"""AlphaEvolve optimization loop for the BraggNN Exo kernel on Gemmini.

The normal entry point searches the Exo schedule against the currently deployed
FireSim bitstream and does not modify the hardware.  The experimental HW + SW
co-design implementation is kept in this module but is not called by the CLI.

The retained co-design implementation has two halves per attempt:
  1. HW: OpenCode (Gemini on Vertex AI) edits the Gemmini Chisel sources on the
     FireSim node, then the design is built (Verilator) and a FireSim bitstream
     build is started in the background.
  2. SW: AlphaEvolve searches over the Exo schedule (exo/braggnn_schedule.py) for
     that hardware. Each candidate is compiled with Exo in the exo container and
     run on FireSim; the score is the average cycles per patch.
The best schedule of an attempt seeds the next attempt, and the outcome is fed
back to the HW LLM (implement -> optimize, or debug after a failure).
"""

import argparse
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import ray

from chia.base.ChiaFunction import get

from alphaevolve_flow import DEFAULT_CONFIG, SEED_PROGRAM, run_alphaevolve_search
from chipyard_ops import capture_chisel_baseline
from constants import (
    CHIPYARD_DIFF_SUBMODULES,
    CHIPYARD_WRITABLE_DIRS,
    DEFAULT_OUTPUT_BASE,
    FIRESIM_CHIPYARD_PATH,
    MAX_ATTEMPTS,
)
from dumper import Dumper, dump_llm
from firesim import (
    SandboxedBashTool,
    chisel_build,
    chisel_build_mock,
    collect_chisel_diff,
    firesim_buildbitstream_mock,
    start_bitstream_build,
)
from hammer_ppa import run_ppa_synthesis
from llm import debug, implement, make_llm, optimize

logger = logging.getLogger(__name__)

FEEDBACK_LOG_TAIL_CHARS = 4000


def _tail(text: Optional[str], limit: int = FEEDBACK_LOG_TAIL_CHARS) -> str:
    if text is None:
        return ""
    return text if len(text) <= limit else f"...(truncated)...\n{text[-limit:]}"


def _best_cycles(run) -> Optional[float]:
    """Measured avg cycles of the run's best program (attached by
    run_alphaevolve_search), or None if no candidate worked."""
    if run is None or run.terminal_status == "error" or not run.best_program:
        return None
    return (run.best_metrics or {}).get("cycles")


def _cycles_score(run) -> Optional[float]:
    cycles = _best_cycles(run)
    return None if cycles is None else -cycles


def _default_out_dir() -> str:
    """Where a run writes by default: the same place braggnn_loop.py uses, an
    absolute path under $HOME on the head node. A relative one would land in
    the temporary directory Ray unpacks the job into and be hard to find."""
    return str(
        Path.home() / "braggnn_loop_runs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    )


def _eval_dir(output_dir: str) -> str:
    """Where BraggnnEvaluator writes chia_eval_log.jsonl and candidates/.

    It runs inside the evolver container, which has its own filesystem and no
    access to the driver's home, so this stays a relative path there; the files
    are read back onto the driver with _read_eval_log / _read_candidates."""
    return os.path.join(DEFAULT_OUTPUT_BASE, os.path.basename(output_dir.rstrip("/")))


def _run_sw_only(dump: Dumper, output_dir: str, config_path: str):
    """One AlphaEvolve search with no FireSim node (laptop test): no HW LLM,
    Chisel build or bitstream, and each candidate's "cycles" is its ELF size.
    It exercises the Exo build and the search plumbing; the scores mean nothing."""
    logger.warning("--sw-only: no FireSim runs; cycles are ELF sizes, not measurements")
    run = run_alphaevolve_search(
        dump, 0, SEED_PROGRAM, ray.put((True, "[sw-only] no bitstream build", "")),
        config_path=config_path, eval_dir=_eval_dir(output_dir),
        hw_context="(not available: --sw-only test run, no FireSim node)",
        fake_run=True,
    )
    logger.info(
        "sw-only search finished: status=%s, iterations=%s, best=%s",
        run.terminal_status, run.iteration_count, run.best_metrics,
    )
    if run.best_program:
        dump.text("best_braggnn_schedule.py", run.best_program)
    return run


def _run_hw_flow(
    output_dir: Optional[str] = None,
    iterations: int = MAX_ATTEMPTS,
    config_path: str = DEFAULT_CONFIG,
    mock_chisel_build: bool = False,
    mock_bitstream: bool = False,
    sw_only: bool = False,
):
    if not ray.is_initialized():
        ray.init(address="auto")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    if output_dir is None:
        output_dir = _default_out_dir()

    dump = Dumper(output_dir)
    logger.info("Output directory: %s", output_dir)

    if sw_only:
        return _run_sw_only(dump, output_dir, config_path)

    baseline = get(
        capture_chisel_baseline.options(resources={"manager": 0.01}).chia_remote(
            FIRESIM_CHIPYARD_PATH, CHIPYARD_DIFF_SUBMODULES
        )
    )

    def _new_chipyard_bash() -> SandboxedBashTool:
        return SandboxedBashTool(
            name="chipyard_bash",
            work_dir=FIRESIM_CHIPYARD_PATH,
            writable_dirs=CHIPYARD_WRITABLE_DIRS,
            timeout_seconds=300,
            task_options={"resources": {"manager": 1}},
        )

    chipyard_bash = _new_chipyard_bash()

    logger.info("Creating LLM Node")
    llm = make_llm(chipyard_bash)

    feedback = None
    feedback_kind = None  # "failure" | "success" -- which LLM prompt feedback needs
    run = None
    best_run = None
    best_score = None
    exo_seed = SEED_PROGRAM
    for attempt in range(iterations):
        if feedback is None:
            impl = implement(llm, chipyard_bash)
            dump_llm(dump, "implement", impl)
            logger.info("Implement finished (success=%s)", impl.success)
        elif feedback_kind == "success":
            impl = optimize(llm, chipyard_bash, feedback)
            dump_llm(dump, f"optimize_attempt{attempt}", impl)
            logger.info("Optimize finished (attempt %d, success=%s)", attempt + 1, impl.success)
        else:
            impl = debug(llm, chipyard_bash, feedback)
            dump_llm(dump, f"debug_attempt{attempt}", impl)
            logger.info("Debug finished (attempt %d, success=%s)", attempt + 1, impl.success)

        chipyard_bash.stop()

        diffs = collect_chisel_diff(dump, attempt, baseline)
        build = chisel_build_mock if mock_chisel_build else chisel_build
        artifact = build(dump, attempt)

        if not artifact.success:
            logger.error("Build failure (attempt %d)", attempt + 1)
            feedback = (
                f"chisel_build failed (attempt {attempt + 1}):\n\n"
                f"STDOUT:\n{_tail(artifact.stdout)}\n\nSTDERR:\n{_tail(artifact.stderr)}"
            )
            feedback_kind = "failure"
            if attempt + 1 < iterations:
                chipyard_bash = _new_chipyard_bash()
            continue

        if diffs is None:
            logger.error(
                "collect_chisel_diff failed (attempt %d), skipping "
                "AlphaEvolve search", attempt + 1,
            )
            feedback = (
                f"collect_chisel_diff failed (attempt {attempt + 1}): could not "
                "capture the Chisel diff for this attempt."
            )
            feedback_kind = "failure"
            if attempt + 1 < iterations:
                chipyard_bash = _new_chipyard_bash()
            continue

        if mock_bitstream:
            bitstream_ref = firesim_buildbitstream_mock.chia_remote()
        else:
            bitstream_ref = start_bitstream_build()

        run = run_alphaevolve_search(
            dump, attempt, exo_seed, bitstream_ref,
            config_path=config_path, eval_dir=_eval_dir(output_dir),
            hw_change_summary=_tail(impl.result),
        )

        if (
            run.terminal_status != "error"
            and run.best_program
            and _best_cycles(run) is not None
        ):
            exo_seed = run.best_program

            score = _cycles_score(run)
            if score is not None and (best_score is None or score > best_score):
                best_run, best_score = run, score
                logger.info(
                    "New best result (attempt %d): cycles=%s", attempt + 1, -score,
                )

            metrics = run.best_metrics or {}
            cycles = metrics.get("cycles")
            subpixel_error = metrics.get("subpixel_error")
            logger.info(
                "Attempt %d succeeded: cycles=%s, subpixel_error=%s -- "
                "continuing to refine the design",
                attempt + 1, cycles, subpixel_error,
            )
            # Best-effort PPA (Genus synthesis, sky130+SRAM22) on this
            # attempt's hardware, now that it has a validated FireSim result.
            # Never blocks/fails the loop; see hammer_ppa.run_ppa_synthesis.
            ppa = run_ppa_synthesis(dump, attempt)
            logger.info(
                "Hammer PPA (attempt %d): %s", attempt + 1,
                "OK" if ppa.success else f"FAILED at {ppa.stage}",
            )
            if ppa.success and ppa.final_area_rpt:
                ppa_note = (
                    "\n\nHammer PPA (Genus synthesis, sky130+SRAM22) for this "
                    f"attempt's hardware:\n{_tail(ppa.final_area_rpt)}"
                )
            else:
                ppa_note = (
                    f"\n\nHammer PPA (Genus synthesis, sky130+SRAM22) FAILED at "
                    f"{ppa.stage} for this attempt's hardware; area/timing not "
                    "available this round."
                )
            feedback = (
                f"Attempt {attempt + 1} succeeded and was validated on FireSim "
                f"hardware:\n  cycles = {cycles}\n  subpixel_error = {subpixel_error}\n\n"
                f"Best result so far: cycles = {-best_score if best_score is not None else 'n/a'}."
                f"{ppa_note}"
            )
            feedback_kind = "success"
            if attempt + 1 < iterations:
                chipyard_bash = _new_chipyard_bash()
            continue

        logger.error("AlphaEvolve search failure (attempt %d)", attempt + 1)
        feedback = (
            f"alphaevolve search failed (attempt {attempt + 1}): no candidate "
            f"was successfully evaluated (iterations={run.iteration_count}, "
            f"status={run.terminal_status}).\n\n{_tail(run.error_message)}"
        )
        feedback_kind = "failure"
        if attempt + 1 < iterations:
            chipyard_bash = _new_chipyard_bash()

    logger.info("Loop finished (%d attempt(s))", iterations)
    result = best_run or run
    if best_run is not None and best_run.best_program:
        dump.text("best_braggnn_schedule.py", best_run.best_program)
    return result


def run_flow(
    output_dir: Optional[str] = None,
    config_path: str = DEFAULT_CONFIG,
    sw_only: bool = False,
):
    """Run one software search against the currently deployed hardware."""
    if not ray.is_initialized():
        ray.init(address="auto")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    if output_dir is None:
        output_dir = _default_out_dir()

    dump = Dumper(output_dir)
    logger.info("Output directory: %s", output_dir)

    if sw_only:
        return _run_sw_only(dump, output_dir, config_path)

    # AlphaEvolve still accepts a readiness ref because the retained co-design
    # path starts a bitstream build in parallel.  In the normal path the
    # existing bitstream is already ready, so resolve it immediately.
    firesim_ready = ray.put(
        (True, "[fixed-hardware] using the deployed FireSim bitstream", "")
    )
    run = run_alphaevolve_search(
        dump,
        0,
        SEED_PROGRAM,
        firesim_ready,
        config_path=config_path,
        eval_dir=_eval_dir(output_dir),
    )
    logger.info(
        "Software search finished: status=%s, iterations=%s, best=%s",
        run.terminal_status,
        run.iteration_count,
        run.best_metrics,
    )
    # With no working candidate the top-scored program is the seed or a failure.
    if _best_cycles(run) is not None:
        dump.text("best_braggnn_schedule.py", run.best_program)
    return run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        default=_default_out_dir(),
        help="Results directory (default: ~/braggnn_loop_runs/<timestamp>)",
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="AlphaEvolve config")
    parser.add_argument(
        "--sw-only", action="store_true",
        help="Laptop test without a FireSim node: only the AlphaEvolve search, "
             "with fake runs (cycles = ELF size)",
    )
    args = parser.parse_args()
    run_flow(
        output_dir=args.out_dir,
        config_path=args.config,
        sw_only=args.sw_only,
    )


if __name__ == "__main__":
    main()
