"""AlphaEvolve optimization loop for the BraggNN Exo kernel on Gemmini.

By default the CLI searches the Exo schedule against the currently deployed
FireSim bitstream and does not modify the hardware.  With --hw it runs the
HW + SW co-design loop instead, which has two halves per attempt:
  1. HW: OpenCode (Gemini on Vertex AI) edits the Gemmini Chisel sources on the
     FireSim node, then the design is elaborated by Hammer's buildfile (the
     build check), and the FireSim bitstream build and Genus synthesis
     (PPA) are started in the background.
  2. SW: AlphaEvolve searches over the Exo schedule (exo/braggnn_schedule.py) for
     that hardware. Each candidate is compiled with Exo in the exo container and
     run on FireSim; the score is the average cycles per patch.
The best schedule of an attempt seeds the next attempt, and the outcome is fed
back to the HW LLM (implement -> optimize, or debug after a failure).
"""

import argparse
import json
import logging
import os
import re
import signal
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import ray

from chia.base.ChiaFunction import chia_cancel, get

from alphaevolve_flow import DEFAULT_CONFIG, SEED_PROGRAM, run_alphaevolve_search
from chipyard_ops import capture_chisel_baseline, collect_diff, restore_diff
from constants import (
    BUILD_CONFIG,
    CHIPYARD_DIFF_SUBMODULES,
    CHIPYARD_PATH,
    CHIPYARD_WRITABLE_DIRS,
    DEFAULT_OUTPUT_BASE,
    FIRESIM_GENERATED_SV,
    FPGA_RTL_SIZE_LIMIT_BYTES,
    MAX_ATTEMPTS,
)
from dumper import Dumper, dump_llm
from firesim import (
    SandboxedBashTool,
    collect_chisel_diff,
    firesim_buildbitstream_mock,
    read_gemmini_params_h,
    start_bitstream_build,
)
from exo_compiler import check_exo_compatible
from hammer_ppa import PPA_FLOW_LABEL, PpaJob
from llm import debug, implement, make_llm, optimize

logger = logging.getLogger(__name__)

FEEDBACK_LOG_TAIL_CHARS = 4000


def _tail(text: Optional[str], limit: int = FEEDBACK_LOG_TAIL_CHARS) -> str:
    if text is None:
        return ""
    return text if len(text) <= limit else f"...(truncated)...\n{text[-limit:]}"


# Lines that name an elaboration error. A Chisel stack trace is ~20 KB and
# its tail is only Chisel/Scala internals, so _tail() alone loses the cause.
_ERROR_LINE_RE = re.compile(
    r"Exception|Caused by:|\[error\]|requirement failed|assertion failed|^\s*at gemmini\."
)


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _error_digest(text: Optional[str], max_lines: int = 30) -> str:
    """The first *max_lines* distinct error lines of *text* (exception
    headers, 'Caused by', sbt [error], and gemmini stack frames)."""
    seen, out = set(), []
    for line in _ANSI_RE.sub("", text or "").splitlines():
        line = line.rstrip()
        # sbt's [warn] lines mention exception types in import lists
        if line.lstrip().startswith("[warn]"):
            continue
        if _ERROR_LINE_RE.search(line) and line not in seen:
            seen.add(line)
            out.append(line)
            if len(out) >= max_lines:
                break
    return "\n".join(out) or "(none found)"


def _best_cycles(run) -> Optional[float]:
    """Measured avg cycles of the run's best program (attached by
    run_alphaevolve_search), or None if no candidate worked."""
    if run is None or run.terminal_status == "error" or not run.best_program:
        return None
    return (run.best_metrics or {}).get("cycles")


def _cycles_score(run) -> Optional[float]:
    cycles = _best_cycles(run)
    return None if cycles is None else -cycles


def _log_attempt(output_dir: str, attempt: int, feedback: str) -> None:
    """Append an attempt's outcome (the feedback the HW LLM gets) to
    attempts.txt as soon as it is known, so a stopped run keeps its history."""
    with open(Path(output_dir) / "attempts.txt", "a") as f:
        f.write(f"===== attempt {attempt + 1} =====\n{feedback}\n\n")


def _save_best(output_dir: str, attempt: int, run, diffs: dict, ppa) -> None:
    """The best attempt so far -- its schedule, Chisel diff and numbers --
    rewritten on every improvement, without the Dumper's timestamp prefix."""
    out = Path(output_dir)
    (out / "best_braggnn_schedule.py").write_text(run.best_program)
    (out / "best_chisel.diff").write_text("\n\n".join(
        f"# ===== diff: {repo or '(chipyard root)'} =====\n{text}"
        for repo, text in diffs.items() if text
    ))
    metrics = run.best_metrics or {}
    (out / "best.json").write_text(json.dumps({
        "attempt": attempt + 1,
        "cycles": metrics.get("cycles"),
        "subpixel_error": metrics.get("subpixel_error"),
        "ppa_success": ppa.success,
        "total_area_um2": ppa.total_area_um2,
        "worst_slack_ns": ppa.worst_slack_ns,
        "total_power_w": ppa.total_power_w,
        "ppa_obj_dir": ppa.obj_dir,
    }, indent=2))


def _load_resume(run_dir: str, output_dir: str):
    """State to continue a stopped --hw run whose last attempt succeeded:
    (next attempt index, that attempt's feedback, its best schedule as the
    seed, its score). The Chisel of that attempt must still be in the
    checkout. best.* are copied into *output_dir* so the new run starts with
    them, and resumed_from.txt records where they came from."""
    src = Path(run_dir)
    best = json.loads((src / "best.json").read_text())
    done = int(best["attempt"])  # 1-based number of the attempt resumed from
    blocks = re.split(r"^===== attempt (\d+) =====\n", (src / "attempts.txt").read_text(),
                      flags=re.MULTILINE)
    by_attempt = {int(n): text.strip() for n, text in zip(blocks[1::2], blocks[2::2])}
    if done not in by_attempt:
        raise SystemExit(f"--resume-from: no feedback for attempt {done} in {src / 'attempts.txt'}")
    out = Path(output_dir)
    for name in ("best.json", "best_braggnn_schedule.py", "best_chisel.diff"):
        if (src / name).exists():
            (out / name).write_text((src / name).read_text())
    (out / "resumed_from.txt").write_text(f"{src.resolve()} (after attempt {done})\n")
    return (done, by_attempt[done], (src / "best_braggnn_schedule.py").read_text(),
            -float(best["cycles"]))


def _fresh_fpga_rtl_size(since: float) -> Optional[int]:
    """Size of the FPGA RTL the running bitstream build wrote, or None until
    it has written one (the file left by an earlier build is older than
    *since*). Read directly: the build dir is on NFS, visible from here."""
    try:
        st = os.stat(FIRESIM_GENERATED_SV)
    except OSError:
        return None
    return st.st_size if st.st_mtime > since else None


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
    mock_bitstream: bool = False,
    sw_only: bool = False,
    resume_from: Optional[str] = None,
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
            CHIPYARD_PATH, CHIPYARD_DIFF_SUBMODULES
        )
    )
    # The Chisel diff of the hardware last built (at first: the unedited
    # checkout, which already differs from HEAD). An attempt whose diff equals
    # it changed nothing, so it is skipped instead of rebuilding the same design.
    last_diffs = get(
        collect_diff.options(resources={"manager": 0.05}).chia_remote(
            CHIPYARD_PATH, CHIPYARD_DIFF_SUBMODULES, baseline
        )
    )[1]

    def _new_chipyard_bash() -> SandboxedBashTool:
        return SandboxedBashTool(
            name="chipyard_bash",
            work_dir=CHIPYARD_PATH,
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
    first_attempt = 0
    if resume_from:
        first_attempt, feedback, exo_seed, best_score = _load_resume(resume_from, output_dir)
        feedback_kind = "success"
        logger.info("Resuming from %s: continuing at attempt %d (best cycles %s)",
                    resume_from, first_attempt + 1, -best_score)
    # The last design that built and was measured, to revert to when an
    # attempt's hardware is too large to build or its bitstream build fails
    # (the checkout the run starts from counts as good: the unedited design,
    # or the design of the attempt resumed from).
    good_diffs = last_diffs
    good_label = f"attempt {first_attempt}" if resume_from else "the unedited starting"
    good_cycles = -best_score if best_score is not None else None

    def _revert_to_good() -> str:
        """Restore the checkout to good_diffs; the sentence to add to the
        feedback. Called with chipyard_bash stopped (it holds 'manager')."""
        nonlocal last_diffs
        ok, msg = get(restore_diff.options(resources={"manager": 0.05}).chia_remote(
            CHIPYARD_PATH, CHIPYARD_DIFF_SUBMODULES, baseline, good_diffs))
        logger.info("Revert to %s design: ok=%s (%s)", good_label, ok, msg)
        if not ok:
            return (f"\n\n(Reverting your change failed -- {msg} -- so the checkout "
                    "still holds it; undo it yourself.)")
        last_diffs = good_diffs
        cycles = f" ({good_cycles:.0f} cycles)" if good_cycles is not None else ""
        return (f"\n\nYour change has been reverted: the checkout is back at "
                f"{good_label}'s design{cycles}. Start again from there with a "
                "smaller change.")

    logged_feedback = None  # a skipped attempt keeps feedback; log it once
    for attempt in range(first_attempt, iterations):
        if feedback is not None and feedback is not logged_feedback:
            _log_attempt(output_dir, attempt - 1, feedback)
            logged_feedback = feedback
        impl = None
        llm_error = None
        try:
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
        except Exception as e:  # noqa: BLE001 -- an LLM/API error must not end a long run
            llm_error = f"{type(e).__name__}: {e}"
            logger.error("LLM call failed (attempt %d): %s", attempt + 1, llm_error)

        chipyard_bash.stop()

        diffs = collect_chisel_diff(dump, attempt, baseline)
        if diffs is not None and diffs == last_diffs:
            reason = (f"the LLM call failed ({_tail(llm_error, 500)})" if llm_error
                      else "the LLM changed no Chisel file")
            logger.error("Attempt %d made no hardware change (%s); skipping it",
                         attempt + 1, reason)
            with open(Path(output_dir) / "attempts.txt", "a") as f:
                f.write(f"===== attempt {attempt + 1} =====\nskipped: {reason}\n\n")
            # feedback / feedback_kind stay as they were, so the next attempt
            # repeats the same request.
            if attempt + 1 < iterations:
                chipyard_bash = _new_chipyard_bash()
            continue
        if diffs is not None:
            last_diffs = diffs
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

        # Hammer's buildfile elaborates the design, so it doubles as the
        # check that the Chisel edit builds. It runs before the bitstream
        # build: both run sbt in the same chipyard checkout.
        ppa_job = PpaJob(dump, attempt)
        elab = ppa_job.elaborate()
        if elab is not None and not elab.success:
            logger.error("Elaboration failure (attempt %d)", attempt + 1)
            feedback = (
                f"Hammer buildfile (elaboration of {BUILD_CONFIG}) failed "
                f"(attempt {attempt + 1}).\n\n"
                f"Key lines (the cause is usually the first 'Caused by' and the "
                f"first 'at gemmini.' frame):\n{_error_digest(elab.stdout + elab.stderr)}\n\n"
                f"STDOUT:\n{_tail(elab.stdout)}\n\nSTDERR:\n{_tail(elab.stderr)}"
            )
            feedback_kind = "failure"
            if attempt + 1 < iterations:
                chipyard_bash = _new_chipyard_bash()
            continue

        # The vlsi worker is the loop's only build check and the only source
        # of an up-to-date gemmini_params.h, so without it the attempt cannot
        # be evaluated. That is an infrastructure problem, not the LLM's, so
        # stop instead of feeding it back as a failure.
        if elab is None:
            raise RuntimeError(
                f"vlsi worker unavailable (attempt {attempt + 1}); see "
                f"{ppa_job.out / 'summary.json'}. Check that the chia-sky130 "
                'container is up and no other job holds {"chipyard": 1}.'
            )

        # Elaboration regenerates gemmini_params.h from the edited Configs.scala;
        # read it now, before the bitstream build elaborates again, and build
        # every AlphaEvolve candidate against it.
        params_h = get(
            read_gemmini_params_h.options(resources={"manager": 0.05}).chia_remote()
        )
        dump.text(f"gemmini_params_attempt{attempt}.h", params_h)
        unsupported = check_exo_compatible(params_h)
        if unsupported:
            logger.error("Hardware not targetable by Exo (attempt %d): %s",
                         attempt + 1, unsupported)
            ppa_job.cancel()
            feedback = (
                f"Attempt {attempt + 1}'s hardware cannot be programmed by the "
                "Exo SW search, so it was not built or measured. The "
                f"gemmini_params.h generated from your Configs.scala has: "
                f"{unsupported}. Keep the mesh 16 wide (meshRows * tileRows == "
                "meshColumns * tileColumns == 16) with int8 inputs/weights and "
                "int32 accumulators."
            )
            feedback_kind = "failure"
            if attempt + 1 < iterations:
                chipyard_bash = _new_chipyard_bash()
            continue

        build_started = time.time()
        if mock_bitstream:
            bitstream_ref = firesim_buildbitstream_mock.chia_remote()
        else:
            bitstream_ref = start_bitstream_build()
        # Genus synthesis overlaps the bitstream build and the SW search.
        ppa_job.start_syn()

        try:
            # Wait for the bitstream before searching: no candidate can be
            # measured without it, the evolver's idle_timeout_s (2 h) is
            # shorter than a build, and the next attempt's Chisel edit must
            # not start while the build is still elaborating. Meanwhile,
            # cancel the build as soon as its FPGA RTL shows it is too large
            # for Vivado to finish (see FPGA_RTL_SIZE_LIMIT_BYTES).
            rtl_bytes = None   # set once the file's size is final
            last_size = None   # it is final when unchanged over one poll
            while not ray.wait([bitstream_ref], timeout=60)[0]:
                if mock_bitstream or rtl_bytes is not None:
                    continue
                size = _fresh_fpga_rtl_size(build_started)
                if size is not None and size == last_size:
                    rtl_bytes = size
                last_size = size
                if rtl_bytes is not None:
                    logger.info("FPGA RTL for attempt %d: %.1f MB (limit %.1f MB)",
                                attempt + 1, rtl_bytes / 1e6,
                                FPGA_RTL_SIZE_LIMIT_BYTES / 1e6)
                if rtl_bytes is not None and rtl_bytes > FPGA_RTL_SIZE_LIMIT_BYTES:
                    break
            if rtl_bytes is not None and rtl_bytes > FPGA_RTL_SIZE_LIMIT_BYTES:
                logger.error("FPGA RTL too large (attempt %d): %.1f MB; cancelling "
                             "the bitstream build", attempt + 1, rtl_bytes / 1e6)
                chia_cancel(bitstream_ref, force=True)
                ppa_job.cancel()
                feedback = (
                    f"Attempt {attempt + 1}'s hardware is too large for the FPGA "
                    f"build, so it was not built or measured: its generated FPGA "
                    f"RTL is {rtl_bytes / 1e6:.1f} MB, over the "
                    f"{FPGA_RTL_SIZE_LIMIT_BYTES / 1e6:.0f} MB limit. For "
                    "reference, the unedited design is 27 MB, a 45.2 MB design "
                    "built in about 4.5 hours, and 53-54 MB designs never got "
                    "through Vivado's RTL elaboration. Undo or scale back the "
                    "change that grew the RTL (reservation-station entries and "
                    "queue lengths, bank counts, sub-banks, dual ports, tile "
                    "shape, scale units), or pay for it by removing hardware "
                    "the software does not use (see 'What the software uses' in "
                    "your instructions)."
                ) + _revert_to_good()
                feedback_kind = "failure"
                if attempt + 1 < iterations:
                    chipyard_bash = _new_chipyard_bash()
                continue
            bitstream_ok, bitstream_out, bitstream_err = get(bitstream_ref)
            dump.text(f"bitstream_attempt{attempt}.log",
                      f"STDOUT:\n{bitstream_out}\n\nSTDERR:\n{bitstream_err}")
            if not bitstream_ok:
                logger.error("Bitstream build failure (attempt %d)", attempt + 1)
                ppa_job.cancel()
                feedback = (
                    f"FireSim bitstream build failed (attempt {attempt + 1}), so "
                    "nothing was measured:\n\n"
                    f"STDOUT:\n{_tail(bitstream_out)}\n\nSTDERR:\n{_tail(bitstream_err)}"
                ) + _revert_to_good()
                feedback_kind = "failure"
                if attempt + 1 < iterations:
                    chipyard_bash = _new_chipyard_bash()
                continue
            run = run_alphaevolve_search(
                dump, attempt, exo_seed, bitstream_ref,
                config_path=config_path, eval_dir=_eval_dir(output_dir),
                hw_change_summary=_tail(impl.result if impl else ""),
                hw_context=params_h, gemmini_params_h=params_h,
            )
        except BaseException:
            ppa_job.cancel()
            # The build runs in its own process group (firesim.bash), so it
            # would outlive this driver; chia_cancel kills that group.
            try:
                chia_cancel(bitstream_ref, force=True)
            except Exception as e:  # noqa: BLE001
                logger.warning("Could not cancel the bitstream build: %s", e)
            raise

        if (
            run.terminal_status != "error"
            and run.best_program
            and _best_cycles(run) is not None
        ):
            exo_seed = run.best_program

            good_diffs = diffs
            good_label = f"attempt {attempt + 1}"
            good_cycles = _best_cycles(run)

            score = _cycles_score(run)
            improved = score is not None and (best_score is None or score > best_score)
            if improved:
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
            # Best-effort PPA (Genus synthesis, sky130 + SRAM macros) on this
            # attempt's hardware: wait for the background syn so its numbers
            # go into this feedback. Never fails the loop; see hammer_ppa.PpaJob.
            ppa = ppa_job.result()
            logger.info(
                "Hammer PPA (attempt %d): %s", attempt + 1,
                "OK" if ppa.success else f"FAILED at {ppa.stage}",
            )
            if improved:
                _save_best(output_dir, attempt, run, diffs, ppa)
            if ppa.success:
                ppa_note = (
                    f"\n\nHammer PPA (Genus synthesis, {PPA_FLOW_LABEL}) for this "
                    f"attempt's hardware:\n{ppa.summary()}"
                )
            else:
                ppa_note = (
                    f"\n\nHammer PPA (Genus synthesis, {PPA_FLOW_LABEL}) FAILED at "
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
        ppa_job.cancel()  # no validated result, so this hardware's PPA is moot
        feedback = (
            f"alphaevolve search failed (attempt {attempt + 1}): no candidate "
            f"was successfully evaluated (iterations={run.iteration_count}, "
            f"status={run.terminal_status}).\n\n{_tail(run.error_message)}"
        )
        feedback_kind = "failure"
        if attempt + 1 < iterations:
            chipyard_bash = _new_chipyard_bash()

    if feedback is not None and feedback is not logged_feedback:
        _log_attempt(output_dir, iterations - 1, feedback)
    logger.info("Loop finished (%d attempt(s))", iterations)
    result = best_run or run
    if best_run is not None and best_run.best_program:
        dump.text("best_braggnn_schedule.py", best_run.best_program)
    return result


def run_flow(
    output_dir: Optional[str] = None,
    config_path: str = DEFAULT_CONFIG,
    sw_only: bool = False,
    gemmini_params_h_path: Optional[str] = None,
    max_iterations: Optional[int] = None,
):
    """Run one software search against the currently deployed hardware.

    *gemmini_params_h_path* is that hardware's gemmini_params.h (e.g. a --hw
    run's gemmini_params_attempt<N>.h when the deployed bitstream is one the
    co-design loop built): candidates are compiled against it and the prompt
    shows it. Without it the build uses the shipped exo/include header and
    the prompt the chipyard checkout's."""
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
    params_h = Path(gemmini_params_h_path).read_text() if gemmini_params_h_path else None
    if params_h is not None:
        problem = check_exo_compatible(params_h)
        if problem:
            raise SystemExit(f"{gemmini_params_h_path}: {problem}")
        dump.text("gemmini_params.h", params_h)
    run = run_alphaevolve_search(
        dump,
        0,
        SEED_PROGRAM,
        firesim_ready,
        config_path=config_path,
        eval_dir=_eval_dir(output_dir),
        hw_context=params_h,
        gemmini_params_h=params_h,
        max_iterations=max_iterations,
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


def _exit_on_sigterm(signum, frame) -> None:
    raise SystemExit(f"terminated by signal {signum}")


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
    parser.add_argument(
        "--hw", action="store_true",
        help="Run the HW + SW co-design loop (Chisel edit -> Hammer "
             "elaboration/synthesis, bitstream build -> AlphaEvolve on FireSim)",
    )
    parser.add_argument(
        "--iterations", type=int, default=MAX_ATTEMPTS,
        help="--hw only: number of hardware attempts",
    )
    parser.add_argument(
        "--gemmini-params-h", default=None,
        help="without --hw: the deployed hardware's gemmini_params.h (e.g. a "
             "--hw run's gemmini_params_attempt<N>.h); candidates are built "
             "against it",
    )
    parser.add_argument(
        "--max-iterations", type=int, default=None,
        help="without --hw: candidates to evaluate (overrides the config's "
             "max_iterations)",
    )
    parser.add_argument(
        "--mock-bitstream", action="store_true",
        help="--hw only: skip the bitstream build and reuse the last built "
             "hwdb entry",
    )
    parser.add_argument(
        "--resume-from", default=None,
        help="--hw only: a previous run's output dir whose last attempt "
             "succeeded; continue after that attempt with its feedback and best "
             "schedule (its Chisel must still be in the checkout)",
    )
    args = parser.parse_args()
    # `ray job stop` sends SIGTERM; turn it into SystemExit so the finally
    # blocks run (killing the detached evolver actor) before Ray's SIGKILL.
    signal.signal(signal.SIGTERM, _exit_on_sigterm)
    if args.hw:
        _run_hw_flow(
            output_dir=args.out_dir,
            iterations=args.iterations,
            config_path=args.config,
            mock_bitstream=args.mock_bitstream,
            sw_only=args.sw_only,
            resume_from=args.resume_from,
        )
    else:
        run_flow(
            output_dir=args.out_dir,
            config_path=args.config,
            sw_only=args.sw_only,
            gemmini_params_h_path=args.gemmini_params_h,
            max_iterations=args.max_iterations,
        )


if __name__ == "__main__":
    main()
