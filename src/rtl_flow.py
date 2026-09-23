"""--rtl loop: RTL edits checked on Verilator, then parameters, then FireSim +
AlphaEvolve, fed back to the HW LLM.

Each outer iteration:
  1. Baseline: Verilator run of the currently accepted design with the fixed
     schedules (verilator_eval: the seed, the best schedule so far, the fusion
     reference).
  2. RTL phase (rtl_iterations steps): OpenCode makes one microarchitecture
     change, verilator_eval checks it (ISA unchanged, predictions bit-identical
     to the seed) and times it. A passing, faster change is accepted (the
     Gemmini sources are snapshotted); anything else is reverted to the last
     accepted snapshot, and OpenCode gets the reason either way.
  3. Parameter phase (param_iterations steps): the same, for Configs.scala.
  4. If anything was accepted: FireSim bitstream build, then AlphaEvolve
     searches schedules on it (config_alphaevolve.yaml's max_iterations). Its
     best schedule seeds the next search and the next Verilator runs, and its
     numbers go back to OpenCode in the next iteration.

Only the Gemmini sources (generators/gemmini and its gemmini-rocc-tests
submodule) are snapshotted and restored; edits the LLM makes elsewhere in its
writable dirs are not reverted.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import ray

from chia.base.ChiaFunction import get

from alphaevolve_flow import DEFAULT_CONFIG, SEED_PROGRAM, run_alphaevolve_search
from chipyard_ops import restore_repos, snapshot_repos
from constants import CHIPYARD_PATH, CHIPYARD_WRITABLE_DIRS, DEFAULT_OUTPUT_BASE
from dumper import Dumper, dump_llm
from exo_compiler import check_exo_compatible
from firesim import (
    SandboxedBashTool,
    firesim_buildbitstream_mock,
    read_gemmini_params_h,
    start_bitstream_build,
)
from llm import make_llm, param_tune, rtl_edit
from verilator_eval import (
    DEFAULT_PATCHES,
    evaluate_rtl,
    isa_fingerprint,
    verilator_cycles,
)

logger = logging.getLogger(__name__)

TAIL_CHARS = 4000


def _tail(text: Optional[str], limit: int = TAIL_CHARS) -> str:
    text = text or ""
    return text if len(text) <= limit else f"...(truncated)...\n{text[-limit:]}"


def _new_bash() -> SandboxedBashTool:
    return SandboxedBashTool(
        name="chipyard_bash",
        work_dir=CHIPYARD_PATH,
        writable_dirs=CHIPYARD_WRITABLE_DIRS,
        timeout_seconds=300,
        task_options={"resources": {"manager": 1}},
    )


def _snapshot() -> dict:
    return get(snapshot_repos.options(resources={"manager": 0.01}).chia_remote(CHIPYARD_PATH))


def _restore(snap: dict) -> None:
    get(restore_repos.options(resources={"manager": 0.01}).chia_remote(CHIPYARD_PATH, snap))


def _snapshot_diff(snap: dict) -> str:
    """Readable form of a snapshot: each repo's diff plus its untracked files."""
    parts = []
    for repo, state in snap.items():
        if state["diff"]:
            parts.append(f"# ===== {repo} =====\n"
                         + state["diff"].decode(errors="replace"))
        for path, (kind, data) in state["untracked"].items():
            body = f"-> {data}" if kind == "link" else data.decode(errors="replace")
            parts.append(f"# ===== {repo}/{path} (untracked {kind}) =====\n{body}")
    return "\n".join(parts)


@dataclass
class _State:
    llm: object
    dump: Dumper
    out: Path
    isa_reference: dict
    n_patches: int
    accepted_snap: dict                      # Gemmini sources of the accepted design
    best_source: Optional[str] = None        # best AlphaEvolve schedule, None = the seed
    best_cycles: float = float("inf")        # Verilator cycles of the accepted design
    baseline_cycles: float = float("inf")    # ... at the start of this iteration
    accepted_notes: list = field(default_factory=list)   # this iteration's accepted changes
    earlier_notes: list = field(default_factory=list)    # ... of earlier iterations that were built
    last_step: str = ""                      # result of the previous LLM step
    firesim_feedback: str = ""               # outcome of the last FireSim + AlphaEvolve run

    def log(self, text: str) -> None:
        logger.info(text.splitlines()[0] if text else "")
        with open(self.out / "rtl_steps.txt", "a") as f:
            f.write(text.rstrip() + "\n\n")


def _context(state: _State, outer: int, phase: str, step: int, steps: int) -> str:
    notes = "\n".join(f"- {n}" for n in state.accepted_notes) or "(none yet)"
    earlier = "\n".join(f"- {n}" for n in state.earlier_notes) or "(none)"
    return (
        f"## Where the loop is\n\n"
        f"Hardware iteration {outer + 1}, {phase} step {step + 1} of {steps}.\n"
        f"Verilator mean cycles/patch: {state.baseline_cycles:,.0f} at the start of "
        f"this iteration, {state.best_cycles:,.0f} for the design in the tree now.\n\n"
        f"Changes accepted and built in earlier iterations (in the tree now):\n{earlier}\n\n"
        f"Changes accepted in this iteration (they are in the tree now):\n{notes}\n\n"
        f"Result of your previous step:\n{state.last_step or '(none yet)'}\n\n"
        f"FireSim + AlphaEvolve result of the last hardware built:\n"
        f"{state.firesim_feedback or '(none yet)'}\n"
    )


def _improve(state: _State, outer: int, phase: str, ask: Callable, steps: int) -> None:
    """Up to *steps* LLM edits, each kept only if Verilator accepts it."""
    for step in range(steps):
        label = f"{phase} {outer + 1}.{step + 1}"
        bash = _new_bash()
        reply, error = "", None
        try:
            impl = ask(state.llm, bash, _context(state, outer, phase, step, steps))
            dump_llm(state.dump, f"{phase}_o{outer}_s{step}", impl)
            reply = impl.result or ""
        except Exception as e:  # noqa: BLE001 -- an LLM/API error must not end a long run
            error = f"{type(e).__name__}: {e}"
        finally:
            bash.stop()

        if error:
            _restore(state.accepted_snap)
            state.last_step = f"The LLM call failed ({_tail(error, 500)}); nothing was kept."
            state.log(f"[{label}] LLM call failed: {error}")
            continue

        result = evaluate_rtl(best_source=state.best_source, n_patches=state.n_patches,
                              isa_reference=state.isa_reference)
        cycles = verilator_cycles(result)
        if cycles is not None and cycles < state.best_cycles:
            verdict = (f"ACCEPTED: {cycles:,.0f} cycles/patch "
                       f"(was {state.best_cycles:,.0f}); it stays in the tree.")
            state.accepted_snap = _snapshot()
            state.best_cycles = cycles
            state.accepted_notes.append(f"{label}: {_tail(reply, 600)}")
            (state.out / f"accepted_{phase}_o{outer}_s{step}.diff").write_text(
                _snapshot_diff(state.accepted_snap))
        else:
            _restore(state.accepted_snap)
            why = result.reason or (
                f"correct, but not faster ({cycles:,.0f} vs {state.best_cycles:,.0f} "
                "cycles/patch)")
            verdict = f"REJECTED and reverted: {why}."
        state.last_step = f"{result.summary()}\n{verdict}"
        state.log(f"[{label}] {verdict}\n{result.summary()}")


def run_rtl_flow(
    output_dir: str,
    iterations: int = 1,
    rtl_iterations: int = 3,
    param_iterations: int = 2,
    config_path: str = DEFAULT_CONFIG,
    n_patches: int = DEFAULT_PATCHES,
    mock_bitstream: bool = False,
    mvout_spad: bool = False,
):
    if not ray.is_initialized():
        ray.init(address="auto")
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    dump = Dumper(output_dir)
    logger.info("Output directory: %s", output_dir)

    bash = _new_bash()
    llm = make_llm(bash)
    bash.stop()
    start_snap = _snapshot()
    (Path(output_dir) / "start_state.diff").write_text(_snapshot_diff(start_snap))
    state = _State(
        llm=llm, dump=dump, out=Path(output_dir),
        isa_reference=get(isa_fingerprint.chia_remote(CHIPYARD_PATH)),
        n_patches=n_patches, accepted_snap=start_snap,
    )
    exo_seed = SEED_PROGRAM
    best_firesim = None
    run = None

    for outer in range(iterations):
        iteration_snap = state.accepted_snap
        state.accepted_notes = []
        baseline = evaluate_rtl(best_source=state.best_source, n_patches=n_patches,
                                isa_reference=state.isa_reference)
        state.log(f"[baseline {outer + 1}] {baseline.summary()}")
        if not baseline.passed:
            raise RuntimeError(
                "the accepted design fails its own Verilator check, so no change "
                f"can be judged:\n{baseline.summary()}\n{_tail(baseline.sim_build_log_tail, 1500)}")
        state.baseline_cycles = state.best_cycles = verilator_cycles(baseline)

        _improve(state, outer, "rtl", rtl_edit, rtl_iterations)
        _improve(state, outer, "param", param_tune, param_iterations)
        _restore(state.accepted_snap)

        if state.accepted_snap is iteration_snap:
            state.firesim_feedback = (
                f"Hardware iteration {outer + 1} accepted no change, so no bitstream "
                "was built; the hardware is the same as before.")
            state.log(f"[iteration {outer + 1}] no accepted change; skipping FireSim")
            continue

        hw_summary = "\n".join(state.accepted_notes)
        bitstream_ref = (firesim_buildbitstream_mock.chia_remote() if mock_bitstream
                         else start_bitstream_build())
        ok, bitstream_out, bitstream_err = get(bitstream_ref)
        dump.text(f"bitstream_o{outer}.log", f"STDOUT:\n{bitstream_out}\n\nSTDERR:\n{bitstream_err}")
        if not ok:
            _restore(iteration_snap)
            state.accepted_snap = iteration_snap
            state.firesim_feedback = (
                f"The FireSim bitstream build of hardware iteration {outer + 1} failed, "
                "so its changes were reverted:\n" + _tail(bitstream_err or bitstream_out, 2000))
            state.log(f"[iteration {outer + 1}] bitstream build failed; reverted")
            continue

        params_h = get(read_gemmini_params_h.options(resources={"manager": 0.05}).chia_remote())
        dump.text(f"gemmini_params_o{outer}.h", params_h)
        unsupported = check_exo_compatible(params_h)
        if unsupported:
            _restore(iteration_snap)
            state.accepted_snap = iteration_snap
            state.firesim_feedback = (
                f"Hardware iteration {outer + 1} cannot be programmed by the Exo "
                f"search ({unsupported}), so its changes were reverted.")
            state.log(f"[iteration {outer + 1}] not Exo-compatible: {unsupported}; reverted")
            continue

        state.earlier_notes += state.accepted_notes
        run = run_alphaevolve_search(
            dump, outer, exo_seed, bitstream_ref,
            config_path=config_path,
            eval_dir=f"{DEFAULT_OUTPUT_BASE}/{Path(output_dir).name}",
            hw_change_summary=_tail(hw_summary, 3000),
            hw_context=params_h, gemmini_params_h=params_h,
            mvout_spad=mvout_spad,
        )
        metrics = run.best_metrics or {}
        cycles = metrics.get("cycles") if run.best_program else None
        if cycles is None:
            state.firesim_feedback = (
                f"On hardware iteration {outer + 1}'s FireSim bitstream the AlphaEvolve "
                f"search found no working schedule (status {run.terminal_status}).")
        else:
            exo_seed = state.best_source = run.best_program
            previous = best_firesim
            if best_firesim is None or cycles < best_firesim:
                best_firesim = cycles
                Path(output_dir, "best_braggnn_schedule.py").write_text(run.best_program)
                Path(output_dir, "best_rtl.diff").write_text(_snapshot_diff(state.accepted_snap))
                Path(output_dir, "best.json").write_text(json.dumps(
                    {"iteration": outer + 1, **metrics}, indent=2))
            state.firesim_feedback = (
                f"On hardware iteration {outer + 1}'s FireSim bitstream, AlphaEvolve's "
                f"best schedule runs at {cycles:,.0f} cycles/patch "
                f"(subpixel_error {metrics.get('subpixel_error')}); "
                f"best on earlier hardware: "
                f"{f'{previous:,.0f}' if previous is not None else 'none yet'}.")
        state.log(f"[iteration {outer + 1}] {state.firesim_feedback}")

    logger.info("RTL loop finished (%d iteration(s)); best FireSim cycles: %s",
                iterations, best_firesim)
    return run
