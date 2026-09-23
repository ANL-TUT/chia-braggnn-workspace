"""Verilator check for Gemmini RTL edits: ISA unchanged, output bit-exact.

Before an RTL change goes to a (~5 h) FireSim bitstream, it is built as a
Chipyard Verilator simulator on the firesim node and runs fixed schedules (the
seed, optionally the best one found so far, and the fusion reference) on the
first few patches. A change passes only if

  - no ISA-defining file under generators/gemmini changed (ISA_FILES,
    compared by content with a reference or with HEAD), and
  - every schedule prints exactly the seed's per-patch prediction errors: a
    microarchitectural change must not change the arithmetic, so anything but
    a bit-identical result is a bug (the SW search's 0.5 px gate does not
    apply here).

Several schedules are run because RTL bugs such as a too-early completion ack
only show up under particular instruction sequences. Cycles are reported per
schedule to compare RTL variants with each other; Verilator's memory model is
not FireSim's, so do not compare them with FireSim numbers.

    uv run chia job submit ... -- python src/verilator_eval.py [--best PATH]
"""

import argparse
import json
import logging
import os
import re
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import ray

from chia.base.ChiaFunction import ChiaFunction, get
from chia.chipyard.chisel_build_node import ChiselBuildNode
from chia.chipyard.state_def import BuildTarget
from chia.chipyard.verilator_run_node import VerilatorRunNode

from constants import BUILD_CONFIG, CHIPYARD_PATH, EXO_DIR
from exo_compiler import build_candidate_elf

logger = logging.getLogger(__name__)

# Files under generators/gemmini that define the ISA: the RoCC command
# encodings and funct codes, the scratchpad/accumulator address encoding the
# commands carry, and the C macros that issue them. An RTL edit may change
# how commands execute, not what they are.
ISA_FILES = (
    "src/main/scala/gemmini/GemminiISA.scala",
    "src/main/scala/gemmini/LocalAddr.scala",
    "software/gemmini-rocc-tests/include/gemmini.h",
)

# Per-patch (x, y) prediction errors in px that the seed prints
# (braggnn_main.c, three decimals). Predictions are int8 steps of 11/127 =
# 0.087 px, so any changed prediction moves an error far beyond the 0.0005
# print rounding.
SEED_PATCH_ERRORS = [
    (-0.344, 0.636),
    (0.191, -0.001),
    (0.466, -0.011),
    (0.022, 0.354),
    (0.276, -0.219),
    (-0.777, 0.034),
    (-0.026, 0.014),
    (0.001, 0.022),
    (0.044, 0.162),
    (-0.039, 0.061),
]
_PRINT_ROUNDING = 5e-4

# Patch 0 runs cold (caches, TLB), so run at least two and report the warm ones.
DEFAULT_PATCHES = 3
SIM_BUILD_TIMEOUT_SECONDS = 4 * 3600
SIM_BUILD_MAKE_JOBS = 16
SIM_RUN_TIMEOUT_SECONDS = 3 * 3600
# Runs share the firesim node's "manager" resource with the chipyard shell
# and builds; a fraction lets a few schedules simulate at once.
SIM_RUN_RESOURCES = {"manager": 0.25}
SIM_WORK_DIR = "/tmp/chia-verilator-eval"

_PATCH_RE = re.compile(
    r"patch (\d+): cycles=(\d+) err=\((-?[\d.]+), (-?[\d.]+)\) px"
)


# ── ISA check ────────────────────────────────────────────────────────

def _read(path: str) -> Optional[bytes]:
    try:
        with open(path, "rb") as f:
            return f.read()
    except FileNotFoundError:
        return None


@ChiaFunction(resources={"manager": 0.01})
def isa_fingerprint(chipyard_path: str) -> dict:
    """sha256 of every ISA_FILES file (None if missing), keyed by its path
    under generators/gemmini. Compared by content rather than with git diff
    because gemmini.h lives in the gemmini-rocc-tests submodule, whose edits a
    diff of the gemmini repo does not show."""
    import hashlib
    root = os.path.join(chipyard_path, "generators", "gemmini")
    out = {}
    for rel in ISA_FILES:
        data = _read(os.path.join(root, rel))
        out[rel] = hashlib.sha256(data).hexdigest() if data is not None else None
    return out


@ChiaFunction(resources={"manager": 0.01})
def head_isa_fingerprint(chipyard_path: str) -> dict:
    """Like isa_fingerprint, but of each file as committed at HEAD of the repo
    (or submodule) that contains it."""
    import hashlib
    root = os.path.join(chipyard_path, "generators", "gemmini")
    out = {}
    for rel in ISA_FILES:
        path = os.path.join(root, rel)
        done = subprocess.run(
            ["git", "-C", os.path.dirname(path), "show",
             f"HEAD:./{os.path.basename(path)}"],
            capture_output=True,
        )
        out[rel] = (hashlib.sha256(done.stdout).hexdigest()
                    if done.returncode == 0 else None)
    return out


def isa_changes(reference: dict, current: dict) -> list:
    return sorted(rel for rel in ISA_FILES if reference.get(rel) != current.get(rel))


# ── Result types ─────────────────────────────────────────────────────

@dataclass
class ScheduleRun:
    name: str
    stage: str                 # "ok", or where it failed: "elf", "sim", "output"
    matched: bool = False      # predictions bit-identical to the seed's
    patch_cycles: list = field(default_factory=list)
    warm_avg_cycles: Optional[float] = None   # mean over patches 1.. (patch 0 is cold)
    mismatches: list = field(default_factory=list)
    log_tail: str = ""


@dataclass
class VerilatorEvalResult:
    passed: bool
    reason: str                # why it failed, "" when passed
    isa_changed: list = field(default_factory=list)
    sim_build_log_tail: str = ""
    runs: list = field(default_factory=list)

    def summary(self) -> str:
        """Plain-text report, e.g. as feedback to the RTL-editing LLM."""
        lines = [f"Verilator RTL check: {'PASSED' if self.passed else 'FAILED'}"
                 + (f" ({self.reason})" if self.reason else "")]
        for run in self.runs:
            cycles = (f"{run.warm_avg_cycles:,.0f} warm cycles/patch"
                      if run.warm_avg_cycles is not None else "no cycles")
            verdict = "output matches the seed" if run.matched else f"FAILED at {run.stage}"
            lines.append(f"  {run.name}: {cycles}, {verdict}")
            lines += [f"    {m}" for m in run.mismatches[:4]]
        if self.isa_changed:
            lines.append("  ISA files changed: " + ", ".join(self.isa_changed))
        return "\n".join(lines)


# ── Steps ────────────────────────────────────────────────────────────

def _schedules(best_source: Optional[str], include_fusion: bool) -> dict:
    """name -> Exo program source for every schedule to run."""
    schedules = {"seed": (EXO_DIR / "braggnn_schedule.py").read_text()}
    if best_source:
        schedules["best"] = best_source
    if include_fusion:
        schedules["fusion_ref"] = (EXO_DIR / "braggnn_schedule_fusion.py").read_text()
    return schedules


def _check_output(name: str, text: str, n_patches: int) -> ScheduleRun:
    found = sorted(_PATCH_RE.findall(text), key=lambda m: int(m[0]))
    cycles = [int(c) for _, c, _, _ in found]
    errors = [(float(x), float(y)) for _, _, x, y in found]
    warm = cycles[1:] or cycles
    run = ScheduleRun(
        name=name, stage="ok", patch_cycles=cycles,
        warm_avg_cycles=sum(warm) / len(warm) if warm else None,
        log_tail=text[-2000:],
    )
    if len(errors) != n_patches:
        run.stage = "output"
        run.mismatches = [f"expected {n_patches} patch lines, got {len(errors)}"]
        return run
    run.mismatches = [
        f"patch {i}: err=({x:.3f}, {y:.3f}) px, seed ({sx:.3f}, {sy:.3f})"
        for i, ((x, y), (sx, sy)) in enumerate(zip(errors, SEED_PATCH_ERRORS))
        if abs(x - sx) > _PRINT_ROUNDING or abs(y - sy) > _PRINT_ROUNDING
    ]
    run.matched = not run.mismatches
    if not run.matched:
        run.stage = "output"
    return run


def evaluate_rtl(
    best_source: Optional[str] = None,
    include_fusion: bool = True,
    n_patches: int = DEFAULT_PATCHES,
    isa_reference: Optional[dict] = None,
    clean_build: bool = False,
) -> VerilatorEvalResult:
    """Check the RTL currently in CHIPYARD_PATH's generators/gemmini.

    *best_source* is an extra Exo program to run next to the seed.
    *isa_reference* is an isa_fingerprint() the ISA files must still match
    (e.g. taken when a loop starts); default: as committed at HEAD.
    """
    if not 1 <= n_patches <= len(SEED_PATCH_ERRORS):
        raise ValueError(f"n_patches must be 1..{len(SEED_PATCH_ERRORS)}")

    if isa_reference is None:
        isa_reference = get(head_isa_fingerprint.chia_remote(CHIPYARD_PATH))
    isa_changed = isa_changes(
        isa_reference, get(isa_fingerprint.chia_remote(CHIPYARD_PATH)))
    if isa_changed:
        return VerilatorEvalResult(
            passed=False, reason="the ISA changed", isa_changed=isa_changed)

    # The ELFs (exo container) and the simulator (firesim node) build in parallel.
    schedules = _schedules(best_source, include_fusion)
    elf_refs = {
        name: build_candidate_elf.chia_remote(src, None, f"-DEVAL_PATCHES={n_patches}")
        for name, src in schedules.items()
    }
    builder = ChiselBuildNode(
        chipyard_path=CHIPYARD_PATH,
        config=BUILD_CONFIG,
        target=BuildTarget.VERILATOR,
        make_jobs=SIM_BUILD_MAKE_JOBS,
        timeout_seconds=SIM_BUILD_TIMEOUT_SECONDS,
        clean=clean_build,
    )
    logger.info("Building the Verilator simulator for %s", BUILD_CONFIG)
    artifact = get(builder.build.options(resources={"manager": 1}).chia_remote(builder))
    build_tail = (artifact.stdout + artifact.stderr)[-3000:]
    if not artifact.success:
        return VerilatorEvalResult(
            passed=False, reason="the Verilator build failed",
            sim_build_log_tail=build_tail)
    artifact_ref = ray.put(artifact)  # one copy of the simulator for all runs

    runs, run_refs = [], {}
    runner = VerilatorRunNode(logging_level=logging.INFO)
    for name, ref in elf_refs.items():
        elf = get(ref)
        if elf["elf"] is None:
            runs.append(ScheduleRun(name=name, stage="elf", log_tail=elf["log"][-2000:]))
            continue
        elf_name = f"braggnn_{name}.riscv"
        run_refs[name] = runner.run.options(resources=SIM_RUN_RESOURCES).chia_remote(
            runner, artifact_ref, elf["elf"], elf_name, SIM_WORK_DIR,
            plusargs={"+loadmem": elf_name},
            timeout_seconds=SIM_RUN_TIMEOUT_SECONDS,
            verbose=False,
        )
    for name, ref in run_refs.items():
        result = get(ref)
        # Program (UART / HTIF) output lands in the simulator's stdout (.log).
        text = (result.log or "") + "\n" + (result.out or "")
        run = _check_output(name, text, n_patches)
        if not result.success and run.matched:
            run.stage, run.matched = "sim", False
            run.mismatches = [f"simulator exited with {result.returncode}"]
        runs.append(run)

    failed = [run.name for run in runs if not run.matched]
    return VerilatorEvalResult(
        passed=not failed,
        reason=f"not bit-exact with the seed: {', '.join(failed)}" if failed else "",
        sim_build_log_tail=build_tail,
        runs=runs,
    )


def verilator_cycles(result: VerilatorEvalResult) -> Optional[float]:
    """One number to compare RTL variants by: the mean warm cycles/patch over
    the schedules run; None unless the check passed."""
    cycles = [run.warm_avg_cycles for run in result.runs
              if run.warm_avg_cycles is not None]
    if not result.passed or not cycles:
        return None
    return sum(cycles) / len(cycles)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--best", help="Extra schedule file to run next to the seed")
    parser.add_argument("--no-fusion", action="store_true",
                        help="Skip the fusion reference schedule")
    parser.add_argument("--patches", type=int, default=DEFAULT_PATCHES)
    parser.add_argument("--clean-build", action="store_true")
    parser.add_argument("--out", help="Write the result as JSON here")
    args = parser.parse_args()

    ray.init(address="auto")
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    result = evaluate_rtl(
        best_source=Path(args.best).read_text() if args.best else None,
        include_fusion=not args.no_fusion,
        n_patches=args.patches,
        clean_build=args.clean_build,
    )
    print(result.summary())
    if args.out:
        Path(args.out).write_text(json.dumps(asdict(result), indent=2))
    raise SystemExit(0 if result.passed else 1)


if __name__ == "__main__":
    main()
