"""Run prebuilt BraggNN ELFs on the firesim node's Verilator simulator.

The Verilator counterpart of run_elf.py: builds the Chipyard simulator for
BUILD_CONFIG from the RTL currently in CHIPYARD_PATH's generators/gemmini
(incremental make, so an unchanged tree only re-checks), then runs every ELF
on it in parallel with +loadmem. Build the ELFs with -DEVAL_PATCHES=n (a few
patches; Verilator runs ~100x slower than FireSim). Reports per ELF the
per-patch cycles, the warm average (patch 0 is cold), how many patches'
predictions differ from the seed's (verilator_eval.SEED_PATCH_ERRORS), per-layer marks, and the tail of the log on failure
(Chisel assertions abort the simulation, unlike on FireSim).

    python src/verilator_elf.py /abs/build/x-base/braggnn.riscv /abs/build/x-new/braggnn.riscv
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import ray

from chia.base.ChiaFunction import ChiaFunction, get
from chia.chipyard.chisel_build_node import ChiselBuildNode
from chia.chipyard.state_def import BuildTarget
from chia.chipyard.verilator_run_node import VerilatorRunNode

from constants import BUILD_CONFIG, CHIPYARD_PATH
from run_elf import _MARK_RE, _PATCH_RE
from verilator_eval import (SEED_PATCH_ERRORS, SIM_BUILD_MAKE_JOBS,
                            SIM_BUILD_TIMEOUT_SECONDS, SIM_RUN_RESOURCES, SIM_WORK_DIR)

logger = logging.getLogger(__name__)


def build_sim(clean: bool):
    builder = ChiselBuildNode(
        chipyard_path=CHIPYARD_PATH,
        config=BUILD_CONFIG,
        target=BuildTarget.VERILATOR,
        make_jobs=SIM_BUILD_MAKE_JOBS,
        timeout_seconds=SIM_BUILD_TIMEOUT_SECONDS,
        clean=clean,
    )
    return get(builder.build.options(resources={"manager": 1}).chia_remote(builder))


@ChiaFunction(resources=SIM_RUN_RESOURCES)
def run_traced(artifact, elf: bytes, name: str, timeout: int, prefix: str):
    """Run with +verbose (the harness prints Chisel printfs only then, next to
    Rocket's per-instruction trace) and keep, of the huge stderr, only the
    lines that start with *prefix*, filtered here on the node."""
    runner = VerilatorRunNode(logging_level=logging.INFO)
    result = VerilatorRunNode.run(runner, artifact, elf, name, SIM_WORK_DIR,
                                  plusargs={"+loadmem": name}, timeout_seconds=timeout,
                                  verbose=True)
    result.out = "\n".join(l for l in (result.out or "").splitlines() if l.startswith(prefix))
    return result


def parse(path: str, result) -> dict:
    text = (result.log or "") + "\n" + (result.out or "")
    patches = sorted(((int(i), int(c), x, y) for i, c, x, y in _PATCH_RE.findall(text)))
    cycles = [c for _, c, _, _ in patches]
    warm = cycles[1:] or cycles
    return {
        "elf": path,
        "returncode": result.returncode,
        "patch_cycles": cycles,
        "warm_avg": sum(warm) / len(warm) if warm else None,
        "patch_errors": {i: (x, y) for i, _, x, y in patches},
        "marks": [(name, int(c)) for _, name, c in _MARK_RE.findall(text)],
        # With few patches braggnn_main.c's 0.5 px check can FAIL (exit 1)
        # although the run is fine; the log matters only when patches are missing.
        "log_tail": "" if patches else text[-2500:],
    }


def seed_mismatches(patch_errors: dict) -> int:
    return sum(1 for i, (x, y) in patch_errors.items()
               if (float(x), float(y)) != SEED_PATCH_ERRORS[int(i)])


def report(results: list[dict]) -> str:
    lines = []
    for r in results:
        lines.append(f"== {r['elf']}")
        if r["warm_avg"] is not None:
            lines.append(f"  warm avg {r['warm_avg']:,.0f} cycles/patch  per patch {r['patch_cycles']}  "
                         f"patches differing from the seed: "
                         f"{seed_mismatches(r['patch_errors'])}/{len(r['patch_errors'])}")
        for name, c in r["marks"]:
            lines.append(f"    {name:16s} {c:8,d}")
        if r["log_tail"]:
            lines.append(f"  simulator exited with {r['returncode']}:\n{r['log_tail']}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("elfs", nargs="+", help="absolute paths")
    parser.add_argument("--clean-build", action="store_true")
    parser.add_argument("--trace", metavar="PREFIX",
                        help="run with +verbose and save the stderr lines starting with PREFIX "
                             "(RTL printfs) in the per-ELF .log")
    parser.add_argument("--timeout", type=int, default=3 * 3600,
                        help="seconds per ELF before the simulation is killed")
    parser.add_argument("--out-dir", default=None,
                        help="default: ~/braggnn_loop_runs/<timestamp>_verilator")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    if not ray.is_initialized():
        ray.init(address="auto")
    out = Path(args.out_dir or Path.home() / "braggnn_loop_runs"
               / f"{datetime.now():%Y%m%d_%H%M%S}_verilator")
    out.mkdir(parents=True, exist_ok=True)

    elfs = {p: Path(p).read_bytes() for p in args.elfs}
    logger.info("Building the Verilator simulator for %s", BUILD_CONFIG)
    artifact = build_sim(args.clean_build)
    if not artifact.success:
        tail = (artifact.stdout + artifact.stderr)[-3000:]
        (out / "build_fail.txt").write_text(tail)
        raise SystemExit(f"Verilator build failed:\n{tail}")
    artifact_ref = ray.put(artifact)

    runner = VerilatorRunNode(logging_level=logging.INFO)
    refs = {}
    for i, (path, elf) in enumerate(elfs.items()):
        name = f"elf{i}_{Path(path).parent.name}.riscv"
        if args.trace:
            refs[path] = run_traced.chia_remote(artifact_ref, elf, name, args.timeout, args.trace)
            continue
        refs[path] = runner.run.options(resources=SIM_RUN_RESOURCES).chia_remote(
            runner, artifact_ref, elf, name, SIM_WORK_DIR,
            plusargs={"+loadmem": name}, timeout_seconds=args.timeout, verbose=False)
    results = []
    for i, (path, ref) in enumerate(refs.items()):
        result = get(ref)
        # the full simulator output, for RTL printf traces
        (out / f"elf{i}_{Path(path).parent.name}.log").write_text(
            (result.log or "") + "\n" + (result.out or ""))
        results.append(parse(path, result))
    (out / "runs.json").write_text(json.dumps(results, indent=2))
    (out / "runs.txt").write_text(report(results) + "\n")
    print(report(results))
    logger.info("Saved to %s", out)


if __name__ == "__main__":
    main()
