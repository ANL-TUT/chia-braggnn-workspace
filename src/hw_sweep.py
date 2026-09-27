"""Sweep RTL parameter variants on the firesim node's Verilator.

For each patch (relative to the RTL currently on the node): apply it, rebuild
the simulator, run the given ELFs, revert it; then print one table of warm
cycles per (variant, ELF), with '!' marking predictions that differ from the
seed. The unpatched RTL runs first as the reference.

    python src/hw_sweep.py --patches rtl_patches/P_stat4.patch rtl_patches/P_st16.patch \\
        --elfs /abs/build/x/braggnn.riscv /abs/build/y/braggnn.riscv
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import ray

from chia.base.ChiaFunction import get
from chia.chipyard.verilator_run_node import VerilatorRunNode

from node_rtl import node_git
from verilator_elf import build_sim, parse, seed_mismatches
from verilator_eval import SIM_RUN_RESOURCES, SIM_WORK_DIR

logger = logging.getLogger(__name__)


def run_variant(elfs: dict, timeout: int) -> dict:
    artifact = build_sim(clean=False)
    if not artifact.success:
        return {p: {"error": "build failed"} for p in elfs}
    artifact_ref = ray.put(artifact)
    runner = VerilatorRunNode(logging_level=logging.INFO)
    refs = {}
    for i, (path, elf) in enumerate(elfs.items()):
        name = f"elf{i}_{Path(path).parent.name}.riscv"
        refs[path] = runner.run.options(resources=SIM_RUN_RESOURCES).chia_remote(
            runner, artifact_ref, elf, name, SIM_WORK_DIR,
            plusargs={"+loadmem": name}, timeout_seconds=timeout, verbose=False)
    out = {}
    for path, ref in refs.items():
        r = parse(path, get(ref))
        out[path] = {"warm_avg": r["warm_avg"], "patch_cycles": r["patch_cycles"],
                     "mismatches": seed_mismatches(r["patch_errors"]),
                     "patches": len(r["patch_errors"])}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--patches", nargs="+", required=True)
    parser.add_argument("--elfs", nargs="+", required=True)
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    if not ray.is_initialized():
        ray.init(address="auto")
    out = Path.home() / "braggnn_loop_runs" / f"{datetime.now():%Y%m%d_%H%M%S}_hwsweep"
    out.mkdir(parents=True, exist_ok=True)
    elfs = {p: Path(p).read_bytes() for p in args.elfs}

    results = {"(current)": run_variant(elfs, args.timeout)}
    (out / "results.json").write_text(json.dumps(results, indent=2))
    for patch in args.patches:
        text = Path(patch).read_text()
        applied = get(node_git.chia_remote("apply", text))
        if "Applied" not in applied:
            results[patch] = {"error": applied[-500:]}
            continue
        logger.info("Variant %s", patch)
        try:
            results[patch] = run_variant(elfs, args.timeout)
        finally:
            reverted = get(node_git.chia_remote("revert", text))
            if "Applied" not in reverted:
                raise SystemExit(f"could not revert {patch}:\n{reverted}")
        (out / "results.json").write_text(json.dumps(results, indent=2))

    names = [Path(p).parent.name.replace("braggnn_tune_opus-", "") for p in args.elfs]
    lines = ["variant".ljust(28) + "".join(n.rjust(16) for n in names)]
    for variant, res in results.items():
        cells = []
        for p in args.elfs:
            r = res.get(p, {})
            if "warm_avg" not in r or r["warm_avg"] is None:
                cells.append("error".rjust(16))
            else:
                bad = "!" if r["mismatches"] else " "
                cells.append(f"{r['warm_avg']:,.0f}{bad}".rjust(16))
        lines.append(Path(variant).stem.ljust(28) + "".join(cells))
    (out / "table.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    logger.info("Saved to %s", out)


if __name__ == "__main__":
    main()
