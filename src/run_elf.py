"""Run prebuilt BraggNN ELFs (e.g. from scripts/build_c_tune.sh) on FireSim.

Each ELF runs once on the deployed bitstream through firesim.run_workload.
Prints, per ELF: average cycles per patch, average error, how many patches'
predictions differ from the first ELF's (the reference -- give the unmodified
build first), and the per-layer marks if it was built with -DCHIA_LAYER_MARKS.
Paths must be readable by the job driver, which runs on the head node, so
pass absolute paths (the build/ directory is not part of the job upload).

    python src/run_elf.py /abs/build/braggnn_tune_opus-base/braggnn.riscv \\
                          /abs/build/braggnn_tune_opus-all/braggnn.riscv
"""

import argparse
import json
import logging
import re
from datetime import datetime
from pathlib import Path

import ray

from chia.base.ChiaFunction import get

from firesim import run_workload

logger = logging.getLogger(__name__)

_PATCH_RE = re.compile(r"^patch (\d+): cycles=(\d+) err=\(([-\d.]+), ([-\d.]+)\) px", re.M)
_AVG_RE = re.compile(r"Avg cycles: (\d+)")
_ERR_RE = re.compile(r"Avg error: \(([\d.]+), ([\d.]+)\)")
_MARK_RE = re.compile(r"^mark (\d+) (\S+): (\d+)", re.M)


def run(path: str, timeout: int) -> dict:
    elf = Path(path).read_bytes()
    # A binary that hangs Gemmini would otherwise hold the FPGA for
    # run_workload's default 4 h; firesim kill runs on timeout.
    result = get(run_workload.chia_remote(elf, timeout_seconds=timeout))
    uart = "\n".join(result.uartlogs.values())
    avg, err = _AVG_RE.search(uart), _ERR_RE.search(uart)
    return {
        "elf": path,
        "returncode": result.returncode,
        "avg_cycles": int(avg.group(1)) if avg else None,
        "avg_error": [float(err.group(1)), float(err.group(2))] if err else None,
        "patch_errors": {int(i): (x, y) for i, _, x, y in _PATCH_RE.findall(uart)},
        "marks": [(name, int(c)) for _, name, c in _MARK_RE.findall(uart)],
        # debug prints of a -DCHIA_DEBUG_* build
        "debug": [l for l in uart.splitlines() if l.startswith(("spadcheck", "debug"))],
        "uart_tail": uart[-1500:] if not avg else "",
    }


def report(results: list[dict]) -> str:
    ref = results[0]["patch_errors"] if results else {}
    lines = []
    for r in results:
        diff = sum(1 for i, e in r["patch_errors"].items() if ref.get(i) != e)
        lines.append(f"== {r['elf']}")
        if r["avg_cycles"] is None:
            lines.append(f"  no result (returncode {r['returncode']}):\n{r['uart_tail']}")
            continue
        lines.append(f"  avg cycles/patch {r['avg_cycles']:,}  avg error {r['avg_error']} px  "
                     f"patches differing from the first ELF: {diff}/{len(r['patch_errors'])}")
        for name, c in r["marks"]:
            lines.append(f"    {name:16s} {c:8,d}")
        for l in r.get("debug", []):
            lines.append(f"    {l}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("elfs", nargs="+", help="absolute paths; the first is the reference")
    parser.add_argument("--timeout", type=int, default=600,
                        help="seconds per ELF before the simulation is killed (default 600)")
    parser.add_argument("--out-dir", default=None,
                        help="default: ~/braggnn_loop_runs/<timestamp>_elf")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    if not ray.is_initialized():
        ray.init(address="auto")
    out = Path(args.out_dir or Path.home() / "braggnn_loop_runs"
               / f"{datetime.now():%Y%m%d_%H%M%S}_elf")
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for path in args.elfs:
        logger.info("Running %s", path)
        results.append(run(path, args.timeout))
        (out / "elf_runs.json").write_text(json.dumps(results, indent=2))
        (out / "elf_runs.txt").write_text(report(results) + "\n")
    print(report(results))
    logger.info("Saved to %s", out)


if __name__ == "__main__":
    main()
