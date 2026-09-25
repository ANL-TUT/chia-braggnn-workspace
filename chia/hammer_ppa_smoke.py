"""Smoke test of hammer_ppa.PpaJob on the current, unedited Chisel.

Runs the same Hammer steps as one attempt of the HW loop (buildfile ->
gemmini_params.h check -> syn -> report parsing) without the LLM, the
bitstream build or FireSim, and prints the PPA summary.
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path

import ray

from chia.base.ChiaFunction import get

from dumper import Dumper
from exo_compiler import check_exo_compatible
from firesim import read_gemmini_params_h
from hammer_ppa import PpaJob


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        default=str(
            Path.home() / "braggnn_loop_runs"
            / f"hammer_smoke_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        ),
    )
    args = parser.parse_args()

    ray.init(address="auto")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    dump = Dumper(args.out_dir)
    job = PpaJob(dump, 0)

    elab = job.elaborate()
    if elab is None:
        raise SystemExit(f"vlsi worker unavailable; see {job.out / 'summary.json'}")
    if not elab.success:
        raise SystemExit(f"buildfile failed:\n{elab.stderr[-4000:]}")
    print(f"buildfile OK -> {job.obj_dir}", flush=True)

    params_h = get(
        read_gemmini_params_h.options(resources={"manager": 0.05}).chia_remote()
    )
    dump.text("gemmini_params.h", params_h)
    print(f"check_exo_compatible: {check_exo_compatible(params_h) or 'OK'}", flush=True)

    job.start_syn()
    ppa = job.result()
    print(f"syn {'OK' if ppa.success else f'FAILED at {ppa.stage}'} -> {job.out}")
    print(ppa.summary())
    if not ppa.success:
        raise SystemExit(ppa.stderr_tail)


if __name__ == "__main__":
    main()
