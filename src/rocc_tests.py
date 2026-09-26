"""Run gemmini-rocc-tests' bare-metal tests on the firesim node's Verilator.

A regression check for Gemmini RTL edits beyond BraggNN: builds the Verilator
simulator for BUILD_CONFIG from the RTL in CHIPYARD_PATH's generators/gemmini
(incremental), builds the bare-metal tests there with the node's toolchain
(their gemmini_params.h is the one the simulator build just generated), and
runs each test on the simulator. Each test checks Gemmini's results against a
CPU reference and exits 0 on success.

    python src/rocc_tests.py                  # the default correctness set
    python src/rocc_tests.py conv matmul_ws   # just these
"""

import argparse
import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path

import ray

from chia.base.ChiaFunction import ChiaFunction, get
from chia.chipyard.verilator_run_node import VerilatorRunNode

from constants import CHIPYARD_PATH
from verilator_elf import build_sim
from verilator_eval import SIM_WORK_DIR

logger = logging.getLogger(__name__)

TESTS_DIR = f"{CHIPYARD_PATH}/generators/gemmini/software/gemmini-rocc-tests"

# bareMetalC/Makefile's list without the performance / scaffolding programs
# (conv_perf, conv_dw_perf, tiled_matmul_ws_perf, gemmini_counter, template).
DEFAULT_TESTS = [
    "mvin_mvout", "mvin_mvout_zeros", "mvin_mvout_stride", "mvin_mvout_block_stride",
    "mvin_mvout_acc", "mvin_mvout_acc_zero_stride", "mvin_mvout_acc_stride",
    "mvin_mvout_acc_full", "mvin_mvout_acc_full_stride", "mvin_mvout_spad",
    "matmul_os", "matmul_ws", "matmul", "matmul_spad", "raw_hazard", "aligned", "padded",
    "mvin_scale", "conv", "conv_stride", "conv_rect", "conv_rect_pool", "conv_with_pool",
    "conv_with_rot180", "conv_with_kernel_dilation", "conv_with_input_dilation",
    "conv_with_input_dilation_and_rot180", "conv_with_input_dilation_and_neg_padding",
    "conv_trans_output_1203", "conv_trans_weight_1203", "conv_trans_weight_0132",
    "conv_trans_input_3120", "conv_trans_input_3120_with_kernel_dilation",
    "conv_first_layer", "conv_dw", "tiled_matmul_os", "tiled_matmul_ws", "tiled_matmul_ws_At",
    "tiled_matmul_ws_Bt", "tiled_matmul_ws_full_C", "tiled_matmul_ws_low_D",
    "tiled_matmul_ws_igelu", "tiled_matmul_ws_layernorm", "tiled_matmul_ws_softmax",
    "tiled_matmul_cpu", "tiled_matmul_option", "transpose", "matrix_add", "resadd",
    "resadd_stride", "global_average",
]


@ChiaFunction(resources={"manager": 0.01})
def build_tests(tests: list) -> dict:
    """Build <test>-baremetal for each test on the node; returns the ELFs
    (None where the build failed), the make log tail and the header's
    feature lines."""
    targets = " ".join(f"{t}-baremetal" for t in tests)
    script = f"""
set -e
source {CHIPYARD_PATH}/env.sh
cd {TESTS_DIR}
if [ ! -d build ]; then autoconf && mkdir build && cd build && ../configure && cd ..; fi
make -C build/bareMetalC -k -j16 BAREMETAL_ONLY=1 {targets}
"""
    done = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    elfs = {}
    for t in tests:
        path = Path(TESTS_DIR, "build", "bareMetalC", f"{t}-baremetal")
        elfs[t] = path.read_bytes() if path.exists() else None
    header = Path(TESTS_DIR, "include", "gemmini_params.h").read_text()
    features = [l for l in header.splitlines()
                if l.startswith(("#define HAS_", "#define NORM_STAT_IDS", "#define DIM ",
                                 "#define BANK_NUM", "#define ACC_ROWS"))]
    return {"elfs": elfs, "returncode": done.returncode,
            "log": (done.stdout + done.stderr)[-4000:], "header": features}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tests", nargs="*", default=DEFAULT_TESTS)
    parser.add_argument("--timeout", type=int, default=1800,
                        help="seconds per test before the simulation is killed")
    parser.add_argument("--parallel", type=int, default=4,
                        help="simulations at once on the node (share of its 'manager' resource)")
    parser.add_argument("--out-dir", default=None,
                        help="default: ~/braggnn_loop_runs/<timestamp>_rocc")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    if not ray.is_initialized():
        ray.init(address="auto")
    out = Path(args.out_dir or Path.home() / "braggnn_loop_runs"
               / f"{datetime.now():%Y%m%d_%H%M%S}_rocc")
    out.mkdir(parents=True, exist_ok=True)

    # The simulator build first: it (re)generates the tests' gemmini_params.h.
    artifact = build_sim(clean=False)
    if not artifact.success:
        raise SystemExit("Verilator build failed:\n" + (artifact.stdout + artifact.stderr)[-3000:])
    artifact_ref = ray.put(artifact)

    built = get(build_tests.chia_remote(args.tests))
    logger.info("gemmini_params.h: %s", "; ".join(built["header"]))
    (out / "build.log").write_text(built["log"])

    runner = VerilatorRunNode(logging_level=logging.INFO)
    refs = {}
    results = {}
    for t, elf in built["elfs"].items():
        if elf is None:
            results[t] = {"status": "BUILD FAILED"}
            continue
        name = f"{t}-baremetal"
        refs[t] = runner.run.options(resources={"manager": 1.0 / args.parallel}).chia_remote(
            runner, artifact_ref, elf, name, SIM_WORK_DIR,
            plusargs={"+loadmem": name}, timeout_seconds=args.timeout, verbose=False)
    for t, ref in refs.items():
        r = get(ref)
        text = (r.log or "") + "\n" + (r.out or "")
        (out / f"{t}.log").write_text(text)
        results[t] = {"status": "PASS" if r.success else f"FAIL (exit {r.returncode})",
                      "tail": "" if r.success else text[-1500:]}
    (out / "results.json").write_text(json.dumps({"header": built["header"],
                                                 "results": results}, indent=2))
    passed = [t for t, r in results.items() if r["status"] == "PASS"]
    lines = [f"gemmini_params.h: {'; '.join(built['header'])}",
             f"{len(passed)}/{len(results)} passed"]
    lines += [f"  {t}: {r['status']}" for t, r in results.items() if r["status"] != "PASS"]
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    logger.info("Saved to %s", out)


if __name__ == "__main__":
    main()
