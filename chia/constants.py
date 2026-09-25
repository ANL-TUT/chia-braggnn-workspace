import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Repo-relative paths (this package's own dir + the chia framework checkout)
# ---------------------------------------------------------------------------
PACKAGE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = Path(__file__).resolve().parents[1]   # workspace root (PACKAGE_DIR is chia/)

load_dotenv(_REPO_ROOT / ".env")

# ---------------------------------------------------------------------------
# Prompts (loaded from prompts/, with ${VAR} placeholders substituted)
# ---------------------------------------------------------------------------
PROMPTS_DIR = _REPO_ROOT / "prompts"
# The shipped Exo program (seed) and its gemmini.py, read on the driver.
EXO_DIR = _REPO_ROOT / "exo"

# ---------------------------------------------------------------------------
# LLM (HW-loop implement/debug agent)
# ---------------------------------------------------------------------------
OPENCODE_MODEL = "google-vertex/gemini-3.8-flash"
LLM_SYSTEM_MESSAGE = (
    "You are an expert Chisel / RISC-V engineer specializing in RoCC "
    "accelerators for the Chipyard / Rocket / Gemmini ecosystem."
)
LLM_TIMEOUT_SECONDS = 3600

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
DEFAULT_OUTPUT_BASE = "./results"

# ---------------------------------------------------------------------------
# Chisel diff capture
# ---------------------------------------------------------------------------
# Submodules collect_diff inspects in addition to the root chipyard repo. The
# accelerator + config edits land in generators/chipyard (the root repo, always
# captured under the "" key); these are included so a debugger edit that strays
# into a submodule (e.g. BOOM) is still recorded.
CHIPYARD_DIFF_SUBMODULES = [
    "generators/gemmini",
    "generators/rocket-chip",
    "generators/rocket-chip-inclusive-cache",
    "generators/rocket-chip-blocks",
]

# ---------------------------------------------------------------------------
# Chipyard checkout (NFS), shared by the firesim node -- braggnn_loop.py edits
# Chisel and builds there directly -- and the vlsi node, which mounts it at
# the same path. Set CHIPYARD_PATH in .env; cluster.yaml uses it for that
# mount too.
# ---------------------------------------------------------------------------
CHIPYARD_PATH = os.environ.get("CHIPYARD_PATH", "/nfs/app/chipyard")

CHIPYARD_WRITABLE_DIRS = [
    f"{CHIPYARD_PATH}/generators/gemmini",
    f"{CHIPYARD_PATH}/sims/firesim",
    f"{CHIPYARD_PATH}/software/firemarshal",
]

# ---------------------------------------------------------------------------
# Chipyard CONFIG the HW loop elaborates (hammer_ppa.PpaJob)
# ---------------------------------------------------------------------------
BUILD_CONFIG = "GemminiRocketConfig"

# ---------------------------------------------------------------------------
# Exo / BraggNN SW loop
# ---------------------------------------------------------------------------
GEMMINI_ROCC_TESTS_DIR = f"{CHIPYARD_PATH}/generators/gemmini/software/gemmini-rocc-tests"
GEMMINI_PARAMS_H_PATH = f"{GEMMINI_ROCC_TESTS_DIR}/include/gemmini_params.h"

# FireSim (bitstream registration / build; see firesim.py)
# ---------------------------------------------------------------------------
FIRESIM_DEPLOY_DIR = f"{CHIPYARD_PATH}/sims/firesim/deploy"

# Name of the hwdb entry produced by config_build.yaml's build recipe --
# fixed across builds, so each new buildbitstream overwrites the same file
# at FIRESIM_HWDB_ENTRIES_DIR/FIRESIM_HWDB_ENTRY_NAME with the latest
# bitstream_tar path. We give the copy appended to config_hwdb.yaml a
# timestamped, unique key instead of reusing this name.
FIRESIM_HWDB_ENTRY_NAME = "alveo_u250_firesim_full_gemmini_rocket_singlecore_no_nic"
FIRESIM_HWDB_ENTRIES_DIR = f"{FIRESIM_DEPLOY_DIR}/built-hwdb-entries"
FIRESIM_CONFIG_HWDB_PATH = f"{FIRESIM_DEPLOY_DIR}/config_hwdb.yaml"
FIRESIM_CONFIG_RUNTIME_PATH = f"{FIRESIM_DEPLOY_DIR}/config_runtime.yaml"
# bitstream synthesis: the unedited design takes ~3.6 h and a 1.4x larger
# RTL ~4 h; one that is still running at 8 h (e.g. RTL doubled,
# synth_design elaborating for hours) is treated as too large for the FPGA flow.
FIRESIM_BUILD_TIMEOUT_SECONDS = 8 * 3600
# The FPGA RTL that the build's replace_rtl step writes (~10 min in), on NFS
# under config_build.yaml's default_build_dir. Its size predicts whether
# Vivado can finish: 27 MB (unedited), 37.8 MB and 45.2 MB built in 3.6,
# 4 and 4.5 h (Vivado's RTL elaboration: 2, 16 and 37 min), while two
# 53-54 MB designs stayed in that elaboration for hours and never finished.
# A build whose RTL is over the limit is cancelled and fed back as too large.
FIRESIM_GENERATED_SV = (
    "/nfs/app/firesim-build/platforms/xilinx_alveo_u250/"
    "cl_xilinx_alveo_u250-firesim-FireSim-FireSimGemminiRocketConfig-"
    "BaseXilinxAlveoU250Config/design/FireSim-generated.sv"
)
FPGA_RTL_SIZE_LIMIT_BYTES = 50_000_000
# Per candidate: infrasetup is the slow step; a normal BraggNN run takes ~1-2
# min, so a hung candidate must not hold the FPGA for long (it is killed here,
# not left running after the evaluator's own ray.get timeout gives up).
FIRESIM_INFRASETUP_TIMEOUT_SECONDS = 1800
FIRESIM_RUNWORKLOAD_TIMEOUT_SECONDS = 900
FIRESIM_KILL_TIMEOUT_SECONDS = 300


# ---------------------------------------------------------------------------
# VLSI PPA (Hammer + Genus + sky130 + SRAM22/OpenRAM macros).
# ---------------------------------------------------------------------------
VLSI_TOP = "ChipTop"
# site-sky130.yml and openram-sky130.yml are generated by the chia-sky130
# image's entrypoint.sh at container start (from the baked-in sky130A /
# sram22 / OpenRAM paths) -- not ours to write. MacroCompiler maps
# single-port memories onto SRAM22 and dual-port ones (e.g. Gemmini's
# accumulator with acc_singleported = false) onto OpenRAM 1rw1r macros; the
# latter also needs the OpenRAM entries in the Hammer plugin's
# sram-cache.json, which the container merges in when started with
# WITH_OPENRAM=1 (cluster.yaml). See gemmini-sky130-sram22's
# docs/openram-1r1w.md.
# No retime-module.yml: retiming forces the flattened AccumulatorScale into
# one serial Genus partition that takes hours (and grows with the datapath,
# e.g. has_normalizations), so the loop synthesizes without it; expect
# negative slack even on the unedited design.
# See gemmini-sky130-sram22's README (env-setup) "Retiming (optional)".
VLSI_INPUT_CONFS = [
    "example-tools.yml", "example-sky130.yml", "site-sky130.yml",
    "openram-sky130.yml",
]
VLSI_OBJ_DIR_ROOT = f"{CHIPYARD_PATH}/vlsi/build-chia-ppa"
VLSI_BUILDFILE_TIMEOUT_SECONDS = 1800
# How long to wait for the vlsi worker's {"chipyard": 1} slot before giving
# up on PPA for the attempt.
VLSI_PG_READY_TIMEOUT_SECONDS = 600
# make syn stops after the syn_map reports (hammer_ppa.SYN_STOP_AFTER_STEP),
# ~3.5 h in; a full run's syn_opt did not finish in 6 h without retiming.
VLSI_SYN_TIMEOUT_SECONDS = 8 * 3600

# ---------------------------------------------------------------------------
# Loop
# ---------------------------------------------------------------------------
MAX_ATTEMPTS = 1
