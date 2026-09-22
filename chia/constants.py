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
LLM_TIMEOUT_SECONDS = 1800

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
# Chipyard checkout on the firesim node -- braggnn_loop.py edits Chisel and
# builds there directly. Override with FIRESIM_CHIPYARD_PATH in .env.
# ---------------------------------------------------------------------------
FIRESIM_CHIPYARD_PATH = os.environ.get("FIRESIM_CHIPYARD_PATH", "/nfs/app/chipyard")

CHIPYARD_WRITABLE_DIRS = [
    f"{FIRESIM_CHIPYARD_PATH}/generators/gemmini",
    f"{FIRESIM_CHIPYARD_PATH}/sims/firesim",
    f"{FIRESIM_CHIPYARD_PATH}/software/firemarshal",
]

# ---------------------------------------------------------------------------
# Chisel build (firesim.chisel_build -> ChiselBuildNode)
# ---------------------------------------------------------------------------
BUILD_CONFIG = "GemminiRocketConfig"
BUILD_CONFIG_PACKAGE = "chipyard"
CHISEL_BUILD_TIMEOUT_SECONDS = 60000
CHISEL_BUILD_MAKE_JOBS = 4

# ---------------------------------------------------------------------------
# Exo / BraggNN SW loop
# ---------------------------------------------------------------------------
GEMMINI_ROCC_TESTS_DIR = f"{FIRESIM_CHIPYARD_PATH}/generators/gemmini/software/gemmini-rocc-tests"
GEMMINI_PARAMS_H_PATH = f"{GEMMINI_ROCC_TESTS_DIR}/include/gemmini_params.h"

# Per-candidate work dirs on the exo_compiler container.
EXO_WORK_ROOT = "/home/ray/braggnn-alphaevolve"

# ---------------------------------------------------------------------------
# FireSim (bitstream registration / build; see firesim.py)
# ---------------------------------------------------------------------------
FIRESIM_DEPLOY_DIR = f"{FIRESIM_CHIPYARD_PATH}/sims/firesim/deploy"

# Name of the hwdb entry produced by config_build.yaml's build recipe --
# fixed across builds, so each new buildbitstream overwrites the same file
# at FIRESIM_HWDB_ENTRIES_DIR/FIRESIM_HWDB_ENTRY_NAME with the latest
# bitstream_tar path. We give the copy appended to config_hwdb.yaml a
# timestamped, unique key instead of reusing this name.
FIRESIM_HWDB_ENTRY_NAME = "alveo_u250_firesim_gemmini_rocket_singlecore_no_nic"
FIRESIM_HWDB_ENTRIES_DIR = f"{FIRESIM_DEPLOY_DIR}/built-hwdb-entries"
FIRESIM_CONFIG_HWDB_PATH = f"{FIRESIM_DEPLOY_DIR}/config_hwdb.yaml"
FIRESIM_CONFIG_RUNTIME_PATH = f"{FIRESIM_DEPLOY_DIR}/config_runtime.yaml"
FIRESIM_BUILD_TIMEOUT_SECONDS = 6 * 3600  # bitstream synthesis: hours
# Per candidate: infrasetup is the slow step; a normal BraggNN run takes ~1-2
# min, so a hung candidate must not hold the FPGA for long (it is killed here,
# not left running after the evaluator's own ray.get timeout gives up).
FIRESIM_INFRASETUP_TIMEOUT_SECONDS = 1800
FIRESIM_RUNWORKLOAD_TIMEOUT_SECONDS = 900
FIRESIM_KILL_TIMEOUT_SECONDS = 300

# ---------------------------------------------------------------------------
# Loop
# ---------------------------------------------------------------------------
MAX_ATTEMPTS = 1
