import json
import logging
import os
import re
import signal
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from ray import ObjectRef

from chia.base.ChiaFunction import ChiaFunction, get
from chia.base.tools.BashTool import BashTool

from chipyard_ops import collect_diff
from constants import (
    CHIPYARD_DIFF_SUBMODULES,
    CHIPYARD_PATH,
    FIRESIM_BUILD_TIMEOUT_SECONDS,
    FIRESIM_CONFIG_HWDB_PATH,
    FIRESIM_CONFIG_RUNTIME_PATH,
    FIRESIM_DEPLOY_DIR,
    FIRESIM_HWDB_ENTRIES_DIR,
    FIRESIM_HWDB_ENTRY_NAME,
    GEMMINI_PARAMS_H_PATH,
)
from dumper import Dumper

logger = logging.getLogger(__name__)

FIRESIM_DIR = f"{CHIPYARD_PATH}/sims/firesim"
HWDB = FIRESIM_CONFIG_HWDB_PATH
BUILD_RECIPES = f"{FIRESIM_DEPLOY_DIR}/config_build_recipes.yaml"


BOOT_BINARY_NAME = "chia-bare-bin"
BOOT_BINARY = Path(FIRESIM_DEPLOY_DIR) / "workloads" / "chia-bare" / BOOT_BINARY_NAME


def stage_bare_workload(elf: bytes) -> None:
    workloads = Path(FIRESIM_DIR) / "deploy" / "workloads"
    (workloads / "chia-bare").mkdir(parents=True, exist_ok=True)
    BOOT_BINARY.write_bytes(elf)
    (workloads / "chia-bare" / f"{BOOT_BINARY_NAME}-dwarf").write_bytes(elf)
    (workloads / "chia-bare.json").write_text(
        json.dumps(
            {
                "benchmark_name": "chia-bare",
                "common_bootbinary": BOOT_BINARY_NAME,
                "common_rootfs": None,
                "common_outputs": [],
                "common_simulation_outputs": ["uartlog"],
            }
        )
    )


@dataclass
class RunResult:
    returncode: int
    stdout: str
    stderr: str
    uartlogs: dict[str, str]


def bash(script: str, timeout: int) -> subprocess.CompletedProcess:
    """Run *script* in the FireSim manager env. It gets its own process group,
    and a timeout kills that whole group: killing only bash would leave the
    firesim manager and Vivado running as orphans in the shared build dir.
    Raises subprocess.TimeoutExpired (with the output so far) on timeout."""
    prefix = f"cd {FIRESIM_DIR} && source sourceme-manager.sh && "
    proc = subprocess.Popen(
        ["bash", "-l", "-c", prefix + script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        stdout, stderr = proc.communicate()
        raise subprocess.TimeoutExpired(proc.args, timeout, output=stdout, stderr=stderr)
    return subprocess.CompletedProcess(proc.args, proc.returncode, stdout, stderr)


@ChiaFunction(resources={"FPGA": 1})
def run_workload(elf: bytes, timeout_seconds: int = 14400) -> RunResult:
    stage_bare_workload(elf)

    try:
        done = bash(
            f"firesim infrasetup -a {HWDB} -r {BUILD_RECIPES} && "
            f"firesim runworkload -a {HWDB} -r {BUILD_RECIPES}",
            timeout_seconds,
        )
        returncode, stdout, stderr = done.returncode, done.stdout, done.stderr
    except subprocess.TimeoutExpired as expired:
        returncode = 124
        killed = bash(f"firesim kill -a {HWDB} -r {BUILD_RECIPES}", 600)
        stdout = f"{expired.stdout or ''}{killed.stdout}"
        stderr = f"{expired.stderr or ''}{killed.stderr}"

    found = re.search(r"See results in:\s*(\S+)", stdout)
    results_dir = Path(found.group(1)) if found else None
    uartlogs = (
        {
            path.parent.name: path.read_text()
            for path in sorted(results_dir.glob("*/uartlog"))
        }
        if results_dir
        else {}
    )

    return RunResult(
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
        uartlogs=uartlogs,
    )


# ── HW loop helpers (Chisel edit/build, bitstream) ───────────────────

# System dirs bind-mounted read-only into the sandbox so the shell and basic
# tools can still exec; everything else on the firesim node is invisible.
_SANDBOX_RO_SYSTEM_DIRS = ["/usr", "/bin", "/sbin", "/lib", "/lib64", "/etc", "/opt"]


class SandboxedBashTool(BashTool):
    """chipyard_bash, confined with bubblewrap: nothing outside work_dir is
    visible, and only writable_dirs (under work_dir) are writable -- the
    rest of work_dir is read-only."""

    def __init__(
        self, name: str, work_dir: str, writable_dirs: list[str],
        timeout_seconds: int = 120, task_options: Optional[dict] = None,
    ):
        self.writable_dirs = writable_dirs
        super().__init__(name, work_dir=work_dir, timeout_seconds=timeout_seconds,
                          task_options=task_options)

    def run_command(self, command: str) -> str:
        # --unshare-pid: without it the command sees and can signal every
        # process of this user on the node; the LLM has twice run
        # `kill -9 -1` after a command timed out, which killed the raylet
        # and this tool's own server.
        bwrap_cmd = ["bwrap", "--die-with-parent", "--unshare-pid", "--dev", "/dev",
                     "--proc", "/proc", "--tmpfs", "/tmp"]
        for d in _SANDBOX_RO_SYSTEM_DIRS:
            if os.path.isdir(d):
                bwrap_cmd += ["--ro-bind", d, d]
        bwrap_cmd += ["--ro-bind", self.work_dir, self.work_dir]
        for d in self.writable_dirs:
            bwrap_cmd += ["--bind", d, d]
        bwrap_cmd += ["--chdir", self.work_dir]
        bwrap_cmd += ["sh", "-c", command]

        try:
            proc = subprocess.Popen(
                bwrap_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
        except Exception as e:
            return f"Error: {e}"

        try:
            stdout, stderr = proc.communicate(timeout=self.timeout_seconds)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                proc.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            return f"Error: command timed out after {self.timeout_seconds}s"
        except Exception as e:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            return f"Error: {e}"

        output = ""
        if stdout:
            output += stdout
        if stderr:
            output += "\nSTDERR:\n" + stderr
        if proc.returncode != 0:
            output += f"\n[exit code: {proc.returncode}]"
        return output or "(no output)"


@ChiaFunction(resources={"manager": 0.05})
def read_gemmini_params_h() -> str:
    with open(GEMMINI_PARAMS_H_PATH) as f:
        return f.read()


def collect_chisel_diff(
    dump: Dumper, attempt: int, baseline: dict[str, str] | None = None,
) -> dict[str, str] | None:
    try:
        err, diffs = get(
            collect_diff.options(resources={"manager": 0.05}).chia_remote(
                CHIPYARD_PATH, CHIPYARD_DIFF_SUBMODULES, baseline
            )
        )
    except Exception as e:  # noqa: BLE001 - diagnostic only
        logger.warning("collect_diff failed (attempt %d): %s", attempt, e)
        return None
    if err:
        logger.warning("collect_diff returned error=%d (attempt %d)", err, attempt)
        return None
    dump.json(f"chisel_diff_attempt{attempt}.json", diffs)
    parts = []
    for repo, text in diffs.items():
        if not text:
            continue
        label = repo or "(chipyard root)"
        parts.append(f"# ===== diff: {label} =====\n{text}")
    dump.text(f"chisel_diff_attempt{attempt}.diff", "\n\n".join(parts))
    logger.info("Collected chisel diff (attempt %d): %d repo(s) changed",
                attempt, sum(1 for t in diffs.values() if t))
    return diffs


def _register_hwdb_entry() -> tuple[bool, str]:
    """After a successful buildbitstream, firesim writes the new hwdb entry
    (same fixed name every build) to FIRESIM_HWDB_ENTRIES_DIR. Copy it into
    config_hwdb.yaml under a timestamped, unique key, and point
    config_runtime.yaml's default_hw_config at that key."""
    entry_path = os.path.join(FIRESIM_HWDB_ENTRIES_DIR, FIRESIM_HWDB_ENTRY_NAME)
    if not os.path.exists(entry_path):
        return (False, f"hwdb entry file not found: {entry_path}")
    with open(entry_path) as f:
        block = f.read()

    key_line = f"{FIRESIM_HWDB_ENTRY_NAME}:"
    if not block.startswith(key_line):
        return (False, f"unexpected hwdb entry format:\n{block}")

    unique_name = f"{FIRESIM_HWDB_ENTRY_NAME}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    unique_block = block.replace(key_line, f"{unique_name}:", 1)

    with open(FIRESIM_CONFIG_HWDB_PATH, "a") as f:
        f.write("\n" + unique_block)

    with open(FIRESIM_CONFIG_RUNTIME_PATH) as f:
        runtime_text = f.read()
    new_runtime_text, n = re.subn(
        r"(?m)^(\s*default_hw_config:\s*)\S+",
        lambda m: m.group(1) + unique_name,
        runtime_text, count=1,
    )
    if n == 0:
        return (False, f"default_hw_config: line not found in {FIRESIM_CONFIG_RUNTIME_PATH}")
    with open(FIRESIM_CONFIG_RUNTIME_PATH, "w") as f:
        f.write(new_runtime_text)

    return (True, unique_name)


@ChiaFunction(resources={"FPGA": 1})
def firesim_buildbitstream() -> tuple[bool, str, str]:
    try:
        r = bash("firesim buildbitstream", FIRESIM_BUILD_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired as expired:
        return (False, expired.stdout or "", (expired.stderr or "") + (
            f"\n[bitstream build timed out after {FIRESIM_BUILD_TIMEOUT_SECONDS // 3600} h "
            "and was killed; the design is likely too large for the FPGA flow]"
        ))
    if r.returncode != 0:
        return (False, r.stdout, r.stderr)

    reg_ok, reg_msg = _register_hwdb_entry()
    if not reg_ok:
        return (False, r.stdout, r.stderr + f"\n[hwdb registration failed: {reg_msg}]")
    return (True, r.stdout, r.stderr + f"\n[registered hwdb entry: {reg_msg}]")


def start_bitstream_build() -> ObjectRef:
    logger.info("Starting FireSim bitstream build in the background")
    return firesim_buildbitstream.options(resources={"FPGA": 1}).chia_remote()


@ChiaFunction(resources={"manager": 0.05})
def firesim_buildbitstream_mock() -> tuple[bool, str, str]:
    """Test double for firesim_buildbitstream: skips the real (hours-long)
    Vivado run and reuses the hwdb entry file already left behind by a
    previous real build (FIRESIM_HWDB_ENTRIES_DIR/FIRESIM_HWDB_ENTRY_NAME,
    pointing at an already-built bitstream_tar), exercising the real
    _register_hwdb_entry() logic against it."""
    reg_ok, reg_msg = _register_hwdb_entry()
    if not reg_ok:
        return (False, "", f"[mock] hwdb registration failed: {reg_msg}")
    return (True, "[mock] buildbitstream skipped", f"[mock] registered hwdb entry: {reg_msg}")
