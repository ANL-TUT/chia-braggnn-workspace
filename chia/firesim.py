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
from chia.chipyard.chisel_build_node import ChiselBuildNode
from chia.chipyard.state_def import BuildArtifact, BuildTarget

from chipyard_ops import collect_diff
from constants import (
    BUILD_CONFIG,
    BUILD_CONFIG_PACKAGE,
    CHIPYARD_DIFF_SUBMODULES,
    CHISEL_BUILD_MAKE_JOBS,
    CHISEL_BUILD_TIMEOUT_SECONDS,
    FIRESIM_BUILD_TIMEOUT_SECONDS,
    FIRESIM_CHIPYARD_PATH,
    FIRESIM_CONFIG_HWDB_PATH,
    FIRESIM_CONFIG_RUNTIME_PATH,
    FIRESIM_DEPLOY_DIR,
    FIRESIM_HWDB_ENTRIES_DIR,
    FIRESIM_HWDB_ENTRY_NAME,
    FIRESIM_INFRASETUP_TIMEOUT_SECONDS,
    FIRESIM_KILL_TIMEOUT_SECONDS,
    FIRESIM_RUNWORKLOAD_TIMEOUT_SECONDS,
    GEMMINI_PARAMS_H_PATH,
)
from dumper import Dumper

logger = logging.getLogger(__name__)

FIRESIM_DIR = f"{FIRESIM_CHIPYARD_PATH}/sims/firesim"
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
    prefix = f"cd {FIRESIM_DIR} && source sourceme-manager.sh && "
    return subprocess.run(
        ["bash", "-l", "-c", prefix + script],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _firesim_kill() -> str:
    """Best-effort `firesim kill`: tears down any simulation still holding the
    FPGA/XDMA module (e.g. from a hung or timed-out runworkload) so the next
    infrasetup can unload the driver. Never raises."""
    try:
        r = bash(f"firesim kill -a {HWDB} -r {BUILD_RECIPES}", FIRESIM_KILL_TIMEOUT_SECONDS)
        return r.stdout + r.stderr
    except Exception as e:  # noqa: BLE001 -- cleanup must not mask the real result
        return f"[chia] firesim kill failed: {type(e).__name__}: {e}"


# ── Fast relaunch: infrasetup once per hardware build, then only swap the boot
# binary and `firesim runworkload` per candidate. infrasetup is the slow, fragile
# step (build driver, unload XDMA, flash the FPGA, reload XDMA); the bare-metal
# ELF is passed to the sim driver as `+prog0=<name>` from the sim slot directory,
# so a new candidate only needs that one file replaced. Anything we cannot verify
# falls back to the full infrasetup path.
FIRESIM_FAST_RELAUNCH = os.environ.get("FIRESIM_FAST_RELAUNCH", "1") != "0"
FIRESIM_INFRASETUP_STAMP = os.path.join(FIRESIM_DEPLOY_DIR, ".chia_infrasetup_stamp")
_SSH_OPTS = ["-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20"]


def _current_hw_config() -> Optional[str]:
    """default_hw_config from config_runtime.yaml -- _register_hwdb_entry gives it
    a unique timestamped name per hardware build, so it identifies which
    bitstream the last infrasetup flashed."""
    try:
        with open(FIRESIM_CONFIG_RUNTIME_PATH) as f:
            m = re.search(r"(?m)^\s*default_hw_config:\s*(\S+)", f.read())
        return m.group(1) if m else None
    except OSError:
        return None


def _infrasetup_is_current() -> bool:
    hw = _current_hw_config()
    if hw is None:
        return False
    try:
        with open(FIRESIM_INFRASETUP_STAMP) as f:
            return f.read().strip() == hw
    except OSError:
        return False


def _write_infrasetup_stamp() -> None:
    hw = _current_hw_config()
    if hw:
        with open(FIRESIM_INFRASETUP_STAMP, "w") as f:
            f.write(hw)


def _clear_infrasetup_stamp() -> None:
    try:
        os.remove(FIRESIM_INFRASETUP_STAMP)
    except OSError:
        pass


def _runfarm_target() -> tuple:
    """(host, sim_dir) of the run farm host: env overrides FIRESIM_RUNFARM_HOST /
    FIRESIM_SIM_DIR, else parsed from config_runtime.yaml (recipe_arg_overrides)
    and its base recipe."""
    import yaml

    host = os.environ.get("FIRESIM_RUNFARM_HOST")
    sim_dir = os.environ.get("FIRESIM_SIM_DIR")
    if host and sim_dir:
        return host, sim_dir
    with open(FIRESIM_CONFIG_RUNTIME_PATH) as f:
        cfg = yaml.safe_load(f)
    run_farm = cfg.get("run_farm", {})
    overrides = run_farm.get("recipe_arg_overrides") or {}
    hosts = overrides.get("run_farm_hosts_to_use")
    sim_dir = sim_dir or overrides.get("default_simulation_dir")
    if not (host or hosts) or not sim_dir:
        with open(os.path.join(FIRESIM_DEPLOY_DIR, run_farm["base_recipe"])) as f:
            base_args = (yaml.safe_load(f) or {}).get("args", {})
        hosts = hosts or base_args.get("run_farm_hosts_to_use")
        sim_dir = sim_dir or base_args.get("default_simulation_dir")
    if not host:
        first = hosts[0]
        host = next(iter(first)) if isinstance(first, dict) else str(first)
    if not (host and sim_dir):
        raise RuntimeError("cannot determine run farm host / sim dir")
    return host, sim_dir


def _swap_bootbinary() -> tuple:
    """Copy the freshly staged boot binary over the one already in the run farm's
    sim slot. Returns (ok, message)."""
    try:
        host, sim_dir = _runfarm_target()
        slot = f"{sim_dir.rstrip('/')}/sim_slot_0"
        is_local = host.split("@")[-1] in ("localhost", "127.0.0.1")

        def sh(cmd: str):
            argv = ["bash", "-c", cmd] if is_local else ["ssh", *_SSH_OPTS, host, cmd]
            return subprocess.run(argv, capture_output=True, text=True, timeout=120)

        # The slot's copy is named "<jobname>-<basename>"; require exactly one.
        ls = sh(f"ls {slot}/*-{BOOT_BINARY_NAME} {slot}/rsyncdir/*-{BOOT_BINARY_NAME} 2>/dev/null")
        remote_files = [l for l in ls.stdout.split() if l]
        in_slot = [p for p in remote_files if "/rsyncdir/" not in p]
        if len(in_slot) != 1:
            return False, f"expected one boot binary in {slot}, found {in_slot!r}"
        for dest in remote_files:
            if is_local:
                r = subprocess.run(["cp", str(BOOT_BINARY), dest], capture_output=True, text=True, timeout=120)
            else:
                r = subprocess.run(
                    ["scp", *_SSH_OPTS, str(BOOT_BINARY), f"{host}:{dest}"],
                    capture_output=True, text=True, timeout=120,
                )
            if r.returncode != 0:
                return False, f"copy to {dest} failed: {r.stderr[-300:]}"
        return True, f"swapped {os.path.basename(in_slot[0])} on {host}"
    except Exception as e:  # noqa: BLE001 -- any doubt -> fall back to full infrasetup
        return False, f"{type(e).__name__}: {e}"


# "[user@host] Slot 0, Job <name> completed!" -- printed by the runworkload
# monitor once the target program has exited, before results are copied back.
_JOB_COMPLETED_RE = re.compile(r"Slot \d+, Job \S+ completed!")


@ChiaFunction(resources={"FPGA": 1})
def run_workload(elf: bytes) -> RunResult:
    stage_bare_workload(elf)

    print("[firesim] firesim kill (clear leftovers)", flush=True)
    _firesim_kill()

    fast = False
    infra_stdout = infra_stderr = ""
    if FIRESIM_FAST_RELAUNCH and _infrasetup_is_current():
        ok, msg = _swap_bootbinary()
        print(f"[firesim] fast relaunch (skip infrasetup): {msg}", flush=True)
        fast = ok
    if not fast:
        print("[firesim] firesim infrasetup", flush=True)
        _clear_infrasetup_stamp()
        infrasetup = f"firesim infrasetup -a {HWDB} -r {BUILD_RECIPES}"
        infra = bash(infrasetup, FIRESIM_INFRASETUP_TIMEOUT_SECONDS)
        if infra.returncode != 0 and "firesim-remove-xdma-module" in infra.stdout:
            # XDMA still busy: a previous simulation is holding it. Kill and retry once.
            print("[firesim] infrasetup hit busy XDMA -- kill + retry", flush=True)
            _firesim_kill()
            infra = bash(infrasetup, FIRESIM_INFRASETUP_TIMEOUT_SECONDS)
        if infra.returncode != 0:
            return RunResult(
                returncode=infra.returncode, stdout=infra.stdout, stderr=infra.stderr,
                uartlogs={},
            )
        if FIRESIM_FAST_RELAUNCH:
            _write_infrasetup_stamp()
        infra_stdout, infra_stderr = infra.stdout, infra.stderr

    print("[firesim] firesim runworkload", flush=True)
    try:
        done = bash(
            f"firesim runworkload -a {HWDB} -r {BUILD_RECIPES}",
            FIRESIM_RUNWORKLOAD_TIMEOUT_SECONDS,
        )
        returncode, stdout, stderr = done.returncode, done.stdout, done.stderr
    except subprocess.TimeoutExpired as expired:
        # The candidate's bare-metal program hung; without this the simulation
        # keeps the FPGA/XDMA and every later infrasetup fails.
        _clear_infrasetup_stamp()
        kill_out = _firesim_kill()

        def text(v):
            return v.decode(errors="replace") if isinstance(v, bytes) else (v or "")

        return RunResult(
            returncode=124,
            stdout=infra_stdout + text(expired.stdout),
            stderr=(
                f"[chia] firesim runworkload timed out after "
                f"{FIRESIM_RUNWORKLOAD_TIMEOUT_SECONDS}s (candidate likely hangs); "
                f"ran firesim kill:\n{kill_out}"
            ),
            uartlogs={},
        )
    if returncode != 0 and _JOB_COMPLETED_RE.search(stdout) is None:
        _clear_infrasetup_stamp()
        _firesim_kill()

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
    if not any("Avg cycles" in log for log in uartlogs.values()):
        # A run that produced no result (driver abort, hung sim, crashed program)
        # can leave the FPGA/XDMA in a bad state, and the fast relaunch skips the
        # infrasetup that would reset it. Do not trust it next time: force a full
        # infrasetup (unload/flash/reload XDMA) for the next candidate.
        _clear_infrasetup_stamp()

    return RunResult(
        returncode=returncode,
        stdout=infra_stdout + stdout,
        stderr=infra_stderr + stderr,
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
        bwrap_cmd = ["bwrap", "--die-with-parent", "--dev", "/dev",
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
                FIRESIM_CHIPYARD_PATH, CHIPYARD_DIFF_SUBMODULES, baseline
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


def chisel_build(dump: Dumper, attempt: int):
    node = ChiselBuildNode(
        chipyard_path=FIRESIM_CHIPYARD_PATH,
        config=BUILD_CONFIG,
        config_package=BUILD_CONFIG_PACKAGE,
        target=BuildTarget.VERILATOR,
        make_jobs=CHISEL_BUILD_MAKE_JOBS,
        timeout_seconds=CHISEL_BUILD_TIMEOUT_SECONDS,
    )
    logger.info("Building %s (attempt %d)", BUILD_CONFIG, attempt)
    artifact = get(
        node.build.options(resources={"manager": 1}).chia_remote(node)
    )
    dump.text(f"chisel_build_attempt{attempt}.stdout.txt", artifact.stdout)
    dump.text(f"chisel_build_attempt{attempt}.stderr.txt", artifact.stderr)
    logger.info("Build %s (rc=%s)", "OK" if artifact.success else "FAILED", artifact.returncode)
    return artifact


def chisel_build_mock(dump: Dumper, attempt: int) -> BuildArtifact:
    """Test double for chisel_build: skips the real (slow) Verilator build
    and returns a fabricated, always-successful BuildArtifact."""
    logger.info("Building %s (attempt %d) [mock]", BUILD_CONFIG, attempt)
    artifact = BuildArtifact(
        name="chipyard",
        simulator_binary_content=b"",
        simulator_binary_name="",
        config=BUILD_CONFIG,
        config_package=BUILD_CONFIG_PACKAGE,
        target=BuildTarget.VERILATOR,
        success=True,
        stdout="[mock] chisel_build skipped",
        stderr="",
        returncode=0,
    )
    dump.text(f"chisel_build_attempt{attempt}.stdout.txt", artifact.stdout)
    dump.text(f"chisel_build_attempt{attempt}.stderr.txt", artifact.stderr)
    logger.info("Build %s (rc=%s) [mock]", "OK" if artifact.success else "FAILED", artifact.returncode)
    return artifact


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
    r = bash("firesim buildbitstream", FIRESIM_BUILD_TIMEOUT_SECONDS)
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
