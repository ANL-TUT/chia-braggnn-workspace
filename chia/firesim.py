import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from chia.base.ChiaFunction import ChiaFunction

CY_DIR = "/nfs/app/chipyard"
FIRESIM_DIR = f"{CY_DIR}/sims/firesim"
HWDB = f"{CY_DIR}/sims/firesim-staging/sample_config_hwdb.yaml"
BUILD_RECIPES = f"{CY_DIR}/sims/firesim-staging/sample_config_build_recipes.yaml"


def stage_bare_workload(elf: bytes) -> None:
    workloads = Path(FIRESIM_DIR) / "deploy" / "workloads"
    (workloads / "chia-bare").mkdir(parents=True, exist_ok=True)
    (workloads / "chia-bare" / "chia-bare-bin").write_bytes(elf)
    (workloads / "chia-bare" / "chia-bare-bin-dwarf").write_bytes(elf)
    (workloads / "chia-bare.json").write_text(
        json.dumps(
            {
                "benchmark_name": "chia-bare",
                "common_bootbinary": "chia-bare-bin",
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


@ChiaFunction(resources={"firesim": 1})
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
