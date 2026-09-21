import shutil
import subprocess
import tempfile
from pathlib import Path

from chia.base.ChiaFunction import ChiaFunction

# Relative paths resolve inside the job's working_dir package that Ray unpacks
# on the worker, so these read the code shipped with `chia job submit`.
SHIPPED_EXO_DIR = Path("exo")
# Harness files and directories copied next to the generated C: the C driver
# and the Gemmini headers (include/, rocc-software/).
HARNESS_FILES = (
    "braggnn_main.c",
    "braggnn_data.h",
    "xprintf.c",
    "xprintf.h",
)
HARNESS_DIRS = ("include", "rocc-software")
# Gemmini scratchpad allocators that exo.platforms.gemmini code calls; they ship
# inside the exo package installed in the container (exo/libs).
GEMM_MALLOC_FILES = (
    "gemm_malloc.c",
    "gemm_malloc.h",
    "gemm_acc_malloc.c",
    "gemm_acc_malloc.h",
)
MAKE_COMMAND = [
    "make",
    "-f",
    "/opt/riscv-harness/Makefile",
    "TARGET=verilator",
    "PROGRAM=braggnn",
    "SRCS=braggnn_main.c braggnn_schedule.c xprintf.c gemm_malloc.c gemm_acc_malloc.c",
    # gemm_malloc.h uses uint32_t without including <stdint.h>.
    "EXTRA_CFLAGS=-I. -include stdint.h -include include/gemmini.h",
    "EXTRA_LDFLAGS=",
]
BUILD_TIMEOUT_SECONDS = 600


@ChiaFunction(resources={"exo_build": 1})
def prepare_work_dir(work_dir: str, source_dir: str | None = None) -> None:
    """Seed work_dir with braggnn_schedule.py from source_dir (default: the shipped one).

    Also copies the C harness (with Gemmini headers and allocators) into
    work_dir/harness, and next to braggnn_schedule.py puts gemmini.py (the
    instrs), braggnn_reference.py (the *_on_cpu specs it schedules) and
    braggnn_schedule_lowlevel.py (a read-only reference schedule). build_elf always
    uses the shipped C harness, but it runs exocc on
    work_dir/braggnn_schedule.py, and exocc puts that file's directory on
    sys.path, so the modules the agent sees are the ones the build uses.
    """
    import exo  # exo-lang, installed in the container

    work = Path(work_dir)
    harness = work / "harness"
    harness.mkdir(parents=True, exist_ok=True)
    shutil.copy(
        SHIPPED_EXO_DIR / "braggnn_schedule.py", work / "braggnn_schedule.orig.py"
    )
    source = Path(source_dir) if source_dir else SHIPPED_EXO_DIR
    shutil.copy(source / "braggnn_schedule.py", work / "braggnn_schedule.py")
    shutil.copy(SHIPPED_EXO_DIR / "gemmini.py", work / "gemmini.py")
    shutil.copy(SHIPPED_EXO_DIR / "braggnn_reference.py", work / "braggnn_reference.py")
    shutil.copy(
        SHIPPED_EXO_DIR / "braggnn_schedule_lowlevel.py",
        work / "braggnn_schedule_lowlevel.py",
    )
    for name in HARNESS_FILES:
        shutil.copy(SHIPPED_EXO_DIR / name, harness / name)
    for name in HARNESS_DIRS:
        shutil.copytree(SHIPPED_EXO_DIR / name, harness / name, dirs_exist_ok=True)
    for name in GEMM_MALLOC_FILES:
        shutil.copy(Path(exo.__file__).parent / "libs" / name, harness / name)


@ChiaFunction(resources={"exo_build": 1})
def build_elf(work_dir: str) -> dict:
    """Compile work_dir/braggnn_schedule.py with Exo and link braggnn.riscv.

    Returns {"source", "elf", "stage", "log"}; "elf" is None on failure.
    Keep everything inside this function and return plain types: workers cannot
    import this module by name, so module-level helpers or classes would not
    deserialize there.
    """
    import exo  # exo-lang, installed in the container

    source_path = Path(work_dir) / "braggnn_schedule.py"
    if not source_path.exists():
        log = f"{source_path} is missing"
        return {"source": "", "elf": None, "stage": "llm", "log": log}
    source = source_path.read_text()

    steps = [
        ("exo", ["exocc", str(source_path), "-o", ".", "--stem", "braggnn_schedule"]),
        ("build", MAKE_COMMAND),
    ]
    with tempfile.TemporaryDirectory(prefix="chia-braggnn-build-") as tmp:
        build = Path(tmp)
        for name in HARNESS_FILES:
            shutil.copy(SHIPPED_EXO_DIR / name, build / name)
        for name in HARNESS_DIRS:
            shutil.copytree(SHIPPED_EXO_DIR / name, build / name)
        for name in GEMM_MALLOC_FILES:
            shutil.copy(Path(exo.__file__).parent / "libs" / name, build / name)
        for stage, cmd in steps:
            try:
                done = subprocess.run(
                    cmd,
                    cwd=build,
                    capture_output=True,
                    text=True,
                    timeout=BUILD_TIMEOUT_SECONDS,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                log = f"{cmd[0]} timed out after {BUILD_TIMEOUT_SECONDS}s"
                return {"source": source, "elf": None, "stage": stage, "log": log}
            if done.returncode != 0:
                log = done.stdout + done.stderr
                return {"source": source, "elf": None, "stage": stage, "log": log}
        elf = (build / "braggnn.riscv").read_bytes()
    return {"source": source, "elf": elf, "stage": "build", "log": ""}
