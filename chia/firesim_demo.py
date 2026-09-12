from pathlib import Path

from chia.base.ChiaFunction import get
from chia.chipyard.riscv_build_node import RiscvBuildNode
from firesim import run_workload

from exo import exo_compile


def main() -> None:
    generated = get(
        exo_compile.chia_remote(
            source=Path("exo/braggnn_exo.py").read_bytes(), stem="braggnn_exo"
        )
    )

    riscv = RiscvBuildNode(timeout_seconds=600)
    art = get(
        riscv.build_program.chia_remote(
            riscv,
            {
                "braggnn_main.c": Path("exo/braggnn_main.c").read_bytes(),
                "braggnn_data.h": Path("exo/braggnn_data.h").read_bytes(),
                "xprintf.c": Path("exo/xprintf.c").read_bytes(),
                "xprintf.h": Path("exo/xprintf.h").read_bytes(),
            }
            | generated,
            [
                "make",
                "-f",
                "/opt/riscv-harness/Makefile",
                "TARGET=verilator",
                "PROGRAM=braggnn",
                "SRCS=braggnn_main.c braggnn_exo.c xprintf.c",
                "EXTRA_CFLAGS=-I.",
                "EXTRA_LDFLAGS=",
            ],
            "/tmp/chia-braggnn-build",
            ["braggnn.riscv"],
        )
    )
    if not art.success:
        raise SystemExit(art.stdout + art.stderr)

    result = get(run_workload.chia_remote(elf=art.files["braggnn.riscv"]))
    uartlog = "\n".join(result.uartlogs.values())
    print(uartlog)
    if "*** PASSED ***" not in uartlog:
        raise SystemExit(result.stderr or result.stdout or "no uartlog collected")


if __name__ == "__main__":
    main()
