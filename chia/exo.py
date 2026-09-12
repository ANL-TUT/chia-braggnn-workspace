import subprocess
import tempfile
from pathlib import Path

from chia.base.ChiaFunction import ChiaFunction


@ChiaFunction(resources={"exo": 1})
def exo_compile(source: bytes, stem: str) -> dict[str, bytes]:
    with tempfile.TemporaryDirectory(prefix="chia-exo-") as tmp:
        work = Path(tmp)
        (work / f"{stem}.py").write_bytes(source)
        done = subprocess.run(
            [
                "exocc",
                str(work / f"{stem}.py"),
                "-o",
                str(work / "out"),
                "--stem",
                stem,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if done.returncode != 0:
            raise RuntimeError(
                f"exo compile failed (rc={done.returncode})\n{done.stderr[-2000:]}"
            )
        return {
            f"{stem}.{ext}": (work / "out" / f"{stem}.{ext}").read_bytes()
            for ext in ("c", "h")
        }
