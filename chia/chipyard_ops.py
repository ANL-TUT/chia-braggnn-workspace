import os
import subprocess

from chia.base.ChiaFunction import ChiaFunction


def _rev_parse(repo_path: str) -> str:
    return subprocess.run(
        ["git", "-C", repo_path, "rev-parse", "HEAD"],
        capture_output=True, text=True,
    ).stdout.strip()


@ChiaFunction(resources={"manager": 0.01})
def capture_chisel_baseline(
    chipyard_path: str,
    submodules: list[str] | None = None,
) -> dict[str, str]:
    submodules = submodules or []
    baseline = {"": _rev_parse(chipyard_path)}
    for sm in submodules:
        baseline[sm] = _rev_parse(os.path.join(chipyard_path, sm))
    return baseline


def _untracked_diff(repo_path: str) -> str:
    listed = subprocess.run(
        ["git", "-C", repo_path, "ls-files", "--others", "--exclude-standard"],
        capture_output=True, text=True,
    ).stdout.split()
    parts = []
    for f in listed:
        # --no-index exits 1 when files differ; we only consume stdout.
        r = subprocess.run(
            ["git", "-C", repo_path, "diff", "--no-index", "--", "/dev/null", f],
            capture_output=True, text=True,
        )
        if r.stdout:
            parts.append(r.stdout)
    return "".join(parts)


@ChiaFunction(resources={"manager": 0.05})
def collect_diff(
    chipyard_path: str,
    submodules: list[str] | None = None,
    baseline: dict[str, str] | None = None,
) -> tuple[int, dict[str, str]]:
    submodules = submodules or []
    baseline = baseline or {}
    for sm in submodules:
        if not os.path.exists(os.path.join(chipyard_path, sm, ".git")):
            return (1, {})

    diffs: dict[str, str] = {}
    root_cmd = ["git", "-C", chipyard_path, "diff", "--ignore-submodules=all"]
    if baseline.get(""):
        root_cmd.append(baseline[""])
    root = subprocess.run(root_cmd, capture_output=True, text=True).stdout
    diffs[""] = root + _untracked_diff(chipyard_path)
    for sm in submodules:
        sm_path = os.path.join(chipyard_path, sm)
        sm_cmd = ["git", "-C", sm_path, "diff"]
        if baseline.get(sm):
            sm_cmd.append(baseline[sm])
        tracked = subprocess.run(sm_cmd, capture_output=True, text=True).stdout
        diffs[sm] = tracked + _untracked_diff(sm_path)
    return (0, diffs)
