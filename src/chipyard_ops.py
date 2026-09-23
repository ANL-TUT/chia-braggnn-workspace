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


# The root repo's own Chisel sources. Its other untracked files are build
# output (vlsi/build-*, vlsi/generated-src-*: hundreds of MB of Verilog and
# Genus DBs) or unrelated copies (e.g. generators/gemmini.bak), and the
# generators/* submodules are diffed separately.
ROOT_UNTRACKED_PATHSPECS = ["generators/chipyard"]


def _untracked_diff(repo_path: str, pathspecs: list[str] | None = None) -> str:
    listed = subprocess.run(
        ["git", "-C", repo_path, "ls-files", "--others", "--exclude-standard",
         "--", *(pathspecs or [])],
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
    diffs[""] = root + _untracked_diff(chipyard_path, ROOT_UNTRACKED_PATHSPECS)
    for sm in submodules:
        sm_path = os.path.join(chipyard_path, sm)
        sm_cmd = ["git", "-C", sm_path, "diff"]
        if baseline.get(sm):
            sm_cmd.append(baseline[sm])
        tracked = subprocess.run(sm_cmd, capture_output=True, text=True).stdout
        diffs[sm] = tracked + _untracked_diff(sm_path)
    return (0, diffs)


# ── Snapshot / restore of the Gemmini sources ────────────────────────
# The RTL loop keeps only accepted edits: after a rejected one it puts the
# tree back to the last accepted state. The LLM's sandbox cannot run git on
# these repos (their .git dirs live outside it), so the loop does it. Each
# repo is saved as its working-tree diff against HEAD plus its untracked
# files, so the snapshot also carries edits that predate the loop.
GEMMINI_REPOS = (
    "generators/gemmini",
    "generators/gemmini/software/gemmini-rocc-tests",
)


def _untracked(repo: str) -> list:
    return subprocess.run(
        ["git", "-C", repo, "ls-files", "--others", "--exclude-standard", "-z"],
        capture_output=True, text=True, check=True,
    ).stdout.split("\0")[:-1]


@ChiaFunction(resources={"manager": 0.01})
def snapshot_repos(chipyard_path: str, repos: tuple = GEMMINI_REPOS) -> dict:
    """{repo: {"diff": binary diff vs HEAD, "untracked": {path: entry}}}, where
    an entry is ("file", bytes) or ("link", target). Symlinks are kept as links,
    never followed: a checkout can hold dangling or looping ones (e.g. an
    untracked generators/gemmini/gemmini pointing at itself)."""
    snap = {}
    for rel in repos:
        repo = os.path.join(chipyard_path, rel)
        diff = subprocess.run(
            ["git", "-C", repo, "diff", "--binary", "--ignore-submodules=all", "HEAD"],
            capture_output=True, check=True,
        ).stdout
        untracked = {}
        for path in _untracked(repo):
            full = os.path.join(repo, path)
            if os.path.islink(full):
                untracked[path] = ("link", os.readlink(full))
            else:
                with open(full, "rb") as f:
                    untracked[path] = ("file", f.read())
        snap[rel] = {"diff": diff, "untracked": untracked}
    return snap


@ChiaFunction(resources={"manager": 0.01})
def restore_repos(chipyard_path: str, snap: dict) -> None:
    """Make each repo in *snap* look exactly as when the snapshot was taken:
    tracked files back to HEAD plus the saved diff, untracked files created
    since removed, saved untracked files rewritten. Anything else edited in
    these repos meanwhile is discarded."""
    for rel, state in snap.items():
        repo = os.path.join(chipyard_path, rel)
        subprocess.run(["git", "-C", repo, "checkout", "HEAD", "--", "."],
                       capture_output=True, check=True)
        for path in _untracked(repo):
            if path not in state["untracked"]:
                os.remove(os.path.join(repo, path))  # removes a link, not its target
        for path, (kind, data) in state["untracked"].items():
            full = os.path.join(repo, path)
            os.makedirs(os.path.dirname(full), exist_ok=True)
            if kind == "link":
                if os.path.islink(full) and os.readlink(full) == data:
                    continue
                if os.path.lexists(full):
                    os.remove(full)
                os.symlink(data, full)
            else:
                if os.path.islink(full):
                    os.remove(full)  # write the file itself, not through a link
                with open(full, "wb") as f:
                    f.write(data)
        if state["diff"]:
            subprocess.run(["git", "-C", repo, "apply", "--binary", "-"],
                           input=state["diff"], capture_output=True, check=True)
