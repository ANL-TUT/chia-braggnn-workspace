import os
import re
import subprocess
import tempfile

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
    return _collect_diff(chipyard_path, submodules, baseline)


def _collect_diff(
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


def _without_nested_submodules(patch: str) -> str:
    """*patch* minus the per-file sections that only move a nested
    submodule's pointer ("Subproject commit" lines, e.g. gemmini's
    gemmini-rocc-tests, dirty from the generated gemmini_params.h): git
    apply cannot restore those, and they are build byproducts here."""
    parts = re.split(r"(?=^diff --git )", patch, flags=re.MULTILINE)
    return "".join(p for p in parts if "Subproject commit" not in p)


@ChiaFunction(resources={"manager": 0.05})
def restore_diff(
    chipyard_path: str,
    submodules: list[str],
    baseline: dict[str, str],
    target: dict[str, str],
) -> tuple[bool, str]:
    """Put each of *submodules* back to the state *target* (a collect_diff
    result) records: reset its tracked files to its baseline commit, remove
    its untracked, non-ignored files, and re-apply target's diff. Only the
    submodules are touched -- they are where the HW LLM can write; the root
    repo is read-only to it and also holds hand-made changes (e.g.
    vlsi/example-vlsi-sky130). Returns (ok, message)."""
    err, current = _collect_diff(chipyard_path, submodules, baseline)
    if err:
        return (False, "collect_diff failed before the restore")
    notes = []
    for sm in submodules:
        want = _without_nested_submodules(target.get(sm, ""))
        if _without_nested_submodules(current.get(sm, "")) == want:
            continue
        path = os.path.join(chipyard_path, sm)
        for cmd in (["git", "-C", path, "checkout", baseline.get(sm) or "HEAD", "--", "."],
                    ["git", "-C", path, "clean", "-fdq", "--", "."]):
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                return (False, f"{' '.join(cmd)} failed: {r.stderr.strip()}")
        if want.strip():
            with tempfile.NamedTemporaryFile("w", suffix=".patch", delete=False) as f:
                f.write(want)
            r = subprocess.run(["git", "-C", path, "apply", "--whitespace=nowarn", f.name],
                               capture_output=True, text=True)
            os.unlink(f.name)
            if r.returncode != 0:
                return (False, f"git apply in {sm} failed: {r.stderr.strip()}")
        notes.append(sm)
    err, after = _collect_diff(chipyard_path, submodules, baseline)
    mismatched = [sm for sm in submodules
                  if _without_nested_submodules(after.get(sm, ""))
                  != _without_nested_submodules(target.get(sm, ""))]
    if err or mismatched:
        return (False, f"restored {notes}, but still differs in {mismatched}")
    return (True, f"restored {notes or 'nothing (already there)'}")
