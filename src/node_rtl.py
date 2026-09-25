"""Inspect or patch the gemmini RTL on the firesim node (no ssh from here).

    python src/node_rtl.py status              # HEAD, git status, diff --stat
    python src/node_rtl.py diff                # full working-tree diff
    python src/node_rtl.py apply my.patch      # git apply (a diff relative to the gemmini repo root)
    python src/node_rtl.py revert my.patch     # git apply -R
    python src/node_rtl.py cat src/main/scala/gemmini/Normalizer.scala

The tree is CHIPYARD_PATH/generators/gemmini, shared by the node's chipyards:
revert what did not work.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import ray

from chia.base.ChiaFunction import ChiaFunction, get

from constants import CHIPYARD_PATH

GEMMINI = f"{CHIPYARD_PATH}/generators/gemmini"


@ChiaFunction(resources={"manager": 0.01})
def node_git(action: str, arg: str = "") -> str:
    import os

    def run(*cmd, stdin=None):
        p = subprocess.run(cmd, cwd=GEMMINI, input=stdin, capture_output=True, text=True)
        return p.stdout + p.stderr + ("" if p.returncode == 0 else f"[exit {p.returncode}]\n")

    root = os.path.realpath(GEMMINI)
    if action == "status":
        return (f"{GEMMINI} -> {root}\n" + run("git", "log", "--oneline", "-3")
                + run("git", "status", "--short", "--untracked-files=no") + run("git", "diff", "--stat"))
    if action == "diff":
        return run("git", "diff")
    if action == "apply":
        return run("git", "apply", "--verbose", "-", stdin=arg)
    if action == "revert":
        return run("git", "apply", "-R", "--verbose", "-", stdin=arg)
    if action == "cat":
        return Path(GEMMINI, arg).read_text()
    raise ValueError(action)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["status", "diff", "apply", "revert", "cat"])
    parser.add_argument("arg", nargs="?", default="")
    args = parser.parse_args()
    if not ray.is_initialized():
        ray.init(address="auto")
    arg = Path(args.arg).read_text() if args.action in ("apply", "revert") else args.arg
    out = get(node_git.chia_remote(args.action, arg))
    sys.stdout.write(out)


if __name__ == "__main__":
    main()
