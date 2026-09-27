"""List or kill processes on the firesim node (no ssh from here).

    python src/node_ps.py list simulator-chipyard     # processes whose command line matches
    python src/node_ps.py kill conv_with_pool-baremetal

`kill` sends SIGKILL to the matching simulator processes only (command lines
that also contain "simulator-"), e.g. to end one long Verilator test early:
VerilatorRunNode then returns it like a timeout (exit -9).
"""

import argparse
import subprocess
import sys

import ray

from chia.base.ChiaFunction import ChiaFunction, get


@ChiaFunction(resources={"manager": 0.01})
def node_ps(action: str, pattern: str) -> str:
    out = subprocess.run(["ps", "-eo", "pid,etimes,args"], capture_output=True, text=True).stdout
    rows = [l for l in out.splitlines()[1:] if pattern in l and "node_ps" not in l]
    lines = [r[:220] for r in rows]
    if action == "kill":
        for r in rows:
            if "simulator-" in r:
                pid = int(r.split()[0])
                subprocess.run(["kill", "-9", str(pid)])
                lines.append(f"killed {pid}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["list", "kill"])
    parser.add_argument("pattern")
    args = parser.parse_args()
    if not ray.is_initialized():
        ray.init(address="auto")
    sys.stdout.write(get(node_ps.chia_remote(args.action, args.pattern)))


if __name__ == "__main__":
    main()
