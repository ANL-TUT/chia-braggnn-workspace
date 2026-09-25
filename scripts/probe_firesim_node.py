"""Print which Gemmini source, hwdb entry and gemmini_params.h the FireSim node
uses, under $CHIPYARD_PATH (pass it in the job's runtime env to probe another
checkout, e.g. ~/chipyard)."""
import subprocess

import ray

ray.init(address="auto")


@ray.remote(resources={"manager": 0.01}, num_cpus=0)
def probe():
    sh = r'''
C=${CHIPYARD_PATH:-$HOME/chia-experiments/chipyard}
echo "conda before: ${CONDA_PREFIX:-none}"
source $C/env.sh >/dev/null 2>&1 && echo "conda after sourcing $C/env.sh: ${CONDA_PREFIX:-none}"
(cd $C/sims/firesim && source sourceme-manager.sh --skip-ssh-setup >/dev/null 2>&1 && export PATH=$C/sims/firesim/deploy:$PATH && echo "firesim: $(command -v firesim)  python: $(command -v python3)")
cd $C/generators/gemmini && { echo "gemmini HEAD: $(git log -1 --format='%h %ad %s' --date=iso)"; git log --oneline -6; echo "dirty:"; git status --short | head -8; }
D=$C/sims/firesim/deploy
grep -n "default_hw_config" $D/config_runtime.yaml
H=$(grep -oP "^\s*default_hw_config:\s*\K\S+" $D/config_runtime.yaml | head -1)
echo "active hw config: $H"; cat $D/built-hwdb-entries/$H 2>/dev/null
grep -n -A6 "^$H:" $D/config_hwdb.yaml | head -8
ls -la --time-style=long-iso $D/built-hwdb-entries 2>/dev/null | tail -4
G=$(readlink -f $C/generators/gemmini)
echo "gemmini dir: $G"
for f in AccumulatorScale.scala GemminiConfigs.scala; do
  echo "mtime $f: $(date -d @$(stat -c %Y $G/src/main/scala/gemmini/$f) '+%F %T %z')"; done
P=$C/generators/gemmini/software/gemmini-rocc-tests/include/gemmini_params.h
echo "gemmini_params.h sha256: $(sha256sum $P | cut -c1-16)"
grep -E "^#define (DIM|ADDR_LEN|BANK_NUM|BANK_ROWS|ACC_ROWS|MAX_BYTES|MAX_BLOCK_LEN|MAX_BLOCK_LEN_ACC|HAS_NORMALIZATIONS|NORM_STAT_IDS)\b" $P
'''
    return subprocess.run(["bash", "-lc", sh], capture_output=True, text=True).stdout


print(ray.get(probe.remote()))
