#!/usr/bin/env bash
# Cross-compile a hand-written BraggNN implementation (one C file defining
# braggnn_eval, e.g. exo/braggnn_tune_opus.c) with the same harness, flags and
# Gemmini headers as the Exo candidates (src/exo_compiler.py), in the local
# chia-exo image. Output: build/<stem>/braggnn.riscv
#
#   [NAME=variant] scripts/build_c_tune.sh exo/braggnn_tune_opus.c [EXTRA_CFLAGS...]
# NAME puts the build in build/<stem>-<NAME>/ so several variants can coexist.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
SRC=$(realpath "$1"); shift
STEM=$(basename "$SRC" .c)
OUT="$REPO/build/$STEM${NAME:+-$NAME}"
IMAGE=${IMAGE:-docker.io/library/chia-exo:latest}
PODMAN=podman; [ -f /run/.containerenv ] && command -v flatpak-spawn >/dev/null && PODMAN="flatpak-spawn --host podman"  # toolbox: use the host podman

rm -rf "$OUT"; mkdir -p "$OUT"
cp "$REPO"/exo/{braggnn_main.c,braggnn_data.h,xprintf.c,xprintf.h} "$OUT"/
cp -r "$REPO"/exo/include "$REPO"/exo/rocc-software "$OUT"/
cp "$SRC" "$OUT/$STEM.c"
# braggnn_main.c includes braggnn_schedule.h: declare the one entry point.
{ echo '#pragma once'; echo '#include <stdint.h>'
  sed -n '/^void braggnn_eval(/{s/ *{ *$/;/;p;q}' "$SRC"; } > "$OUT/braggnn_schedule.h"
grep -q "braggnn_eval" "$OUT/braggnn_schedule.h" || { echo "no braggnn_eval definition line in $SRC" >&2; exit 1; }

$PODMAN run --rm --userns=keep-id:uid=1000,gid=1000 -v "$OUT":/w:Z -w /w "$IMAGE" bash -lc "
  L=\$(python -c 'import exo,os;print(os.path.join(os.path.dirname(exo.__file__),\"libs\"))' 2>/dev/null)
  cp \$L/gemm_malloc.[ch] \$L/gemm_acc_malloc.[ch] .
  make -s -f /opt/riscv-harness/Makefile TARGET=verilator PROGRAM=braggnn \
    'SRCS=braggnn_main.c $STEM.c xprintf.c gemm_malloc.c gemm_acc_malloc.c' \
    'EXTRA_CFLAGS=-I. -include stdint.h -include include/gemmini.h $*' EXTRA_LDFLAGS=
" 2>&1 | grep -v -E "Warning|Cython|tree = Parsing" || true
test -f "$OUT/braggnn.riscv" && ls -la "$OUT/braggnn.riscv"
