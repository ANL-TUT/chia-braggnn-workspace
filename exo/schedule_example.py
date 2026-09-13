# ruff: noqa: F821
# Example schedules for braggnn_exo.py.
#
# Append this file to the end of braggnn_exo.py; it only adds new code and
# rebinds braggnn_inference, leaving every existing @proc untouched. Every step
# is a checked exo.API_scheduling operation, so the result is equivalent to the
# original by construction.
#
# 1. Gemmini matmul: move the integer accumulation of matmul_transA (the NLB
#    attention matmul) onto Gemmini. Gemmini instrs are introduced with
#    `replace`, which only succeeds when the statements match the instr's
#    semantics:
#      - per (i1, j1) the 9x32x9 sum of products is split into two 16-wide k
#        blocks, loaded with ld_i8_id1/ld_i8_id2 and accumulated with
#        matmul_acc_i8
#      - the accumulator is cleared with zero_acc_i32 and read back with
#        st_acc_i32
#      - the float scaling, rounding and saturation stay on the CPU
#    Gemmini scratchpad/accumulator buffers need an innermost dimension of
#    exactly 16, hence the resize_dim / expand_dim to 16. The scheduled proc is
#    swapped in with call_eqv.
#
# 2. Loop fusion via inline: inline conv2 and the leaky3 that follows it into
#    braggnn_inference, then fuse their [7, 7, 32] loop nests so each output
#    element is requantized right after it is written.

from exo.API_scheduling import (
    call_eqv,
    divide_loop,
    expand_dim,
    fission,
    fuse,
    inline,
    lift_alloc,
    rearrange_dim,
    reorder_loops,
    replace,
    resize_dim,
    set_memory,
    simplify,
    stage_mem,
)
from exo.libs.memories import DRAM_STATIC, GEMM_ACCUM, GEMM_SCRATCH
from exo.platforms.gemmini import (
    ld_i8_id1,
    ld_i8_id2,
    matmul_acc_i8,
    st_acc_i32,
    zero_acc_i32,
)


def _nest(p, pattern, depth):
    """Cursor to the loop `depth` levels above the statement matching `pattern`."""
    c = p.find(pattern)
    for _ in range(depth):
        c = c.parent()
    return c


# --------------------------------------------------------------------------- #
# 1. Gemmini matmul
# --------------------------------------------------------------------------- #


def _schedule_matmul_transA_gemmini(p):
    # Loop order i1, j1, i2, j2 and 16-wide k blocks.
    p = reorder_loops(p, "i2 j1")
    p = divide_loop(p, "k", 16, ["ko", "ki"], perfect=True)

    # Turn the per-element `sum` into a [9, 16] accumulator buffer and split the
    # nest into clear / accumulate / scale parts.
    p = expand_dim(p, "sum : _", 16, "j2")
    p = lift_alloc(p, "sum : _")
    p = expand_dim(p, "sum : _", CONV1_DIM, "i2")
    p = lift_alloc(p, "sum : _")
    p = fission(p, p.find("sum[_] = 0.0").after(), n_lifts=2)
    p = fission(p, p.find_loop("ko").after(), n_lifts=2)
    p = reorder_loops(p, "j2 ko")
    p = reorder_loops(p, "i2 ko")
    p = simplify(p)

    # B block: 16 rows (k) x 9 cols (j2), loaded into a [16, 16] scratchpad.
    p = stage_mem(
        p, p.find_loop("ko").body()[0], "B[16 * ko:16 * ko + 16, j1, 0:9]", "bs"
    )
    p = resize_dim(p, "bs : _", 1, 16, 0)
    # A block: transposed CPU copy At[i2, k], then a [9, 16] scratchpad block.
    p = stage_mem(p, p.find_loop("ko"), "A[0:32, i1, 0:9]", "At")
    p = rearrange_dim(p, "At : _", [1, 0])
    p = stage_mem(p, _nest(p, "sum[_] += _", 3), "At[0:9, 16 * ko:16 * ko + 16]", "as_")
    # CPU-readable copy of the accumulator for the scale loop.
    p = stage_mem(p, _nest(p, "C[_] = _", 2), "sum[0:9, 0:9]", "sum_out")
    p = simplify(p)
    # Buffers created by scheduling default to DRAM (heap malloc); keep the CPU
    # side on static storage.
    p = set_memory(p, "At : _", DRAM_STATIC)
    p = set_memory(p, "sum_out : _", DRAM_STATIC)

    # Move buffers to Gemmini memories and swap the loops for Gemmini instrs.
    p = set_memory(p, "sum : _", GEMM_ACCUM)
    p = set_memory(p, "as_ : _", GEMM_SCRATCH)
    p = set_memory(p, "bs : _", GEMM_SCRATCH)
    p = replace(p, _nest(p, "sum[_] = 0.0", 2), zero_acc_i32)
    p = replace(p, _nest(p, "bs[_] = B[_]", 2), ld_i8_id2)
    p = replace(p, _nest(p, "as_[_] = At[_]", 2), ld_i8_id1)
    p = replace(p, _nest(p, "sum[_] += _", 3), matmul_acc_i8)
    p = replace(p, _nest(p, "sum_out[_] = sum[_]", 2), st_acc_i32)
    return simplify(p)


matmul_transA_gemmini = rename(
    _schedule_matmul_transA_gemmini(matmul_transA), "matmul_transA_gemmini"
)
_nlb = call_eqv(nlb, "matmul_transA(_)", matmul_transA_gemmini)
_inf = call_eqv(braggnn_inference, "nlb(_)", _nlb)


# --------------------------------------------------------------------------- #
# 2. Loop fusion via inline
# --------------------------------------------------------------------------- #


def _fuse_conv2_leaky3(p):
    # conv2 writes conv2_out[oh, ow, oc] and leaky3 then rewrites the same
    # element, so after inlining both the loop nests can be fused level by level.
    p = inline(p, "conv2(_)")
    p = inline(p, "leaky3(_)")
    p = fuse(p, "for oh in _:_", "for i in _:_")
    p = fuse(p, "for ow in _:_", "for j in _:_")
    p = fuse(p, "for oc in _:_", "for k in _:_")
    return simplify(p)


braggnn_inference = rename(_fuse_conv2_leaky3(_inf), "braggnn_inference")
