# ruff: noqa: F821

from __future__ import annotations

from exo.API_scheduling import (
    call_eqv,
    commute_expr,
    divide_loop,
    expand_dim,
    extract_subproc,
    inline,
    insert_noop_call,
    mult_dim,
    mult_loops,
    rearrange_dim,
    rename,
    reorder_loops,
    reorder_stmts,
    replace,
    rewrite_expr,
    set_memory,
    simplify,
    split_write,
    unroll_loop,
)
from exo.libs.externs import relu, select
from exo.libs.memories import DRAM_STATIC, GEMM_ACCUM, GEMM_SCRATCH
from exo.platforms.gemmini import (
    acc_scale,
    clamp,
    ld_i8_id1,
    ld_i8_id2,
    matmul_acc_i8,
    old_fission_after,
    old_lift_alloc,
    old_reorder,
    zero_acc_i32,
)
from gemmini import (
    fence,
    ld_acc_i8_scaled,
    ld_acc_i8_scaled_acc,
    ld_acc_i32_bias,
    ld_acc_i32_col,
    ld_i8_col,
    ld_i8_im2col,
    make_loop_matmul_fc,
    make_loop_softmax_flat,
    matmul_acc_i8_trans_b,
    st_acc_i8_act,
    st_acc_i8_no_act,
    st_acc_i8_relu,
)

from exo import DRAM, proc

INPUT_DIM = 11
CONV1_DIM = 9
CONV2_DIM = 7
CONV3_DIM = 5
CONV1_FILTERS = 64
CONV2_FILTERS = 32
CONV3_FILTERS = 8
FC1_UNITS = 16
FC2_UNITS = 8
FC3_UNITS = 4
FC4_UNITS = 2
OUTPUT_UNITS = 2

NLB_ROWS = CONV1_DIM * CONV1_DIM
NLB_SOFTMAX_TILE = 72

SOFTMAX_INPUT_SCALE = 0.028342675417661667


def max_softmax_shift(in_scale):
    qln2 = int(0.693147 / in_scale)
    qln2_inv = 65536 // qln2
    return (255 * qln2_inv) // 65536 + 1


SOFTMAX_MAX_SHIFT = max_softmax_shift(SOFTMAX_INPUT_SCALE)


def fence_after(p, pattern):
    return insert_noop_call(p, p.find(pattern).after(), fence, [])


@proc
def conv_on_cpu(
    in_dim: size,
    in_ch: size,
    out_ch: size,
    k: size,
    out_dim: size,
    inp: i8[in_dim, in_dim, in_ch] @ DRAM,
    weights: i8[k, k, in_ch, out_ch] @ DRAM,
    bias: i32[out_ch] @ DRAM,
    output: i8[out_dim, out_dim, out_ch] @ DRAM,
    scale: f32 @ DRAM,
):
    assert out_dim == in_dim - k + 1

    for orow in seq(0, out_dim):
        for ocol in seq(0, out_dim):
            for och in seq(0, out_ch):
                res: i32
                res = bias[och]

                for krow in seq(0, k):
                    for kcol in seq(0, k):
                        for kch in seq(0, in_ch):
                            w_s: i8 @ DRAM
                            w_s = weights[krow, kcol, kch, och]
                            i_s: i8 @ DRAM
                            i_s = inp[orow + krow, ocol + kcol, kch]
                            a2: i32
                            b2: i32
                            a2 = i_s
                            b2 = w_s
                            res += a2 * b2

                src_tmp: i32
                src_tmp = res
                tmp_res1: f32
                acc_scale(src_tmp, tmp_res1, scale)
                tmp_res2: i8
                clamp(tmp_res1, tmp_res2)
                output[orow, ocol, och] = tmp_res2


@proc
def conv_relu_on_cpu(
    in_dim: size,
    in_ch: size,
    out_ch: size,
    k: size,
    out_dim: size,
    inp: i8[in_dim, in_dim, in_ch] @ DRAM,
    weights: i8[k, k, in_ch, out_ch] @ DRAM,
    bias: i32[out_ch] @ DRAM,
    output: i8[out_dim, out_dim, out_ch] @ DRAM,
    scale: f32 @ DRAM,
    post_scale: f32 @ DRAM,
):
    assert out_dim == in_dim - k + 1

    for orow in seq(0, out_dim):
        for ocol in seq(0, out_dim):
            for och in seq(0, out_ch):
                res: i32
                res = bias[och]

                for krow in seq(0, k):
                    for kcol in seq(0, k):
                        for kch in seq(0, in_ch):
                            w_s: i8 @ DRAM
                            w_s = weights[krow, kcol, kch, och]
                            i_s: i8 @ DRAM
                            i_s = inp[orow + krow, ocol + kcol, kch]
                            a2: i32
                            b2: i32
                            a2 = i_s
                            b2 = w_s
                            res += a2 * b2

                tmp_scale: f32
                tmp_scale = scale * post_scale
                src_tmp: i32
                src_tmp = res
                tmp_res1: f32
                acc_scale(src_tmp, tmp_res1, tmp_scale)
                tmp_res2: i8
                clamp(tmp_res1, tmp_res2)
                tmp_res2 = relu(tmp_res2)
                output[orow, ocol, och] = tmp_res2


@proc
def fc_on_cpu(
    in_ch: size,
    in_h: size,
    in_w: size,
    out_features: size,
    inp: i8[in_ch, in_h, in_w] @ DRAM,
    weights: i8[out_features, in_ch, in_h, in_w] @ DRAM,
    bias: i32[out_features] @ DRAM,
    output: i8[out_features, 1, 1] @ DRAM,
    scale: f32 @ DRAM,
):
    for j in seq(0, out_features):
        res: i32
        res = bias[j]
        for kch in seq(0, in_ch):
            for krow in seq(0, in_h):
                for kcol in seq(0, in_w):
                    w_s: i8 @ DRAM
                    w_s = weights[j, kch, krow, kcol]
                    i_s: i8 @ DRAM
                    i_s = inp[kch, krow, kcol]
                    a2: i32
                    b2: i32
                    a2 = i_s
                    b2 = w_s
                    res += a2 * b2

        src_tmp: i32
        src_tmp = res
        tmp_res1: f32
        acc_scale(src_tmp, tmp_res1, scale)
        tmp_res2: i8
        clamp(tmp_res1, tmp_res2)
        output[j, 0, 0] = tmp_res2


@proc
def fc_relu_on_cpu(
    in_ch: size,
    in_h: size,
    in_w: size,
    out_features: size,
    inp: i8[in_ch, in_h, in_w] @ DRAM,
    weights: i8[out_features, in_ch, in_h, in_w] @ DRAM,
    bias: i32[out_features] @ DRAM,
    output: i8[out_features, 1, 1] @ DRAM,
    scale: f32 @ DRAM,
    post_scale: f32 @ DRAM,
):
    for j in seq(0, out_features):
        res: i32
        res = bias[j]
        for kch in seq(0, in_ch):
            for krow in seq(0, in_h):
                for kcol in seq(0, in_w):
                    w_s: i8 @ DRAM
                    w_s = weights[j, kch, krow, kcol]
                    i_s: i8 @ DRAM
                    i_s = inp[kch, krow, kcol]
                    a2: i32
                    b2: i32
                    a2 = i_s
                    b2 = w_s
                    res += a2 * b2

        tmp_scale: f32
        tmp_scale = scale * post_scale
        src_tmp: i32
        src_tmp = res
        tmp_res1: f32
        acc_scale(src_tmp, tmp_res1, tmp_scale)
        tmp_res2: i8
        clamp(tmp_res1, tmp_res2)
        tmp_res2 = relu(tmp_res2)
        output[j, 0, 0] = tmp_res2


@proc
def matmul_trans_b_on_cpu(
    A: i8[CONV1_DIM, CONV1_DIM, CONV2_FILTERS] @ DRAM,
    B: i8[CONV1_DIM, CONV1_DIM, CONV2_FILTERS] @ DRAM,
    C: i8[CONV1_DIM, CONV1_DIM, CONV1_DIM, CONV1_DIM] @ DRAM,
    scale: f32 @ DRAM,
):
    for i1 in seq(0, CONV1_DIM):
        for i2 in seq(0, CONV1_DIM):
            for j1 in seq(0, CONV1_DIM):
                for j2 in seq(0, CONV1_DIM):
                    res: i32
                    res = 0
                    for k in seq(0, CONV2_FILTERS):
                        a: i8 @ DRAM
                        a = A[i1, i2, k]
                        b: i8 @ DRAM
                        b = B[j1, j2, k]
                        a2: i32
                        b2: i32
                        a2 = a
                        b2 = b
                        res += a2 * b2

                    src_tmp: i32
                    src_tmp = res
                    tmp_res1: f32
                    acc_scale(src_tmp, tmp_res1, scale)
                    tmp_res2: i8
                    clamp(tmp_res1, tmp_res2)
                    C[i1, i2, j1, j2] = tmp_res2


@proc
def matmul_on_cpu(
    A: i8[CONV1_DIM, CONV1_DIM, CONV1_DIM, CONV1_DIM] @ DRAM,
    B: i8[CONV1_DIM, CONV1_DIM, CONV2_FILTERS] @ DRAM,
    C: i8[CONV1_DIM, CONV1_DIM, CONV2_FILTERS] @ DRAM,
    scale: f32 @ DRAM,
):
    for i1 in seq(0, CONV1_DIM):
        for i2 in seq(0, CONV1_DIM):
            for j in seq(0, CONV2_FILTERS):
                res: i32
                res = 0
                for k1 in seq(0, CONV1_DIM):
                    for k2 in seq(0, CONV1_DIM):
                        a: i8 @ DRAM
                        a = A[i1, i2, k1, k2]
                        b: i8 @ DRAM
                        b = B[k1, k2, j]
                        a2: i32
                        b2: i32
                        a2 = a
                        b2 = b
                        res += a2 * b2

                src_tmp: i32
                src_tmp = res
                tmp_res1: f32
                acc_scale(src_tmp, tmp_res1, scale)
                tmp_res2: i8
                clamp(tmp_res1, tmp_res2)
                C[i1, i2, j] = tmp_res2


@proc
def softmax_on_cpu(
    A: i8[CONV1_DIM, CONV1_DIM, CONV1_DIM, CONV1_DIM] @ DRAM,
    C: i8[CONV1_DIM, CONV1_DIM, CONV1_DIM, CONV1_DIM] @ DRAM,
    in_scale: f32 @ DRAM,
    out_scale: f32 @ DRAM,
):
    for i1 in seq(0, CONV1_DIM):
        for i2 in seq(0, CONV1_DIM):
            two16: i32
            two16 = 65536
            qln2: i32
            qln2_f: f32
            qln2_f = 0.693147 / in_scale
            qln2 = qln2_f
            qln2_inv: i32
            qln2_inv = two16 / qln2
            qb: i32
            qb_f: f32
            qb_f = 1.353 / in_scale
            qb = qb_f
            qc: i32
            qc_f: f32
            qc_f = 0.344 / (0.3585 * in_scale * in_scale)
            qc = qc_f

            max_q: i32
            max_q = A[i1, i2, 0, 0]
            for j1 in seq(0, CONV1_DIM):
                for j2 in seq(0, CONV1_DIM):
                    a2: i32
                    a2 = A[i1, i2, j1, j2]
                    max_q = select(max_q, a2, a2, max_q)

            exp_buf: i32[CONV1_DIM, CONV1_DIM] @ DRAM_STATIC
            sum_exp: i32
            sum_exp = 0
            for j1 in seq(0, CONV1_DIM):
                for j2 in seq(0, CONV1_DIM):
                    a2: i32
                    a2 = A[i1, i2, j1, j2]
                    neg_q: i32
                    neg_q = max_q - a2
                    z: i32
                    z = neg_q * qln2_inv / two16
                    qp: i32
                    qp = z * qln2 - neg_q
                    qpb: i32
                    qpb = qp + qb
                    q_exp: i32
                    q_exp = qpb * qpb + qc

                    zero: i32
                    zero = 0
                    shifts_left: i32
                    shifts_left = z
                    for s in seq(0, SOFTMAX_MAX_SHIFT):
                        halved: i32
                        halved = q_exp / 2
                        q_exp = select(zero, shifts_left, halved, q_exp)
                        next_left: i32
                        next_left = shifts_left - 1
                        shifts_left = select(zero, shifts_left, next_left, shifts_left)

                    exp_buf[j1, j2] = q_exp
                    sum_exp += q_exp

            denom: f32
            denom = sum_exp
            factor: f32
            factor = 127.0 / denom
            out_factor: f32
            out_factor = 1.0 / (127.0 * out_scale)
            tmp_scale: f32
            tmp_scale = factor * out_factor
            for j1 in seq(0, CONV1_DIM):
                for j2 in seq(0, CONV1_DIM):
                    src_tmp: i32
                    src_tmp = exp_buf[j1, j2]
                    tmp_res1: f32
                    acc_scale(src_tmp, tmp_res1, tmp_scale)
                    tmp_res2: i8
                    clamp(tmp_res1, tmp_res2)
                    C[i1, i2, j1, j2] = tmp_res2


@proc
def resadd_on_cpu(
    A_scale: f32 @ DRAM,
    B_scale: f32 @ DRAM,
    C_scale: f32 @ DRAM,
    A: i8[CONV1_DIM, CONV1_DIM, CONV1_FILTERS] @ DRAM,
    B: i8[CONV1_DIM, CONV1_DIM, CONV1_FILTERS] @ DRAM,
    C: i8[CONV1_DIM, CONV1_DIM, CONV1_FILTERS] @ DRAM,
):
    for i1 in seq(0, CONV1_DIM):
        for i2 in seq(0, CONV1_DIM):
            for j in seq(0, CONV1_FILTERS):
                a_src: i32
                a_tmp: f32
                a_q: i8
                a2: i32
                a_src = A[i1, i2, j]
                acc_scale(a_src, a_tmp, A_scale)
                clamp(a_tmp, a_q)
                a2 = a_q

                b_src: i32
                b_tmp: f32
                b_q: i8
                b2: i32
                b_src = B[i1, i2, j]
                acc_scale(b_src, b_tmp, B_scale)
                clamp(b_tmp, b_q)
                b2 = b_q

                res: i32
                src_tmp: i32
                tmp_res1: f32
                tmp_res2: i8
                res = a2 + b2
                src_tmp = res
                acc_scale(src_tmp, tmp_res1, C_scale)
                clamp(tmp_res1, tmp_res2)
                tmp_res2 = relu(tmp_res2)
                C[i1, i2, j] = tmp_res2


def sched_fc(name, in_ch, in_h, in_w, out_features, act=False):
    cpu = rename(fc_relu_on_cpu if act else fc_on_cpu, f"{name}_cpu")
    cpu = cpu.partial_eval(in_ch, in_h, in_w, out_features)

    do_fc = make_loop_matmul_fc(f"do_{name}", in_ch, in_h, in_w, out_features, act)

    gemmini = rename(cpu, name)
    gemmini = replace(gemmini, "for j in _:_", do_fc)
    gemmini = fence_after(gemmini, f"do_{name}(_)")

    return gemmini, cpu


def sched_fc_flat_lowlevel(name, in_ch, in_h, in_w, out_features, act=False):
    assert in_w <= 16
    assert out_features <= 16

    cpu = rename(fc_relu_on_cpu if act else fc_on_cpu, f"{name}_cpu")
    cpu = cpu.partial_eval(in_ch, in_h, in_w, out_features)

    gemmini = rename(cpu, name)
    gemmini = old_lift_alloc(gemmini, "res : _", n_lifts=1)
    gemmini = expand_dim(gemmini, "res : _", "16", "0")
    gemmini = rearrange_dim(gemmini, "res : _", [1, 0])
    gemmini = old_fission_after(gemmini, "res[_] = _", n_lifts=1)
    gemmini = old_fission_after(gemmini, "for kch in _:_", n_lifts=1)
    gemmini = old_reorder(gemmini, "j kch")
    gemmini = old_reorder(gemmini, "j krow")
    gemmini = old_lift_alloc(gemmini, "w_s : _", n_lifts=1, size=16)
    gemmini = old_lift_alloc(gemmini, "w_s : _", n_lifts=1)
    gemmini = old_lift_alloc(gemmini, "i_s : _", n_lifts=1)
    gemmini = old_lift_alloc(gemmini, "i_s : _", n_lifts=1, keep_dims=False)
    gemmini = expand_dim(gemmini, "i_s : _", "16", "0")
    gemmini = rearrange_dim(gemmini, "i_s : _", [1, 0])
    gemmini = old_fission_after(gemmini, "w_s[_] = _", n_lifts=2)
    gemmini = old_fission_after(gemmini, "i_s[_] = _", n_lifts=2)
    gemmini = divide_loop(gemmini, "j #3", 1, ["j_o", "j_i"], perfect=True)
    gemmini = divide_loop(gemmini, "j #2", 1, ["j_o", "j_i"], perfect=True)
    gemmini = simplify(gemmini)
    gemmini = reorder_stmts(gemmini, gemmini.find("a2 : _").expand(0, 1))
    gemmini = reorder_stmts(gemmini, gemmini.find("a2 = _").expand(0, 1))
    gemmini = commute_expr(gemmini, "a2 * b2")
    gemmini = rewrite_expr(gemmini, gemmini.find("b2 = _").rhs().idx()[0], "j_o")
    gemmini = rewrite_expr(gemmini, gemmini.find("a2 = _").rhs().idx()[1], "j_i")
    gemmini = rewrite_expr(gemmini, gemmini.find("res[_] += _").idx()[0], "j_o")
    gemmini = rewrite_expr(gemmini, gemmini.find("res[_] += _").idx()[1], "j_i")
    gemmini = rewrite_expr(gemmini, gemmini.find("src_tmp = _").rhs().idx()[0], "j_o")
    gemmini = rewrite_expr(gemmini, gemmini.find("src_tmp = _").rhs().idx()[1], "j_i")
    gemmini = rewrite_expr(gemmini, gemmini.find("output[_] = _").idx()[0], "j_o")
    gemmini = rewrite_expr(gemmini, gemmini.find("output[_] = _").idx()[1], "j_i")
    gemmini = set_memory(gemmini, "res : _", GEMM_ACCUM)
    gemmini = set_memory(gemmini, "w_s : _", GEMM_SCRATCH)
    gemmini = set_memory(gemmini, "i_s : _", GEMM_SCRATCH)
    gemmini = replace(gemmini, "for j in _:_ #0", ld_acc_i32_col)
    gemmini = replace(gemmini, "for j in _:_ #0", ld_i8_id1)
    gemmini = replace(gemmini, "for kcol in _:_ #0", ld_i8_col)
    gemmini = replace(gemmini, "for j_o in _:_ #0", matmul_acc_i8)
    if act:
        gemmini = replace(gemmini, "for j_o in _:_ #0", st_acc_i8_relu)
        gemmini = fence_after(gemmini, "st_acc_i8_relu(_)")
    else:
        gemmini = replace(gemmini, "for j_o in _:_ #0", st_acc_i8_no_act)
        gemmini = fence_after(gemmini, "st_acc_i8_no_act(_)")

    return gemmini, cpu


def sched_fc_lowlevel(name, in_ch, out_features, act=False):
    assert in_ch <= 16
    assert out_features <= 16

    cpu = rename(fc_relu_on_cpu if act else fc_on_cpu, f"{name}_cpu")
    cpu = cpu.partial_eval(in_ch, 1, 1, out_features)

    gemmini = rename(cpu, name)
    gemmini = unroll_loop(gemmini, "krow")
    gemmini = unroll_loop(gemmini, "kcol")
    gemmini = old_lift_alloc(gemmini, "res : _", n_lifts=1)
    gemmini = expand_dim(gemmini, "res : _", "16", "0")
    gemmini = rearrange_dim(gemmini, "res : _", [1, 0])
    gemmini = old_fission_after(gemmini, "res[_] = _", n_lifts=1)
    gemmini = old_fission_after(gemmini, "for kch in _:_", n_lifts=1)
    gemmini = old_lift_alloc(gemmini, "w_s : _", n_lifts=1, size=16)
    gemmini = old_lift_alloc(gemmini, "w_s : _", n_lifts=1)
    gemmini = old_lift_alloc(gemmini, "i_s : _", n_lifts=1)
    gemmini = old_lift_alloc(gemmini, "i_s : _", n_lifts=1, keep_dims=False)
    gemmini = expand_dim(gemmini, "i_s : _", "16", "0")
    gemmini = rearrange_dim(gemmini, "i_s : _", [1, 0])
    gemmini = old_fission_after(gemmini, "w_s[_] = _", n_lifts=2)
    gemmini = old_fission_after(gemmini, "i_s[_] = _", n_lifts=2)
    gemmini = divide_loop(gemmini, "j #3", 1, ["j_o", "j_i"], perfect=True)
    gemmini = divide_loop(gemmini, "j #2", 1, ["j_o", "j_i"], perfect=True)
    gemmini = simplify(gemmini)
    gemmini = reorder_stmts(gemmini, gemmini.find("a2 : _").expand(0, 1))
    gemmini = reorder_stmts(gemmini, gemmini.find("a2 = _").expand(0, 1))
    gemmini = commute_expr(gemmini, "a2 * b2")
    gemmini = rewrite_expr(gemmini, gemmini.find("b2 = _").rhs().idx()[0], "j_o")
    gemmini = rewrite_expr(gemmini, gemmini.find("a2 = _").rhs().idx()[1], "j_i")
    gemmini = rewrite_expr(gemmini, gemmini.find("res[_] += _").idx()[0], "j_o")
    gemmini = rewrite_expr(gemmini, gemmini.find("res[_] += _").idx()[1], "j_i")
    gemmini = rewrite_expr(gemmini, gemmini.find("src_tmp = _").rhs().idx()[0], "j_o")
    gemmini = rewrite_expr(gemmini, gemmini.find("src_tmp = _").rhs().idx()[1], "j_i")
    gemmini = rewrite_expr(gemmini, gemmini.find("output[_] = _").idx()[0], "j_o")
    gemmini = rewrite_expr(gemmini, gemmini.find("output[_] = _").idx()[1], "j_i")
    gemmini = set_memory(gemmini, "res : _", GEMM_ACCUM)
    gemmini = set_memory(gemmini, "w_s : _", GEMM_SCRATCH)
    gemmini = set_memory(gemmini, "i_s : _", GEMM_SCRATCH)
    gemmini = replace(gemmini, "for j in _:_ #0", ld_acc_i32_col)
    gemmini = replace(gemmini, "for j in _:_ #0", ld_i8_id1)
    gemmini = replace(gemmini, "for kch in _:_ #0", ld_i8_col)
    gemmini = replace(gemmini, "for j_o in _:_ #0", matmul_acc_i8)
    if act:
        gemmini = replace(gemmini, "for j_o in _:_ #0", st_acc_i8_relu)
        gemmini = fence_after(gemmini, "st_acc_i8_relu(_)")
    else:
        gemmini = replace(gemmini, "for j_o in _:_ #0", st_acc_i8_no_act)
        gemmini = fence_after(gemmini, "st_acc_i8_no_act(_)")

    return gemmini, cpu


def sched_conv_lowlevel(name, in_dim, in_ch, out_ch, k, act=False):
    out_dim = in_dim - k + 1
    och_blk = min(out_ch, 16)
    kch_blk = min(in_ch, 16)
    assert out_dim <= 16
    assert in_ch % kch_blk == 0
    assert out_ch % och_blk == 0

    cpu = rename(conv_relu_on_cpu if act else conv_on_cpu, f"{name}_cpu")
    cpu = cpu.partial_eval(in_dim, in_ch, out_ch, k, out_dim)

    gemmini = rename(cpu, name)
    gemmini = divide_loop(gemmini, "och", och_blk, ["och_o", "och_i"], perfect=True)
    gemmini = divide_loop(gemmini, "kch", kch_blk, ["kch_o", "kch_i"], perfect=True)
    gemmini = old_reorder(gemmini, "ocol och_o")
    gemmini = old_lift_alloc(gemmini, "res : _", n_lifts=1, size=16)
    gemmini = old_lift_alloc(gemmini, "res : _", n_lifts=1)
    gemmini = old_fission_after(gemmini, "res[_] = _", n_lifts=2)
    gemmini = old_fission_after(gemmini, "for krow in _:_", n_lifts=2)
    gemmini = old_reorder(gemmini, "och_i krow")
    gemmini = old_reorder(gemmini, "och_i kcol")
    gemmini = old_reorder(gemmini, "och_i kch_o")
    gemmini = old_reorder(gemmini, "ocol krow")
    gemmini = old_reorder(gemmini, "ocol kcol")
    gemmini = old_reorder(gemmini, "ocol kch_o")
    gemmini = simplify(gemmini)
    gemmini = old_lift_alloc(gemmini, "w_s : _", n_lifts=1)
    gemmini = old_lift_alloc(gemmini, "w_s : _", n_lifts=1, mode="col", size=16)
    gemmini = old_lift_alloc(gemmini, "w_s : _", n_lifts=1, keep_dims=False)
    gemmini = old_lift_alloc(gemmini, "i_s : _", n_lifts=1, size=16)
    gemmini = old_lift_alloc(gemmini, "i_s : _", n_lifts=1, keep_dims=False)
    gemmini = old_lift_alloc(gemmini, "i_s : _", n_lifts=1)
    gemmini = old_fission_after(gemmini, "w_s[_] = _", n_lifts=3)
    gemmini = old_fission_after(gemmini, "i_s[_] = _", n_lifts=3)
    gemmini = set_memory(gemmini, "res : _", GEMM_ACCUM)
    gemmini = set_memory(gemmini, "w_s : _", GEMM_SCRATCH)
    gemmini = set_memory(gemmini, "i_s : _", GEMM_SCRATCH)
    gemmini = old_reorder(gemmini, "och_i kch_i")
    gemmini = replace(gemmini, "for ocol in _:_ #0", ld_acc_i32_bias)
    gemmini = replace(gemmini, "for kch_i in _:_ #0", ld_i8_id1)
    gemmini = replace(gemmini, "for ocol in _:_ #0", ld_i8_id2)
    gemmini = old_reorder(gemmini, "kch_i och_i")
    gemmini = replace(gemmini, "for ocol in _:_ #0", matmul_acc_i8)
    if act:
        gemmini = replace(gemmini, "for ocol in _:_ #0", st_acc_i8_relu)
    else:
        gemmini = replace(gemmini, "for ocol in _:_ #0", st_acc_i8_no_act)
    gemmini = insert_noop_call(gemmini, gemmini.find_loop("orow").after(), fence, [])

    return gemmini, cpu


def sched_conv_im2col_lowlevel(name, in_dim, in_ch, out_ch, k):
    out_dim = in_dim - k + 1
    och_blk = min(out_ch, 16)
    assert out_dim <= 16
    assert in_dim <= 16
    assert in_ch * k <= 16
    assert out_ch % och_blk == 0

    cpu = rename(conv_on_cpu, f"{name}_cpu")
    cpu = cpu.partial_eval(in_dim, in_ch, out_ch, k, out_dim)

    gemmini = rename(cpu, name)
    gemmini = divide_loop(gemmini, "och", och_blk, ["och_o", "och_i"], perfect=True)
    gemmini = mult_loops(gemmini, gemmini.find_loop("kcol"), "k")
    gemmini = simplify(gemmini)
    gemmini = old_reorder(gemmini, "ocol och_o")
    gemmini = old_lift_alloc(gemmini, "res : _", n_lifts=1, size=16)
    gemmini = old_lift_alloc(gemmini, "res : _", n_lifts=1)
    gemmini = old_fission_after(gemmini, "res[_] = _", n_lifts=2)
    gemmini = old_fission_after(gemmini, "for krow in _:_", n_lifts=2)
    gemmini = old_reorder(gemmini, "och_i krow")
    gemmini = old_reorder(gemmini, "och_i k")
    gemmini = old_reorder(gemmini, "ocol krow")
    gemmini = old_reorder(gemmini, "ocol k")
    gemmini = simplify(gemmini)
    gemmini = old_lift_alloc(gemmini, "w_s : _", n_lifts=1)
    gemmini = old_lift_alloc(gemmini, "w_s : _", n_lifts=1, keep_dims=False)
    gemmini = old_lift_alloc(gemmini, "w_s : _", n_lifts=1, size=16)
    gemmini = old_lift_alloc(gemmini, "i_s : _", n_lifts=1, keep_dims=False)
    gemmini = old_lift_alloc(gemmini, "i_s : _", n_lifts=1, size=in_dim)
    gemmini = old_lift_alloc(gemmini, "i_s : _", n_lifts=1, mode="col", size=16)
    gemmini = old_fission_after(gemmini, "w_s[_] = _", n_lifts=3)
    gemmini = old_fission_after(gemmini, "i_s[_] = _", n_lifts=3)
    gemmini = set_memory(gemmini, "res : _", GEMM_ACCUM)
    gemmini = set_memory(gemmini, "w_s : _", GEMM_SCRATCH)
    gemmini = set_memory(gemmini, "i_s : _", GEMM_SCRATCH)
    gemmini = old_reorder(gemmini, "k ocol")
    gemmini = simplify(gemmini)
    gemmini = replace(gemmini, "for ocol in _:_ #0", ld_acc_i32_bias)
    gemmini = replace(gemmini, "for k in _:_ #0", ld_i8_id1)
    gemmini = replace(gemmini, "for ocol in _:_ #0", ld_i8_im2col)
    gemmini = old_reorder(gemmini, "k och_i")
    gemmini = replace(gemmini, "for ocol in _:_ #0", matmul_acc_i8)
    gemmini = replace(gemmini, "for ocol in _:_ #0", st_acc_i8_no_act)
    gemmini = insert_noop_call(gemmini, gemmini.find_loop("orow").after(), fence, [])

    return gemmini, cpu


conv1, conv1_cpu = sched_conv_im2col_lowlevel("conv1", INPUT_DIM, 1, CONV1_FILTERS, 3)
conv2, conv2_cpu = sched_conv_lowlevel(
    "conv2", CONV1_DIM, CONV1_FILTERS, CONV2_FILTERS, 3, act=True
)
conv3, conv3_cpu = sched_conv_lowlevel(
    "conv3", CONV2_DIM, CONV2_FILTERS, CONV3_FILTERS, 3, act=True
)

fc1, fc1_cpu = sched_fc_flat_lowlevel(
    "fc1", CONV3_FILTERS, CONV3_DIM, CONV3_DIM, FC1_UNITS, act=True
)
fc2, fc2_cpu = sched_fc_lowlevel("fc2", FC1_UNITS, FC2_UNITS, act=True)
fc3, fc3_cpu = sched_fc_lowlevel("fc3", FC2_UNITS, FC3_UNITS, act=True)
fc4, fc4_cpu = sched_fc_lowlevel("fc4", FC3_UNITS, FC4_UNITS, act=True)
fc_output, fc_output_cpu = sched_fc_lowlevel("fc_output", FC4_UNITS, OUTPUT_UNITS)


nlb_qkv_conv_cpu = rename(conv_on_cpu, "nlb_qkv_conv_cpu").partial_eval(
    CONV1_DIM, CONV1_FILTERS, CONV2_FILTERS, 1, CONV1_DIM
)
nlb_out_conv_cpu = rename(conv_on_cpu, "nlb_out_conv_cpu").partial_eval(
    CONV1_DIM, CONV2_FILTERS, CONV1_FILTERS, 1, CONV1_DIM
)
matmul_theta_phi_cpu = rename(matmul_trans_b_on_cpu, "matmul_theta_phi_cpu")
nlb_softmax_cpu = rename(softmax_on_cpu, "nlb_softmax_cpu")
matmul_attention_g_cpu = rename(matmul_on_cpu, "matmul_attention_g_cpu")
resadd_relu_cpu = rename(resadd_on_cpu, "resadd_relu_cpu")


@proc
def nlb_cpu(
    inp: i8[CONV1_DIM, CONV1_DIM, CONV1_FILTERS] @ DRAM,
    nlb_theta_weights: i8[1, 1, CONV1_FILTERS, CONV2_FILTERS] @ DRAM,
    nlb_theta_bias: i32[CONV2_FILTERS] @ DRAM,
    nlb_theta_scale: f32 @ DRAM,
    nlb_phi_weights: i8[1, 1, CONV1_FILTERS, CONV2_FILTERS] @ DRAM,
    nlb_phi_bias: i32[CONV2_FILTERS] @ DRAM,
    nlb_phi_scale: f32 @ DRAM,
    nlb_g_weights: i8[1, 1, CONV1_FILTERS, CONV2_FILTERS] @ DRAM,
    nlb_g_bias: i32[CONV2_FILTERS] @ DRAM,
    nlb_g_scale: f32 @ DRAM,
    nlb_matmul_scale: f32 @ DRAM,
    softmax_input_scale: f32 @ DRAM,
    softmax_output_scale: f32 @ DRAM,
    nlb_matmul_1_scale: f32 @ DRAM,
    nlb_out_weights: i8[1, 1, CONV2_FILTERS, CONV1_FILTERS] @ DRAM,
    nlb_out_bias: i32[CONV1_FILTERS] @ DRAM,
    nlb_out_scale: f32 @ DRAM,
    nlb_add_a_scale: f32 @ DRAM,
    nlb_add_b_scale: f32 @ DRAM,
    resadd_post_scale: f32 @ DRAM,
    output: i8[CONV1_DIM, CONV1_DIM, CONV1_FILTERS] @ DRAM,
):
    nlb_theta_out: i8[CONV1_DIM, CONV1_DIM, CONV2_FILTERS] @ DRAM_STATIC
    nlb_qkv_conv_cpu(
        inp, nlb_theta_weights, nlb_theta_bias, nlb_theta_out, nlb_theta_scale
    )

    nlb_phi_out: i8[CONV1_DIM, CONV1_DIM, CONV2_FILTERS] @ DRAM_STATIC
    nlb_qkv_conv_cpu(inp, nlb_phi_weights, nlb_phi_bias, nlb_phi_out, nlb_phi_scale)

    nlb_g_out: i8[CONV1_DIM, CONV1_DIM, CONV2_FILTERS] @ DRAM_STATIC
    nlb_qkv_conv_cpu(inp, nlb_g_weights, nlb_g_bias, nlb_g_out, nlb_g_scale)

    attention_raw: i8[CONV1_DIM, CONV1_DIM, CONV1_DIM, CONV1_DIM] @ DRAM_STATIC
    matmul_theta_phi_cpu(nlb_theta_out, nlb_phi_out, attention_raw, nlb_matmul_scale)

    attention_out: i8[CONV1_DIM, CONV1_DIM, CONV1_DIM, CONV1_DIM] @ DRAM_STATIC
    nlb_softmax_cpu(
        attention_raw, attention_out, softmax_input_scale, softmax_output_scale
    )

    attended_output: i8[CONV1_DIM, CONV1_DIM, CONV2_FILTERS] @ DRAM_STATIC
    matmul_attention_g_cpu(
        attention_out, nlb_g_out, attended_output, nlb_matmul_1_scale
    )

    nlb_output: i8[CONV1_DIM, CONV1_DIM, CONV1_FILTERS] @ DRAM_STATIC
    nlb_out_conv_cpu(
        attended_output, nlb_out_weights, nlb_out_bias, nlb_output, nlb_out_scale
    )

    resadd_relu_cpu(
        nlb_add_b_scale, nlb_add_a_scale, resadd_post_scale, inp, nlb_output, output
    )


@proc
def braggnn_inference_cpu(
    fp32_input: f32[INPUT_DIM, INPUT_DIM, 1] @ DRAM,
    conv1_weights: i8[3, 3, 1, CONV1_FILTERS] @ DRAM,
    conv1_bias: i32[CONV1_FILTERS] @ DRAM,
    conv1_scale: f32 @ DRAM,
    nlb_theta_weights: i8[1, 1, CONV1_FILTERS, CONV2_FILTERS] @ DRAM,
    nlb_theta_bias: i32[CONV2_FILTERS] @ DRAM,
    nlb_theta_scale: f32 @ DRAM,
    nlb_phi_weights: i8[1, 1, CONV1_FILTERS, CONV2_FILTERS] @ DRAM,
    nlb_phi_bias: i32[CONV2_FILTERS] @ DRAM,
    nlb_phi_scale: f32 @ DRAM,
    nlb_g_weights: i8[1, 1, CONV1_FILTERS, CONV2_FILTERS] @ DRAM,
    nlb_g_bias: i32[CONV2_FILTERS] @ DRAM,
    nlb_g_scale: f32 @ DRAM,
    nlb_matmul_scale: f32 @ DRAM,
    softmax_input_scale: f32 @ DRAM,
    softmax_output_scale: f32 @ DRAM,
    nlb_matmul_1_scale: f32 @ DRAM,
    nlb_out_weights: i8[1, 1, CONV2_FILTERS, CONV1_FILTERS] @ DRAM,
    nlb_out_bias: i32[CONV1_FILTERS] @ DRAM,
    nlb_out_scale: f32 @ DRAM,
    nlb_add_a_scale: f32 @ DRAM,
    nlb_add_b_scale: f32 @ DRAM,
    resadd_post_scale: f32 @ DRAM,
    conv2_weights: i8[3, 3, CONV1_FILTERS, CONV2_FILTERS] @ DRAM,
    conv2_bias: i32[CONV2_FILTERS] @ DRAM,
    conv2_scale: f32 @ DRAM,
    conv2_post_scale: f32 @ DRAM,
    conv3_weights: i8[3, 3, CONV2_FILTERS, CONV3_FILTERS] @ DRAM,
    conv3_bias: i32[CONV3_FILTERS] @ DRAM,
    conv3_scale: f32 @ DRAM,
    conv3_post_scale: f32 @ DRAM,
    fc1_weights: i8[FC1_UNITS, CONV3_FILTERS, CONV3_DIM, CONV3_DIM] @ DRAM,
    fc1_bias: i32[FC1_UNITS] @ DRAM,
    fc1_scale: f32 @ DRAM,
    fc1_post_scale: f32 @ DRAM,
    fc2_weights: i8[FC2_UNITS, FC1_UNITS, 1, 1] @ DRAM,
    fc2_bias: i32[FC2_UNITS] @ DRAM,
    fc2_scale: f32 @ DRAM,
    fc2_post_scale: f32 @ DRAM,
    fc3_weights: i8[FC3_UNITS, FC2_UNITS, 1, 1] @ DRAM,
    fc3_bias: i32[FC3_UNITS] @ DRAM,
    fc3_scale: f32 @ DRAM,
    fc3_post_scale: f32 @ DRAM,
    fc4_weights: i8[FC4_UNITS, FC3_UNITS, 1, 1] @ DRAM,
    fc4_bias: i32[FC4_UNITS] @ DRAM,
    fc4_scale: f32 @ DRAM,
    fc4_post_scale: f32 @ DRAM,
    output_weights: i8[OUTPUT_UNITS, FC4_UNITS, 1, 1] @ DRAM,
    output_bias: i32[OUTPUT_UNITS] @ DRAM,
    output_scale: f32 @ DRAM,
    output: i8[OUTPUT_UNITS, 1, 1] @ DRAM,
):
    inp: i8[INPUT_DIM, INPUT_DIM, 1] @ DRAM_STATIC
    for h in seq(0, INPUT_DIM):
        for w in seq(0, INPUT_DIM):
            q: f32
            q = fp32_input[h, w, 0] * 127.0
            inp[h, w, 0] = q

    conv1_out: i8[CONV1_DIM, CONV1_DIM, CONV1_FILTERS] @ DRAM_STATIC
    conv1_cpu(inp, conv1_weights, conv1_bias, conv1_out, conv1_scale)

    nlb_out: i8[CONV1_DIM, CONV1_DIM, CONV1_FILTERS] @ DRAM_STATIC
    nlb_cpu(
        conv1_out,
        nlb_theta_weights,
        nlb_theta_bias,
        nlb_theta_scale,
        nlb_phi_weights,
        nlb_phi_bias,
        nlb_phi_scale,
        nlb_g_weights,
        nlb_g_bias,
        nlb_g_scale,
        nlb_matmul_scale,
        softmax_input_scale,
        softmax_output_scale,
        nlb_matmul_1_scale,
        nlb_out_weights,
        nlb_out_bias,
        nlb_out_scale,
        nlb_add_a_scale,
        nlb_add_b_scale,
        resadd_post_scale,
        nlb_out,
    )

    conv2_out: i8[CONV2_DIM, CONV2_DIM, CONV2_FILTERS] @ DRAM_STATIC
    conv2_cpu(
        nlb_out, conv2_weights, conv2_bias, conv2_out, conv2_scale, conv2_post_scale
    )

    conv3_out: i8[CONV3_DIM, CONV3_DIM, CONV3_FILTERS] @ DRAM_STATIC
    conv3_cpu(
        conv2_out, conv3_weights, conv3_bias, conv3_out, conv3_scale, conv3_post_scale
    )

    flattened: i8[CONV3_FILTERS, CONV3_DIM, CONV3_DIM] @ DRAM_STATIC
    for ch in seq(0, CONV3_FILTERS):
        for r in seq(0, CONV3_DIM):
            for c in seq(0, CONV3_DIM):
                flattened[ch, r, c] = conv3_out[r, c, ch]

    fc1_out: i8[FC1_UNITS, 1, 1] @ DRAM_STATIC
    fc1_cpu(flattened, fc1_weights, fc1_bias, fc1_out, fc1_scale, fc1_post_scale)

    fc2_out: i8[FC2_UNITS, 1, 1] @ DRAM_STATIC
    fc2_cpu(fc1_out, fc2_weights, fc2_bias, fc2_out, fc2_scale, fc2_post_scale)

    fc3_out: i8[FC3_UNITS, 1, 1] @ DRAM_STATIC
    fc3_cpu(fc2_out, fc3_weights, fc3_bias, fc3_out, fc3_scale, fc3_post_scale)

    fc4_out: i8[FC4_UNITS, 1, 1] @ DRAM_STATIC
    fc4_cpu(fc3_out, fc4_weights, fc4_bias, fc4_out, fc4_scale, fc4_post_scale)

    fc_output_cpu(fc4_out, output_weights, output_bias, output, output_scale)


def tile_matmul(sub, n_reg, do_mm, n_kblk=1, b_col=False):
    sub = old_lift_alloc(sub, "res : _", n_lifts=1, size=16)
    sub = old_lift_alloc(sub, "res : _", n_lifts=1)
    sub = old_fission_after(sub, "res[_] = _", n_lifts=2)
    sub = old_fission_after(sub, "for k_o in _:_", n_lifts=2)
    sub = old_reorder(sub, "j_i k_o")
    sub = old_reorder(sub, "i_i k_o")
    sub = simplify(sub)
    sub = old_lift_alloc(sub, "b : _", n_lifts=1)
    if b_col:
        sub = old_lift_alloc(sub, "b : _", n_lifts=1, mode="col")
    else:
        sub = old_lift_alloc(sub, "b : _", n_lifts=1)
    sub = old_lift_alloc(sub, "b : _", n_lifts=1, keep_dims=False)
    sub = old_lift_alloc(sub, "a : _", n_lifts=1, size=16)
    sub = old_lift_alloc(sub, "a : _", n_lifts=1, keep_dims=False)
    sub = old_lift_alloc(sub, "a : _", n_lifts=1)
    sub = old_fission_after(sub, "b[_] = _", n_lifts=3)
    sub = old_fission_after(sub, "a[_] = _", n_lifts=3)
    for r in range(n_reg):
        sub = set_memory(sub, f"res : _ #{r}", GEMM_ACCUM)
    for r in range(n_reg * n_kblk):
        sub = set_memory(sub, f"b : _ #{r}", GEMM_SCRATCH)
        sub = set_memory(sub, f"a : _ #{r}", GEMM_SCRATCH)
    for _ in range(n_reg):
        sub = replace(sub, "for i_i in _:_ #0", zero_acc_i32)
        for _ in range(n_kblk):
            sub = replace(sub, "for i_i in _:_ #0", ld_i8_id2)
            if b_col:
                sub = reorder_loops(sub, "for j_i in _:_ #0")
                sub = replace(sub, "for k_i in _:_ #0", ld_i8_id1)
            else:
                sub = replace(sub, "for j_i in _:_ #0", ld_i8_id1)
            sub = replace(sub, "for i_i in _:_ #0", do_mm)
        sub = replace(sub, "for i_i in _:_ #0", st_acc_i8_no_act)
    return sub


def sched_conv_sub(sub, name, in_ch, out_ch):
    och_blk, kch_blk = min(out_ch, 16), min(in_ch, 16)
    g = rename(sub, name)
    g = divide_loop(g, "och", och_blk, ["och_o", "och_i"], perfect=True)
    g = divide_loop(g, "kch", kch_blk, ["kch_o", "kch_i"], perfect=True)
    g = old_reorder(g, "ocol och_o")
    g = old_lift_alloc(g, "res : _", n_lifts=1, size=16)
    g = old_lift_alloc(g, "res : _", n_lifts=1)
    g = old_fission_after(g, "res[_] = _", n_lifts=2)
    g = old_fission_after(g, "for krow in _:_", n_lifts=2)
    for a, b in (
        ("och_i", "krow"),
        ("och_i", "kcol"),
        ("och_i", "kch_o"),
        ("ocol", "krow"),
        ("ocol", "kcol"),
        ("ocol", "kch_o"),
    ):
        g = old_reorder(g, f"{a} {b}")
    g = simplify(g)
    g = old_lift_alloc(g, "w_s : _", n_lifts=1)
    g = old_lift_alloc(g, "w_s : _", n_lifts=1, mode="col", size=16)
    g = old_lift_alloc(g, "w_s : _", n_lifts=1, keep_dims=False)
    g = old_lift_alloc(g, "i_s : _", n_lifts=1, size=16)
    g = old_lift_alloc(g, "i_s : _", n_lifts=1, keep_dims=False)
    g = old_lift_alloc(g, "i_s : _", n_lifts=1)
    g = old_fission_after(g, "w_s[_] = _", n_lifts=3)
    g = old_fission_after(g, "i_s[_] = _", n_lifts=3)
    g = set_memory(g, "res : _", GEMM_ACCUM)
    g = set_memory(g, "w_s : _", GEMM_SCRATCH)
    g = set_memory(g, "i_s : _", GEMM_SCRATCH)
    g = old_reorder(g, "och_i kch_i")
    g = replace(g, "for ocol in _:_ #0", ld_acc_i32_bias)
    g = replace(g, "for kch_i in _:_ #0", ld_i8_id1)
    g = replace(g, "for ocol in _:_ #0", ld_i8_id2)
    g = old_reorder(g, "kch_i och_i")
    g = replace(g, "for ocol in _:_ #0", matmul_acc_i8)
    g = replace(g, "for ocol in _:_ #0", st_acc_i8_no_act)
    return insert_noop_call(g, g.find_loop("orow").after(), fence, [])


def schedule_nlb():
    p = rename(nlb_cpu, "nlb")
    for pat in ("nlb_qkv_conv_cpu(_)",) * 3 + (
        "matmul_theta_phi_cpu(_)",
        "nlb_softmax_cpu(_)",
        "matmul_attention_g_cpu(_)",
        "nlb_out_conv_cpu(_)",
        "resadd_relu_cpu(_)",
    ):
        p = inline(p, pat)
    for buf, pairs in (
        ("nlb_theta_out", [(0, 1)]),
        ("nlb_phi_out", [(0, 1)]),
        ("nlb_g_out", [(0, 1)]),
        ("attention_raw", [(0, 1), (1, 2)]),
        ("attention_out", [(0, 1), (1, 2)]),
        ("attended_output", [(0, 1)]),
        ("nlb_output", [(0, 1)]),
    ):
        for hi, lo in pairs:
            p = mult_dim(p, f"{buf} : _", hi, lo)
    p = simplify(p)

    p, qkv = extract_subproc(p, p.find_loop("orow #0"), "qkv_conv_flat")
    p = replace(p, p.find_loop("orow #0"), qkv)
    p = replace(p, p.find_loop("orow #0"), qkv)
    p, tp = extract_subproc(p, p.find_loop("i1 #0"), "theta_phi_flat")
    p, sm = extract_subproc(p, p.find_loop("i1 #0"), "softmax_flat")
    p, ag = extract_subproc(p, p.find_loop("i1 #0"), "attention_g_flat")
    p, oc = extract_subproc(p, p.find_loop("orow #0"), "out_conv_flat")
    p, ra = extract_subproc(p, p.find_loop("i1 #0"), "resadd_flat")

    tp2 = mult_loops(tp, tp.find_loop("i1"), "i")
    tp2 = mult_loops(tp2, tp2.find_loop("j1"), "j")
    tp2 = simplify(tp2)
    tp2 = divide_loop(tp2, "i", 16, ["i_o", "i_i"], tail="cut")
    tp2 = divide_loop(tp2, "j", 16, ["j_o", "j_i"], tail="cut")
    tp2 = divide_loop(tp2, "j", 16, ["j_o", "j_i"], tail="cut")
    tp2 = old_fission_after(tp2, "for j_o in _:_ #0", n_lifts=1)
    tp2 = old_fission_after(tp2, "for j_o in _:_ #1", n_lifts=1)
    tp2 = old_reorder(tp2, "i_i j_o")
    tp2 = old_reorder(tp2, "i_i j_i")
    tp2 = old_reorder(tp2, "j_i i_i")
    for _ in range(4):
        tp2 = divide_loop(tp2, "k", 16, ["k_o", "k_i"], perfect=True)
    tp2 = simplify(tp2)
    tp2 = tile_matmul(rename(tp2, "theta_phi_gemm"), 4, matmul_acc_i8_trans_b)

    ag2 = mult_loops(ag, ag.find_loop("i1"), "i")
    ag2 = mult_loops(ag2, ag2.find_loop("k1"), "k")
    ag2 = simplify(ag2)
    ag2 = divide_loop(ag2, "i", 16, ["i_o", "i_i"], tail="cut")
    ag2 = divide_loop(ag2, "j", 16, ["j_o", "j_i"], perfect=True)
    ag2 = divide_loop(ag2, "j", 16, ["j_o", "j_i"], perfect=True)
    ag2 = old_reorder(ag2, "i_i j_o")
    ag2 = simplify(ag2)
    for _ in range(2):
        ag2 = divide_loop(ag2, "k", 16, ["k_o", "k_i"], tail="cut")
    ag2 = divide_loop(ag2, "k_i #3", 1, ["k_o", "k_i"], perfect=True)
    ag2 = divide_loop(ag2, "k_i #1", 1, ["k_o", "k_i"], perfect=True)
    ag2 = simplify(ag2)
    ag2 = tile_matmul(
        rename(ag2, "attention_g_gemm"), 2, matmul_acc_i8, n_kblk=2, b_col=True
    )

    sm2 = mult_loops(sm, sm.find_loop("i1"), "i")
    for _ in range(3):
        sm2 = mult_loops(sm2, sm2.find_loop("j1"), "j")
    sm2 = mult_dim(sm2, "exp_buf : _", 0, 1)
    sm2 = simplify(sm2)
    sm2 = rename(sm2, "softmax_gemm")
    sm2 = divide_loop(sm2, "i", NLB_SOFTMAX_TILE, ["i_o", "i_i"], tail="cut")
    do_sm = make_loop_softmax_flat("do_softmax", NLB_ROWS, SOFTMAX_MAX_SHIFT)
    sm2 = replace(sm2, sm2.find_loop("i_o").body(), do_sm)
    sm2 = replace(sm2, sm2.find_loop("i_o").next(), do_sm)
    sm2 = insert_noop_call(sm2, sm2.find_loop("i_o").next().after(), fence, [])

    qkv2 = sched_conv_sub(qkv, "qkv_conv_gemm", CONV1_FILTERS, CONV2_FILTERS)
    oc2 = sched_conv_sub(oc, "out_conv_gemm", CONV2_FILTERS, CONV1_FILTERS)

    ra2 = rename(ra, "resadd_gemm")
    ra2 = divide_loop(ra2, "j", 16, ["j_o", "j_i"], perfect=True)
    ra2 = old_reorder(ra2, "i2 j_o")
    ra2 = old_lift_alloc(ra2, "res : _", n_lifts=1, size=16)
    ra2 = old_lift_alloc(ra2, "res : _", n_lifts=1)
    ra2 = split_write(ra2, "res[_] = _")
    for stmt in (
        "tmp_res2 : _",
        "tmp_res1 : _",
        "src_tmp : _",
        "b2 = b_q",
        "clamp(b_tmp, b_q)",
        "acc_scale(b_src, b_tmp, nlb_add_a_scale)",
        "b_src = _",
        "b2 : _",
        "b_q : _",
        "b_tmp : _",
        "b_src : _",
    ):
        ra2 = reorder_stmts(ra2, f"{stmt} ; res[_] = a2")
    for stmt in ("tmp_res2 : _", "tmp_res1 : _", "src_tmp : _"):
        ra2 = reorder_stmts(ra2, f"{stmt} ; res[_] += b2")
    ra2 = old_fission_after(ra2, "res[_] = a2", n_lifts=2)
    ra2 = old_fission_after(ra2, "res[_] += b2", n_lifts=2)
    ra2 = set_memory(ra2, "res : _", GEMM_ACCUM)
    ra2 = replace(ra2, "for i2 in _:_ #0", ld_acc_i8_scaled)
    ra2 = fence_after(ra2, "ld_acc_i8_scaled(_)")
    ra2 = replace(ra2, "for i2 in _:_ #0", ld_acc_i8_scaled_acc)
    ra2 = replace(ra2, "for i2 in _:_ #0", st_acc_i8_act)
    ra2 = insert_noop_call(ra2, ra2.find_loop("i1").after(), fence, [])

    for old, new in (
        (qkv, qkv2),
        (tp, tp2),
        (sm, sm2),
        (ag, ag2),
        (oc, oc2),
        (ra, ra2),
    ):
        for _ in range(len(p.find_all(f"{old.name()}(_)"))):
            p = call_eqv(p, f"{old.name()}(_)", new)
    return p


nlb = schedule_nlb()


def schedule_braggnn():
    gemmini = rename(braggnn_inference_cpu, "braggnn_inference")
    gemmini = call_eqv(gemmini, "conv1_cpu(_)", conv1)
    gemmini = call_eqv(gemmini, "nlb_cpu(_)", nlb)
    gemmini = call_eqv(gemmini, "conv2_cpu(_)", conv2)
    gemmini = call_eqv(gemmini, "conv3_cpu(_)", conv3)
    gemmini = call_eqv(gemmini, "fc1_cpu(_)", fc1)
    gemmini = call_eqv(gemmini, "fc2_cpu(_)", fc2)
    gemmini = call_eqv(gemmini, "fc3_cpu(_)", fc3)
    gemmini = call_eqv(gemmini, "fc4_cpu(_)", fc4)
    gemmini = call_eqv(gemmini, "fc_output_cpu(_)", fc_output)
    return gemmini


braggnn_inference = schedule_braggnn()

__all__ = ["braggnn_inference"]
