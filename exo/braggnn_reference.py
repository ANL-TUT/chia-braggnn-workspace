# ruff: noqa: F821

from __future__ import annotations

from exo.API_scheduling import rename
from exo.libs.externs import relu, select
from exo.libs.memories import DRAM_STATIC
from exo.platforms.gemmini import acc_scale, clamp
from gemmini import layer_mark, rdcycle

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

SOFTMAX_INPUT_SCALE = 0.028342675417661667


def max_softmax_shift(in_scale):
    qln2 = int(0.693147 / in_scale)
    qln2_inv = 65536 // qln2
    return (255 * qln2_inv) // 65536 + 1


SOFTMAX_MAX_SHIFT = max_softmax_shift(SOFTMAX_INPUT_SCALE)


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


conv1_cpu = rename(conv_on_cpu, "conv1_cpu").partial_eval(
    INPUT_DIM, 1, CONV1_FILTERS, 3, CONV1_DIM
)
conv2_cpu = rename(conv_relu_on_cpu, "conv2_cpu").partial_eval(
    CONV1_DIM, CONV1_FILTERS, CONV2_FILTERS, 3, CONV2_DIM
)
conv3_cpu = rename(conv_relu_on_cpu, "conv3_cpu").partial_eval(
    CONV2_DIM, CONV2_FILTERS, CONV3_FILTERS, 3, CONV3_DIM
)
nlb_qkv_conv_cpu = rename(conv_on_cpu, "nlb_qkv_conv_cpu").partial_eval(
    CONV1_DIM, CONV1_FILTERS, CONV2_FILTERS, 1, CONV1_DIM
)
nlb_out_conv_cpu = rename(conv_on_cpu, "nlb_out_conv_cpu").partial_eval(
    CONV1_DIM, CONV2_FILTERS, CONV1_FILTERS, 1, CONV1_DIM
)

fc1_cpu = rename(fc_relu_on_cpu, "fc1_cpu").partial_eval(
    CONV3_FILTERS, CONV3_DIM, CONV3_DIM, FC1_UNITS
)
fc2_cpu = rename(fc_relu_on_cpu, "fc2_cpu").partial_eval(FC1_UNITS, 1, 1, FC2_UNITS)
fc3_cpu = rename(fc_relu_on_cpu, "fc3_cpu").partial_eval(FC2_UNITS, 1, 1, FC3_UNITS)
fc4_cpu = rename(fc_relu_on_cpu, "fc4_cpu").partial_eval(FC3_UNITS, 1, 1, FC4_UNITS)
fc_output_cpu = rename(fc_on_cpu, "fc_output_cpu").partial_eval(
    FC4_UNITS, 1, 1, OUTPUT_UNITS
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
    layer_mark(4)

    attention_raw: i8[CONV1_DIM, CONV1_DIM, CONV1_DIM, CONV1_DIM] @ DRAM_STATIC
    matmul_theta_phi_cpu(nlb_theta_out, nlb_phi_out, attention_raw, nlb_matmul_scale)
    layer_mark(5)

    attention_out: i8[CONV1_DIM, CONV1_DIM, CONV1_DIM, CONV1_DIM] @ DRAM_STATIC
    nlb_softmax_cpu(
        attention_raw, attention_out, softmax_input_scale, softmax_output_scale
    )
    layer_mark(6)

    attended_output: i8[CONV1_DIM, CONV1_DIM, CONV2_FILTERS] @ DRAM_STATIC
    matmul_attention_g_cpu(
        attention_out, nlb_g_out, attended_output, nlb_matmul_1_scale
    )
    layer_mark(7)

    nlb_output: i8[CONV1_DIM, CONV1_DIM, CONV1_FILTERS] @ DRAM_STATIC
    nlb_out_conv_cpu(
        attended_output, nlb_out_weights, nlb_out_bias, nlb_output, nlb_out_scale
    )
    layer_mark(8)

    resadd_relu_cpu(
        nlb_add_b_scale, nlb_add_a_scale, resadd_post_scale, inp, nlb_output, output
    )
    layer_mark(9)


@proc
def braggnn_inference_cpu(
    fp32_input: f32[INPUT_DIM, INPUT_DIM] @ DRAM,
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
    layer_mark(1)
    inp: i8[INPUT_DIM, INPUT_DIM, 1] @ DRAM_STATIC
    for h in seq(0, INPUT_DIM):
        for w in seq(0, INPUT_DIM):
            q: f32
            q = fp32_input[h, w] * 127.0
            inp[h, w, 0] = q
    layer_mark(2)

    conv1_out: i8[CONV1_DIM, CONV1_DIM, CONV1_FILTERS] @ DRAM_STATIC
    conv1_cpu(inp, conv1_weights, conv1_bias, conv1_out, conv1_scale)
    layer_mark(3)

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
    layer_mark(10)

    conv3_out: i8[CONV3_DIM, CONV3_DIM, CONV3_FILTERS] @ DRAM_STATIC
    conv3_cpu(
        conv2_out, conv3_weights, conv3_bias, conv3_out, conv3_scale, conv3_post_scale
    )
    layer_mark(11)

    flattened: i8[CONV3_FILTERS, CONV3_DIM, CONV3_DIM] @ DRAM_STATIC
    for ch in seq(0, CONV3_FILTERS):
        for r in seq(0, CONV3_DIM):
            for c in seq(0, CONV3_DIM):
                flattened[ch, r, c] = conv3_out[r, c, ch]

    fc1_out: i8[FC1_UNITS, 1, 1] @ DRAM_STATIC
    fc1_cpu(flattened, fc1_weights, fc1_bias, fc1_out, fc1_scale, fc1_post_scale)
    layer_mark(12)

    fc2_out: i8[FC2_UNITS, 1, 1] @ DRAM_STATIC
    fc2_cpu(fc1_out, fc2_weights, fc2_bias, fc2_out, fc2_scale, fc2_post_scale)

    fc3_out: i8[FC3_UNITS, 1, 1] @ DRAM_STATIC
    fc3_cpu(fc2_out, fc3_weights, fc3_bias, fc3_out, fc3_scale, fc3_post_scale)

    fc4_out: i8[FC4_UNITS, 1, 1] @ DRAM_STATIC
    fc4_cpu(fc3_out, fc4_weights, fc4_bias, fc4_out, fc4_scale, fc4_post_scale)

    fc_output_cpu(fc4_out, output_weights, output_bias, output, output_scale)
    layer_mark(13)


@proc
def braggnn_eval_cpu(
    n_patches: size,
    fp32_inputs: f32[n_patches, INPUT_DIM, INPUT_DIM] @ DRAM,
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
    cycles: i32[n_patches] @ DRAM,
    outputs: i8[n_patches, OUTPUT_UNITS, 1, 1] @ DRAM,
):
    conv1_weights_: i8[3, 3, 1, CONV1_FILTERS] @ DRAM_STATIC
    for i0 in seq(0, 3):
        for i1 in seq(0, 3):
            for i2 in seq(0, 1):
                for i3 in seq(0, CONV1_FILTERS):
                    conv1_weights_[i0, i1, i2, i3] = conv1_weights[i0, i1, i2, i3]
    conv1_bias_: i32[CONV1_FILTERS] @ DRAM_STATIC
    for i0 in seq(0, CONV1_FILTERS):
        conv1_bias_[i0] = conv1_bias[i0]
    nlb_theta_weights_: i8[1, 1, CONV1_FILTERS, CONV2_FILTERS] @ DRAM_STATIC
    for i0 in seq(0, 1):
        for i1 in seq(0, 1):
            for i2 in seq(0, CONV1_FILTERS):
                for i3 in seq(0, CONV2_FILTERS):
                    nlb_theta_weights_[i0, i1, i2, i3] = nlb_theta_weights[
                        i0, i1, i2, i3
                    ]
    nlb_theta_bias_: i32[CONV2_FILTERS] @ DRAM_STATIC
    for i0 in seq(0, CONV2_FILTERS):
        nlb_theta_bias_[i0] = nlb_theta_bias[i0]
    nlb_phi_weights_: i8[1, 1, CONV1_FILTERS, CONV2_FILTERS] @ DRAM_STATIC
    for i0 in seq(0, 1):
        for i1 in seq(0, 1):
            for i2 in seq(0, CONV1_FILTERS):
                for i3 in seq(0, CONV2_FILTERS):
                    nlb_phi_weights_[i0, i1, i2, i3] = nlb_phi_weights[i0, i1, i2, i3]
    nlb_phi_bias_: i32[CONV2_FILTERS] @ DRAM_STATIC
    for i0 in seq(0, CONV2_FILTERS):
        nlb_phi_bias_[i0] = nlb_phi_bias[i0]
    nlb_g_weights_: i8[1, 1, CONV1_FILTERS, CONV2_FILTERS] @ DRAM_STATIC
    for i0 in seq(0, 1):
        for i1 in seq(0, 1):
            for i2 in seq(0, CONV1_FILTERS):
                for i3 in seq(0, CONV2_FILTERS):
                    nlb_g_weights_[i0, i1, i2, i3] = nlb_g_weights[i0, i1, i2, i3]
    nlb_g_bias_: i32[CONV2_FILTERS] @ DRAM_STATIC
    for i0 in seq(0, CONV2_FILTERS):
        nlb_g_bias_[i0] = nlb_g_bias[i0]
    nlb_out_weights_: i8[1, 1, CONV2_FILTERS, CONV1_FILTERS] @ DRAM_STATIC
    for i0 in seq(0, 1):
        for i1 in seq(0, 1):
            for i2 in seq(0, CONV2_FILTERS):
                for i3 in seq(0, CONV1_FILTERS):
                    nlb_out_weights_[i0, i1, i2, i3] = nlb_out_weights[i0, i1, i2, i3]
    nlb_out_bias_: i32[CONV1_FILTERS] @ DRAM_STATIC
    for i0 in seq(0, CONV1_FILTERS):
        nlb_out_bias_[i0] = nlb_out_bias[i0]
    conv2_weights_: i8[3, 3, CONV1_FILTERS, CONV2_FILTERS] @ DRAM_STATIC
    for i0 in seq(0, 3):
        for i1 in seq(0, 3):
            for i2 in seq(0, CONV1_FILTERS):
                for i3 in seq(0, CONV2_FILTERS):
                    conv2_weights_[i0, i1, i2, i3] = conv2_weights[i0, i1, i2, i3]
    conv2_bias_: i32[CONV2_FILTERS] @ DRAM_STATIC
    for i0 in seq(0, CONV2_FILTERS):
        conv2_bias_[i0] = conv2_bias[i0]
    conv3_weights_: i8[3, 3, CONV2_FILTERS, CONV3_FILTERS] @ DRAM_STATIC
    for i0 in seq(0, 3):
        for i1 in seq(0, 3):
            for i2 in seq(0, CONV2_FILTERS):
                for i3 in seq(0, CONV3_FILTERS):
                    conv3_weights_[i0, i1, i2, i3] = conv3_weights[i0, i1, i2, i3]
    conv3_bias_: i32[CONV3_FILTERS] @ DRAM_STATIC
    for i0 in seq(0, CONV3_FILTERS):
        conv3_bias_[i0] = conv3_bias[i0]
    fc1_weights_: i8[FC1_UNITS, CONV3_FILTERS, CONV3_DIM, CONV3_DIM] @ DRAM_STATIC
    for i0 in seq(0, FC1_UNITS):
        for i1 in seq(0, CONV3_FILTERS):
            for i2 in seq(0, CONV3_DIM):
                for i3 in seq(0, CONV3_DIM):
                    fc1_weights_[i0, i1, i2, i3] = fc1_weights[i0, i1, i2, i3]
    fc1_bias_: i32[FC1_UNITS] @ DRAM_STATIC
    for i0 in seq(0, FC1_UNITS):
        fc1_bias_[i0] = fc1_bias[i0]
    fc2_weights_: i8[FC2_UNITS, FC1_UNITS, 1, 1] @ DRAM_STATIC
    for i0 in seq(0, FC2_UNITS):
        for i1 in seq(0, FC1_UNITS):
            for i2 in seq(0, 1):
                for i3 in seq(0, 1):
                    fc2_weights_[i0, i1, i2, i3] = fc2_weights[i0, i1, i2, i3]
    fc2_bias_: i32[FC2_UNITS] @ DRAM_STATIC
    for i0 in seq(0, FC2_UNITS):
        fc2_bias_[i0] = fc2_bias[i0]
    fc3_weights_: i8[FC3_UNITS, FC2_UNITS, 1, 1] @ DRAM_STATIC
    for i0 in seq(0, FC3_UNITS):
        for i1 in seq(0, FC2_UNITS):
            for i2 in seq(0, 1):
                for i3 in seq(0, 1):
                    fc3_weights_[i0, i1, i2, i3] = fc3_weights[i0, i1, i2, i3]
    fc3_bias_: i32[FC3_UNITS] @ DRAM_STATIC
    for i0 in seq(0, FC3_UNITS):
        fc3_bias_[i0] = fc3_bias[i0]
    fc4_weights_: i8[FC4_UNITS, FC3_UNITS, 1, 1] @ DRAM_STATIC
    for i0 in seq(0, FC4_UNITS):
        for i1 in seq(0, FC3_UNITS):
            for i2 in seq(0, 1):
                for i3 in seq(0, 1):
                    fc4_weights_[i0, i1, i2, i3] = fc4_weights[i0, i1, i2, i3]
    fc4_bias_: i32[FC4_UNITS] @ DRAM_STATIC
    for i0 in seq(0, FC4_UNITS):
        fc4_bias_[i0] = fc4_bias[i0]
    output_weights_: i8[OUTPUT_UNITS, FC4_UNITS, 1, 1] @ DRAM_STATIC
    for i0 in seq(0, OUTPUT_UNITS):
        for i1 in seq(0, FC4_UNITS):
            for i2 in seq(0, 1):
                for i3 in seq(0, 1):
                    output_weights_[i0, i1, i2, i3] = output_weights[i0, i1, i2, i3]
    output_bias_: i32[OUTPUT_UNITS] @ DRAM_STATIC
    for i0 in seq(0, OUTPUT_UNITS):
        output_bias_[i0] = output_bias[i0]

    for p in seq(0, n_patches):
        fp32_patch: f32[INPUT_DIM, INPUT_DIM] @ DRAM
        for irow in seq(0, INPUT_DIM):
            for icol in seq(0, INPUT_DIM):
                fp32_patch[irow, icol] = fp32_inputs[p, irow, icol]
        pred: i8[OUTPUT_UNITS, 1, 1] @ DRAM
        begin: i32
        begin = rdcycle()
        braggnn_inference_cpu(
            fp32_patch,
            conv1_weights_,
            conv1_bias_,
            conv1_scale,
            nlb_theta_weights_,
            nlb_theta_bias_,
            nlb_theta_scale,
            nlb_phi_weights_,
            nlb_phi_bias_,
            nlb_phi_scale,
            nlb_g_weights_,
            nlb_g_bias_,
            nlb_g_scale,
            nlb_matmul_scale,
            softmax_input_scale,
            softmax_output_scale,
            nlb_matmul_1_scale,
            nlb_out_weights_,
            nlb_out_bias_,
            nlb_out_scale,
            nlb_add_a_scale,
            nlb_add_b_scale,
            resadd_post_scale,
            conv2_weights_,
            conv2_bias_,
            conv2_scale,
            conv2_post_scale,
            conv3_weights_,
            conv3_bias_,
            conv3_scale,
            conv3_post_scale,
            fc1_weights_,
            fc1_bias_,
            fc1_scale,
            fc1_post_scale,
            fc2_weights_,
            fc2_bias_,
            fc2_scale,
            fc2_post_scale,
            fc3_weights_,
            fc3_bias_,
            fc3_scale,
            fc3_post_scale,
            fc4_weights_,
            fc4_bias_,
            fc4_scale,
            fc4_post_scale,
            output_weights_,
            output_bias_,
            output_scale,
            pred,
        )
        end: i32
        end = rdcycle()
        cycles[p] = end - begin
        for k in seq(0, OUTPUT_UNITS):
            outputs[p, k, 0, 0] = pred[k, 0, 0]
