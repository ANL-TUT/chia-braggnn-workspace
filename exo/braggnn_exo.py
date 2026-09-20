# ruff: noqa: F811, F821

from __future__ import annotations

from exo.API_scheduling import (
    call_eqv,
    divide_loop,
    insert_noop_call,
    rename,
    replace,
    simplify,
)
from exo.libs.externs import expf, relu, select
from exo.libs.memories import DRAM_STATIC
from exo.platforms.gemmini import acc_scale, clamp
from gemmini import (
    fence,
    make_loop_conv_ws,
    make_loop_matmul_attention_g,
    make_loop_matmul_fc,
    make_loop_matmul_theta_phi_tile,
    make_loop_resadd_relu,
    make_loop_softmax_tile,
)

from exo import DRAM, proc

__all__ = ["braggnn_inference"]

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


def _make_conv2d(in_h, in_w, in_ch, out_ch, kernel_size, out_h, out_w):
    assert in_h >= out_h + kernel_size - 1
    assert in_w >= out_w + kernel_size - 1

    @proc
    def conv2d(
        input: i8[in_h, in_w, in_ch] @ DRAM,
        weights: i8[kernel_size, kernel_size, in_ch, out_ch] @ DRAM,
        bias: i32[out_ch] @ DRAM,
        output: i8[out_h, out_w, out_ch] @ DRAM,
        scale: f32 @ DRAM,
    ):
        for oh in seq(0, out_h):
            for ow in seq(0, out_w):
                for oc in seq(0, out_ch):
                    sum: i32
                    sum = bias[oc]

                    for kh in seq(0, kernel_size):
                        for kw in seq(0, kernel_size):
                            for ic in seq(0, in_ch):
                                x: i32
                                w: i32
                                x = input[oh + kh, ow + kw, ic]
                                w = weights[kh, kw, ic, oc]
                                sum += x * w

                    tmp: f32
                    acc_scale(sum, tmp, scale)
                    out_val: i8
                    clamp(tmp, out_val)
                    output[oh, ow, oc] = out_val

    return conv2d


def _make_conv2d_relu(in_h, in_w, in_ch, out_ch, kernel_size, out_h, out_w):
    assert in_h >= out_h + kernel_size - 1
    assert in_w >= out_w + kernel_size - 1

    @proc
    def conv2d_relu(
        input: i8[in_h, in_w, in_ch] @ DRAM,
        weights: i8[kernel_size, kernel_size, in_ch, out_ch] @ DRAM,
        bias: i32[out_ch] @ DRAM,
        output: i8[out_h, out_w, out_ch] @ DRAM,
        scale: f32 @ DRAM,
        relu_scale: f32 @ DRAM,
    ):
        for oh in seq(0, out_h):
            for ow in seq(0, out_w):
                for oc in seq(0, out_ch):
                    sum: i32
                    sum = bias[oc]

                    for kh in seq(0, kernel_size):
                        for kw in seq(0, kernel_size):
                            for ic in seq(0, in_ch):
                                x: i32
                                w: i32
                                x = input[oh + kh, ow + kw, ic]
                                w = weights[kh, kw, ic, oc]
                                sum += x * w

                    sf: f32
                    sf = scale * relu_scale
                    tmp: f32
                    acc_scale(sum, tmp, sf)
                    out_val: i8
                    clamp(tmp, out_val)
                    out_val = relu(out_val)
                    output[oh, ow, oc] = out_val

    return conv2d_relu


conv1 = rename(
    _make_conv2d(INPUT_DIM, INPUT_DIM, 1, CONV1_FILTERS, 3, CONV1_DIM, CONV1_DIM),
    "conv1",
)
conv2 = rename(
    _make_conv2d_relu(
        CONV1_DIM, CONV1_DIM, CONV1_FILTERS, CONV2_FILTERS, 3, CONV2_DIM, CONV2_DIM
    ),
    "conv2",
)
conv3 = rename(
    _make_conv2d_relu(
        CONV2_DIM, CONV2_DIM, CONV2_FILTERS, CONV3_FILTERS, 3, CONV3_DIM, CONV3_DIM
    ),
    "conv3",
)
nlb_qkv_conv = rename(
    _make_conv2d(
        CONV1_DIM, CONV1_DIM, CONV1_FILTERS, CONV2_FILTERS, 1, CONV1_DIM, CONV1_DIM
    ),
    "nlb_qkv_conv",
)
nlb_out_conv = rename(
    _make_conv2d(
        CONV1_DIM, CONV1_DIM, CONV2_FILTERS, CONV1_FILTERS, 1, CONV1_DIM, CONV1_DIM
    ),
    "nlb_out_conv",
)


def _make_fc(d0, d1, d2, out_features):
    @proc
    def fc(
        input: i8[d0, d1, d2] @ DRAM,
        weights: i8[out_features, d0, d1, d2] @ DRAM,
        bias: i32[out_features] @ DRAM,
        output: i8[out_features, 1, 1] @ DRAM,
        scale: f32 @ DRAM,
    ):
        for j in seq(0, out_features):
            sum: i32
            sum = bias[j]
            for k0 in seq(0, d0):
                for k1 in seq(0, d1):
                    for k2 in seq(0, d2):
                        x: i32
                        w: i32
                        x = input[k0, k1, k2]
                        w = weights[j, k0, k1, k2]
                        sum += x * w

            tmp: f32
            acc_scale(sum, tmp, scale)
            out_val: i8
            clamp(tmp, out_val)
            output[j, 0, 0] = out_val

    return fc


def _make_fc_relu(d0, d1, d2, out_features):
    @proc
    def fc_relu(
        input: i8[d0, d1, d2] @ DRAM,
        weights: i8[out_features, d0, d1, d2] @ DRAM,
        bias: i32[out_features] @ DRAM,
        output: i8[out_features, 1, 1] @ DRAM,
        scale: f32 @ DRAM,
        relu_scale: f32 @ DRAM,
    ):
        for j in seq(0, out_features):
            sum: i32
            sum = bias[j]
            for k0 in seq(0, d0):
                for k1 in seq(0, d1):
                    for k2 in seq(0, d2):
                        x: i32
                        w: i32
                        x = input[k0, k1, k2]
                        w = weights[j, k0, k1, k2]
                        sum += x * w

            sf: f32
            sf = scale * relu_scale
            tmp: f32
            acc_scale(sum, tmp, sf)
            out_val: i8
            clamp(tmp, out_val)
            out_val = relu(out_val)
            output[j, 0, 0] = out_val

    return fc_relu


fc1 = rename(_make_fc_relu(CONV3_FILTERS, CONV3_DIM, CONV3_DIM, FC1_UNITS), "fc1")
fc2 = rename(_make_fc_relu(FC1_UNITS, 1, 1, FC2_UNITS), "fc2")
fc3 = rename(_make_fc_relu(FC2_UNITS, 1, 1, FC3_UNITS), "fc3")
fc4 = rename(_make_fc_relu(FC3_UNITS, 1, 1, FC4_UNITS), "fc4")
fc_output = rename(_make_fc(FC4_UNITS, 1, 1, OUTPUT_UNITS), "fc_output")


@proc
def matmul_theta_phi(
    A: i8[CONV1_DIM, CONV1_DIM, CONV2_FILTERS] @ DRAM,
    B: i8[CONV1_DIM, CONV1_DIM, CONV2_FILTERS] @ DRAM,
    C: i8[CONV1_DIM, CONV1_DIM, CONV1_DIM, CONV1_DIM] @ DRAM,
    scale: f32 @ DRAM,
):
    for i1 in seq(0, CONV1_DIM):
        for i2 in seq(0, CONV1_DIM):
            for j1 in seq(0, CONV1_DIM):
                for j2 in seq(0, CONV1_DIM):
                    sum: i32
                    sum = 0
                    for k in seq(0, CONV2_FILTERS):
                        a: i32
                        b: i32
                        a = A[i1, i2, k]
                        b = B[j1, j2, k]
                        sum += a * b

                    tmp: f32
                    acc_scale(sum, tmp, scale)
                    out_val: i8
                    clamp(tmp, out_val)
                    C[i1, i2, j1, j2] = out_val


@proc
def matmul_attention_g(
    A: i8[CONV1_DIM, CONV1_DIM, CONV1_DIM, CONV1_DIM] @ DRAM,
    B: i8[CONV1_DIM, CONV1_DIM, CONV2_FILTERS] @ DRAM,
    C: i8[CONV1_DIM, CONV1_DIM, CONV2_FILTERS] @ DRAM,
    scale: f32 @ DRAM,
):
    for i1 in seq(0, CONV1_DIM):
        for i2 in seq(0, CONV1_DIM):
            for j in seq(0, CONV2_FILTERS):
                sum: i32
                sum = 0
                for k1 in seq(0, CONV1_DIM):
                    for k2 in seq(0, CONV1_DIM):
                        a: i32
                        b: i32
                        a = A[i1, i2, k1, k2]
                        b = B[k1, k2, j]
                        sum += a * b

                tmp: f32
                acc_scale(sum, tmp, scale)
                out_val: i8
                clamp(tmp, out_val)
                C[i1, i2, j] = out_val


@proc
def nlb_softmax(
    A: i8[CONV1_DIM, CONV1_DIM, CONV1_DIM, CONV1_DIM] @ DRAM,
    C: i8[CONV1_DIM, CONV1_DIM, CONV1_DIM, CONV1_DIM] @ DRAM,
    in_scale: f32 @ DRAM,
    out_scale: f32 @ DRAM,
):
    for i1 in seq(0, CONV1_DIM):
        for i2 in seq(0, CONV1_DIM):
            buf: f32[CONV1_DIM, CONV1_DIM] @ DRAM_STATIC
            max_q: f32
            max_q = A[i1, i2, 0, 0]
            for j1 in seq(0, CONV1_DIM):
                for j2 in seq(0, CONV1_DIM):
                    v: f32
                    v = A[i1, i2, j1, j2]
                    max_q = select(max_q, v, v, max_q)

            sum_exp: f32
            sum_exp = 0.0
            for j1 in seq(0, CONV1_DIM):
                for j2 in seq(0, CONV1_DIM):
                    av: f32
                    e: f32
                    av = A[i1, i2, j1, j2]
                    e = expf(in_scale * (av - max_q))
                    buf[j1, j2] = e
                    sum_exp += e

            for j1 in seq(0, CONV1_DIM):
                for j2 in seq(0, CONV1_DIM):
                    tmp: f32
                    out_val: i8
                    tmp = buf[j1, j2] / sum_exp / out_scale
                    clamp(tmp, out_val)
                    C[i1, i2, j1, j2] = out_val


@proc
def resadd_relu(
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
                a: i32
                atmp: f32
                a8: i8
                ai: i32
                a = A[i1, i2, j]
                acc_scale(a, atmp, A_scale)
                clamp(atmp, a8)
                ai = a8

                b: i32
                btmp: f32
                b8: i8
                bi: i32
                b = B[i1, i2, j]
                acc_scale(b, btmp, B_scale)
                clamp(btmp, b8)
                bi = b8

                si: i32
                ctmp: f32
                out_val: i8
                si = ai + bi
                acc_scale(si, ctmp, C_scale)
                clamp(ctmp, out_val)
                out_val = relu(out_val)
                C[i1, i2, j] = out_val


@proc
def nlb(
    input: i8[CONV1_DIM, CONV1_DIM, CONV1_FILTERS] @ DRAM,
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
    leaky1_scale: f32 @ DRAM,
    output: i8[CONV1_DIM, CONV1_DIM, CONV1_FILTERS] @ DRAM,
):
    nlb_theta_out: i8[CONV1_DIM, CONV1_DIM, CONV2_FILTERS] @ DRAM_STATIC
    nlb_qkv_conv(
        input, nlb_theta_weights, nlb_theta_bias, nlb_theta_out, nlb_theta_scale
    )

    nlb_phi_out: i8[CONV1_DIM, CONV1_DIM, CONV2_FILTERS] @ DRAM_STATIC
    nlb_qkv_conv(input, nlb_phi_weights, nlb_phi_bias, nlb_phi_out, nlb_phi_scale)

    nlb_g_out: i8[CONV1_DIM, CONV1_DIM, CONV2_FILTERS] @ DRAM_STATIC
    nlb_qkv_conv(input, nlb_g_weights, nlb_g_bias, nlb_g_out, nlb_g_scale)

    attention_raw: i8[CONV1_DIM, CONV1_DIM, CONV1_DIM, CONV1_DIM] @ DRAM_STATIC
    matmul_theta_phi(nlb_theta_out, nlb_phi_out, attention_raw, nlb_matmul_scale)

    attention_out: i8[CONV1_DIM, CONV1_DIM, CONV1_DIM, CONV1_DIM] @ DRAM_STATIC
    nlb_softmax(attention_raw, attention_out, softmax_input_scale, softmax_output_scale)

    attended_output: i8[CONV1_DIM, CONV1_DIM, CONV2_FILTERS] @ DRAM_STATIC
    matmul_attention_g(attention_out, nlb_g_out, attended_output, nlb_matmul_1_scale)

    nlb_output: i8[CONV1_DIM, CONV1_DIM, CONV1_FILTERS] @ DRAM_STATIC
    nlb_out_conv(
        attended_output, nlb_out_weights, nlb_out_bias, nlb_output, nlb_out_scale
    )

    resadd_relu(
        nlb_add_b_scale, nlb_add_a_scale, leaky1_scale, input, nlb_output, output
    )


@proc
def braggnn_inference(
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
    leaky1_scale: f32 @ DRAM,
    conv2_weights: i8[3, 3, CONV1_FILTERS, CONV2_FILTERS] @ DRAM,
    conv2_bias: i32[CONV2_FILTERS] @ DRAM,
    conv2_scale: f32 @ DRAM,
    leaky3_scale: f32 @ DRAM,
    conv3_weights: i8[3, 3, CONV2_FILTERS, CONV3_FILTERS] @ DRAM,
    conv3_bias: i32[CONV3_FILTERS] @ DRAM,
    conv3_scale: f32 @ DRAM,
    leaky5_scale: f32 @ DRAM,
    fc1_weights: i8[FC1_UNITS, CONV3_FILTERS, CONV3_DIM, CONV3_DIM] @ DRAM,
    fc1_bias: i32[FC1_UNITS] @ DRAM,
    fc1_scale: f32 @ DRAM,
    dense1_leaky_scale: f32 @ DRAM,
    fc2_weights: i8[FC2_UNITS, FC1_UNITS, 1, 1] @ DRAM,
    fc2_bias: i32[FC2_UNITS] @ DRAM,
    fc2_scale: f32 @ DRAM,
    dense3_leaky_scale: f32 @ DRAM,
    fc3_weights: i8[FC3_UNITS, FC2_UNITS, 1, 1] @ DRAM,
    fc3_bias: i32[FC3_UNITS] @ DRAM,
    fc3_scale: f32 @ DRAM,
    dense5_leaky_scale: f32 @ DRAM,
    fc4_weights: i8[FC4_UNITS, FC3_UNITS, 1, 1] @ DRAM,
    fc4_bias: i32[FC4_UNITS] @ DRAM,
    fc4_scale: f32 @ DRAM,
    dense7_leaky_scale: f32 @ DRAM,
    output_weights: i8[OUTPUT_UNITS, FC4_UNITS, 1, 1] @ DRAM,
    output_bias: i32[OUTPUT_UNITS] @ DRAM,
    output_scale: f32 @ DRAM,
    output: i8[OUTPUT_UNITS, 1, 1] @ DRAM,
):
    input: i8[INPUT_DIM, INPUT_DIM, 1] @ DRAM_STATIC
    for h in seq(0, INPUT_DIM):
        for w in seq(0, INPUT_DIM):
            q: f32
            q = fp32_input[h, w, 0] * 127.0
            input[h, w, 0] = q

    conv1_out: i8[CONV1_DIM, CONV1_DIM, CONV1_FILTERS] @ DRAM_STATIC
    conv1(input, conv1_weights, conv1_bias, conv1_out, conv1_scale)

    nlb_out: i8[CONV1_DIM, CONV1_DIM, CONV1_FILTERS] @ DRAM_STATIC
    nlb(
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
        leaky1_scale,
        nlb_out,
    )

    conv2_out: i8[CONV2_DIM, CONV2_DIM, CONV2_FILTERS] @ DRAM_STATIC
    conv2(nlb_out, conv2_weights, conv2_bias, conv2_out, conv2_scale, leaky3_scale)

    conv3_out: i8[CONV3_DIM, CONV3_DIM, CONV3_FILTERS] @ DRAM_STATIC
    conv3(conv2_out, conv3_weights, conv3_bias, conv3_out, conv3_scale, leaky5_scale)

    flattened: i8[CONV3_FILTERS, CONV3_DIM, CONV3_DIM] @ DRAM_STATIC
    for ch in seq(0, CONV3_FILTERS):
        for r in seq(0, CONV3_DIM):
            for c in seq(0, CONV3_DIM):
                flattened[ch, r, c] = conv3_out[r, c, ch]

    fc1_out: i8[FC1_UNITS, 1, 1] @ DRAM_STATIC
    fc1(flattened, fc1_weights, fc1_bias, fc1_out, fc1_scale, dense1_leaky_scale)

    fc2_out: i8[FC2_UNITS, 1, 1] @ DRAM_STATIC
    fc2(fc1_out, fc2_weights, fc2_bias, fc2_out, fc2_scale, dense3_leaky_scale)

    fc3_out: i8[FC3_UNITS, 1, 1] @ DRAM_STATIC
    fc3(fc2_out, fc3_weights, fc3_bias, fc3_out, fc3_scale, dense5_leaky_scale)

    fc4_out: i8[FC4_UNITS, 1, 1] @ DRAM_STATIC
    fc4(fc3_out, fc4_weights, fc4_bias, fc4_out, fc4_scale, dense7_leaky_scale)

    fc_output(fc4_out, output_weights, output_bias, output, output_scale)


def _swap_all(p, old, new):
    for _ in range(len(p.find_all(f"{old}(_)"))):
        p = call_eqv(p, f"{old}(_)", new)
    return p


def _conv(p, do, name):
    return rename(replace(p, p.find_loop("oh"), do), name)


_do_conv1 = make_loop_conv_ws("do_conv1", INPUT_DIM, 1, CONV1_FILTERS, 3)
_do_conv2 = make_loop_conv_ws(
    "do_conv2", CONV1_DIM, CONV1_FILTERS, CONV2_FILTERS, 3, relu_act=True
)
_do_conv3 = make_loop_conv_ws(
    "do_conv3", CONV2_DIM, CONV2_FILTERS, CONV3_FILTERS, 3, relu_act=True
)
_do_qkv_conv = make_loop_conv_ws(
    "do_qkv_conv", CONV1_DIM, CONV1_FILTERS, CONV2_FILTERS, 1
)
_do_out_conv = make_loop_conv_ws(
    "do_out_conv", CONV1_DIM, CONV2_FILTERS, CONV1_FILTERS, 1
)

conv1_gemm = _conv(conv1, _do_conv1, "conv1_gemm")
conv2_gemm = _conv(conv2, _do_conv2, "conv2_gemm")
conv3_gemm = _conv(conv3, _do_conv3, "conv3_gemm")
nlb_qkv_conv_gemm = _conv(nlb_qkv_conv, _do_qkv_conv, "nlb_qkv_conv_gemm")
nlb_out_conv_gemm = _conv(nlb_out_conv, _do_out_conv, "nlb_out_conv_gemm")


def _fc(p, do, name):
    return rename(replace(p, p.find_loop("j"), do), name)


_do_fc1 = make_loop_matmul_fc(
    "do_fc1", CONV3_FILTERS, CONV3_DIM, CONV3_DIM, FC1_UNITS, relu_act=True
)
_do_fc2 = make_loop_matmul_fc("do_fc2", FC1_UNITS, 1, 1, FC2_UNITS, relu_act=True)
_do_fc3 = make_loop_matmul_fc("do_fc3", FC2_UNITS, 1, 1, FC3_UNITS, relu_act=True)
_do_fc4 = make_loop_matmul_fc("do_fc4", FC3_UNITS, 1, 1, FC4_UNITS, relu_act=True)
_do_fc_output = make_loop_matmul_fc("do_fc_output", FC4_UNITS, 1, 1, OUTPUT_UNITS)

fc1_gemm = _fc(fc1, _do_fc1, "fc1_gemm")
fc2_gemm = _fc(fc2, _do_fc2, "fc2_gemm")
fc3_gemm = _fc(fc3, _do_fc3, "fc3_gemm")
fc4_gemm = _fc(fc4, _do_fc4, "fc4_gemm")
fc_output_gemm = _fc(fc_output, _do_fc_output, "fc_output_gemm")


_do_theta_phi = make_loop_matmul_theta_phi_tile(
    "do_theta_phi", CONV2_FILTERS, CONV1_DIM
)


def _theta_phi(p):
    p = divide_loop(p, p.find_loop("i1"), 8, ["i1o", "i1i"], tail="cut")
    p = simplify(p)
    p = replace(p, p.find_loop("i1o").body(), _do_theta_phi)
    p = replace(p, p.find_loop("i1o").next(), _do_theta_phi)
    p = insert_noop_call(p, p.find_loop("i1o").next().after(), fence, [])
    return rename(p, "matmul_theta_phi_gemm")


matmul_theta_phi_gemm = _theta_phi(matmul_theta_phi)


_do_softmax = make_loop_softmax_tile("do_softmax", CONV1_DIM)


def _softmax(p):
    p = divide_loop(p, p.find_loop("i1"), 8, ["i1o", "i1i"], tail="cut")
    p = simplify(p)
    p = replace(p, p.find_loop("i1o").body(), _do_softmax)
    p = replace(p, p.find_loop("i1o").next(), _do_softmax)
    p = insert_noop_call(p, p.find_loop("i1o").next().after(), fence, [])
    return rename(p, "nlb_softmax_gemm")


nlb_softmax_gemm = _softmax(nlb_softmax)


_do_attention_g = make_loop_matmul_attention_g(
    "do_attention_g", CONV2_FILTERS, CONV1_DIM
)


def _attention_g(p):
    p = replace(p, p.find_loop("i1"), _do_attention_g)
    return rename(p, "matmul_attention_g_gemm")


matmul_attention_g_gemm = _attention_g(matmul_attention_g)


_do_resadd = make_loop_resadd_relu("do_resadd", CONV1_DIM, CONV1_FILTERS)


def _resadd(p):
    p = replace(p, p.find_loop("i1"), _do_resadd)
    return rename(p, "resadd_relu_gemm")


resadd_relu_gemm = _resadd(resadd_relu)


def _nlb_gemm():
    p = nlb
    for old, new in (
        ("nlb_qkv_conv", nlb_qkv_conv_gemm),
        ("matmul_theta_phi", matmul_theta_phi_gemm),
        ("nlb_softmax", nlb_softmax_gemm),
        ("matmul_attention_g", matmul_attention_g_gemm),
        ("nlb_out_conv", nlb_out_conv_gemm),
        ("resadd_relu", resadd_relu_gemm),
    ):
        p = _swap_all(p, old, new)
    return rename(p, "nlb_gemm")


nlb_gemm = _nlb_gemm()


def _inference_gemm():
    p = call_eqv(braggnn_inference, "nlb(_)", nlb_gemm)
    for old, new in (
        ("conv1", conv1_gemm),
        ("conv2", conv2_gemm),
        ("conv3", conv3_gemm),
        ("fc1", fc1_gemm),
        ("fc2", fc2_gemm),
        ("fc3", fc3_gemm),
        ("fc4", fc4_gemm),
        ("fc_output", fc_output_gemm),
    ):
        p = _swap_all(p, old, new)
    return rename(p, "braggnn_inference")


braggnn_inference = _inference_gemm()
