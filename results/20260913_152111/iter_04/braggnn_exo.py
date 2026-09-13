# ruff: noqa: F821

from __future__ import annotations

from exo.API_scheduling import rename
from exo.libs.externs import fmaxf, relu, select
from exo.libs.memories import DRAM_STATIC

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


def _make_leaky_relu_requant(d0, d1, d2):
    @proc
    def leaky_relu_requant(values: i8[d0, d1, d2] @ DRAM, scale: f32 @ DRAM):
        for i in seq(0, d0):
            for j in seq(0, d1):
                for k in seq(0, d2):
                    x: f32
                    val: f32
                    x = values[i, j, k]
                    val = relu(x) * scale
                    val = val + select(val, 0.0, -0.5, 0.5)
                    val = fmaxf(-128.0, select(val, 127.0, val, 127.0))
                    values[i, j, k] = val

    return leaky_relu_requant


leaky1 = rename(_make_leaky_relu_requant(CONV1_DIM, CONV1_DIM, CONV1_FILTERS), "leaky1")
leaky3 = rename(_make_leaky_relu_requant(CONV2_DIM, CONV2_DIM, CONV2_FILTERS), "leaky3")
leaky5 = rename(_make_leaky_relu_requant(CONV3_FILTERS, CONV3_DIM, CONV3_DIM), "leaky5")
dense1_leaky = rename(_make_leaky_relu_requant(FC1_UNITS, 1, 1), "dense1_leaky")
dense3_leaky = rename(_make_leaky_relu_requant(FC2_UNITS, 1, 1), "dense3_leaky")
dense5_leaky = rename(_make_leaky_relu_requant(FC3_UNITS, 1, 1), "dense5_leaky")
dense7_leaky = rename(_make_leaky_relu_requant(FC4_UNITS, 1, 1), "dense7_leaky")


def _make_conv2d(in_h, in_w, in_ch, out_ch, kernel_size, out_h, out_w):
    assert in_h >= out_h + kernel_size - 1
    assert in_w >= out_w + kernel_size - 1

    @proc
    def conv2d(
        input: i8[in_h, in_w, in_ch] @ DRAM,
        weights: i8[out_ch, kernel_size, kernel_size, in_ch] @ DRAM,
        bias: i32[out_ch] @ DRAM,
        output: i8[out_h, out_w, out_ch] @ DRAM,
        acc_scale: f32 @ DRAM,
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
                                w = weights[oc, kh, kw, ic]
                                sum += x * w

                    val: f32
                    val = sum
                    val = val * acc_scale
                    val = val + select(val, 0.0, -0.5, 0.5)
                    val = fmaxf(-128.0, select(val, 127.0, val, 127.0))
                    output[oh, ow, oc] = val

    return conv2d


conv1 = rename(
    _make_conv2d(INPUT_DIM, INPUT_DIM, 1, CONV1_FILTERS, 3, CONV1_DIM, CONV1_DIM),
    "conv1",
)
conv2 = rename(
    _make_conv2d(
        CONV1_DIM, CONV1_DIM, CONV1_FILTERS, CONV2_FILTERS, 3, CONV2_DIM, CONV2_DIM
    ),
    "conv2",
)
conv3 = rename(
    _make_conv2d(
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
        acc_scale: f32 @ DRAM,
    ):
        # in_features = d0 * d1 * d2
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
            val: f32
            val = sum
            val = val * acc_scale
            val = val + select(val, 0.0, -0.5, 0.5)
            val = fmaxf(-128.0, select(val, 127.0, val, 127.0))
            output[j, 0, 0] = val

    return fc


fc1 = rename(
    _make_fc(CONV3_FILTERS, CONV3_DIM, CONV3_DIM, FC1_UNITS),
    "fc1",
)
fc2 = rename(_make_fc(FC1_UNITS, 1, 1, FC2_UNITS), "fc2")
fc3 = rename(_make_fc(FC2_UNITS, 1, 1, FC3_UNITS), "fc3")
fc4 = rename(_make_fc(FC3_UNITS, 1, 1, FC4_UNITS), "fc4")
fc_output = rename(_make_fc(FC4_UNITS, 1, 1, OUTPUT_UNITS), "fc_output")


@proc
def matmul_transA(
    A: i8[CONV2_FILTERS, CONV1_DIM, CONV1_DIM] @ DRAM,
    B: i8[CONV2_FILTERS, CONV1_DIM, CONV1_DIM] @ DRAM,
    C: i8[CONV1_DIM, CONV1_DIM, CONV1_DIM, CONV1_DIM] @ DRAM,
    acc_scale: f32 @ DRAM,
):
    # M = N = CONV1_DIM * CONV1_DIM
    for i1 in seq(0, CONV1_DIM):
        for i2 in seq(0, CONV1_DIM):
            for j1 in seq(0, CONV1_DIM):
                for j2 in seq(0, CONV1_DIM):
                    sum: i32
                    sum = 0.0
                    for k in seq(0, CONV2_FILTERS):
                        a: i32
                        b: i32
                        a = A[k, i1, i2]
                        b = B[k, j1, j2]
                        sum += a * b
                    val: f32
                    val = sum
                    val = val * acc_scale
                    val = val + select(val, 0.0, -0.5, 0.5)
                    val = fmaxf(-128.0, select(val, 127.0, val, 127.0))
                    C[i1, i2, j1, j2] = val


@proc
def matmul_transB(
    A: i8[CONV1_DIM, CONV1_DIM, CONV1_DIM, CONV1_DIM] @ DRAM,
    B: i8[CONV2_FILTERS, CONV1_DIM, CONV1_DIM] @ DRAM,
    C: i8[CONV1_DIM, CONV1_DIM, CONV2_FILTERS] @ DRAM,
    acc_scale: f32 @ DRAM,
):
    # M = K = CONV1_DIM * CONV1_DIM
    for i1 in seq(0, CONV1_DIM):
        for i2 in seq(0, CONV1_DIM):
            for j in seq(0, CONV2_FILTERS):
                sum: i32
                sum = 0.0
                for k1 in seq(0, CONV1_DIM):
                    for k2 in seq(0, CONV1_DIM):
                        a: i32
                        b: i32
                        a = A[i1, i2, k1, k2]
                        b = B[j, k1, k2]
                        sum += a * b
                val: f32
                val = sum
                val = val * acc_scale
                val = val + select(val, 0.0, -0.5, 0.5)
                val = fmaxf(-128.0, select(val, 127.0, val, 127.0))
                C[i1, i2, j] = val


@proc
def resadd(
    A_scale: f32 @ DRAM,
    B_scale: f32 @ DRAM,
    A: i8[CONV1_DIM, CONV1_DIM, CONV1_FILTERS] @ DRAM,
    B: i8[CONV1_DIM, CONV1_DIM, CONV1_FILTERS] @ DRAM,
    C: i8[CONV1_DIM, CONV1_DIM, CONV1_FILTERS] @ DRAM,
):
    # rows = CONV1_DIM * CONV1_DIM
    for i1 in seq(0, CONV1_DIM):
        for i2 in seq(0, CONV1_DIM):
            for j in seq(0, CONV1_FILTERS):
                a: f32
                b: f32
                val: f32
                a = A[i1, i2, j]
                b = B[i1, i2, j]
                val = a * A_scale + b * B_scale
                val = val + select(val, 0.0, -0.5, 0.5)
                val = fmaxf(-128.0, select(val, 127.0, val, 127.0))
                C[i1, i2, j] = val


@proc
def nlb(
    input: i8[CONV1_DIM, CONV1_DIM, CONV1_FILTERS] @ DRAM,
    nlb_theta_weights: i8[CONV2_FILTERS, 1, 1, CONV1_FILTERS] @ DRAM,
    nlb_theta_bias: i32[CONV2_FILTERS] @ DRAM,
    nlb_theta_scale: f32 @ DRAM,
    nlb_phi_weights: i8[CONV2_FILTERS, 1, 1, CONV1_FILTERS] @ DRAM,
    nlb_phi_bias: i32[CONV2_FILTERS] @ DRAM,
    nlb_phi_scale: f32 @ DRAM,
    nlb_g_weights: i8[CONV2_FILTERS, 1, 1, CONV1_FILTERS] @ DRAM,
    nlb_g_bias: i32[CONV2_FILTERS] @ DRAM,
    nlb_g_scale: f32 @ DRAM,
    nlb_matmul_scale: f32 @ DRAM,
    softmax_input_scale: f32 @ DRAM,
    softmax_output_scale: f32 @ DRAM,
    nlb_matmul_1_scale: f32 @ DRAM,
    nlb_out_weights: i8[CONV1_FILTERS, 1, 1, CONV2_FILTERS] @ DRAM,
    nlb_out_bias: i32[CONV1_FILTERS] @ DRAM,
    nlb_out_scale: f32 @ DRAM,
    nlb_add_a_scale: f32 @ DRAM,
    nlb_add_b_scale: f32 @ DRAM,
    output: i8[CONV1_DIM, CONV1_DIM, CONV1_FILTERS] @ DRAM,
):
    # --- Theta 1x1 conv: 9x9x64 -> 9x9x32 ---
    theta_out: i8[CONV1_DIM, CONV1_DIM, CONV2_FILTERS] @ DRAM_STATIC
    nlb_qkv_conv(input, nlb_theta_weights, nlb_theta_bias, theta_out, nlb_theta_scale)

    # Reshape NHWC [9x9x32] -> [32][81]
    theta_reshaped: i8[CONV2_FILTERS, CONV1_DIM, CONV1_DIM] @ DRAM_STATIC
    for c in seq(0, CONV2_FILTERS):
        for h in seq(0, CONV1_DIM):
            for w in seq(0, CONV1_DIM):
                theta_reshaped[c, h, w] = theta_out[h, w, c]

    # --- Phi 1x1 conv: 9x9x64 -> 9x9x32 ---
    phi_out: i8[CONV1_DIM, CONV1_DIM, CONV2_FILTERS] @ DRAM_STATIC
    nlb_qkv_conv(input, nlb_phi_weights, nlb_phi_bias, phi_out, nlb_phi_scale)

    phi_reshaped: i8[CONV2_FILTERS, CONV1_DIM, CONV1_DIM] @ DRAM_STATIC
    for c in seq(0, CONV2_FILTERS):
        for h in seq(0, CONV1_DIM):
            for w in seq(0, CONV1_DIM):
                phi_reshaped[c, h, w] = phi_out[h, w, c]

    # --- G 1x1 conv: 9x9x64 -> 9x9x32 ---
    g_out: i8[CONV1_DIM, CONV1_DIM, CONV2_FILTERS] @ DRAM_STATIC
    nlb_qkv_conv(input, nlb_g_weights, nlb_g_bias, g_out, nlb_g_scale)

    g_reshaped: i8[CONV2_FILTERS, CONV1_DIM, CONV1_DIM] @ DRAM_STATIC
    for c in seq(0, CONV2_FILTERS):
        for h in seq(0, CONV1_DIM):
            for w in seq(0, CONV1_DIM):
                g_reshaped[c, h, w] = g_out[h, w, c]

    # --- Attention: theta^T @ phi -> [81][81] ---
    # theta_reshaped[32][81], phi_reshaped[32][81]
    # C[81][81] = theta^T[81][32] @ phi[32][81]
    attention: i8[CONV1_DIM, CONV1_DIM, CONV1_DIM, CONV1_DIM] @ DRAM_STATIC
    matmul_transA(theta_reshaped, phi_reshaped, attention, nlb_matmul_scale)

    # --- Softmax (per row, matching Gemmini Taylor approximation) ---
    for i1 in seq(0, CONV1_DIM):
        for i2 in seq(0, CONV1_DIM):
            row_float: f32[CONV1_DIM, CONV1_DIM] @ DRAM_STATIC
            max_val: f32 @ DRAM
            max_val = -1000000000.0
            for j1 in seq(0, CONV1_DIM):
                for j2 in seq(0, CONV1_DIM):
                    row_float[j1, j2] = attention[i1, i2, j1, j2]
                    row_float[j1, j2] = row_float[j1, j2] * softmax_input_scale
                    max_val = fmaxf(max_val, row_float[j1, j2])

            sum_exp: f32 @ DRAM
            sum_exp = 0.0
            for j1 in seq(0, CONV1_DIM):
                for j2 in seq(0, CONV1_DIM):
                    x: f32
                    x2: f32
                    x3: f32
                    x4: f32
                    exp_val: f32
                    x = row_float[j1, j2] - max_val
                    x2 = x * x
                    x3 = x2 * x
                    x4 = x2 * x2
                    exp_val = 1.0 + x
                    exp_val += x2 * 0.5
                    exp_val += x3 * 0.166667
                    exp_val += x4 * 0.041667
                    exp_val = select(exp_val, 0.0, 0.0001, exp_val)
                    exp_val = select(-8.0, x, exp_val, 0.0001)
                    row_float[j1, j2] = exp_val
                    sum_exp += exp_val

            for j1 in seq(0, CONV1_DIM):
                for j2 in seq(0, CONV1_DIM):
                    softmax_val: f32
                    quantized: f32
                    softmax_val = row_float[j1, j2] / sum_exp
                    quantized = softmax_val / softmax_output_scale + 0.5
                    quantized = fmaxf(
                        -128.0, select(quantized, 127.0, quantized, 127.0)
                    )
                    attention[i1, i2, j1, j2] = quantized

    # --- Attended output: attention @ g^T -> [81][32] ---
    # attention[81][81], g_reshaped[32][81]
    # C[81][32] = attention[81][81] @ g^T[81][32]
    attended: i8[CONV1_DIM, CONV1_DIM, CONV2_FILTERS] @ DRAM_STATIC
    matmul_transB(attention, g_reshaped, attended, nlb_matmul_1_scale)

    # Reshape [81][32] -> NHWC [9][9][32]
    tpg_output: i8[CONV1_DIM, CONV1_DIM, CONV2_FILTERS] @ DRAM_STATIC
    for c in seq(0, CONV2_FILTERS):
        for h in seq(0, CONV1_DIM):
            for w in seq(0, CONV1_DIM):
                tpg_output[h, w, c] = attended[h, w, c]

    # --- out_cnn 1x1 conv: 9x9x32 -> 9x9x64 ---
    nlb_conv_out: i8[CONV1_DIM, CONV1_DIM, CONV1_FILTERS] @ DRAM_STATIC
    nlb_out_conv(tpg_output, nlb_out_weights, nlb_out_bias, nlb_conv_out, nlb_out_scale)

    # --- Residual add: output = input * B_scale + nlb_conv_out * A_scale ---
    resadd(nlb_add_b_scale, nlb_add_a_scale, input, nlb_conv_out, output)


@proc
def braggnn_inference(
    fp32_input: f32[INPUT_DIM, INPUT_DIM, 1] @ DRAM,
    conv1_weights: i8[CONV1_FILTERS, 3, 3, 1] @ DRAM,
    conv1_bias: i32[CONV1_FILTERS] @ DRAM,
    conv1_scale: f32 @ DRAM,
    nlb_theta_weights: i8[CONV2_FILTERS, 1, 1, CONV1_FILTERS] @ DRAM,
    nlb_theta_bias: i32[CONV2_FILTERS] @ DRAM,
    nlb_theta_scale: f32 @ DRAM,
    nlb_phi_weights: i8[CONV2_FILTERS, 1, 1, CONV1_FILTERS] @ DRAM,
    nlb_phi_bias: i32[CONV2_FILTERS] @ DRAM,
    nlb_phi_scale: f32 @ DRAM,
    nlb_g_weights: i8[CONV2_FILTERS, 1, 1, CONV1_FILTERS] @ DRAM,
    nlb_g_bias: i32[CONV2_FILTERS] @ DRAM,
    nlb_g_scale: f32 @ DRAM,
    nlb_matmul_scale: f32 @ DRAM,
    softmax_input_scale: f32 @ DRAM,
    softmax_output_scale: f32 @ DRAM,
    nlb_matmul_1_scale: f32 @ DRAM,
    nlb_out_weights: i8[CONV1_FILTERS, 1, 1, CONV2_FILTERS] @ DRAM,
    nlb_out_bias: i32[CONV1_FILTERS] @ DRAM,
    nlb_out_scale: f32 @ DRAM,
    nlb_add_a_scale: f32 @ DRAM,
    nlb_add_b_scale: f32 @ DRAM,
    leaky1_scale: f32 @ DRAM,
    conv2_weights: i8[CONV2_FILTERS, 3, 3, CONV1_FILTERS] @ DRAM,
    conv2_bias: i32[CONV2_FILTERS] @ DRAM,
    conv2_scale: f32 @ DRAM,
    leaky3_scale: f32 @ DRAM,
    conv3_weights: i8[CONV3_FILTERS, 3, 3, CONV2_FILTERS] @ DRAM,
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
    # QuantizeLinear: y_scale = 0.007874015718698502 -> 1/0.007874 = 127
    input: i8[INPUT_DIM, INPUT_DIM, 1] @ DRAM_STATIC
    for h in seq(0, INPUT_DIM):
        for w in seq(0, INPUT_DIM):
            q: f32
            q = fp32_input[h, w, 0] * 127.0
            input[h, w, 0] = q

    # Conv1: 11x11x1 -> 9x9x64
    conv1_out: i8[CONV1_DIM, CONV1_DIM, CONV1_FILTERS] @ DRAM_STATIC
    conv1(input, conv1_weights, conv1_bias, conv1_out, conv1_scale)

    # Non-Local Block: 9x9x64 -> 9x9x64
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
        nlb_out,
    )

    # LeakyReLU + requant (NLB add output scale -> Conv2 input scale)
    leaky1(nlb_out, leaky1_scale)

    # Conv2: 9x9x64 -> 7x7x32
    conv2_out: i8[CONV2_DIM, CONV2_DIM, CONV2_FILTERS] @ DRAM_STATIC
    conv2(nlb_out, conv2_weights, conv2_bias, conv2_out, conv2_scale)

    # LeakyReLU + requant (Conv2 output scale -> Conv3 input scale)
    leaky3(conv2_out, leaky3_scale)

    # Conv3: 7x7x32 -> 5x5x8
    conv3_out: i8[CONV3_DIM, CONV3_DIM, CONV3_FILTERS] @ DRAM_STATIC
    conv3(conv2_out, conv3_weights, conv3_bias, conv3_out, conv3_scale)

    # Flatten: NHWC [5][5][8] -> NCHW order [8][5][5] = 200 elements
    flattened: i8[CONV3_FILTERS, CONV3_DIM, CONV3_DIM] @ DRAM_STATIC
    for ch in seq(0, CONV3_FILTERS):
        for r in seq(0, CONV3_DIM):
            for c in seq(0, CONV3_DIM):
                flattened[ch, r, c] = conv3_out[r, c, ch]

    # LeakyReLU + requant (Conv3 output scale -> FC1 input scale)
    leaky5(flattened, leaky5_scale)

    # FC1: 200 -> 16
    fc1_out: i8[FC1_UNITS, 1, 1] @ DRAM_STATIC
    fc1(flattened, fc1_weights, fc1_bias, fc1_out, fc1_scale)

    dense1_leaky(fc1_out, dense1_leaky_scale)

    # FC2: 16 -> 8
    fc2_out: i8[FC2_UNITS, 1, 1] @ DRAM_STATIC
    fc2(fc1_out, fc2_weights, fc2_bias, fc2_out, fc2_scale)

    dense3_leaky(fc2_out, dense3_leaky_scale)

    # FC3: 8 -> 4
    fc3_out: i8[FC3_UNITS, 1, 1] @ DRAM_STATIC
    fc3(fc2_out, fc3_weights, fc3_bias, fc3_out, fc3_scale)

    dense5_leaky(fc3_out, dense5_leaky_scale)

    # FC4: 4 -> 2
    fc4_out: i8[FC4_UNITS, 1, 1] @ DRAM_STATIC
    fc4(fc3_out, fc4_weights, fc4_bias, fc4_out, fc4_scale)

    dense7_leaky(fc4_out, dense7_leaky_scale)

    # Output: 2 -> 2
    fc_output(fc4_out, output_weights, output_bias, output, output_scale)


# --------------------------------------------------------------------------- #
# Scheduling BraggNN for RISC-V Rocket + Gemmini accelerator
# --------------------------------------------------------------------------- #

from exo.API_scheduling import (
    call_eqv,
    divide_loop,
    expand_dim,
    fission,
    lift_alloc,
    rearrange_dim,
    reorder_loops,
    replace,
    resize_dim,
    set_memory,
    simplify,
    stage_mem,
    unroll_loop,
)
from exo.libs.memories import DRAM_STATIC, GEMM_ACCUM, GEMM_SCRATCH
from exo.platforms.gemmini import (
    ld_acc_i32,
    ld_i8_id1,
    ld_i8_id2,
    matmul_acc_i8,
    st_acc_i32,
    zero_acc_i32,
)


def _nest(p, pattern, depth):
    """Cursor to the loop depth levels above the statement matching pattern."""
    c = p.find(pattern)
    for _ in range(depth):
        c = c.parent()
    return c


def _lift_all(proc, buf_name):
    """Lift buffer allocation as far up the proc hierarchy as possible."""
    while True:
        try:
            proc = lift_alloc(proc, f"{buf_name} : _")
        except Exception:
            break
    return proc


# --------------------------------------------------------------------------- #
# 1. Gemmini Matmuls: NLB Attention Matmuls (matmul_transA and matmul_transB)
# --------------------------------------------------------------------------- #


def _schedule_matmul_transA_gemmini(p):
    # Pre-transpose A[0:32, 0:9, 0:9] to At_all[9, 9, 32] contiguously
    p = stage_mem(p, p.find_loop("i1"), "A[0:32, 0:9, 0:9]", "At_all")
    p = rearrange_dim(p, "At_all : _", [1, 2, 0])
    p = simplify(p)
    p = reorder_loops(p, "i0 i1")
    p = reorder_loops(p, "i0 i2")
    p = divide_loop(p, "i0", 8, ["i0o", "i0i"], perfect=True)
    p = unroll_loop(p, "i0i")
    p = simplify(p)

    p = reorder_loops(p, "i2 j1")
    p = divide_loop(p, "k", 16, ["ko", "ki"], perfect=True)

    p = expand_dim(p, "sum : _", 16, "j2")
    p = lift_alloc(p, "sum : _")
    p = expand_dim(p, "sum : _", CONV1_DIM, "i2")
    p = lift_alloc(p, "sum : _")
    p = fission(p, p.find("sum[_] = 0.0").after(), n_lifts=2)
    p = fission(p, p.find_loop("ko").after(), n_lifts=2)
    p = reorder_loops(p, "j2 ko")
    p = reorder_loops(p, "i2 ko")
    p = simplify(p)

    p = stage_mem(
        p, p.find_loop("ko").body()[0], "B[16 * ko:16 * ko + 16, j1, 0:9]", "bs"
    )
    p = resize_dim(p, "bs : _", 1, 16, 0)
    p = stage_mem(p, _nest(p, "sum[_] += _", 3), "At_all[i1, 0:9, 16 * ko:16 * ko + 16]", "as_")
    p = stage_mem(p, _nest(p, "C[_] = _", 2), "sum[0:9, 0:9]", "sum_out")
    p = simplify(p)

    for buf in ["as_", "bs", "At_all", "sum", "sum_out"]:
        p = _lift_all(p, buf)

    p = set_memory(p, "At_all : _", DRAM_STATIC)
    p = set_memory(p, "sum_out : _", DRAM_STATIC)
    p = set_memory(p, "sum : _", GEMM_ACCUM)
    p = set_memory(p, "as_ : _", GEMM_SCRATCH)
    p = set_memory(p, "bs : _", GEMM_SCRATCH)

    p = replace(p, _nest(p, "sum[_] = 0.0", 2), zero_acc_i32)
    p = replace(p, _nest(p, "bs[_] = B[_]", 2), ld_i8_id2)
    p = replace(p, _nest(p, "as_[_] = At_all[_]", 2), ld_i8_id1)
    p = replace(p, _nest(p, "sum[_] += _", 3), matmul_acc_i8)
    p = replace(p, _nest(p, "sum_out[_] = sum[_]", 2), st_acc_i32)
    p = simplify(p)
    p = unroll_loop(p, "j2")
    return simplify(p)


def _schedule_matmul_transB_gemmini(p):
    # Pre-transpose B[0:32, 0:9, 0:9] to Bt_all[9, 9, 32] contiguously
    p = stage_mem(p, p.find_loop("i1"), "B[0:32, 0:9, 0:9]", "Bt_all")
    p = rearrange_dim(p, "Bt_all : _", [1, 2, 0])
    p = simplify(p)
    p = reorder_loops(p, "i0 i1")
    p = reorder_loops(p, "i0 i2")
    p = divide_loop(p, "i0", 8, ["i0o", "i0i"], perfect=True)
    p = unroll_loop(p, "i0i")
    p = simplify(p)

    p = divide_loop(p, "j", 16, ["jo", "ji"], perfect=True)
    p = reorder_loops(p, "i2 jo")

    p = expand_dim(p, "sum : _", 16, "ji")
    p = lift_alloc(p, "sum : _")
    p = expand_dim(p, "sum : _", 9, "i2")
    p = lift_alloc(p, "sum : _")
    p = fission(p, p.find("sum[_] = 0.0").after(), n_lifts=2)
    p = fission(p, p.find_loop("k1").after(), n_lifts=2)
    p = reorder_loops(p, "ji k1")
    p = reorder_loops(p, "i2 k1")
    p = simplify(p)

    p = stage_mem(p, _nest(p, "sum[_] += _", 3), "A[i1, 0:9, k1, 0:9]", "as_")
    p = resize_dim(p, "as_ : _", 1, 16, 0)
    p = stage_mem(p, _nest(p, "sum[_] += _", 3), "Bt_all[k1, 0:9, 16 * jo:16 * jo + 16]", "bs")
    p = stage_mem(p, _nest(p, "C[_] = _", 2), "sum[0:9, 0:16]", "sum_out")
    p = simplify(p)

    for buf in ["as_", "bs", "Bt_all", "sum", "sum_out"]:
        p = _lift_all(p, buf)

    p = set_memory(p, "Bt_all : _", DRAM_STATIC)
    p = set_memory(p, "sum_out : _", DRAM_STATIC)
    p = set_memory(p, "sum : _", GEMM_ACCUM)
    p = set_memory(p, "as_ : _", GEMM_SCRATCH)
    p = set_memory(p, "bs : _", GEMM_SCRATCH)

    p = replace(p, _nest(p, "sum[_] = 0.0", 2), zero_acc_i32)
    p = replace(p, _nest(p, "as_[_] = A[_]", 2), ld_i8_id1)
    p = replace(p, _nest(p, "bs[_] = Bt_all[_]", 2), ld_i8_id2)
    p = replace(p, _nest(p, "sum[_] += _", 3), matmul_acc_i8)
    p = replace(p, _nest(p, "sum_out[_] = sum[_]", 2), st_acc_i32)
    p = simplify(p)

    p = divide_loop(p, "ji", 8, ["jio", "jii"], perfect=True)
    p = unroll_loop(p, "jii")
    return simplify(p)


# --------------------------------------------------------------------------- #
# 2. Gemmini Convolutions: conv2, conv3, nlb_qkv_conv, nlb_out_conv
# --------------------------------------------------------------------------- #


def _schedule_conv2_gemmini(p):
    # Pre-transpose weights to [3, 3, 64, 32] contiguously
    p = stage_mem(p, p.find_loop("oh"), "weights[0:32, 0:3, 0:3, 0:64]", "weights_copy")
    p = rearrange_dim(p, "weights_copy : _", [1, 2, 3, 0])
    p = simplify(p)
    p = reorder_loops(p, "i0 i1")
    p = reorder_loops(p, "i0 i2")
    p = reorder_loops(p, "i0 i3")
    p = divide_loop(p, "i0", 16, ["i0o", "i0i"], perfect=True)
    p = unroll_loop(p, "i0i")
    p = unroll_loop(p, "i1")
    for _ in range(3):
        p = unroll_loop(p, "i2")
    p = simplify(p)

    p = divide_loop(p, "oc", 16, ["oco", "oci"], perfect=True)
    p = reorder_loops(p, "ow oco")

    p = expand_dim(p, "sum : _", 16, "oci")
    p = lift_alloc(p, "sum : _")
    p = expand_dim(p, "sum : _", 7, "ow")
    p = lift_alloc(p, "sum : _")
    p = fission(p, p.find("sum[_] = bias[_]").after(), n_lifts=2)
    p = fission(p, p.find_loop("kh").after(), n_lifts=2)
    p = reorder_loops(p, "oci kh")
    p = reorder_loops(p, "ow kh")
    p = simplify(p)

    p = stage_mem(p, p.find_loop("kh"), "sum[0:7, 0:16]", "acc")
    p = simplify(p)
    p = reorder_loops(p, "oci kw")
    p = reorder_loops(p, "ow kw")
    p = divide_loop(p, "ic", 16, ["ico", "ici"], perfect=True)
    p = reorder_loops(p, "oci ico")
    p = reorder_loops(p, "ow ico")
    p = simplify(p)

    p = stage_mem(
        p,
        _nest(p, "acc[_] += _", 3),
        "input[kh + oh, kw:kw + 7, 16 * ico:16 * ico + 16]",
        "as_",
    )
    p = stage_mem(
        p,
        _nest(p, "acc[_] += _", 3),
        "weights_copy[kh, kw, 16 * ico:16 * ico + 16, 16 * oco:16 * oco + 16]",
        "bs",
    )
    p = simplify(p)

    for buf in ["as_", "bs", "weights_copy", "acc", "sum"]:
        p = _lift_all(p, buf)

    p = set_memory(p, "sum : _", DRAM_STATIC)
    p = set_memory(p, "weights_copy : _", DRAM_STATIC)
    p = set_memory(p, "acc : _", GEMM_ACCUM)
    p = set_memory(p, "as_ : _", GEMM_SCRATCH)
    p = set_memory(p, "bs : _", GEMM_SCRATCH)

    p = replace(p, _nest(p, "acc[_] = sum[_]", 2), ld_acc_i32)
    p = replace(p, _nest(p, "as_[_] = input[_]", 2), ld_i8_id1)
    p = replace(p, _nest(p, "bs[_] = weights_copy[_]", 2), ld_i8_id2)
    p = replace(p, _nest(p, "acc[_] += _", 3), matmul_acc_i8)
    p = replace(p, _nest(p, "sum[_] = acc[_]", 2), st_acc_i32)
    p = simplify(p)

    for _ in range(2):
        p = divide_loop(p, "oci", 8, ["ocio", "ocii"], perfect=True)
        p = unroll_loop(p, "ocii")
    return simplify(p)


def _schedule_conv3_gemmini(p):
    # Offload conv3 to Gemmini with contiguous pre-transposed weights
    p = stage_mem(p, p.find_loop("oh"), "weights[0:8, 0:3, 0:3, 0:32]", "weights_copy")
    p = rearrange_dim(p, "weights_copy : _", [1, 2, 3, 0])
    p = simplify(p)
    p = reorder_loops(p, "i0 i1")
    p = reorder_loops(p, "i0 i2")
    p = reorder_loops(p, "i0 i3")
    p = unroll_loop(p, "i0")
    p = unroll_loop(p, "i1")
    for _ in range(3):
        p = unroll_loop(p, "i2")
    p = simplify(p)

    p = expand_dim(p, "sum : _", 16, "oc")
    p = lift_alloc(p, "sum : _")
    p = expand_dim(p, "sum : _", 5, "ow")
    p = lift_alloc(p, "sum : _")
    p = fission(p, p.find("sum[_] = bias[_]").after(), n_lifts=2)
    p = fission(p, p.find_loop("kh").after(), n_lifts=2)
    p = reorder_loops(p, "oc kh")
    p = reorder_loops(p, "ow kh")
    p = simplify(p)

    p = stage_mem(p, p.find_loop("kh"), "sum[0:5, 0:16]", "acc")
    p = simplify(p)
    p = reorder_loops(p, "oc kw")
    p = reorder_loops(p, "ow kw")
    p = divide_loop(p, "ic", 16, ["ico", "ici"], perfect=True)
    p = reorder_loops(p, "oc ico")
    p = reorder_loops(p, "ow ico")
    p = simplify(p)

    p = stage_mem(
        p,
        _nest(p, "acc[_] += _", 3),
        "input[kh + oh, kw:kw + 5, 16 * ico:16 * ico + 16]",
        "as_",
    )
    p = stage_mem(
        p,
        _nest(p, "acc[_] += _", 3),
        "weights_copy[kh, kw, 16 * ico:16 * ico + 16, 0:8]",
        "bs",
    )
    p = resize_dim(p, "bs : _", 1, 16, 0)
    p = simplify(p)

    for buf in ["as_", "bs", "weights_copy", "acc", "sum"]:
        p = _lift_all(p, buf)

    p = set_memory(p, "sum : _", DRAM_STATIC)
    p = set_memory(p, "weights_copy : _", DRAM_STATIC)
    p = set_memory(p, "acc : _", GEMM_ACCUM)
    p = set_memory(p, "as_ : _", GEMM_SCRATCH)
    p = set_memory(p, "bs : _", GEMM_SCRATCH)

    p = replace(p, _nest(p, "acc[_] = sum[_]", 2), ld_acc_i32)
    p = replace(p, _nest(p, "as_[_] = input[_]", 2), ld_i8_id1)
    p = replace(p, _nest(p, "bs[_] = weights_copy[_]", 2), ld_i8_id2)
    p = replace(p, _nest(p, "acc[_] += _", 3), matmul_acc_i8)
    p = replace(p, _nest(p, "sum[_] = acc[_]", 2), st_acc_i32)
    p = simplify(p)

    for _ in range(2):
        p = unroll_loop(p, "oc")
    return simplify(p)


def _schedule_nlb_conv_gemmini(p, out_h, in_ch, out_ch):
    # Pre-transpose weights to [1, 1, in_ch, out_ch] contiguously
    p = stage_mem(p, p.find_loop("oh"), f"weights[0:{out_ch}, 0:1, 0:1, 0:{in_ch}]", "weights_copy")
    p = rearrange_dim(p, "weights_copy : _", [1, 2, 3, 0])
    p = simplify(p)
    p = reorder_loops(p, "i0 i1")
    p = reorder_loops(p, "i0 i2")
    p = reorder_loops(p, "i0 i3")
    p = divide_loop(p, "i0", 16, ["i0o", "i0i"], perfect=True)
    p = unroll_loop(p, "i0i")
    p = unroll_loop(p, "i1")
    p = unroll_loop(p, "i2")
    p = simplify(p)

    p = divide_loop(p, "oc", 16, ["oco", "oci"], perfect=True)
    p = reorder_loops(p, "ow oco")

    p = expand_dim(p, "sum : _", 16, "oci")
    p = lift_alloc(p, "sum : _")
    p = expand_dim(p, "sum : _", out_h, "ow")
    p = lift_alloc(p, "sum : _")
    p = fission(p, p.find("sum[_] = bias[_]").after(), n_lifts=2)
    p = fission(p, p.find_loop("kh").after(), n_lifts=2)
    p = reorder_loops(p, "oci kh")
    p = reorder_loops(p, "ow kh")
    p = simplify(p)

    p = stage_mem(p, p.find_loop("kh"), f"sum[0:{out_h}, 0:16]", "acc")
    p = simplify(p)
    p = reorder_loops(p, "oci kw")
    p = reorder_loops(p, "ow kw")
    p = divide_loop(p, "ic", 16, ["ico", "ici"], perfect=True)
    p = reorder_loops(p, "oci ico")
    p = reorder_loops(p, "ow ico")
    p = simplify(p)

    p = stage_mem(
        p,
        _nest(p, "acc[_] += _", 3),
        f"input[kh + oh, kw:kw + {out_h}, 16 * ico:16 * ico + 16]",
        "as_",
    )
    p = stage_mem(
        p,
        _nest(p, "acc[_] += _", 3),
        f"weights_copy[kh, kw, 16 * ico:16 * ico + 16, 16 * oco:16 * oco + 16]",
        "bs",
    )
    p = simplify(p)

    for buf in ["as_", "bs", "weights_copy", "acc", "sum"]:
        p = _lift_all(p, buf)

    p = set_memory(p, "sum : _", DRAM_STATIC)
    p = set_memory(p, "weights_copy : _", DRAM_STATIC)
    p = set_memory(p, "acc : _", GEMM_ACCUM)
    p = set_memory(p, "as_ : _", GEMM_SCRATCH)
    p = set_memory(p, "bs : _", GEMM_SCRATCH)

    p = replace(p, _nest(p, "acc[_] = sum[_]", 2), ld_acc_i32)
    p = replace(p, _nest(p, "as_[_] = input[_]", 2), ld_i8_id1)
    p = replace(p, _nest(p, "bs[_] = weights_copy[_]", 2), ld_i8_id2)
    p = replace(p, _nest(p, "acc[_] += _", 3), matmul_acc_i8)
    p = replace(p, _nest(p, "sum[_] = acc[_]", 2), st_acc_i32)
    p = simplify(p)

    for _ in range(2):
        p = divide_loop(p, "oci", 8, ["ocio", "ocii"], perfect=True)
        p = unroll_loop(p, "ocii")
    return simplify(p)


# --------------------------------------------------------------------------- #
# 3. CPU Kernel Scheduling (Tiling, Unrolling, and Register Caching)
# --------------------------------------------------------------------------- #


def _schedule_conv1(p):
    p = unroll_loop(p, "kh")
    for _ in range(3):
        p = unroll_loop(p, "kw")
    for _ in range(9):
        p = unroll_loop(p, "ic")
    p = divide_loop(p, "oc", 4, ["oco", "oci"], perfect=True)
    p = unroll_loop(p, "oci")
    p = divide_loop(p, "ow", 3, ["owo", "owi"], perfect=True)
    p = reorder_loops(p, "owi oco")
    p = unroll_loop(p, "owi")
    return simplify(p)


def _schedule_resadd(p):
    p = divide_loop(p, "j", 8, ["jo", "ji"], perfect=True)
    p = unroll_loop(p, "ji")
    return simplify(p)


def _schedule_leaky_k(p, unroll_factor):
    p = divide_loop(p, "k", unroll_factor, ["ko", "ki"], perfect=True)
    p = unroll_loop(p, "ki")
    return simplify(p)


def _schedule_fc1(p):
    p = unroll_loop(p, "k1")
    for _ in range(5):
        p = unroll_loop(p, "k2")
    p = divide_loop(p, "k0", 4, ["k0o", "k0i"], perfect=True)
    p = unroll_loop(p, "k0i")
    return simplify(p)


def _schedule_small_fc(p):
    p = unroll_loop(p, "k1")
    p = unroll_loop(p, "k2")
    p = unroll_loop(p, "k0")
    return simplify(p)


def _schedule_dense_leaky(p):
    p = unroll_loop(p, "j")
    p = unroll_loop(p, "k")
    return simplify(p)


# Build scheduled procedures
matmul_transA_gemmini = rename(_schedule_matmul_transA_gemmini(matmul_transA), "matmul_transA_gemmini")
matmul_transB_gemmini = rename(_schedule_matmul_transB_gemmini(matmul_transB), "matmul_transB_gemmini")
conv2_gemmini = rename(_schedule_conv2_gemmini(conv2), "conv2_gemmini")
conv3_gemmini = rename(_schedule_conv3_gemmini(conv3), "conv3_gemmini")
nlb_qkv_conv_gemmini = rename(_schedule_nlb_conv_gemmini(nlb_qkv_conv, 9, 64, 32), "nlb_qkv_conv_gemmini")
nlb_out_conv_gemmini = rename(_schedule_nlb_conv_gemmini(nlb_out_conv, 9, 32, 64), "nlb_out_conv_gemmini")

conv1_opt = rename(_schedule_conv1(conv1), "conv1_opt")
resadd_opt = rename(_schedule_resadd(resadd), "resadd_opt")
leaky1_opt = rename(_schedule_leaky_k(leaky1, 16), "leaky1_opt")
leaky3_opt = rename(_schedule_leaky_k(leaky3, 16), "leaky3_opt")
leaky5_opt = rename(unroll_loop(unroll_loop(leaky5, "k"), "j"), "leaky5_opt")

fc1_opt = rename(_schedule_fc1(fc1), "fc1_opt")
fc2_opt = rename(_schedule_small_fc(fc2), "fc2_opt")
fc3_opt = rename(_schedule_small_fc(fc3), "fc3_opt")
fc4_opt = rename(_schedule_small_fc(fc4), "fc4_opt")
fc_output_opt = rename(_schedule_small_fc(fc_output), "fc_output_opt")

dense1_leaky_opt = rename(_schedule_dense_leaky(dense1_leaky), "dense1_leaky_opt")
dense3_leaky_opt = rename(_schedule_dense_leaky(dense3_leaky), "dense3_leaky_opt")
dense5_leaky_opt = rename(_schedule_dense_leaky(dense5_leaky), "dense5_leaky_opt")
dense7_leaky_opt = rename(_schedule_dense_leaky(dense7_leaky), "dense7_leaky_opt")

# Substitute into nlb
_nlb = nlb
_nlb = call_eqv(_nlb, "matmul_transA(_)", matmul_transA_gemmini)
_nlb = call_eqv(_nlb, "matmul_transB(_)", matmul_transB_gemmini)
for _ in range(3):
    _nlb = call_eqv(_nlb, "nlb_qkv_conv(_)", nlb_qkv_conv_gemmini)
_nlb = call_eqv(_nlb, "nlb_out_conv(_)", nlb_out_conv_gemmini)
_nlb = call_eqv(_nlb, "resadd(_)", resadd_opt)
for _ in range(4):
    _nlb = divide_loop(_nlb, "c", 8, ["co", "ci"], perfect=True)
    _nlb = unroll_loop(_nlb, "ci")
    _nlb = unroll_loop(_nlb, "w")
for _ in range(3):
    _nlb = unroll_loop(_nlb, "j2")
_nlb = simplify(_nlb)
_nlb = rename(_nlb, "nlb_gemmini")

# Substitute into braggnn_inference
_inf = braggnn_inference
_inf = call_eqv(_inf, "conv1(_)", conv1_opt)
_inf = call_eqv(_inf, "nlb(_)", _nlb)
_inf = call_eqv(_inf, "leaky1(_)", leaky1_opt)
_inf = call_eqv(_inf, "conv2(_)", conv2_gemmini)
_inf = call_eqv(_inf, "leaky3(_)", leaky3_opt)
_inf = call_eqv(_inf, "conv3(_)", conv3_gemmini)
_inf = call_eqv(_inf, "leaky5(_)", leaky5_opt)
_inf = call_eqv(_inf, "fc1(_)", fc1_opt)
_inf = call_eqv(_inf, "dense1_leaky(_)", dense1_leaky_opt)
_inf = call_eqv(_inf, "fc2(_)", fc2_opt)
_inf = call_eqv(_inf, "dense3_leaky(_)", dense3_leaky_opt)
_inf = call_eqv(_inf, "fc3(_)", fc3_opt)
_inf = call_eqv(_inf, "dense5_leaky(_)", dense5_leaky_opt)
_inf = call_eqv(_inf, "fc4(_)", fc4_opt)
_inf = call_eqv(_inf, "dense7_leaky(_)", dense7_leaky_opt)
_inf = call_eqv(_inf, "fc_output(_)", fc_output_opt)

_inf = unroll_loop(_inf, "w")
_inf = unroll_loop(_inf, "h")
_inf = unroll_loop(_inf, "r")
for _ in range(5):
    _inf = unroll_loop(_inf, "c")
_inf = unroll_loop(_inf, "ch")
_inf = simplify(_inf)

braggnn_inference = rename(_inf, "braggnn_inference")
