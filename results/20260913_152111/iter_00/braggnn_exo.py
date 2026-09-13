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
