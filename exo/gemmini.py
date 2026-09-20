# ruff: noqa: F821, RUF016

from __future__ import annotations

from exo.API_scheduling import rename
from exo.libs.externs import expf, relu, select
from exo.libs.memories import DRAM_STATIC
from exo.platforms.gemmini import acc_scale, clamp

from exo import DRAM, instr

_gemm_fence = "gemmini_fence();"


@instr(_gemm_fence)
def fence():
    pass


def _blocks(n):
    return (n + 15) // 16, (n + 15) // 16 * 16 - n


def _gemm_loop_conv_ws(in_dim, in_ch, out_ch, k, relu_act):
    out_dim = in_dim - k + 1
    max_pixels_per_row = 1 if in_ch > 16 else min(16 // in_ch, k)
    act = "RELU" if relu_act else "NO_ACTIVATION"
    scale = "({scale}[0] * {relu_scale}[0])" if relu_act else "{scale}[0]"
    return (
        f"gemmini_extended_config_st(({out_ch}), {act}, "
        + scale
        + ");\n"
        + "gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, 0, 1, 1, 0, (0), 0);\n"
        + f"gemmini_loop_conv_ws(1, {in_dim}, {in_dim}, {in_ch}, {out_ch}, "
        f"{out_dim}, {out_dim}, {out_dim}, {out_dim}, 1, 0, {k}, 1, 1, 1, 0, "
        f"1, {out_dim}, {out_dim}, {out_ch}, {k}, {k}, {in_ch}, "
        f"0, 0, 0, 0, 0, 0, 0, 0, {out_dim}, {out_dim}, "
        "{weights}, {output}, {bias}, {input}, 0, 1, 0, 0, 0, "
        f"{act}, 0, 0, 0, 0, {max_pixels_per_row}, {in_ch}, {out_ch}, {out_ch}, 0, 1, 1);\n"
        + "gemmini_fence();"
    )


def make_loop_conv_ws(name, in_dim, in_ch, out_ch, k, relu_act=False):
    out_dim = in_dim - k + 1
    c_str = _gemm_loop_conv_ws(in_dim, in_ch, out_ch, k, relu_act)

    if relu_act:

        @instr(c_str)
        def loop_conv_ws(
            input: i8[in_dim, in_dim, in_ch] @ DRAM,
            weights: i8[k, k, in_ch, out_ch] @ DRAM,
            bias: i32[out_ch] @ DRAM,
            output: i8[out_dim, out_dim, out_ch] @ DRAM,
            scale: f32 @ DRAM,
            relu_scale: f32 @ DRAM,
        ):
            for oh in seq(0, out_dim):
                for ow in seq(0, out_dim):
                    for oc in seq(0, out_ch):
                        sum: i32
                        sum = bias[oc]

                        for kh in seq(0, k):
                            for kw in seq(0, k):
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

    else:

        @instr(c_str)
        def loop_conv_ws(
            input: i8[in_dim, in_dim, in_ch] @ DRAM,
            weights: i8[k, k, in_ch, out_ch] @ DRAM,
            bias: i32[out_ch] @ DRAM,
            output: i8[out_dim, out_dim, out_ch] @ DRAM,
            scale: f32 @ DRAM,
        ):
            for oh in seq(0, out_dim):
                for ow in seq(0, out_dim):
                    for oc in seq(0, out_ch):
                        sum: i32
                        sum = bias[oc]

                        for kh in seq(0, k):
                            for kw in seq(0, k):
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

    return rename(loop_conv_ws, name)


def _gemm_loop_matmul_fc(in_features, out_features, relu_act):
    J, pad_J = _blocks(out_features)
    K, pad_K = _blocks(in_features)
    act = "RELU" if relu_act else "NO_ACTIVATION"
    scale = "({scale}[0] * {relu_scale}[0])" if relu_act else "{scale}[0]"
    return (
        "gemmini_extended_config_ex(WEIGHT_STATIONARY, (0), 0, 1, (0), (1));\n"
        + f"gemmini_extended_config_st((1), {act}, "
        + scale
        + ");\n"
        + f"gemmini_extended3_config_ld(({in_features}), 1.0f, false, (0));\n"
        + f"gemmini_extended3_config_ld(({in_features}), 1.0f, false, (1));\n"
        + "gemmini_extended3_config_ld((1), 1.0f, false, (2));\n"
        + f"gemmini_loop_ws(1, {J}, {K}, 15, {pad_J}, {pad_K}, "
        "{input}, {weights}, {bias}, {output}, "
        f"{in_features}, {in_features}, {out_features}, {out_features}, "
        f"0, 1, 0, 0, 1, {act}, 1, 1, 0);\n" + "gemmini_fence();"
    )


def make_loop_matmul_fc(name, d0, d1, d2, out_features, relu_act=False):
    in_features = d0 * d1 * d2
    c_str = _gemm_loop_matmul_fc(in_features, out_features, relu_act)

    if relu_act:

        @instr(c_str)
        def loop_matmul_fc(
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

    else:

        @instr(c_str)
        def loop_matmul_fc(
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

    return rename(loop_matmul_fc, name)


def _gemm_loop_matmul_theta_phi(m, d):
    dd = d * d
    J, pad_J = _blocks(dd)
    K, pad_K = _blocks(m)
    rows = "(" + str(d) + " * {n} + 15) / 16"
    pad_rows = rows + " * 16 - " + str(d) + " * {n}"
    return (
        "gemmini_extended_config_ex(WEIGHT_STATIONARY, (0), 0, 1, (0), (1));\n"
        + f"gemmini_extended_config_st(({dd}), NO_ACTIVATION, "
        + "{scale}[0]);\n"
        + f"gemmini_extended3_config_ld(({m}), 1.0f, false, (0));\n"
        + f"gemmini_extended3_config_ld(({m}), 1.0f, false, (1));\n"
        + f"gemmini_extended3_config_ld(({dd}), 1.0f, false, (2));\n"
        + f"gemmini_loop_ws({rows}, {J}, {K}, {pad_rows}, {pad_J}, {pad_K}, "
        "&{A_data}, &{B_data}, 0, &{C_data}, "
        f"{m}, {m}, {dd}, {dd}, 0, 1, 0, 0, 0, NO_ACTIVATION, 1, 1, 0);"
    )


def make_loop_matmul_theta_phi_tile(name, m, d):
    @instr(_gemm_loop_matmul_theta_phi(m, d))
    def loop_matmul_theta_phi_tile(
        n: size,
        A: [i8][n, d, m] @ DRAM,
        B: [i8][d, d, m] @ DRAM,
        C: [i8][n, d, d, d] @ DRAM,
        scale: f32 @ DRAM,
    ):
        for i1 in seq(0, n):
            for i2 in seq(0, d):
                for j1 in seq(0, d):
                    for j2 in seq(0, d):
                        sum: i32
                        sum = 0
                        for k in seq(0, m):
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

    return rename(loop_matmul_theta_phi_tile, name)


def _gemm_loop_matmul_attention_g(m, d):
    dd = d * d
    I, pad_I = _blocks(dd)
    J, pad_J = _blocks(m)
    return (
        "gemmini_extended_config_ex(WEIGHT_STATIONARY, (0), 0, 1, (0), (0));\n"
        + f"gemmini_extended_config_st(({m}), NO_ACTIVATION, "
        + "{scale}[0]);\n"
        + f"gemmini_extended3_config_ld(({dd}), 1.0f, false, (0));\n"
        + f"gemmini_extended3_config_ld(({m}), 1.0f, false, (1));\n"
        + f"gemmini_extended3_config_ld(({m}), 1.0f, false, (2));\n"
        + f"gemmini_loop_ws({I}, {J}, {I}, {pad_I}, {pad_J}, {pad_I}, "
        "{A}, {B}, 0, {C}, "
        f"{dd}, {m}, {m}, {m}, 0, 0, 0, 0, 0, NO_ACTIVATION, 1, 1, 0);\n"
        + "gemmini_fence();"
    )


def make_loop_matmul_attention_g(name, m, d):
    @instr(_gemm_loop_matmul_attention_g(m, d))
    def loop_matmul_attention_g(
        A: i8[d, d, d, d] @ DRAM,
        B: i8[d, d, m] @ DRAM,
        C: i8[d, d, m] @ DRAM,
        scale: f32 @ DRAM,
    ):
        for i1 in seq(0, d):
            for i2 in seq(0, d):
                for j in seq(0, m):
                    sum: i32
                    sum = 0
                    for k1 in seq(0, d):
                        for k2 in seq(0, d):
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

    return rename(loop_matmul_attention_g, name)


def _gemm_loop_resadd_relu(d, ch):
    I, pad_I = _blocks(d * d)
    J, pad_J = _blocks(ch)
    return (
        f"gemmini_extended_config_st(({ch}), RELU, "
        + "{C_scale}[0]);\n"
        + "gemmini_extended_config_ex(WEIGHT_STATIONARY, (0), 0, 1, (0), (0));\n"
        + f"gemmini_extended4_config_ld(({ch}), "
        + "{A_scale}[0], true, DIM, (0));\n"
        + f"gemmini_extended4_config_ld(({ch}), "
        + "{B_scale}[0], true, DIM, (1));\n"
        + f"gemmini_loop_ws({I}, {J}, 0, {pad_I}, {pad_J}, 0, "
        "{A}, {B}, 0, {C}, "
        f"{ch}, {ch}, 0, {ch}, 0, 0, 0, 0, 0, RELU, 0, 0, 1);\n" + "gemmini_fence();"
    )


def make_loop_resadd_relu(name, d, ch):
    @instr(_gemm_loop_resadd_relu(d, ch))
    def loop_resadd_relu(
        A_scale: f32 @ DRAM,
        B_scale: f32 @ DRAM,
        C_scale: f32 @ DRAM,
        A: i8[d, d, ch] @ DRAM,
        B: i8[d, d, ch] @ DRAM,
        C: i8[d, d, ch] @ DRAM,
    ):
        for i1 in seq(0, d):
            for i2 in seq(0, d):
                for j in seq(0, ch):
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

    return rename(loop_resadd_relu, name)


def _gemm_loop_softmax(d):
    dd = d * d
    J, pad_J = _blocks(dd)
    rows = "(" + str(d) + " * {n} + 15) / 16"
    pad_rows = rows + " * 16 - " + str(d) + " * {n}"
    entries = ", ".join(f"[{(dd + 1) * i}] = 1" for i in range(dd))
    qln2 = "(int)(0.693147 / {in_scale}[0])"
    qb = "(int32_t)(1.353f / {in_scale}[0])"
    qc = "(int32_t)(0.344f / (0.3585f * {in_scale}[0] * {in_scale}[0]))"
    return (
        "gemmini_extended_config_ex(WEIGHT_STATIONARY, (0), 0, 1, (0), (0));\n"
        + f"gemmini_extended_config_st(({dd}), 0, "
        + "1.0f / (127.0f * {out_scale}[0]));\n"
        + f"gemmini_extended3_config_ld(({dd}), 1.0f, false, (0));\n"
        + f"gemmini_extended3_config_ld(({dd}), 1.0f, false, (1));\n"
        + f"gemmini_extended3_config_ld(({dd}), 1.0f, false, (2));\n"
        + "gemmini_config_norm("
        + qln2
        + ", 0, 0, 1, 0, "
        + qb
        + ", "
        + qc
        + ");\n"
        + "gemmini_config_norm((65536 / "
        + qln2
        + "), 1, 0, 1, 0, "
        + qb
        + ", "
        + qc
        + ");\n"
        + "{{ "
        + f"static const int8_t identity[{dd} * {dd}] = "
        + "{{"
        + entries
        + "}};\n"
        + f"gemmini_loop_ws({rows}, {J}, {J}, {pad_rows}, {pad_J}, {pad_J}, "
        "&{A_data}, identity, 0, &{C_data}, "
        f"{dd}, {dd}, {dd}, {dd}, "
        "0, 0, 0, 0, 0, SOFTMAX, 1, 1, 0); }}"
    )


def make_loop_softmax_tile(name, d):
    @instr(_gemm_loop_softmax(d))
    def loop_softmax_tile(
        n: size,
        A: [i8][n, d, d, d] @ DRAM,
        C: [i8][n, d, d, d] @ DRAM,
        in_scale: f32 @ DRAM,
        out_scale: f32 @ DRAM,
    ):
        for i1 in seq(0, n):
            for i2 in seq(0, d):
                buf: f32[d, d] @ DRAM_STATIC
                max_q: f32
                max_q = A[i1, i2, 0, 0]
                for j1 in seq(0, d):
                    for j2 in seq(0, d):
                        v: f32
                        v = A[i1, i2, j1, j2]
                        max_q = select(max_q, v, v, max_q)

                sum_exp: f32
                sum_exp = 0.0
                for j1 in seq(0, d):
                    for j2 in seq(0, d):
                        av: f32
                        e: f32
                        av = A[i1, i2, j1, j2]
                        e = expf(in_scale * (av - max_q))
                        buf[j1, j2] = e
                        sum_exp += e

                for j1 in seq(0, d):
                    for j2 in seq(0, d):
                        tmp: f32
                        out_val: i8
                        tmp = buf[j1, j2] / sum_exp / out_scale
                        clamp(tmp, out_val)
                        C[i1, i2, j1, j2] = out_val

    return rename(loop_softmax_tile, name)
