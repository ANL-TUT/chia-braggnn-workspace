# ruff: noqa: F821
"""Hand-tuned Gemmini instrs for braggnn_schedule_fast.py (the Exo form of
braggnn_tune_opus.c, add-gcp-cluster branch).

Kept out of gemmini.py on purpose: gemmini.py is pasted into the SW search's
prompt, and these would hand it the answer the search is meant to find on its
own. braggnn_schedule_fast.py is a measured target, not a reference for it.
"""

from __future__ import annotations

from exo.API_scheduling import make_instr, rename
from exo.libs.externs import relu
from exo.platforms.gemmini import acc_scale, clamp
from gemmini import (
    _blocks,
    _gemm_loop_matmul_fc,
    make_loop_conv_ws,
    make_loop_softmax,
)

from exo import DRAM, instr

# gemmini_loop_ws with extra LOOP_WS rs2 bits: bits 3..7 mark the loop's ldA,
# ldB, ldD, ex and st stages as already started and completed.
_gemm_loop_ws_skip_global = r"""
#ifndef CHIA_LOOP_WS_SKIP
#define CHIA_LOOP_WS_SKIP
#define LOOP_SKIP_LDB (1 << 4)
#define gemmini_loop_ws_skip(I, J, K, pad_I, pad_J, pad_K, A, B, D, C, A_stride, B_stride, D_stride, C_stride, A_transpose, B_transpose, full_C, low_D, ex_accumulate, act, a_spad_id, b_spad_id, is_resadd, skips) \
  { \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(pad_K) << 32) | ((uint64_t)(pad_J) << 16) | (uint64_t)(pad_I), ((uint64_t)(K) << 32) | ((uint64_t)(J) << 16) | (uint64_t)(I), k_LOOP_WS_CONFIG_BOUNDS) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, A, B, k_LOOP_WS_CONFIG_ADDRS_AB) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, D, C, k_LOOP_WS_CONFIG_ADDRS_DC) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, A_stride, B_stride, k_LOOP_WS_CONFIG_STRIDES_AB) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, D_stride, C_stride, k_LOOP_WS_CONFIG_STRIDES_DC) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(a_spad_id) << 18) | ((uint64_t)(b_spad_id) << 16) | ((uint64_t)(act) << 8) | ((low_D) << 2) | ((full_C) << 1) | (ex_accumulate), (skips) | ((is_resadd) << 2) | ((B_transpose) << 1) | (A_transpose), k_LOOP_WS) \
  }
#endif
"""


def _gemm_loop_conv1x1_matmul(in_dim, in_ch, out_ch, reuse_a):
    # C[pixels x out_ch] = A[pixels x in_ch] * W[in_ch x out_ch] + bias, the
    # bias a D row repeated with D stride 0 (as tiled_matmul_outer issues it).
    I, pad_I = _blocks(in_dim * in_dim)
    J, pad_J = _blocks(out_ch)
    K, pad_K = _blocks(in_ch)
    inp = "NULL" if reuse_a else "{inp}"
    return (
        "gemmini_extended_config_ex(WEIGHT_STATIONARY, 0, 0, 1, 0, 0);\n"
        + f"gemmini_extended_config_st({out_ch}, NO_ACTIVATION, "
        + "{scale}[0]);\n"
        + f"gemmini_extended3_config_ld({in_ch}, 1.0f, false, 0);\n"
        + f"gemmini_extended3_config_ld({out_ch}, 1.0f, false, 1);\n"
        + "gemmini_extended3_config_ld(0, 1.0f, false, 2);\n"
        + f"gemmini_loop_ws({I}, {J}, {K}, {pad_I}, {pad_J}, {pad_K}, "
        + inp
        + ", {weights}, {bias}, {output}, "
        f"{in_ch}, {out_ch}, 0, {out_ch}, "
        "0, 0, 0, 0, 1, NO_ACTIVATION, 1, 1, 0);"
    )


def make_loop_conv1x1_matmul(name, in_dim, in_ch, out_ch, reuse_a=False):
    """A 1x1 conv without activation (the semantics of make_loop_conv_ws with
    k = 1) issued as a gemmini_loop_ws matmul instead of gemmini_loop_conv_ws.

    With reuse_a the loop's A address is NULL: it reuses the input that the
    previous loop_ws left in scratchpad region a_spad_id = 1 (gemmini.h's
    a_reuse path). Such a call is only correct directly after another
    make_loop_conv1x1_matmul call on the same input with the same in_dim and
    in_ch, with no Gemmini command in between; Exo cannot check this."""
    p = make_loop_conv_ws(name, in_dim, in_ch, out_ch, 1)
    return make_instr(p, _gemm_loop_conv1x1_matmul(in_dim, in_ch, out_ch, reuse_a))


def _gemm_loop_softmax_load(d):
    cols = d * d
    rows = f"({d} * {{n}} + 15) / 16"
    pad_rows = f"{rows} * 16 - {d} * {{n}}"
    J, pad_J = _blocks(cols)
    qln2 = "(int)(0.693147 / {in_scale}[0])"
    qb = "(int32_t)(1.353f / {in_scale}[0])"
    qc = "(int32_t)(0.344f / (0.3585f * {in_scale}[0] * {in_scale}[0]))"
    return (
        "gemmini_extended_config_ex(WEIGHT_STATIONARY, 0, 0, 1, 0, 0);\n"
        + f"gemmini_extended_config_st({cols}, 0, "
        + "1.0f / (127.0f * {out_scale}[0]));\n"
        + f"gemmini_config_norm({qln2}, 0, 0, 1, 0, {qb}, {qc});\n"
        + f"gemmini_config_norm((65536 / {qln2}), 1, 0, 1, 0, {qb}, {qc});\n"
        + f"gemmini_extended4_config_ld({cols}, 1.0f, true, DIM, 0);\n"
        + f"gemmini_extended4_config_ld({cols}, 0.0f, true, DIM, 1);\n"
        + f"gemmini_loop_ws_skip({rows}, {J}, 0, {pad_rows}, {pad_J}, 0, "
        "&{A_data}, &{A_data}, 0, &{C_data}, "
        f"{cols}, {cols}, 0, {cols}, "
        "0, 0, 0, 0, 0, SOFTMAX, 0, 0, 1, LOOP_SKIP_LDB);"
    )


def make_loop_softmax_load(name, d, max_shift):
    """make_loop_softmax (same semantics), but the input goes straight into
    the accumulator instead of through a matmul by an identity matrix: a
    resadd-style loop (K = 0) computes acc = 1.0 * A + 0.0 * A, with the ldB
    stage marked done (LOOP_WS rs2 bit 4) so A is loaded once. B = A with
    scale 0 keeps the result the same on hardware that ignores the bit."""
    p = make_loop_softmax(name, d, max_shift)
    return make_instr(p, _gemm_loop_softmax_load(d), _gemm_loop_ws_skip_global)


def make_loop_matmul_fc_hwc(name, in_h, in_w, in_ch, out_features, act=False):
    """An fc layer on an HWC input: the reference's (ch, row, col) flatten
    into `flat` followed by make_loop_matmul_fc's semantics, with the weights
    in (unit, row, col, ch) order -- permute the staged weights once with
    rearrange_dim(..., [0, 2, 3, 1]) to get it. The C reads inp directly as
    one in_h * in_w * in_ch row, so the flatten (and `flat`) exists only in
    the spec."""
    c_instr = _gemm_loop_matmul_fc(in_ch * in_h * in_w, out_features, act)

    if act:

        @instr(c_instr)
        def loop_matmul_fc_hwc(
            inp: i8[in_h, in_w, in_ch] @ DRAM,
            flat: i8[in_ch, in_h, in_w] @ DRAM,
            weights: i8[out_features, in_h, in_w, in_ch] @ DRAM,
            bias: i32[out_features] @ DRAM,
            output: i8[out_features, 1, 1] @ DRAM,
            scale: f32 @ DRAM,
            post_scale: f32 @ DRAM,
        ):
            for ch in seq(0, in_ch):
                for r in seq(0, in_h):
                    for c in seq(0, in_w):
                        flat[ch, r, c] = inp[r, c, ch]
            for j in seq(0, out_features):
                res: i32
                res = bias[j]
                for kch in seq(0, in_ch):
                    for krow in seq(0, in_h):
                        for kcol in seq(0, in_w):
                            w_s: i8 @ DRAM
                            w_s = weights[j, krow, kcol, kch]
                            i_s: i8 @ DRAM
                            i_s = flat[kch, krow, kcol]
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

    else:

        @instr(c_instr)
        def loop_matmul_fc_hwc(
            inp: i8[in_h, in_w, in_ch] @ DRAM,
            flat: i8[in_ch, in_h, in_w] @ DRAM,
            weights: i8[out_features, in_h, in_w, in_ch] @ DRAM,
            bias: i32[out_features] @ DRAM,
            output: i8[out_features, 1, 1] @ DRAM,
            scale: f32 @ DRAM,
        ):
            for ch in seq(0, in_ch):
                for r in seq(0, in_h):
                    for c in seq(0, in_w):
                        flat[ch, r, c] = inp[r, c, ch]
            for j in seq(0, out_features):
                res: i32
                res = bias[j]
                for kch in seq(0, in_ch):
                    for krow in seq(0, in_h):
                        for kcol in seq(0, in_w):
                            w_s: i8 @ DRAM
                            w_s = weights[j, krow, kcol, kch]
                            i_s: i8 @ DRAM
                            i_s = flat[kch, krow, kcol]
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

    return rename(loop_matmul_fc_hwc, name)
