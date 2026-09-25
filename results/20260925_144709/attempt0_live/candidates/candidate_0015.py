from __future__ import annotations

from braggnn_reference import (
    CONV1_DIM,
    CONV1_FILTERS,
    CONV2_DIM,
    CONV2_FILTERS,
    CONV3_DIM,
    CONV3_FILTERS,
    FC1_UNITS,
    FC2_UNITS,
    FC3_UNITS,
    FC4_UNITS,
    INPUT_DIM,
    OUTPUT_UNITS,
    SOFTMAX_MAX_SHIFT,
    braggnn_eval_cpu,
    braggnn_inference_cpu,
    conv1_cpu,
    conv2_cpu,
    conv3_cpu,
    fc1_cpu,
    fc2_cpu,
    fc3_cpu,
    fc4_cpu,
    fc_output_cpu,
    matmul_attention_g_cpu,
    matmul_theta_phi_cpu,
    nlb_cpu,
    nlb_out_conv_cpu,
    nlb_qkv_conv_cpu,
    nlb_softmax_cpu,
    resadd_relu_cpu,
)
from exo.API_scheduling import (
    call_eqv,
    divide_loop,
    inline,
    insert_noop_call,
    rename,
    replace,
    simplify,
)
from gemmini import (
    fence,
    make_loop_conv_ws,
    make_loop_matmul,
    make_loop_matmul_fc,
    make_loop_matmul_trans_b,
    make_loop_resadd,
    make_loop_softmax,
)

# EVOLVE-BLOCK-START
from exo import instr, DRAM
from exo.libs.externs import relu
from exo.platforms.gemmini import acc_scale, clamp

def _blocks(n):
    tiles = (n + 15) // 16
    return tiles, tiles * 16 - n


def make_loop_conv1x1_ws(name, out_dim, in_ch, out_ch, act=False):
    I_tiles, pad_I = _blocks(out_dim * out_dim)
    J_tiles, pad_J = _blocks(out_ch)
    K_tiles, pad_K = _blocks(in_ch)
    
    gemm_act = "RELU" if act else "NO_ACTIVATION"
    scale = "({scale}[0] * {post_scale}[0])" if act else "{scale}[0]"
    
    c_instr = (
        "gemmini_extended_config_ex(WEIGHT_STATIONARY, (0), 0, 1, (0), (0));\n"
        + f"gemmini_extended_config_st(({out_ch}), {gemm_act}, "
        + scale
        + ");\n"
        + f"gemmini_extended3_config_ld(({in_ch}), 1.0f, false, (0));\n"
        + f"gemmini_extended3_config_ld(({out_ch}), 1.0f, false, (1));\n"
        + "gemmini_extended3_config_ld((0), 1.0f, false, (2));\n"
        + f"gemmini_loop_ws({I_tiles}, {J_tiles}, {K_tiles}, {pad_I}, {pad_J}, {pad_K}, "
        "{inp}, {weights}, {bias}, {output}, "
        f"{in_ch}, {out_ch}, 0, {out_ch}, "
        "0, 0, 0, 0, 0, "
        f"{gemm_act}, 1, 1, 0);"
    )
    
    if act:
        @instr(c_instr)
        def loop_conv1x1_ws(
            inp: i8[out_dim, out_dim, in_ch] @ DRAM,
            weights: i8[1, 1, in_ch, out_ch] @ DRAM,
            bias: i32[out_ch] @ DRAM,
            output: i8[out_dim, out_dim, out_ch] @ DRAM,
            scale: f32 @ DRAM,
            post_scale: f32 @ DRAM,
        ):
            for orow in seq(0, out_dim):
                for ocol in seq(0, out_dim):
                    for och in seq(0, out_ch):
                        res: i32
                        res = bias[och]
                        for krow in seq(0, 1):
                            for kcol in seq(0, 1):
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
    else:
        @instr(c_instr)
        def loop_conv1x1_ws(
            inp: i8[out_dim, out_dim, in_ch] @ DRAM,
            weights: i8[1, 1, in_ch, out_ch] @ DRAM,
            bias: i32[out_ch] @ DRAM,
            output: i8[out_dim, out_dim, out_ch] @ DRAM,
            scale: f32 @ DRAM,
        ):
            for orow in seq(0, out_dim):
                for ocol in seq(0, out_dim):
                    for och in seq(0, out_ch):
                        res: i32
                        res = bias[och]
                        for krow in seq(0, 1):
                            for kcol in seq(0, 1):
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

    return rename(loop_conv1x1_ws, name)


def sched_conv1x1(cpu, in_dim, in_ch, out_ch, act=False, fence=True):
    name = cpu.name()[: -len("_cpu")]

    do_conv = make_loop_conv1x1_ws(f"do_{name}", in_dim, in_ch, out_ch, act)

    gemmini = rename(cpu, name)
    gemmini = replace(gemmini, "for orow in _:_", do_conv)
    if fence:
        gemmini = fence_after(gemmini, f"do_{name}(_)")

    return gemmini


def fence_after(p, pattern):
    return insert_noop_call(p, p.find(pattern).after(), fence, [])


def sched_conv(cpu, in_dim, in_ch, out_ch, k, act=False, fence=True):
    name = cpu.name()[: -len("_cpu")]

    do_conv = make_loop_conv_ws(f"do_{name}", in_dim, in_ch, out_ch, k, act)

    gemmini = rename(cpu, name)
    gemmini = replace(gemmini, "for orow in _:_", do_conv)
    if fence:
        gemmini = fence_after(gemmini, f"do_{name}(_)")

    return gemmini


def sched_fc(cpu, in_ch, in_h, in_w, out_features, act=False):
    name = cpu.name()[: -len("_cpu")]

    do_fc = make_loop_matmul_fc(f"do_{name}", in_ch, in_h, in_w, out_features, act)

    gemmini = rename(cpu, name)
    gemmini = replace(gemmini, "for j in _:_", do_fc)
    gemmini = fence_after(gemmini, f"do_{name}(_)")

    return gemmini


def sched_matmul_trans_b(cpu):
    name = cpu.name()[: -len("_cpu")]

    do_matmul = make_loop_matmul_trans_b(f"do_{name}", CONV2_FILTERS, CONV1_DIM)

    gemmini = rename(cpu, name)
    gemmini = replace(gemmini, "for i1 in _:_", do_matmul)
    gemmini = fence_after(gemmini, f"do_{name}(_)")

    return gemmini


def sched_softmax(cpu):
    name = cpu.name()[: -len("_cpu")]

    do_softmax = make_loop_softmax(f"do_{name}", CONV1_DIM, SOFTMAX_MAX_SHIFT)

    gemmini = rename(cpu, name)
    gemmini = replace(gemmini, "for i1 in _:_", do_softmax)
    gemmini = fence_after(gemmini, f"do_{name}(_)")

    return gemmini


def sched_matmul(cpu):
    name = cpu.name()[: -len("_cpu")]

    do_matmul = make_loop_matmul(f"do_{name}", CONV2_FILTERS, CONV1_DIM)

    gemmini = rename(cpu, name)
    gemmini = replace(gemmini, "for i1 in _:_", do_matmul)
    gemmini = fence_after(gemmini, f"do_{name}(_)")

    return gemmini


def sched_resadd(cpu):
    name = cpu.name()[: -len("_cpu")]

    do_resadd = make_loop_resadd(f"do_{name}", CONV1_DIM, CONV1_FILTERS)

    gemmini = rename(cpu, name)
    gemmini = replace(gemmini, "for i1 in _:_", do_resadd)
    gemmini = fence_after(gemmini, f"do_{name}(_)")

    return gemmini


conv1 = sched_conv(conv1_cpu, INPUT_DIM, 1, CONV1_FILTERS, 3)
conv2 = sched_conv(conv2_cpu, CONV1_DIM, CONV1_FILTERS, CONV2_FILTERS, 3, act=True)
conv3 = sched_conv(conv3_cpu, CONV2_DIM, CONV2_FILTERS, CONV3_FILTERS, 3, act=True)
nlb_qkv_conv = sched_conv1x1(nlb_qkv_conv_cpu, CONV1_DIM, CONV1_FILTERS, CONV2_FILTERS, fence=False)
nlb_out_conv = sched_conv1x1(nlb_out_conv_cpu, CONV1_DIM, CONV2_FILTERS, CONV1_FILTERS)

fc1 = sched_fc(fc1_cpu, CONV3_FILTERS, CONV3_DIM, CONV3_DIM, FC1_UNITS, act=True)
fc2 = sched_fc(fc2_cpu, FC1_UNITS, 1, 1, FC2_UNITS, act=True)
fc3 = sched_fc(fc3_cpu, FC2_UNITS, 1, 1, FC3_UNITS, act=True)
fc4 = sched_fc(fc4_cpu, FC3_UNITS, 1, 1, FC4_UNITS, act=True)
fc_output = sched_fc(fc_output_cpu, FC4_UNITS, 1, 1, OUTPUT_UNITS)

matmul_theta_phi = sched_matmul_trans_b(matmul_theta_phi_cpu)
nlb_softmax = sched_softmax(nlb_softmax_cpu)
matmul_attention_g = sched_matmul(matmul_attention_g_cpu)
resadd_relu = sched_resadd(resadd_relu_cpu)


def schedule_nlb():
    gemmini = rename(nlb_cpu, "nlb")
    gemmini = call_eqv(gemmini, "nlb_qkv_conv_cpu(_) #0", nlb_qkv_conv)
    gemmini = call_eqv(gemmini, "nlb_qkv_conv_cpu(_) #0", nlb_qkv_conv)
    gemmini = call_eqv(gemmini, "nlb_qkv_conv_cpu(_) #0", nlb_qkv_conv)
    gemmini = insert_noop_call(gemmini, gemmini.find("nlb_qkv_conv(_) #2").after(), fence, [])
    gemmini = call_eqv(gemmini, "matmul_theta_phi_cpu(_)", matmul_theta_phi)
    gemmini = call_eqv(gemmini, "nlb_softmax_cpu(_)", nlb_softmax)
    gemmini = call_eqv(gemmini, "matmul_attention_g_cpu(_)", matmul_attention_g)
    gemmini = call_eqv(gemmini, "nlb_out_conv_cpu(_)", nlb_out_conv)
    gemmini = call_eqv(gemmini, "resadd_relu_cpu(_)", resadd_relu)
    return gemmini


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


def schedule_eval():
    gemmini = rename(braggnn_eval_cpu, "braggnn_eval")
    gemmini = call_eqv(gemmini, "braggnn_inference_cpu(_)", braggnn_inference)
    gemmini = inline(gemmini, "braggnn_inference(_)")
    return gemmini


braggnn_eval = schedule_eval()
# EVOLVE-BLOCK-END


__all__ = ["braggnn_eval"]
