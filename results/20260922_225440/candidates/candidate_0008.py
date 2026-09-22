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
from exo.API_scheduling import unroll_loop, mult_dim, mult_loops
from exo import instr, DRAM
from exo.libs.externs import relu, select
from exo.platforms.gemmini import acc_scale, clamp
from gemmini import _gemm_loop_matmul_fc

NLB_ROW_TILE = 8


def fence_after(p, pattern):
    return insert_noop_call(p, p.find(pattern).after(), fence, [])


def _blocks(n):
    tiles = (n + 15) // 16
    return tiles, tiles * 16 - n


def make_loop_matmul_fc_flat(name, in_features, out_features, act=False):
    c_instr = _gemm_loop_matmul_fc(in_features, out_features, act)

    if act:
        @instr(c_instr)
        def loop_matmul_fc_flat(
            inp: i8[in_features] @ DRAM,
            weights: i8[out_features, in_features] @ DRAM,
            bias: i32[out_features] @ DRAM,
            output: i8[out_features, 1, 1] @ DRAM,
            scale: f32 @ DRAM,
            post_scale: f32 @ DRAM,
        ):
            for j in seq(0, out_features):
                res: i32
                res = bias[j]
                for k in seq(0, in_features):
                    w_s: i8 @ DRAM
                    w_s = weights[j, k]
                    i_s: i8 @ DRAM
                    i_s = inp[k]
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
        def loop_matmul_fc_flat(
            inp: i8[in_features] @ DRAM,
            weights: i8[out_features, in_features] @ DRAM,
            bias: i32[out_features] @ DRAM,
            output: i8[out_features, 1, 1] @ DRAM,
            scale: f32 @ DRAM,
        ):
            for j in seq(0, out_features):
                res: i32
                res = bias[j]
                for k in seq(0, in_features):
                    w_s: i8 @ DRAM
                    w_s = weights[j, k]
                    i_s: i8 @ DRAM
                    i_s = inp[k]
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

    return rename(loop_matmul_fc_flat, name)


in_features = CONV3_FILTERS * CONV3_DIM * CONV3_DIM
out_features = FC1_UNITS
do_fc1_flat = make_loop_matmul_fc_flat("do_fc1_flat", in_features, out_features, act=True)


def sched_conv(cpu, in_dim, in_ch, out_ch, k, act=False):
    name = cpu.name()[: -len("_cpu")]

    do_conv = make_loop_conv_ws(f"do_{name}", in_dim, in_ch, out_ch, k, act)

    gemmini = rename(cpu, name)
    gemmini = replace(gemmini, "for orow in _:_", do_conv)
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
    gemmini = divide_loop(gemmini, "i1", NLB_ROW_TILE, ["i1_o", "i1_i"], tail="cut")
    gemmini = simplify(gemmini)
    gemmini = replace(gemmini, "for i1_i in _:_", do_matmul)
    gemmini = replace(gemmini, gemmini.find_loop("i1_o").next(), do_matmul)
    gemmini = fence_after(gemmini, f"do_{name}(_) #1")

    return gemmini


def sched_softmax(cpu):
    name = cpu.name()[: -len("_cpu")]

    do_softmax = make_loop_softmax(f"do_{name}", CONV1_DIM, SOFTMAX_MAX_SHIFT)

    gemmini = rename(cpu, name)
    gemmini = divide_loop(gemmini, "i1", NLB_ROW_TILE, ["i1_o", "i1_i"], tail="cut")
    gemmini = simplify(gemmini)
    gemmini = replace(gemmini, "for i1_i in _:_", do_softmax)
    gemmini = replace(gemmini, gemmini.find_loop("i1_o").next(), do_softmax)
    gemmini = fence_after(gemmini, f"do_{name}(_) #1")

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
nlb_qkv_conv = sched_conv(nlb_qkv_conv_cpu, CONV1_DIM, CONV1_FILTERS, CONV2_FILTERS, 1)
nlb_out_conv = sched_conv(nlb_out_conv_cpu, CONV1_DIM, CONV2_FILTERS, CONV1_FILTERS, 1)

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
    gemmini = call_eqv(gemmini, "matmul_theta_phi_cpu(_)", matmul_theta_phi)
    gemmini = call_eqv(gemmini, "nlb_softmax_cpu(_)", nlb_softmax)
    gemmini = call_eqv(gemmini, "matmul_attention_g_cpu(_)", matmul_attention_g)
    gemmini = call_eqv(gemmini, "nlb_out_conv_cpu(_)", nlb_out_conv)
    gemmini = call_eqv(gemmini, "resadd_relu_cpu(_)", resadd_relu)

    # Try unrolling CPU loops in nlb
    for l_name in ["i", "j", "i0", "i1", "i2", "och", "k"]:
        for _ in range(4):
            try:
                gemmini = unroll_loop(gemmini, l_name)
            except Exception:
                break
    return gemmini


nlb = schedule_nlb()


def schedule_braggnn():
    gemmini = rename(braggnn_inference_cpu, "braggnn_inference")
    gemmini = call_eqv(gemmini, "conv1_cpu(_)", conv1)
    gemmini = call_eqv(gemmini, "nlb_cpu(_)", nlb)
    gemmini = call_eqv(gemmini, "conv2_cpu(_)", conv2)
    gemmini = call_eqv(gemmini, "conv3_cpu(_)", conv3)
    # Do NOT call_eqv fc1_cpu here. We keep it as fc1_cpu to inline and flatten it later in schedule_eval
    gemmini = call_eqv(gemmini, "fc2_cpu(_)", fc2)
    gemmini = call_eqv(gemmini, "fc3_cpu(_)", fc3)
    gemmini = call_eqv(gemmini, "fc4_cpu(_)", fc4)
    gemmini = call_eqv(gemmini, "fc_output_cpu(_)", fc_output)

    # Try unrolling CPU loops in braggnn_inference
    for l_name in ["i", "j", "i0", "i1", "i2", "och", "k"]:
        for _ in range(4):
            try:
                gemmini = unroll_loop(gemmini, l_name)
            except Exception:
                break
    return gemmini


braggnn_inference = schedule_braggnn()


def schedule_eval():
    gemmini = rename(braggnn_eval_cpu, "braggnn_eval")
    gemmini = call_eqv(gemmini, "braggnn_inference_cpu(_)", braggnn_inference)
    gemmini = inline(gemmini, "braggnn_inference(_)")

    # Inline fc1_cpu
    gemmini = inline(gemmini, "fc1_cpu(_)")

    # Flatten weights and inputs with multiple trial names for robust matching
    for w_name in ["fc1_weights", "fc1_weight", "fc1_w"]:
        try:
            gemmini = mult_dim(gemmini, w_name, 1, 2)
            gemmini = simplify(gemmini)
            gemmini = mult_dim(gemmini, w_name, 1, 2)
            gemmini = simplify(gemmini)
            break
        except Exception:
            pass

    for buf_name in ["fc1_in", "fc1_inp", "fc1_input", "conv3_out"]:
        try:
            gemmini = mult_dim(gemmini, buf_name, 0, 1)
            gemmini = simplify(gemmini)
            gemmini = mult_dim(gemmini, buf_name, 0, 1)
            gemmini = simplify(gemmini)
            break
        except Exception:
            pass

    # Merge loops of fc1
    try:
        loop_kch = gemmini.find_loop("kch")
        gemmini = mult_loops(gemmini, loop_kch, loop_kch.body()[0])
        gemmini = simplify(gemmini)

        loop_kch2 = gemmini.find_loop("kch")
        gemmini = mult_loops(gemmini, loop_kch2, loop_kch2.body()[0])
        gemmini = simplify(gemmini)
    except Exception:
        pass

    # Replace with do_fc1_flat
    try:
        loop_j = gemmini.find_loop("j")
        gemmini = replace(gemmini, loop_j, do_fc1_flat)
        gemmini = fence_after(gemmini, "do_fc1_flat(_)")
    except Exception:
        pass

    # Try unrolling CPU loops in braggnn_eval
    for l_name in ["i", "j", "i0", "i1", "i2", "och", "k"]:
        for _ in range(4):
            try:
                gemmini = unroll_loop(gemmini, l_name)
            except Exception:
                break

    return gemmini


braggnn_eval = schedule_eval()
# EVOLVE-BLOCK-END


__all__ = ["braggnn_eval"]
