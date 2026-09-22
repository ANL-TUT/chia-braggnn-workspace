from __future__ import annotations

import sys

sys.setrecursionlimit(10000)

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
    reorder_loops,
    replace,
    set_memory,
    simplify,
    unroll_loop,
)
from exo.libs.memories import DRAM_STATIC
from gemmini import (
    fence,
    make_loop_conv_ws,
    make_loop_matmul,
    make_loop_matmul_fc,
    make_loop_matmul_trans_b,
    make_loop_resadd,
    make_loop_softmax,
)

NLB_ROW_TILE = 8


def fence_after(p, pattern):
    return insert_noop_call(p, p.find(pattern).after(), fence, [])


def sched_conv_nofence(cpu, in_dim, in_ch, out_ch, k, act=False):
    name = cpu.name()[: -len("_cpu")]

    do_conv = make_loop_conv_ws(f"do_{name}", in_dim, in_ch, out_ch, k, act)

    gemmini = rename(cpu, name)
    gemmini = replace(gemmini, "for orow in _:_", do_conv)

    return gemmini


def sched_conv(cpu, in_dim, in_ch, out_ch, k, act=False):
    name = cpu.name()[: -len("_cpu")]

    do_conv = make_loop_conv_ws(f"do_{name}", in_dim, in_ch, out_ch, k, act)

    gemmini = rename(cpu, name)
    gemmini = replace(gemmini, "for orow in _:_", do_conv)
    gemmini = fence_after(gemmini, f"do_{name}(_)")

    return gemmini


def sched_fc_nofence(cpu, in_ch, in_h, in_w, out_features, act=False):
    name = cpu.name()[: -len("_cpu")]

    do_fc = make_loop_matmul_fc(f"do_{name}", in_ch, in_h, in_w, out_features, act)

    gemmini = rename(cpu, name)
    gemmini = replace(gemmini, "for j in _:_", do_fc)

    return gemmini


def sched_fc(cpu, in_ch, in_h, in_w, out_features, act=False):
    name = cpu.name()[: -len("_cpu")]

    do_fc = make_loop_matmul_fc(f"do_{name}", in_ch, in_h, in_w, out_features, act)

    gemmini = rename(cpu, name)
    gemmini = replace(gemmini, "for j in _:_", do_fc)
    gemmini = fence_after(gemmini, f"do_{name}(_)")

    return gemmini


def sched_fc_cpu(cpu, act=False):
    name = cpu.name()[: -len("_cpu")]

    p = rename(cpu, name)
    p = unroll_loop(p, "kcol")
    p = unroll_loop(p, "krow")
    p = unroll_loop(p, "kch")
    p = unroll_loop(p, "j")
    p = simplify(p)
    while True:
        try:
            p = inline(p, "clamp(_)")
        except Exception:
            break

    return simplify(p)


def sched_matmul_trans_b_nofence(cpu):
    name = cpu.name()[: -len("_cpu")]

    do_matmul = make_loop_matmul_trans_b(f"do_{name}", CONV2_FILTERS, CONV1_DIM)

    gemmini = rename(cpu, name)
    gemmini = divide_loop(gemmini, "i1", NLB_ROW_TILE, ["i1_o", "i1_i"], tail="cut")
    gemmini = simplify(gemmini)
    gemmini = replace(gemmini, "for i1_i in _:_", do_matmul)
    gemmini = replace(gemmini, gemmini.find_loop("i1_o").next(), do_matmul)
    gemmini = unroll_loop(gemmini, "i1_o")
    gemmini = simplify(gemmini)

    return gemmini


def sched_softmax(cpu):
    name = cpu.name()[: -len("_cpu")]

    do_softmax = make_loop_softmax(f"do_{name}", CONV1_DIM, SOFTMAX_MAX_SHIFT)

    gemmini = rename(cpu, name)
    gemmini = divide_loop(gemmini, "i1", NLB_ROW_TILE, ["i1_o", "i1_i"], tail="cut")
    gemmini = simplify(gemmini)
    gemmini = replace(gemmini, "for i1_i in _:_", do_softmax)
    gemmini = replace(gemmini, gemmini.find_loop("i1_o").next(), do_softmax)
    gemmini = unroll_loop(gemmini, "i1_o")
    gemmini = simplify(gemmini)
    gemmini = fence_after(gemmini, f"do_{name}(_) #1")

    return gemmini


def sched_matmul_nofence(cpu):
    name = cpu.name()[: -len("_cpu")]

    do_matmul = make_loop_matmul(f"do_{name}", CONV2_FILTERS, CONV1_DIM)

    gemmini = rename(cpu, name)
    gemmini = replace(gemmini, "for i1 in _:_", do_matmul)

    return gemmini


def sched_resadd(cpu):
    name = cpu.name()[: -len("_cpu")]

    do_resadd = make_loop_resadd(f"do_{name}", CONV1_DIM, CONV1_FILTERS)

    gemmini = rename(cpu, name)
    gemmini = replace(gemmini, "for i1 in _:_", do_resadd)
    gemmini = fence_after(gemmini, f"do_{name}(_)")

    return gemmini


conv1 = sched_conv_nofence(conv1_cpu, INPUT_DIM, 1, CONV1_FILTERS, 3)
conv2 = sched_conv_nofence(conv2_cpu, CONV1_DIM, CONV1_FILTERS, CONV2_FILTERS, 3, act=True)
conv3 = sched_conv(conv3_cpu, CONV2_DIM, CONV2_FILTERS, CONV3_FILTERS, 3, act=True)
nlb_qkv_conv = sched_conv_nofence(
    nlb_qkv_conv_cpu, CONV1_DIM, CONV1_FILTERS, CONV2_FILTERS, 1
)
nlb_out_conv = sched_conv_nofence(
    nlb_out_conv_cpu, CONV1_DIM, CONV2_FILTERS, CONV1_FILTERS, 1
)

fc1 = sched_fc_nofence(fc1_cpu, CONV3_FILTERS, CONV3_DIM, CONV3_DIM, FC1_UNITS, act=True)
fc2 = sched_fc(fc2_cpu, FC1_UNITS, 1, 1, FC2_UNITS, act=True)
fc3 = sched_fc_cpu(fc3_cpu, act=True)
fc4 = sched_fc_cpu(fc4_cpu, act=True)
fc_output = sched_fc_cpu(fc_output_cpu)

matmul_theta_phi = sched_matmul_trans_b_nofence(matmul_theta_phi_cpu)
nlb_softmax = sched_softmax(nlb_softmax_cpu)
matmul_attention_g = sched_matmul_nofence(matmul_attention_g_cpu)
resadd_relu = sched_resadd(resadd_relu_cpu)


def schedule_nlb():
    gemmini = rename(nlb_cpu, "nlb")
    gemmini = call_eqv(gemmini, "nlb_qkv_conv_cpu(_) #0", nlb_qkv_conv)
    gemmini = call_eqv(gemmini, "nlb_qkv_conv_cpu(_) #0", nlb_qkv_conv)
    gemmini = call_eqv(gemmini, "nlb_qkv_conv_cpu(_) #0", nlb_qkv_conv)
    gemmini = fence_after(gemmini, "nlb_qkv_conv(_) #2")
    gemmini = call_eqv(gemmini, "matmul_theta_phi_cpu(_)", matmul_theta_phi)
    gemmini = call_eqv(gemmini, "nlb_softmax_cpu(_)", nlb_softmax)
    gemmini = call_eqv(gemmini, "matmul_attention_g_cpu(_)", matmul_attention_g)
    gemmini = call_eqv(gemmini, "nlb_out_conv_cpu(_)", nlb_out_conv)
    gemmini = call_eqv(gemmini, "resadd_relu_cpu(_)", resadd_relu)
    for pat in [
        "nlb_qkv_conv(_) #0",
        "nlb_qkv_conv(_) #0",
        "nlb_qkv_conv(_) #0",
        "matmul_theta_phi(_)",
        "nlb_softmax(_)",
        "matmul_attention_g(_)",
        "nlb_out_conv(_)",
        "resadd_relu(_)",
    ]:
        gemmini = inline(gemmini, pat)
    return gemmini


nlb = schedule_nlb()


def schedule_braggnn():
    gemmini = rename(braggnn_inference_cpu, "braggnn_inference")
    gemmini = unroll_loop(gemmini, "w")
    gemmini = reorder_loops(gemmini, "ch r")
    gemmini = reorder_loops(gemmini, "ch c")
    gemmini = unroll_loop(gemmini, "ch")
    gemmini = unroll_loop(gemmini, "c")
    gemmini = simplify(gemmini)

    gemmini = call_eqv(gemmini, "conv1_cpu(_)", conv1)
    gemmini = call_eqv(gemmini, "nlb_cpu(_)", nlb)
    gemmini = call_eqv(gemmini, "conv2_cpu(_)", conv2)
    gemmini = call_eqv(gemmini, "conv3_cpu(_)", conv3)
    gemmini = call_eqv(gemmini, "fc1_cpu(_)", fc1)
    gemmini = call_eqv(gemmini, "fc2_cpu(_)", fc2)
    gemmini = call_eqv(gemmini, "fc3_cpu(_)", fc3)
    gemmini = call_eqv(gemmini, "fc4_cpu(_)", fc4)
    gemmini = call_eqv(gemmini, "fc_output_cpu(_)", fc_output)

    for name in [
        "conv1",
        "nlb",
        "conv2",
        "conv3",
        "fc1",
        "fc2",
        "fc3",
        "fc4",
        "fc_output",
    ]:
        gemmini = inline(gemmini, f"{name}(_)")
    return simplify(gemmini)


braggnn_inference = schedule_braggnn()


def schedule_eval():
    gemmini = rename(braggnn_eval_cpu, "braggnn_eval")
    gemmini = call_eqv(gemmini, "braggnn_inference_cpu(_)", braggnn_inference)
    gemmini = inline(gemmini, "braggnn_inference(_)")
    gemmini = set_memory(gemmini, "fp32_patch : _", DRAM_STATIC)
    gemmini = set_memory(gemmini, "pred : _", DRAM_STATIC)
    return gemmini


braggnn_eval = schedule_eval()

__all__ = ["braggnn_eval"]
