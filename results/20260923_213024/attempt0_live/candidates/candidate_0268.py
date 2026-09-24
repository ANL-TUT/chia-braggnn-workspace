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
def fence_after(p, pattern):
    return insert_noop_call(p, p.find(pattern).after(), fence, [])


def sched_conv(cpu, in_dim, in_ch, out_ch, k, act=False, fence=True, suffix=""):
    name = cpu.name()[: -len("_cpu")] + suffix

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
nlb_qkv_conv = sched_conv(nlb_qkv_conv_cpu, CONV1_DIM, CONV1_FILTERS, CONV2_FILTERS, 1)
nlb_qkv_conv_nofence = sched_conv(nlb_qkv_conv_cpu, CONV1_DIM, CONV1_FILTERS, CONV2_FILTERS, 1, fence=False, suffix="_nofence")
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
    gemmini = call_eqv(gemmini, "nlb_qkv_conv_cpu(_) #0", nlb_qkv_conv_nofence)
    gemmini = call_eqv(gemmini, "nlb_qkv_conv_cpu(_) #0", nlb_qkv_conv_nofence)
    gemmini = call_eqv(gemmini, "nlb_qkv_conv_cpu(_) #0", nlb_qkv_conv_nofence)
    gemmini = fence_after(gemmini, "nlb_qkv_conv_nofence(_) #2")
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
    from exo.API_scheduling import unroll_loop, lift_alloc, fuse, reorder_loops
    gemmini = rename(braggnn_eval_cpu, "braggnn_eval")
    gemmini = call_eqv(gemmini, "braggnn_inference_cpu(_)", braggnn_inference)
    gemmini = inline(gemmini, "braggnn_inference(_)")
    
    # Inline all scheduled layer procedures to eliminate function call overhead inside the patch loop.
    for inline_pat in [
        "conv1(_)", "nlb(_)", "conv2(_)", "conv3(_)",
        "fc1(_)", "fc2(_)", "fc3(_)", "fc4(_)", "fc_output(_)",
        "nlb_qkv_conv_nofence(_)", "matmul_theta_phi(_)",
        "nlb_softmax(_)", "matmul_attention_g(_)",
        "nlb_out_conv(_)", "resadd_relu(_)"
    ]:
        while True:
            try:
                gemmini = inline(gemmini, inline_pat)
            except Exception:
                break

    # Lift allocations to the top level of braggnn_eval to make fusion possible and save overhead
    for alloc_name in ["patch", "quantized_patch", "conv1_out", "nlb_out", "conv2_out", "conv3_out", "fc1_in", "fc1_out", "fc2_out", "fc3_out", "fc4_out", "res"]:
        for _ in range(5):
            try:
                gemmini = lift_alloc(gemmini, alloc_name)
            except Exception:
                break

    # Reorder the NCHW flattening loops so that the memory access is contiguous (ch is innermost)
    try:
        gemmini = reorder_loops(gemmini, "ch", "r")
        gemmini = reorder_loops(gemmini, "ch", "c")
    except Exception:
        pass

    # Fuse input copying and quantization loops to save CPU overhead
    try:
        loop_irow = gemmini.find_loop("irow")
        loop_h = gemmini.find_loop("h")
        gemmini = fuse(gemmini, loop_irow, loop_h)
        loop_icol = gemmini.find_loop("icol")
        loop_w = gemmini.find_loop("w")
        gemmini = fuse(gemmini, loop_icol, loop_w)
    except Exception:
        pass

    # Target specific CPU loops inside the patch loop 'p' and unroll them completely.
    # This avoids touching any weight-staging loops outside 'p'.
    for name in ["icol", "irow", "w", "h", "c", "r", "ch", "k"]:
        while True:
            try:
                p_loop = gemmini.find_loop("p")
                loop = p_loop.find_loop(name)
                gemmini = unroll_loop(gemmini, loop)
            except Exception:
                break
                
    gemmini = simplify(gemmini)
    return gemmini


braggnn_eval = schedule_eval()
# EVOLVE-BLOCK-END


__all__ = ["braggnn_eval"]
