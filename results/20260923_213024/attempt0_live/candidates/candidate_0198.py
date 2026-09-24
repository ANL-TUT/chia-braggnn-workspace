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
    from exo.API_scheduling import unroll_loop, mult_dim, reuse_buffer
    from exo.API_cursors import AllocCursor, ForCursor
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

    # Find allocations in gemmini to dynamically locate the buffers for NCHW flatten copy.
    alloc_names = []
    def walk(body):
        for stmt in body:
            if isinstance(stmt, AllocCursor):
                try:
                    alloc_names.append(stmt.name())
                except Exception:
                    alloc_names.append(stmt.name)
            if hasattr(stmt, "body"):
                try:
                    walk(stmt.body())
                except Exception:
                    pass
    walk(gemmini.body())
    
    conv3_alloc = None
    fc1_alloc = None
    for name in alloc_names:
        if "conv3" in name and ("out" in name or "res" in name):
            conv3_alloc = name
        if "fc1" in name and ("in" in name or "inp" in name):
            fc1_alloc = name

    # Eliminate the NCHW flatten CPU copy loop between conv3 and fc1 by reusing the buffer.
    if conv3_alloc and fc1_alloc:
        try:
            gemmini = mult_dim(gemmini, f"{fc1_alloc} : _", 1, 2)
            gemmini = mult_dim(gemmini, f"{fc1_alloc} : _", 0, 1)
            gemmini = mult_dim(gemmini, f"{conv3_alloc} : _", 1, 2)
            gemmini = mult_dim(gemmini, f"{conv3_alloc} : _", 0, 1)
            gemmini = reuse_buffer(gemmini, fc1_alloc, conv3_alloc)
            gemmini = simplify(gemmini)
        except Exception as e:
            print(f"Failed to reuse {fc1_alloc} and {conv3_alloc}: {e}")

    # Optimize and reuse intermediate buffers in NLB to reduce DRAM footprint and improve L1 cache locality.
    theta_phi_alloc = None
    softmax_out_alloc = None
    q_alloc = None
    attention_g_alloc = None
    
    for name in alloc_names:
        if "theta_phi" in name:
            theta_phi_alloc = name
        elif "softmax_out" in name:
            softmax_out_alloc = name
        elif "nlb" in name and "q" in name:
            q_alloc = name
        elif "attention_g" in name:
            attention_g_alloc = name

    if theta_phi_alloc and softmax_out_alloc:
        try:
            gemmini = reuse_buffer(gemmini, softmax_out_alloc, theta_phi_alloc)
            gemmini = simplify(gemmini)
        except Exception as e:
            print(f"Failed to reuse {softmax_out_alloc} and {theta_phi_alloc}: {e}")

    if q_alloc and attention_g_alloc:
        try:
            gemmini = reuse_buffer(gemmini, attention_g_alloc, q_alloc)
            gemmini = simplify(gemmini)
        except Exception as e:
            print(f"Failed to reuse {attention_g_alloc} and {q_alloc}: {e}")
    
    # Target and dynamically unroll all CPU loops inside the patch loop 'p' completely.
    # This avoids touching any weight-staging loops outside 'p' while maximizing performance.
    p_loop = gemmini.find_loop("p")
    inner_loop_names = set()
    def find_inner_loop_names(body):
        for stmt in body:
            if isinstance(stmt, ForCursor):
                inner_loop_names.add(stmt.name())
                find_inner_loop_names(stmt.body())
            elif hasattr(stmt, "body"):
                try:
                    find_inner_loop_names(stmt.body())
                except Exception:
                    pass
    find_inner_loop_names(p_loop.body())

    for name in sorted(list(inner_loop_names)):
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
