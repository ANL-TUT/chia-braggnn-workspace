"""BraggNN schedule with the hand-tuned Gemmini calls of braggnn_tune_opus.c.

The same algorithm as braggnn_schedule.py, with the default OPT_* set of
braggnn_tune_opus.c (add-gcp-cluster branch, 34,092 cycles/patch on FireSim
against 39,953 for the Exo build it started from), expressed as Exo instrs so
the result is still checked against braggnn_reference.py:

- the NLB theta / phi / g matmuls (no row tiling) share one fence;
- the three NLB 1x1 convs run as loop_ws matmuls (bias as a D row with stride
  0), and the second and third reuse the input the first left in the
  scratchpad (A = NULL);
- softmax loads its input straight into the accumulator (resadd-style
  A*1 + A*0 with K = 0, the ldB stage marked done) instead of multiplying by
  an 81x81 identity;
- fc1 reads conv3's HWC output directly: its staged weights are permuted once
  (rearrange_dim), so the per-patch flatten is gone.

The instrs are gemmini_fast.py's make_loop_conv1x1_matmul, make_loop_softmax_load
and make_loop_matmul_fc_hwc. Their C relies on Gemmini features beyond what
braggnn_schedule.py uses: LOOP_WS's skip bits (rs2 bits 3..7) and A reuse
across consecutive loop_ws calls.
"""

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
    inline,
    insert_noop_call,
    rearrange_dim,
    rename,
    reorder_stmts,
    replace,
)
from gemmini import (
    fence,
    make_loop_conv_ws,
    make_loop_matmul,
    make_loop_matmul_fc,
    make_loop_matmul_trans_b,
    make_loop_resadd,
)
from gemmini_fast import (
    make_loop_conv1x1_matmul,
    make_loop_matmul_fc_hwc,
    make_loop_softmax_load,
)


def fence_after(p, pattern):
    return insert_noop_call(p, p.find(pattern).after(), fence, [])


# ── per-operator schedules ────────────────────────────────────────────────


def sched_conv(cpu, in_dim, in_ch, out_ch, k, act=False, with_fence=True):
    name = cpu.name()[: -len("_cpu")]
    do_conv = make_loop_conv_ws(f"do_{name}", in_dim, in_ch, out_ch, k, act)
    gemmini = rename(cpu, name)
    gemmini = replace(gemmini, "for orow in _:_", do_conv)
    if with_fence:
        gemmini = fence_after(gemmini, f"do_{name}(_)")
    return gemmini


def sched_fc(cpu, in_ch, in_h, in_w, out_features, act=False):
    name = cpu.name()[: -len("_cpu")]
    do_fc = make_loop_matmul_fc(f"do_{name}", in_ch, in_h, in_w, out_features, act)
    gemmini = rename(cpu, name)
    gemmini = replace(gemmini, "for j in _:_", do_fc)
    gemmini = fence_after(gemmini, f"do_{name}(_)")
    return gemmini


def sched_qkv(name, reuse_a):
    # No fence: schedule_nlb puts one after the third.
    do_qkv = make_loop_conv1x1_matmul(
        f"do_{name}", CONV1_DIM, CONV1_FILTERS, CONV2_FILTERS, reuse_a
    )
    gemmini = rename(nlb_qkv_conv_cpu, name)
    return replace(gemmini, "for orow in _:_", do_qkv)


def sched_whole(cpu, do_op):
    # One loop over all 81 rows (no NLB row tiling).
    name = cpu.name()[: -len("_cpu")]
    gemmini = rename(cpu, name)
    gemmini = replace(gemmini, "for i1 in _:_", do_op)
    return fence_after(gemmini, f"{do_op.name()}(_)")


conv1 = sched_conv(conv1_cpu, INPUT_DIM, 1, CONV1_FILTERS, 3)
conv2 = sched_conv(conv2_cpu, CONV1_DIM, CONV1_FILTERS, CONV2_FILTERS, 3, act=True)
conv3 = sched_conv(conv3_cpu, CONV2_DIM, CONV2_FILTERS, CONV3_FILTERS, 3, act=True)
nlb_out_conv = sched_conv(nlb_out_conv_cpu, CONV1_DIM, CONV2_FILTERS, CONV1_FILTERS, 1)

nlb_qkv_load = sched_qkv("nlb_qkv_load", reuse_a=False)
nlb_qkv_reuse = sched_qkv("nlb_qkv_reuse", reuse_a=True)

fc2 = sched_fc(fc2_cpu, FC1_UNITS, 1, 1, FC2_UNITS, act=True)
fc3 = sched_fc(fc3_cpu, FC2_UNITS, 1, 1, FC3_UNITS, act=True)
fc4 = sched_fc(fc4_cpu, FC3_UNITS, 1, 1, FC4_UNITS, act=True)
fc_output = sched_fc(fc_output_cpu, FC4_UNITS, 1, 1, OUTPUT_UNITS)

matmul_theta_phi = sched_whole(
    matmul_theta_phi_cpu,
    make_loop_matmul_trans_b("do_matmul_theta_phi", CONV2_FILTERS, CONV1_DIM),
)
nlb_softmax = sched_whole(
    nlb_softmax_cpu,
    make_loop_softmax_load("do_nlb_softmax", CONV1_DIM, SOFTMAX_MAX_SHIFT),
)
matmul_attention_g = sched_whole(
    matmul_attention_g_cpu,
    make_loop_matmul("do_matmul_attention_g", CONV2_FILTERS, CONV1_DIM),
)
resadd_relu = sched_whole(
    resadd_relu_cpu, make_loop_resadd("do_resadd_relu", CONV1_DIM, CONV1_FILTERS)
)


def schedule_nlb():
    gemmini = rename(nlb_cpu, "nlb")
    gemmini = call_eqv(gemmini, "nlb_qkv_conv_cpu(_) #0", nlb_qkv_load)
    gemmini = call_eqv(gemmini, "nlb_qkv_conv_cpu(_) #0", nlb_qkv_reuse)
    gemmini = call_eqv(gemmini, "nlb_qkv_conv_cpu(_) #0", nlb_qkv_reuse)
    gemmini = fence_after(gemmini, "nlb_qkv_reuse(_) #1")
    gemmini = call_eqv(gemmini, "matmul_theta_phi_cpu(_)", matmul_theta_phi)
    gemmini = call_eqv(gemmini, "nlb_softmax_cpu(_)", nlb_softmax)
    gemmini = call_eqv(gemmini, "matmul_attention_g_cpu(_)", matmul_attention_g)
    gemmini = call_eqv(gemmini, "nlb_out_conv_cpu(_)", nlb_out_conv)
    gemmini = call_eqv(gemmini, "resadd_relu_cpu(_)", resadd_relu)
    return gemmini


nlb = schedule_nlb()


def schedule_braggnn():
    # fc1 stays a call here: schedule_eval schedules it together with the
    # weight staging it needs.
    gemmini = rename(braggnn_inference_cpu, "braggnn_inference")
    gemmini = call_eqv(gemmini, "conv1_cpu(_)", conv1)
    gemmini = call_eqv(gemmini, "nlb_cpu(_)", nlb)
    gemmini = call_eqv(gemmini, "conv2_cpu(_)", conv2)
    gemmini = call_eqv(gemmini, "conv3_cpu(_)", conv3)
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
    gemmini = inline(gemmini, "fc1_cpu(_)")
    # Stage fc1's weights once in (unit, row, col, ch) order, the order of
    # conv3's HWC output, so fc1 can read that output without the flatten.
    gemmini = rearrange_dim(gemmini, "fc1_weights_ : _", [0, 2, 3, 1])
    # Move fc1_out's allocation above the flatten so the flatten and the fc1
    # loop are adjacent, then replace both with the one instr.
    flatten = gemmini.find_loop("ch")
    gemmini = reorder_stmts(gemmini, flatten.expand(0, 1))
    flatten = gemmini.find_loop("ch")
    do_fc1 = make_loop_matmul_fc_hwc(
        "do_fc1", CONV3_DIM, CONV3_DIM, CONV3_FILTERS, FC1_UNITS, act=True
    )
    gemmini = replace(gemmini, flatten.expand(0, 1), do_fc1)
    gemmini = fence_after(gemmini, "do_fc1(_)")
    return gemmini


braggnn_eval = schedule_eval()

__all__ = ["braggnn_eval"]
