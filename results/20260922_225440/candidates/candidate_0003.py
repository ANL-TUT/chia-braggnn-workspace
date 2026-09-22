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
from exo import DRAM, seq, instr
from exo.core.LoopIR import LoopIR
from exo.platforms.gemmini import acc_scale, clamp
from exo.libs.externs import relu, select
from exo.API_scheduling import (
    unroll_loop,
    fuse,
    reorder_loops,
    set_memory,
    mult_dim,
    rearrange_dim,
    stage_mem,
    expand_dim,
    lift_alloc,
    fission,
    remove_loop,
    delete_buffer,
    bind_expr,
    bind_map,
    mult_loops,
)

def fence_after(p, pattern):
    return insert_noop_call(p, p.find(pattern).after(), fence, [])


def _blocks(n):
    tiles = (n + 15) // 16
    return tiles, tiles * 16 - n


def _gemm_loop_matmul_fc(in_features, out_features, act):
    J, pad_J = _blocks(out_features)
    K, pad_K = _blocks(in_features)
    gemm_act = "RELU" if act else "NO_ACTIVATION"
    scale = "({scale}[0] * {post_scale}[0])" if act else "{scale}[0]"
    return (
        "gemmini_extended_config_ex(WEIGHT_STATIONARY, (0), 0, 1, (0), (1));\n"
        + f"gemmini_extended_config_st((1), {gemm_act}, "
        + scale
        + ");\n"
        + f"gemmini_extended3_config_ld(({in_features}), 1.0f, false, (0));\n"
        + f"gemmini_extended3_config_ld(({in_features}), 1.0f, false, (1));\n"
        + "gemmini_extended3_config_ld((1), 1.0f, false, (2));\n"
        + f"gemmini_loop_ws(1, {J}, {K}, 15, {pad_J}, {pad_K}, "
        "{inp}, {weights}, {bias}, {output}, "
        f"{in_features}, {in_features}, {out_features}, {out_features}, "
        f"0, 1, 0, 0, 1, {gemm_act}, 1, 1, 0);"
    )


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
nlb_out_conv = sched_conv(nlb_out_conv_cpu, CONV1_DIM, CONV2_FILTERS, CONV1_FILTERS, 1)

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
    return gemmini


nlb = schedule_nlb()


def schedule_braggnn():
    gemmini = rename(braggnn_inference_cpu, "braggnn_inference")
    gemmini = call_eqv(gemmini, "conv1_cpu(_)", conv1)
    gemmini = call_eqv(gemmini, "nlb_cpu(_)", nlb)
    gemmini = call_eqv(gemmini, "conv2_cpu(_)", conv2)
    gemmini = call_eqv(gemmini, "conv3_cpu(_)", conv3)
    # We do NOT scheduled fc1_cpu here; we schedule it flat at the eval level!
    gemmini = call_eqv(gemmini, "fc2_cpu(_)", fc2)
    gemmini = call_eqv(gemmini, "fc3_cpu(_)", fc3)
    gemmini = call_eqv(gemmini, "fc4_cpu(_)", fc4)
    gemmini = call_eqv(gemmini, "fc_output_cpu(_)", fc_output)
    return gemmini


braggnn_inference = schedule_braggnn()


def find_alloc_or_arg_by_prefix(proc, prefix):
    for arg in proc.args:
        if arg.name.startswith(prefix):
            return arg.name
    def search(stmts):
        for stmt in stmts:
            if isinstance(stmt, LoopIR.Alloc):
                if stmt.name.startswith(prefix):
                    return stmt.name
            elif isinstance(stmt, LoopIR.For):
                res = search(stmt.body)
                if res: return res
            elif isinstance(stmt, LoopIR.If):
                res = search(stmt.body)
                if res: return res
                if stmt.orelse:
                    res = search(stmt.orelse)
                    if res: return res
        return None
    return search(proc.body)


def find_loops_recursive(stmts, inside_patch_loop=False):
    loops = []
    for stmt in stmts:
        if isinstance(stmt, LoopIR.For):
            if inside_patch_loop:
                loops.append(stmt)
            loops.extend(find_loops_recursive(stmt.body, inside_patch_loop=True))
        elif isinstance(stmt, LoopIR.If):
            loops.extend(find_loops_recursive(stmt.body, inside_patch_loop))
            if stmt.orelse:
                loops.extend(find_loops_recursive(stmt.orelse, inside_patch_loop))
    return loops


def optimize_cpu_loops(proc):
    while True:
        all_loops = find_loops_recursive(proc.body, inside_patch_loop=False)
        unrolled_any = False
        for stmt in all_loops:
            if isinstance(stmt.hi, LoopIR.Const) and isinstance(stmt.lo, LoopIR.Const):
                bound = stmt.hi.val - stmt.lo.val
                if bound <= 16:
                    try:
                        loop_cursor = proc.find(f"for {stmt.name} in seq({stmt.lo.val}, {stmt.hi.val}): _")
                        proc = unroll_loop(proc, loop_cursor)
                        proc = simplify(proc)
                        unrolled_any = True
                        break
                    except Exception:
                        pass
        if not unrolled_any:
            break
    return proc


def schedule_eval():
    gemmini = rename(braggnn_eval_cpu, "braggnn_eval")
    gemmini = call_eqv(gemmini, "braggnn_inference_cpu(_)", braggnn_inference)
    gemmini = inline(gemmini, "braggnn_inference(_)")

    # Inline fc1_cpu
    gemmini = inline(gemmini, "fc1_cpu(_)")

    # Find buffer names dynamically
    fc1_weights_name = find_alloc_or_arg_by_prefix(gemmini, "fc1_weights")
    conv3_out_name = find_alloc_or_arg_by_prefix(gemmini, "conv3_out")

    # Fold fc1 weights: [16, 8, 5, 5] -> [16, 200]
    gemmini = mult_dim(gemmini, fc1_weights_name, 2, 25)
    gemmini = simplify(gemmini)
    gemmini = mult_dim(gemmini, fc1_weights_name, 1, 200)
    gemmini = simplify(gemmini)

    # Fold conv3 output buffer: [8, 5, 5] -> [200]
    gemmini = mult_dim(gemmini, conv3_out_name, 1, 25)
    gemmini = simplify(gemmini)
    gemmini = mult_dim(gemmini, conv3_out_name, 0, 200)
    gemmini = simplify(gemmini)

    # Merge inner loops of inlined fc1
    kch_loop = gemmini.find_loop("kch")
    gemmini = mult_loops(gemmini, kch_loop, "k_combined")
    gemmini = simplify(gemmini)
    k_comb_loop = gemmini.find_loop("k_combined")
    gemmini = mult_loops(gemmini, k_comb_loop, "k")
    gemmini = simplify(gemmini)

    # Replace with flat matmul loop instruction
    do_fc1 = make_loop_matmul_fc_flat("do_fc1", 200, 16, act=True)
    j_loop = gemmini.find_loop("j")
    gemmini = replace(gemmini, j_loop, do_fc1)
    gemmini = fence_after(gemmini, "do_fc1(_)")

    # Optimize and unroll all timed CPU loops inside the patch loop
    gemmini = optimize_cpu_loops(gemmini)

    return gemmini


braggnn_eval = schedule_eval()
# EVOLVE-BLOCK-END


__all__ = ["braggnn_eval"]
