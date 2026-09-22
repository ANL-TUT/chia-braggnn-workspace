from __future__ import annotations

from typing import ClassVar

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
from exo.API_cursors import ForCursor, InvalidCursor
from exo.API_scheduling import (
    autofission,
    call_eqv,
    delete_buffer,
    delete_config,
    divide_dim,
    divide_loop,
    expand_dim,
    fission,
    fuse,
    inline,
    inline_assign,
    inline_window,
    insert_noop_call,
    lift_alloc,
    rearrange_dim,
    remove_loop,
    rename,
    reorder_loops,
    reorder_stmts,
    replace,
    resize_dim,
    set_memory,
    simplify,
    stage_mem,
    unroll_loop,
)
from exo.libs.memories import GEMM_ACCUM, GEMM_SCRATCH
from exo.platforms.gemmini import old_fission_after
from gemmini import (
    DRAM_ALIGNED,
    fence,
    ld_acc_i32_repeat,
    ld_acc_i32_repeat_v2,
    ld_i8_block_id1,
    ld_i8_block_id1_v2,
    ld_i8_block_id2,
    ld_i8_block_id2_v2,
    make_loop_conv_ws,
    make_loop_matmul,
    make_loop_matmul_fc,
    make_loop_matmul_trans_b,
    make_loop_resadd,
    make_loop_softmax,
    matmul_acc_i8_trans_b,
    matmul_acc_i8_trans_b_v2,
    st_acc_i8_relu,
    st_acc_i8_relu_v2,
)

NLB_ROW_TILE = 8


class GEMM_SCRATCH_FIXED(GEMM_SCRATCH):
    addresses: ClassVar[dict[str, int]] = {
        "input_tmp": 1,
        "input_tmp_1": 1,
        "weights_tmp": 4096,
        "weights_tmp_1": 4096,
    }

    @classmethod
    def global_(cls):
        return "#include <stdint.h>\n#include <include/gemmini.h>"

    @classmethod
    def alloc(cls, new_name, prim_type, shape, srcinfo):
        try:
            addr = cls.addresses[new_name]
        except KeyError as e:
            raise RuntimeError(f"No fixed scratchpad address for {new_name}") from e
        return f"{prim_type} *{new_name} = ({prim_type} *)(uintptr_t){addr}u;"

    @classmethod
    def free(cls, new_name, prim_type, shape, srcinfo):
        return ""


class GEMM_ACCUM_FIXED(GEMM_ACCUM):
    addresses: ClassVar[dict[str, int]] = {
        "res": 0x80000000,
        "res_1": 0x80000000,
    }

    @classmethod
    def global_(cls):
        return "#include <stdint.h>\n#include <include/gemmini.h>"

    @classmethod
    def alloc(cls, new_name, prim_type, shape, srcinfo):
        try:
            addr = cls.addresses[new_name]
        except KeyError as e:
            raise RuntimeError(f"No fixed accumulator address for {new_name}") from e
        return f"{prim_type} *{new_name} = ({prim_type} *)(uintptr_t)0x{addr:08x}u;"

    @classmethod
    def free(cls, new_name, prim_type, shape, srcinfo):
        return ""


def fence_after(p, pattern):
    return insert_noop_call(p, p.find(pattern).after(), fence, [])


def hoist_config(p, pattern, n_lifts, n_reorders=0):
    p = fission(p, p.find(pattern).after(), n_lifts=n_lifts)
    for _ in range(n_lifts):
        p = remove_loop(p, p.find(pattern).parent())
    for _ in range(n_reorders):
        p = reorder_stmts(p, p.find(pattern).expand(1, 0))
    return p


def hoist_config_after_store(p, pattern, n_lifts, n_reorders=0):
    p = old_fission_after(p, pattern, n_lifts=n_lifts)
    for _ in range(n_lifts):
        parent = p.find(pattern).parent()
        if not isinstance(parent, ForCursor):
            break
        p = remove_loop(p, parent)
    for _ in range(n_reorders):
        p = reorder_stmts(p, p.find(pattern).expand(1, 0))
    return p


def outer_copy_loop(p, pattern, patch_loop="p"):
    c = p.find(pattern)
    while isinstance(c.parent(), ForCursor) and c.parent().name() != patch_loop:
        c = c.parent()
    return c


def lift_weight_transpose(p, buf, patch_loop="p"):
    p = lift_alloc(p, f"{buf}: _", n_lifts=1)
    while True:
        c = outer_copy_loop(p, f"{buf}[_] = _", patch_loop)
        if isinstance(c.prev(), InvalidCursor):
            break
        p = reorder_stmts(p, c.expand(1, 0))
    p = autofission(
        p, outer_copy_loop(p, f"{buf}[_] = _", patch_loop).after(), n_lifts=1
    )
    parent = outer_copy_loop(p, f"{buf}[_] = _", patch_loop).parent()
    if isinstance(parent, ForCursor) and parent.name() == patch_loop:
        p = remove_loop(p, parent)
    return p


def sched_conv(cpu, in_dim, in_ch, out_ch, k, act=False):
    name = cpu.name()[: -len("_cpu")]

    do_conv = make_loop_conv_ws(f"do_{name}", in_dim, in_ch, out_ch, k, act)

    gemmini = rename(cpu, name)
    gemmini = replace(gemmini, "for orow in _:_", do_conv)
    gemmini = fence_after(gemmini, f"do_{name}(_)")

    return gemmini


def sched_conv_fusion(cpu, in_dim, in_ch, out_ch, k):
    name = cpu.name()[: -len("_cpu")]
    out_dim = in_dim - k + 1
    och_blk = min(out_ch, 16)
    kch_blk = min(in_ch, 16)
    assert out_dim <= 16
    assert in_ch % kch_blk == 0
    assert out_ch % och_blk == 0

    ohwi = f"{name}_weights_ohwi"

    gemmini = rename(cpu, name)
    gemmini = stage_mem(
        gemmini,
        "for orow in _:_",
        f"weights[0:{k}, 0:{k}, 0:{in_ch}, 0:{out_ch}]",
        ohwi,
    )
    gemmini = set_memory(gemmini, f"{ohwi}: _", DRAM_ALIGNED)
    gemmini = rearrange_dim(gemmini, f"{ohwi}: _", [0, 1, 3, 2])
    gemmini = simplify(gemmini)
    gemmini = stage_mem(
        gemmini,
        "for orow in _:_",
        f"inp[0:{in_dim}, 0:{in_dim}, 0:{in_ch}]",
        "input_tmp",
    )
    gemmini = stage_mem(
        gemmini,
        "for orow in _:_",
        f"{ohwi}[0:{k}, 0:{k}, 0:{out_ch}, 0:{in_ch}]",
        "weights_tmp",
    )
    gemmini = simplify(gemmini)
    gemmini = reorder_stmts(
        gemmini, gemmini.find("weights_tmp: _").expand(1, 0)
    )

    gemmini = divide_loop(
        gemmini, "kch", kch_blk, ["kch_o", "kch_i"], perfect=True
    )
    gemmini = divide_loop(
        gemmini, "och", och_blk, ["och_o", "och_i"], perfect=True
    )
    gemmini = reorder_loops(gemmini, "ocol och_o")
    gemmini = expand_dim(gemmini, "res: _", och_blk, "och_i")
    gemmini = expand_dim(gemmini, "res: _", out_dim, "ocol")
    gemmini = expand_dim(gemmini, "res: _", out_ch // och_blk, "och_o")
    gemmini = expand_dim(gemmini, "res: _", out_dim, "orow")
    gemmini = lift_alloc(gemmini, "res: _", n_lifts=4)
    for _ in range(2):
        gemmini = reorder_stmts(gemmini, gemmini.find("res: _").expand(1, 0))
    gemmini = fission(gemmini, gemmini.find("res[_] = _").after(), n_lifts=4)
    gemmini = fission(gemmini, gemmini.find_loop("krow").after(), n_lifts=4)

    for loop in ("krow", "kcol", "kch_o"):
        gemmini = reorder_loops(gemmini, "och_i " + loop)
        gemmini = reorder_loops(gemmini, "ocol " + loop)

    gemmini = divide_loop(
        gemmini, "i3 #1", kch_blk, ["i3_o", "i3_i"], perfect=True
    )
    gemmini = divide_dim(gemmini, "weights_tmp: _", 3, kch_blk)
    gemmini = divide_loop(
        gemmini, "i2 #2", och_blk, ["i2_o", "i2_i"], perfect=True
    )
    gemmini = divide_dim(gemmini, "weights_tmp: _", 2, och_blk)
    gemmini = divide_loop(
        gemmini, "i2 #1", kch_blk, ["i2_o", "i2_i"], perfect=True
    )
    gemmini = divide_dim(gemmini, "input_tmp: _", 2, kch_blk)
    gemmini = simplify(gemmini)
    gemmini = rearrange_dim(gemmini, "input_tmp: _", [0, 2, 1, 3])
    gemmini = rearrange_dim(
        gemmini, "weights_tmp: _", [0, 1, 2, 4, 3, 5]
    )
    if och_blk < 16:
        gemmini = resize_dim(gemmini, "res: _", 3, 16, 0)
        gemmini = simplify(gemmini)

    gemmini = set_memory(gemmini, "input_tmp: _", GEMM_SCRATCH_FIXED)
    gemmini = set_memory(gemmini, "weights_tmp: _", GEMM_SCRATCH_FIXED)
    gemmini = set_memory(gemmini, "res: _", GEMM_ACCUM_FIXED)

    gemmini = replace(gemmini, "for i1 in _:_ #1", ld_i8_block_id1)
    gemmini = call_eqv(gemmini, ld_i8_block_id1, ld_i8_block_id1_v2)
    gemmini = inline(gemmini, ld_i8_block_id1_v2)
    gemmini = inline_window(gemmini, "src = _")
    gemmini = inline_window(gemmini, "dst = _")
    gemmini = simplify(gemmini)
    gemmini = replace(gemmini, "for i2_i in _:_ #0", ld_i8_block_id2)
    gemmini = call_eqv(gemmini, ld_i8_block_id2, ld_i8_block_id2_v2)
    gemmini = inline(gemmini, ld_i8_block_id2_v2)
    gemmini = inline_window(gemmini, "src = _")
    gemmini = inline_window(gemmini, "dst = _")
    gemmini = simplify(gemmini)

    gemmini = inline_assign(gemmini, "w_s = _")
    gemmini = inline_assign(gemmini, "i_s = _")
    gemmini = delete_buffer(gemmini, "w_s: _")
    gemmini = delete_buffer(gemmini, "i_s: _")
    gemmini = replace(gemmini, "for ocol in _:_ #0", ld_acc_i32_repeat)
    gemmini = call_eqv(gemmini, ld_acc_i32_repeat, ld_acc_i32_repeat_v2)
    gemmini = inline(gemmini, ld_acc_i32_repeat_v2)
    gemmini = inline_window(gemmini, "src = _")
    gemmini = inline_window(gemmini, "dst = _")
    gemmini = simplify(gemmini)
    gemmini = replace(gemmini, "for ocol in _:_ #0", matmul_acc_i8_trans_b)
    gemmini = call_eqv(
        gemmini, matmul_acc_i8_trans_b, matmul_acc_i8_trans_b_v2
    )
    gemmini = inline(gemmini, matmul_acc_i8_trans_b_v2)
    gemmini = inline_window(gemmini, "A = _")
    gemmini = inline_window(gemmini, "B = _")
    gemmini = inline_window(gemmini, "C = _")
    gemmini = simplify(gemmini)
    gemmini = replace(gemmini, "for ocol in _:_ #0", st_acc_i8_relu)
    gemmini = call_eqv(gemmini, st_acc_i8_relu, st_acc_i8_relu_v2)
    gemmini = inline(gemmini, st_acc_i8_relu_v2)
    gemmini = inline_window(gemmini, "src = _")
    gemmini = inline_window(gemmini, "dst = _")
    gemmini = simplify(gemmini)

    if out_ch < 16:
        for _ in range(3):
            gemmini = unroll_loop(gemmini, "och_o")

    gemmini = hoist_config(gemmini, "config_ld_i8_block_id1(_)", 1)
    gemmini = hoist_config(gemmini, "config_ld_i8_block_id2(_)", 3, 1)
    gemmini = hoist_config(
        gemmini, "config_ld_repeat(_)", 2 if out_ch > 16 else 1, 2
    )
    gemmini = hoist_config(
        gemmini, "config_matmul_trans_b(_)", 5 if out_ch > 16 else 4, 3
    )
    gemmini = hoist_config_after_store(
        gemmini, "config_st_i8_relu(_)", 2 if out_ch > 16 else 1, 4
    )

    for loop in ("kch_o", "kcol"):
        gemmini = unroll_loop(gemmini, loop)
    if out_ch < 16:
        gemmini = unroll_loop(gemmini, "krow")

    gemmini = fuse(gemmini, "orow #1", "orow #2")
    gemmini = fuse(gemmini, "orow", "orow #1")
    if out_ch > 16:
        gemmini = fuse(gemmini, "och_o #1", "och_o #2")

    gemmini = insert_noop_call(gemmini, gemmini.find_loop("orow").after(), fence, [])

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
conv2 = sched_conv_fusion(
    conv2_cpu, CONV1_DIM, CONV1_FILTERS, CONV2_FILTERS, 3
)
conv3 = sched_conv_fusion(
    conv3_cpu, CONV2_DIM, CONV2_FILTERS, CONV3_FILTERS, 3
)
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
    return gemmini


nlb = schedule_nlb()


def schedule_braggnn():
    gemmini = rename(braggnn_inference_cpu, "braggnn_inference")
    gemmini = call_eqv(gemmini, "conv1_cpu(_)", conv1)
    gemmini = call_eqv(gemmini, "nlb_cpu(_)", nlb)
    gemmini = call_eqv(gemmini, "conv2_cpu(_)", conv2)
    gemmini = call_eqv(gemmini, "conv3_cpu(_)", conv3)
    gemmini = inline(gemmini, "conv2(_)")
    gemmini = inline(gemmini, "conv3(_)")
    gemmini = delete_config(gemmini, "config_matmul_trans_b(_) #1")
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
    for buf in ("conv2_weights_ohwi", "conv3_weights_ohwi"):
        gemmini = lift_weight_transpose(gemmini, buf)
    for buf in (
        "conv2_bias_",
        "conv3_bias_",
        "nlb_out",
        "conv2_out",
        "conv3_out",
    ):
        gemmini = set_memory(gemmini, f"{buf}: _", DRAM_ALIGNED)
    return gemmini


braggnn_eval = schedule_eval()

__all__ = ["braggnn_eval"]
