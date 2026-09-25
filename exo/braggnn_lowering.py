"""Loop-nest lowering of a conv layer, as a fixed, pre-checked building block.

`lower_conv(conv_cpu, in_dim, in_ch, out_ch, k)` schedules a ReLU conv spec
from braggnn_reference.py (conv2_cpu / conv3_cpu) onto Gemmini's low-level
instrs instead of the `make_loop_conv_ws` hardware-loop macro: weights
re-laid out once, input staged to the scratchpad once per kernel column, the
output rows and columns merged into one axis and tiled by 16 so each matmul
fills the array's rows, results kept in the accumulator and stored with the
activation. It is the recipe of braggnn_schedule_fusion.py (conv2 / conv3),
moved here so a schedule can use it with one call and it always passes Exo's
checks. Read-only for the schedule search.
"""

from __future__ import annotations

from typing import ClassVar

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
    inline,
    inline_assign,
    inline_window,
    insert_noop_call,
    lift_alloc,
    mult_dim,
    mult_loops,
    rearrange_dim,
    remove_loop,
    rename,
    reorder_loops,
    reorder_stmts,
    replace,
    resize_dim,
    rewrite_expr,
    set_memory,
    simplify,
    stage_mem,
)
from exo.libs.memories import DRAM_STATIC, GEMM_ACCUM, GEMM_SCRATCH
from exo.platforms.gemmini import old_fission_after
from gemmini import (
    fence,
    ld_acc_i32_repeat,
    ld_acc_i32_repeat_v2,
    ld_i8_block_strided_id1,
    ld_i8_block_strided_id1_v2,
    ld_i8_block_strided_id2,
    ld_i8_block_strided_id2_v2,
    make_loop_conv_ws,
    make_loop_matmul,
    make_loop_matmul_fc,
    make_loop_matmul_trans_b,
    make_loop_resadd,
    make_loop_softmax,
    matmul_acc_i8_trans_b,
    matmul_acc_i8_trans_b_v2,
    st_acc_i8_act,
    st_acc_i8_act_v2,
)


class GEMM_SCRATCH_FIXED(GEMM_SCRATCH):
    addresses: ClassVar[dict[str, int]] = {
        "input_tmp": 1,
        "input_tmp_1": 1,
        "weights_tmp": 4096,
        "weights_tmp_1": 6144,
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


def hoist_acc_scale(p, n_lifts):
    p = lift_alloc(p, "tmp_scale : _", n_lifts=n_lifts)
    while not isinstance(p.find("tmp_scale = _").prev(), InvalidCursor):
        p = reorder_stmts(p, p.find("tmp_scale = _").expand(1, 0))
    p = fission(p, p.find("tmp_scale = _").after(), n_lifts=n_lifts)
    for _ in range(n_lifts):
        p = remove_loop(p, p.find("tmp_scale = _").parent())
    return p


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


def lift_out_of_patch_loop(p, pattern, patch_loop="p"):
    while True:
        c = outer_copy_loop(p, pattern, patch_loop)
        if isinstance(c.prev(), InvalidCursor):
            break
        p = reorder_stmts(p, c.expand(1, 0))
    p = autofission(
        p, outer_copy_loop(p, pattern, patch_loop).after(), n_lifts=1
    )
    parent = outer_copy_loop(p, pattern, patch_loop).parent()
    if isinstance(parent, ForCursor) and parent.name() == patch_loop:
        p = remove_loop(p, parent)
    return p


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
    gemmini = set_memory(gemmini, f"{ohwi}: _", DRAM_STATIC)
    gemmini = rearrange_dim(gemmini, f"{ohwi}: _", [0, 1, 3, 2])
    gemmini = simplify(gemmini)
    gemmini = stage_mem(
        gemmini,
        "for orow in _:_",
        f"{ohwi}[0:{k}, 0:{k}, 0:{out_ch}, 0:{in_ch}]",
        "weights_tmp",
    )
    gemmini = simplify(gemmini)

    gemmini = divide_loop(
        gemmini, "kch", kch_blk, ["kch_o", "kch_i"], perfect=True
    )
    gemmini = divide_loop(
        gemmini, "och", och_blk, ["och_o", "och_i"], perfect=True
    )
    gemmini = expand_dim(gemmini, "res: _", och_blk, "och_i")
    gemmini = expand_dim(gemmini, "res: _", out_dim, "ocol")
    gemmini = expand_dim(gemmini, "res: _", out_ch // och_blk, "och_o")
    gemmini = expand_dim(gemmini, "res: _", out_dim, "orow")
    gemmini = lift_alloc(gemmini, "res: _", n_lifts=4)
    gemmini = fission(gemmini, gemmini.find("res[_] = _").after(), n_lifts=4)
    gemmini = fission(gemmini, gemmini.find_loop("krow").after(), n_lifts=4)
    gemmini = inline_assign(gemmini, "w_s = _")
    gemmini = inline_assign(gemmini, "i_s = _")
    gemmini = delete_buffer(gemmini, "w_s: _")
    gemmini = delete_buffer(gemmini, "i_s: _")
    gemmini = simplify(gemmini)

    for loop in ("krow", "och_i", "och_o", "ocol", "orow"):
        gemmini = reorder_loops(gemmini, f"{loop} kcol")
    gemmini = stage_mem(
        gemmini,
        "for orow in _:_ #1",
        f"inp[0:{in_dim}, kcol:kcol+{out_dim}, 0:{in_ch}]",
        "input_tmp",
    )
    gemmini = simplify(gemmini)
    gemmini = expand_dim(gemmini, "input_tmp: _", k, "kcol")
    gemmini = lift_alloc(gemmini, "input_tmp: _", n_lifts=1)
    gemmini = simplify(gemmini)
    gemmini = autofission(
        gemmini,
        gemmini.find("input_tmp[_] = _").parent().parent().parent().after(),
        n_lifts=1,
    )
    gemmini = simplify(gemmini)

    gemmini = rearrange_dim(gemmini, "res: _", [1, 0, 2, 3])
    gemmini = simplify(mult_dim(gemmini, "res: _", 1, 2))
    gemmini = divide_loop(
        gemmini, "i2 #2", kch_blk, ["i2_o", "i2_i"], perfect=True
    )
    gemmini = divide_dim(gemmini, "input_tmp: _", 3, kch_blk)
    gemmini = simplify(gemmini)
    gemmini = rearrange_dim(gemmini, "input_tmp: _", [0, 3, 1, 2, 4])
    gemmini = simplify(mult_dim(gemmini, "input_tmp: _", 2, 3))
    gemmini = divide_loop(
        gemmini, "i3 #1", kch_blk, ["i3_o", "i3_i"], perfect=True
    )
    gemmini = divide_dim(gemmini, "weights_tmp: _", 3, kch_blk)
    gemmini = divide_loop(
        gemmini, "i2 #1", och_blk, ["i2_o", "i2_i"], perfect=True
    )
    gemmini = divide_dim(gemmini, "weights_tmp: _", 2, och_blk)
    gemmini = simplify(gemmini)
    gemmini = rearrange_dim(gemmini, "weights_tmp: _", [0, 1, 2, 4, 3, 5])
    if och_blk < 16:
        gemmini = resize_dim(gemmini, "res: _", 2, 16, 0)
        gemmini = simplify(gemmini)

    gemmini = simplify(mult_loops(gemmini, gemmini.find_loop("orow #1"), "m"))
    gemmini = rewrite_expr(
        gemmini,
        gemmini.find("a2 = _").rhs().idx()[2],
        f"m + {out_dim} * krow",
    )
    gemmini = simplify(gemmini)
    gemmini = simplify(mult_loops(gemmini, gemmini.find_loop("orow"), "m"))
    for _ in range(2):
        gemmini = divide_loop(gemmini, "m", 16, ["m_o", "m_i"], tail="cut")
    gemmini = simplify(gemmini)

    for _ in range(2):
        gemmini = reorder_loops(gemmini, "m_i och_o")
    for _ in range(2):
        gemmini = reorder_loops(gemmini, "m_i och_o")
        gemmini = reorder_loops(gemmini, "och_i krow")
        gemmini = reorder_loops(gemmini, "m_i krow")
        gemmini = reorder_loops(gemmini, "och_i kch_o")
        gemmini = reorder_loops(gemmini, "m_i kch_o")
    gemmini = reorder_loops(gemmini, "ocol och_o")

    gemmini = set_memory(gemmini, "input_tmp: _", GEMM_SCRATCH_FIXED)
    gemmini = set_memory(gemmini, "weights_tmp: _", GEMM_SCRATCH_FIXED)
    gemmini = set_memory(gemmini, "res: _", GEMM_ACCUM_FIXED)

    gemmini = replace(gemmini, "for i2_i in _:_ #0", ld_i8_block_strided_id2)
    gemmini = call_eqv(gemmini, ld_i8_block_strided_id2, ld_i8_block_strided_id2_v2)
    gemmini = inline(gemmini, ld_i8_block_strided_id2_v2)
    gemmini = inline_window(gemmini, "src = _")
    gemmini = inline_window(gemmini, "dst = _")
    gemmini = simplify(gemmini)

    gemmini = replace(gemmini, "for i1 in _:_ #2", ld_i8_block_strided_id1)
    gemmini = call_eqv(gemmini, ld_i8_block_strided_id1, ld_i8_block_strided_id1_v2)
    gemmini = inline(gemmini, ld_i8_block_strided_id1_v2)
    gemmini = inline_window(gemmini, "src = _")
    gemmini = inline_window(gemmini, "dst = _")
    gemmini = simplify(gemmini)

    for _ in range(2):
        gemmini = replace(gemmini, "for m_i in _:_ #0", ld_acc_i32_repeat)
        gemmini = call_eqv(gemmini, ld_acc_i32_repeat, ld_acc_i32_repeat_v2)
        gemmini = inline(gemmini, ld_acc_i32_repeat_v2)
        gemmini = inline_window(gemmini, "src = _")
        gemmini = inline_window(gemmini, "dst = _")
        gemmini = simplify(gemmini)

    for _ in range(2):
        gemmini = replace(gemmini, "for m_i in _:_ #0", matmul_acc_i8_trans_b)
        gemmini = call_eqv(
            gemmini, matmul_acc_i8_trans_b, matmul_acc_i8_trans_b_v2
        )
        gemmini = inline(gemmini, matmul_acc_i8_trans_b_v2)
        gemmini = inline_window(gemmini, "A = _")
        gemmini = inline_window(gemmini, "B = _")
        gemmini = inline_window(gemmini, "C = _")
        gemmini = simplify(gemmini)

    gemmini = hoist_acc_scale(gemmini, 4)
    gemmini = replace(gemmini, "for ocol in _:_ #0", st_acc_i8_act)
    gemmini = call_eqv(gemmini, st_acc_i8_act, st_acc_i8_act_v2)
    gemmini = inline(gemmini, st_acc_i8_act_v2)
    gemmini = inline_window(gemmini, "src = _")
    gemmini = inline_window(gemmini, "dst = _")
    gemmini = simplify(gemmini)

    gemmini = hoist_config(gemmini, "config_ld_i8_block_id2(_)", 3)
    gemmini = hoist_config(gemmini, "config_ld_repeat(_)", 2)
    gemmini = delete_config(gemmini, "config_ld_repeat(_) #1")
    gemmini = hoist_config(gemmini, "config_ld_i8_block_id1(_)", 2)
    gemmini = hoist_config(gemmini, "config_matmul_trans_b(_)", 5)
    gemmini = delete_config(gemmini, "config_matmul_trans_b(_) #1")
    gemmini = hoist_config_after_store(
        gemmini, "config_st_acc_i8(_)", 2, 0
    )
    gemmini = insert_noop_call(
        gemmini, gemmini.find_loop("orow").after(), fence, []
    )

    return gemmini


lower_conv = sched_conv_fusion


def finish_lowering(eval_proc, layers):
    """The step every lower_conv use needs in `braggnn_eval` (after
    `braggnn_inference` is inlined into it): inline each lowered layer, whose
    weight re-layout into `<layer>_weights_ohwi` then sits in the patch loop,
    and hoist that one-time re-layout out of the patch loop. Without this the
    weights are re-laid out on the CPU for every patch (conv2: ~119k instead of
    ~7.7k cycles). The re-layout only reorders the weights, like the seed's own
    weight staging before the patch loop; no per-patch work moves out.

        braggnn_eval = finish_lowering(braggnn_eval, ["conv2", "conv3"])
    """
    for layer in layers:
        eval_proc = inline(eval_proc, f"{layer}(_)")
    for layer in layers:
        buf = f"{layer}_weights_ohwi"
        eval_proc = lift_alloc(eval_proc, f"{buf}: _", n_lifts=1)
        eval_proc = lift_out_of_patch_loop(eval_proc, f"{buf}[_] = _")
    return eval_proc
