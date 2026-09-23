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
    add_loop,
    autofission,
    call_eqv,
    cut_loop,
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
    join_loops,
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
    shift_loop,
    simplify,
    stage_mem,
    unroll_loop,
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
    st_acc_i8_act_spad,
    st_acc_i8_act_spad_v2,
    st_acc_i8_act_v2,
)

NLB_ROW_TILE = 8


class GEMM_SCRATCH_FIXED(GEMM_SCRATCH):
    addresses: ClassVar[dict[str, int]] = {
        "input_tmp": 1,
        # conv3's input is written by conv2's mvout_spad while conv2's input
        # may still be read, so it must not alias conv2's (rows 1..756).
        "input_tmp_1": 1024,
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
        # conv3's bias load is issued before conv2's result is moved out.
        "res_1": 0x80000000 + 128,
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


def sched_conv(cpu, in_dim, in_ch, out_ch, k, act=False):
    name = cpu.name()[: -len("_cpu")]

    do_conv = make_loop_conv_ws(f"do_{name}", in_dim, in_ch, out_ch, k, act)

    gemmini = rename(cpu, name)
    gemmini = replace(gemmini, "for orow in _:_", do_conv)
    gemmini = fence_after(gemmini, f"do_{name}(_)")

    return gemmini


def sched_conv_fusion(cpu, in_dim, in_ch, out_ch, k, fused_in=False, fused_out=False):
    """fused_in / fused_out leave the input staging / the output store as
    plain loops so that fuse_conv2_conv3 can connect them."""
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

    if not fused_in:
        gemmini = replace(gemmini, "for i1 in _:_ #2", ld_i8_block_strided_id1)
        gemmini = call_eqv(
            gemmini, ld_i8_block_strided_id1, ld_i8_block_strided_id1_v2
        )
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
    if not fused_out:
        gemmini = replace(gemmini, "for ocol in _:_ #0", st_acc_i8_act)
        gemmini = call_eqv(gemmini, st_acc_i8_act, st_acc_i8_act_v2)
        gemmini = inline(gemmini, st_acc_i8_act_v2)
        gemmini = inline_window(gemmini, "src = _")
        gemmini = inline_window(gemmini, "dst = _")
        gemmini = simplify(gemmini)

    gemmini = hoist_config(gemmini, "config_ld_i8_block_id2(_)", 3)
    gemmini = hoist_config(gemmini, "config_ld_repeat(_)", 2)
    gemmini = delete_config(gemmini, "config_ld_repeat(_) #1")
    if not fused_in:
        gemmini = hoist_config(gemmini, "config_ld_i8_block_id1(_)", 2)
    gemmini = hoist_config(gemmini, "config_matmul_trans_b(_)", 5)
    gemmini = delete_config(gemmini, "config_matmul_trans_b(_) #1")
    if not fused_out:
        gemmini = hoist_config_after_store(
            gemmini, "config_st_acc_i8(_)", 2, 0
        )
        gemmini = insert_noop_call(
            gemmini, gemmini.find_loop("orow").after(), fence, []
        )

    return gemmini


def _const(expr_cursor):
    return expr_cursor.value()


def _loop_range(loop):
    return _const(loop.lo()), _const(loop.hi())


def _pos(stmt):
    n = 0
    while not isinstance(stmt.prev(), InvalidCursor):
        stmt = stmt.prev()
        n += 1
    return n


def _nest(stmt, depth):
    for _ in range(depth):
        stmt = stmt.parent()
    return stmt


def fuse_conv2_conv3(p):
    """Move conv2's output straight from the accumulator into conv3's
    scratchpad input with mvout_spad, instead of mvout to conv2_out in DRAM
    and mvin back.

    conv3 stages its input once per kernel column kcol (input_tmp[kcol, ...]
    holds columns kcol..kcol+4 of conv2_out), so each conv2 output column is
    needed up to three times. The store loop is cut into the column ranges
    [0,1) [1,2) [2,5) [5,6) [6,7), each piece is duplicated as many times as
    it is read (add_loop + unroll_loop on the idempotent store), and every
    copy is fused with the staging piece that reads it; inline_assign then
    forwards the stored value, and conv2_out is deleted once nothing reads
    or writes it. Finally the pieces of one kcol are joined back into a
    5-column loop and replaced with the mvout_spad instruction.
    """
    # conv2's store: for orow, och_o, ocol, och_i: conv2_out[...] = ...
    # conv3's staging: for kcol, i0, i1, i2_o, i2_i: input_tmp[...] = conv2_out[...]
    store_depth = 4  # stmt -> och_i -> ocol -> och_o -> orow
    for cut in (1, 2, 5, 6):
        ocol = _nest(p.find_all("conv2_out[_] = _")[-1], 2)
        p = cut_loop(p, ocol, cut)
    for piece in range(4):
        ocol = _nest(p.find_all("conv2_out[_] = _")[piece], 2)
        p = fission(p, ocol.after(), n_lifts=2)

    # How many kernel columns read each piece.
    copies = {(0, 1): 1, (1, 2): 2, (2, 5): 3, (5, 6): 2, (6, 7): 1}
    idx = 0
    for (lo, hi), n in copies.items():
        if n > 1:
            nest = _nest(p.find_all("conv2_out[_] = _")[idx], store_depth)
            p = add_loop(p, nest, "dup", n)
            p = unroll_loop(p, p.find_loop("dup"))
        idx += n

    # Staging: one copy per kcol, in (i0, i2_o, i1, i2_i) order, cut into
    # the same column pieces, each with its i1 loop shifted to the column.
    stage_depth = 4  # stmt -> i2_i -> i1 -> i2_o -> i0 (after reorder)
    kcol = _nest(p.find("input_tmp[_] = _"), 5)
    p = reorder_loops(p, _nest(p.find("input_tmp[_] = _"), 3))
    p = unroll_loop(p, kcol)
    stage_cuts = {0: (1, 2), 1: (1, 4), 2: (3, 4)}
    for k in (0, 1, 2):
        for cut in reversed(stage_cuts[k]):
            stmt = [
                s for s in p.find_all("input_tmp[_] = _")
                if _const(s.idx()[0]) == k
            ][0]
            p = cut_loop(p, _nest(stmt, 2), cut)
        for piece in range(2):
            stmt = [
                s for s in p.find_all("input_tmp[_] = _")
                if _const(s.idx()[0]) == k
            ][piece]
            p = fission(p, _nest(stmt, 2).after(), n_lifts=2)
        for piece in range(3):
            stmt = [
                s for s in p.find_all("input_tmp[_] = _")
                if _const(s.idx()[0]) == k
            ][piece]
            i1 = _nest(stmt, 2)
            if k:
                p = shift_loop(p, i1, _const(i1.lo()) + k)
    p = simplify(p)

    # The staging's allocation must precede the pieces it is fused with.
    def first_store():
        return _nest(p.find_all("conv2_out[_] = _")[0], store_depth)

    while _pos(p.find_all("input_tmp : _")[1]) > _pos(first_store()):
        p = reorder_stmts(p, p.find_all("input_tmp : _")[1].expand(1, 0))

    # Pair the last remaining store copy with the staging piece that reads
    # the same columns for the same kcol, move that piece up next to it,
    # fuse the two nests and forward the stored value.
    order = [
        ((6, 7), 2), ((5, 6), 2), ((5, 6), 1), ((2, 5), 2), ((2, 5), 1),
        ((2, 5), 0), ((1, 2), 1), ((1, 2), 0), ((0, 1), 0),
    ]
    for cols, k in order:
        store = p.find_all("conv2_out[_] = _")[-1]
        assert _loop_range(_nest(store, 2)) == cols, cols
        while True:
            store_nest = _nest(p.find_all("conv2_out[_] = _")[-1], store_depth)
            stage = [
                s for s in p.find_all("input_tmp[_] = _")
                if _const(s.idx()[0]) == k
                and _loop_range(_nest(s, 2)) == cols
                and "conv2_out" in str(s)
            ][0]
            stage_nest = _nest(stage, stage_depth)
            if _pos(stage_nest) == _pos(store_nest) + 1:
                break
            p = reorder_stmts(p, stage_nest.expand(1, 0))
        for depth in (store_depth, 3, 2, 1):
            store_loop = _nest(p.find_all("conv2_out[_] = _")[-1], depth)
            p = fuse(p, store_loop, store_loop.next())
        p = inline_assign(p, p.find_all("conv2_out[_] = _")[-1])
    p = delete_buffer(p, "conv2_out : _")
    p = simplify(p)

    # Now nine nests write input_tmp[k, och_o, ocol - k + 5 * orow, och_i]
    # from res. Sort them by (kcol, first column) so each kcol's pieces are
    # adjacent, then fuse their orow / och_o loops and join the column loops.
    def pieces():
        return [
            (_const(s.idx()[0]), _const(_nest(s, 2).lo()), _nest(s, store_depth))
            for s in p.find_all("input_tmp[_] = _")
        ]

    for _ in range(len(pieces())):
        for i in range(len(pieces()) - 1):
            a, b = pieces()[i], pieces()[i + 1]
            if a[:2] > b[:2]:
                p = reorder_stmts(p, b[2].expand(1, 0))

    for k in (0, 1, 2):
        for depth in (store_depth, 3):
            for _ in range(2):
                first = [s for s in p.find_all("input_tmp[_] = _")
                         if _const(s.idx()[0]) == k][0]
                loop = _nest(first, depth)
                p = fuse(p, loop, loop.next())
        for _ in range(2):
            first = [s for s in p.find_all("input_tmp[_] = _")
                     if _const(s.idx()[0]) == k][0]
            ocol = _nest(first, 2)
            p = join_loops(p, ocol, ocol.next())
        if k:
            first = [s for s in p.find_all("input_tmp[_] = _")
                     if _const(s.idx()[0]) == k][0]
            p = shift_loop(p, _nest(first, 2), 0)
        p = simplify(p)

    # Each (kcol, orow, och_o) is now a 5 x 16 block copy: one mvout_spad.
    for k in (0, 1, 2):
        first = [s for s in p.find_all("input_tmp[_] = _")
                 if _const(s.idx()[0]) == k][0]
        p = replace(p, _nest(first, 2), st_acc_i8_act_spad)
    for _ in range(3):
        p = call_eqv(p, "st_acc_i8_act_spad(_)", st_acc_i8_act_spad_v2)
        p = inline(p, "st_acc_i8_act_spad_v2(_)")
        p = inline_window(p, "src = _")
        p = inline_window(p, "dst = _")
        p = simplify(p)

    # Lift each kcol's store config out of its orow / och_o loops; the three
    # write the same scale, stride and activation, so keep only the first.
    for i in range(3):
        cfg = [c for c in p.find_all("config_st_acc_i8(_)")
               if isinstance(c.parent(), ForCursor)][0]
        p = fission(p, cfg.after(), n_lifts=2)
        for _ in range(2):
            cfg = [c for c in p.find_all("config_st_acc_i8(_)")
                   if isinstance(c.parent(), ForCursor)][0]
            p = remove_loop(p, cfg.parent())
    # config_st_acc_i8 #0..#2 are these three; #3 is conv3's own.
    for _ in range(2):
        p = delete_config(p, "config_st_acc_i8(_) #1")
    return p


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
    conv2_cpu, CONV1_DIM, CONV1_FILTERS, CONV2_FILTERS, 3, fused_out=True
)
conv3 = sched_conv_fusion(
    conv3_cpu, CONV2_DIM, CONV2_FILTERS, CONV3_FILTERS, 3, fused_in=True
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
    gemmini = fuse_conv2_conv3(gemmini)
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
        gemmini = lift_alloc(gemmini, f"{buf}: _", n_lifts=1)
        gemmini = lift_out_of_patch_loop(gemmini, f"{buf}[_] = _")
    return gemmini


braggnn_eval = schedule_eval()

__all__ = ["braggnn_eval"]
