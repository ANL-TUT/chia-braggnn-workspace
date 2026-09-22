from __future__ import annotations

from braggnn_reference import (
    CONV1_DIM,
    CONV1_FILTERS,
    CONV2_DIM,
    CONV2_FILTERS,
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
    nlb_cpu,
)
from exo.API_cursors import InvalidCursor
from exo.API_scheduling import (
    call_eqv,
    commute_expr,
    divide_loop,
    expand_dim,
    extract_subproc,
    fission,
    inline,
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
    rewrite_expr,
    set_memory,
    simplify,
    split_write,
    unroll_loop,
)
from exo.libs.memories import GEMM_ACCUM, GEMM_SCRATCH
from exo.platforms.gemmini import (
    ld_i8_id1,
    ld_i8_id2,
    matmul_acc_i8,
    old_fission_after,
    old_lift_alloc,
    old_reorder,
    zero_acc_i32,
)
from gemmini import (
    fence,
    ld_acc_i8_scaled,
    ld_acc_i8_scaled_acc,
    ld_acc_i32_col,
    ld_acc_i32_repeat,
    ld_i8_col,
    ld_i8_im2col,
    make_loop_softmax_flat,
    matmul_acc_i8_trans_b,
    st_acc_i8_act,
    st_acc_i8_no_act,
)

NLB_ROWS = CONV1_DIM * CONV1_DIM
NLB_SOFTMAX_TILE = 72


st_acc_i8_act_decls_first = rename(st_acc_i8_act, "st_acc_i8_act_decls_first")
for _pair in (
    "src_tmp = _ ; tmp : _",
    "acc_scale(_) ; tmp2 : _",
    "src_tmp = _ ; tmp2 : _",
):
    st_acc_i8_act_decls_first = reorder_stmts(
        st_acc_i8_act_decls_first, st_acc_i8_act_decls_first.find(_pair)
    )


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


def sched_fc_lowlevel(cpu, in_ch, out_features, act=False):
    name = cpu.name()[: -len("_cpu")]
    assert in_ch <= 16
    assert out_features <= 16

    gemmini = rename(cpu, name)
    gemmini = unroll_loop(gemmini, "krow")
    gemmini = unroll_loop(gemmini, "kcol")
    gemmini = old_lift_alloc(gemmini, "res : _", n_lifts=1)
    gemmini = expand_dim(gemmini, "res : _", "16", "0")
    gemmini = rearrange_dim(gemmini, "res : _", [1, 0])
    gemmini = old_fission_after(gemmini, "res[_] = _", n_lifts=1)
    gemmini = old_fission_after(gemmini, "for kch in _:_", n_lifts=1)
    gemmini = old_lift_alloc(gemmini, "w_s : _", n_lifts=1, size=16)
    gemmini = old_lift_alloc(gemmini, "w_s : _", n_lifts=1)
    gemmini = old_lift_alloc(gemmini, "i_s : _", n_lifts=1)
    gemmini = old_lift_alloc(gemmini, "i_s : _", n_lifts=1, keep_dims=False)
    gemmini = expand_dim(gemmini, "i_s : _", "16", "0")
    gemmini = rearrange_dim(gemmini, "i_s : _", [1, 0])
    gemmini = old_fission_after(gemmini, "w_s[_] = _", n_lifts=2)
    gemmini = old_fission_after(gemmini, "i_s[_] = _", n_lifts=2)
    gemmini = divide_loop(gemmini, "j #3", 1, ["j_o", "j_i"], perfect=True)
    gemmini = divide_loop(gemmini, "j #2", 1, ["j_o", "j_i"], perfect=True)
    gemmini = simplify(gemmini)
    gemmini = reorder_stmts(gemmini, gemmini.find("a2 : _").expand(0, 1))
    gemmini = reorder_stmts(gemmini, gemmini.find("a2 = _").expand(0, 1))
    gemmini = commute_expr(gemmini, "a2 * b2")
    gemmini = rewrite_expr(gemmini, gemmini.find("b2 = _").rhs().idx()[0], "j_o")
    gemmini = rewrite_expr(gemmini, gemmini.find("a2 = _").rhs().idx()[1], "j_i")
    gemmini = rewrite_expr(gemmini, gemmini.find("res[_] += _").idx()[0], "j_o")
    gemmini = rewrite_expr(gemmini, gemmini.find("res[_] += _").idx()[1], "j_i")
    gemmini = rewrite_expr(gemmini, gemmini.find("src_tmp = _").rhs().idx()[0], "j_o")
    gemmini = rewrite_expr(gemmini, gemmini.find("src_tmp = _").rhs().idx()[1], "j_i")
    gemmini = rewrite_expr(gemmini, gemmini.find("output[_] = _").idx()[0], "j_o")
    gemmini = rewrite_expr(gemmini, gemmini.find("output[_] = _").idx()[1], "j_i")
    gemmini = set_memory(gemmini, "res : _", GEMM_ACCUM)
    gemmini = set_memory(gemmini, "w_s : _", GEMM_SCRATCH)
    gemmini = set_memory(gemmini, "i_s : _", GEMM_SCRATCH)
    gemmini = replace(gemmini, "for j in _:_ #0", ld_acc_i32_col)
    gemmini = replace(gemmini, "for j in _:_ #0", ld_i8_id1)
    gemmini = replace(gemmini, "for kch in _:_ #0", ld_i8_col)
    gemmini = replace(gemmini, "for j_o in _:_ #0", matmul_acc_i8)
    if act:
        gemmini = hoist_acc_scale(gemmini, 2)
        gemmini = replace(gemmini, "for j_o in _:_ #0", st_acc_i8_act)
        gemmini = fence_after(gemmini, "st_acc_i8_act(_)")
    else:
        gemmini = replace(gemmini, "for j_o in _:_ #0", st_acc_i8_no_act)
        gemmini = fence_after(gemmini, "st_acc_i8_no_act(_)")

    return gemmini


def sched_conv_lowlevel(cpu, in_dim, in_ch, out_ch, k, act=False):
    name = cpu.name()[: -len("_cpu")]
    out_dim = in_dim - k + 1
    och_blk = min(out_ch, 16)
    kch_blk = min(in_ch, 16)
    assert out_dim <= 16
    assert in_ch % kch_blk == 0
    assert out_ch % och_blk == 0

    gemmini = rename(cpu, name)
    gemmini = divide_loop(gemmini, "och", och_blk, ["och_o", "och_i"], perfect=True)
    gemmini = divide_loop(gemmini, "kch", kch_blk, ["kch_o", "kch_i"], perfect=True)
    gemmini = old_reorder(gemmini, "ocol och_o")
    gemmini = old_lift_alloc(gemmini, "res : _", n_lifts=1, size=16)
    gemmini = old_lift_alloc(gemmini, "res : _", n_lifts=1)
    gemmini = old_fission_after(gemmini, "res[_] = _", n_lifts=2)
    gemmini = old_fission_after(gemmini, "for krow in _:_", n_lifts=2)
    gemmini = old_reorder(gemmini, "och_i krow")
    gemmini = old_reorder(gemmini, "och_i kcol")
    gemmini = old_reorder(gemmini, "och_i kch_o")
    gemmini = old_reorder(gemmini, "ocol krow")
    gemmini = old_reorder(gemmini, "ocol kcol")
    gemmini = old_reorder(gemmini, "ocol kch_o")
    gemmini = simplify(gemmini)
    gemmini = old_lift_alloc(gemmini, "w_s : _", n_lifts=1)
    gemmini = old_lift_alloc(gemmini, "w_s : _", n_lifts=1, mode="col", size=16)
    gemmini = old_lift_alloc(gemmini, "w_s : _", n_lifts=1, keep_dims=False)
    gemmini = old_lift_alloc(gemmini, "i_s : _", n_lifts=1, size=16)
    gemmini = old_lift_alloc(gemmini, "i_s : _", n_lifts=1, keep_dims=False)
    gemmini = old_lift_alloc(gemmini, "i_s : _", n_lifts=1)
    gemmini = old_fission_after(gemmini, "w_s[_] = _", n_lifts=3)
    gemmini = old_fission_after(gemmini, "i_s[_] = _", n_lifts=3)
    gemmini = set_memory(gemmini, "res : _", GEMM_ACCUM)
    gemmini = set_memory(gemmini, "w_s : _", GEMM_SCRATCH)
    gemmini = set_memory(gemmini, "i_s : _", GEMM_SCRATCH)
    gemmini = old_reorder(gemmini, "och_i kch_i")
    gemmini = replace(gemmini, "for ocol in _:_ #0", ld_acc_i32_repeat)
    gemmini = replace(gemmini, "for kch_i in _:_ #0", ld_i8_id1)
    gemmini = replace(gemmini, "for ocol in _:_ #0", ld_i8_id2)
    gemmini = old_reorder(gemmini, "kch_i och_i")
    gemmini = replace(gemmini, "for ocol in _:_ #0", matmul_acc_i8)
    if act:
        gemmini = hoist_acc_scale(gemmini, 2)
        gemmini = replace(gemmini, "for ocol in _:_ #0", st_acc_i8_act)
    else:
        gemmini = replace(gemmini, "for ocol in _:_ #0", st_acc_i8_no_act)
    gemmini = insert_noop_call(gemmini, gemmini.find_loop("orow").after(), fence, [])

    return gemmini


def sched_conv_im2col_lowlevel(cpu, in_dim, in_ch, out_ch, k):
    name = cpu.name()[: -len("_cpu")]
    out_dim = in_dim - k + 1
    och_blk = min(out_ch, 16)
    assert out_dim <= 16
    assert in_dim <= 16
    assert in_ch * k <= 16
    assert out_ch % och_blk == 0

    gemmini = rename(cpu, name)
    gemmini = divide_loop(gemmini, "och", och_blk, ["och_o", "och_i"], perfect=True)
    gemmini = mult_loops(gemmini, gemmini.find_loop("kcol"), "k")
    gemmini = simplify(gemmini)
    gemmini = old_reorder(gemmini, "ocol och_o")
    gemmini = old_lift_alloc(gemmini, "res : _", n_lifts=1, size=16)
    gemmini = old_lift_alloc(gemmini, "res : _", n_lifts=1)
    gemmini = old_fission_after(gemmini, "res[_] = _", n_lifts=2)
    gemmini = old_fission_after(gemmini, "for krow in _:_", n_lifts=2)
    gemmini = old_reorder(gemmini, "och_i krow")
    gemmini = old_reorder(gemmini, "och_i k")
    gemmini = old_reorder(gemmini, "ocol krow")
    gemmini = old_reorder(gemmini, "ocol k")
    gemmini = simplify(gemmini)
    gemmini = old_lift_alloc(gemmini, "w_s : _", n_lifts=1)
    gemmini = old_lift_alloc(gemmini, "w_s : _", n_lifts=1, keep_dims=False)
    gemmini = old_lift_alloc(gemmini, "w_s : _", n_lifts=1, size=16)
    gemmini = old_lift_alloc(gemmini, "i_s : _", n_lifts=1, keep_dims=False)
    gemmini = old_lift_alloc(gemmini, "i_s : _", n_lifts=1, size=in_dim)
    gemmini = old_lift_alloc(gemmini, "i_s : _", n_lifts=1, mode="col", size=16)
    gemmini = old_fission_after(gemmini, "w_s[_] = _", n_lifts=3)
    gemmini = old_fission_after(gemmini, "i_s[_] = _", n_lifts=3)
    gemmini = set_memory(gemmini, "res : _", GEMM_ACCUM)
    gemmini = set_memory(gemmini, "w_s : _", GEMM_SCRATCH)
    gemmini = set_memory(gemmini, "i_s : _", GEMM_SCRATCH)
    gemmini = old_reorder(gemmini, "k ocol")
    gemmini = simplify(gemmini)
    gemmini = replace(gemmini, "for ocol in _:_ #0", ld_acc_i32_repeat)
    gemmini = replace(gemmini, "for k in _:_ #0", ld_i8_id1)
    gemmini = replace(gemmini, "for ocol in _:_ #0", ld_i8_im2col)
    gemmini = old_reorder(gemmini, "k och_i")
    gemmini = replace(gemmini, "for ocol in _:_ #0", matmul_acc_i8)
    gemmini = replace(gemmini, "for ocol in _:_ #0", st_acc_i8_no_act)
    gemmini = insert_noop_call(gemmini, gemmini.find_loop("orow").after(), fence, [])

    return gemmini


conv1 = sched_conv_im2col_lowlevel(conv1_cpu, INPUT_DIM, 1, CONV1_FILTERS, 3)
conv2 = sched_conv_lowlevel(
    conv2_cpu, CONV1_DIM, CONV1_FILTERS, CONV2_FILTERS, 3, act=True
)
conv3 = sched_conv_lowlevel(
    conv3_cpu, CONV2_DIM, CONV2_FILTERS, CONV3_FILTERS, 3, act=True
)

fc2 = sched_fc_lowlevel(fc2_cpu, FC1_UNITS, FC2_UNITS, act=True)
fc3 = sched_fc_lowlevel(fc3_cpu, FC2_UNITS, FC3_UNITS, act=True)
fc4 = sched_fc_lowlevel(fc4_cpu, FC3_UNITS, FC4_UNITS, act=True)
fc_output = sched_fc_lowlevel(fc_output_cpu, FC4_UNITS, OUTPUT_UNITS)


def tile_matmul(sub, n_reg, do_mm, n_kblk=1, b_col=False):
    sub = old_lift_alloc(sub, "res : _", n_lifts=1, size=16)
    sub = old_lift_alloc(sub, "res : _", n_lifts=1)
    sub = old_fission_after(sub, "res[_] = _", n_lifts=2)
    sub = old_fission_after(sub, "for k_o in _:_", n_lifts=2)
    sub = old_reorder(sub, "j_i k_o")
    sub = old_reorder(sub, "i_i k_o")
    sub = simplify(sub)
    sub = old_lift_alloc(sub, "b : _", n_lifts=1)
    if b_col:
        sub = old_lift_alloc(sub, "b : _", n_lifts=1, mode="col")
    else:
        sub = old_lift_alloc(sub, "b : _", n_lifts=1)
    sub = old_lift_alloc(sub, "b : _", n_lifts=1, keep_dims=False)
    sub = old_lift_alloc(sub, "a : _", n_lifts=1, size=16)
    sub = old_lift_alloc(sub, "a : _", n_lifts=1, keep_dims=False)
    sub = old_lift_alloc(sub, "a : _", n_lifts=1)
    sub = old_fission_after(sub, "b[_] = _", n_lifts=3)
    sub = old_fission_after(sub, "a[_] = _", n_lifts=3)
    for r in range(n_reg):
        sub = set_memory(sub, f"res : _ #{r}", GEMM_ACCUM)
    for r in range(n_reg * n_kblk):
        sub = set_memory(sub, f"b : _ #{r}", GEMM_SCRATCH)
        sub = set_memory(sub, f"a : _ #{r}", GEMM_SCRATCH)
    for _ in range(n_reg):
        sub = replace(sub, "for i_i in _:_ #0", zero_acc_i32)
        for _ in range(n_kblk):
            sub = replace(sub, "for i_i in _:_ #0", ld_i8_id2)
            if b_col:
                sub = reorder_loops(sub, "for j_i in _:_ #0")
                sub = replace(sub, "for k_i in _:_ #0", ld_i8_id1)
            else:
                sub = replace(sub, "for j_i in _:_ #0", ld_i8_id1)
            sub = replace(sub, "for i_i in _:_ #0", do_mm)
        sub = replace(sub, "for i_i in _:_ #0", st_acc_i8_no_act)
    return sub


def sched_conv_sub(sub, name, in_ch, out_ch):
    och_blk, kch_blk = min(out_ch, 16), min(in_ch, 16)
    g = rename(sub, name)
    g = divide_loop(g, "och", och_blk, ["och_o", "och_i"], perfect=True)
    g = divide_loop(g, "kch", kch_blk, ["kch_o", "kch_i"], perfect=True)
    g = old_reorder(g, "ocol och_o")
    g = old_lift_alloc(g, "res : _", n_lifts=1, size=16)
    g = old_lift_alloc(g, "res : _", n_lifts=1)
    g = old_fission_after(g, "res[_] = _", n_lifts=2)
    g = old_fission_after(g, "for krow in _:_", n_lifts=2)
    for a, b in (
        ("och_i", "krow"),
        ("och_i", "kcol"),
        ("och_i", "kch_o"),
        ("ocol", "krow"),
        ("ocol", "kcol"),
        ("ocol", "kch_o"),
    ):
        g = old_reorder(g, f"{a} {b}")
    g = simplify(g)
    g = old_lift_alloc(g, "w_s : _", n_lifts=1)
    g = old_lift_alloc(g, "w_s : _", n_lifts=1, mode="col", size=16)
    g = old_lift_alloc(g, "w_s : _", n_lifts=1, keep_dims=False)
    g = old_lift_alloc(g, "i_s : _", n_lifts=1, size=16)
    g = old_lift_alloc(g, "i_s : _", n_lifts=1, keep_dims=False)
    g = old_lift_alloc(g, "i_s : _", n_lifts=1)
    g = old_fission_after(g, "w_s[_] = _", n_lifts=3)
    g = old_fission_after(g, "i_s[_] = _", n_lifts=3)
    g = set_memory(g, "res : _", GEMM_ACCUM)
    g = set_memory(g, "w_s : _", GEMM_SCRATCH)
    g = set_memory(g, "i_s : _", GEMM_SCRATCH)
    g = old_reorder(g, "och_i kch_i")
    g = replace(g, "for ocol in _:_ #0", ld_acc_i32_repeat)
    g = replace(g, "for kch_i in _:_ #0", ld_i8_id1)
    g = replace(g, "for ocol in _:_ #0", ld_i8_id2)
    g = old_reorder(g, "kch_i och_i")
    g = replace(g, "for ocol in _:_ #0", matmul_acc_i8)
    g = replace(g, "for ocol in _:_ #0", st_acc_i8_no_act)
    return insert_noop_call(g, g.find_loop("orow").after(), fence, [])


def schedule_nlb():
    p = rename(nlb_cpu, "nlb")
    for pat in ("nlb_qkv_conv_cpu(_)",) * 3 + (
        "matmul_theta_phi_cpu(_)",
        "nlb_softmax_cpu(_)",
        "matmul_attention_g_cpu(_)",
        "nlb_out_conv_cpu(_)",
        "resadd_relu_cpu(_)",
    ):
        p = inline(p, pat)
    for buf, pairs in (
        ("nlb_theta_out", [(0, 1)]),
        ("nlb_phi_out", [(0, 1)]),
        ("nlb_g_out", [(0, 1)]),
        ("attention_raw", [(0, 1), (1, 2)]),
        ("attention_out", [(0, 1), (1, 2)]),
        ("attended_output", [(0, 1)]),
        ("nlb_output", [(0, 1)]),
    ):
        for hi, lo in pairs:
            p = mult_dim(p, f"{buf} : _", hi, lo)
    p = simplify(p)

    p, qkv = extract_subproc(p, p.find_loop("orow #0"), "qkv_conv_flat")
    p = replace(p, p.find_loop("orow #0"), qkv)
    p = replace(p, p.find_loop("orow #0"), qkv)
    p, tp = extract_subproc(p, p.find_loop("i1 #0"), "theta_phi_flat")
    p, sm = extract_subproc(p, p.find_loop("i1 #0"), "softmax_flat")
    p, ag = extract_subproc(p, p.find_loop("i1 #0"), "attention_g_flat")
    p, oc = extract_subproc(p, p.find_loop("orow #0"), "out_conv_flat")
    p, ra = extract_subproc(p, p.find_loop("i1 #0"), "resadd_flat")

    tp2 = mult_loops(tp, tp.find_loop("i1"), "i")
    tp2 = mult_loops(tp2, tp2.find_loop("j1"), "j")
    tp2 = simplify(tp2)
    tp2 = divide_loop(tp2, "i", 16, ["i_o", "i_i"], tail="cut")
    tp2 = divide_loop(tp2, "j", 16, ["j_o", "j_i"], tail="cut")
    tp2 = divide_loop(tp2, "j", 16, ["j_o", "j_i"], tail="cut")
    tp2 = old_fission_after(tp2, "for j_o in _:_ #0", n_lifts=1)
    tp2 = old_fission_after(tp2, "for j_o in _:_ #1", n_lifts=1)
    tp2 = old_reorder(tp2, "i_i j_o")
    tp2 = old_reorder(tp2, "i_i j_i")
    tp2 = old_reorder(tp2, "j_i i_i")
    for _ in range(4):
        tp2 = divide_loop(tp2, "k", 16, ["k_o", "k_i"], perfect=True)
    tp2 = simplify(tp2)
    tp2 = tile_matmul(rename(tp2, "theta_phi_gemm"), 4, matmul_acc_i8_trans_b)

    ag2 = mult_loops(ag, ag.find_loop("i1"), "i")
    ag2 = mult_loops(ag2, ag2.find_loop("k1"), "k")
    ag2 = simplify(ag2)
    ag2 = divide_loop(ag2, "i", 16, ["i_o", "i_i"], tail="cut")
    ag2 = divide_loop(ag2, "j", 16, ["j_o", "j_i"], perfect=True)
    ag2 = divide_loop(ag2, "j", 16, ["j_o", "j_i"], perfect=True)
    ag2 = old_reorder(ag2, "i_i j_o")
    ag2 = simplify(ag2)
    for _ in range(2):
        ag2 = divide_loop(ag2, "k", 16, ["k_o", "k_i"], tail="cut")
    ag2 = divide_loop(ag2, "k_i #3", 1, ["k_o", "k_i"], perfect=True)
    ag2 = divide_loop(ag2, "k_i #1", 1, ["k_o", "k_i"], perfect=True)
    ag2 = simplify(ag2)
    ag2 = tile_matmul(
        rename(ag2, "attention_g_gemm"), 2, matmul_acc_i8, n_kblk=2, b_col=True
    )

    sm2 = mult_loops(sm, sm.find_loop("i1"), "i")
    for _ in range(3):
        sm2 = mult_loops(sm2, sm2.find_loop("j1"), "j")
    sm2 = mult_dim(sm2, "exp_buf : _", 0, 1)
    sm2 = simplify(sm2)
    sm2 = rename(sm2, "softmax_gemm")
    sm2 = divide_loop(sm2, "i", NLB_SOFTMAX_TILE, ["i_o", "i_i"], tail="cut")
    do_sm = make_loop_softmax_flat("do_softmax", NLB_ROWS, SOFTMAX_MAX_SHIFT)
    sm2 = replace(sm2, sm2.find_loop("i_o").body(), do_sm)
    sm2 = replace(sm2, sm2.find_loop("i_o").next(), do_sm)
    sm2 = insert_noop_call(sm2, sm2.find_loop("i_o").next().after(), fence, [])

    qkv2 = sched_conv_sub(qkv, "qkv_conv_gemm", CONV1_FILTERS, CONV2_FILTERS)
    oc2 = sched_conv_sub(oc, "out_conv_gemm", CONV2_FILTERS, CONV1_FILTERS)

    ra2 = rename(ra, "resadd_gemm")
    ra2 = divide_loop(ra2, "j", 16, ["j_o", "j_i"], perfect=True)
    ra2 = old_reorder(ra2, "i2 j_o")
    ra2 = old_lift_alloc(ra2, "res : _", n_lifts=1, size=16)
    ra2 = old_lift_alloc(ra2, "res : _", n_lifts=1)
    ra2 = split_write(ra2, "res[_] = _")
    for stmt in (
        "tmp_res2 : _",
        "tmp_res1 : _",
        "src_tmp : _",
        "b2 = b_q",
        "clamp(b_tmp, b_q)",
        "acc_scale(b_src, b_tmp, nlb_add_a_scale)",
        "b_src = _",
        "b2 : _",
        "b_q : _",
        "b_tmp : _",
        "b_src : _",
    ):
        ra2 = reorder_stmts(ra2, f"{stmt} ; res[_] = a2")
    for stmt in ("tmp_res2 : _", "tmp_res1 : _", "src_tmp : _"):
        ra2 = reorder_stmts(ra2, f"{stmt} ; res[_] += b2")
    ra2 = old_fission_after(ra2, "res[_] = a2", n_lifts=2)
    ra2 = old_fission_after(ra2, "res[_] += b2", n_lifts=2)
    ra2 = set_memory(ra2, "res : _", GEMM_ACCUM)
    ra2 = replace(ra2, "for i2 in _:_ #0", ld_acc_i8_scaled)
    ra2 = fence_after(ra2, "ld_acc_i8_scaled(_)")
    ra2 = replace(ra2, "for i2 in _:_ #0", ld_acc_i8_scaled_acc)
    ra2 = replace(ra2, "for i2 in _:_ #0", st_acc_i8_act_decls_first)
    ra2 = insert_noop_call(ra2, ra2.find_loop("i1").after(), fence, [])

    for old, new in (
        (qkv, qkv2),
        (tp, tp2),
        (sm, sm2),
        (ag, ag2),
        (oc, oc2),
        (ra, ra2),
    ):
        for _ in range(len(p.find_all(f"{old.name()}(_)"))):
            p = call_eqv(p, f"{old.name()}(_)", new)
    return p


nlb = schedule_nlb()


def schedule_braggnn():
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


def sched_fc1_flat(gemmini):
    gemmini = inline(gemmini, "fc1_cpu(_)")
    for buf, hi, lo in (
        ("fc1_weights_", 2, 3),
        ("fc1_weights_", 1, 2),
        ("flattened", 1, 2),
        ("flattened", 0, 1),
    ):
        gemmini = simplify(mult_dim(gemmini, f"{buf} : _", hi, lo))
    gemmini = simplify(mult_loops(gemmini, gemmini.find_loop("krow"), "krow"))
    gemmini = simplify(mult_loops(gemmini, gemmini.find_loop("kch"), "k"))

    gemmini = divide_loop(gemmini, "k #0", 16, ["k_o", "k_i"], tail="cut")
    gemmini = simplify(gemmini)
    gemmini = old_lift_alloc(gemmini, "res : _", n_lifts=1)
    gemmini = expand_dim(gemmini, "res : _", "16", "0")
    gemmini = rearrange_dim(gemmini, "res : _", [1, 0])
    gemmini = old_fission_after(gemmini, "res[_] = _", n_lifts=1)
    gemmini = old_fission_after(gemmini, "for k_o in _:_", n_lifts=1)
    gemmini = old_fission_after(gemmini, "for k_i in _:_ #1", n_lifts=1)
    gemmini = old_reorder(gemmini, "j k_o")
    gemmini = old_lift_alloc(gemmini, "w_s : _", n_lifts=1, size=16)
    gemmini = old_lift_alloc(gemmini, "w_s : _", n_lifts=1)
    gemmini = old_lift_alloc(gemmini, "i_s : _", n_lifts=1, size=16)
    gemmini = old_lift_alloc(gemmini, "i_s : _", n_lifts=1, keep_dims=False)
    for n in (1, 0):
        gemmini = expand_dim(gemmini, f"i_s : _ #{n}", "16", "0")
        gemmini = rearrange_dim(gemmini, f"i_s : _ #{n}", [1, 0])
    gemmini = old_fission_after(gemmini, "w_s[_] = _", n_lifts=2)
    gemmini = old_fission_after(gemmini, "i_s[_] = _", n_lifts=2)
    gemmini = simplify(gemmini)

    for n in (5, 4, 2):
        gemmini = divide_loop(gemmini, f"j #{n}", 1, ["j_o", "j_i"], perfect=True)
    gemmini = simplify(gemmini)
    for n in (1, 0):
        gemmini = reorder_stmts(gemmini, gemmini.find(f"a2 : _ #{n}").expand(0, 1))
        gemmini = reorder_stmts(gemmini, gemmini.find(f"a2 = _ #{n}").expand(0, 1))
        gemmini = commute_expr(gemmini, f"a2 * b2 #{n}")
        gemmini = rewrite_expr(
            gemmini, gemmini.find(f"b2 = _ #{n}").rhs().idx()[0], "j_o"
        )
        gemmini = rewrite_expr(
            gemmini, gemmini.find(f"a2 = _ #{n}").rhs().idx()[1], "j_i"
        )
        gemmini = rewrite_expr(
            gemmini, gemmini.find(f"res[_] += _ #{n}").idx()[0], "j_o"
        )
        gemmini = rewrite_expr(
            gemmini, gemmini.find(f"res[_] += _ #{n}").idx()[1], "j_i"
        )
    gemmini = rewrite_expr(gemmini, gemmini.find("src_tmp = _").rhs().idx()[0], "j_o")
    gemmini = rewrite_expr(gemmini, gemmini.find("src_tmp = _").rhs().idx()[1], "j_i")
    gemmini = rewrite_expr(gemmini, gemmini.find("fc1_out[_] = _").idx()[0], "j_o")
    gemmini = rewrite_expr(gemmini, gemmini.find("fc1_out[_] = _").idx()[1], "j_i")

    gemmini = set_memory(gemmini, "res : _", GEMM_ACCUM)
    for n in (1, 0):
        gemmini = set_memory(gemmini, f"w_s : _ #{n}", GEMM_SCRATCH)
        gemmini = set_memory(gemmini, f"i_s : _ #{n}", GEMM_SCRATCH)
    gemmini = replace(gemmini, "for j in _:_ #0", ld_acc_i32_col)
    for _ in range(2):
        gemmini = replace(gemmini, "for j in _:_ #0", ld_i8_id1)
        gemmini = replace(gemmini, "for k_i in _:_ #0", ld_i8_col)
        gemmini = replace(gemmini, "for j_o in _:_ #0", matmul_acc_i8)
    gemmini = hoist_acc_scale(gemmini, 2)
    gemmini = replace(gemmini, "for j_o in _:_ #0", st_acc_i8_act)
    return fence_after(gemmini, "st_acc_i8_act(_)")


def schedule_eval():
    gemmini = rename(braggnn_eval_cpu, "braggnn_eval")
    gemmini = call_eqv(gemmini, "braggnn_inference_cpu(_)", braggnn_inference)
    gemmini = inline(gemmini, "braggnn_inference(_)")
    return sched_fc1_flat(gemmini)


braggnn_eval = schedule_eval()

__all__ = ["braggnn_eval"]
