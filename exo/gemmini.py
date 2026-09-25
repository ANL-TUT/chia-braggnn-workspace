# ruff: noqa: F821, RUF016

from __future__ import annotations

from exo.API_scheduling import (
    bind_config,
    make_instr,
    rename,
    reorder_stmts,
    replace,
    simplify,
    write_config,
)
from exo.core.extern import Extern, _EErr
from exo.core.LoopIR import T
from exo.libs.externs import relu, select
from exo.libs.memories import DRAM_STATIC, GEMM_ACCUM, GEMM_SCRATCH
from exo.platforms.gemmini import (
    ConfigStore,
    _gemm_st_acc_i8,
    acc_scale,
    clamp,
    config_st_acc_i8,
    old_fission_after,
    set_prec_mem,
    st_acc_i8,
)

from exo import DRAM, config, instr


class _RdCycle(Extern):
    def __init__(self):
        super().__init__("rdcycle")

    def typecheck(self, args):
        if len(args) != 0:
            raise _EErr(f"expected 0 arguments, got {len(args)}")
        return T.i32

    def globl(self, prim_type):
        return (
            "static inline int32_t _rdcycle(void) {\n"
            "  uint64_t c;\n"
            '  asm volatile("rdcycle %0" : "=r"(c));\n'
            "  return (int32_t)c;\n"
            "}\n"
        )

    def compile(self, args, prim_type):
        return "_rdcycle()"


rdcycle = _RdCycle()

_gemm_fence = "gemmini_fence();"


@instr(_gemm_fence)
def fence():
    pass


# Per-layer profiling. braggnn_reference.py puts a layer_mark(id) after every
# layer (id 1 at the start of the patch); braggnn_main.c runs braggnn_eval a
# second time with chia_profile_layers set, and then each mark fences Gemmini
# and charges the cycles since the previous mark to layer `id` (names in
# braggnn_main.c). In the timed pass a mark is a load and a branch.
@instr(
    "if (chia_profile_layers) chia_layer_mark({id});",
    c_global="extern volatile int chia_profile_layers;\n"
    "void chia_layer_mark(int id);\n",
)
def layer_mark(id: size):
    pass


def make_config_ld(ld_id=0):
    def new_config_ld():
        if ld_id == 0:

            @config
            class ConfigLoad:
                src_stride: stride
                block_stride: stride

            return ConfigLoad
        if ld_id == 1:

            @config
            class ConfigLoad_id1:
                src_stride: stride
                block_stride: stride

            return ConfigLoad_id1

        @config
        class ConfigLoad_id2:
            src_stride: stride
            block_stride: stride

        return ConfigLoad_id2

    config_load = new_config_ld()

    if ld_id == 0:

        @instr("gemmini_extended3_config_ld(0, 1.0f, 0, 0);")
        def config_ld_repeat():
            config_load.src_stride = 0
            config_load.block_stride = 16

        return config_load, config_ld_repeat

    @instr(
        "gemmini_extended4_config_ld({src_stride}, 1.0f, 0, "
        "{block_stride}/16, " + str(ld_id) + ");"
    )
    def config_ld_i8_block(src_stride: stride, block_stride: stride):
        config_load.src_stride = src_stride
        config_load.block_stride = block_stride

    return config_load, rename(
        config_ld_i8_block, f"config_ld_i8_block_id{ld_id}"
    )


ConfigLoad, config_ld_repeat = make_config_ld()
ConfigLoad_id1, config_ld_i8_block_id1 = make_config_ld(1)
ConfigLoad_id2, config_ld_i8_block_id2 = make_config_ld(2)


_gemm_do_ld_i8_block_strided_id1 = (
    "gemmini_extended_mvin2(((uintptr_t) &{src_data}), "
    "((uint32_t )(uintptr_t) &{dst_data}), 16*{m}, {n});"
)
_gemm_do_ld_i8_block_strided_id2 = (
    "gemmini_extended_mvin3(((uintptr_t) &{src_data}), "
    "((uint32_t )(uintptr_t) &{dst_data}), 16*{m}, {n});"
)
_gemm_config_ld_i8_block_strided_id1 = (
    "gemmini_extended4_config_ld({src}.strides[0], 1.0f, 0, "
    "{dst}.strides[0]/16, 1);\n"
)
_gemm_config_ld_i8_block_strided_id2 = (
    "gemmini_extended4_config_ld({src}.strides[0], 1.0f, 0, "
    "{dst}.strides[0]/16, 2);\n"
)
_gemm_ld_i8_block_strided_id1 = (
    _gemm_config_ld_i8_block_strided_id1 + _gemm_do_ld_i8_block_strided_id1
)
_gemm_ld_i8_block_strided_id2 = (
    _gemm_config_ld_i8_block_strided_id2 + _gemm_do_ld_i8_block_strided_id2
)


@instr(_gemm_ld_i8_block_strided_id1)
def ld_i8_block_strided(
    n: size,
    m: size,
    src: [i8][n, 16 * m] @ DRAM,
    dst: [i8][m, n, 16] @ GEMM_SCRATCH,
):
    assert n <= 16
    assert m <= 4
    assert stride(src, 1) == 1
    assert stride(dst, 1) == 16
    assert stride(dst, 2) == 1

    for i in seq(0, n):
        for j in seq(0, m):
            for k in seq(0, 16):
                dst[j, i, k] = src[i, k + 16 * j]


do_ld_i8_block_strided_id1 = make_instr(
    rename(ld_i8_block_strided, "do_ld_i8_block_strided_id1"),
    _gemm_do_ld_i8_block_strided_id1,
)
do_ld_i8_block_strided_id2 = make_instr(
    rename(ld_i8_block_strided, "do_ld_i8_block_strided_id2"),
    _gemm_do_ld_i8_block_strided_id2,
)


def make_ld_i8_block_strided(name, ld_id, p=ld_i8_block_strided, split=False):
    if ld_id == 1:
        config_load = ConfigLoad_id1
        config_ld = config_ld_i8_block_id1
        do_ld = do_ld_i8_block_strided_id1
        gemm_c = _gemm_ld_i8_block_strided_id1
    else:
        assert ld_id == 2
        config_load = ConfigLoad_id2
        config_ld = config_ld_i8_block_id2
        do_ld = do_ld_i8_block_strided_id2
        gemm_c = _gemm_ld_i8_block_strided_id2

    p = rename(p, name)
    if split:
        p = write_config(
            p, p.body().before(), config_load, "block_stride", "stride(dst, 0)"
        )
        p = write_config(
            p, p.body().before(), config_load, "src_stride", "stride(src, 0)"
        )
        p = replace(
            p,
            f"{config_load.name()}.src_stride = _ ; "
            f"{config_load.name()}.block_stride = _",
            config_ld,
        )
        p = replace(p, "for i in _:_", do_ld)
        return p
    return make_instr(p, gemm_c)


ld_i8_block_strided_id1 = make_ld_i8_block_strided(
    "ld_i8_block_strided_id1", 1
)
ld_i8_block_strided_id1_v2 = make_ld_i8_block_strided(
    "ld_i8_block_strided_id1_v2", 1, split=True
)
ld_i8_block_strided_id2 = make_ld_i8_block_strided(
    "ld_i8_block_strided_id2", 2
)
ld_i8_block_strided_id2_v2 = make_ld_i8_block_strided(
    "ld_i8_block_strided_id2_v2", 2, split=True
)


_gemm_config_ld_repeat = "gemmini_extended3_config_ld(0, 1.0f, 0, 0);"
_gemm_do_ld_acc_i32_repeat = (
    "gemmini_extended_mvin(((uintptr_t) &{src_data}), "
    "((uint32_t )(uintptr_t) &{dst_data}), {m}, {n});"
)


@instr(_gemm_config_ld_repeat + "\n" + _gemm_do_ld_acc_i32_repeat)
def ld_acc_i32_repeat(
    n: size,
    m: size,
    src: [i32][m] @ DRAM,
    dst: [i32][n, 16] @ GEMM_ACCUM,
):
    assert n <= 16
    assert m <= 16
    assert stride(src, 0) == 1
    assert stride(dst, 0) == 16
    assert stride(dst, 1) == 1

    for i in seq(0, n):
        for j in seq(0, m):
            dst[i, j] = src[j]


do_ld_acc_i32_repeat = make_instr(
    rename(ld_acc_i32_repeat, "do_ld_acc_i32_repeat"), _gemm_do_ld_acc_i32_repeat
)


def make_ld_acc_i32_repeat_v2(p=ld_acc_i32_repeat):
    p = rename(p, "ld_acc_i32_repeat_v2")
    p = write_config(p, p.body().before(), ConfigLoad, "block_stride", "16")
    p = write_config(p, p.body().before(), ConfigLoad, "src_stride", "0")
    p = replace(
        p,
        "ConfigLoad.src_stride = _ ; ConfigLoad.block_stride = _",
        config_ld_repeat,
    )
    p = replace(p, "for i in _:_", do_ld_acc_i32_repeat)
    return p


ld_acc_i32_repeat_v2 = make_ld_acc_i32_repeat_v2()


def make_st_acc_i8(name, act, p=st_acc_i8):
    p = rename(p.partial_eval(act=act), name)
    return make_instr(
        simplify(p), _gemm_st_acc_i8.replace("{act}", "RELU" if act else "0")
    )


st_acc_i8_no_act = make_st_acc_i8("st_acc_i8_no_act", False)
st_acc_i8_act = make_st_acc_i8("st_acc_i8_act", True)

_gemm_do_st_acc_i8 = (
    "gemmini_extended_mvout( ((uint64_t) &{dst_data}), "
    "(uint32_t) &{src_data}, {m}, {n} );"
)


@instr(_gemm_do_st_acc_i8)
def do_st_acc_i8_act(
    n: size,
    m: size,
    src: [i32][n, 16] @ GEMM_ACCUM,
    dst: [i8][n, m] @ DRAM,
):
    assert n <= 16
    assert m <= 16
    assert stride(dst, 1) == 1
    assert stride(src, 0) == 16
    assert stride(src, 1) == 1

    for i in seq(0, n):
        for j in seq(0, m):
            src_tmp: i32
            src_tmp = src[i, j]
            tmp: f32
            acc_scale(src_tmp, tmp, ConfigStore.scale)
            tmp2: i8
            clamp(tmp, tmp2)
            tmp2 = relu(tmp2)
            dst[i, j] = tmp2


def make_st_acc_i8_act_v2(p=st_acc_i8_act):
    p = rename(p, "st_acc_i8_act_v2")
    p = bind_config(p, "scale", ConfigStore, "scale")
    for pair in (
        "tmp : _ ; ConfigStore.scale = _",
        "src_tmp = _ ; ConfigStore.scale = _",
        "src_tmp : _ ; ConfigStore.scale = _",
    ):
        p = reorder_stmts(p, p.find(pair))
    p = old_fission_after(p, "ConfigStore.scale = _", n_lifts=2)
    p = write_config(
        p,
        p.find("ConfigStore.scale = _").after(),
        ConfigStore,
        "dst_stride",
        "stride(dst, 0)",
    )
    p = write_config(
        p,
        p.find("ConfigStore.dst_stride = _").after(),
        ConfigStore,
        "act",
        "True",
    )
    p = replace(p, "for i in _:_", do_st_acc_i8_act)
    p = replace(
        p,
        "ConfigStore.scale = _ ; ConfigStore.dst_stride = _ ; "
        "ConfigStore.act = _",
        config_st_acc_i8,
    )
    return p


st_acc_i8_act_v2 = make_st_acc_i8_act_v2()


_gemm_matmul_acc_trans_b = (
    "gemmini_extended_config_ex(WS, 0, 0, 1, 0, 1);\n"
    + "gemmini_extended_preload((uint32_t)(&{B_data}), "
    + "(uint32_t)(&{C_data}) | 0x40000000, {M}, {K}, {M}, {N});\n"
    + "gemmini_extended_compute_preloaded((uint32_t)(&{A_data}), "
    + "~((uint32_t)0), {K}, {N}, 16, 16);"
)


@instr(_gemm_matmul_acc_trans_b)
def matmul_acc_i8_trans_b(
    N: size,
    M: size,
    K: size,
    A: [i8][N, 16] @ GEMM_SCRATCH,
    B: [i8][M, 16] @ GEMM_SCRATCH,
    C: [i32][N, 16] @ GEMM_ACCUM,
):
    assert N <= 16
    assert M <= 16
    assert K <= 16

    for i in seq(0, N):
        for j in seq(0, M):
            for k in seq(0, K):
                a: i32
                b: i32
                a = A[i, k]
                b = B[j, k]
                C[i, j] += a * b


def new_config_matmul_trans_b():
    @config
    class ConfigMatmulTransB:
        done: bool

    return ConfigMatmulTransB


ConfigMatmulTransB = new_config_matmul_trans_b()

_gemm_config_matmul_trans_b = "gemmini_extended_config_ex(WS, 0, 0, 1, 0, 1);"


@instr(_gemm_config_matmul_trans_b)
def config_matmul_trans_b():
    ConfigMatmulTransB.done = True


_gemm_do_matmul_acc_trans_b = (
    "gemmini_extended_preload((uint32_t)(&{B_data}), "
    + "(uint32_t)(&{C_data}) | 0x40000000, {M}, {K}, {M}, {N});\n"
    + "gemmini_extended_compute_preloaded((uint32_t)(&{A_data}), "
    + "~((uint32_t)0), {K}, {N}, 16, 16);"
)

do_matmul_acc_i8_trans_b = make_instr(
    rename(matmul_acc_i8_trans_b, "do_matmul_acc_i8_trans_b"),
    _gemm_do_matmul_acc_trans_b,
)


def make_matmul_acc_i8_trans_b_v2(p=matmul_acc_i8_trans_b):
    p = rename(p, "matmul_acc_i8_trans_b_v2")
    p = write_config(p, p.body().before(), ConfigMatmulTransB, "done", "True")
    p = replace(p, "for i in _:_", do_matmul_acc_i8_trans_b)
    p = replace(p, "ConfigMatmulTransB.done = True", config_matmul_trans_b)
    p = make_instr(p, _gemm_do_matmul_acc_trans_b)
    return p


matmul_acc_i8_trans_b_v2 = make_matmul_acc_i8_trans_b_v2()


def _gemm_ld_acc_i8_scaled(acc):
    ld_id = "1" if acc else "0"
    mvin = "gemmini_extended_mvin2" if acc else "gemmini_extended_mvin"
    dst = "((uint32_t) &{dst_data}) | 0x40000000" if acc else "((uint32_t) &{dst_data})"
    return (
        "gemmini_extended4_config_ld({src}.strides[0]*1, "
        + "{scale}[0], true, DIM, "
        + ld_id
        + ");\n"
        + mvin
        + "( &{src_data}, "
        + dst
        + ", {m}, {n} );"
    )


@instr(_gemm_ld_acc_i8_scaled(False))
def ld_acc_i8_scaled(
    n: size,
    m: size,
    scale: f32 @ DRAM,
    src: [i8][n, m] @ DRAM,
    dst: [i32][n, 16] @ GEMM_ACCUM,
):
    assert n <= 16
    assert m <= 16
    assert stride(src, 1) == 1
    assert stride(dst, 0) == 16
    assert stride(dst, 1) == 1

    for i in seq(0, n):
        for j in seq(0, m):
            a_src: i32
            a_tmp: f32
            a_q: i8
            a2: i32
            a_src = src[i, j]
            acc_scale(a_src, a_tmp, scale)
            clamp(a_tmp, a_q)
            a2 = a_q
            dst[i, j] = a2


@instr(_gemm_ld_acc_i8_scaled(True))
def ld_acc_i8_scaled_acc(
    n: size,
    m: size,
    scale: f32 @ DRAM,
    src: [i8][n, m] @ DRAM,
    dst: [i32][n, 16] @ GEMM_ACCUM,
):
    assert n <= 16
    assert m <= 16
    assert stride(src, 1) == 1
    assert stride(dst, 0) == 16
    assert stride(dst, 1) == 1

    for i in seq(0, n):
        for j in seq(0, m):
            b_src: i32
            b_tmp: f32
            b_q: i8
            b2: i32
            b_src = src[i, j]
            acc_scale(b_src, b_tmp, scale)
            clamp(b_tmp, b_q)
            b2 = b_q
            dst[i, j] += b2


_gemm_ld_i8_col = (
    "gemmini_extended3_config_ld({src}.strides[0]*1, 1.0f, 0, 2);\n"
    + "gemmini_extended_mvin3( &{src_data}, "
    + "((uint64_t) &{dst_data}), 1, {n} );"
)


@instr(_gemm_ld_i8_col)
def ld_i8_col(
    n: size,
    src: [i8][n] @ DRAM,
    dst: [i8][n, 16] @ GEMM_SCRATCH,
):
    assert n <= 16
    assert stride(src, 0) == 1
    assert stride(dst, 0) == 16
    assert stride(dst, 1) == 1

    for i in seq(0, n):
        dst[i, 0] = src[i]


_gemm_ld_acc_i32_col = (
    "gemmini_extended3_config_ld({src}.strides[0]*4, 1.0f, 0, 0);\n"
    + "gemmini_extended_mvin( ((uint64_t) &{src_data}), "
    + "((uint32_t) &{dst_data}), 1, {n} );"
)

ld_acc_i32_col = rename(ld_i8_col, "ld_acc_i32_col")
ld_acc_i32_col = set_prec_mem(ld_acc_i32_col, "src", "i32", DRAM)
ld_acc_i32_col = set_prec_mem(ld_acc_i32_col, "dst", "i32", GEMM_ACCUM)
ld_acc_i32_col = make_instr(ld_acc_i32_col, _gemm_ld_acc_i32_col)


def _gemm_loop_softmax_flat(cols, row_scale=1):
    n_rows = "{n}" if row_scale == 1 else f"{row_scale} * {{n}}"
    rows = f"({n_rows} + 15) / 16"
    pad_rows = f"{rows} * 16 - {n_rows}"
    J, pad_J = _blocks(cols)
    entries = ", ".join(f"[{(cols + 1) * i}] = 1" for i in range(cols))
    qln2 = "(int)(0.693147 / {in_scale}[0])"
    qb = "(int32_t)(1.353f / {in_scale}[0])"
    qc = "(int32_t)(0.344f / (0.3585f * {in_scale}[0] * {in_scale}[0]))"
    return (
        "gemmini_extended_config_ex(WEIGHT_STATIONARY, (0), 0, 1, (0), (0));\n"
        + f"gemmini_extended_config_st(({cols}), 0, "
        + "1.0f / (127.0f * {out_scale}[0]));\n"
        + f"gemmini_extended3_config_ld(({cols}), 1.0f, false, (0));\n"
        + f"gemmini_extended3_config_ld(({cols}), 1.0f, false, (1));\n"
        + f"gemmini_extended3_config_ld(({cols}), 1.0f, false, (2));\n"
        + "gemmini_config_norm("
        + qln2
        + ", 0, 0, 1, 0, "
        + qb
        + ", "
        + qc
        + ");\n"
        + "gemmini_config_norm((65536 / "
        + qln2
        + "), 1, 0, 1, 0, "
        + qb
        + ", "
        + qc
        + ");\n"
        + "{{ "
        + f"static const int8_t identity[{cols} * {cols}] = "
        + "{{"
        + entries
        + "}};\n"
        + f"gemmini_loop_ws({rows}, {J}, {J}, {pad_rows}, {pad_J}, {pad_J}, "
        "&{A_data}, identity, 0, &{C_data}, "
        f"{cols}, {cols}, {cols}, {cols}, "
        "0, 0, 0, 0, 0, SOFTMAX, 1, 1, 0); }}"
    )


def make_loop_softmax_flat(name, cols, max_shift):
    @instr(_gemm_loop_softmax_flat(cols))
    def loop_softmax_flat(
        n: size,
        A: [i8][n, cols] @ DRAM,
        C: [i8][n, cols] @ DRAM,
        in_scale: f32 @ DRAM,
        out_scale: f32 @ DRAM,
    ):
        for i in seq(0, n):
            two16: i32
            two16 = 65536
            qln2: i32
            qln2_f: f32
            qln2_f = 0.693147 / in_scale
            qln2 = qln2_f
            qln2_inv: i32
            qln2_inv = two16 / qln2
            qb: i32
            qb_f: f32
            qb_f = 1.353 / in_scale
            qb = qb_f
            qc: i32
            qc_f: f32
            qc_f = 0.344 / (0.3585 * in_scale * in_scale)
            qc = qc_f

            max_q: i32
            max_q = A[i, 0]
            for j in seq(0, cols):
                a2: i32
                a2 = A[i, j]
                max_q = select(max_q, a2, a2, max_q)

            exp_buf: i32[cols] @ DRAM_STATIC
            sum_exp: i32
            sum_exp = 0
            for j in seq(0, cols):
                a2: i32
                a2 = A[i, j]
                neg_q: i32
                neg_q = max_q - a2
                z: i32
                z = neg_q * qln2_inv / two16
                qp: i32
                qp = z * qln2 - neg_q
                qpb: i32
                qpb = qp + qb
                q_exp: i32
                q_exp = qpb * qpb + qc

                zero: i32
                zero = 0
                shifts_left: i32
                shifts_left = z
                for s in seq(0, max_shift):
                    halved: i32
                    halved = q_exp / 2
                    q_exp = select(zero, shifts_left, halved, q_exp)
                    next_left: i32
                    next_left = shifts_left - 1
                    shifts_left = select(zero, shifts_left, next_left, shifts_left)

                exp_buf[j] = q_exp
                sum_exp += q_exp

            denom: f32
            denom = sum_exp
            factor: f32
            factor = 127.0 / denom
            out_factor: f32
            out_factor = 1.0 / (127.0 * out_scale)
            tmp_scale: f32
            tmp_scale = factor * out_factor
            for j in seq(0, cols):
                src_tmp: i32
                src_tmp = exp_buf[j]
                tmp_res1: f32
                acc_scale(src_tmp, tmp_res1, tmp_scale)
                tmp_res2: i8
                clamp(tmp_res1, tmp_res2)
                C[i, j] = tmp_res2

    return rename(loop_softmax_flat, name)


_gemm_ld_i8_im2col = (
    "gemmini_extended5_config_ld({src}.strides[0]*1, 1.0f, false, DIM, {w}, 2);\n"
    + "gemmini_extended_mvin3( &{src_data}, "
    + "((uint64_t) &{dst_data}), 1, {n} + {w} - 1 );"
)


@instr(_gemm_ld_i8_im2col)
def ld_i8_im2col(
    n: size,
    w: size,
    src: [i8][n + w - 1] @ DRAM,
    dst: [i8][n + w - 1, 16] @ GEMM_SCRATCH,
):
    assert n <= 16
    assert w <= 16
    assert stride(src, 0) == 1
    assert stride(dst, 0) == 16
    assert stride(dst, 1) == 1

    for i in seq(0, n):
        for j in seq(0, w):
            dst[i, j] = src[i + j]


def _blocks(n):
    tiles = (n + 15) // 16
    return tiles, tiles * 16 - n


def _gemm_loop_conv_ws(in_dim, in_ch, out_ch, k, act):
    out_dim = in_dim - k + 1
    max_pixels_per_row = 1 if in_ch > 16 else min(16 // in_ch, k)
    gemm_act = "RELU" if act else "NO_ACTIVATION"
    scale = "({scale}[0] * {post_scale}[0])" if act else "{scale}[0]"
    return (
        f"gemmini_extended_config_st(({out_ch}), {gemm_act}, "
        + scale
        + ");\n"
        + "gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, 0, 1, 1, 0, (0), 0);\n"
        + f"gemmini_loop_conv_ws(1, {in_dim}, {in_dim}, {in_ch}, {out_ch}, "
        f"{out_dim}, {out_dim}, {out_dim}, {out_dim}, 1, 0, {k}, 1, 1, 1, 0, "
        f"1, {out_dim}, {out_dim}, {out_ch}, {k}, {k}, {in_ch}, "
        "0, 0, 0, 0, 0, 0, 0, 0, "
        f"{out_dim}, {out_dim}, "
        "{weights}, {output}, {bias}, {inp}, 0, 1, 0, 0, 0, "
        f"{gemm_act}, 0, 0, 0, 0, {max_pixels_per_row}, "
        f"{in_ch}, {out_ch}, {out_ch}, 0, 1, 1);"
    )


def make_loop_conv_ws(name, in_dim, in_ch, out_ch, k, act=False):
    out_dim = in_dim - k + 1
    c_instr = _gemm_loop_conv_ws(in_dim, in_ch, out_ch, k, act)

    if act:

        @instr(c_instr)
        def loop_conv_ws(
            inp: i8[in_dim, in_dim, in_ch] @ DRAM,
            weights: i8[k, k, in_ch, out_ch] @ DRAM,
            bias: i32[out_ch] @ DRAM,
            output: i8[out_dim, out_dim, out_ch] @ DRAM,
            scale: f32 @ DRAM,
            post_scale: f32 @ DRAM,
        ):
            for orow in seq(0, out_dim):
                for ocol in seq(0, out_dim):
                    for och in seq(0, out_ch):
                        res: i32
                        res = bias[och]

                        for krow in seq(0, k):
                            for kcol in seq(0, k):
                                for kch in seq(0, in_ch):
                                    w_s: i8 @ DRAM
                                    w_s = weights[krow, kcol, kch, och]
                                    i_s: i8 @ DRAM
                                    i_s = inp[orow + krow, ocol + kcol, kch]
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
                        output[orow, ocol, och] = tmp_res2

    else:

        @instr(c_instr)
        def loop_conv_ws(
            inp: i8[in_dim, in_dim, in_ch] @ DRAM,
            weights: i8[k, k, in_ch, out_ch] @ DRAM,
            bias: i32[out_ch] @ DRAM,
            output: i8[out_dim, out_dim, out_ch] @ DRAM,
            scale: f32 @ DRAM,
        ):
            for orow in seq(0, out_dim):
                for ocol in seq(0, out_dim):
                    for och in seq(0, out_ch):
                        res: i32
                        res = bias[och]

                        for krow in seq(0, k):
                            for kcol in seq(0, k):
                                for kch in seq(0, in_ch):
                                    w_s: i8 @ DRAM
                                    w_s = weights[krow, kcol, kch, och]
                                    i_s: i8 @ DRAM
                                    i_s = inp[orow + krow, ocol + kcol, kch]
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
                        output[orow, ocol, och] = tmp_res2

    return rename(loop_conv_ws, name)


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


def make_loop_matmul_fc(name, in_ch, in_h, in_w, out_features, act=False):
    c_instr = _gemm_loop_matmul_fc(in_ch * in_h * in_w, out_features, act)

    if act:

        @instr(c_instr)
        def loop_matmul_fc(
            inp: i8[in_ch, in_h, in_w] @ DRAM,
            weights: i8[out_features, in_ch, in_h, in_w] @ DRAM,
            bias: i32[out_features] @ DRAM,
            output: i8[out_features, 1, 1] @ DRAM,
            scale: f32 @ DRAM,
            post_scale: f32 @ DRAM,
        ):
            for j in seq(0, out_features):
                res: i32
                res = bias[j]
                for kch in seq(0, in_ch):
                    for krow in seq(0, in_h):
                        for kcol in seq(0, in_w):
                            w_s: i8 @ DRAM
                            w_s = weights[j, kch, krow, kcol]
                            i_s: i8 @ DRAM
                            i_s = inp[kch, krow, kcol]
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
        def loop_matmul_fc(
            inp: i8[in_ch, in_h, in_w] @ DRAM,
            weights: i8[out_features, in_ch, in_h, in_w] @ DRAM,
            bias: i32[out_features] @ DRAM,
            output: i8[out_features, 1, 1] @ DRAM,
            scale: f32 @ DRAM,
        ):
            for j in seq(0, out_features):
                res: i32
                res = bias[j]
                for kch in seq(0, in_ch):
                    for krow in seq(0, in_h):
                        for kcol in seq(0, in_w):
                            w_s: i8 @ DRAM
                            w_s = weights[j, kch, krow, kcol]
                            i_s: i8 @ DRAM
                            i_s = inp[kch, krow, kcol]
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

    return rename(loop_matmul_fc, name)


def _gemm_loop_matmul_trans_b(m, d):
    dd = d * d
    J, pad_J = _blocks(dd)
    K, pad_K = _blocks(m)
    rows = "(" + str(d) + " * {n} + 15) / 16"
    pad_rows = rows + " * 16 - " + str(d) + " * {n}"
    return (
        "gemmini_extended_config_ex(WEIGHT_STATIONARY, (0), 0, 1, (0), (1));\n"
        + f"gemmini_extended_config_st(({dd}), NO_ACTIVATION, "
        + "{scale}[0]);\n"
        + f"gemmini_extended3_config_ld(({m}), 1.0f, false, (0));\n"
        + f"gemmini_extended3_config_ld(({m}), 1.0f, false, (1));\n"
        + f"gemmini_extended3_config_ld(({dd}), 1.0f, false, (2));\n"
        + f"gemmini_loop_ws({rows}, {J}, {K}, {pad_rows}, {pad_J}, {pad_K}, "
        "&{A_data}, &{B_data}, 0, &{C_data}, "
        f"{m}, {m}, {dd}, {dd}, 0, 1, 0, 0, 0, NO_ACTIVATION, 1, 1, 0);"
    )


def make_loop_matmul_trans_b(name, m, d):
    @instr(_gemm_loop_matmul_trans_b(m, d))
    def loop_matmul_trans_b(
        n: size,
        A: [i8][n, d, m] @ DRAM,
        B: [i8][d, d, m] @ DRAM,
        C: [i8][n, d, d, d] @ DRAM,
        scale: f32 @ DRAM,
    ):
        for i1 in seq(0, n):
            for i2 in seq(0, d):
                for j1 in seq(0, d):
                    for j2 in seq(0, d):
                        res: i32
                        res = 0
                        for k in seq(0, m):
                            a: i8 @ DRAM
                            a = A[i1, i2, k]
                            b: i8 @ DRAM
                            b = B[j1, j2, k]
                            a2: i32
                            b2: i32
                            a2 = a
                            b2 = b
                            res += a2 * b2

                        src_tmp: i32
                        src_tmp = res
                        tmp_res1: f32
                        acc_scale(src_tmp, tmp_res1, scale)
                        tmp_res2: i8
                        clamp(tmp_res1, tmp_res2)
                        C[i1, i2, j1, j2] = tmp_res2

    return rename(loop_matmul_trans_b, name)


def _gemm_loop_matmul(m, d):
    dd = d * d
    I, pad_I = _blocks(dd)
    J, pad_J = _blocks(m)
    return (
        "gemmini_extended_config_ex(WEIGHT_STATIONARY, (0), 0, 1, (0), (0));\n"
        + f"gemmini_extended_config_st(({m}), NO_ACTIVATION, "
        + "{scale}[0]);\n"
        + f"gemmini_extended3_config_ld(({dd}), 1.0f, false, (0));\n"
        + f"gemmini_extended3_config_ld(({m}), 1.0f, false, (1));\n"
        + f"gemmini_extended3_config_ld(({m}), 1.0f, false, (2));\n"
        + f"gemmini_loop_ws({I}, {J}, {I}, {pad_I}, {pad_J}, {pad_I}, "
        "{A}, {B}, 0, {C}, "
        f"{dd}, {m}, {m}, {m}, 0, 0, 0, 0, 0, NO_ACTIVATION, 1, 1, 0);"
    )


def make_loop_matmul(name, m, d):
    @instr(_gemm_loop_matmul(m, d))
    def loop_matmul(
        A: i8[d, d, d, d] @ DRAM,
        B: i8[d, d, m] @ DRAM,
        C: i8[d, d, m] @ DRAM,
        scale: f32 @ DRAM,
    ):
        for i1 in seq(0, d):
            for i2 in seq(0, d):
                for j in seq(0, m):
                    res: i32
                    res = 0
                    for k1 in seq(0, d):
                        for k2 in seq(0, d):
                            a: i8 @ DRAM
                            a = A[i1, i2, k1, k2]
                            b: i8 @ DRAM
                            b = B[k1, k2, j]
                            a2: i32
                            b2: i32
                            a2 = a
                            b2 = b
                            res += a2 * b2

                    src_tmp: i32
                    src_tmp = res
                    tmp_res1: f32
                    acc_scale(src_tmp, tmp_res1, scale)
                    tmp_res2: i8
                    clamp(tmp_res1, tmp_res2)
                    C[i1, i2, j] = tmp_res2

    return rename(loop_matmul, name)


def _gemm_loop_resadd(d, ch):
    I, pad_I = _blocks(d * d)
    J, pad_J = _blocks(ch)
    return (
        f"gemmini_extended_config_st(({ch}), RELU, "
        + "{C_scale}[0]);\n"
        + "gemmini_extended_config_ex(WEIGHT_STATIONARY, (0), 0, 1, (0), (0));\n"
        + f"gemmini_extended4_config_ld(({ch}), "
        + "{A_scale}[0], true, DIM, (0));\n"
        + f"gemmini_extended4_config_ld(({ch}), "
        + "{B_scale}[0], true, DIM, (1));\n"
        + f"gemmini_loop_ws({I}, {J}, 0, {pad_I}, {pad_J}, 0, "
        "{A}, {B}, 0, {C}, "
        f"{ch}, {ch}, 0, {ch}, 0, 0, 0, 0, 0, RELU, 0, 0, 1);"
    )


def make_loop_resadd(name, d, ch):
    @instr(_gemm_loop_resadd(d, ch))
    def loop_resadd(
        A_scale: f32 @ DRAM,
        B_scale: f32 @ DRAM,
        C_scale: f32 @ DRAM,
        A: i8[d, d, ch] @ DRAM,
        B: i8[d, d, ch] @ DRAM,
        C: i8[d, d, ch] @ DRAM,
    ):
        for i1 in seq(0, d):
            for i2 in seq(0, d):
                for j in seq(0, ch):
                    a_src: i32
                    a_tmp: f32
                    a_q: i8
                    a2: i32
                    a_src = A[i1, i2, j]
                    acc_scale(a_src, a_tmp, A_scale)
                    clamp(a_tmp, a_q)
                    a2 = a_q

                    b_src: i32
                    b_tmp: f32
                    b_q: i8
                    b2: i32
                    b_src = B[i1, i2, j]
                    acc_scale(b_src, b_tmp, B_scale)
                    clamp(b_tmp, b_q)
                    b2 = b_q

                    res: i32
                    src_tmp: i32
                    tmp_res1: f32
                    tmp_res2: i8
                    res = a2 + b2
                    src_tmp = res
                    acc_scale(src_tmp, tmp_res1, C_scale)
                    clamp(tmp_res1, tmp_res2)
                    tmp_res2 = relu(tmp_res2)
                    C[i1, i2, j] = tmp_res2

    return rename(loop_resadd, name)


def make_loop_softmax(name, d, max_shift):
    @instr(_gemm_loop_softmax_flat(d * d, row_scale=d))
    def loop_softmax(
        n: size,
        A: [i8][n, d, d, d] @ DRAM,
        C: [i8][n, d, d, d] @ DRAM,
        in_scale: f32 @ DRAM,
        out_scale: f32 @ DRAM,
    ):
        for i1 in seq(0, n):
            for i2 in seq(0, d):
                two16: i32
                two16 = 65536
                qln2: i32
                qln2_f: f32
                qln2_f = 0.693147 / in_scale
                qln2 = qln2_f
                qln2_inv: i32
                qln2_inv = two16 / qln2
                qb: i32
                qb_f: f32
                qb_f = 1.353 / in_scale
                qb = qb_f
                qc: i32
                qc_f: f32
                qc_f = 0.344 / (0.3585 * in_scale * in_scale)
                qc = qc_f

                max_q: i32
                max_q = A[i1, i2, 0, 0]
                for j1 in seq(0, d):
                    for j2 in seq(0, d):
                        a2: i32
                        a2 = A[i1, i2, j1, j2]
                        max_q = select(max_q, a2, a2, max_q)

                exp_buf: i32[d, d] @ DRAM_STATIC
                sum_exp: i32
                sum_exp = 0
                for j1 in seq(0, d):
                    for j2 in seq(0, d):
                        a2: i32
                        a2 = A[i1, i2, j1, j2]
                        neg_q: i32
                        neg_q = max_q - a2
                        z: i32
                        z = neg_q * qln2_inv / two16
                        qp: i32
                        qp = z * qln2 - neg_q
                        qpb: i32
                        qpb = qp + qb
                        q_exp: i32
                        q_exp = qpb * qpb + qc

                        zero: i32
                        zero = 0
                        shifts_left: i32
                        shifts_left = z
                        for s in seq(0, max_shift):
                            halved: i32
                            halved = q_exp / 2
                            q_exp = select(zero, shifts_left, halved, q_exp)
                            next_left: i32
                            next_left = shifts_left - 1
                            shifts_left = select(
                                zero, shifts_left, next_left, shifts_left
                            )

                        exp_buf[j1, j2] = q_exp
                        sum_exp += q_exp

                denom: f32
                denom = sum_exp
                factor: f32
                factor = 127.0 / denom
                out_factor: f32
                out_factor = 1.0 / (127.0 * out_scale)
                tmp_scale: f32
                tmp_scale = factor * out_factor
                for j1 in seq(0, d):
                    for j2 in seq(0, d):
                        src_tmp: i32
                        src_tmp = exp_buf[j1, j2]
                        tmp_res1: f32
                        acc_scale(src_tmp, tmp_res1, tmp_scale)
                        tmp_res2: i8
                        clamp(tmp_res1, tmp_res2)
                        C[i1, i2, j1, j2] = tmp_res2

    return rename(loop_softmax, name)
