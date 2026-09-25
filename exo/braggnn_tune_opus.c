// BraggNN inference on Rocket + Gemmini (DIM 16), hand-tuned in plain C.
//
// Starting point: the C that Exo generated for the best AlphaEvolve schedule
// (39,950 cycles/patch on FireSim: shared fence over the three NLB qkv convs,
// no NLB row tiling, unrolled CPU loops). Every Gemmini call below is that
// schedule's, unless an OPT_* flag below replaces it. Each optimization is a
// compile-time switch so its effect can be measured on its own; all of them
// must keep the predictions bit-identical to the seed's.
//
//   OPT_FC1_NO_FLATTEN  fc1 reads conv3's HWC output directly: its weights
//                       are permuted once, before the patch loop (like the
//                       seed's weight staging), so the per-patch CPU flatten
//                       disappears. Integer sums, so the result is identical.
//   OPT_SOFTMAX_LOAD    the softmax loop loads its 81x81 input straight into
//                       the accumulator (resadd-style: A*1 + A*0, K = 0)
//                       instead of multiplying it by an 81x81 identity matrix.
//   OPT_TAIL_ON_CPU     fc2, fc3, fc4 and fc_output (172 MACs in total) run on
//                       the CPU with Gemmini's own output arithmetic
//                       (scale_and_sat: round-half-even scale, saturate, ReLU),
//                       saving four tiny Gemmini loops and their fences.
//   OPT_SOFTMAX_ONE_LOAD  with OPT_SOFTMAX_LOAD: mark the loop's B load as already
//                       done (LOOP_WS rs2 bit 4, LoopMatmul's ldb_started/_completed),
//                       so A is loaded once. In a resadd loop ldA overwrites the
//                       accumulator and ldB adds to it, and the store waits for
//                       both loaders, so the result is the same A. B stays = A
//                       with scale 0, so hardware that ignores the bit is still
//                       correct.
//   OPT_NLB_SPAD_CHAIN  with OPT_SOFTMAX_LOAD: the softmax loop stores its 81x81
//                       output into the scratchpad (LOOP_WS spad_only store,
//                       LoopMatmulStCSpad) instead of DRAM, and attention*g reads
//                       it there as A (a_ex_spad_id = 0 -> a_addr_start, ldA
//                       marked done): one DRAM store + reload less. Tile (i,j) of
//                       the store lands at dst + (i*6+j)*16, exactly where the
//                       execute unit reads A tile (i,k) of a K = 6 loop.
//   OPT_QKV_SPAD_CHAIN  with OPT_QKV_LOOP_WS: the three qkv matmuls store theta, phi
//                       and g into the scratchpad (spad_only store; no activation,
//                       so no normalizer involved), and theta*phi^T reads both its
//                       operands there, attention*g its B (g): the loads of those
//                       loops are marked done. Scratchpad banks (4 x 4096 rows):
//                       loop-loaded A in bank 0, B in bank 1 (the default regions),
//                       theta and g in bank 2, phi in bank 3, so every loop reads
//                       A and B from different banks and the qkv stores do not hit
//                       the banks that the qkv loop itself reads.
//   OPT_CONV1_IM2COL    conv1 (one input channel) as a matmul: the CPU writes the
//                       quantized input straight into an 81x16 im2col matrix
//                       (9 taps, zero-padded), Gemmini multiplies it by the
//                       9x64 weights (+ bias). Same integer sums.
//   OPT_QKV_LOOP_WS     the three NLB 1x1 convs (same input) as loop_ws matmuls;
//   OPT_QKV_REUSE_A     with it, the second and third skip reloading the input
//                       from DRAM (A = NULL, same a_spad_id: gemmini.h's a_reuse).
//
// Default: the best measured combination (FireSim, predictions identical to
// the seed's): OPT_FC1_NO_FLATTEN + OPT_SOFTMAX_LOAD + OPT_SOFTMAX_ONE_LOAD +
// OPT_QKV_LOOP_WS + OPT_QKV_REUSE_A, 34,092 cycles/patch against 39,953 for
// OPT_BASELINE (the Exo build's calls). OPT_CONV1_IM2COL (+2.2k in conv1) and
// OPT_TAIL_ON_CPU (slower than four tiny Gemmini loops) measured worse, and
// OPT_NLB_SPAD_CHAIN (29,652) changes the predictions (9 of 10 patches), so
// they stay off.
//
// Build:  scripts/build_c_tune.sh exo/braggnn_tune_opus.c [-DOPT_BASELINE] [-DOPT_...]
// Profile per layer: add -DCHIA_LAYER_MARKS (braggnn_main.c prints the marks).

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "braggnn_schedule.h"

#ifndef OPT_BASELINE
#define OPT_FC1_NO_FLATTEN
#define OPT_SOFTMAX_LOAD
#define OPT_SOFTMAX_ONE_LOAD
#define OPT_QKV_LOOP_WS
#define OPT_QKV_REUSE_A
#endif

#ifdef CHIA_LAYER_MARKS
void chia_mark(int i, const char *name);
#define MARK(i, name) do { gemmini_fence(); chia_mark((i), (name)); } while (0)
#else
#define MARK(i, name) do { } while (0)
#endif

// Every buffer Gemmini reads or writes is 64-byte aligned: where the harness's
// arrays happen to land otherwise changes DMA cost (the same Gemmini calls
// measured 44.8k cycles on the harness arrays vs 39.95k in the Exo build,
// which stages every weight into its own static buffer).
#define ALIGNED __attribute__((aligned(64)))
#define STAGE(dst, src) memcpy((dst), (src), sizeof(dst))

static inline int32_t rdcycle32(void) {
  uint64_t c;
  asm volatile("rdcycle %0" : "=r"(c));
  return (int32_t)c;
}

// gemmini_loop_ws with extra LOOP_WS rs2 bits: bits 3..7 mark the ldA, ldB,
// ldD, ex and st stages of the loop as already started and completed.
#define LOOP_SKIP_LDA (1 << 3)
#define LOOP_SKIP_LDB (1 << 4)
#define LOOP_SPAD_ONLY (1 << 9)
// OPT_QKV_SPAD_CHAIN placement (12 tiles = 192 rows each; bank = row / 4096).
#define SPAD_BANK_ROWS 4096
#define THETA_SPAD_ROW (2 * SPAD_BANK_ROWS)
#define G_SPAD_ROW (2 * SPAD_BANK_ROWS + 256)
#define PHI_SPAD_ROW (3 * SPAD_BANK_ROWS)
#define LOOP_REGION1_A_START 0
#define LOOP_REGION1_B_END (2 * SPAD_BANK_ROWS)  // max_addr / concurrent_loops

#define LOOP_BOUNDS(I, J, K, pad_I, pad_J, pad_K)                                             \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC,                                                        \
                           ((uint64_t)(pad_K) << 32) | ((uint64_t)(pad_J) << 16) | (pad_I),     \
                           ((uint64_t)(K) << 32) | ((uint64_t)(J) << 16) | (I),                  \
                           k_LOOP_WS_CONFIG_BOUNDS)

// Scratchpad rows for the softmax output in OPT_NLB_SPAD_CHAIN: 36 tiles of 16
// rows, clear of the loop A region (from row 0) and B region (ending at
// max_addr / concurrent_loops) that the neighbouring loops use.
#define SOFTMAX_SPAD_ROW 4096
#define gemmini_loop_ws_skip(I, J, K, pad_I, pad_J, pad_K, A, B, D, C, A_stride, B_stride, D_stride, C_stride, A_transpose, B_transpose, full_C, low_D, ex_accumulate, act, a_spad_id, b_spad_id, is_resadd, skips) \
  { \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(pad_K) << 32) | ((uint64_t)(pad_J) << 16) | (uint64_t)(pad_I), ((uint64_t)(K) << 32) | ((uint64_t)(J) << 16) | (uint64_t)(I), k_LOOP_WS_CONFIG_BOUNDS) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, A, B, k_LOOP_WS_CONFIG_ADDRS_AB) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, D, C, k_LOOP_WS_CONFIG_ADDRS_DC) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, A_stride, B_stride, k_LOOP_WS_CONFIG_STRIDES_AB) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, D_stride, C_stride, k_LOOP_WS_CONFIG_STRIDES_DC) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(a_spad_id) << 18) | ((uint64_t)(b_spad_id) << 16) | ((uint64_t)(act) << 8) | ((low_D) << 2) | ((full_C) << 1) | (ex_accumulate), (skips) | ((is_resadd) << 2) | ((B_transpose) << 1) | (A_transpose), k_LOOP_WS) \
  }

// ---------------------------------------------------------------- conv layers

static void conv1(const int8_t *inp, const int8_t *w, const int32_t *bias,
                  int8_t *out, float scale) {
  gemmini_extended_config_st(64, NO_ACTIVATION, scale);
  gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, 0, 1, 1, 0, 0, 0);
  gemmini_loop_conv_ws(1, 11, 11, 1, 64, 9, 9, 9, 9, 1, 0, 3, 1, 1, 1, 0, 1, 9, 9, 64, 3, 3, 1,
                       0, 0, 0, 0, 0, 0, 0, 0, 9, 9, w, out, bias, inp, 0, 1, 0, 0, 0,
                       NO_ACTIVATION, 0, 0, 0, 0, 3, 1, 64, 64, 0, 1, 1);
  gemmini_fence();
}

static void conv2(const int8_t *inp, const int8_t *w, const int32_t *bias,
                  int8_t *out, float scale) {
  gemmini_extended_config_st(32, RELU, scale);
  gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, 0, 1, 1, 0, 0, 0);
  gemmini_loop_conv_ws(1, 9, 9, 64, 32, 7, 7, 7, 7, 1, 0, 3, 1, 1, 1, 0, 1, 7, 7, 32, 3, 3, 64,
                       0, 0, 0, 0, 0, 0, 0, 0, 7, 7, w, out, bias, inp, 0, 1, 0, 0, 0,
                       RELU, 0, 0, 0, 0, 1, 64, 32, 32, 0, 1, 1);
  gemmini_fence();
}

static void conv3(const int8_t *inp, const int8_t *w, const int32_t *bias,
                  int8_t *out, float scale) {
  gemmini_extended_config_st(8, RELU, scale);
  gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, 0, 1, 1, 0, 0, 0);
  gemmini_loop_conv_ws(1, 7, 7, 32, 8, 5, 5, 5, 5, 1, 0, 3, 1, 1, 1, 0, 1, 5, 5, 8, 3, 3, 32,
                       0, 0, 0, 0, 0, 0, 0, 0, 5, 5, w, out, bias, inp, 0, 1, 0, 0, 0,
                       RELU, 0, 0, 0, 0, 1, 32, 8, 8, 0, 1, 1);
  gemmini_fence();
}

#ifdef OPT_CONV1_IM2COL
// C[81 x 64] = im2col[81 x 16] * W[16 x 64] + bias; rows 9..15 of W are zero.
static void conv1_matmul(const int8_t *cols, const int8_t *w16, const int32_t *bias,
                         int8_t *out, float scale) {
  gemmini_extended_config_ex(WEIGHT_STATIONARY, 0, 0, 1, 0, 0);
  gemmini_extended_config_st(64, NO_ACTIVATION, scale);
  gemmini_extended3_config_ld(16, 1.0f, false, 0);
  gemmini_extended3_config_ld(64, 1.0f, false, 1);
  gemmini_extended3_config_ld(0, 1.0f, false, 2);
  gemmini_loop_ws(6, 4, 1, 15, 0, 0, cols, w16, bias, out, 16, 64, 0, 64,
                  0, 0, 0, 0, 1, NO_ACTIVATION, 1, 1, 0);
  gemmini_fence();
}
#endif

// 1x1 conv, 9x9 pixels: in_ch -> out_ch. No fence (the caller fences).
static void conv1x1(const int8_t *inp, int in_ch, const int8_t *w, const int32_t *bias,
                    int8_t *out, int out_ch, float scale) {
  gemmini_extended_config_st(out_ch, NO_ACTIVATION, scale);
  gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, 0, 1, 1, 0, 0, 0);
  gemmini_loop_conv_ws(1, 9, 9, in_ch, out_ch, 9, 9, 9, 9, 1, 0, 1, 1, 1, 1, 0, 1, 9, 9, out_ch,
                       1, 1, in_ch, 0, 0, 0, 0, 0, 0, 0, 0, 9, 9, w, out, bias, inp, 0, 1, 0, 0,
                       0, NO_ACTIVATION, 0, 0, 0, 0, 1, in_ch, out_ch, out_ch, 0, 1, 1);
}

#ifdef OPT_QKV_LOOP_WS
// The same 1x1 conv as a matmul: C[81 x 32] = A[81 x 64] * W[64 x 32] + bias,
// the bias a repeating D row (D stride 0), as tiled_matmul_outer issues it.
// A second call with inp == NULL reuses the A that the previous call left in
// scratchpad region a_spad_id = 1 (gemmini.h's a_reuse path).
static void qkv_matmul(const int8_t *inp, const int8_t *w, const int32_t *bias, int8_t *out,
                       float scale) {
  gemmini_extended_config_ex(WEIGHT_STATIONARY, 0, 0, 1, 0, 0);
  gemmini_extended_config_st(32, NO_ACTIVATION, scale);
  gemmini_extended3_config_ld(64, 1.0f, false, 0);
  gemmini_extended3_config_ld(32, 1.0f, false, 1);
  gemmini_extended3_config_ld(0, 1.0f, false, 2);
  // I = 6 tiles of 81 rows (pad 15), J = 2 tiles of 32, K = 4 tiles of 64.
  gemmini_loop_ws(6, 2, 4, 15, 0, 0, inp, w, bias, out, 64, 32, 0, 32,
                  0, 0, 0, 0, 1, NO_ACTIVATION, 1, 1, 0);
}
#endif

#ifdef OPT_QKV_SPAD_CHAIN
// qkv_matmul, but C goes to scratchpad row dst_row (spad_only store). With
// spad_only the execute unit takes A / B from the SPAD_AB addresses, so set
// them to where the id-1 loaders put A and B (region 1). inp == NULL reuses A.
static void qkv_matmul_to_spad(const int8_t *inp, const int8_t *w, const int32_t *bias,
                               uint32_t dst_row, float scale) {
  gemmini_extended_config_ex(WEIGHT_STATIONARY, 0, 0, 1, 0, 0);
  gemmini_extended_config_st(32, NO_ACTIVATION, scale);
  gemmini_extended3_config_ld(64, 1.0f, false, 0);
  gemmini_extended3_config_ld(32, 1.0f, false, 1);
  gemmini_extended3_config_ld(0, 1.0f, false, 2);
  LOOP_BOUNDS(6, 2, 4, 15, 0, 0);
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, inp, w, k_LOOP_WS_CONFIG_ADDRS_AB);
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, bias, 0, k_LOOP_WS_CONFIG_ADDRS_DC);
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, 64, 32, k_LOOP_WS_CONFIG_STRIDES_AB);
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, 0, 0, k_LOOP_WS_CONFIG_STRIDES_DC);
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, LOOP_REGION1_A_START, LOOP_REGION1_B_END,
                           k_LOOP_WS_CONFIG_SPAD_AB);
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC,
                           ((uint64_t)1 << 18) | ((uint64_t)1 << 16) |
                               ((uint64_t)NO_ACTIVATION << 8) | 1 /* ex_accumulate: bias */,
                           ((uint64_t)dst_row << 32) | LOOP_SPAD_ONLY, k_LOOP_WS);
}
#endif

// ------------------------------------------------------------------ NLB ops

// C[81 x 81] = theta[81 x 32] * phi[81 x 32]^T
static void matmul_theta_phi(const int8_t *A, const int8_t *B, int8_t *C, float scale) {
  gemmini_extended_config_ex(WEIGHT_STATIONARY, 0, 0, 1, 0, 1);
  gemmini_extended_config_st(81, NO_ACTIVATION, scale);
  gemmini_extended3_config_ld(32, 1.0f, false, 0);
  gemmini_extended3_config_ld(32, 1.0f, false, 1);
  gemmini_extended3_config_ld(81, 1.0f, false, 2);
  gemmini_loop_ws(6, 6, 2, 15, 15, 0, A, B, 0, C, 32, 32, 81, 81, 0, 1, 0, 0, 0,
                  NO_ACTIVATION, 1, 1, 0);
  gemmini_fence();
}

#ifdef OPT_QKV_SPAD_CHAIN
// theta * phi^T with theta and phi already in the scratchpad (banks 2 and 3).
static void matmul_theta_phi_spad(int8_t *C, float scale) {
  gemmini_extended_config_ex(WEIGHT_STATIONARY, 0, 0, 1, 0, 1);
  gemmini_extended_config_st(81, NO_ACTIVATION, scale);
  gemmini_extended3_config_ld(32, 1.0f, false, 0);
  gemmini_extended3_config_ld(32, 1.0f, false, 1);
  gemmini_extended3_config_ld(81, 1.0f, false, 2);
  LOOP_BOUNDS(6, 6, 2, 15, 15, 0);
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, 0, 0, k_LOOP_WS_CONFIG_ADDRS_AB);
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, 0, C, k_LOOP_WS_CONFIG_ADDRS_DC);
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, 32, 32, k_LOOP_WS_CONFIG_STRIDES_AB);
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, 81, 81, k_LOOP_WS_CONFIG_STRIDES_DC);
  // B tiles (j,k) sit at b_addr_end - 6*2*16 + (j*2 + k)*16.
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, THETA_SPAD_ROW, PHI_SPAD_ROW + 6 * 2 * DIM,
                           k_LOOP_WS_CONFIG_SPAD_AB);
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, (uint64_t)NO_ACTIVATION << 8,
                           LOOP_SKIP_LDA | LOOP_SKIP_LDB | (1 << 1) /* B_transpose */,
                           k_LOOP_WS);
  gemmini_fence();
}
#endif

static void softmax_81(const int8_t *A, int8_t *C, float in_scale, float out_scale) {
  gemmini_extended_config_ex(WEIGHT_STATIONARY, 0, 0, 1, 0, 0);
  gemmini_extended_config_st(81, 0, 1.0f / (127.0f * out_scale));
#ifndef OPT_SOFTMAX_LOAD
  gemmini_extended3_config_ld(81, 1.0f, false, 0);
  gemmini_extended3_config_ld(81, 1.0f, false, 1);
  gemmini_extended3_config_ld(81, 1.0f, false, 2);
#endif
  gemmini_config_norm((int)(0.693147 / in_scale), 0, 0, 1, 0, (int32_t)(1.353f / in_scale),
                      (int32_t)(0.344f / (0.3585f * in_scale * in_scale)));
  gemmini_config_norm((65536 / (int)(0.693147 / in_scale)), 1, 0, 1, 0,
                      (int32_t)(1.353f / in_scale),
                      (int32_t)(0.344f / (0.3585f * in_scale * in_scale)));
#ifdef OPT_SOFTMAX_LOAD
  // acc = 1.0 * A + 0.0 * A: the input goes straight into the accumulator.
  gemmini_extended4_config_ld(81, 1.0f, true, DIM, 0);
  gemmini_extended4_config_ld(81, 0.0f, true, DIM, 1);
#ifdef OPT_SOFTMAX_ONE_LOAD
  gemmini_loop_ws_skip(6, 6, 0, 15, 15, 0, A, A, 0, C, 81, 81, 0, 81, 0, 0, 0, 0, 0, SOFTMAX, 0, 0,
                       1, LOOP_SKIP_LDB);
#else
  gemmini_loop_ws(6, 6, 0, 15, 15, 0, A, A, 0, C, 81, 81, 0, 81, 0, 0, 0, 0, 0, SOFTMAX, 0, 0, 1);
#endif
#else
  static const int8_t identity[81 * 81] = {
      [0] = 1, [82] = 1, [164] = 1, [246] = 1, [328] = 1, [410] = 1, [492] = 1, [574] = 1,
      [656] = 1, [738] = 1, [820] = 1, [902] = 1, [984] = 1, [1066] = 1, [1148] = 1,
      [1230] = 1, [1312] = 1, [1394] = 1, [1476] = 1, [1558] = 1, [1640] = 1, [1722] = 1,
      [1804] = 1, [1886] = 1, [1968] = 1, [2050] = 1, [2132] = 1, [2214] = 1, [2296] = 1,
      [2378] = 1, [2460] = 1, [2542] = 1, [2624] = 1, [2706] = 1, [2788] = 1, [2870] = 1,
      [2952] = 1, [3034] = 1, [3116] = 1, [3198] = 1, [3280] = 1, [3362] = 1, [3444] = 1,
      [3526] = 1, [3608] = 1, [3690] = 1, [3772] = 1, [3854] = 1, [3936] = 1, [4018] = 1,
      [4100] = 1, [4182] = 1, [4264] = 1, [4346] = 1, [4428] = 1, [4510] = 1, [4592] = 1,
      [4674] = 1, [4756] = 1, [4838] = 1, [4920] = 1, [5002] = 1, [5084] = 1, [5166] = 1,
      [5248] = 1, [5330] = 1, [5412] = 1, [5494] = 1, [5576] = 1, [5658] = 1, [5740] = 1,
      [5822] = 1, [5904] = 1, [5986] = 1, [6068] = 1, [6150] = 1, [6232] = 1, [6314] = 1,
      [6396] = 1, [6478] = 1, [6560] = 1};
  gemmini_loop_ws(6, 6, 6, 15, 15, 15, A, identity, 0, C, 81, 81, 81, 81, 0, 0, 0, 0, 0,
                  SOFTMAX, 1, 1, 0);
#endif
  gemmini_fence();
}

#ifdef OPT_NLB_SPAD_CHAIN
// Softmax as in OPT_SOFTMAX_LOAD, but the output goes to scratchpad rows
// SOFTMAX_SPAD_ROW.. (spad_only store) instead of DRAM.
static void softmax_81_to_spad(const int8_t *A, float in_scale, float out_scale) {
  gemmini_extended_config_ex(WEIGHT_STATIONARY, 0, 0, 1, 0, 0);
  gemmini_extended_config_st(81, 0, 1.0f / (127.0f * out_scale));
  gemmini_config_norm((int)(0.693147 / in_scale), 0, 0, 1, 0, (int32_t)(1.353f / in_scale),
                      (int32_t)(0.344f / (0.3585f * in_scale * in_scale)));
  gemmini_config_norm((65536 / (int)(0.693147 / in_scale)), 1, 0, 1, 0,
                      (int32_t)(1.353f / in_scale),
                      (int32_t)(0.344f / (0.3585f * in_scale * in_scale)));
  gemmini_extended4_config_ld(81, 1.0f, true, DIM, 0);
  gemmini_extended4_config_ld(81, 0.0f, true, DIM, 1);
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)0 << 32) | ((uint64_t)15 << 16) | 15,
                           ((uint64_t)0 << 32) | ((uint64_t)6 << 16) | 6, k_LOOP_WS_CONFIG_BOUNDS);
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, A, A, k_LOOP_WS_CONFIG_ADDRS_AB);
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, 0, 0, k_LOOP_WS_CONFIG_ADDRS_DC);
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, 81, 81, k_LOOP_WS_CONFIG_STRIDES_AB);
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, 0, 81, k_LOOP_WS_CONFIG_STRIDES_DC);
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)SOFTMAX << 8),
                           ((uint64_t)SOFTMAX_SPAD_ROW << 32) | LOOP_SPAD_ONLY
#ifdef OPT_SOFTMAX_ONE_LOAD
                               | LOOP_SKIP_LDB
#endif
                               | (1 << 2) /* is_resadd */,
                           k_LOOP_WS);
  gemmini_fence();
}

// attention*g with A = the softmax output already in the scratchpad.
static void matmul_attention_g_spad(const int8_t *B, int8_t *C, float scale) {
  gemmini_extended_config_ex(WEIGHT_STATIONARY, 0, 0, 1, 0, 0);
  gemmini_extended_config_st(32, NO_ACTIVATION, scale);
  gemmini_extended3_config_ld(81, 1.0f, false, 0);
  gemmini_extended3_config_ld(32, 1.0f, false, 1);
  gemmini_extended3_config_ld(32, 1.0f, false, 2);
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)15 << 32) | ((uint64_t)0 << 16) | 15,
                           ((uint64_t)6 << 32) | ((uint64_t)2 << 16) | 6, k_LOOP_WS_CONFIG_BOUNDS);
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, 0, B, k_LOOP_WS_CONFIG_ADDRS_AB);
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, 0, C, k_LOOP_WS_CONFIG_ADDRS_DC);
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, 81, 32, k_LOOP_WS_CONFIG_STRIDES_AB);
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, 32, 32, k_LOOP_WS_CONFIG_STRIDES_DC);
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, SOFTMAX_SPAD_ROW, 0, k_LOOP_WS_CONFIG_SPAD_AB);
  // a_ex_spad_id = 0 (read A from a_addr_start), b_spad_id = 1, no activation.
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)1 << 16) | ((uint64_t)NO_ACTIVATION << 8),
                           LOOP_SKIP_LDA, k_LOOP_WS);
  gemmini_fence();
}
#endif

#ifdef OPT_QKV_SPAD_CHAIN
// attention * g with g already in the scratchpad (bank 2); A loads into
// region 1 (bank 0) as usual.
static void matmul_attention_g_gspad(const int8_t *A, int8_t *C, float scale) {
  gemmini_extended_config_ex(WEIGHT_STATIONARY, 0, 0, 1, 0, 0);
  gemmini_extended_config_st(32, NO_ACTIVATION, scale);
  gemmini_extended3_config_ld(81, 1.0f, false, 0);
  gemmini_extended3_config_ld(32, 1.0f, false, 1);
  gemmini_extended3_config_ld(32, 1.0f, false, 2);
  LOOP_BOUNDS(6, 2, 6, 15, 0, 15);
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, A, 0, k_LOOP_WS_CONFIG_ADDRS_AB);
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, 0, C, k_LOOP_WS_CONFIG_ADDRS_DC);
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, 81, 32, k_LOOP_WS_CONFIG_STRIDES_AB);
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, 32, 32, k_LOOP_WS_CONFIG_STRIDES_DC);
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, 0, G_SPAD_ROW + 6 * 2 * DIM, k_LOOP_WS_CONFIG_SPAD_AB);
  // a_spad_id = 1 (region 1, where ldA puts A), b_ex_spad_id = 0 (b_addr_end).
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)1 << 18) | ((uint64_t)NO_ACTIVATION << 8),
                           LOOP_SKIP_LDB, k_LOOP_WS);
  gemmini_fence();
}
#endif

// C[81 x 32] = attention[81 x 81] * g[81 x 32]
static void matmul_attention_g(const int8_t *A, const int8_t *B, int8_t *C, float scale) {
  gemmini_extended_config_ex(WEIGHT_STATIONARY, 0, 0, 1, 0, 0);
  gemmini_extended_config_st(32, NO_ACTIVATION, scale);
  gemmini_extended3_config_ld(81, 1.0f, false, 0);
  gemmini_extended3_config_ld(32, 1.0f, false, 1);
  gemmini_extended3_config_ld(32, 1.0f, false, 2);
  gemmini_loop_ws(6, 2, 6, 15, 0, 15, A, B, 0, C, 81, 32, 32, 32, 0, 0, 0, 0, 0, NO_ACTIVATION,
                  1, 1, 0);
  gemmini_fence();
}

// C = relu(round(a_scale * A + b_scale * B) * c_scale), 81 x 64
static void resadd_relu(float a_scale, float b_scale, float c_scale, const int8_t *A,
                        const int8_t *B, int8_t *C) {
  gemmini_extended_config_st(64, RELU, c_scale);
  gemmini_extended_config_ex(WEIGHT_STATIONARY, 0, 0, 1, 0, 0);
  gemmini_extended4_config_ld(64, a_scale, true, DIM, 0);
  gemmini_extended4_config_ld(64, b_scale, true, DIM, 1);
  gemmini_loop_ws(6, 4, 0, 15, 0, 0, A, B, 0, C, 64, 64, 0, 64, 0, 0, 0, 0, 0, RELU, 0, 0, 1);
  gemmini_fence();
}

// ------------------------------------------------------------------ FC layers

// out[1 x n_out] = in[1 x n_in] * W^T + bias, W is [n_out x n_in] (row-major).
static void fc_gemmini(const int8_t *in, int n_in, const int8_t *W, const int32_t *bias,
                       int8_t *out, int n_out, int act, float scale) {
  const int K = (n_in + DIM - 1) / DIM, pad_K = K * DIM - n_in;
  const int J = (n_out + DIM - 1) / DIM, pad_J = J * DIM - n_out;
  gemmini_extended_config_ex(WEIGHT_STATIONARY, 0, 0, 1, 0, 1);
  gemmini_extended_config_st(1, act, scale);
  gemmini_extended3_config_ld(n_in, 1.0f, false, 0);
  gemmini_extended3_config_ld(n_in, 1.0f, false, 1);
  gemmini_extended3_config_ld(1, 1.0f, false, 2);
  gemmini_loop_ws(1, J, K, 15, pad_J, pad_K, in, W, bias, out, n_in, n_in, n_out, n_out,
                  0, 1, 0, 0, 1, act, 1, 1, 0);
  gemmini_fence();
}

#ifdef OPT_TAIL_ON_CPU
// Same arithmetic as the Gemmini loop above: int32 accumulation with the bias,
// then scale_and_sat (round-half-even scaling, saturation, activation).
static inline void fc_cpu(const int8_t *in, int n_in, const int8_t *W, const int32_t *bias,
                          int8_t *out, int n_out, int act, float scale) {
  for (int o = 0; o < n_out; o++) {
    acc_t acc = bias[o];
    const int8_t *w = W + o * n_in;
    for (int i = 0; i < n_in; i++) acc += (acc_t)in[i] * w[i];
    out[o] = scale_and_sat(acc, act, scale, 0);
  }
}
#define FC_TAIL fc_cpu
#else
#define FC_TAIL fc_gemmini
#endif

// ------------------------------------------------------------ entry point

void braggnn_eval(void *ctxt, int_fast32_t n_patches, const float *fp32_inputs, const int8_t *conv1_weights, const int32_t *conv1_bias, const float *conv1_scale, const int8_t *nlb_theta_weights, const int32_t *nlb_theta_bias, const float *nlb_theta_scale, const int8_t *nlb_phi_weights, const int32_t *nlb_phi_bias, const float *nlb_phi_scale, const int8_t *nlb_g_weights, const int32_t *nlb_g_bias, const float *nlb_g_scale, const float *nlb_matmul_scale, const float *softmax_input_scale, const float *softmax_output_scale, const float *nlb_matmul_1_scale, const int8_t *nlb_out_weights, const int32_t *nlb_out_bias, const float *nlb_out_scale, const float *nlb_add_a_scale, const float *nlb_add_b_scale, const float *resadd_post_scale, const int8_t *conv2_weights, const int32_t *conv2_bias, const float *conv2_scale, const float *conv2_post_scale, const int8_t *conv3_weights, const int32_t *conv3_bias, const float *conv3_scale, const float *conv3_post_scale, const int8_t *fc1_weights, const int32_t *fc1_bias, const float *fc1_scale, const float *fc1_post_scale, const int8_t *fc2_weights, const int32_t *fc2_bias, const float *fc2_scale, const float *fc2_post_scale, const int8_t *fc3_weights, const int32_t *fc3_bias, const float *fc3_scale, const float *fc3_post_scale, const int8_t *fc4_weights, const int32_t *fc4_bias, const float *fc4_scale, const float *fc4_post_scale, const int8_t *output_weights, const int32_t *output_bias, const float *output_scale, int32_t *cycles, int8_t *outputs) {
  (void)ctxt;
  // Stage every weight and bias once, before the patch loop, into aligned
  // static buffers (as the Exo build does); fc1's is also re-laid out by
  // OPT_FC1_NO_FLATTEN.
  static int8_t w_conv1[9 * 64] ALIGNED, w_theta[64 * 32] ALIGNED, w_phi[64 * 32] ALIGNED,
      w_g[64 * 32] ALIGNED, w_nlb_out[32 * 64] ALIGNED, w_conv2[9 * 64 * 32] ALIGNED,
      w_conv3[9 * 32 * 8] ALIGNED, w_fc1[16 * 200] ALIGNED, w_fc2[8 * 16] ALIGNED,
      w_fc3[4 * 8] ALIGNED, w_fc4[2 * 4] ALIGNED, w_out[2 * 2] ALIGNED;
  static int32_t b_conv1[64] ALIGNED, b_theta[32] ALIGNED, b_phi[32] ALIGNED, b_g[32] ALIGNED,
      b_nlb_out[64] ALIGNED, b_conv2[32] ALIGNED, b_conv3[8] ALIGNED, b_fc1[16] ALIGNED,
      b_fc2[8] ALIGNED, b_fc3[4] ALIGNED, b_fc4[2] ALIGNED, b_out[2] ALIGNED;
  STAGE(w_conv1, conv1_weights); STAGE(b_conv1, conv1_bias);
  STAGE(w_theta, nlb_theta_weights); STAGE(b_theta, nlb_theta_bias);
  STAGE(w_phi, nlb_phi_weights); STAGE(b_phi, nlb_phi_bias);
  STAGE(w_g, nlb_g_weights); STAGE(b_g, nlb_g_bias);
  STAGE(w_nlb_out, nlb_out_weights); STAGE(b_nlb_out, nlb_out_bias);
  STAGE(w_conv2, conv2_weights); STAGE(b_conv2, conv2_bias);
  STAGE(w_conv3, conv3_weights); STAGE(b_conv3, conv3_bias);
  STAGE(b_fc1, fc1_bias);
  STAGE(w_fc2, fc2_weights); STAGE(b_fc2, fc2_bias);
  STAGE(w_fc3, fc3_weights); STAGE(b_fc3, fc3_bias);
  STAGE(w_fc4, fc4_weights); STAGE(b_fc4, fc4_bias);
  STAGE(w_out, output_weights); STAGE(b_out, output_bias);
#ifdef OPT_CONV1_IM2COL
  // conv1 weights [krow][kcol][1][64] = [9 x 64], padded to 16 rows of zeros.
  static int8_t w_conv1_16[16 * 64] ALIGNED;
  memcpy(w_conv1_16, conv1_weights, 9 * 64);
  static int8_t cols[81 * 16] ALIGNED;  // columns 9..15 stay zero
#endif
#ifdef OPT_FC1_NO_FLATTEN
  // flattened[ch*25 + pix] = conv3_out[pix*8 + ch], so permute each output's
  // 200 weights from (ch, pix) to (pix, ch) order.
  for (int o = 0; o < 16; o++)
    for (int ch = 0; ch < 8; ch++)
      for (int pix = 0; pix < 25; pix++)
        w_fc1[o * 200 + pix * 8 + ch] = fc1_weights[o * 200 + ch * 25 + pix];
#else
  STAGE(w_fc1, fc1_weights);
#endif

  const float s_conv2 = conv2_scale[0] * conv2_post_scale[0];
  const float s_conv3 = conv3_scale[0] * conv3_post_scale[0];
  const float s_fc1 = fc1_scale[0] * fc1_post_scale[0];
  const float s_fc2 = fc2_scale[0] * fc2_post_scale[0];
  const float s_fc3 = fc3_scale[0] * fc3_post_scale[0];
  const float s_fc4 = fc4_scale[0] * fc4_post_scale[0];

  static int8_t inp[11 * 11] ALIGNED;
  static int8_t conv1_out[81 * 64] ALIGNED;
#ifndef OPT_QKV_SPAD_CHAIN  // with it, theta / phi / g live only in the scratchpad
  static int8_t theta[81 * 32] ALIGNED, phi[81 * 32] ALIGNED, g[81 * 32] ALIGNED;
#endif
  static int8_t attention_raw[81 * 81] ALIGNED, attention[81 * 81] ALIGNED;
  static int8_t attended[81 * 32] ALIGNED, nlb_conv_out[81 * 64] ALIGNED, nlb_out[81 * 64] ALIGNED;
  static int8_t conv2_out[49 * 32] ALIGNED, conv3_out[25 * 8] ALIGNED;
#ifndef OPT_FC1_NO_FLATTEN
  static int8_t flattened[200] ALIGNED;
#endif
  static int8_t fc1_out[16] ALIGNED, fc2_out[8] ALIGNED, fc3_out[4] ALIGNED, fc4_out[2] ALIGNED,
      pred[2] ALIGNED;

  for (int_fast32_t p = 0; p < n_patches; p++) {
    const float *patch = fp32_inputs + p * 121;
    int32_t begin = rdcycle32();
    MARK(0, "start");

    for (int i = 0; i < 121; i++) inp[i] = (int8_t)(patch[i] * 127.0f);
#ifdef OPT_CONV1_IM2COL
    for (int r = 0; r < 9; r++)
      for (int c = 0; c < 9; c++) {
        int8_t *row = cols + (r * 9 + c) * 16;
        const int8_t *src = inp + r * 11 + c;
        row[0] = src[0];  row[1] = src[1];  row[2] = src[2];
        row[3] = src[11]; row[4] = src[12]; row[5] = src[13];
        row[6] = src[22]; row[7] = src[23]; row[8] = src[24];
      }
    conv1_matmul(cols, w_conv1_16, b_conv1, conv1_out, conv1_scale[0]);
#else
    conv1(inp, w_conv1, b_conv1, conv1_out, conv1_scale[0]);
#endif
    MARK(1, "conv1");

    // NLB: the three 1x1 convs read the same input and are independent, so
    // one fence covers all three.
#if defined(OPT_QKV_SPAD_CHAIN)
    qkv_matmul_to_spad(conv1_out, w_theta, b_theta, THETA_SPAD_ROW, nlb_theta_scale[0]);
    qkv_matmul_to_spad(NULL, w_phi, b_phi, PHI_SPAD_ROW, nlb_phi_scale[0]);
    qkv_matmul_to_spad(NULL, w_g, b_g, G_SPAD_ROW, nlb_g_scale[0]);
#elif defined(OPT_QKV_LOOP_WS)
    qkv_matmul(conv1_out, w_theta, b_theta, theta, nlb_theta_scale[0]);
#ifdef OPT_QKV_REUSE_A
    qkv_matmul(NULL, w_phi, b_phi, phi, nlb_phi_scale[0]);
    qkv_matmul(NULL, w_g, b_g, g, nlb_g_scale[0]);
#else
    qkv_matmul(conv1_out, w_phi, b_phi, phi, nlb_phi_scale[0]);
    qkv_matmul(conv1_out, w_g, b_g, g, nlb_g_scale[0]);
#endif
#else
    conv1x1(conv1_out, 64, w_theta, b_theta, theta, 32, nlb_theta_scale[0]);
    conv1x1(conv1_out, 64, w_phi, b_phi, phi, 32, nlb_phi_scale[0]);
    conv1x1(conv1_out, 64, w_g, b_g, g, 32, nlb_g_scale[0]);
#endif
    gemmini_fence();
    MARK(2, "nlb_qkv");
#ifdef OPT_QKV_SPAD_CHAIN
    matmul_theta_phi_spad(attention_raw, nlb_matmul_scale[0]);
#else
    matmul_theta_phi(theta, phi, attention_raw, nlb_matmul_scale[0]);
#endif
    MARK(3, "theta_phi");
#ifdef OPT_NLB_SPAD_CHAIN
    softmax_81_to_spad(attention_raw, softmax_input_scale[0], softmax_output_scale[0]);
    MARK(4, "softmax");
    matmul_attention_g_spad(g, attended, nlb_matmul_1_scale[0]);
#else
    softmax_81(attention_raw, attention, softmax_input_scale[0], softmax_output_scale[0]);
    MARK(4, "softmax");
#ifdef OPT_QKV_SPAD_CHAIN
    matmul_attention_g_gspad(attention, attended, nlb_matmul_1_scale[0]);
#else
    matmul_attention_g(attention, g, attended, nlb_matmul_1_scale[0]);
#endif
#endif
    MARK(5, "attention_g");
    conv1x1(attended, 32, w_nlb_out, b_nlb_out, nlb_conv_out, 64, nlb_out_scale[0]);
    gemmini_fence();
    MARK(6, "nlb_out_conv");
    resadd_relu(nlb_add_b_scale[0], nlb_add_a_scale[0], resadd_post_scale[0], conv1_out,
                nlb_conv_out, nlb_out);
    MARK(7, "resadd");

    conv2(nlb_out, w_conv2, b_conv2, conv2_out, s_conv2);
    MARK(8, "conv2");
    conv3(conv2_out, w_conv3, b_conv3, conv3_out, s_conv3);
    MARK(9, "conv3");

#ifdef OPT_FC1_NO_FLATTEN
    fc_gemmini(conv3_out, 200, w_fc1, b_fc1, fc1_out, 16, RELU, s_fc1);
#else
    for (int ch = 0; ch < 8; ch++)
      for (int pix = 0; pix < 25; pix++) flattened[ch * 25 + pix] = conv3_out[pix * 8 + ch];
    fc_gemmini(flattened, 200, w_fc1, b_fc1, fc1_out, 16, RELU, s_fc1);
#endif
    MARK(10, "fc1");
    FC_TAIL(fc1_out, 16, w_fc2, b_fc2, fc2_out, 8, RELU, s_fc2);
    FC_TAIL(fc2_out, 8, w_fc3, b_fc3, fc3_out, 4, RELU, s_fc3);
    FC_TAIL(fc3_out, 4, w_fc4, b_fc4, fc4_out, 2, RELU, s_fc4);
    FC_TAIL(fc4_out, 2, w_out, b_out, pred, 2, NO_ACTIVATION, output_scale[0]);
    MARK(11, "fc2..output");

    int32_t end = rdcycle32();
    cycles[p] = end - begin;
    outputs[p * 2] = pred[0];
    outputs[p * 2 + 1] = pred[1];
  }
}
