#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "braggnn_data.h"
#include "braggnn_schedule.h"
#include "xprintf.h"

#define DEQUANT_SCALE (11.0f / 127.0f)
#define MAX_AVG_ERROR_PX 0.5f

// The seed's int8 predictions (exo/braggnn_schedule.py on FireSim,
// results/20260922_171212/iter_00). Every schedule computes the same
// arithmetic, so its predictions must equal these exactly; a schedule whose
// instr C is wrong (or that drops a needed fence) shows up here even when its
// error stays under MAX_AVG_ERROR_PX.
static const int8_t golden_preds[NUM_TEST_PATCHES][OUTPUT_UNITS] = {
    {58, 70}, {68, 60}, {72, 68}, {68, 70}, {72, 66},
    {54, 59}, {68, 61}, {58, 62}, {63, 61}, {67, 63},
};

// Per-layer profiling (gemmini.py's layer_mark, placed by
// braggnn_reference.py). The timed pass runs with chia_profile_layers = 0;
// a second pass sets it, and each mark then fences Gemmini and charges the
// cycles since the previous mark to its layer. Ids match braggnn_reference.py
// (1 = start of the patch; an Exo size is never 0).
#define CHIA_NUM_MARKS 14
static const char *const chia_mark_names[CHIA_NUM_MARKS] = {
    "", "start", "input_quantize", "conv1", "nlb_qkv", "nlb_theta_phi",
    "nlb_softmax", "nlb_attention_g", "nlb_out_conv", "nlb_resadd", "conv2",
    "conv3", "flatten_fc1", "fc2_to_output",
};
volatile int chia_profile_layers = 0;
static uint64_t chia_mark_cycles[CHIA_NUM_MARKS];
static uint32_t chia_mark_hits[CHIA_NUM_MARKS];
static uint64_t chia_mark_prev;

void chia_layer_mark(int id) {
  gemmini_fence();
  uint64_t c;
  asm volatile("rdcycle %0" : "=r"(c));
  if (id > 1 && id < CHIA_NUM_MARKS) {
    chia_mark_cycles[id] += c - chia_mark_prev;
    chia_mark_hits[id]++;
  }
  chia_mark_prev = c;
}

static float patches[NUM_TEST_PATCHES][INPUT_DIM][INPUT_DIM];
static int8_t preds[NUM_TEST_PATCHES][OUTPUT_UNITS][1][1];
static int32_t patch_cycles[NUM_TEST_PATCHES];
static int8_t profiled_preds[NUM_TEST_PATCHES][OUTPUT_UNITS][1][1];
static int32_t profiled_cycles[NUM_TEST_PATCHES];

static void run_eval(int32_t *cycles, int8_t *outputs) {
  braggnn_eval(
      NULL, NUM_TEST_PATCHES, (const float *)patches,
      (const int8_t *)conv1_weights_flat, conv1_bias,
      &(float){CNN_LAYERS_0_CONV_QUANT_ACC_SCALE},
      (const int8_t *)nlb_theta_weights_flat, nlb_theta_bias,
      &(float){NLB_THETA_LAYER_CONV_QUANT_ACC_SCALE},
      (const int8_t *)nlb_phi_weights_flat, nlb_phi_bias,
      &(float){NLB_PHI_LAYER_CONV_QUANT_ACC_SCALE},
      (const int8_t *)nlb_g_weights_flat, nlb_g_bias,
      &(float){NLB_G_LAYER_CONV_QUANT_ACC_SCALE},
      &(float){NLB_MATMUL_QUANT_ACC_SCALE}, &(float){SOFTMAX_INPUT_SCALE},
      &(float){SOFTMAX_OUTPUT_SCALE}, &(float){NLB_MATMUL_1_QUANT_ACC_SCALE},
      (const int8_t *)nlb_out_weights_flat, nlb_out_bias,
      &(float){NLB_OUT_CNN_CONV_QUANT_ACC_SCALE}, &(float){NLB_ADD_A_SCALE},
      &(float){NLB_ADD_B_SCALE},
      &(float){CNN_LAYERS_1_LEAKYRELU_QUANT_ACC_SCALE},
      (const int8_t *)conv2_weights_flat, conv2_bias,
      &(float){CNN_LAYERS_2_CONV_QUANT_ACC_SCALE},
      &(float){CNN_LAYERS_3_LEAKYRELU_QUANT_ACC_SCALE},
      (const int8_t *)conv3_weights_flat, conv3_bias,
      &(float){CNN_LAYERS_4_CONV_QUANT_ACC_SCALE},
      &(float){CNN_LAYERS_5_LEAKYRELU_QUANT_ACC_SCALE},
      (const int8_t *)fc1_weights, fc1_bias,
      &(float){DENSE_LAYERS_0_GEMM_ACC_SCALE},
      &(float){DENSE_LAYERS_1_LEAKYRELU_QUANT_ACC_SCALE},
      (const int8_t *)fc2_weights, fc2_bias,
      &(float){DENSE_LAYERS_2_GEMM_ACC_SCALE},
      &(float){DENSE_LAYERS_3_LEAKYRELU_QUANT_ACC_SCALE},
      (const int8_t *)fc3_weights, fc3_bias,
      &(float){DENSE_LAYERS_4_GEMM_ACC_SCALE},
      &(float){DENSE_LAYERS_5_LEAKYRELU_QUANT_ACC_SCALE},
      (const int8_t *)fc4_weights, fc4_bias,
      &(float){DENSE_LAYERS_6_GEMM_ACC_SCALE},
      &(float){DENSE_LAYERS_7_LEAKYRELU_QUANT_ACC_SCALE},
      (const int8_t *)output_weights, output_bias,
      &(float){DENSE_LAYERS_8_GEMM_ACC_SCALE}, cycles, outputs);
}

static int count_mismatches(int8_t p[NUM_TEST_PATCHES][OUTPUT_UNITS][1][1]) {
  int n = 0;
  for (size_t i = 0; i < NUM_TEST_PATCHES; i++)
    for (size_t k = 0; k < OUTPUT_UNITS; k++)
      if (p[i][k][0][0] != golden_preds[i][k]) {
        n++;
        break;
      }
  return n;
}

int main(void) {
  xdev_out(putchar);

  for (size_t i = 0; i < NUM_TEST_PATCHES; i++)
    for (size_t r = 0; r < INPUT_DIM; r++)
      for (size_t c = 0; c < INPUT_DIM; c++)
        patches[i][r][c] = test_inputs[i][r * INPUT_DIM + c];

  run_eval(patch_cycles, (int8_t *)preds);
  chia_profile_layers = 1;
  run_eval(profiled_cycles, (int8_t *)profiled_preds);
  chia_profile_layers = 0;

  uint64_t total_cycles = 0;
  float total_x_error = 0.0f;
  float total_y_error = 0.0f;

  for (size_t i = 0; i < NUM_TEST_PATCHES; i++) {
    uint64_t cycles = (uint64_t)(uint32_t)patch_cycles[i];
    total_cycles += cycles;

    float err_x = preds[i][0][0][0] * DEQUANT_SCALE - test_labels[i][0];
    float err_y = preds[i][1][0][0] * DEQUANT_SCALE - test_labels[i][1];
    total_x_error += fabsf(err_x);
    total_y_error += fabsf(err_y);

    xprintf("patch %u: cycles=%lu err=(%.3f, %.3f) px\n", (unsigned)i,
            (unsigned long)cycles, err_x, err_y);
  }

  float avg_x_error = total_x_error / NUM_TEST_PATCHES;
  float avg_y_error = total_y_error / NUM_TEST_PATCHES;
  int mismatches = count_mismatches(preds);
  int profiled_mismatches = count_mismatches(profiled_preds);
  int failed = avg_x_error > MAX_AVG_ERROR_PX || avg_y_error > MAX_AVG_ERROR_PX ||
               mismatches > 0;

  // Per-layer cycles of the profiled pass, averaged over the patches. A mark
  // the schedule dropped or moved is simply missing or shifted here.
  for (int i = 2; i < CHIA_NUM_MARKS; i++)
    if (chia_mark_hits[i])
      xprintf("Layer %s: %lu\n", chia_mark_names[i],
              (unsigned long)(chia_mark_cycles[i] / NUM_TEST_PATCHES));
  xprintf("Prediction mismatches: %d of %d patches (profiled pass: %d)%s\n",
          mismatches, NUM_TEST_PATCHES, profiled_mismatches,
          mismatches ? " FAIL" : "");

  xprintf("Avg cycles: %lu\n",
          (unsigned long)(total_cycles / NUM_TEST_PATCHES));
  xprintf("Avg error: (%.3f, %.3f) px (tolerance %.3f px)%s\n", avg_x_error,
          avg_y_error, (float)MAX_AVG_ERROR_PX, failed ? " FAIL" : "");

  return failed;
}
