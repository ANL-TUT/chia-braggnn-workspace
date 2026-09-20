#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "braggnn_data.h"
#include "braggnn_inference.h"
#include "xprintf.h"

#define DEQUANT_SCALE (11.0f / 127.0f)
#define MAX_AVG_ERROR_PX 0.5f

static inline uint64_t rdcycle(void) {
  uint64_t cycles;
  asm volatile("rdcycle %0" : "=r"(cycles));
  return cycles;
}

int main(void) {
  xdev_out(putchar);

  elem_t pred[OUTPUT_UNITS];
  uint64_t total_cycles = 0;
  float total_x_error = 0.0f;
  float total_y_error = 0.0f;

  for (size_t i = 0; i < NUM_TEST_PATCHES; i++) {
    uint64_t start = rdcycle();
    braggnn_inference(
        NULL, test_inputs[i], (const elem_t *)conv1_weights_flat, conv1_bias,
        &(float){CNN_LAYERS_0_CONV_QUANT_ACC_SCALE},
        (const elem_t *)nlb_theta_weights_flat, nlb_theta_bias,
        &(float){NLB_THETA_LAYER_CONV_QUANT_ACC_SCALE},
        (const elem_t *)nlb_phi_weights_flat, nlb_phi_bias,
        &(float){NLB_PHI_LAYER_CONV_QUANT_ACC_SCALE},
        (const elem_t *)nlb_g_weights_flat, nlb_g_bias,
        &(float){NLB_G_LAYER_CONV_QUANT_ACC_SCALE},
        &(float){NLB_MATMUL_QUANT_ACC_SCALE}, &(float){SOFTMAX_INPUT_SCALE},
        &(float){SOFTMAX_OUTPUT_SCALE}, &(float){NLB_MATMUL_1_QUANT_ACC_SCALE},
        (const elem_t *)nlb_out_weights_flat, nlb_out_bias,
        &(float){NLB_OUT_CNN_CONV_QUANT_ACC_SCALE}, &(float){NLB_ADD_A_SCALE},
        &(float){NLB_ADD_B_SCALE},
        &(float){CNN_LAYERS_1_LEAKYRELU_QUANT_ACC_SCALE},
        (const elem_t *)conv2_weights_flat, conv2_bias,
        &(float){CNN_LAYERS_2_CONV_QUANT_ACC_SCALE},
        &(float){CNN_LAYERS_3_LEAKYRELU_QUANT_ACC_SCALE},
        (const elem_t *)conv3_weights_flat, conv3_bias,
        &(float){CNN_LAYERS_4_CONV_QUANT_ACC_SCALE},
        &(float){CNN_LAYERS_5_LEAKYRELU_QUANT_ACC_SCALE},
        (const elem_t *)fc1_weights, fc1_bias,
        &(float){DENSE_LAYERS_0_GEMM_ACC_SCALE},
        &(float){DENSE_LAYERS_1_LEAKYRELU_QUANT_ACC_SCALE},
        (const elem_t *)fc2_weights, fc2_bias,
        &(float){DENSE_LAYERS_2_GEMM_ACC_SCALE},
        &(float){DENSE_LAYERS_3_LEAKYRELU_QUANT_ACC_SCALE},
        (const elem_t *)fc3_weights, fc3_bias,
        &(float){DENSE_LAYERS_4_GEMM_ACC_SCALE},
        &(float){DENSE_LAYERS_5_LEAKYRELU_QUANT_ACC_SCALE},
        (const elem_t *)fc4_weights, fc4_bias,
        &(float){DENSE_LAYERS_6_GEMM_ACC_SCALE},
        &(float){DENSE_LAYERS_7_LEAKYRELU_QUANT_ACC_SCALE},
        (const elem_t *)output_weights, output_bias,
        &(float){DENSE_LAYERS_8_GEMM_ACC_SCALE}, pred);
    uint64_t cycles = rdcycle() - start;
    total_cycles += cycles;

    float err_x = pred[0] * DEQUANT_SCALE - test_labels[i][0];
    float err_y = pred[1] * DEQUANT_SCALE - test_labels[i][1];
    total_x_error += fabsf(err_x);
    total_y_error += fabsf(err_y);

    xprintf("patch %u: cycles=%lu err=(%.3f, %.3f) px\n", (unsigned)i,
            (unsigned long)cycles, err_x, err_y);
  }

  float avg_x_error = total_x_error / NUM_TEST_PATCHES;
  float avg_y_error = total_y_error / NUM_TEST_PATCHES;
  int failed = avg_x_error > MAX_AVG_ERROR_PX || avg_y_error > MAX_AVG_ERROR_PX;

  xprintf("Avg cycles: %lu\n",
          (unsigned long)(total_cycles / NUM_TEST_PATCHES));
  xprintf("Avg error: (%.3f, %.3f) px (tolerance %.3f px)%s\n", avg_x_error,
          avg_y_error, (float)MAX_AVG_ERROR_PX, failed ? " FAIL" : "");

  return failed;
}
