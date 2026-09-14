#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "braggnn_data.h"
#include "include/gemmini.h"
#include "braggnn_inference.h"

// Convolution operation using Gemmini's tiled_conv_auto
// weights_flat: pre-computed flattened weight matrix [patch_size][out_channels]
void gemmini_conv2d(int batch, int in_rows, int in_cols, int in_channels,
                    int out_channels, int kernel_dim, int stride, int padding,
                    elem_t *input, elem_t *weights_flat, acc_t *bias,
                    elem_t *output, bool relu, acc_scale_t acc_scale) {

  int out_rows = (in_rows + 2 * padding - kernel_dim) / stride + 1;
  int out_cols = (in_cols + 2 * padding - kernel_dim) / stride + 1;

  tiled_conv_auto(batch, in_rows, in_cols, in_channels, out_channels, out_rows,
                  out_cols, stride, 1, 1, padding, kernel_dim, false, false,
                  false, false, false, (elem_t *)input, (elem_t *)weights_flat,
                  (acc_t *)bias, (elem_t *)output, relu ? RELU : NO_ACTIVATION,
                  acc_scale, 0, 0, 0, WS);

  // tiled_conv_auto does not have a fence (unlike tiled_matmul_outer).
  gemmini_fence();
}

// Fully connected layer using Gemmini
void gemmini_fc(int batch, int in_features, int out_features, elem_t *input,
                elem_t *weights, acc_t *bias, elem_t *output, bool relu,
                acc_scale_t acc_scale) {
  tiled_matmul_auto(batch, out_features, in_features, (elem_t *)input,
                    (elem_t *)weights, (acc_t *)bias, (elem_t *)output,
                    in_features, in_features, out_features, out_features,
                    MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
                    MVIN_SCALE_IDENTITY, relu ? RELU : NO_ACTIVATION, acc_scale,
                    0, false, false, true, false, false, 0, WS);
}

// Non-Local Block implementation
void gemmini_nlb(elem_t *input, elem_t *output) {
  // Theta conv: output NHWC [1][9][9][32], memory layout = [81][32]
  static elem_t nlb_theta_out[CONV1_CHANNELS][CONV1_DIM][CONV1_DIM]
                             [CONV2_FILTERS];
  gemmini_conv2d(BATCH, CONV1_DIM, CONV1_DIM, CONV1_FILTERS, CONV2_FILTERS,
                 NLB_THETA_KERNEL, 1, 0, (elem_t *)input,
                 (elem_t *)nlb_theta_weights_flat, (acc_t *)nlb_theta_bias,
                 (elem_t *)nlb_theta_out, false,
                 NLB_THETA_LAYER_CONV_QUANT_ACC_SCALE);

  // Phi conv
  static elem_t nlb_phi_out[CONV1_CHANNELS][CONV1_DIM][CONV1_DIM]
                           [CONV2_FILTERS];
  gemmini_conv2d(BATCH, CONV1_DIM, CONV1_DIM, CONV1_FILTERS, CONV2_FILTERS,
                 NLB_PHI_KERNEL, 1, 0, (elem_t *)input,
                 (elem_t *)nlb_phi_weights_flat, (acc_t *)nlb_phi_bias,
                 (elem_t *)nlb_phi_out, false,
                 NLB_PHI_LAYER_CONV_QUANT_ACC_SCALE);

  // G conv
  static elem_t nlb_g_out[CONV2_CHANNELS][CONV1_DIM][CONV1_DIM][CONV2_FILTERS];
  gemmini_conv2d(BATCH, CONV1_DIM, CONV1_DIM, CONV1_FILTERS, CONV2_FILTERS,
                 NLB_G_KERNEL, 1, 0, (elem_t *)input,
                 (elem_t *)nlb_g_weights_flat, (acc_t *)nlb_g_bias,
                 (elem_t *)nlb_g_out, false, NLB_G_LAYER_CONV_QUANT_ACC_SCALE);

  // Step 1: theta @ phi^T → int8 (matches PTQ QuantizeLinear)
  static elem_t attention_raw[CONV1_2D][CONV1_2D];
  tiled_matmul_auto(
      CONV1_2D, CONV1_2D, CONV2_FILTERS, // dim_I=81, dim_J=81, dim_K=32
      (elem_t *)nlb_theta_out,           // A [81x32], stride=32
      (elem_t *)nlb_phi_out,             // B [81x32], transposed to [32x81]
      NULL, (elem_t *)attention_raw, CONV2_FILTERS, CONV2_FILTERS, CONV1_2D,
      CONV1_2D, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
      NO_ACTIVATION, NLB_MATMUL_QUANT_ACC_SCALE, 0, false, false, true, false,
      false, 0, WS);

  // Step 2: Identity matmul + Gemmini I-BERT SOFTMAX
  // attention_raw @ identity_81 with SOFTMAX activation
  // Hardware computes: output = iexp(q-max) * (127 / sum_exp) * acc_scale
  // We want: output = softmax(q) / SOFTMAX_OUTPUT_SCALE
  // So: acc_scale = 1 / (127 * SOFTMAX_OUTPUT_SCALE)
  // bert_scale = SOFTMAX_INPUT_SCALE (dequant scale for I-BERT constants)
  static elem_t attention_out[CONV1_2D][CONV1_2D];
  tiled_matmul_auto(CONV1_2D, CONV1_2D, CONV1_2D,
                    (elem_t *)attention_raw,       // A [81][81]
                    (elem_t *)identity_81,         // B [81][81]
                    NULL, (elem_t *)attention_out, // C [81][81]
                    CONV1_2D, CONV1_2D, CONV1_2D, CONV1_2D, MVIN_SCALE_IDENTITY,
                    MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, SOFTMAX,
                    1.0f / (127.0f * SOFTMAX_OUTPUT_SCALE), SOFTMAX_INPUT_SCALE,
                    false, false, false, false, false, 0, WS);

  // Step 3: attended = softmax(attention) @ g = [81][81] @ [81][32] = [81][32]
  static elem_t attended_output[CONV1_2D][CONV2_FILTERS];
  tiled_matmul_auto(CONV1_2D, CONV2_FILTERS, CONV1_2D,
                    (elem_t *)attention_out, // A [81][81]
                    (elem_t *)nlb_g_out,     // B [81][32]
                    NULL, (elem_t *)attended_output, CONV1_2D, CONV2_FILTERS,
                    CONV2_FILTERS, CONV2_FILTERS, MVIN_SCALE_IDENTITY,
                    MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, NO_ACTIVATION,
                    NLB_MATMUL_1_QUANT_ACC_SCALE, 0.0, false, false, false,
                    false, false, 0, WS);

  // Out conv: attended_output [81][32] is NHWC [9][9][32] in memory
  static elem_t nlb_output[CONV1_DIM][CONV1_DIM][64];
  gemmini_conv2d(BATCH, CONV1_DIM, CONV1_DIM, CONV2_FILTERS, CONV1_FILTERS,
                 NLB_OUT_KERNEL, 1, 0, (elem_t *)attended_output,
                 (elem_t *)nlb_out_weights_flat, (acc_t *)nlb_out_bias,
                 (elem_t *)nlb_output, false, NLB_OUT_CNN_CONV_QUANT_ACC_SCALE);

  // resadd with RELU + fused requant scale
  tiled_resadd_auto(CONV1_2D, CONV1_FILTERS,
                    NLB_ADD_B_SCALE, // A_scale for input (skip connection)
                    NLB_ADD_A_SCALE, // B_scale for nlb_output (out_cnn)
                    CNN_LAYERS_1_LEAKYRELU_QUANT_ACC_SCALE, // C_scale
                    (elem_t *)input, (elem_t *)nlb_output, (elem_t *)output,
                    true, WS); // relu=true
}

void gemmini_inference(float *fp32_input, elem_t *output) {

  // QuantizeLinear: y_scale = 0.007874015718698502
  elem_t input[BATCH][INPUT_DIM][INPUT_DIM][INPUT_CHANNELS];
  const acc_t y_scale = 127.0f; // 1/0.007874 ≈ 127
  for (int i = 0; i < BATCH * INPUT_DIM * INPUT_DIM * INPUT_CHANNELS; i++) {
    ((elem_t *)input)[i] = (elem_t)(((float *)fp32_input)[i] * y_scale);
  }

  static elem_t conv1_out[CONV1_CHANNELS][CONV1_DIM][CONV1_DIM]
                         [CONV1_FILTERS]; // 11x11 -> 9x9x64 (3x3, no pad)
  static elem_t nlb_out[CONV1_CHANNELS][CONV1_DIM][CONV1_DIM]
                       [CONV1_FILTERS]; // NLB output: 9x9x64
  static elem_t conv2_out[CONV2_CHANNELS][CONV2_DIM][CONV2_DIM]
                         [CONV2_FILTERS]; // 9x9 -> 7x7x32 (3x3, no pad)
  static elem_t conv3_out[CONV3_KERNEL][CONV3_DIM][CONV3_DIM]
                         [CONV3_FILTERS]; // 7x7 -> 5x5x8 (3x3, no pad)

  static elem_t fc1_out[FC1_UNITS];
  static elem_t fc2_out[FC2_UNITS];
  static elem_t fc3_out[FC3_UNITS];
  static elem_t fc4_out[FC4_UNITS];

  gemmini_conv2d(BATCH, INPUT_DIM, INPUT_DIM, INPUT_CHANNELS, CONV1_FILTERS,
                 CONV1_KERNEL, 1, 0, (elem_t *)input,
                 (elem_t *)conv1_weights_flat, (acc_t *)conv1_bias,
                 (elem_t *)conv1_out, false, CNN_LAYERS_0_CONV_QUANT_ACC_SCALE);
  // printf("[DEBUG] Conv1 done\n");
  // printf("conv1_out ch0-7@(0,0): %d, %d, %d, %d, %d, %d, %d, %d\n",
  //        conv1_out[0][0][0][0], conv1_out[0][0][0][1], conv1_out[0][0][0][2],
  //        conv1_out[0][0][0][3], conv1_out[0][0][0][4], conv1_out[0][0][0][5],
  //        conv1_out[0][0][0][6], conv1_out[0][0][0][7]);
  // printf("conv1_out ch0-7@(4,4): %d, %d, %d, %d, %d, %d, %d, %d\n",
  //        conv1_out[0][4][4][0], conv1_out[0][4][4][1], conv1_out[0][4][4][2],
  //        conv1_out[0][4][4][3], conv1_out[0][4][4][4], conv1_out[0][4][4][5],
  //        conv1_out[0][4][4][6], conv1_out[0][4][4][7]);

  gemmini_nlb((elem_t *)conv1_out, (elem_t *)nlb_out);

  gemmini_conv2d(BATCH, CONV1_DIM, CONV1_DIM, CONV1_FILTERS, CONV2_FILTERS,
                 CONV2_KERNEL, 1, 0, (elem_t *)nlb_out,
                 (elem_t *)conv2_weights_flat, (acc_t *)conv2_bias,
                 (elem_t *)conv2_out, true,
                 CNN_LAYERS_2_CONV_QUANT_ACC_SCALE *
                     CNN_LAYERS_3_LEAKYRELU_QUANT_ACC_SCALE);

  gemmini_conv2d(BATCH, CONV2_DIM, CONV2_DIM, CONV2_FILTERS, CONV3_FILTERS,
                 CONV3_KERNEL, 1, 0, (elem_t *)conv2_out,
                 (elem_t *)conv3_weights_flat, (acc_t *)conv3_bias,
                 (elem_t *)conv3_out, true,
                 CNN_LAYERS_4_CONV_QUANT_ACC_SCALE *
                     CNN_LAYERS_5_LEAKYRELU_QUANT_ACC_SCALE);

  // Flatten: 5x5x8 -> 200 (NCHW order to match ONNX FC weight layout)
  elem_t flattened[CONV3_FLATTENED];
  int idx = 0;
  for (int ch = 0; ch < CONV3_FILTERS; ch++) {
    for (int r = 0; r < CONV3_DIM; r++) {
      for (int c = 0; c < CONV3_DIM; c++) {
        flattened[idx++] = conv3_out[0][r][c][ch];
      }
    }
  }

  gemmini_fc(CONV3_CHANNELS, CONV3_FLATTENED, FC1_UNITS, (elem_t *)flattened,
             (elem_t *)fc1_weights, (acc_t *)fc1_bias, (elem_t *)fc1_out, true,
             DENSE_LAYERS_0_GEMM_ACC_SCALE);
  // printf("fc1_out: %d, %d, %d\n", fc1_out[0], fc1_out[1], fc1_out[2]);
  gemmini_fc(1, FC1_UNITS, FC2_UNITS, (elem_t *)fc1_out, (elem_t *)fc2_weights,
             (acc_t *)fc2_bias, (elem_t *)fc2_out, true,
             DENSE_LAYERS_2_GEMM_ACC_SCALE);

  // printf("fc2_out: %d, %d, %d\n", fc2_out[0], fc2_out[1], fc2_out[2]);
  gemmini_fc(1, FC2_UNITS, FC3_UNITS, (elem_t *)fc2_out, (elem_t *)fc3_weights,
             (acc_t *)fc3_bias, (elem_t *)fc3_out, true,
             DENSE_LAYERS_4_GEMM_ACC_SCALE);

  // printf("fc3_out: %d, %d, %d\n", fc3_out[0], fc3_out[1], fc3_out[2]);
  gemmini_fc(1, FC3_UNITS, FC4_UNITS, (elem_t *)fc3_out, (elem_t *)fc4_weights,
             (acc_t *)fc4_bias, (elem_t *)fc4_out, true,
             DENSE_LAYERS_6_GEMM_ACC_SCALE);

  // printf("fc4_out: %d, %d, %d\n", fc4_out[0], fc4_out[1], fc4_out[2]);
  gemmini_fc(1, FC4_UNITS, OUTPUT_UNITS, (elem_t *)fc4_out,
             (elem_t *)output_weights, (acc_t *)output_bias, (elem_t *)output,
             false, DENSE_LAYERS_8_GEMM_ACC_SCALE);
}

// Adapt the original fixed-model implementation to the shared Exo ABI.
// Weights and scales come from braggnn_data.h; their ABI arguments are unused.
void braggnn_inference(
    void *ctxt, const float *fp32_input, const int8_t *conv1_weights,
    const int32_t *conv1_bias, const float *conv1_scale,
    const int8_t *nlb_theta_weights, const int32_t *nlb_theta_bias,
    const float *nlb_theta_scale, const int8_t *nlb_phi_weights,
    const int32_t *nlb_phi_bias, const float *nlb_phi_scale,
    const int8_t *nlb_g_weights, const int32_t *nlb_g_bias,
    const float *nlb_g_scale, const float *nlb_matmul_scale,
    const float *softmax_input_scale, const float *softmax_output_scale,
    const float *nlb_matmul_1_scale, const int8_t *nlb_out_weights,
    const int32_t *nlb_out_bias, const float *nlb_out_scale,
    const float *nlb_add_a_scale, const float *nlb_add_b_scale,
    const float *leaky1_scale, const int8_t *conv2_weights,
    const int32_t *conv2_bias, const float *conv2_scale,
    const float *leaky3_scale, const int8_t *conv3_weights,
    const int32_t *conv3_bias, const float *conv3_scale,
    const float *leaky5_scale, const int8_t *fc1_weights,
    const int32_t *fc1_bias, const float *fc1_scale,
    const float *dense1_leaky_scale, const int8_t *fc2_weights,
    const int32_t *fc2_bias, const float *fc2_scale,
    const float *dense3_leaky_scale, const int8_t *fc3_weights,
    const int32_t *fc3_bias, const float *fc3_scale,
    const float *dense5_leaky_scale, const int8_t *fc4_weights,
    const int32_t *fc4_bias, const float *fc4_scale,
    const float *dense7_leaky_scale, const int8_t *output_weights,
    const int32_t *output_bias, const float *output_scale, int8_t *output) {
  (void)ctxt;
  (void)conv1_weights;
  (void)conv1_bias;
  (void)conv1_scale;
  (void)nlb_theta_weights;
  (void)nlb_theta_bias;
  (void)nlb_theta_scale;
  (void)nlb_phi_weights;
  (void)nlb_phi_bias;
  (void)nlb_phi_scale;
  (void)nlb_g_weights;
  (void)nlb_g_bias;
  (void)nlb_g_scale;
  (void)nlb_matmul_scale;
  (void)softmax_input_scale;
  (void)softmax_output_scale;
  (void)nlb_matmul_1_scale;
  (void)nlb_out_weights;
  (void)nlb_out_bias;
  (void)nlb_out_scale;
  (void)nlb_add_a_scale;
  (void)nlb_add_b_scale;
  (void)leaky1_scale;
  (void)conv2_weights;
  (void)conv2_bias;
  (void)conv2_scale;
  (void)leaky3_scale;
  (void)conv3_weights;
  (void)conv3_bias;
  (void)conv3_scale;
  (void)leaky5_scale;
  (void)fc1_weights;
  (void)fc1_bias;
  (void)fc1_scale;
  (void)dense1_leaky_scale;
  (void)fc2_weights;
  (void)fc2_bias;
  (void)fc2_scale;
  (void)dense3_leaky_scale;
  (void)fc3_weights;
  (void)fc3_bias;
  (void)fc3_scale;
  (void)dense5_leaky_scale;
  (void)fc4_weights;
  (void)fc4_bias;
  (void)fc4_scale;
  (void)dense7_leaky_scale;
  (void)output_weights;
  (void)output_bias;
  (void)output_scale;

  gemmini_inference((float *)fp32_input, output);
}
