#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

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
    const float *resadd_post_scale, const int8_t *conv2_weights,
    const int32_t *conv2_bias, const float *conv2_scale,
    const float *conv2_post_scale, const int8_t *conv3_weights,
    const int32_t *conv3_bias, const float *conv3_scale,
    const float *conv3_post_scale, const int8_t *fc1_weights,
    const int32_t *fc1_bias, const float *fc1_scale,
    const float *fc1_post_scale, const int8_t *fc2_weights,
    const int32_t *fc2_bias, const float *fc2_scale,
    const float *fc2_post_scale, const int8_t *fc3_weights,
    const int32_t *fc3_bias, const float *fc3_scale,
    const float *fc3_post_scale, const int8_t *fc4_weights,
    const int32_t *fc4_bias, const float *fc4_scale,
    const float *fc4_post_scale, const int8_t *output_weights,
    const int32_t *output_bias, const float *output_scale, int8_t *output);

#ifdef __cplusplus
}
#endif
