
#pragma once
#ifndef BRAGGNN_EXO_H
#define BRAGGNN_EXO_H

#ifdef __cplusplus
extern "C" {
#endif


#include <stdint.h>
#include <stdbool.h>

// Compiler feature macros adapted from Hedley (public domain)
// https://github.com/nemequ/hedley

#if defined(__has_builtin)
#  define EXO_HAS_BUILTIN(builtin) __has_builtin(builtin)
#else
#  define EXO_HAS_BUILTIN(builtin) (0)
#endif

#if EXO_HAS_BUILTIN(__builtin_assume)
#  define EXO_ASSUME(expr) __builtin_assume(expr)
#elif EXO_HAS_BUILTIN(__builtin_unreachable)
#  define EXO_ASSUME(expr) \
      ((void)((expr) ? 1 : (__builtin_unreachable(), 1)))
#else
#  define EXO_ASSUME(expr) ((void)(expr))
#endif


#ifndef EXO_WIN_2I32
#define EXO_WIN_2I32
struct exo_win_2i32{
    int32_t * const data;
    const int_fast32_t strides[2];
};
#endif
#ifndef EXO_WIN_2I32C
#define EXO_WIN_2I32C
struct exo_win_2i32c{
    const int32_t * const data;
    const int_fast32_t strides[2];
};
#endif
#ifndef EXO_WIN_2I8
#define EXO_WIN_2I8
struct exo_win_2i8{
    int8_t * const data;
    const int_fast32_t strides[2];
};
#endif
#ifndef EXO_WIN_2I8C
#define EXO_WIN_2I8C
struct exo_win_2i8c{
    const int8_t * const data;
    const int_fast32_t strides[2];
};
#endif
// braggnn_inference(
//     fp32_input : f32[11, 11, 1] @DRAM,
//     conv1_weights : i8[64, 3, 3, 1] @DRAM,
//     conv1_bias : i32[64] @DRAM,
//     conv1_scale : f32 @DRAM,
//     nlb_theta_weights : i8[32, 1, 1, 64] @DRAM,
//     nlb_theta_bias : i32[32] @DRAM,
//     nlb_theta_scale : f32 @DRAM,
//     nlb_phi_weights : i8[32, 1, 1, 64] @DRAM,
//     nlb_phi_bias : i32[32] @DRAM,
//     nlb_phi_scale : f32 @DRAM,
//     nlb_g_weights : i8[32, 1, 1, 64] @DRAM,
//     nlb_g_bias : i32[32] @DRAM,
//     nlb_g_scale : f32 @DRAM,
//     nlb_matmul_scale : f32 @DRAM,
//     softmax_input_scale : f32 @DRAM,
//     softmax_output_scale : f32 @DRAM,
//     nlb_matmul_1_scale : f32 @DRAM,
//     nlb_out_weights : i8[64, 1, 1, 32] @DRAM,
//     nlb_out_bias : i32[64] @DRAM,
//     nlb_out_scale : f32 @DRAM,
//     nlb_add_a_scale : f32 @DRAM,
//     nlb_add_b_scale : f32 @DRAM,
//     leaky1_scale : f32 @DRAM,
//     conv2_weights : i8[32, 3, 3, 64] @DRAM,
//     conv2_bias : i32[32] @DRAM,
//     conv2_scale : f32 @DRAM,
//     leaky3_scale : f32 @DRAM,
//     conv3_weights : i8[8, 3, 3, 32] @DRAM,
//     conv3_bias : i32[8] @DRAM,
//     conv3_scale : f32 @DRAM,
//     leaky5_scale : f32 @DRAM,
//     fc1_weights : i8[16, 8, 5, 5] @DRAM,
//     fc1_bias : i32[16] @DRAM,
//     fc1_scale : f32 @DRAM,
//     dense1_leaky_scale : f32 @DRAM,
//     fc2_weights : i8[8, 16, 1, 1] @DRAM,
//     fc2_bias : i32[8] @DRAM,
//     fc2_scale : f32 @DRAM,
//     dense3_leaky_scale : f32 @DRAM,
//     fc3_weights : i8[4, 8, 1, 1] @DRAM,
//     fc3_bias : i32[4] @DRAM,
//     fc3_scale : f32 @DRAM,
//     dense5_leaky_scale : f32 @DRAM,
//     fc4_weights : i8[2, 4, 1, 1] @DRAM,
//     fc4_bias : i32[2] @DRAM,
//     fc4_scale : f32 @DRAM,
//     dense7_leaky_scale : f32 @DRAM,
//     output_weights : i8[2, 2, 1, 1] @DRAM,
//     output_bias : i32[2] @DRAM,
//     output_scale : f32 @DRAM,
//     output : i8[2, 1, 1] @DRAM
// )
void braggnn_inference( void *ctxt, const float* fp32_input, const int8_t* conv1_weights, const int32_t* conv1_bias, const float* conv1_scale, const int8_t* nlb_theta_weights, const int32_t* nlb_theta_bias, const float* nlb_theta_scale, const int8_t* nlb_phi_weights, const int32_t* nlb_phi_bias, const float* nlb_phi_scale, const int8_t* nlb_g_weights, const int32_t* nlb_g_bias, const float* nlb_g_scale, const float* nlb_matmul_scale, const float* softmax_input_scale, const float* softmax_output_scale, const float* nlb_matmul_1_scale, const int8_t* nlb_out_weights, const int32_t* nlb_out_bias, const float* nlb_out_scale, const float* nlb_add_a_scale, const float* nlb_add_b_scale, const float* leaky1_scale, const int8_t* conv2_weights, const int32_t* conv2_bias, const float* conv2_scale, const float* leaky3_scale, const int8_t* conv3_weights, const int32_t* conv3_bias, const float* conv3_scale, const float* leaky5_scale, const int8_t* fc1_weights, const int32_t* fc1_bias, const float* fc1_scale, const float* dense1_leaky_scale, const int8_t* fc2_weights, const int32_t* fc2_bias, const float* fc2_scale, const float* dense3_leaky_scale, const int8_t* fc3_weights, const int32_t* fc3_bias, const float* fc3_scale, const float* dense5_leaky_scale, const int8_t* fc4_weights, const int32_t* fc4_bias, const float* fc4_scale, const float* dense7_leaky_scale, const int8_t* output_weights, const int32_t* output_bias, const float* output_scale, int8_t* output );



#ifdef __cplusplus
}
#endif
#endif  // BRAGGNN_EXO_H
