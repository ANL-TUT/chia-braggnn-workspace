#include "braggnn_exo.h"

#include <stdio.h>
#include <stdlib.h>

#include <stdio.h>
#include <stdlib.h>

#include <include/gemmini.h>
#include "gemm_acc_malloc.h"
#include <include/gemmini.h>
#include "gemm_malloc.h"
#include <math.h>
float _relu_float(float x) {
    if (x > 0.0) return x;
    else return 0.0;
}

float _select_float(float x,float v,float y,float z) {
    if (x < v) return y;
    else return z;
}

// conv1_opt(
//     input : i8[11, 11, 1] @DRAM,
//     weights : i8[64, 3, 3, 1] @DRAM,
//     bias : i32[64] @DRAM,
//     output : i8[9, 9, 64] @DRAM,
//     acc_scale : f32 @DRAM
// )
static void conv1_opt( void *ctxt, const int8_t* input, const int8_t* weights, const int32_t* bias, int8_t* output, const float* acc_scale );

// conv2_gemmini(
//     input : i8[9, 9, 64] @DRAM,
//     weights : i8[32, 3, 3, 64] @DRAM,
//     bias : i32[32] @DRAM,
//     output : i8[7, 7, 32] @DRAM,
//     acc_scale : f32 @DRAM
// )
static void conv2_gemmini( void *ctxt, const int8_t* input, const int8_t* weights, const int32_t* bias, int8_t* output, const float* acc_scale );

// conv3_gemmini(
//     input : i8[7, 7, 32] @DRAM,
//     weights : i8[8, 3, 3, 32] @DRAM,
//     bias : i32[8] @DRAM,
//     output : i8[5, 5, 8] @DRAM,
//     acc_scale : f32 @DRAM
// )
static void conv3_gemmini( void *ctxt, const int8_t* input, const int8_t* weights, const int32_t* bias, int8_t* output, const float* acc_scale );

// dense1_leaky_opt(
//     values : i8[16, 1, 1] @DRAM,
//     scale : f32 @DRAM
// )
static void dense1_leaky_opt( void *ctxt, int8_t* values, const float* scale );

// dense3_leaky_opt(
//     values : i8[8, 1, 1] @DRAM,
//     scale : f32 @DRAM
// )
static void dense3_leaky_opt( void *ctxt, int8_t* values, const float* scale );

// dense5_leaky_opt(
//     values : i8[4, 1, 1] @DRAM,
//     scale : f32 @DRAM
// )
static void dense5_leaky_opt( void *ctxt, int8_t* values, const float* scale );

// dense7_leaky_opt(
//     values : i8[2, 1, 1] @DRAM,
//     scale : f32 @DRAM
// )
static void dense7_leaky_opt( void *ctxt, int8_t* values, const float* scale );

// fc1_opt(
//     input : i8[8, 5, 5] @DRAM,
//     weights : i8[16, 8, 5, 5] @DRAM,
//     bias : i32[16] @DRAM,
//     output : i8[16, 1, 1] @DRAM,
//     acc_scale : f32 @DRAM
// )
static void fc1_opt( void *ctxt, const int8_t* input, const int8_t* weights, const int32_t* bias, int8_t* output, const float* acc_scale );

// fc2_opt(
//     input : i8[16, 1, 1] @DRAM,
//     weights : i8[8, 16, 1, 1] @DRAM,
//     bias : i32[8] @DRAM,
//     output : i8[8, 1, 1] @DRAM,
//     acc_scale : f32 @DRAM
// )
static void fc2_opt( void *ctxt, const int8_t* input, const int8_t* weights, const int32_t* bias, int8_t* output, const float* acc_scale );

// fc3_opt(
//     input : i8[8, 1, 1] @DRAM,
//     weights : i8[4, 8, 1, 1] @DRAM,
//     bias : i32[4] @DRAM,
//     output : i8[4, 1, 1] @DRAM,
//     acc_scale : f32 @DRAM
// )
static void fc3_opt( void *ctxt, const int8_t* input, const int8_t* weights, const int32_t* bias, int8_t* output, const float* acc_scale );

// fc4_opt(
//     input : i8[4, 1, 1] @DRAM,
//     weights : i8[2, 4, 1, 1] @DRAM,
//     bias : i32[2] @DRAM,
//     output : i8[2, 1, 1] @DRAM,
//     acc_scale : f32 @DRAM
// )
static void fc4_opt( void *ctxt, const int8_t* input, const int8_t* weights, const int32_t* bias, int8_t* output, const float* acc_scale );

// fc_output_opt(
//     input : i8[2, 1, 1] @DRAM,
//     weights : i8[2, 2, 1, 1] @DRAM,
//     bias : i32[2] @DRAM,
//     output : i8[2, 1, 1] @DRAM,
//     acc_scale : f32 @DRAM
// )
static void fc_output_opt( void *ctxt, const int8_t* input, const int8_t* weights, const int32_t* bias, int8_t* output, const float* acc_scale );

// leaky1_opt(
//     values : i8[9, 9, 64] @DRAM,
//     scale : f32 @DRAM
// )
static void leaky1_opt( void *ctxt, int8_t* values, const float* scale );

// leaky3_opt(
//     values : i8[7, 7, 32] @DRAM,
//     scale : f32 @DRAM
// )
static void leaky3_opt( void *ctxt, int8_t* values, const float* scale );

// leaky5_opt(
//     values : i8[8, 5, 5] @DRAM,
//     scale : f32 @DRAM
// )
static void leaky5_opt( void *ctxt, int8_t* values, const float* scale );

// matmul_transA_gemmini(
//     A : i8[32, 9, 9] @DRAM,
//     B : i8[32, 9, 9] @DRAM,
//     C : i8[9, 9, 9, 9] @DRAM,
//     acc_scale : f32 @DRAM
// )
static void matmul_transA_gemmini( void *ctxt, const int8_t* A, const int8_t* B, int8_t* C, const float* acc_scale );

// matmul_transB_gemmini(
//     A : i8[9, 9, 9, 9] @DRAM,
//     B : i8[32, 9, 9] @DRAM,
//     C : i8[9, 9, 32] @DRAM,
//     acc_scale : f32 @DRAM
// )
static void matmul_transB_gemmini( void *ctxt, const int8_t* A, const int8_t* B, int8_t* C, const float* acc_scale );

// nlb_gemmini(
//     input : i8[9, 9, 64] @DRAM,
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
//     output : i8[9, 9, 64] @DRAM
// )
static void nlb_gemmini( void *ctxt, const int8_t* input, const int8_t* nlb_theta_weights, const int32_t* nlb_theta_bias, const float* nlb_theta_scale, const int8_t* nlb_phi_weights, const int32_t* nlb_phi_bias, const float* nlb_phi_scale, const int8_t* nlb_g_weights, const int32_t* nlb_g_bias, const float* nlb_g_scale, const float* nlb_matmul_scale, const float* softmax_input_scale, const float* softmax_output_scale, const float* nlb_matmul_1_scale, const int8_t* nlb_out_weights, const int32_t* nlb_out_bias, const float* nlb_out_scale, const float* nlb_add_a_scale, const float* nlb_add_b_scale, int8_t* output );

// nlb_out_conv_gemmini(
//     input : i8[9, 9, 32] @DRAM,
//     weights : i8[64, 1, 1, 32] @DRAM,
//     bias : i32[64] @DRAM,
//     output : i8[9, 9, 64] @DRAM,
//     acc_scale : f32 @DRAM
// )
static void nlb_out_conv_gemmini( void *ctxt, const int8_t* input, const int8_t* weights, const int32_t* bias, int8_t* output, const float* acc_scale );

// nlb_qkv_conv_gemmini(
//     input : i8[9, 9, 64] @DRAM,
//     weights : i8[32, 1, 1, 64] @DRAM,
//     bias : i32[32] @DRAM,
//     output : i8[9, 9, 32] @DRAM,
//     acc_scale : f32 @DRAM
// )
static void nlb_qkv_conv_gemmini( void *ctxt, const int8_t* input, const int8_t* weights, const int32_t* bias, int8_t* output, const float* acc_scale );

// resadd_opt(
//     A_scale : f32 @DRAM,
//     B_scale : f32 @DRAM,
//     A : i8[9, 9, 64] @DRAM,
//     B : i8[9, 9, 64] @DRAM,
//     C : i8[9, 9, 64] @DRAM
// )
static void resadd_opt( void *ctxt, const float* A_scale, const float* B_scale, const int8_t* A, const int8_t* B, int8_t* C );

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
void braggnn_inference( void *ctxt, const float* fp32_input, const int8_t* conv1_weights, const int32_t* conv1_bias, const float* conv1_scale, const int8_t* nlb_theta_weights, const int32_t* nlb_theta_bias, const float* nlb_theta_scale, const int8_t* nlb_phi_weights, const int32_t* nlb_phi_bias, const float* nlb_phi_scale, const int8_t* nlb_g_weights, const int32_t* nlb_g_bias, const float* nlb_g_scale, const float* nlb_matmul_scale, const float* softmax_input_scale, const float* softmax_output_scale, const float* nlb_matmul_1_scale, const int8_t* nlb_out_weights, const int32_t* nlb_out_bias, const float* nlb_out_scale, const float* nlb_add_a_scale, const float* nlb_add_b_scale, const float* leaky1_scale, const int8_t* conv2_weights, const int32_t* conv2_bias, const float* conv2_scale, const float* leaky3_scale, const int8_t* conv3_weights, const int32_t* conv3_bias, const float* conv3_scale, const float* leaky5_scale, const int8_t* fc1_weights, const int32_t* fc1_bias, const float* fc1_scale, const float* dense1_leaky_scale, const int8_t* fc2_weights, const int32_t* fc2_bias, const float* fc2_scale, const float* dense3_leaky_scale, const int8_t* fc3_weights, const int32_t* fc3_bias, const float* fc3_scale, const float* dense5_leaky_scale, const int8_t* fc4_weights, const int32_t* fc4_bias, const float* fc4_scale, const float* dense7_leaky_scale, const int8_t* output_weights, const int32_t* output_bias, const float* output_scale, int8_t* output ) {
static int8_t input[11 * 11 * 1];
float q;
q = fp32_input[0] * 127.0f;
input[0] = (int8_t)(q);
float q_1;
q_1 = fp32_input[1] * 127.0f;
input[1] = (int8_t)(q_1);
float q_2;
q_2 = fp32_input[2] * 127.0f;
input[2] = (int8_t)(q_2);
float q_3;
q_3 = fp32_input[3] * 127.0f;
input[3] = (int8_t)(q_3);
float q_4;
q_4 = fp32_input[4] * 127.0f;
input[4] = (int8_t)(q_4);
float q_5;
q_5 = fp32_input[5] * 127.0f;
input[5] = (int8_t)(q_5);
float q_6;
q_6 = fp32_input[6] * 127.0f;
input[6] = (int8_t)(q_6);
float q_7;
q_7 = fp32_input[7] * 127.0f;
input[7] = (int8_t)(q_7);
float q_8;
q_8 = fp32_input[8] * 127.0f;
input[8] = (int8_t)(q_8);
float q_9;
q_9 = fp32_input[9] * 127.0f;
input[9] = (int8_t)(q_9);
float q_10;
q_10 = fp32_input[10] * 127.0f;
input[10] = (int8_t)(q_10);
float q_11;
q_11 = fp32_input[11] * 127.0f;
input[11] = (int8_t)(q_11);
float q_12;
q_12 = fp32_input[12] * 127.0f;
input[12] = (int8_t)(q_12);
float q_13;
q_13 = fp32_input[13] * 127.0f;
input[13] = (int8_t)(q_13);
float q_14;
q_14 = fp32_input[14] * 127.0f;
input[14] = (int8_t)(q_14);
float q_15;
q_15 = fp32_input[15] * 127.0f;
input[15] = (int8_t)(q_15);
float q_16;
q_16 = fp32_input[16] * 127.0f;
input[16] = (int8_t)(q_16);
float q_17;
q_17 = fp32_input[17] * 127.0f;
input[17] = (int8_t)(q_17);
float q_18;
q_18 = fp32_input[18] * 127.0f;
input[18] = (int8_t)(q_18);
float q_19;
q_19 = fp32_input[19] * 127.0f;
input[19] = (int8_t)(q_19);
float q_20;
q_20 = fp32_input[20] * 127.0f;
input[20] = (int8_t)(q_20);
float q_21;
q_21 = fp32_input[21] * 127.0f;
input[21] = (int8_t)(q_21);
float q_22;
q_22 = fp32_input[22] * 127.0f;
input[22] = (int8_t)(q_22);
float q_23;
q_23 = fp32_input[23] * 127.0f;
input[23] = (int8_t)(q_23);
float q_24;
q_24 = fp32_input[24] * 127.0f;
input[24] = (int8_t)(q_24);
float q_25;
q_25 = fp32_input[25] * 127.0f;
input[25] = (int8_t)(q_25);
float q_26;
q_26 = fp32_input[26] * 127.0f;
input[26] = (int8_t)(q_26);
float q_27;
q_27 = fp32_input[27] * 127.0f;
input[27] = (int8_t)(q_27);
float q_28;
q_28 = fp32_input[28] * 127.0f;
input[28] = (int8_t)(q_28);
float q_29;
q_29 = fp32_input[29] * 127.0f;
input[29] = (int8_t)(q_29);
float q_30;
q_30 = fp32_input[30] * 127.0f;
input[30] = (int8_t)(q_30);
float q_31;
q_31 = fp32_input[31] * 127.0f;
input[31] = (int8_t)(q_31);
float q_32;
q_32 = fp32_input[32] * 127.0f;
input[32] = (int8_t)(q_32);
float q_33;
q_33 = fp32_input[33] * 127.0f;
input[33] = (int8_t)(q_33);
float q_34;
q_34 = fp32_input[34] * 127.0f;
input[34] = (int8_t)(q_34);
float q_35;
q_35 = fp32_input[35] * 127.0f;
input[35] = (int8_t)(q_35);
float q_36;
q_36 = fp32_input[36] * 127.0f;
input[36] = (int8_t)(q_36);
float q_37;
q_37 = fp32_input[37] * 127.0f;
input[37] = (int8_t)(q_37);
float q_38;
q_38 = fp32_input[38] * 127.0f;
input[38] = (int8_t)(q_38);
float q_39;
q_39 = fp32_input[39] * 127.0f;
input[39] = (int8_t)(q_39);
float q_40;
q_40 = fp32_input[40] * 127.0f;
input[40] = (int8_t)(q_40);
float q_41;
q_41 = fp32_input[41] * 127.0f;
input[41] = (int8_t)(q_41);
float q_42;
q_42 = fp32_input[42] * 127.0f;
input[42] = (int8_t)(q_42);
float q_43;
q_43 = fp32_input[43] * 127.0f;
input[43] = (int8_t)(q_43);
float q_44;
q_44 = fp32_input[44] * 127.0f;
input[44] = (int8_t)(q_44);
float q_45;
q_45 = fp32_input[45] * 127.0f;
input[45] = (int8_t)(q_45);
float q_46;
q_46 = fp32_input[46] * 127.0f;
input[46] = (int8_t)(q_46);
float q_47;
q_47 = fp32_input[47] * 127.0f;
input[47] = (int8_t)(q_47);
float q_48;
q_48 = fp32_input[48] * 127.0f;
input[48] = (int8_t)(q_48);
float q_49;
q_49 = fp32_input[49] * 127.0f;
input[49] = (int8_t)(q_49);
float q_50;
q_50 = fp32_input[50] * 127.0f;
input[50] = (int8_t)(q_50);
float q_51;
q_51 = fp32_input[51] * 127.0f;
input[51] = (int8_t)(q_51);
float q_52;
q_52 = fp32_input[52] * 127.0f;
input[52] = (int8_t)(q_52);
float q_53;
q_53 = fp32_input[53] * 127.0f;
input[53] = (int8_t)(q_53);
float q_54;
q_54 = fp32_input[54] * 127.0f;
input[54] = (int8_t)(q_54);
float q_55;
q_55 = fp32_input[55] * 127.0f;
input[55] = (int8_t)(q_55);
float q_56;
q_56 = fp32_input[56] * 127.0f;
input[56] = (int8_t)(q_56);
float q_57;
q_57 = fp32_input[57] * 127.0f;
input[57] = (int8_t)(q_57);
float q_58;
q_58 = fp32_input[58] * 127.0f;
input[58] = (int8_t)(q_58);
float q_59;
q_59 = fp32_input[59] * 127.0f;
input[59] = (int8_t)(q_59);
float q_60;
q_60 = fp32_input[60] * 127.0f;
input[60] = (int8_t)(q_60);
float q_61;
q_61 = fp32_input[61] * 127.0f;
input[61] = (int8_t)(q_61);
float q_62;
q_62 = fp32_input[62] * 127.0f;
input[62] = (int8_t)(q_62);
float q_63;
q_63 = fp32_input[63] * 127.0f;
input[63] = (int8_t)(q_63);
float q_64;
q_64 = fp32_input[64] * 127.0f;
input[64] = (int8_t)(q_64);
float q_65;
q_65 = fp32_input[65] * 127.0f;
input[65] = (int8_t)(q_65);
float q_66;
q_66 = fp32_input[66] * 127.0f;
input[66] = (int8_t)(q_66);
float q_67;
q_67 = fp32_input[67] * 127.0f;
input[67] = (int8_t)(q_67);
float q_68;
q_68 = fp32_input[68] * 127.0f;
input[68] = (int8_t)(q_68);
float q_69;
q_69 = fp32_input[69] * 127.0f;
input[69] = (int8_t)(q_69);
float q_70;
q_70 = fp32_input[70] * 127.0f;
input[70] = (int8_t)(q_70);
float q_71;
q_71 = fp32_input[71] * 127.0f;
input[71] = (int8_t)(q_71);
float q_72;
q_72 = fp32_input[72] * 127.0f;
input[72] = (int8_t)(q_72);
float q_73;
q_73 = fp32_input[73] * 127.0f;
input[73] = (int8_t)(q_73);
float q_74;
q_74 = fp32_input[74] * 127.0f;
input[74] = (int8_t)(q_74);
float q_75;
q_75 = fp32_input[75] * 127.0f;
input[75] = (int8_t)(q_75);
float q_76;
q_76 = fp32_input[76] * 127.0f;
input[76] = (int8_t)(q_76);
float q_77;
q_77 = fp32_input[77] * 127.0f;
input[77] = (int8_t)(q_77);
float q_78;
q_78 = fp32_input[78] * 127.0f;
input[78] = (int8_t)(q_78);
float q_79;
q_79 = fp32_input[79] * 127.0f;
input[79] = (int8_t)(q_79);
float q_80;
q_80 = fp32_input[80] * 127.0f;
input[80] = (int8_t)(q_80);
float q_81;
q_81 = fp32_input[81] * 127.0f;
input[81] = (int8_t)(q_81);
float q_82;
q_82 = fp32_input[82] * 127.0f;
input[82] = (int8_t)(q_82);
float q_83;
q_83 = fp32_input[83] * 127.0f;
input[83] = (int8_t)(q_83);
float q_84;
q_84 = fp32_input[84] * 127.0f;
input[84] = (int8_t)(q_84);
float q_85;
q_85 = fp32_input[85] * 127.0f;
input[85] = (int8_t)(q_85);
float q_86;
q_86 = fp32_input[86] * 127.0f;
input[86] = (int8_t)(q_86);
float q_87;
q_87 = fp32_input[87] * 127.0f;
input[87] = (int8_t)(q_87);
float q_88;
q_88 = fp32_input[88] * 127.0f;
input[88] = (int8_t)(q_88);
float q_89;
q_89 = fp32_input[89] * 127.0f;
input[89] = (int8_t)(q_89);
float q_90;
q_90 = fp32_input[90] * 127.0f;
input[90] = (int8_t)(q_90);
float q_91;
q_91 = fp32_input[91] * 127.0f;
input[91] = (int8_t)(q_91);
float q_92;
q_92 = fp32_input[92] * 127.0f;
input[92] = (int8_t)(q_92);
float q_93;
q_93 = fp32_input[93] * 127.0f;
input[93] = (int8_t)(q_93);
float q_94;
q_94 = fp32_input[94] * 127.0f;
input[94] = (int8_t)(q_94);
float q_95;
q_95 = fp32_input[95] * 127.0f;
input[95] = (int8_t)(q_95);
float q_96;
q_96 = fp32_input[96] * 127.0f;
input[96] = (int8_t)(q_96);
float q_97;
q_97 = fp32_input[97] * 127.0f;
input[97] = (int8_t)(q_97);
float q_98;
q_98 = fp32_input[98] * 127.0f;
input[98] = (int8_t)(q_98);
float q_99;
q_99 = fp32_input[99] * 127.0f;
input[99] = (int8_t)(q_99);
float q_100;
q_100 = fp32_input[100] * 127.0f;
input[100] = (int8_t)(q_100);
float q_101;
q_101 = fp32_input[101] * 127.0f;
input[101] = (int8_t)(q_101);
float q_102;
q_102 = fp32_input[102] * 127.0f;
input[102] = (int8_t)(q_102);
float q_103;
q_103 = fp32_input[103] * 127.0f;
input[103] = (int8_t)(q_103);
float q_104;
q_104 = fp32_input[104] * 127.0f;
input[104] = (int8_t)(q_104);
float q_105;
q_105 = fp32_input[105] * 127.0f;
input[105] = (int8_t)(q_105);
float q_106;
q_106 = fp32_input[106] * 127.0f;
input[106] = (int8_t)(q_106);
float q_107;
q_107 = fp32_input[107] * 127.0f;
input[107] = (int8_t)(q_107);
float q_108;
q_108 = fp32_input[108] * 127.0f;
input[108] = (int8_t)(q_108);
float q_109;
q_109 = fp32_input[109] * 127.0f;
input[109] = (int8_t)(q_109);
float q_110;
q_110 = fp32_input[110] * 127.0f;
input[110] = (int8_t)(q_110);
float q_111;
q_111 = fp32_input[111] * 127.0f;
input[111] = (int8_t)(q_111);
float q_112;
q_112 = fp32_input[112] * 127.0f;
input[112] = (int8_t)(q_112);
float q_113;
q_113 = fp32_input[113] * 127.0f;
input[113] = (int8_t)(q_113);
float q_114;
q_114 = fp32_input[114] * 127.0f;
input[114] = (int8_t)(q_114);
float q_115;
q_115 = fp32_input[115] * 127.0f;
input[115] = (int8_t)(q_115);
float q_116;
q_116 = fp32_input[116] * 127.0f;
input[116] = (int8_t)(q_116);
float q_117;
q_117 = fp32_input[117] * 127.0f;
input[117] = (int8_t)(q_117);
float q_118;
q_118 = fp32_input[118] * 127.0f;
input[118] = (int8_t)(q_118);
float q_119;
q_119 = fp32_input[119] * 127.0f;
input[119] = (int8_t)(q_119);
float q_120;
q_120 = fp32_input[120] * 127.0f;
input[120] = (int8_t)(q_120);
static int8_t conv1_out[9 * 9 * 64];
conv1_opt(ctxt,input,conv1_weights,conv1_bias,conv1_out,conv1_scale);
static int8_t nlb_out[9 * 9 * 64];
nlb_gemmini(ctxt,conv1_out,nlb_theta_weights,nlb_theta_bias,nlb_theta_scale,nlb_phi_weights,nlb_phi_bias,nlb_phi_scale,nlb_g_weights,nlb_g_bias,nlb_g_scale,nlb_matmul_scale,softmax_input_scale,softmax_output_scale,nlb_matmul_1_scale,nlb_out_weights,nlb_out_bias,nlb_out_scale,nlb_add_a_scale,nlb_add_b_scale,nlb_out);
leaky1_opt(ctxt,nlb_out,leaky1_scale);
static int8_t conv2_out[7 * 7 * 32];
conv2_gemmini(ctxt,nlb_out,conv2_weights,conv2_bias,conv2_out,conv2_scale);
leaky3_opt(ctxt,conv2_out,leaky3_scale);
static int8_t conv3_out[5 * 5 * 8];
conv3_gemmini(ctxt,conv2_out,conv3_weights,conv3_bias,conv3_out,conv3_scale);
static int8_t flattened[8 * 5 * 5];
flattened[0] = conv3_out[0];
flattened[1] = conv3_out[8];
flattened[2] = conv3_out[16];
flattened[3] = conv3_out[24];
flattened[4] = conv3_out[32];
flattened[5] = conv3_out[40];
flattened[6] = conv3_out[48];
flattened[7] = conv3_out[56];
flattened[8] = conv3_out[64];
flattened[9] = conv3_out[72];
flattened[10] = conv3_out[80];
flattened[11] = conv3_out[88];
flattened[12] = conv3_out[96];
flattened[13] = conv3_out[104];
flattened[14] = conv3_out[112];
flattened[15] = conv3_out[120];
flattened[16] = conv3_out[128];
flattened[17] = conv3_out[136];
flattened[18] = conv3_out[144];
flattened[19] = conv3_out[152];
flattened[20] = conv3_out[160];
flattened[21] = conv3_out[168];
flattened[22] = conv3_out[176];
flattened[23] = conv3_out[184];
flattened[24] = conv3_out[192];
flattened[25] = conv3_out[1];
flattened[26] = conv3_out[9];
flattened[27] = conv3_out[17];
flattened[28] = conv3_out[25];
flattened[29] = conv3_out[33];
flattened[30] = conv3_out[41];
flattened[31] = conv3_out[49];
flattened[32] = conv3_out[57];
flattened[33] = conv3_out[65];
flattened[34] = conv3_out[73];
flattened[35] = conv3_out[81];
flattened[36] = conv3_out[89];
flattened[37] = conv3_out[97];
flattened[38] = conv3_out[105];
flattened[39] = conv3_out[113];
flattened[40] = conv3_out[121];
flattened[41] = conv3_out[129];
flattened[42] = conv3_out[137];
flattened[43] = conv3_out[145];
flattened[44] = conv3_out[153];
flattened[45] = conv3_out[161];
flattened[46] = conv3_out[169];
flattened[47] = conv3_out[177];
flattened[48] = conv3_out[185];
flattened[49] = conv3_out[193];
flattened[50] = conv3_out[2];
flattened[51] = conv3_out[10];
flattened[52] = conv3_out[18];
flattened[53] = conv3_out[26];
flattened[54] = conv3_out[34];
flattened[55] = conv3_out[42];
flattened[56] = conv3_out[50];
flattened[57] = conv3_out[58];
flattened[58] = conv3_out[66];
flattened[59] = conv3_out[74];
flattened[60] = conv3_out[82];
flattened[61] = conv3_out[90];
flattened[62] = conv3_out[98];
flattened[63] = conv3_out[106];
flattened[64] = conv3_out[114];
flattened[65] = conv3_out[122];
flattened[66] = conv3_out[130];
flattened[67] = conv3_out[138];
flattened[68] = conv3_out[146];
flattened[69] = conv3_out[154];
flattened[70] = conv3_out[162];
flattened[71] = conv3_out[170];
flattened[72] = conv3_out[178];
flattened[73] = conv3_out[186];
flattened[74] = conv3_out[194];
flattened[75] = conv3_out[3];
flattened[76] = conv3_out[11];
flattened[77] = conv3_out[19];
flattened[78] = conv3_out[27];
flattened[79] = conv3_out[35];
flattened[80] = conv3_out[43];
flattened[81] = conv3_out[51];
flattened[82] = conv3_out[59];
flattened[83] = conv3_out[67];
flattened[84] = conv3_out[75];
flattened[85] = conv3_out[83];
flattened[86] = conv3_out[91];
flattened[87] = conv3_out[99];
flattened[88] = conv3_out[107];
flattened[89] = conv3_out[115];
flattened[90] = conv3_out[123];
flattened[91] = conv3_out[131];
flattened[92] = conv3_out[139];
flattened[93] = conv3_out[147];
flattened[94] = conv3_out[155];
flattened[95] = conv3_out[163];
flattened[96] = conv3_out[171];
flattened[97] = conv3_out[179];
flattened[98] = conv3_out[187];
flattened[99] = conv3_out[195];
flattened[100] = conv3_out[4];
flattened[101] = conv3_out[12];
flattened[102] = conv3_out[20];
flattened[103] = conv3_out[28];
flattened[104] = conv3_out[36];
flattened[105] = conv3_out[44];
flattened[106] = conv3_out[52];
flattened[107] = conv3_out[60];
flattened[108] = conv3_out[68];
flattened[109] = conv3_out[76];
flattened[110] = conv3_out[84];
flattened[111] = conv3_out[92];
flattened[112] = conv3_out[100];
flattened[113] = conv3_out[108];
flattened[114] = conv3_out[116];
flattened[115] = conv3_out[124];
flattened[116] = conv3_out[132];
flattened[117] = conv3_out[140];
flattened[118] = conv3_out[148];
flattened[119] = conv3_out[156];
flattened[120] = conv3_out[164];
flattened[121] = conv3_out[172];
flattened[122] = conv3_out[180];
flattened[123] = conv3_out[188];
flattened[124] = conv3_out[196];
flattened[125] = conv3_out[5];
flattened[126] = conv3_out[13];
flattened[127] = conv3_out[21];
flattened[128] = conv3_out[29];
flattened[129] = conv3_out[37];
flattened[130] = conv3_out[45];
flattened[131] = conv3_out[53];
flattened[132] = conv3_out[61];
flattened[133] = conv3_out[69];
flattened[134] = conv3_out[77];
flattened[135] = conv3_out[85];
flattened[136] = conv3_out[93];
flattened[137] = conv3_out[101];
flattened[138] = conv3_out[109];
flattened[139] = conv3_out[117];
flattened[140] = conv3_out[125];
flattened[141] = conv3_out[133];
flattened[142] = conv3_out[141];
flattened[143] = conv3_out[149];
flattened[144] = conv3_out[157];
flattened[145] = conv3_out[165];
flattened[146] = conv3_out[173];
flattened[147] = conv3_out[181];
flattened[148] = conv3_out[189];
flattened[149] = conv3_out[197];
flattened[150] = conv3_out[6];
flattened[151] = conv3_out[14];
flattened[152] = conv3_out[22];
flattened[153] = conv3_out[30];
flattened[154] = conv3_out[38];
flattened[155] = conv3_out[46];
flattened[156] = conv3_out[54];
flattened[157] = conv3_out[62];
flattened[158] = conv3_out[70];
flattened[159] = conv3_out[78];
flattened[160] = conv3_out[86];
flattened[161] = conv3_out[94];
flattened[162] = conv3_out[102];
flattened[163] = conv3_out[110];
flattened[164] = conv3_out[118];
flattened[165] = conv3_out[126];
flattened[166] = conv3_out[134];
flattened[167] = conv3_out[142];
flattened[168] = conv3_out[150];
flattened[169] = conv3_out[158];
flattened[170] = conv3_out[166];
flattened[171] = conv3_out[174];
flattened[172] = conv3_out[182];
flattened[173] = conv3_out[190];
flattened[174] = conv3_out[198];
flattened[175] = conv3_out[7];
flattened[176] = conv3_out[15];
flattened[177] = conv3_out[23];
flattened[178] = conv3_out[31];
flattened[179] = conv3_out[39];
flattened[180] = conv3_out[47];
flattened[181] = conv3_out[55];
flattened[182] = conv3_out[63];
flattened[183] = conv3_out[71];
flattened[184] = conv3_out[79];
flattened[185] = conv3_out[87];
flattened[186] = conv3_out[95];
flattened[187] = conv3_out[103];
flattened[188] = conv3_out[111];
flattened[189] = conv3_out[119];
flattened[190] = conv3_out[127];
flattened[191] = conv3_out[135];
flattened[192] = conv3_out[143];
flattened[193] = conv3_out[151];
flattened[194] = conv3_out[159];
flattened[195] = conv3_out[167];
flattened[196] = conv3_out[175];
flattened[197] = conv3_out[183];
flattened[198] = conv3_out[191];
flattened[199] = conv3_out[199];
leaky5_opt(ctxt,flattened,leaky5_scale);
static int8_t fc1_out[16 * 1 * 1];
fc1_opt(ctxt,flattened,fc1_weights,fc1_bias,fc1_out,fc1_scale);
dense1_leaky_opt(ctxt,fc1_out,dense1_leaky_scale);
static int8_t fc2_out[8 * 1 * 1];
fc2_opt(ctxt,fc1_out,fc2_weights,fc2_bias,fc2_out,fc2_scale);
dense3_leaky_opt(ctxt,fc2_out,dense3_leaky_scale);
static int8_t fc3_out[4 * 1 * 1];
fc3_opt(ctxt,fc2_out,fc3_weights,fc3_bias,fc3_out,fc3_scale);
dense5_leaky_opt(ctxt,fc3_out,dense5_leaky_scale);
static int8_t fc4_out[2 * 1 * 1];
fc4_opt(ctxt,fc3_out,fc4_weights,fc4_bias,fc4_out,fc4_scale);
dense7_leaky_opt(ctxt,fc4_out,dense7_leaky_scale);
fc_output_opt(ctxt,fc4_out,output_weights,output_bias,output,output_scale);
}

// conv1_opt(
//     input : i8[11, 11, 1] @DRAM,
//     weights : i8[64, 3, 3, 1] @DRAM,
//     bias : i32[64] @DRAM,
//     output : i8[9, 9, 64] @DRAM,
//     acc_scale : f32 @DRAM
// )
static void conv1_opt( void *ctxt, const int8_t* input, const int8_t* weights, const int32_t* bias, int8_t* output, const float* acc_scale ) {
for (int_fast32_t oh = 0; oh < 9; oh++) {
  for (int_fast32_t ow = 0; ow < 9; ow++) {
    for (int_fast32_t oco = 0; oco < 8; oco++) {
      int32_t sum;
      sum = bias[8 * oco];
      int32_t x;
      int32_t w;
      x = (int32_t)(input[oh * 11 + ow]);
      w = (int32_t)(weights[8 * oco * 9]);
      sum += x * w;
      int32_t x_1;
      int32_t w_1;
      x_1 = (int32_t)(input[oh * 11 + (1 + ow)]);
      w_1 = (int32_t)(weights[8 * oco * 9 + 1]);
      sum += x_1 * w_1;
      int32_t x_2;
      int32_t w_2;
      x_2 = (int32_t)(input[oh * 11 + (2 + ow)]);
      w_2 = (int32_t)(weights[8 * oco * 9 + 2]);
      sum += x_2 * w_2;
      int32_t x_3;
      int32_t w_3;
      x_3 = (int32_t)(input[(1 + oh) * 11 + ow]);
      w_3 = (int32_t)(weights[8 * oco * 9 + 3]);
      sum += x_3 * w_3;
      int32_t x_4;
      int32_t w_4;
      x_4 = (int32_t)(input[(1 + oh) * 11 + (1 + ow)]);
      w_4 = (int32_t)(weights[8 * oco * 9 + 3 + 1]);
      sum += x_4 * w_4;
      int32_t x_5;
      int32_t w_5;
      x_5 = (int32_t)(input[(1 + oh) * 11 + (2 + ow)]);
      w_5 = (int32_t)(weights[8 * oco * 9 + 3 + 2]);
      sum += x_5 * w_5;
      int32_t x_6;
      int32_t w_6;
      x_6 = (int32_t)(input[(2 + oh) * 11 + ow]);
      w_6 = (int32_t)(weights[8 * oco * 9 + 6]);
      sum += x_6 * w_6;
      int32_t x_7;
      int32_t w_7;
      x_7 = (int32_t)(input[(2 + oh) * 11 + (1 + ow)]);
      w_7 = (int32_t)(weights[8 * oco * 9 + 6 + 1]);
      sum += x_7 * w_7;
      int32_t x_8;
      int32_t w_8;
      x_8 = (int32_t)(input[(2 + oh) * 11 + (2 + ow)]);
      w_8 = (int32_t)(weights[8 * oco * 9 + 6 + 2]);
      sum += x_8 * w_8;
      float val;
      val = (float)(sum);
      val = val * *acc_scale;
      val = val + _select_float((float)val, (float)0.0f, (float)-0.5f, (float)0.5f);
      val = fmaxf((float)(-128.0f), (float)(_select_float((float)val, (float)127.0f, (float)val, (float)127.0f)));
      output[oh * 576 + ow * 64 + 8 * oco] = (int8_t)(val);
      int32_t sum_1;
      sum_1 = bias[1 + 8 * oco];
      int32_t x_9;
      int32_t w_9;
      x_9 = (int32_t)(input[oh * 11 + ow]);
      w_9 = (int32_t)(weights[(1 + 8 * oco) * 9]);
      sum_1 += x_9 * w_9;
      int32_t x_10;
      int32_t w_10;
      x_10 = (int32_t)(input[oh * 11 + (1 + ow)]);
      w_10 = (int32_t)(weights[(1 + 8 * oco) * 9 + 1]);
      sum_1 += x_10 * w_10;
      int32_t x_11;
      int32_t w_11;
      x_11 = (int32_t)(input[oh * 11 + (2 + ow)]);
      w_11 = (int32_t)(weights[(1 + 8 * oco) * 9 + 2]);
      sum_1 += x_11 * w_11;
      int32_t x_12;
      int32_t w_12;
      x_12 = (int32_t)(input[(1 + oh) * 11 + ow]);
      w_12 = (int32_t)(weights[(1 + 8 * oco) * 9 + 3]);
      sum_1 += x_12 * w_12;
      int32_t x_13;
      int32_t w_13;
      x_13 = (int32_t)(input[(1 + oh) * 11 + (1 + ow)]);
      w_13 = (int32_t)(weights[(1 + 8 * oco) * 9 + 3 + 1]);
      sum_1 += x_13 * w_13;
      int32_t x_14;
      int32_t w_14;
      x_14 = (int32_t)(input[(1 + oh) * 11 + (2 + ow)]);
      w_14 = (int32_t)(weights[(1 + 8 * oco) * 9 + 3 + 2]);
      sum_1 += x_14 * w_14;
      int32_t x_15;
      int32_t w_15;
      x_15 = (int32_t)(input[(2 + oh) * 11 + ow]);
      w_15 = (int32_t)(weights[(1 + 8 * oco) * 9 + 6]);
      sum_1 += x_15 * w_15;
      int32_t x_16;
      int32_t w_16;
      x_16 = (int32_t)(input[(2 + oh) * 11 + (1 + ow)]);
      w_16 = (int32_t)(weights[(1 + 8 * oco) * 9 + 6 + 1]);
      sum_1 += x_16 * w_16;
      int32_t x_17;
      int32_t w_17;
      x_17 = (int32_t)(input[(2 + oh) * 11 + (2 + ow)]);
      w_17 = (int32_t)(weights[(1 + 8 * oco) * 9 + 6 + 2]);
      sum_1 += x_17 * w_17;
      float val_1;
      val_1 = (float)(sum_1);
      val_1 = val_1 * *acc_scale;
      val_1 = val_1 + _select_float((float)val_1, (float)0.0f, (float)-0.5f, (float)0.5f);
      val_1 = fmaxf((float)(-128.0f), (float)(_select_float((float)val_1, (float)127.0f, (float)val_1, (float)127.0f)));
      output[oh * 576 + ow * 64 + (1 + 8 * oco)] = (int8_t)(val_1);
      int32_t sum_2;
      sum_2 = bias[2 + 8 * oco];
      int32_t x_18;
      int32_t w_18;
      x_18 = (int32_t)(input[oh * 11 + ow]);
      w_18 = (int32_t)(weights[(2 + 8 * oco) * 9]);
      sum_2 += x_18 * w_18;
      int32_t x_19;
      int32_t w_19;
      x_19 = (int32_t)(input[oh * 11 + (1 + ow)]);
      w_19 = (int32_t)(weights[(2 + 8 * oco) * 9 + 1]);
      sum_2 += x_19 * w_19;
      int32_t x_20;
      int32_t w_20;
      x_20 = (int32_t)(input[oh * 11 + (2 + ow)]);
      w_20 = (int32_t)(weights[(2 + 8 * oco) * 9 + 2]);
      sum_2 += x_20 * w_20;
      int32_t x_21;
      int32_t w_21;
      x_21 = (int32_t)(input[(1 + oh) * 11 + ow]);
      w_21 = (int32_t)(weights[(2 + 8 * oco) * 9 + 3]);
      sum_2 += x_21 * w_21;
      int32_t x_22;
      int32_t w_22;
      x_22 = (int32_t)(input[(1 + oh) * 11 + (1 + ow)]);
      w_22 = (int32_t)(weights[(2 + 8 * oco) * 9 + 3 + 1]);
      sum_2 += x_22 * w_22;
      int32_t x_23;
      int32_t w_23;
      x_23 = (int32_t)(input[(1 + oh) * 11 + (2 + ow)]);
      w_23 = (int32_t)(weights[(2 + 8 * oco) * 9 + 3 + 2]);
      sum_2 += x_23 * w_23;
      int32_t x_24;
      int32_t w_24;
      x_24 = (int32_t)(input[(2 + oh) * 11 + ow]);
      w_24 = (int32_t)(weights[(2 + 8 * oco) * 9 + 6]);
      sum_2 += x_24 * w_24;
      int32_t x_25;
      int32_t w_25;
      x_25 = (int32_t)(input[(2 + oh) * 11 + (1 + ow)]);
      w_25 = (int32_t)(weights[(2 + 8 * oco) * 9 + 6 + 1]);
      sum_2 += x_25 * w_25;
      int32_t x_26;
      int32_t w_26;
      x_26 = (int32_t)(input[(2 + oh) * 11 + (2 + ow)]);
      w_26 = (int32_t)(weights[(2 + 8 * oco) * 9 + 6 + 2]);
      sum_2 += x_26 * w_26;
      float val_2;
      val_2 = (float)(sum_2);
      val_2 = val_2 * *acc_scale;
      val_2 = val_2 + _select_float((float)val_2, (float)0.0f, (float)-0.5f, (float)0.5f);
      val_2 = fmaxf((float)(-128.0f), (float)(_select_float((float)val_2, (float)127.0f, (float)val_2, (float)127.0f)));
      output[oh * 576 + ow * 64 + (2 + 8 * oco)] = (int8_t)(val_2);
      int32_t sum_3;
      sum_3 = bias[3 + 8 * oco];
      int32_t x_27;
      int32_t w_27;
      x_27 = (int32_t)(input[oh * 11 + ow]);
      w_27 = (int32_t)(weights[(3 + 8 * oco) * 9]);
      sum_3 += x_27 * w_27;
      int32_t x_28;
      int32_t w_28;
      x_28 = (int32_t)(input[oh * 11 + (1 + ow)]);
      w_28 = (int32_t)(weights[(3 + 8 * oco) * 9 + 1]);
      sum_3 += x_28 * w_28;
      int32_t x_29;
      int32_t w_29;
      x_29 = (int32_t)(input[oh * 11 + (2 + ow)]);
      w_29 = (int32_t)(weights[(3 + 8 * oco) * 9 + 2]);
      sum_3 += x_29 * w_29;
      int32_t x_30;
      int32_t w_30;
      x_30 = (int32_t)(input[(1 + oh) * 11 + ow]);
      w_30 = (int32_t)(weights[(3 + 8 * oco) * 9 + 3]);
      sum_3 += x_30 * w_30;
      int32_t x_31;
      int32_t w_31;
      x_31 = (int32_t)(input[(1 + oh) * 11 + (1 + ow)]);
      w_31 = (int32_t)(weights[(3 + 8 * oco) * 9 + 3 + 1]);
      sum_3 += x_31 * w_31;
      int32_t x_32;
      int32_t w_32;
      x_32 = (int32_t)(input[(1 + oh) * 11 + (2 + ow)]);
      w_32 = (int32_t)(weights[(3 + 8 * oco) * 9 + 3 + 2]);
      sum_3 += x_32 * w_32;
      int32_t x_33;
      int32_t w_33;
      x_33 = (int32_t)(input[(2 + oh) * 11 + ow]);
      w_33 = (int32_t)(weights[(3 + 8 * oco) * 9 + 6]);
      sum_3 += x_33 * w_33;
      int32_t x_34;
      int32_t w_34;
      x_34 = (int32_t)(input[(2 + oh) * 11 + (1 + ow)]);
      w_34 = (int32_t)(weights[(3 + 8 * oco) * 9 + 6 + 1]);
      sum_3 += x_34 * w_34;
      int32_t x_35;
      int32_t w_35;
      x_35 = (int32_t)(input[(2 + oh) * 11 + (2 + ow)]);
      w_35 = (int32_t)(weights[(3 + 8 * oco) * 9 + 6 + 2]);
      sum_3 += x_35 * w_35;
      float val_3;
      val_3 = (float)(sum_3);
      val_3 = val_3 * *acc_scale;
      val_3 = val_3 + _select_float((float)val_3, (float)0.0f, (float)-0.5f, (float)0.5f);
      val_3 = fmaxf((float)(-128.0f), (float)(_select_float((float)val_3, (float)127.0f, (float)val_3, (float)127.0f)));
      output[oh * 576 + ow * 64 + (3 + 8 * oco)] = (int8_t)(val_3);
      int32_t sum_4;
      sum_4 = bias[4 + 8 * oco];
      int32_t x_36;
      int32_t w_36;
      x_36 = (int32_t)(input[oh * 11 + ow]);
      w_36 = (int32_t)(weights[(4 + 8 * oco) * 9]);
      sum_4 += x_36 * w_36;
      int32_t x_37;
      int32_t w_37;
      x_37 = (int32_t)(input[oh * 11 + (1 + ow)]);
      w_37 = (int32_t)(weights[(4 + 8 * oco) * 9 + 1]);
      sum_4 += x_37 * w_37;
      int32_t x_38;
      int32_t w_38;
      x_38 = (int32_t)(input[oh * 11 + (2 + ow)]);
      w_38 = (int32_t)(weights[(4 + 8 * oco) * 9 + 2]);
      sum_4 += x_38 * w_38;
      int32_t x_39;
      int32_t w_39;
      x_39 = (int32_t)(input[(1 + oh) * 11 + ow]);
      w_39 = (int32_t)(weights[(4 + 8 * oco) * 9 + 3]);
      sum_4 += x_39 * w_39;
      int32_t x_40;
      int32_t w_40;
      x_40 = (int32_t)(input[(1 + oh) * 11 + (1 + ow)]);
      w_40 = (int32_t)(weights[(4 + 8 * oco) * 9 + 3 + 1]);
      sum_4 += x_40 * w_40;
      int32_t x_41;
      int32_t w_41;
      x_41 = (int32_t)(input[(1 + oh) * 11 + (2 + ow)]);
      w_41 = (int32_t)(weights[(4 + 8 * oco) * 9 + 3 + 2]);
      sum_4 += x_41 * w_41;
      int32_t x_42;
      int32_t w_42;
      x_42 = (int32_t)(input[(2 + oh) * 11 + ow]);
      w_42 = (int32_t)(weights[(4 + 8 * oco) * 9 + 6]);
      sum_4 += x_42 * w_42;
      int32_t x_43;
      int32_t w_43;
      x_43 = (int32_t)(input[(2 + oh) * 11 + (1 + ow)]);
      w_43 = (int32_t)(weights[(4 + 8 * oco) * 9 + 6 + 1]);
      sum_4 += x_43 * w_43;
      int32_t x_44;
      int32_t w_44;
      x_44 = (int32_t)(input[(2 + oh) * 11 + (2 + ow)]);
      w_44 = (int32_t)(weights[(4 + 8 * oco) * 9 + 6 + 2]);
      sum_4 += x_44 * w_44;
      float val_4;
      val_4 = (float)(sum_4);
      val_4 = val_4 * *acc_scale;
      val_4 = val_4 + _select_float((float)val_4, (float)0.0f, (float)-0.5f, (float)0.5f);
      val_4 = fmaxf((float)(-128.0f), (float)(_select_float((float)val_4, (float)127.0f, (float)val_4, (float)127.0f)));
      output[oh * 576 + ow * 64 + (4 + 8 * oco)] = (int8_t)(val_4);
      int32_t sum_5;
      sum_5 = bias[5 + 8 * oco];
      int32_t x_45;
      int32_t w_45;
      x_45 = (int32_t)(input[oh * 11 + ow]);
      w_45 = (int32_t)(weights[(5 + 8 * oco) * 9]);
      sum_5 += x_45 * w_45;
      int32_t x_46;
      int32_t w_46;
      x_46 = (int32_t)(input[oh * 11 + (1 + ow)]);
      w_46 = (int32_t)(weights[(5 + 8 * oco) * 9 + 1]);
      sum_5 += x_46 * w_46;
      int32_t x_47;
      int32_t w_47;
      x_47 = (int32_t)(input[oh * 11 + (2 + ow)]);
      w_47 = (int32_t)(weights[(5 + 8 * oco) * 9 + 2]);
      sum_5 += x_47 * w_47;
      int32_t x_48;
      int32_t w_48;
      x_48 = (int32_t)(input[(1 + oh) * 11 + ow]);
      w_48 = (int32_t)(weights[(5 + 8 * oco) * 9 + 3]);
      sum_5 += x_48 * w_48;
      int32_t x_49;
      int32_t w_49;
      x_49 = (int32_t)(input[(1 + oh) * 11 + (1 + ow)]);
      w_49 = (int32_t)(weights[(5 + 8 * oco) * 9 + 3 + 1]);
      sum_5 += x_49 * w_49;
      int32_t x_50;
      int32_t w_50;
      x_50 = (int32_t)(input[(1 + oh) * 11 + (2 + ow)]);
      w_50 = (int32_t)(weights[(5 + 8 * oco) * 9 + 3 + 2]);
      sum_5 += x_50 * w_50;
      int32_t x_51;
      int32_t w_51;
      x_51 = (int32_t)(input[(2 + oh) * 11 + ow]);
      w_51 = (int32_t)(weights[(5 + 8 * oco) * 9 + 6]);
      sum_5 += x_51 * w_51;
      int32_t x_52;
      int32_t w_52;
      x_52 = (int32_t)(input[(2 + oh) * 11 + (1 + ow)]);
      w_52 = (int32_t)(weights[(5 + 8 * oco) * 9 + 6 + 1]);
      sum_5 += x_52 * w_52;
      int32_t x_53;
      int32_t w_53;
      x_53 = (int32_t)(input[(2 + oh) * 11 + (2 + ow)]);
      w_53 = (int32_t)(weights[(5 + 8 * oco) * 9 + 6 + 2]);
      sum_5 += x_53 * w_53;
      float val_5;
      val_5 = (float)(sum_5);
      val_5 = val_5 * *acc_scale;
      val_5 = val_5 + _select_float((float)val_5, (float)0.0f, (float)-0.5f, (float)0.5f);
      val_5 = fmaxf((float)(-128.0f), (float)(_select_float((float)val_5, (float)127.0f, (float)val_5, (float)127.0f)));
      output[oh * 576 + ow * 64 + (5 + 8 * oco)] = (int8_t)(val_5);
      int32_t sum_6;
      sum_6 = bias[6 + 8 * oco];
      int32_t x_54;
      int32_t w_54;
      x_54 = (int32_t)(input[oh * 11 + ow]);
      w_54 = (int32_t)(weights[(6 + 8 * oco) * 9]);
      sum_6 += x_54 * w_54;
      int32_t x_55;
      int32_t w_55;
      x_55 = (int32_t)(input[oh * 11 + (1 + ow)]);
      w_55 = (int32_t)(weights[(6 + 8 * oco) * 9 + 1]);
      sum_6 += x_55 * w_55;
      int32_t x_56;
      int32_t w_56;
      x_56 = (int32_t)(input[oh * 11 + (2 + ow)]);
      w_56 = (int32_t)(weights[(6 + 8 * oco) * 9 + 2]);
      sum_6 += x_56 * w_56;
      int32_t x_57;
      int32_t w_57;
      x_57 = (int32_t)(input[(1 + oh) * 11 + ow]);
      w_57 = (int32_t)(weights[(6 + 8 * oco) * 9 + 3]);
      sum_6 += x_57 * w_57;
      int32_t x_58;
      int32_t w_58;
      x_58 = (int32_t)(input[(1 + oh) * 11 + (1 + ow)]);
      w_58 = (int32_t)(weights[(6 + 8 * oco) * 9 + 3 + 1]);
      sum_6 += x_58 * w_58;
      int32_t x_59;
      int32_t w_59;
      x_59 = (int32_t)(input[(1 + oh) * 11 + (2 + ow)]);
      w_59 = (int32_t)(weights[(6 + 8 * oco) * 9 + 3 + 2]);
      sum_6 += x_59 * w_59;
      int32_t x_60;
      int32_t w_60;
      x_60 = (int32_t)(input[(2 + oh) * 11 + ow]);
      w_60 = (int32_t)(weights[(6 + 8 * oco) * 9 + 6]);
      sum_6 += x_60 * w_60;
      int32_t x_61;
      int32_t w_61;
      x_61 = (int32_t)(input[(2 + oh) * 11 + (1 + ow)]);
      w_61 = (int32_t)(weights[(6 + 8 * oco) * 9 + 6 + 1]);
      sum_6 += x_61 * w_61;
      int32_t x_62;
      int32_t w_62;
      x_62 = (int32_t)(input[(2 + oh) * 11 + (2 + ow)]);
      w_62 = (int32_t)(weights[(6 + 8 * oco) * 9 + 6 + 2]);
      sum_6 += x_62 * w_62;
      float val_6;
      val_6 = (float)(sum_6);
      val_6 = val_6 * *acc_scale;
      val_6 = val_6 + _select_float((float)val_6, (float)0.0f, (float)-0.5f, (float)0.5f);
      val_6 = fmaxf((float)(-128.0f), (float)(_select_float((float)val_6, (float)127.0f, (float)val_6, (float)127.0f)));
      output[oh * 576 + ow * 64 + (6 + 8 * oco)] = (int8_t)(val_6);
      int32_t sum_7;
      sum_7 = bias[7 + 8 * oco];
      int32_t x_63;
      int32_t w_63;
      x_63 = (int32_t)(input[oh * 11 + ow]);
      w_63 = (int32_t)(weights[(7 + 8 * oco) * 9]);
      sum_7 += x_63 * w_63;
      int32_t x_64;
      int32_t w_64;
      x_64 = (int32_t)(input[oh * 11 + (1 + ow)]);
      w_64 = (int32_t)(weights[(7 + 8 * oco) * 9 + 1]);
      sum_7 += x_64 * w_64;
      int32_t x_65;
      int32_t w_65;
      x_65 = (int32_t)(input[oh * 11 + (2 + ow)]);
      w_65 = (int32_t)(weights[(7 + 8 * oco) * 9 + 2]);
      sum_7 += x_65 * w_65;
      int32_t x_66;
      int32_t w_66;
      x_66 = (int32_t)(input[(1 + oh) * 11 + ow]);
      w_66 = (int32_t)(weights[(7 + 8 * oco) * 9 + 3]);
      sum_7 += x_66 * w_66;
      int32_t x_67;
      int32_t w_67;
      x_67 = (int32_t)(input[(1 + oh) * 11 + (1 + ow)]);
      w_67 = (int32_t)(weights[(7 + 8 * oco) * 9 + 3 + 1]);
      sum_7 += x_67 * w_67;
      int32_t x_68;
      int32_t w_68;
      x_68 = (int32_t)(input[(1 + oh) * 11 + (2 + ow)]);
      w_68 = (int32_t)(weights[(7 + 8 * oco) * 9 + 3 + 2]);
      sum_7 += x_68 * w_68;
      int32_t x_69;
      int32_t w_69;
      x_69 = (int32_t)(input[(2 + oh) * 11 + ow]);
      w_69 = (int32_t)(weights[(7 + 8 * oco) * 9 + 6]);
      sum_7 += x_69 * w_69;
      int32_t x_70;
      int32_t w_70;
      x_70 = (int32_t)(input[(2 + oh) * 11 + (1 + ow)]);
      w_70 = (int32_t)(weights[(7 + 8 * oco) * 9 + 6 + 1]);
      sum_7 += x_70 * w_70;
      int32_t x_71;
      int32_t w_71;
      x_71 = (int32_t)(input[(2 + oh) * 11 + (2 + ow)]);
      w_71 = (int32_t)(weights[(7 + 8 * oco) * 9 + 6 + 2]);
      sum_7 += x_71 * w_71;
      float val_7;
      val_7 = (float)(sum_7);
      val_7 = val_7 * *acc_scale;
      val_7 = val_7 + _select_float((float)val_7, (float)0.0f, (float)-0.5f, (float)0.5f);
      val_7 = fmaxf((float)(-128.0f), (float)(_select_float((float)val_7, (float)127.0f, (float)val_7, (float)127.0f)));
      output[oh * 576 + ow * 64 + (7 + 8 * oco)] = (int8_t)(val_7);
    }
  }
}
}

// conv2_gemmini(
//     input : i8[9, 9, 64] @DRAM,
//     weights : i8[32, 3, 3, 64] @DRAM,
//     bias : i32[32] @DRAM,
//     output : i8[7, 7, 32] @DRAM,
//     acc_scale : f32 @DRAM
// )
static void conv2_gemmini( void *ctxt, const int8_t* input, const int8_t* weights, const int32_t* bias, int8_t* output, const float* acc_scale ) {
static int8_t weights_copy[3 * 3 * 64 * 32];
for (int_fast32_t i0 = 0; i0 < 32; i0++) {
  for (int_fast32_t i3o = 0; i3o < 8; i3o++) {
    weights_copy[8 * i3o * 32 + i0] = weights[i0 * 576 + 8 * i3o];
    weights_copy[(1 + 8 * i3o) * 32 + i0] = weights[i0 * 576 + (1 + 8 * i3o)];
    weights_copy[(2 + 8 * i3o) * 32 + i0] = weights[i0 * 576 + (2 + 8 * i3o)];
    weights_copy[(3 + 8 * i3o) * 32 + i0] = weights[i0 * 576 + (3 + 8 * i3o)];
    weights_copy[(4 + 8 * i3o) * 32 + i0] = weights[i0 * 576 + (4 + 8 * i3o)];
    weights_copy[(5 + 8 * i3o) * 32 + i0] = weights[i0 * 576 + (5 + 8 * i3o)];
    weights_copy[(6 + 8 * i3o) * 32 + i0] = weights[i0 * 576 + (6 + 8 * i3o)];
    weights_copy[(7 + 8 * i3o) * 32 + i0] = weights[i0 * 576 + (7 + 8 * i3o)];
  }
  for (int_fast32_t i3o = 0; i3o < 8; i3o++) {
    weights_copy[2048 + 8 * i3o * 32 + i0] = weights[i0 * 576 + 64 + 8 * i3o];
    weights_copy[2048 + (1 + 8 * i3o) * 32 + i0] = weights[i0 * 576 + 64 + (1 + 8 * i3o)];
    weights_copy[2048 + (2 + 8 * i3o) * 32 + i0] = weights[i0 * 576 + 64 + (2 + 8 * i3o)];
    weights_copy[2048 + (3 + 8 * i3o) * 32 + i0] = weights[i0 * 576 + 64 + (3 + 8 * i3o)];
    weights_copy[2048 + (4 + 8 * i3o) * 32 + i0] = weights[i0 * 576 + 64 + (4 + 8 * i3o)];
    weights_copy[2048 + (5 + 8 * i3o) * 32 + i0] = weights[i0 * 576 + 64 + (5 + 8 * i3o)];
    weights_copy[2048 + (6 + 8 * i3o) * 32 + i0] = weights[i0 * 576 + 64 + (6 + 8 * i3o)];
    weights_copy[2048 + (7 + 8 * i3o) * 32 + i0] = weights[i0 * 576 + 64 + (7 + 8 * i3o)];
  }
  for (int_fast32_t i3o = 0; i3o < 8; i3o++) {
    weights_copy[4096 + 8 * i3o * 32 + i0] = weights[i0 * 576 + 128 + 8 * i3o];
    weights_copy[4096 + (1 + 8 * i3o) * 32 + i0] = weights[i0 * 576 + 128 + (1 + 8 * i3o)];
    weights_copy[4096 + (2 + 8 * i3o) * 32 + i0] = weights[i0 * 576 + 128 + (2 + 8 * i3o)];
    weights_copy[4096 + (3 + 8 * i3o) * 32 + i0] = weights[i0 * 576 + 128 + (3 + 8 * i3o)];
    weights_copy[4096 + (4 + 8 * i3o) * 32 + i0] = weights[i0 * 576 + 128 + (4 + 8 * i3o)];
    weights_copy[4096 + (5 + 8 * i3o) * 32 + i0] = weights[i0 * 576 + 128 + (5 + 8 * i3o)];
    weights_copy[4096 + (6 + 8 * i3o) * 32 + i0] = weights[i0 * 576 + 128 + (6 + 8 * i3o)];
    weights_copy[4096 + (7 + 8 * i3o) * 32 + i0] = weights[i0 * 576 + 128 + (7 + 8 * i3o)];
  }
  for (int_fast32_t i3o = 0; i3o < 8; i3o++) {
    weights_copy[6144 + 8 * i3o * 32 + i0] = weights[i0 * 576 + 192 + 8 * i3o];
    weights_copy[6144 + (1 + 8 * i3o) * 32 + i0] = weights[i0 * 576 + 192 + (1 + 8 * i3o)];
    weights_copy[6144 + (2 + 8 * i3o) * 32 + i0] = weights[i0 * 576 + 192 + (2 + 8 * i3o)];
    weights_copy[6144 + (3 + 8 * i3o) * 32 + i0] = weights[i0 * 576 + 192 + (3 + 8 * i3o)];
    weights_copy[6144 + (4 + 8 * i3o) * 32 + i0] = weights[i0 * 576 + 192 + (4 + 8 * i3o)];
    weights_copy[6144 + (5 + 8 * i3o) * 32 + i0] = weights[i0 * 576 + 192 + (5 + 8 * i3o)];
    weights_copy[6144 + (6 + 8 * i3o) * 32 + i0] = weights[i0 * 576 + 192 + (6 + 8 * i3o)];
    weights_copy[6144 + (7 + 8 * i3o) * 32 + i0] = weights[i0 * 576 + 192 + (7 + 8 * i3o)];
  }
  for (int_fast32_t i3o = 0; i3o < 8; i3o++) {
    weights_copy[8192 + 8 * i3o * 32 + i0] = weights[i0 * 576 + 192 + 64 + 8 * i3o];
    weights_copy[8192 + (1 + 8 * i3o) * 32 + i0] = weights[i0 * 576 + 192 + 64 + (1 + 8 * i3o)];
    weights_copy[8192 + (2 + 8 * i3o) * 32 + i0] = weights[i0 * 576 + 192 + 64 + (2 + 8 * i3o)];
    weights_copy[8192 + (3 + 8 * i3o) * 32 + i0] = weights[i0 * 576 + 192 + 64 + (3 + 8 * i3o)];
    weights_copy[8192 + (4 + 8 * i3o) * 32 + i0] = weights[i0 * 576 + 192 + 64 + (4 + 8 * i3o)];
    weights_copy[8192 + (5 + 8 * i3o) * 32 + i0] = weights[i0 * 576 + 192 + 64 + (5 + 8 * i3o)];
    weights_copy[8192 + (6 + 8 * i3o) * 32 + i0] = weights[i0 * 576 + 192 + 64 + (6 + 8 * i3o)];
    weights_copy[8192 + (7 + 8 * i3o) * 32 + i0] = weights[i0 * 576 + 192 + 64 + (7 + 8 * i3o)];
  }
  for (int_fast32_t i3o = 0; i3o < 8; i3o++) {
    weights_copy[10240 + 8 * i3o * 32 + i0] = weights[i0 * 576 + 192 + 128 + 8 * i3o];
    weights_copy[10240 + (1 + 8 * i3o) * 32 + i0] = weights[i0 * 576 + 192 + 128 + (1 + 8 * i3o)];
    weights_copy[10240 + (2 + 8 * i3o) * 32 + i0] = weights[i0 * 576 + 192 + 128 + (2 + 8 * i3o)];
    weights_copy[10240 + (3 + 8 * i3o) * 32 + i0] = weights[i0 * 576 + 192 + 128 + (3 + 8 * i3o)];
    weights_copy[10240 + (4 + 8 * i3o) * 32 + i0] = weights[i0 * 576 + 192 + 128 + (4 + 8 * i3o)];
    weights_copy[10240 + (5 + 8 * i3o) * 32 + i0] = weights[i0 * 576 + 192 + 128 + (5 + 8 * i3o)];
    weights_copy[10240 + (6 + 8 * i3o) * 32 + i0] = weights[i0 * 576 + 192 + 128 + (6 + 8 * i3o)];
    weights_copy[10240 + (7 + 8 * i3o) * 32 + i0] = weights[i0 * 576 + 192 + 128 + (7 + 8 * i3o)];
  }
  for (int_fast32_t i3o = 0; i3o < 8; i3o++) {
    weights_copy[12288 + 8 * i3o * 32 + i0] = weights[i0 * 576 + 384 + 8 * i3o];
    weights_copy[12288 + (1 + 8 * i3o) * 32 + i0] = weights[i0 * 576 + 384 + (1 + 8 * i3o)];
    weights_copy[12288 + (2 + 8 * i3o) * 32 + i0] = weights[i0 * 576 + 384 + (2 + 8 * i3o)];
    weights_copy[12288 + (3 + 8 * i3o) * 32 + i0] = weights[i0 * 576 + 384 + (3 + 8 * i3o)];
    weights_copy[12288 + (4 + 8 * i3o) * 32 + i0] = weights[i0 * 576 + 384 + (4 + 8 * i3o)];
    weights_copy[12288 + (5 + 8 * i3o) * 32 + i0] = weights[i0 * 576 + 384 + (5 + 8 * i3o)];
    weights_copy[12288 + (6 + 8 * i3o) * 32 + i0] = weights[i0 * 576 + 384 + (6 + 8 * i3o)];
    weights_copy[12288 + (7 + 8 * i3o) * 32 + i0] = weights[i0 * 576 + 384 + (7 + 8 * i3o)];
  }
  for (int_fast32_t i3o = 0; i3o < 8; i3o++) {
    weights_copy[14336 + 8 * i3o * 32 + i0] = weights[i0 * 576 + 384 + 64 + 8 * i3o];
    weights_copy[14336 + (1 + 8 * i3o) * 32 + i0] = weights[i0 * 576 + 384 + 64 + (1 + 8 * i3o)];
    weights_copy[14336 + (2 + 8 * i3o) * 32 + i0] = weights[i0 * 576 + 384 + 64 + (2 + 8 * i3o)];
    weights_copy[14336 + (3 + 8 * i3o) * 32 + i0] = weights[i0 * 576 + 384 + 64 + (3 + 8 * i3o)];
    weights_copy[14336 + (4 + 8 * i3o) * 32 + i0] = weights[i0 * 576 + 384 + 64 + (4 + 8 * i3o)];
    weights_copy[14336 + (5 + 8 * i3o) * 32 + i0] = weights[i0 * 576 + 384 + 64 + (5 + 8 * i3o)];
    weights_copy[14336 + (6 + 8 * i3o) * 32 + i0] = weights[i0 * 576 + 384 + 64 + (6 + 8 * i3o)];
    weights_copy[14336 + (7 + 8 * i3o) * 32 + i0] = weights[i0 * 576 + 384 + 64 + (7 + 8 * i3o)];
  }
  for (int_fast32_t i3o = 0; i3o < 8; i3o++) {
    weights_copy[16384 + 8 * i3o * 32 + i0] = weights[i0 * 576 + 384 + 128 + 8 * i3o];
    weights_copy[16384 + (1 + 8 * i3o) * 32 + i0] = weights[i0 * 576 + 384 + 128 + (1 + 8 * i3o)];
    weights_copy[16384 + (2 + 8 * i3o) * 32 + i0] = weights[i0 * 576 + 384 + 128 + (2 + 8 * i3o)];
    weights_copy[16384 + (3 + 8 * i3o) * 32 + i0] = weights[i0 * 576 + 384 + 128 + (3 + 8 * i3o)];
    weights_copy[16384 + (4 + 8 * i3o) * 32 + i0] = weights[i0 * 576 + 384 + 128 + (4 + 8 * i3o)];
    weights_copy[16384 + (5 + 8 * i3o) * 32 + i0] = weights[i0 * 576 + 384 + 128 + (5 + 8 * i3o)];
    weights_copy[16384 + (6 + 8 * i3o) * 32 + i0] = weights[i0 * 576 + 384 + 128 + (6 + 8 * i3o)];
    weights_copy[16384 + (7 + 8 * i3o) * 32 + i0] = weights[i0 * 576 + 384 + 128 + (7 + 8 * i3o)];
  }
}
int8_t *as_ = (int8_t*) ((uint64_t)gemm_malloc (16 * 7 * sizeof(int8_t)));
int8_t *bs = (int8_t*) ((uint64_t)gemm_malloc (16 * 16 * sizeof(int8_t)));
int32_t *acc = (int32_t*) ((uint32_t)gemm_acc_malloc (16 * 7 * sizeof(int32_t)));
static int32_t sum[7 * 16];
for (int_fast32_t oh = 0; oh < 7; oh++) {
  for (int_fast32_t oco = 0; oco < 2; oco++) {
    for (int_fast32_t ow = 0; ow < 7; ow++) {
      for (int_fast32_t ocio = 0; ocio < 2; ocio++) {
        sum[ow * 16 + 8 * ocio] = bias[8 * ocio + 16 * oco];
        sum[ow * 16 + (1 + 8 * ocio)] = bias[1 + 8 * ocio + 16 * oco];
        sum[ow * 16 + (2 + 8 * ocio)] = bias[2 + 8 * ocio + 16 * oco];
        sum[ow * 16 + (3 + 8 * ocio)] = bias[3 + 8 * ocio + 16 * oco];
        sum[ow * 16 + (4 + 8 * ocio)] = bias[4 + 8 * ocio + 16 * oco];
        sum[ow * 16 + (5 + 8 * ocio)] = bias[5 + 8 * ocio + 16 * oco];
        sum[ow * 16 + (6 + 8 * ocio)] = bias[6 + 8 * ocio + 16 * oco];
        sum[ow * 16 + (7 + 8 * ocio)] = bias[7 + 8 * ocio + 16 * oco];
      }
    }
    gemmini_extended3_config_ld(((struct exo_win_2i32c){ &sum[0], { 16, 1 } }).strides[0]*4, 1.0f, 0, 0);
gemmini_extended_mvin( ((uint64_t) &sum[0]), ((uint32_t) &*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)acc)) + (0)/16))), (16), (7) );
    for (int_fast32_t kh = 0; kh < 3; kh++) {
      for (int_fast32_t kw = 0; kw < 3; kw++) {
        for (int_fast32_t ico = 0; ico < 4; ico++) {
          gemmini_extended3_config_ld(((struct exo_win_2i8c){ &input[(kh + oh) * (576) + (kw) * (64) + 16 * ico], { 64, 1 } }).strides[0]*1, 1.0f, 0, 1);
gemmini_extended_mvin2( &input[(kh + oh) * (576) + (kw) * (64) + 16 * ico], ((uint64_t) &*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)as_)) + (0)/16))), (16), (7) );
          gemmini_extended3_config_ld(((struct exo_win_2i8c){ &weights_copy[(kh) * (6144) + (kw) * (2048) + (16 * ico) * (32) + 16 * oco], { 32, 1 } }).strides[0]*1, 1.0f, 0, 2);
gemmini_extended_mvin3( &weights_copy[(kh) * (6144) + (kw) * (2048) + (16 * ico) * (32) + 16 * oco], ((uint64_t) &*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)bs)) + (0)/16))), (16), (16) );
          gemmini_extended_config_ex(WS, 0, 0, 1, 0, 0);
gemmini_extended_preload((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)bs)) + (0)/16))), (uint32_t)(&*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)acc)) + (0)/16))) | 0x40000000, (16), (16), (16), (7));
gemmini_extended_compute_preloaded((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)as_)) + (0)/16))), ~((uint32_t)0), (16), (7), 16, 16);
        }
      }
    }
    gemmini_extended_config_st(((struct exo_win_2i32){ &sum[0], { 16, 1 } }).strides[0]*4, 0, 1.0f);
gemmini_extended_mvout( ((uint64_t) &sum[0]), ((uint32_t) &*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)acc)) + (0)/16)) | 0x20000000), (16), (7) );
    for (int_fast32_t ow = 0; ow < 7; ow++) {
      for (int_fast32_t ocio = 0; ocio < 2; ocio++) {
        float val;
        val = (float)(sum[ow * 16 + 8 * ocio]);
        val = val * *acc_scale;
        val = val + _select_float((float)val, (float)0.0f, (float)-0.5f, (float)0.5f);
        val = fmaxf((float)(-128.0f), (float)(_select_float((float)val, (float)127.0f, (float)val, (float)127.0f)));
        output[oh * 224 + ow * 32 + (8 * ocio + 16 * oco)] = (int8_t)(val);
        float val_1;
        val_1 = (float)(sum[ow * 16 + (1 + 8 * ocio)]);
        val_1 = val_1 * *acc_scale;
        val_1 = val_1 + _select_float((float)val_1, (float)0.0f, (float)-0.5f, (float)0.5f);
        val_1 = fmaxf((float)(-128.0f), (float)(_select_float((float)val_1, (float)127.0f, (float)val_1, (float)127.0f)));
        output[oh * 224 + ow * 32 + (1 + 8 * ocio + 16 * oco)] = (int8_t)(val_1);
        float val_2;
        val_2 = (float)(sum[ow * 16 + (2 + 8 * ocio)]);
        val_2 = val_2 * *acc_scale;
        val_2 = val_2 + _select_float((float)val_2, (float)0.0f, (float)-0.5f, (float)0.5f);
        val_2 = fmaxf((float)(-128.0f), (float)(_select_float((float)val_2, (float)127.0f, (float)val_2, (float)127.0f)));
        output[oh * 224 + ow * 32 + (2 + 8 * ocio + 16 * oco)] = (int8_t)(val_2);
        float val_3;
        val_3 = (float)(sum[ow * 16 + (3 + 8 * ocio)]);
        val_3 = val_3 * *acc_scale;
        val_3 = val_3 + _select_float((float)val_3, (float)0.0f, (float)-0.5f, (float)0.5f);
        val_3 = fmaxf((float)(-128.0f), (float)(_select_float((float)val_3, (float)127.0f, (float)val_3, (float)127.0f)));
        output[oh * 224 + ow * 32 + (3 + 8 * ocio + 16 * oco)] = (int8_t)(val_3);
        float val_4;
        val_4 = (float)(sum[ow * 16 + (4 + 8 * ocio)]);
        val_4 = val_4 * *acc_scale;
        val_4 = val_4 + _select_float((float)val_4, (float)0.0f, (float)-0.5f, (float)0.5f);
        val_4 = fmaxf((float)(-128.0f), (float)(_select_float((float)val_4, (float)127.0f, (float)val_4, (float)127.0f)));
        output[oh * 224 + ow * 32 + (4 + 8 * ocio + 16 * oco)] = (int8_t)(val_4);
        float val_5;
        val_5 = (float)(sum[ow * 16 + (5 + 8 * ocio)]);
        val_5 = val_5 * *acc_scale;
        val_5 = val_5 + _select_float((float)val_5, (float)0.0f, (float)-0.5f, (float)0.5f);
        val_5 = fmaxf((float)(-128.0f), (float)(_select_float((float)val_5, (float)127.0f, (float)val_5, (float)127.0f)));
        output[oh * 224 + ow * 32 + (5 + 8 * ocio + 16 * oco)] = (int8_t)(val_5);
        float val_6;
        val_6 = (float)(sum[ow * 16 + (6 + 8 * ocio)]);
        val_6 = val_6 * *acc_scale;
        val_6 = val_6 + _select_float((float)val_6, (float)0.0f, (float)-0.5f, (float)0.5f);
        val_6 = fmaxf((float)(-128.0f), (float)(_select_float((float)val_6, (float)127.0f, (float)val_6, (float)127.0f)));
        output[oh * 224 + ow * 32 + (6 + 8 * ocio + 16 * oco)] = (int8_t)(val_6);
        float val_7;
        val_7 = (float)(sum[ow * 16 + (7 + 8 * ocio)]);
        val_7 = val_7 * *acc_scale;
        val_7 = val_7 + _select_float((float)val_7, (float)0.0f, (float)-0.5f, (float)0.5f);
        val_7 = fmaxf((float)(-128.0f), (float)(_select_float((float)val_7, (float)127.0f, (float)val_7, (float)127.0f)));
        output[oh * 224 + ow * 32 + (7 + 8 * ocio + 16 * oco)] = (int8_t)(val_7);
      }
    }
  }
}
gemm_acc_free((uint32_t)(acc));
gemm_free((uint64_t)(bs));
gemm_free((uint64_t)(as_));
}

// conv3_gemmini(
//     input : i8[7, 7, 32] @DRAM,
//     weights : i8[8, 3, 3, 32] @DRAM,
//     bias : i32[8] @DRAM,
//     output : i8[5, 5, 8] @DRAM,
//     acc_scale : f32 @DRAM
// )
static void conv3_gemmini( void *ctxt, const int8_t* input, const int8_t* weights, const int32_t* bias, int8_t* output, const float* acc_scale ) {
static int8_t weights_copy[3 * 3 * 32 * 8];
for (int_fast32_t i0 = 0; i0 < 8; i0++) {
  for (int_fast32_t i3o = 0; i3o < 4; i3o++) {
    weights_copy[8 * i3o * 8 + i0] = weights[i0 * 288 + 8 * i3o];
    weights_copy[(1 + 8 * i3o) * 8 + i0] = weights[i0 * 288 + (1 + 8 * i3o)];
    weights_copy[(2 + 8 * i3o) * 8 + i0] = weights[i0 * 288 + (2 + 8 * i3o)];
    weights_copy[(3 + 8 * i3o) * 8 + i0] = weights[i0 * 288 + (3 + 8 * i3o)];
    weights_copy[(4 + 8 * i3o) * 8 + i0] = weights[i0 * 288 + (4 + 8 * i3o)];
    weights_copy[(5 + 8 * i3o) * 8 + i0] = weights[i0 * 288 + (5 + 8 * i3o)];
    weights_copy[(6 + 8 * i3o) * 8 + i0] = weights[i0 * 288 + (6 + 8 * i3o)];
    weights_copy[(7 + 8 * i3o) * 8 + i0] = weights[i0 * 288 + (7 + 8 * i3o)];
  }
  for (int_fast32_t i3o = 0; i3o < 4; i3o++) {
    weights_copy[256 + 8 * i3o * 8 + i0] = weights[i0 * 288 + 32 + 8 * i3o];
    weights_copy[256 + (1 + 8 * i3o) * 8 + i0] = weights[i0 * 288 + 32 + (1 + 8 * i3o)];
    weights_copy[256 + (2 + 8 * i3o) * 8 + i0] = weights[i0 * 288 + 32 + (2 + 8 * i3o)];
    weights_copy[256 + (3 + 8 * i3o) * 8 + i0] = weights[i0 * 288 + 32 + (3 + 8 * i3o)];
    weights_copy[256 + (4 + 8 * i3o) * 8 + i0] = weights[i0 * 288 + 32 + (4 + 8 * i3o)];
    weights_copy[256 + (5 + 8 * i3o) * 8 + i0] = weights[i0 * 288 + 32 + (5 + 8 * i3o)];
    weights_copy[256 + (6 + 8 * i3o) * 8 + i0] = weights[i0 * 288 + 32 + (6 + 8 * i3o)];
    weights_copy[256 + (7 + 8 * i3o) * 8 + i0] = weights[i0 * 288 + 32 + (7 + 8 * i3o)];
  }
  for (int_fast32_t i3o = 0; i3o < 4; i3o++) {
    weights_copy[512 + 8 * i3o * 8 + i0] = weights[i0 * 288 + 64 + 8 * i3o];
    weights_copy[512 + (1 + 8 * i3o) * 8 + i0] = weights[i0 * 288 + 64 + (1 + 8 * i3o)];
    weights_copy[512 + (2 + 8 * i3o) * 8 + i0] = weights[i0 * 288 + 64 + (2 + 8 * i3o)];
    weights_copy[512 + (3 + 8 * i3o) * 8 + i0] = weights[i0 * 288 + 64 + (3 + 8 * i3o)];
    weights_copy[512 + (4 + 8 * i3o) * 8 + i0] = weights[i0 * 288 + 64 + (4 + 8 * i3o)];
    weights_copy[512 + (5 + 8 * i3o) * 8 + i0] = weights[i0 * 288 + 64 + (5 + 8 * i3o)];
    weights_copy[512 + (6 + 8 * i3o) * 8 + i0] = weights[i0 * 288 + 64 + (6 + 8 * i3o)];
    weights_copy[512 + (7 + 8 * i3o) * 8 + i0] = weights[i0 * 288 + 64 + (7 + 8 * i3o)];
  }
  for (int_fast32_t i3o = 0; i3o < 4; i3o++) {
    weights_copy[768 + 8 * i3o * 8 + i0] = weights[i0 * 288 + 96 + 8 * i3o];
    weights_copy[768 + (1 + 8 * i3o) * 8 + i0] = weights[i0 * 288 + 96 + (1 + 8 * i3o)];
    weights_copy[768 + (2 + 8 * i3o) * 8 + i0] = weights[i0 * 288 + 96 + (2 + 8 * i3o)];
    weights_copy[768 + (3 + 8 * i3o) * 8 + i0] = weights[i0 * 288 + 96 + (3 + 8 * i3o)];
    weights_copy[768 + (4 + 8 * i3o) * 8 + i0] = weights[i0 * 288 + 96 + (4 + 8 * i3o)];
    weights_copy[768 + (5 + 8 * i3o) * 8 + i0] = weights[i0 * 288 + 96 + (5 + 8 * i3o)];
    weights_copy[768 + (6 + 8 * i3o) * 8 + i0] = weights[i0 * 288 + 96 + (6 + 8 * i3o)];
    weights_copy[768 + (7 + 8 * i3o) * 8 + i0] = weights[i0 * 288 + 96 + (7 + 8 * i3o)];
  }
  for (int_fast32_t i3o = 0; i3o < 4; i3o++) {
    weights_copy[1024 + 8 * i3o * 8 + i0] = weights[i0 * 288 + 96 + 32 + 8 * i3o];
    weights_copy[1024 + (1 + 8 * i3o) * 8 + i0] = weights[i0 * 288 + 96 + 32 + (1 + 8 * i3o)];
    weights_copy[1024 + (2 + 8 * i3o) * 8 + i0] = weights[i0 * 288 + 96 + 32 + (2 + 8 * i3o)];
    weights_copy[1024 + (3 + 8 * i3o) * 8 + i0] = weights[i0 * 288 + 96 + 32 + (3 + 8 * i3o)];
    weights_copy[1024 + (4 + 8 * i3o) * 8 + i0] = weights[i0 * 288 + 96 + 32 + (4 + 8 * i3o)];
    weights_copy[1024 + (5 + 8 * i3o) * 8 + i0] = weights[i0 * 288 + 96 + 32 + (5 + 8 * i3o)];
    weights_copy[1024 + (6 + 8 * i3o) * 8 + i0] = weights[i0 * 288 + 96 + 32 + (6 + 8 * i3o)];
    weights_copy[1024 + (7 + 8 * i3o) * 8 + i0] = weights[i0 * 288 + 96 + 32 + (7 + 8 * i3o)];
  }
  for (int_fast32_t i3o = 0; i3o < 4; i3o++) {
    weights_copy[1280 + 8 * i3o * 8 + i0] = weights[i0 * 288 + 96 + 64 + 8 * i3o];
    weights_copy[1280 + (1 + 8 * i3o) * 8 + i0] = weights[i0 * 288 + 96 + 64 + (1 + 8 * i3o)];
    weights_copy[1280 + (2 + 8 * i3o) * 8 + i0] = weights[i0 * 288 + 96 + 64 + (2 + 8 * i3o)];
    weights_copy[1280 + (3 + 8 * i3o) * 8 + i0] = weights[i0 * 288 + 96 + 64 + (3 + 8 * i3o)];
    weights_copy[1280 + (4 + 8 * i3o) * 8 + i0] = weights[i0 * 288 + 96 + 64 + (4 + 8 * i3o)];
    weights_copy[1280 + (5 + 8 * i3o) * 8 + i0] = weights[i0 * 288 + 96 + 64 + (5 + 8 * i3o)];
    weights_copy[1280 + (6 + 8 * i3o) * 8 + i0] = weights[i0 * 288 + 96 + 64 + (6 + 8 * i3o)];
    weights_copy[1280 + (7 + 8 * i3o) * 8 + i0] = weights[i0 * 288 + 96 + 64 + (7 + 8 * i3o)];
  }
  for (int_fast32_t i3o = 0; i3o < 4; i3o++) {
    weights_copy[1536 + 8 * i3o * 8 + i0] = weights[i0 * 288 + 192 + 8 * i3o];
    weights_copy[1536 + (1 + 8 * i3o) * 8 + i0] = weights[i0 * 288 + 192 + (1 + 8 * i3o)];
    weights_copy[1536 + (2 + 8 * i3o) * 8 + i0] = weights[i0 * 288 + 192 + (2 + 8 * i3o)];
    weights_copy[1536 + (3 + 8 * i3o) * 8 + i0] = weights[i0 * 288 + 192 + (3 + 8 * i3o)];
    weights_copy[1536 + (4 + 8 * i3o) * 8 + i0] = weights[i0 * 288 + 192 + (4 + 8 * i3o)];
    weights_copy[1536 + (5 + 8 * i3o) * 8 + i0] = weights[i0 * 288 + 192 + (5 + 8 * i3o)];
    weights_copy[1536 + (6 + 8 * i3o) * 8 + i0] = weights[i0 * 288 + 192 + (6 + 8 * i3o)];
    weights_copy[1536 + (7 + 8 * i3o) * 8 + i0] = weights[i0 * 288 + 192 + (7 + 8 * i3o)];
  }
  for (int_fast32_t i3o = 0; i3o < 4; i3o++) {
    weights_copy[1792 + 8 * i3o * 8 + i0] = weights[i0 * 288 + 192 + 32 + 8 * i3o];
    weights_copy[1792 + (1 + 8 * i3o) * 8 + i0] = weights[i0 * 288 + 192 + 32 + (1 + 8 * i3o)];
    weights_copy[1792 + (2 + 8 * i3o) * 8 + i0] = weights[i0 * 288 + 192 + 32 + (2 + 8 * i3o)];
    weights_copy[1792 + (3 + 8 * i3o) * 8 + i0] = weights[i0 * 288 + 192 + 32 + (3 + 8 * i3o)];
    weights_copy[1792 + (4 + 8 * i3o) * 8 + i0] = weights[i0 * 288 + 192 + 32 + (4 + 8 * i3o)];
    weights_copy[1792 + (5 + 8 * i3o) * 8 + i0] = weights[i0 * 288 + 192 + 32 + (5 + 8 * i3o)];
    weights_copy[1792 + (6 + 8 * i3o) * 8 + i0] = weights[i0 * 288 + 192 + 32 + (6 + 8 * i3o)];
    weights_copy[1792 + (7 + 8 * i3o) * 8 + i0] = weights[i0 * 288 + 192 + 32 + (7 + 8 * i3o)];
  }
  for (int_fast32_t i3o = 0; i3o < 4; i3o++) {
    weights_copy[2048 + 8 * i3o * 8 + i0] = weights[i0 * 288 + 192 + 64 + 8 * i3o];
    weights_copy[2048 + (1 + 8 * i3o) * 8 + i0] = weights[i0 * 288 + 192 + 64 + (1 + 8 * i3o)];
    weights_copy[2048 + (2 + 8 * i3o) * 8 + i0] = weights[i0 * 288 + 192 + 64 + (2 + 8 * i3o)];
    weights_copy[2048 + (3 + 8 * i3o) * 8 + i0] = weights[i0 * 288 + 192 + 64 + (3 + 8 * i3o)];
    weights_copy[2048 + (4 + 8 * i3o) * 8 + i0] = weights[i0 * 288 + 192 + 64 + (4 + 8 * i3o)];
    weights_copy[2048 + (5 + 8 * i3o) * 8 + i0] = weights[i0 * 288 + 192 + 64 + (5 + 8 * i3o)];
    weights_copy[2048 + (6 + 8 * i3o) * 8 + i0] = weights[i0 * 288 + 192 + 64 + (6 + 8 * i3o)];
    weights_copy[2048 + (7 + 8 * i3o) * 8 + i0] = weights[i0 * 288 + 192 + 64 + (7 + 8 * i3o)];
  }
}
int8_t *as_ = (int8_t*) ((uint64_t)gemm_malloc (16 * 5 * sizeof(int8_t)));
int8_t *bs = (int8_t*) ((uint64_t)gemm_malloc (16 * 16 * sizeof(int8_t)));
int32_t *acc = (int32_t*) ((uint32_t)gemm_acc_malloc (16 * 5 * sizeof(int32_t)));
static int32_t sum[5 * 16];
for (int_fast32_t oh = 0; oh < 5; oh++) {
  for (int_fast32_t ow = 0; ow < 5; ow++) {
    sum[ow * 16] = bias[0];
    sum[ow * 16 + 1] = bias[1];
    sum[ow * 16 + 2] = bias[2];
    sum[ow * 16 + 3] = bias[3];
    sum[ow * 16 + 4] = bias[4];
    sum[ow * 16 + 5] = bias[5];
    sum[ow * 16 + 6] = bias[6];
    sum[ow * 16 + 7] = bias[7];
  }
  gemmini_extended3_config_ld(((struct exo_win_2i32c){ &sum[0], { 16, 1 } }).strides[0]*4, 1.0f, 0, 0);
gemmini_extended_mvin( ((uint64_t) &sum[0]), ((uint32_t) &*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)acc)) + (0)/16))), (16), (5) );
  for (int_fast32_t kh = 0; kh < 3; kh++) {
    for (int_fast32_t kw = 0; kw < 3; kw++) {
      for (int_fast32_t ico = 0; ico < 2; ico++) {
        gemmini_extended3_config_ld(((struct exo_win_2i8c){ &input[(kh + oh) * (224) + (kw) * (32) + 16 * ico], { 32, 1 } }).strides[0]*1, 1.0f, 0, 1);
gemmini_extended_mvin2( &input[(kh + oh) * (224) + (kw) * (32) + 16 * ico], ((uint64_t) &*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)as_)) + (0)/16))), (16), (5) );
        gemmini_extended3_config_ld(((struct exo_win_2i8c){ &weights_copy[(kh) * (768) + (kw) * (256) + (16 * ico) * 8], { 8, 1 } }).strides[0]*1, 1.0f, 0, 2);
gemmini_extended_mvin3( &weights_copy[(kh) * (768) + (kw) * (256) + (16 * ico) * 8], ((uint64_t) &*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)bs)) + (0)/16))), (8), (16) );
        gemmini_extended_config_ex(WS, 0, 0, 1, 0, 0);
gemmini_extended_preload((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)bs)) + (0)/16))), (uint32_t)(&*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)acc)) + (0)/16))) | 0x40000000, (8), (16), (8), (5));
gemmini_extended_compute_preloaded((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)as_)) + (0)/16))), ~((uint32_t)0), (16), (5), 16, 16);
      }
    }
  }
  gemmini_extended_config_st(((struct exo_win_2i32){ &sum[0], { 16, 1 } }).strides[0]*4, 0, 1.0f);
gemmini_extended_mvout( ((uint64_t) &sum[0]), ((uint32_t) &*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)acc)) + (0)/16)) | 0x20000000), (16), (5) );
  for (int_fast32_t ow = 0; ow < 5; ow++) {
    float val;
    val = (float)(sum[ow * 16]);
    val = val * *acc_scale;
    val = val + _select_float((float)val, (float)0.0f, (float)-0.5f, (float)0.5f);
    val = fmaxf((float)(-128.0f), (float)(_select_float((float)val, (float)127.0f, (float)val, (float)127.0f)));
    output[oh * 40 + ow * 8] = (int8_t)(val);
    float val_1;
    val_1 = (float)(sum[ow * 16 + 1]);
    val_1 = val_1 * *acc_scale;
    val_1 = val_1 + _select_float((float)val_1, (float)0.0f, (float)-0.5f, (float)0.5f);
    val_1 = fmaxf((float)(-128.0f), (float)(_select_float((float)val_1, (float)127.0f, (float)val_1, (float)127.0f)));
    output[oh * 40 + ow * 8 + 1] = (int8_t)(val_1);
    float val_2;
    val_2 = (float)(sum[ow * 16 + 2]);
    val_2 = val_2 * *acc_scale;
    val_2 = val_2 + _select_float((float)val_2, (float)0.0f, (float)-0.5f, (float)0.5f);
    val_2 = fmaxf((float)(-128.0f), (float)(_select_float((float)val_2, (float)127.0f, (float)val_2, (float)127.0f)));
    output[oh * 40 + ow * 8 + 2] = (int8_t)(val_2);
    float val_3;
    val_3 = (float)(sum[ow * 16 + 3]);
    val_3 = val_3 * *acc_scale;
    val_3 = val_3 + _select_float((float)val_3, (float)0.0f, (float)-0.5f, (float)0.5f);
    val_3 = fmaxf((float)(-128.0f), (float)(_select_float((float)val_3, (float)127.0f, (float)val_3, (float)127.0f)));
    output[oh * 40 + ow * 8 + 3] = (int8_t)(val_3);
    float val_4;
    val_4 = (float)(sum[ow * 16 + 4]);
    val_4 = val_4 * *acc_scale;
    val_4 = val_4 + _select_float((float)val_4, (float)0.0f, (float)-0.5f, (float)0.5f);
    val_4 = fmaxf((float)(-128.0f), (float)(_select_float((float)val_4, (float)127.0f, (float)val_4, (float)127.0f)));
    output[oh * 40 + ow * 8 + 4] = (int8_t)(val_4);
    float val_5;
    val_5 = (float)(sum[ow * 16 + 5]);
    val_5 = val_5 * *acc_scale;
    val_5 = val_5 + _select_float((float)val_5, (float)0.0f, (float)-0.5f, (float)0.5f);
    val_5 = fmaxf((float)(-128.0f), (float)(_select_float((float)val_5, (float)127.0f, (float)val_5, (float)127.0f)));
    output[oh * 40 + ow * 8 + 5] = (int8_t)(val_5);
    float val_6;
    val_6 = (float)(sum[ow * 16 + 6]);
    val_6 = val_6 * *acc_scale;
    val_6 = val_6 + _select_float((float)val_6, (float)0.0f, (float)-0.5f, (float)0.5f);
    val_6 = fmaxf((float)(-128.0f), (float)(_select_float((float)val_6, (float)127.0f, (float)val_6, (float)127.0f)));
    output[oh * 40 + ow * 8 + 6] = (int8_t)(val_6);
    float val_7;
    val_7 = (float)(sum[ow * 16 + 7]);
    val_7 = val_7 * *acc_scale;
    val_7 = val_7 + _select_float((float)val_7, (float)0.0f, (float)-0.5f, (float)0.5f);
    val_7 = fmaxf((float)(-128.0f), (float)(_select_float((float)val_7, (float)127.0f, (float)val_7, (float)127.0f)));
    output[oh * 40 + ow * 8 + 7] = (int8_t)(val_7);
  }
}
gemm_acc_free((uint32_t)(acc));
gemm_free((uint64_t)(bs));
gemm_free((uint64_t)(as_));
}

// dense1_leaky_opt(
//     values : i8[16, 1, 1] @DRAM,
//     scale : f32 @DRAM
// )
static void dense1_leaky_opt( void *ctxt, int8_t* values, const float* scale ) {
for (int_fast32_t i = 0; i < 16; i++) {
  float x;
  float val;
  x = (float)(values[i]);
  val = _relu_float((float)x) * *scale;
  val = val + _select_float((float)val, (float)0.0f, (float)-0.5f, (float)0.5f);
  val = fmaxf((float)(-128.0f), (float)(_select_float((float)val, (float)127.0f, (float)val, (float)127.0f)));
  values[i] = (int8_t)(val);
}
}

// dense3_leaky_opt(
//     values : i8[8, 1, 1] @DRAM,
//     scale : f32 @DRAM
// )
static void dense3_leaky_opt( void *ctxt, int8_t* values, const float* scale ) {
for (int_fast32_t i = 0; i < 8; i++) {
  float x;
  float val;
  x = (float)(values[i]);
  val = _relu_float((float)x) * *scale;
  val = val + _select_float((float)val, (float)0.0f, (float)-0.5f, (float)0.5f);
  val = fmaxf((float)(-128.0f), (float)(_select_float((float)val, (float)127.0f, (float)val, (float)127.0f)));
  values[i] = (int8_t)(val);
}
}

// dense5_leaky_opt(
//     values : i8[4, 1, 1] @DRAM,
//     scale : f32 @DRAM
// )
static void dense5_leaky_opt( void *ctxt, int8_t* values, const float* scale ) {
for (int_fast32_t i = 0; i < 4; i++) {
  float x;
  float val;
  x = (float)(values[i]);
  val = _relu_float((float)x) * *scale;
  val = val + _select_float((float)val, (float)0.0f, (float)-0.5f, (float)0.5f);
  val = fmaxf((float)(-128.0f), (float)(_select_float((float)val, (float)127.0f, (float)val, (float)127.0f)));
  values[i] = (int8_t)(val);
}
}

// dense7_leaky_opt(
//     values : i8[2, 1, 1] @DRAM,
//     scale : f32 @DRAM
// )
static void dense7_leaky_opt( void *ctxt, int8_t* values, const float* scale ) {
for (int_fast32_t i = 0; i < 2; i++) {
  float x;
  float val;
  x = (float)(values[i]);
  val = _relu_float((float)x) * *scale;
  val = val + _select_float((float)val, (float)0.0f, (float)-0.5f, (float)0.5f);
  val = fmaxf((float)(-128.0f), (float)(_select_float((float)val, (float)127.0f, (float)val, (float)127.0f)));
  values[i] = (int8_t)(val);
}
}

// fc1_opt(
//     input : i8[8, 5, 5] @DRAM,
//     weights : i8[16, 8, 5, 5] @DRAM,
//     bias : i32[16] @DRAM,
//     output : i8[16, 1, 1] @DRAM,
//     acc_scale : f32 @DRAM
// )
static void fc1_opt( void *ctxt, const int8_t* input, const int8_t* weights, const int32_t* bias, int8_t* output, const float* acc_scale ) {
for (int_fast32_t j = 0; j < 16; j++) {
  int32_t sum;
  sum = bias[j];
  for (int_fast32_t k0 = 0; k0 < 8; k0++) {
    int32_t x;
    int32_t w;
    x = (int32_t)(input[k0 * 25]);
    w = (int32_t)(weights[j * 200 + k0 * 25]);
    sum += x * w;
    int32_t x_1;
    int32_t w_1;
    x_1 = (int32_t)(input[k0 * 25 + 1]);
    w_1 = (int32_t)(weights[j * 200 + k0 * 25 + 1]);
    sum += x_1 * w_1;
    int32_t x_2;
    int32_t w_2;
    x_2 = (int32_t)(input[k0 * 25 + 2]);
    w_2 = (int32_t)(weights[j * 200 + k0 * 25 + 2]);
    sum += x_2 * w_2;
    int32_t x_3;
    int32_t w_3;
    x_3 = (int32_t)(input[k0 * 25 + 3]);
    w_3 = (int32_t)(weights[j * 200 + k0 * 25 + 3]);
    sum += x_3 * w_3;
    int32_t x_4;
    int32_t w_4;
    x_4 = (int32_t)(input[k0 * 25 + 4]);
    w_4 = (int32_t)(weights[j * 200 + k0 * 25 + 4]);
    sum += x_4 * w_4;
    int32_t x_5;
    int32_t w_5;
    x_5 = (int32_t)(input[k0 * 25 + 5]);
    w_5 = (int32_t)(weights[j * 200 + k0 * 25 + 5]);
    sum += x_5 * w_5;
    int32_t x_6;
    int32_t w_6;
    x_6 = (int32_t)(input[k0 * 25 + 5 + 1]);
    w_6 = (int32_t)(weights[j * 200 + k0 * 25 + 5 + 1]);
    sum += x_6 * w_6;
    int32_t x_7;
    int32_t w_7;
    x_7 = (int32_t)(input[k0 * 25 + 5 + 2]);
    w_7 = (int32_t)(weights[j * 200 + k0 * 25 + 5 + 2]);
    sum += x_7 * w_7;
    int32_t x_8;
    int32_t w_8;
    x_8 = (int32_t)(input[k0 * 25 + 5 + 3]);
    w_8 = (int32_t)(weights[j * 200 + k0 * 25 + 5 + 3]);
    sum += x_8 * w_8;
    int32_t x_9;
    int32_t w_9;
    x_9 = (int32_t)(input[k0 * 25 + 5 + 4]);
    w_9 = (int32_t)(weights[j * 200 + k0 * 25 + 5 + 4]);
    sum += x_9 * w_9;
    int32_t x_10;
    int32_t w_10;
    x_10 = (int32_t)(input[k0 * 25 + 10]);
    w_10 = (int32_t)(weights[j * 200 + k0 * 25 + 10]);
    sum += x_10 * w_10;
    int32_t x_11;
    int32_t w_11;
    x_11 = (int32_t)(input[k0 * 25 + 10 + 1]);
    w_11 = (int32_t)(weights[j * 200 + k0 * 25 + 10 + 1]);
    sum += x_11 * w_11;
    int32_t x_12;
    int32_t w_12;
    x_12 = (int32_t)(input[k0 * 25 + 10 + 2]);
    w_12 = (int32_t)(weights[j * 200 + k0 * 25 + 10 + 2]);
    sum += x_12 * w_12;
    int32_t x_13;
    int32_t w_13;
    x_13 = (int32_t)(input[k0 * 25 + 10 + 3]);
    w_13 = (int32_t)(weights[j * 200 + k0 * 25 + 10 + 3]);
    sum += x_13 * w_13;
    int32_t x_14;
    int32_t w_14;
    x_14 = (int32_t)(input[k0 * 25 + 10 + 4]);
    w_14 = (int32_t)(weights[j * 200 + k0 * 25 + 10 + 4]);
    sum += x_14 * w_14;
    int32_t x_15;
    int32_t w_15;
    x_15 = (int32_t)(input[k0 * 25 + 15]);
    w_15 = (int32_t)(weights[j * 200 + k0 * 25 + 15]);
    sum += x_15 * w_15;
    int32_t x_16;
    int32_t w_16;
    x_16 = (int32_t)(input[k0 * 25 + 15 + 1]);
    w_16 = (int32_t)(weights[j * 200 + k0 * 25 + 15 + 1]);
    sum += x_16 * w_16;
    int32_t x_17;
    int32_t w_17;
    x_17 = (int32_t)(input[k0 * 25 + 15 + 2]);
    w_17 = (int32_t)(weights[j * 200 + k0 * 25 + 15 + 2]);
    sum += x_17 * w_17;
    int32_t x_18;
    int32_t w_18;
    x_18 = (int32_t)(input[k0 * 25 + 15 + 3]);
    w_18 = (int32_t)(weights[j * 200 + k0 * 25 + 15 + 3]);
    sum += x_18 * w_18;
    int32_t x_19;
    int32_t w_19;
    x_19 = (int32_t)(input[k0 * 25 + 15 + 4]);
    w_19 = (int32_t)(weights[j * 200 + k0 * 25 + 15 + 4]);
    sum += x_19 * w_19;
    int32_t x_20;
    int32_t w_20;
    x_20 = (int32_t)(input[k0 * 25 + 20]);
    w_20 = (int32_t)(weights[j * 200 + k0 * 25 + 20]);
    sum += x_20 * w_20;
    int32_t x_21;
    int32_t w_21;
    x_21 = (int32_t)(input[k0 * 25 + 20 + 1]);
    w_21 = (int32_t)(weights[j * 200 + k0 * 25 + 20 + 1]);
    sum += x_21 * w_21;
    int32_t x_22;
    int32_t w_22;
    x_22 = (int32_t)(input[k0 * 25 + 20 + 2]);
    w_22 = (int32_t)(weights[j * 200 + k0 * 25 + 20 + 2]);
    sum += x_22 * w_22;
    int32_t x_23;
    int32_t w_23;
    x_23 = (int32_t)(input[k0 * 25 + 20 + 3]);
    w_23 = (int32_t)(weights[j * 200 + k0 * 25 + 20 + 3]);
    sum += x_23 * w_23;
    int32_t x_24;
    int32_t w_24;
    x_24 = (int32_t)(input[k0 * 25 + 20 + 4]);
    w_24 = (int32_t)(weights[j * 200 + k0 * 25 + 20 + 4]);
    sum += x_24 * w_24;
  }
  float val;
  val = (float)(sum);
  val = val * *acc_scale;
  val = val + _select_float((float)val, (float)0.0f, (float)-0.5f, (float)0.5f);
  val = fmaxf((float)(-128.0f), (float)(_select_float((float)val, (float)127.0f, (float)val, (float)127.0f)));
  output[j] = (int8_t)(val);
}
}

// fc2_opt(
//     input : i8[16, 1, 1] @DRAM,
//     weights : i8[8, 16, 1, 1] @DRAM,
//     bias : i32[8] @DRAM,
//     output : i8[8, 1, 1] @DRAM,
//     acc_scale : f32 @DRAM
// )
static void fc2_opt( void *ctxt, const int8_t* input, const int8_t* weights, const int32_t* bias, int8_t* output, const float* acc_scale ) {
for (int_fast32_t j = 0; j < 8; j++) {
  int32_t sum;
  sum = bias[j];
  int32_t x;
  int32_t w;
  x = (int32_t)(input[0]);
  w = (int32_t)(weights[j * 16]);
  sum += x * w;
  int32_t x_1;
  int32_t w_1;
  x_1 = (int32_t)(input[1]);
  w_1 = (int32_t)(weights[j * 16 + 1]);
  sum += x_1 * w_1;
  int32_t x_2;
  int32_t w_2;
  x_2 = (int32_t)(input[2]);
  w_2 = (int32_t)(weights[j * 16 + 2]);
  sum += x_2 * w_2;
  int32_t x_3;
  int32_t w_3;
  x_3 = (int32_t)(input[3]);
  w_3 = (int32_t)(weights[j * 16 + 3]);
  sum += x_3 * w_3;
  int32_t x_4;
  int32_t w_4;
  x_4 = (int32_t)(input[4]);
  w_4 = (int32_t)(weights[j * 16 + 4]);
  sum += x_4 * w_4;
  int32_t x_5;
  int32_t w_5;
  x_5 = (int32_t)(input[5]);
  w_5 = (int32_t)(weights[j * 16 + 5]);
  sum += x_5 * w_5;
  int32_t x_6;
  int32_t w_6;
  x_6 = (int32_t)(input[6]);
  w_6 = (int32_t)(weights[j * 16 + 6]);
  sum += x_6 * w_6;
  int32_t x_7;
  int32_t w_7;
  x_7 = (int32_t)(input[7]);
  w_7 = (int32_t)(weights[j * 16 + 7]);
  sum += x_7 * w_7;
  int32_t x_8;
  int32_t w_8;
  x_8 = (int32_t)(input[8]);
  w_8 = (int32_t)(weights[j * 16 + 8]);
  sum += x_8 * w_8;
  int32_t x_9;
  int32_t w_9;
  x_9 = (int32_t)(input[9]);
  w_9 = (int32_t)(weights[j * 16 + 9]);
  sum += x_9 * w_9;
  int32_t x_10;
  int32_t w_10;
  x_10 = (int32_t)(input[10]);
  w_10 = (int32_t)(weights[j * 16 + 10]);
  sum += x_10 * w_10;
  int32_t x_11;
  int32_t w_11;
  x_11 = (int32_t)(input[11]);
  w_11 = (int32_t)(weights[j * 16 + 11]);
  sum += x_11 * w_11;
  int32_t x_12;
  int32_t w_12;
  x_12 = (int32_t)(input[12]);
  w_12 = (int32_t)(weights[j * 16 + 12]);
  sum += x_12 * w_12;
  int32_t x_13;
  int32_t w_13;
  x_13 = (int32_t)(input[13]);
  w_13 = (int32_t)(weights[j * 16 + 13]);
  sum += x_13 * w_13;
  int32_t x_14;
  int32_t w_14;
  x_14 = (int32_t)(input[14]);
  w_14 = (int32_t)(weights[j * 16 + 14]);
  sum += x_14 * w_14;
  int32_t x_15;
  int32_t w_15;
  x_15 = (int32_t)(input[15]);
  w_15 = (int32_t)(weights[j * 16 + 15]);
  sum += x_15 * w_15;
  float val;
  val = (float)(sum);
  val = val * *acc_scale;
  val = val + _select_float((float)val, (float)0.0f, (float)-0.5f, (float)0.5f);
  val = fmaxf((float)(-128.0f), (float)(_select_float((float)val, (float)127.0f, (float)val, (float)127.0f)));
  output[j] = (int8_t)(val);
}
}

// fc3_opt(
//     input : i8[8, 1, 1] @DRAM,
//     weights : i8[4, 8, 1, 1] @DRAM,
//     bias : i32[4] @DRAM,
//     output : i8[4, 1, 1] @DRAM,
//     acc_scale : f32 @DRAM
// )
static void fc3_opt( void *ctxt, const int8_t* input, const int8_t* weights, const int32_t* bias, int8_t* output, const float* acc_scale ) {
for (int_fast32_t j = 0; j < 4; j++) {
  int32_t sum;
  sum = bias[j];
  int32_t x;
  int32_t w;
  x = (int32_t)(input[0]);
  w = (int32_t)(weights[j * 8]);
  sum += x * w;
  int32_t x_1;
  int32_t w_1;
  x_1 = (int32_t)(input[1]);
  w_1 = (int32_t)(weights[j * 8 + 1]);
  sum += x_1 * w_1;
  int32_t x_2;
  int32_t w_2;
  x_2 = (int32_t)(input[2]);
  w_2 = (int32_t)(weights[j * 8 + 2]);
  sum += x_2 * w_2;
  int32_t x_3;
  int32_t w_3;
  x_3 = (int32_t)(input[3]);
  w_3 = (int32_t)(weights[j * 8 + 3]);
  sum += x_3 * w_3;
  int32_t x_4;
  int32_t w_4;
  x_4 = (int32_t)(input[4]);
  w_4 = (int32_t)(weights[j * 8 + 4]);
  sum += x_4 * w_4;
  int32_t x_5;
  int32_t w_5;
  x_5 = (int32_t)(input[5]);
  w_5 = (int32_t)(weights[j * 8 + 5]);
  sum += x_5 * w_5;
  int32_t x_6;
  int32_t w_6;
  x_6 = (int32_t)(input[6]);
  w_6 = (int32_t)(weights[j * 8 + 6]);
  sum += x_6 * w_6;
  int32_t x_7;
  int32_t w_7;
  x_7 = (int32_t)(input[7]);
  w_7 = (int32_t)(weights[j * 8 + 7]);
  sum += x_7 * w_7;
  float val;
  val = (float)(sum);
  val = val * *acc_scale;
  val = val + _select_float((float)val, (float)0.0f, (float)-0.5f, (float)0.5f);
  val = fmaxf((float)(-128.0f), (float)(_select_float((float)val, (float)127.0f, (float)val, (float)127.0f)));
  output[j] = (int8_t)(val);
}
}

// fc4_opt(
//     input : i8[4, 1, 1] @DRAM,
//     weights : i8[2, 4, 1, 1] @DRAM,
//     bias : i32[2] @DRAM,
//     output : i8[2, 1, 1] @DRAM,
//     acc_scale : f32 @DRAM
// )
static void fc4_opt( void *ctxt, const int8_t* input, const int8_t* weights, const int32_t* bias, int8_t* output, const float* acc_scale ) {
for (int_fast32_t j = 0; j < 2; j++) {
  int32_t sum;
  sum = bias[j];
  int32_t x;
  int32_t w;
  x = (int32_t)(input[0]);
  w = (int32_t)(weights[j * 4]);
  sum += x * w;
  int32_t x_1;
  int32_t w_1;
  x_1 = (int32_t)(input[1]);
  w_1 = (int32_t)(weights[j * 4 + 1]);
  sum += x_1 * w_1;
  int32_t x_2;
  int32_t w_2;
  x_2 = (int32_t)(input[2]);
  w_2 = (int32_t)(weights[j * 4 + 2]);
  sum += x_2 * w_2;
  int32_t x_3;
  int32_t w_3;
  x_3 = (int32_t)(input[3]);
  w_3 = (int32_t)(weights[j * 4 + 3]);
  sum += x_3 * w_3;
  float val;
  val = (float)(sum);
  val = val * *acc_scale;
  val = val + _select_float((float)val, (float)0.0f, (float)-0.5f, (float)0.5f);
  val = fmaxf((float)(-128.0f), (float)(_select_float((float)val, (float)127.0f, (float)val, (float)127.0f)));
  output[j] = (int8_t)(val);
}
}

// fc_output_opt(
//     input : i8[2, 1, 1] @DRAM,
//     weights : i8[2, 2, 1, 1] @DRAM,
//     bias : i32[2] @DRAM,
//     output : i8[2, 1, 1] @DRAM,
//     acc_scale : f32 @DRAM
// )
static void fc_output_opt( void *ctxt, const int8_t* input, const int8_t* weights, const int32_t* bias, int8_t* output, const float* acc_scale ) {
for (int_fast32_t j = 0; j < 2; j++) {
  int32_t sum;
  sum = bias[j];
  int32_t x;
  int32_t w;
  x = (int32_t)(input[0]);
  w = (int32_t)(weights[j * 2]);
  sum += x * w;
  int32_t x_1;
  int32_t w_1;
  x_1 = (int32_t)(input[1]);
  w_1 = (int32_t)(weights[j * 2 + 1]);
  sum += x_1 * w_1;
  float val;
  val = (float)(sum);
  val = val * *acc_scale;
  val = val + _select_float((float)val, (float)0.0f, (float)-0.5f, (float)0.5f);
  val = fmaxf((float)(-128.0f), (float)(_select_float((float)val, (float)127.0f, (float)val, (float)127.0f)));
  output[j] = (int8_t)(val);
}
}


/* relying on the following instruction..."
ld_acc_i32(n,m,src,dst)
gemmini_extended3_config_ld({src}.strides[0]*4, 1.0f, 0, 0);
gemmini_extended_mvin( ((uint64_t) &{src_data}), ((uint32_t) &{dst_data}), {m}, {n} );
*/

/* relying on the following instruction..."
ld_i8_id1(n,m,src,dst)
gemmini_extended3_config_ld({src}.strides[0]*1, 1.0f, 0, 1);
gemmini_extended_mvin2( &{src_data}, ((uint64_t) &{dst_data}), {m}, {n} );
*/

/* relying on the following instruction..."
ld_i8_id2(n,m,src,dst)
gemmini_extended3_config_ld({src}.strides[0]*1, 1.0f, 0, 2);
gemmini_extended_mvin3( &{src_data}, ((uint64_t) &{dst_data}), {m}, {n} );
*/
// leaky1_opt(
//     values : i8[9, 9, 64] @DRAM,
//     scale : f32 @DRAM
// )
static void leaky1_opt( void *ctxt, int8_t* values, const float* scale ) {
for (int_fast32_t i = 0; i < 9; i++) {
  for (int_fast32_t j = 0; j < 9; j++) {
    for (int_fast32_t ko = 0; ko < 8; ko++) {
      float x;
      float val;
      x = (float)(values[i * 576 + j * 64 + 8 * ko]);
      val = _relu_float((float)x) * *scale;
      val = val + _select_float((float)val, (float)0.0f, (float)-0.5f, (float)0.5f);
      val = fmaxf((float)(-128.0f), (float)(_select_float((float)val, (float)127.0f, (float)val, (float)127.0f)));
      values[i * 576 + j * 64 + 8 * ko] = (int8_t)(val);
      float x_1;
      float val_1;
      x_1 = (float)(values[i * 576 + j * 64 + (1 + 8 * ko)]);
      val_1 = _relu_float((float)x_1) * *scale;
      val_1 = val_1 + _select_float((float)val_1, (float)0.0f, (float)-0.5f, (float)0.5f);
      val_1 = fmaxf((float)(-128.0f), (float)(_select_float((float)val_1, (float)127.0f, (float)val_1, (float)127.0f)));
      values[i * 576 + j * 64 + (1 + 8 * ko)] = (int8_t)(val_1);
      float x_2;
      float val_2;
      x_2 = (float)(values[i * 576 + j * 64 + (2 + 8 * ko)]);
      val_2 = _relu_float((float)x_2) * *scale;
      val_2 = val_2 + _select_float((float)val_2, (float)0.0f, (float)-0.5f, (float)0.5f);
      val_2 = fmaxf((float)(-128.0f), (float)(_select_float((float)val_2, (float)127.0f, (float)val_2, (float)127.0f)));
      values[i * 576 + j * 64 + (2 + 8 * ko)] = (int8_t)(val_2);
      float x_3;
      float val_3;
      x_3 = (float)(values[i * 576 + j * 64 + (3 + 8 * ko)]);
      val_3 = _relu_float((float)x_3) * *scale;
      val_3 = val_3 + _select_float((float)val_3, (float)0.0f, (float)-0.5f, (float)0.5f);
      val_3 = fmaxf((float)(-128.0f), (float)(_select_float((float)val_3, (float)127.0f, (float)val_3, (float)127.0f)));
      values[i * 576 + j * 64 + (3 + 8 * ko)] = (int8_t)(val_3);
      float x_4;
      float val_4;
      x_4 = (float)(values[i * 576 + j * 64 + (4 + 8 * ko)]);
      val_4 = _relu_float((float)x_4) * *scale;
      val_4 = val_4 + _select_float((float)val_4, (float)0.0f, (float)-0.5f, (float)0.5f);
      val_4 = fmaxf((float)(-128.0f), (float)(_select_float((float)val_4, (float)127.0f, (float)val_4, (float)127.0f)));
      values[i * 576 + j * 64 + (4 + 8 * ko)] = (int8_t)(val_4);
      float x_5;
      float val_5;
      x_5 = (float)(values[i * 576 + j * 64 + (5 + 8 * ko)]);
      val_5 = _relu_float((float)x_5) * *scale;
      val_5 = val_5 + _select_float((float)val_5, (float)0.0f, (float)-0.5f, (float)0.5f);
      val_5 = fmaxf((float)(-128.0f), (float)(_select_float((float)val_5, (float)127.0f, (float)val_5, (float)127.0f)));
      values[i * 576 + j * 64 + (5 + 8 * ko)] = (int8_t)(val_5);
      float x_6;
      float val_6;
      x_6 = (float)(values[i * 576 + j * 64 + (6 + 8 * ko)]);
      val_6 = _relu_float((float)x_6) * *scale;
      val_6 = val_6 + _select_float((float)val_6, (float)0.0f, (float)-0.5f, (float)0.5f);
      val_6 = fmaxf((float)(-128.0f), (float)(_select_float((float)val_6, (float)127.0f, (float)val_6, (float)127.0f)));
      values[i * 576 + j * 64 + (6 + 8 * ko)] = (int8_t)(val_6);
      float x_7;
      float val_7;
      x_7 = (float)(values[i * 576 + j * 64 + (7 + 8 * ko)]);
      val_7 = _relu_float((float)x_7) * *scale;
      val_7 = val_7 + _select_float((float)val_7, (float)0.0f, (float)-0.5f, (float)0.5f);
      val_7 = fmaxf((float)(-128.0f), (float)(_select_float((float)val_7, (float)127.0f, (float)val_7, (float)127.0f)));
      values[i * 576 + j * 64 + (7 + 8 * ko)] = (int8_t)(val_7);
    }
  }
}
}

// leaky3_opt(
//     values : i8[7, 7, 32] @DRAM,
//     scale : f32 @DRAM
// )
static void leaky3_opt( void *ctxt, int8_t* values, const float* scale ) {
for (int_fast32_t i = 0; i < 7; i++) {
  for (int_fast32_t j = 0; j < 7; j++) {
    for (int_fast32_t ko = 0; ko < 4; ko++) {
      float x;
      float val;
      x = (float)(values[i * 224 + j * 32 + 8 * ko]);
      val = _relu_float((float)x) * *scale;
      val = val + _select_float((float)val, (float)0.0f, (float)-0.5f, (float)0.5f);
      val = fmaxf((float)(-128.0f), (float)(_select_float((float)val, (float)127.0f, (float)val, (float)127.0f)));
      values[i * 224 + j * 32 + 8 * ko] = (int8_t)(val);
      float x_1;
      float val_1;
      x_1 = (float)(values[i * 224 + j * 32 + (1 + 8 * ko)]);
      val_1 = _relu_float((float)x_1) * *scale;
      val_1 = val_1 + _select_float((float)val_1, (float)0.0f, (float)-0.5f, (float)0.5f);
      val_1 = fmaxf((float)(-128.0f), (float)(_select_float((float)val_1, (float)127.0f, (float)val_1, (float)127.0f)));
      values[i * 224 + j * 32 + (1 + 8 * ko)] = (int8_t)(val_1);
      float x_2;
      float val_2;
      x_2 = (float)(values[i * 224 + j * 32 + (2 + 8 * ko)]);
      val_2 = _relu_float((float)x_2) * *scale;
      val_2 = val_2 + _select_float((float)val_2, (float)0.0f, (float)-0.5f, (float)0.5f);
      val_2 = fmaxf((float)(-128.0f), (float)(_select_float((float)val_2, (float)127.0f, (float)val_2, (float)127.0f)));
      values[i * 224 + j * 32 + (2 + 8 * ko)] = (int8_t)(val_2);
      float x_3;
      float val_3;
      x_3 = (float)(values[i * 224 + j * 32 + (3 + 8 * ko)]);
      val_3 = _relu_float((float)x_3) * *scale;
      val_3 = val_3 + _select_float((float)val_3, (float)0.0f, (float)-0.5f, (float)0.5f);
      val_3 = fmaxf((float)(-128.0f), (float)(_select_float((float)val_3, (float)127.0f, (float)val_3, (float)127.0f)));
      values[i * 224 + j * 32 + (3 + 8 * ko)] = (int8_t)(val_3);
      float x_4;
      float val_4;
      x_4 = (float)(values[i * 224 + j * 32 + (4 + 8 * ko)]);
      val_4 = _relu_float((float)x_4) * *scale;
      val_4 = val_4 + _select_float((float)val_4, (float)0.0f, (float)-0.5f, (float)0.5f);
      val_4 = fmaxf((float)(-128.0f), (float)(_select_float((float)val_4, (float)127.0f, (float)val_4, (float)127.0f)));
      values[i * 224 + j * 32 + (4 + 8 * ko)] = (int8_t)(val_4);
      float x_5;
      float val_5;
      x_5 = (float)(values[i * 224 + j * 32 + (5 + 8 * ko)]);
      val_5 = _relu_float((float)x_5) * *scale;
      val_5 = val_5 + _select_float((float)val_5, (float)0.0f, (float)-0.5f, (float)0.5f);
      val_5 = fmaxf((float)(-128.0f), (float)(_select_float((float)val_5, (float)127.0f, (float)val_5, (float)127.0f)));
      values[i * 224 + j * 32 + (5 + 8 * ko)] = (int8_t)(val_5);
      float x_6;
      float val_6;
      x_6 = (float)(values[i * 224 + j * 32 + (6 + 8 * ko)]);
      val_6 = _relu_float((float)x_6) * *scale;
      val_6 = val_6 + _select_float((float)val_6, (float)0.0f, (float)-0.5f, (float)0.5f);
      val_6 = fmaxf((float)(-128.0f), (float)(_select_float((float)val_6, (float)127.0f, (float)val_6, (float)127.0f)));
      values[i * 224 + j * 32 + (6 + 8 * ko)] = (int8_t)(val_6);
      float x_7;
      float val_7;
      x_7 = (float)(values[i * 224 + j * 32 + (7 + 8 * ko)]);
      val_7 = _relu_float((float)x_7) * *scale;
      val_7 = val_7 + _select_float((float)val_7, (float)0.0f, (float)-0.5f, (float)0.5f);
      val_7 = fmaxf((float)(-128.0f), (float)(_select_float((float)val_7, (float)127.0f, (float)val_7, (float)127.0f)));
      values[i * 224 + j * 32 + (7 + 8 * ko)] = (int8_t)(val_7);
    }
  }
}
}

// leaky5_opt(
//     values : i8[8, 5, 5] @DRAM,
//     scale : f32 @DRAM
// )
static void leaky5_opt( void *ctxt, int8_t* values, const float* scale ) {
for (int_fast32_t i = 0; i < 8; i++) {
  for (int_fast32_t j = 0; j < 5; j++) {
    float x;
    float val;
    x = (float)(values[i * 25 + j * 5]);
    val = _relu_float((float)x) * *scale;
    val = val + _select_float((float)val, (float)0.0f, (float)-0.5f, (float)0.5f);
    val = fmaxf((float)(-128.0f), (float)(_select_float((float)val, (float)127.0f, (float)val, (float)127.0f)));
    values[i * 25 + j * 5] = (int8_t)(val);
    float x_1;
    float val_1;
    x_1 = (float)(values[i * 25 + j * 5 + 1]);
    val_1 = _relu_float((float)x_1) * *scale;
    val_1 = val_1 + _select_float((float)val_1, (float)0.0f, (float)-0.5f, (float)0.5f);
    val_1 = fmaxf((float)(-128.0f), (float)(_select_float((float)val_1, (float)127.0f, (float)val_1, (float)127.0f)));
    values[i * 25 + j * 5 + 1] = (int8_t)(val_1);
    float x_2;
    float val_2;
    x_2 = (float)(values[i * 25 + j * 5 + 2]);
    val_2 = _relu_float((float)x_2) * *scale;
    val_2 = val_2 + _select_float((float)val_2, (float)0.0f, (float)-0.5f, (float)0.5f);
    val_2 = fmaxf((float)(-128.0f), (float)(_select_float((float)val_2, (float)127.0f, (float)val_2, (float)127.0f)));
    values[i * 25 + j * 5 + 2] = (int8_t)(val_2);
    float x_3;
    float val_3;
    x_3 = (float)(values[i * 25 + j * 5 + 3]);
    val_3 = _relu_float((float)x_3) * *scale;
    val_3 = val_3 + _select_float((float)val_3, (float)0.0f, (float)-0.5f, (float)0.5f);
    val_3 = fmaxf((float)(-128.0f), (float)(_select_float((float)val_3, (float)127.0f, (float)val_3, (float)127.0f)));
    values[i * 25 + j * 5 + 3] = (int8_t)(val_3);
    float x_4;
    float val_4;
    x_4 = (float)(values[i * 25 + j * 5 + 4]);
    val_4 = _relu_float((float)x_4) * *scale;
    val_4 = val_4 + _select_float((float)val_4, (float)0.0f, (float)-0.5f, (float)0.5f);
    val_4 = fmaxf((float)(-128.0f), (float)(_select_float((float)val_4, (float)127.0f, (float)val_4, (float)127.0f)));
    values[i * 25 + j * 5 + 4] = (int8_t)(val_4);
  }
}
}


/* relying on the following instruction..."
matmul_acc_i8(N,M,K,A,B,C)
gemmini_extended_config_ex(WS, 0, 0, 1, 0, 0);
gemmini_extended_preload((uint32_t)(&{B_data}), (uint32_t)(&{C_data}) | 0x40000000, {M}, {K}, {M}, {N});
gemmini_extended_compute_preloaded((uint32_t)(&{A_data}), ~((uint32_t)0), {K}, {N}, 16, 16);
*/
// matmul_transA_gemmini(
//     A : i8[32, 9, 9] @DRAM,
//     B : i8[32, 9, 9] @DRAM,
//     C : i8[9, 9, 9, 9] @DRAM,
//     acc_scale : f32 @DRAM
// )
static void matmul_transA_gemmini( void *ctxt, const int8_t* A, const int8_t* B, int8_t* C, const float* acc_scale ) {
static int8_t At_all[9 * 9 * 32];
for (int_fast32_t i0o = 0; i0o < 4; i0o++) {
  for (int_fast32_t i1 = 0; i1 < 9; i1++) {
    for (int_fast32_t i2 = 0; i2 < 9; i2++) {
      At_all[i1 * 288 + i2 * 32 + 8 * i0o] = A[8 * i0o * 81 + i1 * 9 + i2];
    }
  }
  for (int_fast32_t i1 = 0; i1 < 9; i1++) {
    for (int_fast32_t i2 = 0; i2 < 9; i2++) {
      At_all[i1 * 288 + i2 * 32 + (1 + 8 * i0o)] = A[(1 + 8 * i0o) * 81 + i1 * 9 + i2];
    }
  }
  for (int_fast32_t i1 = 0; i1 < 9; i1++) {
    for (int_fast32_t i2 = 0; i2 < 9; i2++) {
      At_all[i1 * 288 + i2 * 32 + (2 + 8 * i0o)] = A[(2 + 8 * i0o) * 81 + i1 * 9 + i2];
    }
  }
  for (int_fast32_t i1 = 0; i1 < 9; i1++) {
    for (int_fast32_t i2 = 0; i2 < 9; i2++) {
      At_all[i1 * 288 + i2 * 32 + (3 + 8 * i0o)] = A[(3 + 8 * i0o) * 81 + i1 * 9 + i2];
    }
  }
  for (int_fast32_t i1 = 0; i1 < 9; i1++) {
    for (int_fast32_t i2 = 0; i2 < 9; i2++) {
      At_all[i1 * 288 + i2 * 32 + (4 + 8 * i0o)] = A[(4 + 8 * i0o) * 81 + i1 * 9 + i2];
    }
  }
  for (int_fast32_t i1 = 0; i1 < 9; i1++) {
    for (int_fast32_t i2 = 0; i2 < 9; i2++) {
      At_all[i1 * 288 + i2 * 32 + (5 + 8 * i0o)] = A[(5 + 8 * i0o) * 81 + i1 * 9 + i2];
    }
  }
  for (int_fast32_t i1 = 0; i1 < 9; i1++) {
    for (int_fast32_t i2 = 0; i2 < 9; i2++) {
      At_all[i1 * 288 + i2 * 32 + (6 + 8 * i0o)] = A[(6 + 8 * i0o) * 81 + i1 * 9 + i2];
    }
  }
  for (int_fast32_t i1 = 0; i1 < 9; i1++) {
    for (int_fast32_t i2 = 0; i2 < 9; i2++) {
      At_all[i1 * 288 + i2 * 32 + (7 + 8 * i0o)] = A[(7 + 8 * i0o) * 81 + i1 * 9 + i2];
    }
  }
}
int8_t *as_ = (int8_t*) ((uint64_t)gemm_malloc (16 * 9 * sizeof(int8_t)));
int8_t *bs = (int8_t*) ((uint64_t)gemm_malloc (16 * 16 * sizeof(int8_t)));
int32_t *sum = (int32_t*) ((uint32_t)gemm_acc_malloc (16 * 9 * sizeof(int32_t)));
static int32_t sum_out[9 * 9];
for (int_fast32_t i1 = 0; i1 < 9; i1++) {
  for (int_fast32_t j1 = 0; j1 < 9; j1++) {
    gemmini_extended3_config_ld(0, 1.0f, 0, 0);
gemmini_extended_mvin( 0, ((uint64_t) &*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)sum)) + (0)/16))),(9), (9) );
    for (int_fast32_t ko = 0; ko < 2; ko++) {
      gemmini_extended3_config_ld(((struct exo_win_2i8c){ &B[(16 * ko) * (81) + (j1) * 9], { 81, 1 } }).strides[0]*1, 1.0f, 0, 2);
gemmini_extended_mvin3( &B[(16 * ko) * (81) + (j1) * 9], ((uint64_t) &*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)bs)) + (0)/16))), (9), (16) );
      gemmini_extended3_config_ld(((struct exo_win_2i8c){ &At_all[(i1) * (288) + 16 * ko], { 32, 1 } }).strides[0]*1, 1.0f, 0, 1);
gemmini_extended_mvin2( &At_all[(i1) * (288) + 16 * ko], ((uint64_t) &*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)as_)) + (0)/16))), (16), (9) );
      gemmini_extended_config_ex(WS, 0, 0, 1, 0, 0);
gemmini_extended_preload((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)bs)) + (0)/16))), (uint32_t)(&*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)sum)) + (0)/16))) | 0x40000000, (9), (16), (9), (9));
gemmini_extended_compute_preloaded((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)as_)) + (0)/16))), ~((uint32_t)0), (16), (9), 16, 16);
    }
    gemmini_extended_config_st(((struct exo_win_2i32){ &sum_out[0], { 9, 1 } }).strides[0]*4, 0, 1.0f);
gemmini_extended_mvout( ((uint64_t) &sum_out[0]), ((uint32_t) &*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)sum)) + (0)/16)) | 0x20000000), (9), (9) );
    for (int_fast32_t i2 = 0; i2 < 9; i2++) {
      float val;
      val = (float)(sum_out[i2 * 9]);
      val = val * *acc_scale;
      val = val + _select_float((float)val, (float)0.0f, (float)-0.5f, (float)0.5f);
      val = fmaxf((float)(-128.0f), (float)(_select_float((float)val, (float)127.0f, (float)val, (float)127.0f)));
      C[i1 * 729 + i2 * 81 + j1 * 9] = (int8_t)(val);
      float val_1;
      val_1 = (float)(sum_out[i2 * 9 + 1]);
      val_1 = val_1 * *acc_scale;
      val_1 = val_1 + _select_float((float)val_1, (float)0.0f, (float)-0.5f, (float)0.5f);
      val_1 = fmaxf((float)(-128.0f), (float)(_select_float((float)val_1, (float)127.0f, (float)val_1, (float)127.0f)));
      C[i1 * 729 + i2 * 81 + j1 * 9 + 1] = (int8_t)(val_1);
      float val_2;
      val_2 = (float)(sum_out[i2 * 9 + 2]);
      val_2 = val_2 * *acc_scale;
      val_2 = val_2 + _select_float((float)val_2, (float)0.0f, (float)-0.5f, (float)0.5f);
      val_2 = fmaxf((float)(-128.0f), (float)(_select_float((float)val_2, (float)127.0f, (float)val_2, (float)127.0f)));
      C[i1 * 729 + i2 * 81 + j1 * 9 + 2] = (int8_t)(val_2);
      float val_3;
      val_3 = (float)(sum_out[i2 * 9 + 3]);
      val_3 = val_3 * *acc_scale;
      val_3 = val_3 + _select_float((float)val_3, (float)0.0f, (float)-0.5f, (float)0.5f);
      val_3 = fmaxf((float)(-128.0f), (float)(_select_float((float)val_3, (float)127.0f, (float)val_3, (float)127.0f)));
      C[i1 * 729 + i2 * 81 + j1 * 9 + 3] = (int8_t)(val_3);
      float val_4;
      val_4 = (float)(sum_out[i2 * 9 + 4]);
      val_4 = val_4 * *acc_scale;
      val_4 = val_4 + _select_float((float)val_4, (float)0.0f, (float)-0.5f, (float)0.5f);
      val_4 = fmaxf((float)(-128.0f), (float)(_select_float((float)val_4, (float)127.0f, (float)val_4, (float)127.0f)));
      C[i1 * 729 + i2 * 81 + j1 * 9 + 4] = (int8_t)(val_4);
      float val_5;
      val_5 = (float)(sum_out[i2 * 9 + 5]);
      val_5 = val_5 * *acc_scale;
      val_5 = val_5 + _select_float((float)val_5, (float)0.0f, (float)-0.5f, (float)0.5f);
      val_5 = fmaxf((float)(-128.0f), (float)(_select_float((float)val_5, (float)127.0f, (float)val_5, (float)127.0f)));
      C[i1 * 729 + i2 * 81 + j1 * 9 + 5] = (int8_t)(val_5);
      float val_6;
      val_6 = (float)(sum_out[i2 * 9 + 6]);
      val_6 = val_6 * *acc_scale;
      val_6 = val_6 + _select_float((float)val_6, (float)0.0f, (float)-0.5f, (float)0.5f);
      val_6 = fmaxf((float)(-128.0f), (float)(_select_float((float)val_6, (float)127.0f, (float)val_6, (float)127.0f)));
      C[i1 * 729 + i2 * 81 + j1 * 9 + 6] = (int8_t)(val_6);
      float val_7;
      val_7 = (float)(sum_out[i2 * 9 + 7]);
      val_7 = val_7 * *acc_scale;
      val_7 = val_7 + _select_float((float)val_7, (float)0.0f, (float)-0.5f, (float)0.5f);
      val_7 = fmaxf((float)(-128.0f), (float)(_select_float((float)val_7, (float)127.0f, (float)val_7, (float)127.0f)));
      C[i1 * 729 + i2 * 81 + j1 * 9 + 7] = (int8_t)(val_7);
      float val_8;
      val_8 = (float)(sum_out[i2 * 9 + 8]);
      val_8 = val_8 * *acc_scale;
      val_8 = val_8 + _select_float((float)val_8, (float)0.0f, (float)-0.5f, (float)0.5f);
      val_8 = fmaxf((float)(-128.0f), (float)(_select_float((float)val_8, (float)127.0f, (float)val_8, (float)127.0f)));
      C[i1 * 729 + i2 * 81 + j1 * 9 + 8] = (int8_t)(val_8);
    }
  }
}
gemm_acc_free((uint32_t)(sum));
gemm_free((uint64_t)(bs));
gemm_free((uint64_t)(as_));
}

// matmul_transB_gemmini(
//     A : i8[9, 9, 9, 9] @DRAM,
//     B : i8[32, 9, 9] @DRAM,
//     C : i8[9, 9, 32] @DRAM,
//     acc_scale : f32 @DRAM
// )
static void matmul_transB_gemmini( void *ctxt, const int8_t* A, const int8_t* B, int8_t* C, const float* acc_scale ) {
static int8_t Bt_all[9 * 9 * 32];
for (int_fast32_t i0o = 0; i0o < 4; i0o++) {
  for (int_fast32_t i1 = 0; i1 < 9; i1++) {
    for (int_fast32_t i2 = 0; i2 < 9; i2++) {
      Bt_all[i1 * 288 + i2 * 32 + 8 * i0o] = B[8 * i0o * 81 + i1 * 9 + i2];
    }
  }
  for (int_fast32_t i1 = 0; i1 < 9; i1++) {
    for (int_fast32_t i2 = 0; i2 < 9; i2++) {
      Bt_all[i1 * 288 + i2 * 32 + (1 + 8 * i0o)] = B[(1 + 8 * i0o) * 81 + i1 * 9 + i2];
    }
  }
  for (int_fast32_t i1 = 0; i1 < 9; i1++) {
    for (int_fast32_t i2 = 0; i2 < 9; i2++) {
      Bt_all[i1 * 288 + i2 * 32 + (2 + 8 * i0o)] = B[(2 + 8 * i0o) * 81 + i1 * 9 + i2];
    }
  }
  for (int_fast32_t i1 = 0; i1 < 9; i1++) {
    for (int_fast32_t i2 = 0; i2 < 9; i2++) {
      Bt_all[i1 * 288 + i2 * 32 + (3 + 8 * i0o)] = B[(3 + 8 * i0o) * 81 + i1 * 9 + i2];
    }
  }
  for (int_fast32_t i1 = 0; i1 < 9; i1++) {
    for (int_fast32_t i2 = 0; i2 < 9; i2++) {
      Bt_all[i1 * 288 + i2 * 32 + (4 + 8 * i0o)] = B[(4 + 8 * i0o) * 81 + i1 * 9 + i2];
    }
  }
  for (int_fast32_t i1 = 0; i1 < 9; i1++) {
    for (int_fast32_t i2 = 0; i2 < 9; i2++) {
      Bt_all[i1 * 288 + i2 * 32 + (5 + 8 * i0o)] = B[(5 + 8 * i0o) * 81 + i1 * 9 + i2];
    }
  }
  for (int_fast32_t i1 = 0; i1 < 9; i1++) {
    for (int_fast32_t i2 = 0; i2 < 9; i2++) {
      Bt_all[i1 * 288 + i2 * 32 + (6 + 8 * i0o)] = B[(6 + 8 * i0o) * 81 + i1 * 9 + i2];
    }
  }
  for (int_fast32_t i1 = 0; i1 < 9; i1++) {
    for (int_fast32_t i2 = 0; i2 < 9; i2++) {
      Bt_all[i1 * 288 + i2 * 32 + (7 + 8 * i0o)] = B[(7 + 8 * i0o) * 81 + i1 * 9 + i2];
    }
  }
}
int8_t *as_ = (int8_t*) ((uint64_t)gemm_malloc (16 * 9 * sizeof(int8_t)));
int8_t *bs = (int8_t*) ((uint64_t)gemm_malloc (16 * 9 * sizeof(int8_t)));
int32_t *sum = (int32_t*) ((uint32_t)gemm_acc_malloc (16 * 9 * sizeof(int32_t)));
static int32_t sum_out[9 * 16];
for (int_fast32_t i1 = 0; i1 < 9; i1++) {
  for (int_fast32_t jo = 0; jo < 2; jo++) {
    gemmini_extended3_config_ld(0, 1.0f, 0, 0);
gemmini_extended_mvin( 0, ((uint64_t) &*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)sum)) + (0)/16))),(16), (9) );
    for (int_fast32_t k1 = 0; k1 < 9; k1++) {
      gemmini_extended3_config_ld(((struct exo_win_2i8c){ &A[(i1) * (729) + (k1) * 9], { 81, 1 } }).strides[0]*1, 1.0f, 0, 1);
gemmini_extended_mvin2( &A[(i1) * (729) + (k1) * 9], ((uint64_t) &*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)as_)) + (0)/16))), (9), (9) );
      gemmini_extended3_config_ld(((struct exo_win_2i8c){ &Bt_all[(k1) * (288) + 16 * jo], { 32, 1 } }).strides[0]*1, 1.0f, 0, 2);
gemmini_extended_mvin3( &Bt_all[(k1) * (288) + 16 * jo], ((uint64_t) &*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)bs)) + (0)/16))), (16), (9) );
      gemmini_extended_config_ex(WS, 0, 0, 1, 0, 0);
gemmini_extended_preload((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)bs)) + (0)/16))), (uint32_t)(&*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)sum)) + (0)/16))) | 0x40000000, (16), (9), (16), (9));
gemmini_extended_compute_preloaded((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)as_)) + (0)/16))), ~((uint32_t)0), (9), (9), 16, 16);
    }
    gemmini_extended_config_st(((struct exo_win_2i32){ &sum_out[0], { 16, 1 } }).strides[0]*4, 0, 1.0f);
gemmini_extended_mvout( ((uint64_t) &sum_out[0]), ((uint32_t) &*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)sum)) + (0)/16)) | 0x20000000), (16), (9) );
    for (int_fast32_t i2 = 0; i2 < 9; i2++) {
      for (int_fast32_t jio = 0; jio < 2; jio++) {
        float val;
        val = (float)(sum_out[i2 * 16 + 8 * jio]);
        val = val * *acc_scale;
        val = val + _select_float((float)val, (float)0.0f, (float)-0.5f, (float)0.5f);
        val = fmaxf((float)(-128.0f), (float)(_select_float((float)val, (float)127.0f, (float)val, (float)127.0f)));
        C[i1 * 288 + i2 * 32 + (8 * jio + 16 * jo)] = (int8_t)(val);
        float val_1;
        val_1 = (float)(sum_out[i2 * 16 + (1 + 8 * jio)]);
        val_1 = val_1 * *acc_scale;
        val_1 = val_1 + _select_float((float)val_1, (float)0.0f, (float)-0.5f, (float)0.5f);
        val_1 = fmaxf((float)(-128.0f), (float)(_select_float((float)val_1, (float)127.0f, (float)val_1, (float)127.0f)));
        C[i1 * 288 + i2 * 32 + (1 + 8 * jio + 16 * jo)] = (int8_t)(val_1);
        float val_2;
        val_2 = (float)(sum_out[i2 * 16 + (2 + 8 * jio)]);
        val_2 = val_2 * *acc_scale;
        val_2 = val_2 + _select_float((float)val_2, (float)0.0f, (float)-0.5f, (float)0.5f);
        val_2 = fmaxf((float)(-128.0f), (float)(_select_float((float)val_2, (float)127.0f, (float)val_2, (float)127.0f)));
        C[i1 * 288 + i2 * 32 + (2 + 8 * jio + 16 * jo)] = (int8_t)(val_2);
        float val_3;
        val_3 = (float)(sum_out[i2 * 16 + (3 + 8 * jio)]);
        val_3 = val_3 * *acc_scale;
        val_3 = val_3 + _select_float((float)val_3, (float)0.0f, (float)-0.5f, (float)0.5f);
        val_3 = fmaxf((float)(-128.0f), (float)(_select_float((float)val_3, (float)127.0f, (float)val_3, (float)127.0f)));
        C[i1 * 288 + i2 * 32 + (3 + 8 * jio + 16 * jo)] = (int8_t)(val_3);
        float val_4;
        val_4 = (float)(sum_out[i2 * 16 + (4 + 8 * jio)]);
        val_4 = val_4 * *acc_scale;
        val_4 = val_4 + _select_float((float)val_4, (float)0.0f, (float)-0.5f, (float)0.5f);
        val_4 = fmaxf((float)(-128.0f), (float)(_select_float((float)val_4, (float)127.0f, (float)val_4, (float)127.0f)));
        C[i1 * 288 + i2 * 32 + (4 + 8 * jio + 16 * jo)] = (int8_t)(val_4);
        float val_5;
        val_5 = (float)(sum_out[i2 * 16 + (5 + 8 * jio)]);
        val_5 = val_5 * *acc_scale;
        val_5 = val_5 + _select_float((float)val_5, (float)0.0f, (float)-0.5f, (float)0.5f);
        val_5 = fmaxf((float)(-128.0f), (float)(_select_float((float)val_5, (float)127.0f, (float)val_5, (float)127.0f)));
        C[i1 * 288 + i2 * 32 + (5 + 8 * jio + 16 * jo)] = (int8_t)(val_5);
        float val_6;
        val_6 = (float)(sum_out[i2 * 16 + (6 + 8 * jio)]);
        val_6 = val_6 * *acc_scale;
        val_6 = val_6 + _select_float((float)val_6, (float)0.0f, (float)-0.5f, (float)0.5f);
        val_6 = fmaxf((float)(-128.0f), (float)(_select_float((float)val_6, (float)127.0f, (float)val_6, (float)127.0f)));
        C[i1 * 288 + i2 * 32 + (6 + 8 * jio + 16 * jo)] = (int8_t)(val_6);
        float val_7;
        val_7 = (float)(sum_out[i2 * 16 + (7 + 8 * jio)]);
        val_7 = val_7 * *acc_scale;
        val_7 = val_7 + _select_float((float)val_7, (float)0.0f, (float)-0.5f, (float)0.5f);
        val_7 = fmaxf((float)(-128.0f), (float)(_select_float((float)val_7, (float)127.0f, (float)val_7, (float)127.0f)));
        C[i1 * 288 + i2 * 32 + (7 + 8 * jio + 16 * jo)] = (int8_t)(val_7);
      }
    }
  }
}
gemm_acc_free((uint32_t)(sum));
gemm_free((uint64_t)(bs));
gemm_free((uint64_t)(as_));
}

// nlb_gemmini(
//     input : i8[9, 9, 64] @DRAM,
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
//     output : i8[9, 9, 64] @DRAM
// )
static void nlb_gemmini( void *ctxt, const int8_t* input, const int8_t* nlb_theta_weights, const int32_t* nlb_theta_bias, const float* nlb_theta_scale, const int8_t* nlb_phi_weights, const int32_t* nlb_phi_bias, const float* nlb_phi_scale, const int8_t* nlb_g_weights, const int32_t* nlb_g_bias, const float* nlb_g_scale, const float* nlb_matmul_scale, const float* softmax_input_scale, const float* softmax_output_scale, const float* nlb_matmul_1_scale, const int8_t* nlb_out_weights, const int32_t* nlb_out_bias, const float* nlb_out_scale, const float* nlb_add_a_scale, const float* nlb_add_b_scale, int8_t* output ) {
static int8_t theta_out[9 * 9 * 32];
nlb_qkv_conv_gemmini(ctxt,input,nlb_theta_weights,nlb_theta_bias,theta_out,nlb_theta_scale);
static int8_t theta_reshaped[32 * 9 * 9];
for (int_fast32_t co = 0; co < 4; co++) {
  for (int_fast32_t h = 0; h < 9; h++) {
    theta_reshaped[8 * co * 81 + h * 9] = theta_out[h * 288 + 8 * co];
    theta_reshaped[8 * co * 81 + h * 9 + 1] = theta_out[h * 288 + 32 + 8 * co];
    theta_reshaped[8 * co * 81 + h * 9 + 2] = theta_out[h * 288 + 64 + 8 * co];
    theta_reshaped[8 * co * 81 + h * 9 + 3] = theta_out[h * 288 + 96 + 8 * co];
    theta_reshaped[8 * co * 81 + h * 9 + 4] = theta_out[h * 288 + 128 + 8 * co];
    theta_reshaped[8 * co * 81 + h * 9 + 5] = theta_out[h * 288 + 160 + 8 * co];
    theta_reshaped[8 * co * 81 + h * 9 + 6] = theta_out[h * 288 + 192 + 8 * co];
    theta_reshaped[8 * co * 81 + h * 9 + 7] = theta_out[h * 288 + 224 + 8 * co];
    theta_reshaped[8 * co * 81 + h * 9 + 8] = theta_out[h * 288 + 256 + 8 * co];
  }
  for (int_fast32_t h = 0; h < 9; h++) {
    theta_reshaped[(1 + 8 * co) * 81 + h * 9] = theta_out[h * 288 + (1 + 8 * co)];
    theta_reshaped[(1 + 8 * co) * 81 + h * 9 + 1] = theta_out[h * 288 + 32 + (1 + 8 * co)];
    theta_reshaped[(1 + 8 * co) * 81 + h * 9 + 2] = theta_out[h * 288 + 64 + (1 + 8 * co)];
    theta_reshaped[(1 + 8 * co) * 81 + h * 9 + 3] = theta_out[h * 288 + 96 + (1 + 8 * co)];
    theta_reshaped[(1 + 8 * co) * 81 + h * 9 + 4] = theta_out[h * 288 + 128 + (1 + 8 * co)];
    theta_reshaped[(1 + 8 * co) * 81 + h * 9 + 5] = theta_out[h * 288 + 160 + (1 + 8 * co)];
    theta_reshaped[(1 + 8 * co) * 81 + h * 9 + 6] = theta_out[h * 288 + 192 + (1 + 8 * co)];
    theta_reshaped[(1 + 8 * co) * 81 + h * 9 + 7] = theta_out[h * 288 + 224 + (1 + 8 * co)];
    theta_reshaped[(1 + 8 * co) * 81 + h * 9 + 8] = theta_out[h * 288 + 256 + (1 + 8 * co)];
  }
  for (int_fast32_t h = 0; h < 9; h++) {
    theta_reshaped[(2 + 8 * co) * 81 + h * 9] = theta_out[h * 288 + (2 + 8 * co)];
    theta_reshaped[(2 + 8 * co) * 81 + h * 9 + 1] = theta_out[h * 288 + 32 + (2 + 8 * co)];
    theta_reshaped[(2 + 8 * co) * 81 + h * 9 + 2] = theta_out[h * 288 + 64 + (2 + 8 * co)];
    theta_reshaped[(2 + 8 * co) * 81 + h * 9 + 3] = theta_out[h * 288 + 96 + (2 + 8 * co)];
    theta_reshaped[(2 + 8 * co) * 81 + h * 9 + 4] = theta_out[h * 288 + 128 + (2 + 8 * co)];
    theta_reshaped[(2 + 8 * co) * 81 + h * 9 + 5] = theta_out[h * 288 + 160 + (2 + 8 * co)];
    theta_reshaped[(2 + 8 * co) * 81 + h * 9 + 6] = theta_out[h * 288 + 192 + (2 + 8 * co)];
    theta_reshaped[(2 + 8 * co) * 81 + h * 9 + 7] = theta_out[h * 288 + 224 + (2 + 8 * co)];
    theta_reshaped[(2 + 8 * co) * 81 + h * 9 + 8] = theta_out[h * 288 + 256 + (2 + 8 * co)];
  }
  for (int_fast32_t h = 0; h < 9; h++) {
    theta_reshaped[(3 + 8 * co) * 81 + h * 9] = theta_out[h * 288 + (3 + 8 * co)];
    theta_reshaped[(3 + 8 * co) * 81 + h * 9 + 1] = theta_out[h * 288 + 32 + (3 + 8 * co)];
    theta_reshaped[(3 + 8 * co) * 81 + h * 9 + 2] = theta_out[h * 288 + 64 + (3 + 8 * co)];
    theta_reshaped[(3 + 8 * co) * 81 + h * 9 + 3] = theta_out[h * 288 + 96 + (3 + 8 * co)];
    theta_reshaped[(3 + 8 * co) * 81 + h * 9 + 4] = theta_out[h * 288 + 128 + (3 + 8 * co)];
    theta_reshaped[(3 + 8 * co) * 81 + h * 9 + 5] = theta_out[h * 288 + 160 + (3 + 8 * co)];
    theta_reshaped[(3 + 8 * co) * 81 + h * 9 + 6] = theta_out[h * 288 + 192 + (3 + 8 * co)];
    theta_reshaped[(3 + 8 * co) * 81 + h * 9 + 7] = theta_out[h * 288 + 224 + (3 + 8 * co)];
    theta_reshaped[(3 + 8 * co) * 81 + h * 9 + 8] = theta_out[h * 288 + 256 + (3 + 8 * co)];
  }
  for (int_fast32_t h = 0; h < 9; h++) {
    theta_reshaped[(4 + 8 * co) * 81 + h * 9] = theta_out[h * 288 + (4 + 8 * co)];
    theta_reshaped[(4 + 8 * co) * 81 + h * 9 + 1] = theta_out[h * 288 + 32 + (4 + 8 * co)];
    theta_reshaped[(4 + 8 * co) * 81 + h * 9 + 2] = theta_out[h * 288 + 64 + (4 + 8 * co)];
    theta_reshaped[(4 + 8 * co) * 81 + h * 9 + 3] = theta_out[h * 288 + 96 + (4 + 8 * co)];
    theta_reshaped[(4 + 8 * co) * 81 + h * 9 + 4] = theta_out[h * 288 + 128 + (4 + 8 * co)];
    theta_reshaped[(4 + 8 * co) * 81 + h * 9 + 5] = theta_out[h * 288 + 160 + (4 + 8 * co)];
    theta_reshaped[(4 + 8 * co) * 81 + h * 9 + 6] = theta_out[h * 288 + 192 + (4 + 8 * co)];
    theta_reshaped[(4 + 8 * co) * 81 + h * 9 + 7] = theta_out[h * 288 + 224 + (4 + 8 * co)];
    theta_reshaped[(4 + 8 * co) * 81 + h * 9 + 8] = theta_out[h * 288 + 256 + (4 + 8 * co)];
  }
  for (int_fast32_t h = 0; h < 9; h++) {
    theta_reshaped[(5 + 8 * co) * 81 + h * 9] = theta_out[h * 288 + (5 + 8 * co)];
    theta_reshaped[(5 + 8 * co) * 81 + h * 9 + 1] = theta_out[h * 288 + 32 + (5 + 8 * co)];
    theta_reshaped[(5 + 8 * co) * 81 + h * 9 + 2] = theta_out[h * 288 + 64 + (5 + 8 * co)];
    theta_reshaped[(5 + 8 * co) * 81 + h * 9 + 3] = theta_out[h * 288 + 96 + (5 + 8 * co)];
    theta_reshaped[(5 + 8 * co) * 81 + h * 9 + 4] = theta_out[h * 288 + 128 + (5 + 8 * co)];
    theta_reshaped[(5 + 8 * co) * 81 + h * 9 + 5] = theta_out[h * 288 + 160 + (5 + 8 * co)];
    theta_reshaped[(5 + 8 * co) * 81 + h * 9 + 6] = theta_out[h * 288 + 192 + (5 + 8 * co)];
    theta_reshaped[(5 + 8 * co) * 81 + h * 9 + 7] = theta_out[h * 288 + 224 + (5 + 8 * co)];
    theta_reshaped[(5 + 8 * co) * 81 + h * 9 + 8] = theta_out[h * 288 + 256 + (5 + 8 * co)];
  }
  for (int_fast32_t h = 0; h < 9; h++) {
    theta_reshaped[(6 + 8 * co) * 81 + h * 9] = theta_out[h * 288 + (6 + 8 * co)];
    theta_reshaped[(6 + 8 * co) * 81 + h * 9 + 1] = theta_out[h * 288 + 32 + (6 + 8 * co)];
    theta_reshaped[(6 + 8 * co) * 81 + h * 9 + 2] = theta_out[h * 288 + 64 + (6 + 8 * co)];
    theta_reshaped[(6 + 8 * co) * 81 + h * 9 + 3] = theta_out[h * 288 + 96 + (6 + 8 * co)];
    theta_reshaped[(6 + 8 * co) * 81 + h * 9 + 4] = theta_out[h * 288 + 128 + (6 + 8 * co)];
    theta_reshaped[(6 + 8 * co) * 81 + h * 9 + 5] = theta_out[h * 288 + 160 + (6 + 8 * co)];
    theta_reshaped[(6 + 8 * co) * 81 + h * 9 + 6] = theta_out[h * 288 + 192 + (6 + 8 * co)];
    theta_reshaped[(6 + 8 * co) * 81 + h * 9 + 7] = theta_out[h * 288 + 224 + (6 + 8 * co)];
    theta_reshaped[(6 + 8 * co) * 81 + h * 9 + 8] = theta_out[h * 288 + 256 + (6 + 8 * co)];
  }
  for (int_fast32_t h = 0; h < 9; h++) {
    theta_reshaped[(7 + 8 * co) * 81 + h * 9] = theta_out[h * 288 + (7 + 8 * co)];
    theta_reshaped[(7 + 8 * co) * 81 + h * 9 + 1] = theta_out[h * 288 + 32 + (7 + 8 * co)];
    theta_reshaped[(7 + 8 * co) * 81 + h * 9 + 2] = theta_out[h * 288 + 64 + (7 + 8 * co)];
    theta_reshaped[(7 + 8 * co) * 81 + h * 9 + 3] = theta_out[h * 288 + 96 + (7 + 8 * co)];
    theta_reshaped[(7 + 8 * co) * 81 + h * 9 + 4] = theta_out[h * 288 + 128 + (7 + 8 * co)];
    theta_reshaped[(7 + 8 * co) * 81 + h * 9 + 5] = theta_out[h * 288 + 160 + (7 + 8 * co)];
    theta_reshaped[(7 + 8 * co) * 81 + h * 9 + 6] = theta_out[h * 288 + 192 + (7 + 8 * co)];
    theta_reshaped[(7 + 8 * co) * 81 + h * 9 + 7] = theta_out[h * 288 + 224 + (7 + 8 * co)];
    theta_reshaped[(7 + 8 * co) * 81 + h * 9 + 8] = theta_out[h * 288 + 256 + (7 + 8 * co)];
  }
}
static int8_t phi_out[9 * 9 * 32];
nlb_qkv_conv_gemmini(ctxt,input,nlb_phi_weights,nlb_phi_bias,phi_out,nlb_phi_scale);
static int8_t phi_reshaped[32 * 9 * 9];
for (int_fast32_t co = 0; co < 4; co++) {
  for (int_fast32_t h = 0; h < 9; h++) {
    phi_reshaped[8 * co * 81 + h * 9] = phi_out[h * 288 + 8 * co];
    phi_reshaped[8 * co * 81 + h * 9 + 1] = phi_out[h * 288 + 32 + 8 * co];
    phi_reshaped[8 * co * 81 + h * 9 + 2] = phi_out[h * 288 + 64 + 8 * co];
    phi_reshaped[8 * co * 81 + h * 9 + 3] = phi_out[h * 288 + 96 + 8 * co];
    phi_reshaped[8 * co * 81 + h * 9 + 4] = phi_out[h * 288 + 128 + 8 * co];
    phi_reshaped[8 * co * 81 + h * 9 + 5] = phi_out[h * 288 + 160 + 8 * co];
    phi_reshaped[8 * co * 81 + h * 9 + 6] = phi_out[h * 288 + 192 + 8 * co];
    phi_reshaped[8 * co * 81 + h * 9 + 7] = phi_out[h * 288 + 224 + 8 * co];
    phi_reshaped[8 * co * 81 + h * 9 + 8] = phi_out[h * 288 + 256 + 8 * co];
  }
  for (int_fast32_t h = 0; h < 9; h++) {
    phi_reshaped[(1 + 8 * co) * 81 + h * 9] = phi_out[h * 288 + (1 + 8 * co)];
    phi_reshaped[(1 + 8 * co) * 81 + h * 9 + 1] = phi_out[h * 288 + 32 + (1 + 8 * co)];
    phi_reshaped[(1 + 8 * co) * 81 + h * 9 + 2] = phi_out[h * 288 + 64 + (1 + 8 * co)];
    phi_reshaped[(1 + 8 * co) * 81 + h * 9 + 3] = phi_out[h * 288 + 96 + (1 + 8 * co)];
    phi_reshaped[(1 + 8 * co) * 81 + h * 9 + 4] = phi_out[h * 288 + 128 + (1 + 8 * co)];
    phi_reshaped[(1 + 8 * co) * 81 + h * 9 + 5] = phi_out[h * 288 + 160 + (1 + 8 * co)];
    phi_reshaped[(1 + 8 * co) * 81 + h * 9 + 6] = phi_out[h * 288 + 192 + (1 + 8 * co)];
    phi_reshaped[(1 + 8 * co) * 81 + h * 9 + 7] = phi_out[h * 288 + 224 + (1 + 8 * co)];
    phi_reshaped[(1 + 8 * co) * 81 + h * 9 + 8] = phi_out[h * 288 + 256 + (1 + 8 * co)];
  }
  for (int_fast32_t h = 0; h < 9; h++) {
    phi_reshaped[(2 + 8 * co) * 81 + h * 9] = phi_out[h * 288 + (2 + 8 * co)];
    phi_reshaped[(2 + 8 * co) * 81 + h * 9 + 1] = phi_out[h * 288 + 32 + (2 + 8 * co)];
    phi_reshaped[(2 + 8 * co) * 81 + h * 9 + 2] = phi_out[h * 288 + 64 + (2 + 8 * co)];
    phi_reshaped[(2 + 8 * co) * 81 + h * 9 + 3] = phi_out[h * 288 + 96 + (2 + 8 * co)];
    phi_reshaped[(2 + 8 * co) * 81 + h * 9 + 4] = phi_out[h * 288 + 128 + (2 + 8 * co)];
    phi_reshaped[(2 + 8 * co) * 81 + h * 9 + 5] = phi_out[h * 288 + 160 + (2 + 8 * co)];
    phi_reshaped[(2 + 8 * co) * 81 + h * 9 + 6] = phi_out[h * 288 + 192 + (2 + 8 * co)];
    phi_reshaped[(2 + 8 * co) * 81 + h * 9 + 7] = phi_out[h * 288 + 224 + (2 + 8 * co)];
    phi_reshaped[(2 + 8 * co) * 81 + h * 9 + 8] = phi_out[h * 288 + 256 + (2 + 8 * co)];
  }
  for (int_fast32_t h = 0; h < 9; h++) {
    phi_reshaped[(3 + 8 * co) * 81 + h * 9] = phi_out[h * 288 + (3 + 8 * co)];
    phi_reshaped[(3 + 8 * co) * 81 + h * 9 + 1] = phi_out[h * 288 + 32 + (3 + 8 * co)];
    phi_reshaped[(3 + 8 * co) * 81 + h * 9 + 2] = phi_out[h * 288 + 64 + (3 + 8 * co)];
    phi_reshaped[(3 + 8 * co) * 81 + h * 9 + 3] = phi_out[h * 288 + 96 + (3 + 8 * co)];
    phi_reshaped[(3 + 8 * co) * 81 + h * 9 + 4] = phi_out[h * 288 + 128 + (3 + 8 * co)];
    phi_reshaped[(3 + 8 * co) * 81 + h * 9 + 5] = phi_out[h * 288 + 160 + (3 + 8 * co)];
    phi_reshaped[(3 + 8 * co) * 81 + h * 9 + 6] = phi_out[h * 288 + 192 + (3 + 8 * co)];
    phi_reshaped[(3 + 8 * co) * 81 + h * 9 + 7] = phi_out[h * 288 + 224 + (3 + 8 * co)];
    phi_reshaped[(3 + 8 * co) * 81 + h * 9 + 8] = phi_out[h * 288 + 256 + (3 + 8 * co)];
  }
  for (int_fast32_t h = 0; h < 9; h++) {
    phi_reshaped[(4 + 8 * co) * 81 + h * 9] = phi_out[h * 288 + (4 + 8 * co)];
    phi_reshaped[(4 + 8 * co) * 81 + h * 9 + 1] = phi_out[h * 288 + 32 + (4 + 8 * co)];
    phi_reshaped[(4 + 8 * co) * 81 + h * 9 + 2] = phi_out[h * 288 + 64 + (4 + 8 * co)];
    phi_reshaped[(4 + 8 * co) * 81 + h * 9 + 3] = phi_out[h * 288 + 96 + (4 + 8 * co)];
    phi_reshaped[(4 + 8 * co) * 81 + h * 9 + 4] = phi_out[h * 288 + 128 + (4 + 8 * co)];
    phi_reshaped[(4 + 8 * co) * 81 + h * 9 + 5] = phi_out[h * 288 + 160 + (4 + 8 * co)];
    phi_reshaped[(4 + 8 * co) * 81 + h * 9 + 6] = phi_out[h * 288 + 192 + (4 + 8 * co)];
    phi_reshaped[(4 + 8 * co) * 81 + h * 9 + 7] = phi_out[h * 288 + 224 + (4 + 8 * co)];
    phi_reshaped[(4 + 8 * co) * 81 + h * 9 + 8] = phi_out[h * 288 + 256 + (4 + 8 * co)];
  }
  for (int_fast32_t h = 0; h < 9; h++) {
    phi_reshaped[(5 + 8 * co) * 81 + h * 9] = phi_out[h * 288 + (5 + 8 * co)];
    phi_reshaped[(5 + 8 * co) * 81 + h * 9 + 1] = phi_out[h * 288 + 32 + (5 + 8 * co)];
    phi_reshaped[(5 + 8 * co) * 81 + h * 9 + 2] = phi_out[h * 288 + 64 + (5 + 8 * co)];
    phi_reshaped[(5 + 8 * co) * 81 + h * 9 + 3] = phi_out[h * 288 + 96 + (5 + 8 * co)];
    phi_reshaped[(5 + 8 * co) * 81 + h * 9 + 4] = phi_out[h * 288 + 128 + (5 + 8 * co)];
    phi_reshaped[(5 + 8 * co) * 81 + h * 9 + 5] = phi_out[h * 288 + 160 + (5 + 8 * co)];
    phi_reshaped[(5 + 8 * co) * 81 + h * 9 + 6] = phi_out[h * 288 + 192 + (5 + 8 * co)];
    phi_reshaped[(5 + 8 * co) * 81 + h * 9 + 7] = phi_out[h * 288 + 224 + (5 + 8 * co)];
    phi_reshaped[(5 + 8 * co) * 81 + h * 9 + 8] = phi_out[h * 288 + 256 + (5 + 8 * co)];
  }
  for (int_fast32_t h = 0; h < 9; h++) {
    phi_reshaped[(6 + 8 * co) * 81 + h * 9] = phi_out[h * 288 + (6 + 8 * co)];
    phi_reshaped[(6 + 8 * co) * 81 + h * 9 + 1] = phi_out[h * 288 + 32 + (6 + 8 * co)];
    phi_reshaped[(6 + 8 * co) * 81 + h * 9 + 2] = phi_out[h * 288 + 64 + (6 + 8 * co)];
    phi_reshaped[(6 + 8 * co) * 81 + h * 9 + 3] = phi_out[h * 288 + 96 + (6 + 8 * co)];
    phi_reshaped[(6 + 8 * co) * 81 + h * 9 + 4] = phi_out[h * 288 + 128 + (6 + 8 * co)];
    phi_reshaped[(6 + 8 * co) * 81 + h * 9 + 5] = phi_out[h * 288 + 160 + (6 + 8 * co)];
    phi_reshaped[(6 + 8 * co) * 81 + h * 9 + 6] = phi_out[h * 288 + 192 + (6 + 8 * co)];
    phi_reshaped[(6 + 8 * co) * 81 + h * 9 + 7] = phi_out[h * 288 + 224 + (6 + 8 * co)];
    phi_reshaped[(6 + 8 * co) * 81 + h * 9 + 8] = phi_out[h * 288 + 256 + (6 + 8 * co)];
  }
  for (int_fast32_t h = 0; h < 9; h++) {
    phi_reshaped[(7 + 8 * co) * 81 + h * 9] = phi_out[h * 288 + (7 + 8 * co)];
    phi_reshaped[(7 + 8 * co) * 81 + h * 9 + 1] = phi_out[h * 288 + 32 + (7 + 8 * co)];
    phi_reshaped[(7 + 8 * co) * 81 + h * 9 + 2] = phi_out[h * 288 + 64 + (7 + 8 * co)];
    phi_reshaped[(7 + 8 * co) * 81 + h * 9 + 3] = phi_out[h * 288 + 96 + (7 + 8 * co)];
    phi_reshaped[(7 + 8 * co) * 81 + h * 9 + 4] = phi_out[h * 288 + 128 + (7 + 8 * co)];
    phi_reshaped[(7 + 8 * co) * 81 + h * 9 + 5] = phi_out[h * 288 + 160 + (7 + 8 * co)];
    phi_reshaped[(7 + 8 * co) * 81 + h * 9 + 6] = phi_out[h * 288 + 192 + (7 + 8 * co)];
    phi_reshaped[(7 + 8 * co) * 81 + h * 9 + 7] = phi_out[h * 288 + 224 + (7 + 8 * co)];
    phi_reshaped[(7 + 8 * co) * 81 + h * 9 + 8] = phi_out[h * 288 + 256 + (7 + 8 * co)];
  }
}
static int8_t g_out[9 * 9 * 32];
nlb_qkv_conv_gemmini(ctxt,input,nlb_g_weights,nlb_g_bias,g_out,nlb_g_scale);
static int8_t g_reshaped[32 * 9 * 9];
for (int_fast32_t co = 0; co < 4; co++) {
  for (int_fast32_t h = 0; h < 9; h++) {
    g_reshaped[8 * co * 81 + h * 9] = g_out[h * 288 + 8 * co];
    g_reshaped[8 * co * 81 + h * 9 + 1] = g_out[h * 288 + 32 + 8 * co];
    g_reshaped[8 * co * 81 + h * 9 + 2] = g_out[h * 288 + 64 + 8 * co];
    g_reshaped[8 * co * 81 + h * 9 + 3] = g_out[h * 288 + 96 + 8 * co];
    g_reshaped[8 * co * 81 + h * 9 + 4] = g_out[h * 288 + 128 + 8 * co];
    g_reshaped[8 * co * 81 + h * 9 + 5] = g_out[h * 288 + 160 + 8 * co];
    g_reshaped[8 * co * 81 + h * 9 + 6] = g_out[h * 288 + 192 + 8 * co];
    g_reshaped[8 * co * 81 + h * 9 + 7] = g_out[h * 288 + 224 + 8 * co];
    g_reshaped[8 * co * 81 + h * 9 + 8] = g_out[h * 288 + 256 + 8 * co];
  }
  for (int_fast32_t h = 0; h < 9; h++) {
    g_reshaped[(1 + 8 * co) * 81 + h * 9] = g_out[h * 288 + (1 + 8 * co)];
    g_reshaped[(1 + 8 * co) * 81 + h * 9 + 1] = g_out[h * 288 + 32 + (1 + 8 * co)];
    g_reshaped[(1 + 8 * co) * 81 + h * 9 + 2] = g_out[h * 288 + 64 + (1 + 8 * co)];
    g_reshaped[(1 + 8 * co) * 81 + h * 9 + 3] = g_out[h * 288 + 96 + (1 + 8 * co)];
    g_reshaped[(1 + 8 * co) * 81 + h * 9 + 4] = g_out[h * 288 + 128 + (1 + 8 * co)];
    g_reshaped[(1 + 8 * co) * 81 + h * 9 + 5] = g_out[h * 288 + 160 + (1 + 8 * co)];
    g_reshaped[(1 + 8 * co) * 81 + h * 9 + 6] = g_out[h * 288 + 192 + (1 + 8 * co)];
    g_reshaped[(1 + 8 * co) * 81 + h * 9 + 7] = g_out[h * 288 + 224 + (1 + 8 * co)];
    g_reshaped[(1 + 8 * co) * 81 + h * 9 + 8] = g_out[h * 288 + 256 + (1 + 8 * co)];
  }
  for (int_fast32_t h = 0; h < 9; h++) {
    g_reshaped[(2 + 8 * co) * 81 + h * 9] = g_out[h * 288 + (2 + 8 * co)];
    g_reshaped[(2 + 8 * co) * 81 + h * 9 + 1] = g_out[h * 288 + 32 + (2 + 8 * co)];
    g_reshaped[(2 + 8 * co) * 81 + h * 9 + 2] = g_out[h * 288 + 64 + (2 + 8 * co)];
    g_reshaped[(2 + 8 * co) * 81 + h * 9 + 3] = g_out[h * 288 + 96 + (2 + 8 * co)];
    g_reshaped[(2 + 8 * co) * 81 + h * 9 + 4] = g_out[h * 288 + 128 + (2 + 8 * co)];
    g_reshaped[(2 + 8 * co) * 81 + h * 9 + 5] = g_out[h * 288 + 160 + (2 + 8 * co)];
    g_reshaped[(2 + 8 * co) * 81 + h * 9 + 6] = g_out[h * 288 + 192 + (2 + 8 * co)];
    g_reshaped[(2 + 8 * co) * 81 + h * 9 + 7] = g_out[h * 288 + 224 + (2 + 8 * co)];
    g_reshaped[(2 + 8 * co) * 81 + h * 9 + 8] = g_out[h * 288 + 256 + (2 + 8 * co)];
  }
  for (int_fast32_t h = 0; h < 9; h++) {
    g_reshaped[(3 + 8 * co) * 81 + h * 9] = g_out[h * 288 + (3 + 8 * co)];
    g_reshaped[(3 + 8 * co) * 81 + h * 9 + 1] = g_out[h * 288 + 32 + (3 + 8 * co)];
    g_reshaped[(3 + 8 * co) * 81 + h * 9 + 2] = g_out[h * 288 + 64 + (3 + 8 * co)];
    g_reshaped[(3 + 8 * co) * 81 + h * 9 + 3] = g_out[h * 288 + 96 + (3 + 8 * co)];
    g_reshaped[(3 + 8 * co) * 81 + h * 9 + 4] = g_out[h * 288 + 128 + (3 + 8 * co)];
    g_reshaped[(3 + 8 * co) * 81 + h * 9 + 5] = g_out[h * 288 + 160 + (3 + 8 * co)];
    g_reshaped[(3 + 8 * co) * 81 + h * 9 + 6] = g_out[h * 288 + 192 + (3 + 8 * co)];
    g_reshaped[(3 + 8 * co) * 81 + h * 9 + 7] = g_out[h * 288 + 224 + (3 + 8 * co)];
    g_reshaped[(3 + 8 * co) * 81 + h * 9 + 8] = g_out[h * 288 + 256 + (3 + 8 * co)];
  }
  for (int_fast32_t h = 0; h < 9; h++) {
    g_reshaped[(4 + 8 * co) * 81 + h * 9] = g_out[h * 288 + (4 + 8 * co)];
    g_reshaped[(4 + 8 * co) * 81 + h * 9 + 1] = g_out[h * 288 + 32 + (4 + 8 * co)];
    g_reshaped[(4 + 8 * co) * 81 + h * 9 + 2] = g_out[h * 288 + 64 + (4 + 8 * co)];
    g_reshaped[(4 + 8 * co) * 81 + h * 9 + 3] = g_out[h * 288 + 96 + (4 + 8 * co)];
    g_reshaped[(4 + 8 * co) * 81 + h * 9 + 4] = g_out[h * 288 + 128 + (4 + 8 * co)];
    g_reshaped[(4 + 8 * co) * 81 + h * 9 + 5] = g_out[h * 288 + 160 + (4 + 8 * co)];
    g_reshaped[(4 + 8 * co) * 81 + h * 9 + 6] = g_out[h * 288 + 192 + (4 + 8 * co)];
    g_reshaped[(4 + 8 * co) * 81 + h * 9 + 7] = g_out[h * 288 + 224 + (4 + 8 * co)];
    g_reshaped[(4 + 8 * co) * 81 + h * 9 + 8] = g_out[h * 288 + 256 + (4 + 8 * co)];
  }
  for (int_fast32_t h = 0; h < 9; h++) {
    g_reshaped[(5 + 8 * co) * 81 + h * 9] = g_out[h * 288 + (5 + 8 * co)];
    g_reshaped[(5 + 8 * co) * 81 + h * 9 + 1] = g_out[h * 288 + 32 + (5 + 8 * co)];
    g_reshaped[(5 + 8 * co) * 81 + h * 9 + 2] = g_out[h * 288 + 64 + (5 + 8 * co)];
    g_reshaped[(5 + 8 * co) * 81 + h * 9 + 3] = g_out[h * 288 + 96 + (5 + 8 * co)];
    g_reshaped[(5 + 8 * co) * 81 + h * 9 + 4] = g_out[h * 288 + 128 + (5 + 8 * co)];
    g_reshaped[(5 + 8 * co) * 81 + h * 9 + 5] = g_out[h * 288 + 160 + (5 + 8 * co)];
    g_reshaped[(5 + 8 * co) * 81 + h * 9 + 6] = g_out[h * 288 + 192 + (5 + 8 * co)];
    g_reshaped[(5 + 8 * co) * 81 + h * 9 + 7] = g_out[h * 288 + 224 + (5 + 8 * co)];
    g_reshaped[(5 + 8 * co) * 81 + h * 9 + 8] = g_out[h * 288 + 256 + (5 + 8 * co)];
  }
  for (int_fast32_t h = 0; h < 9; h++) {
    g_reshaped[(6 + 8 * co) * 81 + h * 9] = g_out[h * 288 + (6 + 8 * co)];
    g_reshaped[(6 + 8 * co) * 81 + h * 9 + 1] = g_out[h * 288 + 32 + (6 + 8 * co)];
    g_reshaped[(6 + 8 * co) * 81 + h * 9 + 2] = g_out[h * 288 + 64 + (6 + 8 * co)];
    g_reshaped[(6 + 8 * co) * 81 + h * 9 + 3] = g_out[h * 288 + 96 + (6 + 8 * co)];
    g_reshaped[(6 + 8 * co) * 81 + h * 9 + 4] = g_out[h * 288 + 128 + (6 + 8 * co)];
    g_reshaped[(6 + 8 * co) * 81 + h * 9 + 5] = g_out[h * 288 + 160 + (6 + 8 * co)];
    g_reshaped[(6 + 8 * co) * 81 + h * 9 + 6] = g_out[h * 288 + 192 + (6 + 8 * co)];
    g_reshaped[(6 + 8 * co) * 81 + h * 9 + 7] = g_out[h * 288 + 224 + (6 + 8 * co)];
    g_reshaped[(6 + 8 * co) * 81 + h * 9 + 8] = g_out[h * 288 + 256 + (6 + 8 * co)];
  }
  for (int_fast32_t h = 0; h < 9; h++) {
    g_reshaped[(7 + 8 * co) * 81 + h * 9] = g_out[h * 288 + (7 + 8 * co)];
    g_reshaped[(7 + 8 * co) * 81 + h * 9 + 1] = g_out[h * 288 + 32 + (7 + 8 * co)];
    g_reshaped[(7 + 8 * co) * 81 + h * 9 + 2] = g_out[h * 288 + 64 + (7 + 8 * co)];
    g_reshaped[(7 + 8 * co) * 81 + h * 9 + 3] = g_out[h * 288 + 96 + (7 + 8 * co)];
    g_reshaped[(7 + 8 * co) * 81 + h * 9 + 4] = g_out[h * 288 + 128 + (7 + 8 * co)];
    g_reshaped[(7 + 8 * co) * 81 + h * 9 + 5] = g_out[h * 288 + 160 + (7 + 8 * co)];
    g_reshaped[(7 + 8 * co) * 81 + h * 9 + 6] = g_out[h * 288 + 192 + (7 + 8 * co)];
    g_reshaped[(7 + 8 * co) * 81 + h * 9 + 7] = g_out[h * 288 + 224 + (7 + 8 * co)];
    g_reshaped[(7 + 8 * co) * 81 + h * 9 + 8] = g_out[h * 288 + 256 + (7 + 8 * co)];
  }
}
static int8_t attention[9 * 9 * 9 * 9];
matmul_transA_gemmini(ctxt,theta_reshaped,phi_reshaped,attention,nlb_matmul_scale);
for (int_fast32_t i1 = 0; i1 < 9; i1++) {
  for (int_fast32_t i2 = 0; i2 < 9; i2++) {
    static float row_float[9 * 9];
    float max_val;
    max_val = -1000000000.0f;
    for (int_fast32_t j1 = 0; j1 < 9; j1++) {
      row_float[j1 * 9] = (float)(attention[i1 * 729 + i2 * 81 + j1 * 9]);
      row_float[j1 * 9] = row_float[j1 * 9] * *softmax_input_scale;
      max_val = fmaxf((float)(max_val), (float)(row_float[j1 * 9]));
      row_float[j1 * 9 + 1] = (float)(attention[i1 * 729 + i2 * 81 + j1 * 9 + 1]);
      row_float[j1 * 9 + 1] = row_float[j1 * 9 + 1] * *softmax_input_scale;
      max_val = fmaxf((float)(max_val), (float)(row_float[j1 * 9 + 1]));
      row_float[j1 * 9 + 2] = (float)(attention[i1 * 729 + i2 * 81 + j1 * 9 + 2]);
      row_float[j1 * 9 + 2] = row_float[j1 * 9 + 2] * *softmax_input_scale;
      max_val = fmaxf((float)(max_val), (float)(row_float[j1 * 9 + 2]));
      row_float[j1 * 9 + 3] = (float)(attention[i1 * 729 + i2 * 81 + j1 * 9 + 3]);
      row_float[j1 * 9 + 3] = row_float[j1 * 9 + 3] * *softmax_input_scale;
      max_val = fmaxf((float)(max_val), (float)(row_float[j1 * 9 + 3]));
      row_float[j1 * 9 + 4] = (float)(attention[i1 * 729 + i2 * 81 + j1 * 9 + 4]);
      row_float[j1 * 9 + 4] = row_float[j1 * 9 + 4] * *softmax_input_scale;
      max_val = fmaxf((float)(max_val), (float)(row_float[j1 * 9 + 4]));
      row_float[j1 * 9 + 5] = (float)(attention[i1 * 729 + i2 * 81 + j1 * 9 + 5]);
      row_float[j1 * 9 + 5] = row_float[j1 * 9 + 5] * *softmax_input_scale;
      max_val = fmaxf((float)(max_val), (float)(row_float[j1 * 9 + 5]));
      row_float[j1 * 9 + 6] = (float)(attention[i1 * 729 + i2 * 81 + j1 * 9 + 6]);
      row_float[j1 * 9 + 6] = row_float[j1 * 9 + 6] * *softmax_input_scale;
      max_val = fmaxf((float)(max_val), (float)(row_float[j1 * 9 + 6]));
      row_float[j1 * 9 + 7] = (float)(attention[i1 * 729 + i2 * 81 + j1 * 9 + 7]);
      row_float[j1 * 9 + 7] = row_float[j1 * 9 + 7] * *softmax_input_scale;
      max_val = fmaxf((float)(max_val), (float)(row_float[j1 * 9 + 7]));
      row_float[j1 * 9 + 8] = (float)(attention[i1 * 729 + i2 * 81 + j1 * 9 + 8]);
      row_float[j1 * 9 + 8] = row_float[j1 * 9 + 8] * *softmax_input_scale;
      max_val = fmaxf((float)(max_val), (float)(row_float[j1 * 9 + 8]));
    }
    float sum_exp;
    sum_exp = 0.0f;
    for (int_fast32_t j1 = 0; j1 < 9; j1++) {
      float x;
      float x2;
      float x3;
      float x4;
      float exp_val;
      x = row_float[j1 * 9] - max_val;
      x2 = x * x;
      x3 = x2 * x;
      x4 = x2 * x2;
      exp_val = 1.0f + x;
      exp_val += x2 * 0.5f;
      exp_val += x3 * 0.166667f;
      exp_val += x4 * 0.041667f;
      exp_val = _select_float((float)exp_val, (float)0.0f, (float)0.0001f, (float)exp_val);
      exp_val = _select_float((float)-8.0f, (float)x, (float)exp_val, (float)0.0001f);
      row_float[j1 * 9] = exp_val;
      sum_exp += exp_val;
      float x_1;
      float x2_1;
      float x3_1;
      float x4_1;
      float exp_val_1;
      x_1 = row_float[j1 * 9 + 1] - max_val;
      x2_1 = x_1 * x_1;
      x3_1 = x2_1 * x_1;
      x4_1 = x2_1 * x2_1;
      exp_val_1 = 1.0f + x_1;
      exp_val_1 += x2_1 * 0.5f;
      exp_val_1 += x3_1 * 0.166667f;
      exp_val_1 += x4_1 * 0.041667f;
      exp_val_1 = _select_float((float)exp_val_1, (float)0.0f, (float)0.0001f, (float)exp_val_1);
      exp_val_1 = _select_float((float)-8.0f, (float)x_1, (float)exp_val_1, (float)0.0001f);
      row_float[j1 * 9 + 1] = exp_val_1;
      sum_exp += exp_val_1;
      float x_2;
      float x2_2;
      float x3_2;
      float x4_2;
      float exp_val_2;
      x_2 = row_float[j1 * 9 + 2] - max_val;
      x2_2 = x_2 * x_2;
      x3_2 = x2_2 * x_2;
      x4_2 = x2_2 * x2_2;
      exp_val_2 = 1.0f + x_2;
      exp_val_2 += x2_2 * 0.5f;
      exp_val_2 += x3_2 * 0.166667f;
      exp_val_2 += x4_2 * 0.041667f;
      exp_val_2 = _select_float((float)exp_val_2, (float)0.0f, (float)0.0001f, (float)exp_val_2);
      exp_val_2 = _select_float((float)-8.0f, (float)x_2, (float)exp_val_2, (float)0.0001f);
      row_float[j1 * 9 + 2] = exp_val_2;
      sum_exp += exp_val_2;
      float x_3;
      float x2_3;
      float x3_3;
      float x4_3;
      float exp_val_3;
      x_3 = row_float[j1 * 9 + 3] - max_val;
      x2_3 = x_3 * x_3;
      x3_3 = x2_3 * x_3;
      x4_3 = x2_3 * x2_3;
      exp_val_3 = 1.0f + x_3;
      exp_val_3 += x2_3 * 0.5f;
      exp_val_3 += x3_3 * 0.166667f;
      exp_val_3 += x4_3 * 0.041667f;
      exp_val_3 = _select_float((float)exp_val_3, (float)0.0f, (float)0.0001f, (float)exp_val_3);
      exp_val_3 = _select_float((float)-8.0f, (float)x_3, (float)exp_val_3, (float)0.0001f);
      row_float[j1 * 9 + 3] = exp_val_3;
      sum_exp += exp_val_3;
      float x_4;
      float x2_4;
      float x3_4;
      float x4_4;
      float exp_val_4;
      x_4 = row_float[j1 * 9 + 4] - max_val;
      x2_4 = x_4 * x_4;
      x3_4 = x2_4 * x_4;
      x4_4 = x2_4 * x2_4;
      exp_val_4 = 1.0f + x_4;
      exp_val_4 += x2_4 * 0.5f;
      exp_val_4 += x3_4 * 0.166667f;
      exp_val_4 += x4_4 * 0.041667f;
      exp_val_4 = _select_float((float)exp_val_4, (float)0.0f, (float)0.0001f, (float)exp_val_4);
      exp_val_4 = _select_float((float)-8.0f, (float)x_4, (float)exp_val_4, (float)0.0001f);
      row_float[j1 * 9 + 4] = exp_val_4;
      sum_exp += exp_val_4;
      float x_5;
      float x2_5;
      float x3_5;
      float x4_5;
      float exp_val_5;
      x_5 = row_float[j1 * 9 + 5] - max_val;
      x2_5 = x_5 * x_5;
      x3_5 = x2_5 * x_5;
      x4_5 = x2_5 * x2_5;
      exp_val_5 = 1.0f + x_5;
      exp_val_5 += x2_5 * 0.5f;
      exp_val_5 += x3_5 * 0.166667f;
      exp_val_5 += x4_5 * 0.041667f;
      exp_val_5 = _select_float((float)exp_val_5, (float)0.0f, (float)0.0001f, (float)exp_val_5);
      exp_val_5 = _select_float((float)-8.0f, (float)x_5, (float)exp_val_5, (float)0.0001f);
      row_float[j1 * 9 + 5] = exp_val_5;
      sum_exp += exp_val_5;
      float x_6;
      float x2_6;
      float x3_6;
      float x4_6;
      float exp_val_6;
      x_6 = row_float[j1 * 9 + 6] - max_val;
      x2_6 = x_6 * x_6;
      x3_6 = x2_6 * x_6;
      x4_6 = x2_6 * x2_6;
      exp_val_6 = 1.0f + x_6;
      exp_val_6 += x2_6 * 0.5f;
      exp_val_6 += x3_6 * 0.166667f;
      exp_val_6 += x4_6 * 0.041667f;
      exp_val_6 = _select_float((float)exp_val_6, (float)0.0f, (float)0.0001f, (float)exp_val_6);
      exp_val_6 = _select_float((float)-8.0f, (float)x_6, (float)exp_val_6, (float)0.0001f);
      row_float[j1 * 9 + 6] = exp_val_6;
      sum_exp += exp_val_6;
      float x_7;
      float x2_7;
      float x3_7;
      float x4_7;
      float exp_val_7;
      x_7 = row_float[j1 * 9 + 7] - max_val;
      x2_7 = x_7 * x_7;
      x3_7 = x2_7 * x_7;
      x4_7 = x2_7 * x2_7;
      exp_val_7 = 1.0f + x_7;
      exp_val_7 += x2_7 * 0.5f;
      exp_val_7 += x3_7 * 0.166667f;
      exp_val_7 += x4_7 * 0.041667f;
      exp_val_7 = _select_float((float)exp_val_7, (float)0.0f, (float)0.0001f, (float)exp_val_7);
      exp_val_7 = _select_float((float)-8.0f, (float)x_7, (float)exp_val_7, (float)0.0001f);
      row_float[j1 * 9 + 7] = exp_val_7;
      sum_exp += exp_val_7;
      float x_8;
      float x2_8;
      float x3_8;
      float x4_8;
      float exp_val_8;
      x_8 = row_float[j1 * 9 + 8] - max_val;
      x2_8 = x_8 * x_8;
      x3_8 = x2_8 * x_8;
      x4_8 = x2_8 * x2_8;
      exp_val_8 = 1.0f + x_8;
      exp_val_8 += x2_8 * 0.5f;
      exp_val_8 += x3_8 * 0.166667f;
      exp_val_8 += x4_8 * 0.041667f;
      exp_val_8 = _select_float((float)exp_val_8, (float)0.0f, (float)0.0001f, (float)exp_val_8);
      exp_val_8 = _select_float((float)-8.0f, (float)x_8, (float)exp_val_8, (float)0.0001f);
      row_float[j1 * 9 + 8] = exp_val_8;
      sum_exp += exp_val_8;
    }
    for (int_fast32_t j1 = 0; j1 < 9; j1++) {
      float softmax_val;
      float quantized;
      softmax_val = row_float[j1 * 9] / sum_exp;
      quantized = softmax_val / *softmax_output_scale + 0.5f;
      quantized = fmaxf((float)(-128.0f), (float)(_select_float((float)quantized, (float)127.0f, (float)quantized, (float)127.0f)));
      attention[i1 * 729 + i2 * 81 + j1 * 9] = (int8_t)(quantized);
      float softmax_val_1;
      float quantized_1;
      softmax_val_1 = row_float[j1 * 9 + 1] / sum_exp;
      quantized_1 = softmax_val_1 / *softmax_output_scale + 0.5f;
      quantized_1 = fmaxf((float)(-128.0f), (float)(_select_float((float)quantized_1, (float)127.0f, (float)quantized_1, (float)127.0f)));
      attention[i1 * 729 + i2 * 81 + j1 * 9 + 1] = (int8_t)(quantized_1);
      float softmax_val_2;
      float quantized_2;
      softmax_val_2 = row_float[j1 * 9 + 2] / sum_exp;
      quantized_2 = softmax_val_2 / *softmax_output_scale + 0.5f;
      quantized_2 = fmaxf((float)(-128.0f), (float)(_select_float((float)quantized_2, (float)127.0f, (float)quantized_2, (float)127.0f)));
      attention[i1 * 729 + i2 * 81 + j1 * 9 + 2] = (int8_t)(quantized_2);
      float softmax_val_3;
      float quantized_3;
      softmax_val_3 = row_float[j1 * 9 + 3] / sum_exp;
      quantized_3 = softmax_val_3 / *softmax_output_scale + 0.5f;
      quantized_3 = fmaxf((float)(-128.0f), (float)(_select_float((float)quantized_3, (float)127.0f, (float)quantized_3, (float)127.0f)));
      attention[i1 * 729 + i2 * 81 + j1 * 9 + 3] = (int8_t)(quantized_3);
      float softmax_val_4;
      float quantized_4;
      softmax_val_4 = row_float[j1 * 9 + 4] / sum_exp;
      quantized_4 = softmax_val_4 / *softmax_output_scale + 0.5f;
      quantized_4 = fmaxf((float)(-128.0f), (float)(_select_float((float)quantized_4, (float)127.0f, (float)quantized_4, (float)127.0f)));
      attention[i1 * 729 + i2 * 81 + j1 * 9 + 4] = (int8_t)(quantized_4);
      float softmax_val_5;
      float quantized_5;
      softmax_val_5 = row_float[j1 * 9 + 5] / sum_exp;
      quantized_5 = softmax_val_5 / *softmax_output_scale + 0.5f;
      quantized_5 = fmaxf((float)(-128.0f), (float)(_select_float((float)quantized_5, (float)127.0f, (float)quantized_5, (float)127.0f)));
      attention[i1 * 729 + i2 * 81 + j1 * 9 + 5] = (int8_t)(quantized_5);
      float softmax_val_6;
      float quantized_6;
      softmax_val_6 = row_float[j1 * 9 + 6] / sum_exp;
      quantized_6 = softmax_val_6 / *softmax_output_scale + 0.5f;
      quantized_6 = fmaxf((float)(-128.0f), (float)(_select_float((float)quantized_6, (float)127.0f, (float)quantized_6, (float)127.0f)));
      attention[i1 * 729 + i2 * 81 + j1 * 9 + 6] = (int8_t)(quantized_6);
      float softmax_val_7;
      float quantized_7;
      softmax_val_7 = row_float[j1 * 9 + 7] / sum_exp;
      quantized_7 = softmax_val_7 / *softmax_output_scale + 0.5f;
      quantized_7 = fmaxf((float)(-128.0f), (float)(_select_float((float)quantized_7, (float)127.0f, (float)quantized_7, (float)127.0f)));
      attention[i1 * 729 + i2 * 81 + j1 * 9 + 7] = (int8_t)(quantized_7);
      float softmax_val_8;
      float quantized_8;
      softmax_val_8 = row_float[j1 * 9 + 8] / sum_exp;
      quantized_8 = softmax_val_8 / *softmax_output_scale + 0.5f;
      quantized_8 = fmaxf((float)(-128.0f), (float)(_select_float((float)quantized_8, (float)127.0f, (float)quantized_8, (float)127.0f)));
      attention[i1 * 729 + i2 * 81 + j1 * 9 + 8] = (int8_t)(quantized_8);
    }
  }
}
static int8_t attended[9 * 9 * 32];
matmul_transB_gemmini(ctxt,attention,g_reshaped,attended,nlb_matmul_1_scale);
static int8_t tpg_output[9 * 9 * 32];
for (int_fast32_t co = 0; co < 4; co++) {
  for (int_fast32_t h = 0; h < 9; h++) {
    tpg_output[h * 288 + 8 * co] = attended[h * 288 + 8 * co];
    tpg_output[h * 288 + 32 + 8 * co] = attended[h * 288 + 32 + 8 * co];
    tpg_output[h * 288 + 64 + 8 * co] = attended[h * 288 + 64 + 8 * co];
    tpg_output[h * 288 + 96 + 8 * co] = attended[h * 288 + 96 + 8 * co];
    tpg_output[h * 288 + 128 + 8 * co] = attended[h * 288 + 128 + 8 * co];
    tpg_output[h * 288 + 160 + 8 * co] = attended[h * 288 + 160 + 8 * co];
    tpg_output[h * 288 + 192 + 8 * co] = attended[h * 288 + 192 + 8 * co];
    tpg_output[h * 288 + 224 + 8 * co] = attended[h * 288 + 224 + 8 * co];
    tpg_output[h * 288 + 256 + 8 * co] = attended[h * 288 + 256 + 8 * co];
  }
  for (int_fast32_t h = 0; h < 9; h++) {
    tpg_output[h * 288 + (1 + 8 * co)] = attended[h * 288 + (1 + 8 * co)];
    tpg_output[h * 288 + 32 + (1 + 8 * co)] = attended[h * 288 + 32 + (1 + 8 * co)];
    tpg_output[h * 288 + 64 + (1 + 8 * co)] = attended[h * 288 + 64 + (1 + 8 * co)];
    tpg_output[h * 288 + 96 + (1 + 8 * co)] = attended[h * 288 + 96 + (1 + 8 * co)];
    tpg_output[h * 288 + 128 + (1 + 8 * co)] = attended[h * 288 + 128 + (1 + 8 * co)];
    tpg_output[h * 288 + 160 + (1 + 8 * co)] = attended[h * 288 + 160 + (1 + 8 * co)];
    tpg_output[h * 288 + 192 + (1 + 8 * co)] = attended[h * 288 + 192 + (1 + 8 * co)];
    tpg_output[h * 288 + 224 + (1 + 8 * co)] = attended[h * 288 + 224 + (1 + 8 * co)];
    tpg_output[h * 288 + 256 + (1 + 8 * co)] = attended[h * 288 + 256 + (1 + 8 * co)];
  }
  for (int_fast32_t h = 0; h < 9; h++) {
    tpg_output[h * 288 + (2 + 8 * co)] = attended[h * 288 + (2 + 8 * co)];
    tpg_output[h * 288 + 32 + (2 + 8 * co)] = attended[h * 288 + 32 + (2 + 8 * co)];
    tpg_output[h * 288 + 64 + (2 + 8 * co)] = attended[h * 288 + 64 + (2 + 8 * co)];
    tpg_output[h * 288 + 96 + (2 + 8 * co)] = attended[h * 288 + 96 + (2 + 8 * co)];
    tpg_output[h * 288 + 128 + (2 + 8 * co)] = attended[h * 288 + 128 + (2 + 8 * co)];
    tpg_output[h * 288 + 160 + (2 + 8 * co)] = attended[h * 288 + 160 + (2 + 8 * co)];
    tpg_output[h * 288 + 192 + (2 + 8 * co)] = attended[h * 288 + 192 + (2 + 8 * co)];
    tpg_output[h * 288 + 224 + (2 + 8 * co)] = attended[h * 288 + 224 + (2 + 8 * co)];
    tpg_output[h * 288 + 256 + (2 + 8 * co)] = attended[h * 288 + 256 + (2 + 8 * co)];
  }
  for (int_fast32_t h = 0; h < 9; h++) {
    tpg_output[h * 288 + (3 + 8 * co)] = attended[h * 288 + (3 + 8 * co)];
    tpg_output[h * 288 + 32 + (3 + 8 * co)] = attended[h * 288 + 32 + (3 + 8 * co)];
    tpg_output[h * 288 + 64 + (3 + 8 * co)] = attended[h * 288 + 64 + (3 + 8 * co)];
    tpg_output[h * 288 + 96 + (3 + 8 * co)] = attended[h * 288 + 96 + (3 + 8 * co)];
    tpg_output[h * 288 + 128 + (3 + 8 * co)] = attended[h * 288 + 128 + (3 + 8 * co)];
    tpg_output[h * 288 + 160 + (3 + 8 * co)] = attended[h * 288 + 160 + (3 + 8 * co)];
    tpg_output[h * 288 + 192 + (3 + 8 * co)] = attended[h * 288 + 192 + (3 + 8 * co)];
    tpg_output[h * 288 + 224 + (3 + 8 * co)] = attended[h * 288 + 224 + (3 + 8 * co)];
    tpg_output[h * 288 + 256 + (3 + 8 * co)] = attended[h * 288 + 256 + (3 + 8 * co)];
  }
  for (int_fast32_t h = 0; h < 9; h++) {
    tpg_output[h * 288 + (4 + 8 * co)] = attended[h * 288 + (4 + 8 * co)];
    tpg_output[h * 288 + 32 + (4 + 8 * co)] = attended[h * 288 + 32 + (4 + 8 * co)];
    tpg_output[h * 288 + 64 + (4 + 8 * co)] = attended[h * 288 + 64 + (4 + 8 * co)];
    tpg_output[h * 288 + 96 + (4 + 8 * co)] = attended[h * 288 + 96 + (4 + 8 * co)];
    tpg_output[h * 288 + 128 + (4 + 8 * co)] = attended[h * 288 + 128 + (4 + 8 * co)];
    tpg_output[h * 288 + 160 + (4 + 8 * co)] = attended[h * 288 + 160 + (4 + 8 * co)];
    tpg_output[h * 288 + 192 + (4 + 8 * co)] = attended[h * 288 + 192 + (4 + 8 * co)];
    tpg_output[h * 288 + 224 + (4 + 8 * co)] = attended[h * 288 + 224 + (4 + 8 * co)];
    tpg_output[h * 288 + 256 + (4 + 8 * co)] = attended[h * 288 + 256 + (4 + 8 * co)];
  }
  for (int_fast32_t h = 0; h < 9; h++) {
    tpg_output[h * 288 + (5 + 8 * co)] = attended[h * 288 + (5 + 8 * co)];
    tpg_output[h * 288 + 32 + (5 + 8 * co)] = attended[h * 288 + 32 + (5 + 8 * co)];
    tpg_output[h * 288 + 64 + (5 + 8 * co)] = attended[h * 288 + 64 + (5 + 8 * co)];
    tpg_output[h * 288 + 96 + (5 + 8 * co)] = attended[h * 288 + 96 + (5 + 8 * co)];
    tpg_output[h * 288 + 128 + (5 + 8 * co)] = attended[h * 288 + 128 + (5 + 8 * co)];
    tpg_output[h * 288 + 160 + (5 + 8 * co)] = attended[h * 288 + 160 + (5 + 8 * co)];
    tpg_output[h * 288 + 192 + (5 + 8 * co)] = attended[h * 288 + 192 + (5 + 8 * co)];
    tpg_output[h * 288 + 224 + (5 + 8 * co)] = attended[h * 288 + 224 + (5 + 8 * co)];
    tpg_output[h * 288 + 256 + (5 + 8 * co)] = attended[h * 288 + 256 + (5 + 8 * co)];
  }
  for (int_fast32_t h = 0; h < 9; h++) {
    tpg_output[h * 288 + (6 + 8 * co)] = attended[h * 288 + (6 + 8 * co)];
    tpg_output[h * 288 + 32 + (6 + 8 * co)] = attended[h * 288 + 32 + (6 + 8 * co)];
    tpg_output[h * 288 + 64 + (6 + 8 * co)] = attended[h * 288 + 64 + (6 + 8 * co)];
    tpg_output[h * 288 + 96 + (6 + 8 * co)] = attended[h * 288 + 96 + (6 + 8 * co)];
    tpg_output[h * 288 + 128 + (6 + 8 * co)] = attended[h * 288 + 128 + (6 + 8 * co)];
    tpg_output[h * 288 + 160 + (6 + 8 * co)] = attended[h * 288 + 160 + (6 + 8 * co)];
    tpg_output[h * 288 + 192 + (6 + 8 * co)] = attended[h * 288 + 192 + (6 + 8 * co)];
    tpg_output[h * 288 + 224 + (6 + 8 * co)] = attended[h * 288 + 224 + (6 + 8 * co)];
    tpg_output[h * 288 + 256 + (6 + 8 * co)] = attended[h * 288 + 256 + (6 + 8 * co)];
  }
  for (int_fast32_t h = 0; h < 9; h++) {
    tpg_output[h * 288 + (7 + 8 * co)] = attended[h * 288 + (7 + 8 * co)];
    tpg_output[h * 288 + 32 + (7 + 8 * co)] = attended[h * 288 + 32 + (7 + 8 * co)];
    tpg_output[h * 288 + 64 + (7 + 8 * co)] = attended[h * 288 + 64 + (7 + 8 * co)];
    tpg_output[h * 288 + 96 + (7 + 8 * co)] = attended[h * 288 + 96 + (7 + 8 * co)];
    tpg_output[h * 288 + 128 + (7 + 8 * co)] = attended[h * 288 + 128 + (7 + 8 * co)];
    tpg_output[h * 288 + 160 + (7 + 8 * co)] = attended[h * 288 + 160 + (7 + 8 * co)];
    tpg_output[h * 288 + 192 + (7 + 8 * co)] = attended[h * 288 + 192 + (7 + 8 * co)];
    tpg_output[h * 288 + 224 + (7 + 8 * co)] = attended[h * 288 + 224 + (7 + 8 * co)];
    tpg_output[h * 288 + 256 + (7 + 8 * co)] = attended[h * 288 + 256 + (7 + 8 * co)];
  }
}
static int8_t nlb_conv_out[9 * 9 * 64];
nlb_out_conv_gemmini(ctxt,tpg_output,nlb_out_weights,nlb_out_bias,nlb_conv_out,nlb_out_scale);
resadd_opt(ctxt,nlb_add_b_scale,nlb_add_a_scale,input,nlb_conv_out,output);
}

// nlb_out_conv_gemmini(
//     input : i8[9, 9, 32] @DRAM,
//     weights : i8[64, 1, 1, 32] @DRAM,
//     bias : i32[64] @DRAM,
//     output : i8[9, 9, 64] @DRAM,
//     acc_scale : f32 @DRAM
// )
static void nlb_out_conv_gemmini( void *ctxt, const int8_t* input, const int8_t* weights, const int32_t* bias, int8_t* output, const float* acc_scale ) {
static int8_t weights_copy[1 * 1 * 32 * 64];
for (int_fast32_t i0 = 0; i0 < 64; i0++) {
  for (int_fast32_t i3o = 0; i3o < 4; i3o++) {
    weights_copy[8 * i3o * 64 + i0] = weights[i0 * 32 + 8 * i3o];
    weights_copy[(1 + 8 * i3o) * 64 + i0] = weights[i0 * 32 + (1 + 8 * i3o)];
    weights_copy[(2 + 8 * i3o) * 64 + i0] = weights[i0 * 32 + (2 + 8 * i3o)];
    weights_copy[(3 + 8 * i3o) * 64 + i0] = weights[i0 * 32 + (3 + 8 * i3o)];
    weights_copy[(4 + 8 * i3o) * 64 + i0] = weights[i0 * 32 + (4 + 8 * i3o)];
    weights_copy[(5 + 8 * i3o) * 64 + i0] = weights[i0 * 32 + (5 + 8 * i3o)];
    weights_copy[(6 + 8 * i3o) * 64 + i0] = weights[i0 * 32 + (6 + 8 * i3o)];
    weights_copy[(7 + 8 * i3o) * 64 + i0] = weights[i0 * 32 + (7 + 8 * i3o)];
  }
}
int8_t *as_ = (int8_t*) ((uint64_t)gemm_malloc (16 * 9 * sizeof(int8_t)));
int8_t *bs = (int8_t*) ((uint64_t)gemm_malloc (16 * 16 * sizeof(int8_t)));
int32_t *acc = (int32_t*) ((uint32_t)gemm_acc_malloc (16 * 9 * sizeof(int32_t)));
static int32_t sum[9 * 16];
for (int_fast32_t oh = 0; oh < 9; oh++) {
  for (int_fast32_t oco = 0; oco < 4; oco++) {
    for (int_fast32_t ow = 0; ow < 9; ow++) {
      for (int_fast32_t ocio = 0; ocio < 2; ocio++) {
        sum[ow * 16 + 8 * ocio] = bias[8 * ocio + 16 * oco];
        sum[ow * 16 + (1 + 8 * ocio)] = bias[1 + 8 * ocio + 16 * oco];
        sum[ow * 16 + (2 + 8 * ocio)] = bias[2 + 8 * ocio + 16 * oco];
        sum[ow * 16 + (3 + 8 * ocio)] = bias[3 + 8 * ocio + 16 * oco];
        sum[ow * 16 + (4 + 8 * ocio)] = bias[4 + 8 * ocio + 16 * oco];
        sum[ow * 16 + (5 + 8 * ocio)] = bias[5 + 8 * ocio + 16 * oco];
        sum[ow * 16 + (6 + 8 * ocio)] = bias[6 + 8 * ocio + 16 * oco];
        sum[ow * 16 + (7 + 8 * ocio)] = bias[7 + 8 * ocio + 16 * oco];
      }
    }
    gemmini_extended3_config_ld(((struct exo_win_2i32c){ &sum[0], { 16, 1 } }).strides[0]*4, 1.0f, 0, 0);
gemmini_extended_mvin( ((uint64_t) &sum[0]), ((uint32_t) &*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)acc)) + (0)/16))), (16), (9) );
    for (int_fast32_t kh = 0; kh < 1; kh++) {
      for (int_fast32_t kw = 0; kw < 1; kw++) {
        for (int_fast32_t ico = 0; ico < 2; ico++) {
          gemmini_extended3_config_ld(((struct exo_win_2i8c){ &input[(kh + oh) * (288) + (kw) * (32) + 16 * ico], { 32, 1 } }).strides[0]*1, 1.0f, 0, 1);
gemmini_extended_mvin2( &input[(kh + oh) * (288) + (kw) * (32) + 16 * ico], ((uint64_t) &*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)as_)) + (0)/16))), (16), (9) );
          gemmini_extended3_config_ld(((struct exo_win_2i8c){ &weights_copy[(kh) * (2048) + (kw) * (2048) + (16 * ico) * (64) + 16 * oco], { 64, 1 } }).strides[0]*1, 1.0f, 0, 2);
gemmini_extended_mvin3( &weights_copy[(kh) * (2048) + (kw) * (2048) + (16 * ico) * (64) + 16 * oco], ((uint64_t) &*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)bs)) + (0)/16))), (16), (16) );
          gemmini_extended_config_ex(WS, 0, 0, 1, 0, 0);
gemmini_extended_preload((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)bs)) + (0)/16))), (uint32_t)(&*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)acc)) + (0)/16))) | 0x40000000, (16), (16), (16), (9));
gemmini_extended_compute_preloaded((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)as_)) + (0)/16))), ~((uint32_t)0), (16), (9), 16, 16);
        }
      }
    }
    gemmini_extended_config_st(((struct exo_win_2i32){ &sum[0], { 16, 1 } }).strides[0]*4, 0, 1.0f);
gemmini_extended_mvout( ((uint64_t) &sum[0]), ((uint32_t) &*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)acc)) + (0)/16)) | 0x20000000), (16), (9) );
    for (int_fast32_t ow = 0; ow < 9; ow++) {
      for (int_fast32_t ocio = 0; ocio < 2; ocio++) {
        float val;
        val = (float)(sum[ow * 16 + 8 * ocio]);
        val = val * *acc_scale;
        val = val + _select_float((float)val, (float)0.0f, (float)-0.5f, (float)0.5f);
        val = fmaxf((float)(-128.0f), (float)(_select_float((float)val, (float)127.0f, (float)val, (float)127.0f)));
        output[oh * 576 + ow * 64 + (8 * ocio + 16 * oco)] = (int8_t)(val);
        float val_1;
        val_1 = (float)(sum[ow * 16 + (1 + 8 * ocio)]);
        val_1 = val_1 * *acc_scale;
        val_1 = val_1 + _select_float((float)val_1, (float)0.0f, (float)-0.5f, (float)0.5f);
        val_1 = fmaxf((float)(-128.0f), (float)(_select_float((float)val_1, (float)127.0f, (float)val_1, (float)127.0f)));
        output[oh * 576 + ow * 64 + (1 + 8 * ocio + 16 * oco)] = (int8_t)(val_1);
        float val_2;
        val_2 = (float)(sum[ow * 16 + (2 + 8 * ocio)]);
        val_2 = val_2 * *acc_scale;
        val_2 = val_2 + _select_float((float)val_2, (float)0.0f, (float)-0.5f, (float)0.5f);
        val_2 = fmaxf((float)(-128.0f), (float)(_select_float((float)val_2, (float)127.0f, (float)val_2, (float)127.0f)));
        output[oh * 576 + ow * 64 + (2 + 8 * ocio + 16 * oco)] = (int8_t)(val_2);
        float val_3;
        val_3 = (float)(sum[ow * 16 + (3 + 8 * ocio)]);
        val_3 = val_3 * *acc_scale;
        val_3 = val_3 + _select_float((float)val_3, (float)0.0f, (float)-0.5f, (float)0.5f);
        val_3 = fmaxf((float)(-128.0f), (float)(_select_float((float)val_3, (float)127.0f, (float)val_3, (float)127.0f)));
        output[oh * 576 + ow * 64 + (3 + 8 * ocio + 16 * oco)] = (int8_t)(val_3);
        float val_4;
        val_4 = (float)(sum[ow * 16 + (4 + 8 * ocio)]);
        val_4 = val_4 * *acc_scale;
        val_4 = val_4 + _select_float((float)val_4, (float)0.0f, (float)-0.5f, (float)0.5f);
        val_4 = fmaxf((float)(-128.0f), (float)(_select_float((float)val_4, (float)127.0f, (float)val_4, (float)127.0f)));
        output[oh * 576 + ow * 64 + (4 + 8 * ocio + 16 * oco)] = (int8_t)(val_4);
        float val_5;
        val_5 = (float)(sum[ow * 16 + (5 + 8 * ocio)]);
        val_5 = val_5 * *acc_scale;
        val_5 = val_5 + _select_float((float)val_5, (float)0.0f, (float)-0.5f, (float)0.5f);
        val_5 = fmaxf((float)(-128.0f), (float)(_select_float((float)val_5, (float)127.0f, (float)val_5, (float)127.0f)));
        output[oh * 576 + ow * 64 + (5 + 8 * ocio + 16 * oco)] = (int8_t)(val_5);
        float val_6;
        val_6 = (float)(sum[ow * 16 + (6 + 8 * ocio)]);
        val_6 = val_6 * *acc_scale;
        val_6 = val_6 + _select_float((float)val_6, (float)0.0f, (float)-0.5f, (float)0.5f);
        val_6 = fmaxf((float)(-128.0f), (float)(_select_float((float)val_6, (float)127.0f, (float)val_6, (float)127.0f)));
        output[oh * 576 + ow * 64 + (6 + 8 * ocio + 16 * oco)] = (int8_t)(val_6);
        float val_7;
        val_7 = (float)(sum[ow * 16 + (7 + 8 * ocio)]);
        val_7 = val_7 * *acc_scale;
        val_7 = val_7 + _select_float((float)val_7, (float)0.0f, (float)-0.5f, (float)0.5f);
        val_7 = fmaxf((float)(-128.0f), (float)(_select_float((float)val_7, (float)127.0f, (float)val_7, (float)127.0f)));
        output[oh * 576 + ow * 64 + (7 + 8 * ocio + 16 * oco)] = (int8_t)(val_7);
      }
    }
  }
}
gemm_acc_free((uint32_t)(acc));
gemm_free((uint64_t)(bs));
gemm_free((uint64_t)(as_));
}

// nlb_qkv_conv_gemmini(
//     input : i8[9, 9, 64] @DRAM,
//     weights : i8[32, 1, 1, 64] @DRAM,
//     bias : i32[32] @DRAM,
//     output : i8[9, 9, 32] @DRAM,
//     acc_scale : f32 @DRAM
// )
static void nlb_qkv_conv_gemmini( void *ctxt, const int8_t* input, const int8_t* weights, const int32_t* bias, int8_t* output, const float* acc_scale ) {
static int8_t weights_copy[1 * 1 * 64 * 32];
for (int_fast32_t i0 = 0; i0 < 32; i0++) {
  for (int_fast32_t i3o = 0; i3o < 8; i3o++) {
    weights_copy[8 * i3o * 32 + i0] = weights[i0 * 64 + 8 * i3o];
    weights_copy[(1 + 8 * i3o) * 32 + i0] = weights[i0 * 64 + (1 + 8 * i3o)];
    weights_copy[(2 + 8 * i3o) * 32 + i0] = weights[i0 * 64 + (2 + 8 * i3o)];
    weights_copy[(3 + 8 * i3o) * 32 + i0] = weights[i0 * 64 + (3 + 8 * i3o)];
    weights_copy[(4 + 8 * i3o) * 32 + i0] = weights[i0 * 64 + (4 + 8 * i3o)];
    weights_copy[(5 + 8 * i3o) * 32 + i0] = weights[i0 * 64 + (5 + 8 * i3o)];
    weights_copy[(6 + 8 * i3o) * 32 + i0] = weights[i0 * 64 + (6 + 8 * i3o)];
    weights_copy[(7 + 8 * i3o) * 32 + i0] = weights[i0 * 64 + (7 + 8 * i3o)];
  }
}
int8_t *as_ = (int8_t*) ((uint64_t)gemm_malloc (16 * 9 * sizeof(int8_t)));
int8_t *bs = (int8_t*) ((uint64_t)gemm_malloc (16 * 16 * sizeof(int8_t)));
int32_t *acc = (int32_t*) ((uint32_t)gemm_acc_malloc (16 * 9 * sizeof(int32_t)));
static int32_t sum[9 * 16];
for (int_fast32_t oh = 0; oh < 9; oh++) {
  for (int_fast32_t oco = 0; oco < 2; oco++) {
    for (int_fast32_t ow = 0; ow < 9; ow++) {
      for (int_fast32_t ocio = 0; ocio < 2; ocio++) {
        sum[ow * 16 + 8 * ocio] = bias[8 * ocio + 16 * oco];
        sum[ow * 16 + (1 + 8 * ocio)] = bias[1 + 8 * ocio + 16 * oco];
        sum[ow * 16 + (2 + 8 * ocio)] = bias[2 + 8 * ocio + 16 * oco];
        sum[ow * 16 + (3 + 8 * ocio)] = bias[3 + 8 * ocio + 16 * oco];
        sum[ow * 16 + (4 + 8 * ocio)] = bias[4 + 8 * ocio + 16 * oco];
        sum[ow * 16 + (5 + 8 * ocio)] = bias[5 + 8 * ocio + 16 * oco];
        sum[ow * 16 + (6 + 8 * ocio)] = bias[6 + 8 * ocio + 16 * oco];
        sum[ow * 16 + (7 + 8 * ocio)] = bias[7 + 8 * ocio + 16 * oco];
      }
    }
    gemmini_extended3_config_ld(((struct exo_win_2i32c){ &sum[0], { 16, 1 } }).strides[0]*4, 1.0f, 0, 0);
gemmini_extended_mvin( ((uint64_t) &sum[0]), ((uint32_t) &*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)acc)) + (0)/16))), (16), (9) );
    for (int_fast32_t kh = 0; kh < 1; kh++) {
      for (int_fast32_t kw = 0; kw < 1; kw++) {
        for (int_fast32_t ico = 0; ico < 4; ico++) {
          gemmini_extended3_config_ld(((struct exo_win_2i8c){ &input[(kh + oh) * (576) + (kw) * (64) + 16 * ico], { 64, 1 } }).strides[0]*1, 1.0f, 0, 1);
gemmini_extended_mvin2( &input[(kh + oh) * (576) + (kw) * (64) + 16 * ico], ((uint64_t) &*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)as_)) + (0)/16))), (16), (9) );
          gemmini_extended3_config_ld(((struct exo_win_2i8c){ &weights_copy[(kh) * (2048) + (kw) * (2048) + (16 * ico) * (32) + 16 * oco], { 32, 1 } }).strides[0]*1, 1.0f, 0, 2);
gemmini_extended_mvin3( &weights_copy[(kh) * (2048) + (kw) * (2048) + (16 * ico) * (32) + 16 * oco], ((uint64_t) &*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)bs)) + (0)/16))), (16), (16) );
          gemmini_extended_config_ex(WS, 0, 0, 1, 0, 0);
gemmini_extended_preload((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)bs)) + (0)/16))), (uint32_t)(&*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)acc)) + (0)/16))) | 0x40000000, (16), (16), (16), (9));
gemmini_extended_compute_preloaded((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)as_)) + (0)/16))), ~((uint32_t)0), (16), (9), 16, 16);
        }
      }
    }
    gemmini_extended_config_st(((struct exo_win_2i32){ &sum[0], { 16, 1 } }).strides[0]*4, 0, 1.0f);
gemmini_extended_mvout( ((uint64_t) &sum[0]), ((uint32_t) &*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)acc)) + (0)/16)) | 0x20000000), (16), (9) );
    for (int_fast32_t ow = 0; ow < 9; ow++) {
      for (int_fast32_t ocio = 0; ocio < 2; ocio++) {
        float val;
        val = (float)(sum[ow * 16 + 8 * ocio]);
        val = val * *acc_scale;
        val = val + _select_float((float)val, (float)0.0f, (float)-0.5f, (float)0.5f);
        val = fmaxf((float)(-128.0f), (float)(_select_float((float)val, (float)127.0f, (float)val, (float)127.0f)));
        output[oh * 288 + ow * 32 + (8 * ocio + 16 * oco)] = (int8_t)(val);
        float val_1;
        val_1 = (float)(sum[ow * 16 + (1 + 8 * ocio)]);
        val_1 = val_1 * *acc_scale;
        val_1 = val_1 + _select_float((float)val_1, (float)0.0f, (float)-0.5f, (float)0.5f);
        val_1 = fmaxf((float)(-128.0f), (float)(_select_float((float)val_1, (float)127.0f, (float)val_1, (float)127.0f)));
        output[oh * 288 + ow * 32 + (1 + 8 * ocio + 16 * oco)] = (int8_t)(val_1);
        float val_2;
        val_2 = (float)(sum[ow * 16 + (2 + 8 * ocio)]);
        val_2 = val_2 * *acc_scale;
        val_2 = val_2 + _select_float((float)val_2, (float)0.0f, (float)-0.5f, (float)0.5f);
        val_2 = fmaxf((float)(-128.0f), (float)(_select_float((float)val_2, (float)127.0f, (float)val_2, (float)127.0f)));
        output[oh * 288 + ow * 32 + (2 + 8 * ocio + 16 * oco)] = (int8_t)(val_2);
        float val_3;
        val_3 = (float)(sum[ow * 16 + (3 + 8 * ocio)]);
        val_3 = val_3 * *acc_scale;
        val_3 = val_3 + _select_float((float)val_3, (float)0.0f, (float)-0.5f, (float)0.5f);
        val_3 = fmaxf((float)(-128.0f), (float)(_select_float((float)val_3, (float)127.0f, (float)val_3, (float)127.0f)));
        output[oh * 288 + ow * 32 + (3 + 8 * ocio + 16 * oco)] = (int8_t)(val_3);
        float val_4;
        val_4 = (float)(sum[ow * 16 + (4 + 8 * ocio)]);
        val_4 = val_4 * *acc_scale;
        val_4 = val_4 + _select_float((float)val_4, (float)0.0f, (float)-0.5f, (float)0.5f);
        val_4 = fmaxf((float)(-128.0f), (float)(_select_float((float)val_4, (float)127.0f, (float)val_4, (float)127.0f)));
        output[oh * 288 + ow * 32 + (4 + 8 * ocio + 16 * oco)] = (int8_t)(val_4);
        float val_5;
        val_5 = (float)(sum[ow * 16 + (5 + 8 * ocio)]);
        val_5 = val_5 * *acc_scale;
        val_5 = val_5 + _select_float((float)val_5, (float)0.0f, (float)-0.5f, (float)0.5f);
        val_5 = fmaxf((float)(-128.0f), (float)(_select_float((float)val_5, (float)127.0f, (float)val_5, (float)127.0f)));
        output[oh * 288 + ow * 32 + (5 + 8 * ocio + 16 * oco)] = (int8_t)(val_5);
        float val_6;
        val_6 = (float)(sum[ow * 16 + (6 + 8 * ocio)]);
        val_6 = val_6 * *acc_scale;
        val_6 = val_6 + _select_float((float)val_6, (float)0.0f, (float)-0.5f, (float)0.5f);
        val_6 = fmaxf((float)(-128.0f), (float)(_select_float((float)val_6, (float)127.0f, (float)val_6, (float)127.0f)));
        output[oh * 288 + ow * 32 + (6 + 8 * ocio + 16 * oco)] = (int8_t)(val_6);
        float val_7;
        val_7 = (float)(sum[ow * 16 + (7 + 8 * ocio)]);
        val_7 = val_7 * *acc_scale;
        val_7 = val_7 + _select_float((float)val_7, (float)0.0f, (float)-0.5f, (float)0.5f);
        val_7 = fmaxf((float)(-128.0f), (float)(_select_float((float)val_7, (float)127.0f, (float)val_7, (float)127.0f)));
        output[oh * 288 + ow * 32 + (7 + 8 * ocio + 16 * oco)] = (int8_t)(val_7);
      }
    }
  }
}
gemm_acc_free((uint32_t)(acc));
gemm_free((uint64_t)(bs));
gemm_free((uint64_t)(as_));
}

// resadd_opt(
//     A_scale : f32 @DRAM,
//     B_scale : f32 @DRAM,
//     A : i8[9, 9, 64] @DRAM,
//     B : i8[9, 9, 64] @DRAM,
//     C : i8[9, 9, 64] @DRAM
// )
static void resadd_opt( void *ctxt, const float* A_scale, const float* B_scale, const int8_t* A, const int8_t* B, int8_t* C ) {
for (int_fast32_t i1 = 0; i1 < 9; i1++) {
  for (int_fast32_t i2 = 0; i2 < 9; i2++) {
    for (int_fast32_t jo = 0; jo < 8; jo++) {
      float a;
      float b;
      float val;
      a = (float)(A[i1 * 576 + i2 * 64 + 8 * jo]);
      b = (float)(B[i1 * 576 + i2 * 64 + 8 * jo]);
      val = a * *A_scale + b * *B_scale;
      val = val + _select_float((float)val, (float)0.0f, (float)-0.5f, (float)0.5f);
      val = fmaxf((float)(-128.0f), (float)(_select_float((float)val, (float)127.0f, (float)val, (float)127.0f)));
      C[i1 * 576 + i2 * 64 + 8 * jo] = (int8_t)(val);
      float a_1;
      float b_1;
      float val_1;
      a_1 = (float)(A[i1 * 576 + i2 * 64 + (1 + 8 * jo)]);
      b_1 = (float)(B[i1 * 576 + i2 * 64 + (1 + 8 * jo)]);
      val_1 = a_1 * *A_scale + b_1 * *B_scale;
      val_1 = val_1 + _select_float((float)val_1, (float)0.0f, (float)-0.5f, (float)0.5f);
      val_1 = fmaxf((float)(-128.0f), (float)(_select_float((float)val_1, (float)127.0f, (float)val_1, (float)127.0f)));
      C[i1 * 576 + i2 * 64 + (1 + 8 * jo)] = (int8_t)(val_1);
      float a_2;
      float b_2;
      float val_2;
      a_2 = (float)(A[i1 * 576 + i2 * 64 + (2 + 8 * jo)]);
      b_2 = (float)(B[i1 * 576 + i2 * 64 + (2 + 8 * jo)]);
      val_2 = a_2 * *A_scale + b_2 * *B_scale;
      val_2 = val_2 + _select_float((float)val_2, (float)0.0f, (float)-0.5f, (float)0.5f);
      val_2 = fmaxf((float)(-128.0f), (float)(_select_float((float)val_2, (float)127.0f, (float)val_2, (float)127.0f)));
      C[i1 * 576 + i2 * 64 + (2 + 8 * jo)] = (int8_t)(val_2);
      float a_3;
      float b_3;
      float val_3;
      a_3 = (float)(A[i1 * 576 + i2 * 64 + (3 + 8 * jo)]);
      b_3 = (float)(B[i1 * 576 + i2 * 64 + (3 + 8 * jo)]);
      val_3 = a_3 * *A_scale + b_3 * *B_scale;
      val_3 = val_3 + _select_float((float)val_3, (float)0.0f, (float)-0.5f, (float)0.5f);
      val_3 = fmaxf((float)(-128.0f), (float)(_select_float((float)val_3, (float)127.0f, (float)val_3, (float)127.0f)));
      C[i1 * 576 + i2 * 64 + (3 + 8 * jo)] = (int8_t)(val_3);
      float a_4;
      float b_4;
      float val_4;
      a_4 = (float)(A[i1 * 576 + i2 * 64 + (4 + 8 * jo)]);
      b_4 = (float)(B[i1 * 576 + i2 * 64 + (4 + 8 * jo)]);
      val_4 = a_4 * *A_scale + b_4 * *B_scale;
      val_4 = val_4 + _select_float((float)val_4, (float)0.0f, (float)-0.5f, (float)0.5f);
      val_4 = fmaxf((float)(-128.0f), (float)(_select_float((float)val_4, (float)127.0f, (float)val_4, (float)127.0f)));
      C[i1 * 576 + i2 * 64 + (4 + 8 * jo)] = (int8_t)(val_4);
      float a_5;
      float b_5;
      float val_5;
      a_5 = (float)(A[i1 * 576 + i2 * 64 + (5 + 8 * jo)]);
      b_5 = (float)(B[i1 * 576 + i2 * 64 + (5 + 8 * jo)]);
      val_5 = a_5 * *A_scale + b_5 * *B_scale;
      val_5 = val_5 + _select_float((float)val_5, (float)0.0f, (float)-0.5f, (float)0.5f);
      val_5 = fmaxf((float)(-128.0f), (float)(_select_float((float)val_5, (float)127.0f, (float)val_5, (float)127.0f)));
      C[i1 * 576 + i2 * 64 + (5 + 8 * jo)] = (int8_t)(val_5);
      float a_6;
      float b_6;
      float val_6;
      a_6 = (float)(A[i1 * 576 + i2 * 64 + (6 + 8 * jo)]);
      b_6 = (float)(B[i1 * 576 + i2 * 64 + (6 + 8 * jo)]);
      val_6 = a_6 * *A_scale + b_6 * *B_scale;
      val_6 = val_6 + _select_float((float)val_6, (float)0.0f, (float)-0.5f, (float)0.5f);
      val_6 = fmaxf((float)(-128.0f), (float)(_select_float((float)val_6, (float)127.0f, (float)val_6, (float)127.0f)));
      C[i1 * 576 + i2 * 64 + (6 + 8 * jo)] = (int8_t)(val_6);
      float a_7;
      float b_7;
      float val_7;
      a_7 = (float)(A[i1 * 576 + i2 * 64 + (7 + 8 * jo)]);
      b_7 = (float)(B[i1 * 576 + i2 * 64 + (7 + 8 * jo)]);
      val_7 = a_7 * *A_scale + b_7 * *B_scale;
      val_7 = val_7 + _select_float((float)val_7, (float)0.0f, (float)-0.5f, (float)0.5f);
      val_7 = fmaxf((float)(-128.0f), (float)(_select_float((float)val_7, (float)127.0f, (float)val_7, (float)127.0f)));
      C[i1 * 576 + i2 * 64 + (7 + 8 * jo)] = (int8_t)(val_7);
    }
  }
}
}


/* relying on the following instruction..."
st_acc_i32(n,m,src,dst)
gemmini_extended_config_st({dst}.strides[0]*4, 0, 1.0f);
gemmini_extended_mvout( ((uint64_t) &{dst_data}), ((uint32_t) &{src_data} | 0x20000000), {m}, {n} );
*/

/* relying on the following instruction..."
zero_acc_i32(n,m,dst)
gemmini_extended3_config_ld(0, 1.0f, 0, 0);
gemmini_extended_mvin( 0, ((uint64_t) &{dst_data}),{m}, {n} );
*/
