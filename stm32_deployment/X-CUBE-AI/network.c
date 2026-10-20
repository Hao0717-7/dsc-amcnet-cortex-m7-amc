/**
  ******************************************************************************
  * @file    network.c
  * @author  AST Embedded Analytics Research Platform
  * @date    2026-10-17T23:29:17+0800
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */


#include "network.h"
#include "network_data.h"

#include "ai_platform.h"
#include "ai_platform_interface.h"
#include "ai_math_helpers.h"

#include "core_common.h"
#include "core_convert.h"

#include "layers.h"



#undef AI_NET_OBJ_INSTANCE
#define AI_NET_OBJ_INSTANCE g_network
 
#undef AI_NETWORK_MODEL_SIGNATURE
#define AI_NETWORK_MODEL_SIGNATURE     "0xe46731a0d4de825b825999f1ea931c88"

#ifndef AI_TOOLS_REVISION_ID
#define AI_TOOLS_REVISION_ID     ""
#endif

#undef AI_TOOLS_DATE_TIME
#define AI_TOOLS_DATE_TIME   "2026-10-17T23:29:17+0800"

#undef AI_TOOLS_COMPILE_TIME
#define AI_TOOLS_COMPILE_TIME    __DATE__ " " __TIME__

#undef AI_NETWORK_N_BATCHES
#define AI_NETWORK_N_BATCHES         (1)

static ai_ptr g_network_activations_map[1] = AI_C_ARRAY_INIT;
static ai_ptr g_network_weights_map[1] = AI_C_ARRAY_INIT;



/**  Array declarations section  **********************************************/
/* Array#0 */
AI_ARRAY_OBJ_DECLARE(
  input_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 256, AI_STATIC)

/* Array#1 */
AI_ARRAY_OBJ_DECLARE(
  input_QuantizeLinear_Output_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 256, AI_STATIC)

/* Array#2 */
AI_ARRAY_OBJ_DECLARE(
  input_Transpose_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 256, AI_STATIC)

/* Array#3 */
AI_ARRAY_OBJ_DECLARE(
  _Relu_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 3072, AI_STATIC)

/* Array#4 */
AI_ARRAY_OBJ_DECLARE(
  _dsc1_Relu_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 3072, AI_STATIC)

/* Array#5 */
AI_ARRAY_OBJ_DECLARE(
  _dsc1_Relu_1_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1536, AI_STATIC)

/* Array#6 */
AI_ARRAY_OBJ_DECLARE(
  _dsc2_Relu_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1536, AI_STATIC)

/* Array#7 */
AI_ARRAY_OBJ_DECLARE(
  _dsc2_Relu_1_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 768, AI_STATIC)

/* Array#8 */
AI_ARRAY_OBJ_DECLARE(
  _dsc3_Relu_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 768, AI_STATIC)

/* Array#9 */
AI_ARRAY_OBJ_DECLARE(
  _dsc3_Relu_1_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 384, AI_STATIC)

/* Array#10 */
AI_ARRAY_OBJ_DECLARE(
  _dsc3_Relu_1_output_0_0_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 384, AI_STATIC)

/* Array#11 */
AI_ARRAY_OBJ_DECLARE(
  _se_ReduceMean_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 24, AI_STATIC)

/* Array#12 */
AI_ARRAY_OBJ_DECLARE(
  _se_ReduceMean_output_0_Mul_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 24, AI_STATIC)

/* Array#13 */
AI_ARRAY_OBJ_DECLARE(
  _se_ReduceMean_output_0_Mul_0_0__se_Relu_output_0_conversion_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#14 */
AI_ARRAY_OBJ_DECLARE(
  _se_Relu_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 6, AI_STATIC)

/* Array#15 */
AI_ARRAY_OBJ_DECLARE(
  _se_fc2_Gemm_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#16 */
AI_ARRAY_OBJ_DECLARE(
  _se_Sigmoid_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#17 */
AI_ARRAY_OBJ_DECLARE(
  _se_Mul_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 384, AI_STATIC)

/* Array#18 */
AI_ARRAY_OBJ_DECLARE(
  _Reshape_output_0_to_chlast_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 384, AI_STATIC)

/* Array#19 */
AI_ARRAY_OBJ_DECLARE(
  _Reshape_1_output_0_to_chlast_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 384, AI_STATIC)

/* Array#20 */
AI_ARRAY_OBJ_DECLARE(
  _Reshape_1_output_0_to_chfirst_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 384, AI_STATIC)

/* Array#21 */
AI_ARRAY_OBJ_DECLARE(
  _Reshape_1_output_0_to_chfirst_0_0__ReduceMean_output_0_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 384, AI_STATIC)

/* Array#22 */
AI_ARRAY_OBJ_DECLARE(
  _ReduceMean_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 24, AI_STATIC)

/* Array#23 */
AI_ARRAY_OBJ_DECLARE(
  _ReduceMean_output_0_Mul_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 24, AI_STATIC)

/* Array#24 */
AI_ARRAY_OBJ_DECLARE(
  _ReduceMean_output_0_Mul_0_0__classifier_classifier_1_Relu_output_0_conversion_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#25 */
AI_ARRAY_OBJ_DECLARE(
  _classifier_classifier_1_Relu_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#26 */
AI_ARRAY_OBJ_DECLARE(
  logits_QuantizeLinear_Input_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 11, AI_STATIC)

/* Array#27 */
AI_ARRAY_OBJ_DECLARE(
  logits_QuantizeLinear_Input_0_conversion_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 11, AI_STATIC)

/* Array#28 */
AI_ARRAY_OBJ_DECLARE(
  _Relu_output_0_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 48, AI_STATIC)

/* Array#29 */
AI_ARRAY_OBJ_DECLARE(
  _Relu_output_0_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 24, AI_STATIC)

/* Array#30 */
AI_ARRAY_OBJ_DECLARE(
  _dsc1_Relu_output_0_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 72, AI_STATIC)

/* Array#31 */
AI_ARRAY_OBJ_DECLARE(
  _dsc1_Relu_output_0_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 24, AI_STATIC)

/* Array#32 */
AI_ARRAY_OBJ_DECLARE(
  _dsc1_Relu_1_output_0_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 576, AI_STATIC)

/* Array#33 */
AI_ARRAY_OBJ_DECLARE(
  _dsc1_Relu_1_output_0_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 24, AI_STATIC)

/* Array#34 */
AI_ARRAY_OBJ_DECLARE(
  _dsc2_Relu_output_0_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 72, AI_STATIC)

/* Array#35 */
AI_ARRAY_OBJ_DECLARE(
  _dsc2_Relu_output_0_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 24, AI_STATIC)

/* Array#36 */
AI_ARRAY_OBJ_DECLARE(
  _dsc2_Relu_1_output_0_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 576, AI_STATIC)

/* Array#37 */
AI_ARRAY_OBJ_DECLARE(
  _dsc2_Relu_1_output_0_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 24, AI_STATIC)

/* Array#38 */
AI_ARRAY_OBJ_DECLARE(
  _dsc3_Relu_output_0_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 72, AI_STATIC)

/* Array#39 */
AI_ARRAY_OBJ_DECLARE(
  _dsc3_Relu_output_0_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 24, AI_STATIC)

/* Array#40 */
AI_ARRAY_OBJ_DECLARE(
  _dsc3_Relu_1_output_0_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 576, AI_STATIC)

/* Array#41 */
AI_ARRAY_OBJ_DECLARE(
  _dsc3_Relu_1_output_0_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 24, AI_STATIC)

/* Array#42 */
AI_ARRAY_OBJ_DECLARE(
  _se_ReduceMean_output_0_Mul_scale_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 24, AI_STATIC)

/* Array#43 */
AI_ARRAY_OBJ_DECLARE(
  _se_ReduceMean_output_0_Mul_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 24, AI_STATIC)

/* Array#44 */
AI_ARRAY_OBJ_DECLARE(
  _se_Relu_output_0_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 144, AI_STATIC)

/* Array#45 */
AI_ARRAY_OBJ_DECLARE(
  _se_Relu_output_0_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 6, AI_STATIC)

/* Array#46 */
AI_ARRAY_OBJ_DECLARE(
  _se_fc2_Gemm_output_0_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 144, AI_STATIC)

/* Array#47 */
AI_ARRAY_OBJ_DECLARE(
  _se_fc2_Gemm_output_0_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 24, AI_STATIC)

/* Array#48 */
AI_ARRAY_OBJ_DECLARE(
  _classifier_classifier_1_Relu_output_0_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1536, AI_STATIC)

/* Array#49 */
AI_ARRAY_OBJ_DECLARE(
  _classifier_classifier_1_Relu_output_0_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 64, AI_STATIC)

/* Array#50 */
AI_ARRAY_OBJ_DECLARE(
  logits_QuantizeLinear_Input_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 704, AI_STATIC)

/* Array#51 */
AI_ARRAY_OBJ_DECLARE(
  logits_QuantizeLinear_Input_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 11, AI_STATIC)

/* Array#52 */
AI_ARRAY_OBJ_DECLARE(
  _Relu_output_0_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 200, AI_STATIC)

/* Array#53 */
AI_ARRAY_OBJ_DECLARE(
  _dsc1_Relu_output_0_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 217, AI_STATIC)

/* Array#54 */
AI_ARRAY_OBJ_DECLARE(
  _dsc1_Relu_1_output_0_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1344, AI_STATIC)

/* Array#55 */
AI_ARRAY_OBJ_DECLARE(
  _dsc1_Relu_1_output_0_scratch1_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 48, AI_STATIC)

/* Array#56 */
AI_ARRAY_OBJ_DECLARE(
  _dsc2_Relu_output_0_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 217, AI_STATIC)

/* Array#57 */
AI_ARRAY_OBJ_DECLARE(
  _dsc2_Relu_1_output_0_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1344, AI_STATIC)

/* Array#58 */
AI_ARRAY_OBJ_DECLARE(
  _dsc2_Relu_1_output_0_scratch1_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 48, AI_STATIC)

/* Array#59 */
AI_ARRAY_OBJ_DECLARE(
  _dsc3_Relu_output_0_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 217, AI_STATIC)

/* Array#60 */
AI_ARRAY_OBJ_DECLARE(
  _dsc3_Relu_1_output_0_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1344, AI_STATIC)

/* Array#61 */
AI_ARRAY_OBJ_DECLARE(
  _dsc3_Relu_1_output_0_scratch1_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 48, AI_STATIC)

/* Array#62 */
AI_ARRAY_OBJ_DECLARE(
  _se_Relu_output_0_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 24, AI_STATIC)

/* Array#63 */
AI_ARRAY_OBJ_DECLARE(
  _se_fc2_Gemm_output_0_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 6, AI_STATIC)

/* Array#64 */
AI_ARRAY_OBJ_DECLARE(
  _classifier_classifier_1_Relu_output_0_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 24, AI_STATIC)

/* Array#65 */
AI_ARRAY_OBJ_DECLARE(
  logits_QuantizeLinear_Input_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 64, AI_STATIC)

/**  Array metadata declarations section  *************************************/
/* Int quant #0 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_ReduceMean_output_0_Mul_0_0__classifier_classifier_1_Relu_output_0_conversion_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.02770262397825718f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #1 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_Relu_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.020163042470812798f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #2 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_Relu_output_0_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.15178850293159485f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #3 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_Reshape_1_output_0_to_chfirst_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.05636545270681381f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #4 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_Reshape_1_output_0_to_chlast_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.05636545270681381f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #5 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_Reshape_output_0_to_chlast_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.05636545270681381f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #6 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_classifier_classifier_1_Relu_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.043299514800310135f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #7 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_classifier_classifier_1_Relu_output_0_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.014402154833078384f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #8 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_dsc1_Relu_1_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.02371574379503727f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #9 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_dsc1_Relu_1_output_0_scratch1_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.02371574379503727f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #10 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_dsc1_Relu_1_output_0_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00660670455545187f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #11 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_dsc1_Relu_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.025341734290122986f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #12 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_dsc1_Relu_output_0_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.017928512766957283f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #13 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_dsc2_Relu_1_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.016382450237870216f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #14 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_dsc2_Relu_1_output_0_scratch1_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.016382450237870216f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #15 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_dsc2_Relu_1_output_0_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.005915973335504532f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #16 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_dsc2_Relu_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03311488404870033f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #17 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_dsc2_Relu_output_0_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.015195546671748161f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #18 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_dsc3_Relu_1_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.05707817152142525f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #19 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_dsc3_Relu_1_output_0_scratch1_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.05707817152142525f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #20 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_dsc3_Relu_1_output_0_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03374287486076355f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #21 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_dsc3_Relu_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.026817653328180313f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #22 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_dsc3_Relu_output_0_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.056274257600307465f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #23 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_se_Mul_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.05636545270681381f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #24 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_se_ReduceMean_output_0_Mul_0_0__se_Relu_output_0_conversion_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.028052911162376404f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #25 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_se_Relu_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.06134776398539543f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #26 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_se_Relu_output_0_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.01180955395102501f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #27 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_se_Sigmoid_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.003921568859368563f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #28 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_se_fc2_Gemm_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.1363757848739624f),
    AI_PACK_INTQ_ZP(-5)))

/* Int quant #29 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_se_fc2_Gemm_output_0_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.01099038403481245f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #30 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(input_QuantizeLinear_Output_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.002135044429451227f),
    AI_PACK_INTQ_ZP(-4)))

/* Int quant #31 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(input_Transpose_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.002135044429451227f),
    AI_PACK_INTQ_ZP(-4)))

/* Int quant #32 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(logits_QuantizeLinear_Input_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.3500853180885315f),
    AI_PACK_INTQ_ZP(63)))

/* Int quant #33 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(logits_QuantizeLinear_Input_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.027239160612225533f),
    AI_PACK_INTQ_ZP(0)))

/**  Tensor declarations section  *********************************************/
/* Tensor #0 */
AI_TENSOR_OBJ_DECLARE(
  _ReduceMean_output_0_Mul_0_0__classifier_classifier_1_Relu_output_0_conversion_output, AI_STATIC,
  0, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &_ReduceMean_output_0_Mul_0_0__classifier_classifier_1_Relu_output_0_conversion_output_array, &_ReduceMean_output_0_Mul_0_0__classifier_classifier_1_Relu_output_0_conversion_output_array_intq)

/* Tensor #1 */
AI_TENSOR_OBJ_DECLARE(
  _ReduceMean_output_0_Mul_output, AI_STATIC,
  1, 0x0,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 4, 4, 96, 96),
  1, &_ReduceMean_output_0_Mul_output_array, NULL)

/* Tensor #2 */
AI_TENSOR_OBJ_DECLARE(
  _ReduceMean_output_0_output, AI_STATIC,
  2, 0x0,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 4, 4, 96, 96),
  1, &_ReduceMean_output_0_output_array, NULL)

/* Tensor #3 */
AI_TENSOR_OBJ_DECLARE(
  _Relu_output_0_bias, AI_STATIC,
  3, 0x0,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 4, 4, 96, 96),
  1, &_Relu_output_0_bias_array, NULL)

/* Tensor #4 */
AI_TENSOR_OBJ_DECLARE(
  _Relu_output_0_output, AI_STATIC,
  4, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 128), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &_Relu_output_0_output_array, &_Relu_output_0_output_array_intq)

/* Tensor #5 */
AI_TENSOR_OBJ_DECLARE(
  _Relu_output_0_scratch0, AI_STATIC,
  5, 0x0,
  AI_SHAPE_INIT(4, 1, 200, 1, 1), AI_STRIDE_INIT(4, 1, 1, 200, 200),
  1, &_Relu_output_0_scratch0_array, NULL)

/* Tensor #6 */
AI_TENSOR_OBJ_DECLARE(
  _Relu_output_0_weights, AI_STATIC,
  6, 0x1,
  AI_SHAPE_INIT(4, 2, 1, 1, 24), AI_STRIDE_INIT(4, 1, 2, 48, 48),
  1, &_Relu_output_0_weights_array, &_Relu_output_0_weights_array_intq)

/* Tensor #7 */
AI_TENSOR_OBJ_DECLARE(
  _Reshape_1_output_0_to_chfirst_0_0__ReduceMean_output_0_conversion_output, AI_STATIC,
  7, 0x0,
  AI_SHAPE_INIT(4, 1, 24, 1, 16), AI_STRIDE_INIT(4, 4, 4, 96, 96),
  1, &_Reshape_1_output_0_to_chfirst_0_0__ReduceMean_output_0_conversion_output_array, NULL)

/* Tensor #8 */
AI_TENSOR_OBJ_DECLARE(
  _Reshape_1_output_0_to_chfirst_output, AI_STATIC,
  8, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 16), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &_Reshape_1_output_0_to_chfirst_output_array, &_Reshape_1_output_0_to_chfirst_output_array_intq)

/* Tensor #9 */
AI_TENSOR_OBJ_DECLARE(
  _Reshape_1_output_0_to_chlast_output, AI_STATIC,
  9, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 4, 6), AI_STRIDE_INIT(4, 1, 1, 16, 64),
  1, &_Reshape_1_output_0_to_chlast_output_array, &_Reshape_1_output_0_to_chlast_output_array_intq)

/* Tensor #10 */
AI_TENSOR_OBJ_DECLARE(
  _Reshape_1_output_0_to_chlast_output0, AI_STATIC,
  10, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 24), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &_Reshape_1_output_0_to_chlast_output_array, &_Reshape_1_output_0_to_chlast_output_array_intq)

/* Tensor #11 */
AI_TENSOR_OBJ_DECLARE(
  _Reshape_output_0_to_chlast_output, AI_STATIC,
  11, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 24), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &_Reshape_output_0_to_chlast_output_array, &_Reshape_output_0_to_chlast_output_array_intq)

/* Tensor #12 */
AI_TENSOR_OBJ_DECLARE(
  _Reshape_output_0_to_chlast_output0, AI_STATIC,
  12, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 6, 4), AI_STRIDE_INIT(4, 1, 1, 16, 96),
  1, &_Reshape_output_0_to_chlast_output_array, &_Reshape_output_0_to_chlast_output_array_intq)

/* Tensor #13 */
AI_TENSOR_OBJ_DECLARE(
  _classifier_classifier_1_Relu_output_0_bias, AI_STATIC,
  13, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_classifier_classifier_1_Relu_output_0_bias_array, NULL)

/* Tensor #14 */
AI_TENSOR_OBJ_DECLARE(
  _classifier_classifier_1_Relu_output_0_output, AI_STATIC,
  14, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &_classifier_classifier_1_Relu_output_0_output_array, &_classifier_classifier_1_Relu_output_0_output_array_intq)

/* Tensor #15 */
AI_TENSOR_OBJ_DECLARE(
  _classifier_classifier_1_Relu_output_0_scratch0, AI_STATIC,
  15, 0x0,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 2, 2, 48, 48),
  1, &_classifier_classifier_1_Relu_output_0_scratch0_array, NULL)

/* Tensor #16 */
AI_TENSOR_OBJ_DECLARE(
  _classifier_classifier_1_Relu_output_0_weights, AI_STATIC,
  16, 0x1,
  AI_SHAPE_INIT(4, 24, 64, 1, 1), AI_STRIDE_INIT(4, 1, 24, 1536, 1536),
  1, &_classifier_classifier_1_Relu_output_0_weights_array, &_classifier_classifier_1_Relu_output_0_weights_array_intq)

/* Tensor #17 */
AI_TENSOR_OBJ_DECLARE(
  _dsc1_Relu_1_output_0_bias, AI_STATIC,
  17, 0x0,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 4, 4, 96, 96),
  1, &_dsc1_Relu_1_output_0_bias_array, NULL)

/* Tensor #18 */
AI_TENSOR_OBJ_DECLARE(
  _dsc1_Relu_1_output_0_output, AI_STATIC,
  18, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 64), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &_dsc1_Relu_1_output_0_output_array, &_dsc1_Relu_1_output_0_output_array_intq)

/* Tensor #19 */
AI_TENSOR_OBJ_DECLARE(
  _dsc1_Relu_1_output_0_scratch0, AI_STATIC,
  19, 0x0,
  AI_SHAPE_INIT(4, 1, 1344, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1344, 1344),
  1, &_dsc1_Relu_1_output_0_scratch0_array, NULL)

/* Tensor #20 */
AI_TENSOR_OBJ_DECLARE(
  _dsc1_Relu_1_output_0_scratch1, AI_STATIC,
  20, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 2), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &_dsc1_Relu_1_output_0_scratch1_array, &_dsc1_Relu_1_output_0_scratch1_array_intq)

/* Tensor #21 */
AI_TENSOR_OBJ_DECLARE(
  _dsc1_Relu_1_output_0_weights, AI_STATIC,
  21, 0x1,
  AI_SHAPE_INIT(4, 24, 1, 1, 24), AI_STRIDE_INIT(4, 1, 24, 576, 576),
  1, &_dsc1_Relu_1_output_0_weights_array, &_dsc1_Relu_1_output_0_weights_array_intq)

/* Tensor #22 */
AI_TENSOR_OBJ_DECLARE(
  _dsc1_Relu_output_0_bias, AI_STATIC,
  22, 0x0,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 4, 4, 96, 96),
  1, &_dsc1_Relu_output_0_bias_array, NULL)

/* Tensor #23 */
AI_TENSOR_OBJ_DECLARE(
  _dsc1_Relu_output_0_output, AI_STATIC,
  23, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 128), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &_dsc1_Relu_output_0_output_array, &_dsc1_Relu_output_0_output_array_intq)

/* Tensor #24 */
AI_TENSOR_OBJ_DECLARE(
  _dsc1_Relu_output_0_scratch0, AI_STATIC,
  24, 0x0,
  AI_SHAPE_INIT(4, 1, 217, 1, 1), AI_STRIDE_INIT(4, 1, 1, 217, 217),
  1, &_dsc1_Relu_output_0_scratch0_array, NULL)

/* Tensor #25 */
AI_TENSOR_OBJ_DECLARE(
  _dsc1_Relu_output_0_weights, AI_STATIC,
  25, 0x1,
  AI_SHAPE_INIT(4, 24, 1, 3, 1), AI_STRIDE_INIT(4, 1, 24, 24, 24),
  1, &_dsc1_Relu_output_0_weights_array, &_dsc1_Relu_output_0_weights_array_intq)

/* Tensor #26 */
AI_TENSOR_OBJ_DECLARE(
  _dsc2_Relu_1_output_0_bias, AI_STATIC,
  26, 0x0,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 4, 4, 96, 96),
  1, &_dsc2_Relu_1_output_0_bias_array, NULL)

/* Tensor #27 */
AI_TENSOR_OBJ_DECLARE(
  _dsc2_Relu_1_output_0_output, AI_STATIC,
  27, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 32), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &_dsc2_Relu_1_output_0_output_array, &_dsc2_Relu_1_output_0_output_array_intq)

/* Tensor #28 */
AI_TENSOR_OBJ_DECLARE(
  _dsc2_Relu_1_output_0_scratch0, AI_STATIC,
  28, 0x0,
  AI_SHAPE_INIT(4, 1, 1344, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1344, 1344),
  1, &_dsc2_Relu_1_output_0_scratch0_array, NULL)

/* Tensor #29 */
AI_TENSOR_OBJ_DECLARE(
  _dsc2_Relu_1_output_0_scratch1, AI_STATIC,
  29, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 2), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &_dsc2_Relu_1_output_0_scratch1_array, &_dsc2_Relu_1_output_0_scratch1_array_intq)

/* Tensor #30 */
AI_TENSOR_OBJ_DECLARE(
  _dsc2_Relu_1_output_0_weights, AI_STATIC,
  30, 0x1,
  AI_SHAPE_INIT(4, 24, 1, 1, 24), AI_STRIDE_INIT(4, 1, 24, 576, 576),
  1, &_dsc2_Relu_1_output_0_weights_array, &_dsc2_Relu_1_output_0_weights_array_intq)

/* Tensor #31 */
AI_TENSOR_OBJ_DECLARE(
  _dsc2_Relu_output_0_bias, AI_STATIC,
  31, 0x0,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 4, 4, 96, 96),
  1, &_dsc2_Relu_output_0_bias_array, NULL)

/* Tensor #32 */
AI_TENSOR_OBJ_DECLARE(
  _dsc2_Relu_output_0_output, AI_STATIC,
  32, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 64), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &_dsc2_Relu_output_0_output_array, &_dsc2_Relu_output_0_output_array_intq)

/* Tensor #33 */
AI_TENSOR_OBJ_DECLARE(
  _dsc2_Relu_output_0_scratch0, AI_STATIC,
  33, 0x0,
  AI_SHAPE_INIT(4, 1, 217, 1, 1), AI_STRIDE_INIT(4, 1, 1, 217, 217),
  1, &_dsc2_Relu_output_0_scratch0_array, NULL)

/* Tensor #34 */
AI_TENSOR_OBJ_DECLARE(
  _dsc2_Relu_output_0_weights, AI_STATIC,
  34, 0x1,
  AI_SHAPE_INIT(4, 24, 1, 3, 1), AI_STRIDE_INIT(4, 1, 24, 24, 24),
  1, &_dsc2_Relu_output_0_weights_array, &_dsc2_Relu_output_0_weights_array_intq)

/* Tensor #35 */
AI_TENSOR_OBJ_DECLARE(
  _dsc3_Relu_1_output_0_0_conversion_output, AI_STATIC,
  35, 0x0,
  AI_SHAPE_INIT(4, 1, 24, 1, 16), AI_STRIDE_INIT(4, 4, 4, 96, 96),
  1, &_dsc3_Relu_1_output_0_0_conversion_output_array, NULL)

/* Tensor #36 */
AI_TENSOR_OBJ_DECLARE(
  _dsc3_Relu_1_output_0_bias, AI_STATIC,
  36, 0x0,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 4, 4, 96, 96),
  1, &_dsc3_Relu_1_output_0_bias_array, NULL)

/* Tensor #37 */
AI_TENSOR_OBJ_DECLARE(
  _dsc3_Relu_1_output_0_output, AI_STATIC,
  37, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 16), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &_dsc3_Relu_1_output_0_output_array, &_dsc3_Relu_1_output_0_output_array_intq)

/* Tensor #38 */
AI_TENSOR_OBJ_DECLARE(
  _dsc3_Relu_1_output_0_scratch0, AI_STATIC,
  38, 0x0,
  AI_SHAPE_INIT(4, 1, 1344, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1344, 1344),
  1, &_dsc3_Relu_1_output_0_scratch0_array, NULL)

/* Tensor #39 */
AI_TENSOR_OBJ_DECLARE(
  _dsc3_Relu_1_output_0_scratch1, AI_STATIC,
  39, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 2), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &_dsc3_Relu_1_output_0_scratch1_array, &_dsc3_Relu_1_output_0_scratch1_array_intq)

/* Tensor #40 */
AI_TENSOR_OBJ_DECLARE(
  _dsc3_Relu_1_output_0_weights, AI_STATIC,
  40, 0x1,
  AI_SHAPE_INIT(4, 24, 1, 1, 24), AI_STRIDE_INIT(4, 1, 24, 576, 576),
  1, &_dsc3_Relu_1_output_0_weights_array, &_dsc3_Relu_1_output_0_weights_array_intq)

/* Tensor #41 */
AI_TENSOR_OBJ_DECLARE(
  _dsc3_Relu_output_0_bias, AI_STATIC,
  41, 0x0,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 4, 4, 96, 96),
  1, &_dsc3_Relu_output_0_bias_array, NULL)

/* Tensor #42 */
AI_TENSOR_OBJ_DECLARE(
  _dsc3_Relu_output_0_output, AI_STATIC,
  42, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 32), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &_dsc3_Relu_output_0_output_array, &_dsc3_Relu_output_0_output_array_intq)

/* Tensor #43 */
AI_TENSOR_OBJ_DECLARE(
  _dsc3_Relu_output_0_scratch0, AI_STATIC,
  43, 0x0,
  AI_SHAPE_INIT(4, 1, 217, 1, 1), AI_STRIDE_INIT(4, 1, 1, 217, 217),
  1, &_dsc3_Relu_output_0_scratch0_array, NULL)

/* Tensor #44 */
AI_TENSOR_OBJ_DECLARE(
  _dsc3_Relu_output_0_weights, AI_STATIC,
  44, 0x1,
  AI_SHAPE_INIT(4, 24, 1, 3, 1), AI_STRIDE_INIT(4, 1, 24, 24, 24),
  1, &_dsc3_Relu_output_0_weights_array, &_dsc3_Relu_output_0_weights_array_intq)

/* Tensor #45 */
AI_TENSOR_OBJ_DECLARE(
  _se_Mul_output_0_output, AI_STATIC,
  45, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 16), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &_se_Mul_output_0_output_array, &_se_Mul_output_0_output_array_intq)

/* Tensor #46 */
AI_TENSOR_OBJ_DECLARE(
  _se_ReduceMean_output_0_Mul_0_0__se_Relu_output_0_conversion_output, AI_STATIC,
  46, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &_se_ReduceMean_output_0_Mul_0_0__se_Relu_output_0_conversion_output_array, &_se_ReduceMean_output_0_Mul_0_0__se_Relu_output_0_conversion_output_array_intq)

/* Tensor #47 */
AI_TENSOR_OBJ_DECLARE(
  _se_ReduceMean_output_0_Mul_bias, AI_STATIC,
  47, 0x0,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 4, 4, 96, 96),
  1, &_se_ReduceMean_output_0_Mul_bias_array, NULL)

/* Tensor #48 */
AI_TENSOR_OBJ_DECLARE(
  _se_ReduceMean_output_0_Mul_output, AI_STATIC,
  48, 0x0,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 4, 4, 96, 96),
  1, &_se_ReduceMean_output_0_Mul_output_array, NULL)

/* Tensor #49 */
AI_TENSOR_OBJ_DECLARE(
  _se_ReduceMean_output_0_Mul_scale, AI_STATIC,
  49, 0x0,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 4, 4, 96, 96),
  1, &_se_ReduceMean_output_0_Mul_scale_array, NULL)

/* Tensor #50 */
AI_TENSOR_OBJ_DECLARE(
  _se_ReduceMean_output_0_output, AI_STATIC,
  50, 0x0,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 4, 4, 96, 96),
  1, &_se_ReduceMean_output_0_output_array, NULL)

/* Tensor #51 */
AI_TENSOR_OBJ_DECLARE(
  _se_Relu_output_0_bias, AI_STATIC,
  51, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 1, 1), AI_STRIDE_INIT(4, 4, 4, 24, 24),
  1, &_se_Relu_output_0_bias_array, NULL)

/* Tensor #52 */
AI_TENSOR_OBJ_DECLARE(
  _se_Relu_output_0_output, AI_STATIC,
  52, 0x1,
  AI_SHAPE_INIT(4, 1, 6, 1, 1), AI_STRIDE_INIT(4, 1, 1, 6, 6),
  1, &_se_Relu_output_0_output_array, &_se_Relu_output_0_output_array_intq)

/* Tensor #53 */
AI_TENSOR_OBJ_DECLARE(
  _se_Relu_output_0_scratch0, AI_STATIC,
  53, 0x0,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 2, 2, 48, 48),
  1, &_se_Relu_output_0_scratch0_array, NULL)

/* Tensor #54 */
AI_TENSOR_OBJ_DECLARE(
  _se_Relu_output_0_weights, AI_STATIC,
  54, 0x1,
  AI_SHAPE_INIT(4, 24, 6, 1, 1), AI_STRIDE_INIT(4, 1, 24, 144, 144),
  1, &_se_Relu_output_0_weights_array, &_se_Relu_output_0_weights_array_intq)

/* Tensor #55 */
AI_TENSOR_OBJ_DECLARE(
  _se_Sigmoid_output_0_output, AI_STATIC,
  55, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &_se_Sigmoid_output_0_output_array, &_se_Sigmoid_output_0_output_array_intq)

/* Tensor #56 */
AI_TENSOR_OBJ_DECLARE(
  _se_fc2_Gemm_output_0_bias, AI_STATIC,
  56, 0x0,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 4, 4, 96, 96),
  1, &_se_fc2_Gemm_output_0_bias_array, NULL)

/* Tensor #57 */
AI_TENSOR_OBJ_DECLARE(
  _se_fc2_Gemm_output_0_output, AI_STATIC,
  57, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &_se_fc2_Gemm_output_0_output_array, &_se_fc2_Gemm_output_0_output_array_intq)

/* Tensor #58 */
AI_TENSOR_OBJ_DECLARE(
  _se_fc2_Gemm_output_0_scratch0, AI_STATIC,
  58, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 1, 1), AI_STRIDE_INIT(4, 2, 2, 12, 12),
  1, &_se_fc2_Gemm_output_0_scratch0_array, NULL)

/* Tensor #59 */
AI_TENSOR_OBJ_DECLARE(
  _se_fc2_Gemm_output_0_weights, AI_STATIC,
  59, 0x1,
  AI_SHAPE_INIT(4, 6, 24, 1, 1), AI_STRIDE_INIT(4, 1, 6, 144, 144),
  1, &_se_fc2_Gemm_output_0_weights_array, &_se_fc2_Gemm_output_0_weights_array_intq)

/* Tensor #60 */
AI_TENSOR_OBJ_DECLARE(
  input_QuantizeLinear_Output_output, AI_STATIC,
  60, 0x1,
  AI_SHAPE_INIT(4, 1, 128, 1, 2), AI_STRIDE_INIT(4, 1, 1, 128, 128),
  1, &input_QuantizeLinear_Output_output_array, &input_QuantizeLinear_Output_output_array_intq)

/* Tensor #61 */
AI_TENSOR_OBJ_DECLARE(
  input_Transpose_output, AI_STATIC,
  61, 0x1,
  AI_SHAPE_INIT(4, 1, 2, 1, 128), AI_STRIDE_INIT(4, 1, 1, 2, 2),
  1, &input_Transpose_output_array, &input_Transpose_output_array_intq)

/* Tensor #62 */
AI_TENSOR_OBJ_DECLARE(
  input_output, AI_STATIC,
  62, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 1, 2), AI_STRIDE_INIT(4, 4, 4, 512, 512),
  1, &input_output_array, NULL)

/* Tensor #63 */
AI_TENSOR_OBJ_DECLARE(
  logits_QuantizeLinear_Input_0_conversion_output, AI_STATIC,
  63, 0x0,
  AI_SHAPE_INIT(4, 1, 11, 1, 1), AI_STRIDE_INIT(4, 4, 4, 44, 44),
  1, &logits_QuantizeLinear_Input_0_conversion_output_array, NULL)

/* Tensor #64 */
AI_TENSOR_OBJ_DECLARE(
  logits_QuantizeLinear_Input_bias, AI_STATIC,
  64, 0x0,
  AI_SHAPE_INIT(4, 1, 11, 1, 1), AI_STRIDE_INIT(4, 4, 4, 44, 44),
  1, &logits_QuantizeLinear_Input_bias_array, NULL)

/* Tensor #65 */
AI_TENSOR_OBJ_DECLARE(
  logits_QuantizeLinear_Input_output, AI_STATIC,
  65, 0x1,
  AI_SHAPE_INIT(4, 1, 11, 1, 1), AI_STRIDE_INIT(4, 1, 1, 11, 11),
  1, &logits_QuantizeLinear_Input_output_array, &logits_QuantizeLinear_Input_output_array_intq)

/* Tensor #66 */
AI_TENSOR_OBJ_DECLARE(
  logits_QuantizeLinear_Input_scratch0, AI_STATIC,
  66, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 2, 2, 128, 128),
  1, &logits_QuantizeLinear_Input_scratch0_array, NULL)

/* Tensor #67 */
AI_TENSOR_OBJ_DECLARE(
  logits_QuantizeLinear_Input_weights, AI_STATIC,
  67, 0x1,
  AI_SHAPE_INIT(4, 64, 11, 1, 1), AI_STRIDE_INIT(4, 1, 64, 704, 704),
  1, &logits_QuantizeLinear_Input_weights_array, &logits_QuantizeLinear_Input_weights_array_intq)



/**  Layer declarations section  **********************************************/


AI_TENSOR_CHAIN_OBJ_DECLARE(
  logits_QuantizeLinear_Input_0_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &logits_QuantizeLinear_Input_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &logits_QuantizeLinear_Input_0_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  logits_QuantizeLinear_Input_0_conversion_layer, 91,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &logits_QuantizeLinear_Input_0_conversion_chain,
  NULL, &logits_QuantizeLinear_Input_0_conversion_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  logits_QuantizeLinear_Input_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_classifier_classifier_1_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &logits_QuantizeLinear_Input_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &logits_QuantizeLinear_Input_weights, &logits_QuantizeLinear_Input_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &logits_QuantizeLinear_Input_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  logits_QuantizeLinear_Input_layer, 91,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA,
  &logits_QuantizeLinear_Input_chain,
  NULL, &logits_QuantizeLinear_Input_0_conversion_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _classifier_classifier_1_Relu_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_ReduceMean_output_0_Mul_0_0__classifier_classifier_1_Relu_output_0_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_classifier_classifier_1_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_classifier_classifier_1_Relu_output_0_weights, &_classifier_classifier_1_Relu_output_0_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_classifier_classifier_1_Relu_output_0_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  _classifier_classifier_1_Relu_output_0_layer, 88,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA,
  &_classifier_classifier_1_Relu_output_0_chain,
  NULL, &logits_QuantizeLinear_Input_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _ReduceMean_output_0_Mul_0_0__classifier_classifier_1_Relu_output_0_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_ReduceMean_output_0_Mul_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_ReduceMean_output_0_Mul_0_0__classifier_classifier_1_Relu_output_0_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _ReduceMean_output_0_Mul_0_0__classifier_classifier_1_Relu_output_0_conversion_layer, 85,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &_ReduceMean_output_0_Mul_0_0__classifier_classifier_1_Relu_output_0_conversion_chain,
  NULL, &_classifier_classifier_1_Relu_output_0_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _ReduceMean_output_0_Mul_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_ReduceMean_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_ReduceMean_output_0_Mul_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_se_ReduceMean_output_0_Mul_scale, &_se_ReduceMean_output_0_Mul_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _ReduceMean_output_0_Mul_layer, 85,
  BN_TYPE, 0x0, NULL,
  bn, forward_bn,
  &_ReduceMean_output_0_Mul_chain,
  NULL, &_ReduceMean_output_0_Mul_0_0__classifier_classifier_1_Relu_output_0_conversion_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_float _ReduceMean_output_0_neutral_value_data[] = { 0.0f };
AI_ARRAY_OBJ_DECLARE(
    _ReduceMean_output_0_neutral_value, AI_ARRAY_FORMAT_FLOAT,
    _ReduceMean_output_0_neutral_value_data, _ReduceMean_output_0_neutral_value_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  _ReduceMean_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Reshape_1_output_0_to_chfirst_0_0__ReduceMean_output_0_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_ReduceMean_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _ReduceMean_output_0_layer, 85,
  REDUCE_TYPE, 0x0, NULL,
  reduce, forward_reduce,
  &_ReduceMean_output_0_chain,
  NULL, &_ReduceMean_output_0_Mul_layer, AI_STATIC, 
  .operation = ai_sum, 
  .neutral_value = &_ReduceMean_output_0_neutral_value, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _Reshape_1_output_0_to_chfirst_0_0__ReduceMean_output_0_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Reshape_1_output_0_to_chfirst_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Reshape_1_output_0_to_chfirst_0_0__ReduceMean_output_0_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _Reshape_1_output_0_to_chfirst_0_0__ReduceMean_output_0_conversion_layer, 82,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &_Reshape_1_output_0_to_chfirst_0_0__ReduceMean_output_0_conversion_chain,
  NULL, &_ReduceMean_output_0_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _Reshape_1_output_0_to_chfirst_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Reshape_1_output_0_to_chlast_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Reshape_1_output_0_to_chfirst_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _Reshape_1_output_0_to_chfirst_layer, 82,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &_Reshape_1_output_0_to_chfirst_chain,
  NULL, &_Reshape_1_output_0_to_chfirst_0_0__ReduceMean_output_0_conversion_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_HEIGHT, AI_SHAPE_WIDTH, AI_SHAPE_CHANNEL, AI_SHAPE_DEPTH, AI_SHAPE_EXTENSION), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _Reshape_1_output_0_to_chlast_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Reshape_output_0_to_chlast_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Reshape_1_output_0_to_chlast_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _Reshape_1_output_0_to_chlast_layer, 82,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &_Reshape_1_output_0_to_chlast_chain,
  NULL, &_Reshape_1_output_0_to_chfirst_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_CHANNEL, AI_SHAPE_HEIGHT, AI_SHAPE_WIDTH, AI_SHAPE_DEPTH, AI_SHAPE_EXTENSION), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _Reshape_output_0_to_chlast_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_se_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Reshape_output_0_to_chlast_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _Reshape_output_0_to_chlast_layer, 76,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &_Reshape_output_0_to_chlast_chain,
  NULL, &_Reshape_1_output_0_to_chlast_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_HEIGHT, AI_SHAPE_WIDTH, AI_SHAPE_CHANNEL, AI_SHAPE_DEPTH, AI_SHAPE_EXTENSION), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _se_Mul_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_dsc3_Relu_1_output_0_output, &_se_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_se_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _se_Mul_output_0_layer, 73,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &_se_Mul_output_0_chain,
  NULL, &_Reshape_output_0_to_chlast_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)


AI_STATIC_CONST ai_i8 _se_Sigmoid_output_0_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -125, -125, -124, -124, -123, -123, -122, -121, -120, -119, -117, -116, -114, -112, -110, -108, -105, -102, -99, -95, -91, -86, -81, -76, -70, -64, -57, -50, -42, -34, -26, -18, -9, -1, 8, 17, 25, 33, 41, 49, 56, 63, 69, 75, 80, 85, 90, 94, 98, 101, 104, 107, 109, 111, 113, 115, 116, 118, 119, 120, 121, 122, 122, 123, 123, 124, 124, 125, 125, 125, 125, 126, 126, 126, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    _se_Sigmoid_output_0_nl_params, AI_ARRAY_FORMAT_S8,
    _se_Sigmoid_output_0_nl_params_data, _se_Sigmoid_output_0_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  _se_Sigmoid_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_se_fc2_Gemm_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_se_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _se_Sigmoid_output_0_layer, 67,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &_se_Sigmoid_output_0_chain,
  NULL, &_se_Mul_output_0_layer, AI_STATIC, 
  .nl_params = &_se_Sigmoid_output_0_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _se_fc2_Gemm_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_se_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_se_fc2_Gemm_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_se_fc2_Gemm_output_0_weights, &_se_fc2_Gemm_output_0_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_se_fc2_Gemm_output_0_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  _se_fc2_Gemm_output_0_layer, 64,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA,
  &_se_fc2_Gemm_output_0_chain,
  NULL, &_se_Sigmoid_output_0_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _se_Relu_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_se_ReduceMean_output_0_Mul_0_0__se_Relu_output_0_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_se_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_se_Relu_output_0_weights, &_se_Relu_output_0_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_se_Relu_output_0_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  _se_Relu_output_0_layer, 61,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA,
  &_se_Relu_output_0_chain,
  NULL, &_se_fc2_Gemm_output_0_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _se_ReduceMean_output_0_Mul_0_0__se_Relu_output_0_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_se_ReduceMean_output_0_Mul_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_se_ReduceMean_output_0_Mul_0_0__se_Relu_output_0_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _se_ReduceMean_output_0_Mul_0_0__se_Relu_output_0_conversion_layer, 58,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &_se_ReduceMean_output_0_Mul_0_0__se_Relu_output_0_conversion_chain,
  NULL, &_se_Relu_output_0_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _se_ReduceMean_output_0_Mul_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_se_ReduceMean_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_se_ReduceMean_output_0_Mul_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_se_ReduceMean_output_0_Mul_scale, &_se_ReduceMean_output_0_Mul_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _se_ReduceMean_output_0_Mul_layer, 58,
  BN_TYPE, 0x0, NULL,
  bn, forward_bn,
  &_se_ReduceMean_output_0_Mul_chain,
  NULL, &_se_ReduceMean_output_0_Mul_0_0__se_Relu_output_0_conversion_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_float _se_ReduceMean_output_0_neutral_value_data[] = { 0.0f };
AI_ARRAY_OBJ_DECLARE(
    _se_ReduceMean_output_0_neutral_value, AI_ARRAY_FORMAT_FLOAT,
    _se_ReduceMean_output_0_neutral_value_data, _se_ReduceMean_output_0_neutral_value_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  _se_ReduceMean_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_dsc3_Relu_1_output_0_0_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_se_ReduceMean_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _se_ReduceMean_output_0_layer, 58,
  REDUCE_TYPE, 0x0, NULL,
  reduce, forward_reduce,
  &_se_ReduceMean_output_0_chain,
  NULL, &_se_ReduceMean_output_0_Mul_layer, AI_STATIC, 
  .operation = ai_sum, 
  .neutral_value = &_se_ReduceMean_output_0_neutral_value, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _dsc3_Relu_1_output_0_0_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_dsc3_Relu_1_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_dsc3_Relu_1_output_0_0_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _dsc3_Relu_1_output_0_0_conversion_layer, 55,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &_dsc3_Relu_1_output_0_0_conversion_chain,
  NULL, &_se_ReduceMean_output_0_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _dsc3_Relu_1_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_dsc3_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_dsc3_Relu_1_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_dsc3_Relu_1_output_0_weights, &_dsc3_Relu_1_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_dsc3_Relu_1_output_0_scratch0, &_dsc3_Relu_1_output_0_scratch1)
)

AI_LAYER_OBJ_DECLARE(
  _dsc3_Relu_1_output_0_layer, 55,
  OPTIMIZED_CONV2D_TYPE, 0x0, NULL,
  conv2d_nl_pool, forward_conv2d_nl_pool_integer_SSSA,
  &_dsc3_Relu_1_output_0_chain,
  NULL, &_dsc3_Relu_1_output_0_0_conversion_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .pool_size = AI_SHAPE_2D_INIT(1, 2), 
  .pool_stride = AI_SHAPE_2D_INIT(1, 2), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .pool_func = AI_HANDLE_PTR(pool_func_mp_array_integer_INT8), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _dsc3_Relu_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_dsc2_Relu_1_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_dsc3_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_dsc3_Relu_output_0_weights, &_dsc3_Relu_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_dsc3_Relu_output_0_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  _dsc3_Relu_output_0_layer, 49,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_integer_SSSA,
  &_dsc3_Relu_output_0_chain,
  NULL, &_dsc3_Relu_1_output_0_layer, AI_STATIC, 
  .groups = 24, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 0, 1, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _dsc2_Relu_1_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_dsc2_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_dsc2_Relu_1_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_dsc2_Relu_1_output_0_weights, &_dsc2_Relu_1_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_dsc2_Relu_1_output_0_scratch0, &_dsc2_Relu_1_output_0_scratch1)
)

AI_LAYER_OBJ_DECLARE(
  _dsc2_Relu_1_output_0_layer, 46,
  OPTIMIZED_CONV2D_TYPE, 0x0, NULL,
  conv2d_nl_pool, forward_conv2d_nl_pool_integer_SSSA,
  &_dsc2_Relu_1_output_0_chain,
  NULL, &_dsc3_Relu_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .pool_size = AI_SHAPE_2D_INIT(1, 2), 
  .pool_stride = AI_SHAPE_2D_INIT(1, 2), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .pool_func = AI_HANDLE_PTR(pool_func_mp_array_integer_INT8), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _dsc2_Relu_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_dsc1_Relu_1_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_dsc2_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_dsc2_Relu_output_0_weights, &_dsc2_Relu_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_dsc2_Relu_output_0_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  _dsc2_Relu_output_0_layer, 40,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_integer_SSSA,
  &_dsc2_Relu_output_0_chain,
  NULL, &_dsc2_Relu_1_output_0_layer, AI_STATIC, 
  .groups = 24, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 0, 1, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _dsc1_Relu_1_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_dsc1_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_dsc1_Relu_1_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_dsc1_Relu_1_output_0_weights, &_dsc1_Relu_1_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_dsc1_Relu_1_output_0_scratch0, &_dsc1_Relu_1_output_0_scratch1)
)

AI_LAYER_OBJ_DECLARE(
  _dsc1_Relu_1_output_0_layer, 37,
  OPTIMIZED_CONV2D_TYPE, 0x0, NULL,
  conv2d_nl_pool, forward_conv2d_nl_pool_integer_SSSA,
  &_dsc1_Relu_1_output_0_chain,
  NULL, &_dsc2_Relu_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .pool_size = AI_SHAPE_2D_INIT(1, 2), 
  .pool_stride = AI_SHAPE_2D_INIT(1, 2), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .pool_func = AI_HANDLE_PTR(pool_func_mp_array_integer_INT8), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _dsc1_Relu_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_dsc1_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_dsc1_Relu_output_0_weights, &_dsc1_Relu_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_dsc1_Relu_output_0_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  _dsc1_Relu_output_0_layer, 31,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_integer_SSSA,
  &_dsc1_Relu_output_0_chain,
  NULL, &_dsc1_Relu_1_output_0_layer, AI_STATIC, 
  .groups = 24, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 0, 1, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _Relu_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &input_Transpose_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_Relu_output_0_weights, &_Relu_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Relu_output_0_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  _Relu_output_0_layer, 28,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_integer_SSSA,
  &_Relu_output_0_chain,
  NULL, &_dsc1_Relu_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  input_Transpose_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &input_QuantizeLinear_Output_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &input_Transpose_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  input_Transpose_layer, 2,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &input_Transpose_chain,
  NULL, &_Relu_output_0_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_HEIGHT, AI_SHAPE_WIDTH, AI_SHAPE_CHANNEL, AI_SHAPE_DEPTH, AI_SHAPE_EXTENSION), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  input_QuantizeLinear_Output_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &input_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &input_QuantizeLinear_Output_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  input_QuantizeLinear_Output_layer, 8,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &input_QuantizeLinear_Output_chain,
  NULL, &input_Transpose_layer, AI_STATIC, 
)


#if (AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 5804, 1, 1),
    5804, NULL, NULL),
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 4488, 1, 1),
    4488, NULL, NULL),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_IN_NUM, &input_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_OUT_NUM, &logits_QuantizeLinear_Input_0_conversion_output),
  &input_QuantizeLinear_Output_layer, 0x9b96750f, NULL)

#else

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 5804, 1, 1),
      5804, NULL, NULL)
  ),
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 4488, 1, 1),
      4488, NULL, NULL)
  ),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_IN_NUM, &input_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_OUT_NUM, &logits_QuantizeLinear_Input_0_conversion_output),
  &input_QuantizeLinear_Output_layer, 0x9b96750f, NULL)

#endif	/*(AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)*/



/******************************************************************************/
AI_DECLARE_STATIC
ai_bool network_configure_activations(
  ai_network* net_ctx, const ai_network_params* params)
{
  AI_ASSERT(net_ctx)

  if (ai_platform_get_activations_map(g_network_activations_map, 1, params)) {
    /* Updating activations (byte) offsets */
    
    input_QuantizeLinear_Output_output_array.data = AI_PTR(g_network_activations_map[0] + 2912);
    input_QuantizeLinear_Output_output_array.data_start = AI_PTR(g_network_activations_map[0] + 2912);
    input_Transpose_output_array.data = AI_PTR(g_network_activations_map[0] + 3168);
    input_Transpose_output_array.data_start = AI_PTR(g_network_activations_map[0] + 3168);
    _Relu_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 3424);
    _Relu_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 3424);
    _Relu_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 96);
    _Relu_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 96);
    _dsc1_Relu_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 3168);
    _dsc1_Relu_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 3168);
    _dsc1_Relu_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 24);
    _dsc1_Relu_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 24);
    _dsc1_Relu_1_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 3096);
    _dsc1_Relu_1_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 3096);
    _dsc1_Relu_1_output_0_scratch1_array.data = AI_PTR(g_network_activations_map[0] + 4440);
    _dsc1_Relu_1_output_0_scratch1_array.data_start = AI_PTR(g_network_activations_map[0] + 4440);
    _dsc1_Relu_1_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _dsc1_Relu_1_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _dsc2_Relu_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 1536);
    _dsc2_Relu_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 1536);
    _dsc2_Relu_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 1756);
    _dsc2_Relu_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 1756);
    _dsc2_Relu_1_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _dsc2_Relu_1_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _dsc2_Relu_1_output_0_scratch1_array.data = AI_PTR(g_network_activations_map[0] + 1344);
    _dsc2_Relu_1_output_0_scratch1_array.data_start = AI_PTR(g_network_activations_map[0] + 1344);
    _dsc2_Relu_1_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 3292);
    _dsc2_Relu_1_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 3292);
    _dsc3_Relu_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _dsc3_Relu_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _dsc3_Relu_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 220);
    _dsc3_Relu_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 220);
    _dsc3_Relu_1_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 988);
    _dsc3_Relu_1_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 988);
    _dsc3_Relu_1_output_0_scratch1_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _dsc3_Relu_1_output_0_scratch1_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _dsc3_Relu_1_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 2332);
    _dsc3_Relu_1_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 2332);
    _dsc3_Relu_1_output_0_0_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _dsc3_Relu_1_output_0_0_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _se_ReduceMean_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 1536);
    _se_ReduceMean_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 1536);
    _se_ReduceMean_output_0_Mul_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _se_ReduceMean_output_0_Mul_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _se_ReduceMean_output_0_Mul_0_0__se_Relu_output_0_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 96);
    _se_ReduceMean_output_0_Mul_0_0__se_Relu_output_0_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 96);
    _se_Relu_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _se_Relu_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _se_Relu_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 48);
    _se_Relu_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 48);
    _se_fc2_Gemm_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _se_fc2_Gemm_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _se_fc2_Gemm_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 12);
    _se_fc2_Gemm_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 12);
    _se_Sigmoid_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 36);
    _se_Sigmoid_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 36);
    _se_Mul_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 60);
    _se_Mul_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 60);
    _Reshape_output_0_to_chlast_output_array.data = AI_PTR(g_network_activations_map[0] + 444);
    _Reshape_output_0_to_chlast_output_array.data_start = AI_PTR(g_network_activations_map[0] + 444);
    _Reshape_1_output_0_to_chlast_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _Reshape_1_output_0_to_chlast_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _Reshape_1_output_0_to_chfirst_output_array.data = AI_PTR(g_network_activations_map[0] + 384);
    _Reshape_1_output_0_to_chfirst_output_array.data_start = AI_PTR(g_network_activations_map[0] + 384);
    _Reshape_1_output_0_to_chfirst_0_0__ReduceMean_output_0_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 768);
    _Reshape_1_output_0_to_chfirst_0_0__ReduceMean_output_0_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 768);
    _ReduceMean_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _ReduceMean_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _ReduceMean_output_0_Mul_output_array.data = AI_PTR(g_network_activations_map[0] + 96);
    _ReduceMean_output_0_Mul_output_array.data_start = AI_PTR(g_network_activations_map[0] + 96);
    _ReduceMean_output_0_Mul_0_0__classifier_classifier_1_Relu_output_0_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _ReduceMean_output_0_Mul_0_0__classifier_classifier_1_Relu_output_0_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _classifier_classifier_1_Relu_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 24);
    _classifier_classifier_1_Relu_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 24);
    _classifier_classifier_1_Relu_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 72);
    _classifier_classifier_1_Relu_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 72);
    logits_QuantizeLinear_Input_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 136);
    logits_QuantizeLinear_Input_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 136);
    logits_QuantizeLinear_Input_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    logits_QuantizeLinear_Input_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    return true;
  }
  AI_ERROR_TRAP(net_ctx, INIT_FAILED, NETWORK_ACTIVATIONS);
  return false;
}




/******************************************************************************/
AI_DECLARE_STATIC
ai_bool network_configure_weights(
  ai_network* net_ctx, const ai_network_params* params)
{
  AI_ASSERT(net_ctx)

  if (ai_platform_get_weights_map(g_network_weights_map, 1, params)) {
    /* Updating weights (byte) offsets */
    
    _Relu_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _Relu_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 0);
    _Relu_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 0);
    _Relu_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _Relu_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 48);
    _Relu_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 48);
    _dsc1_Relu_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _dsc1_Relu_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 144);
    _dsc1_Relu_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 144);
    _dsc1_Relu_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _dsc1_Relu_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 216);
    _dsc1_Relu_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 216);
    _dsc1_Relu_1_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _dsc1_Relu_1_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 312);
    _dsc1_Relu_1_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 312);
    _dsc1_Relu_1_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _dsc1_Relu_1_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 888);
    _dsc1_Relu_1_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 888);
    _dsc2_Relu_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _dsc2_Relu_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 984);
    _dsc2_Relu_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 984);
    _dsc2_Relu_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _dsc2_Relu_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 1056);
    _dsc2_Relu_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 1056);
    _dsc2_Relu_1_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _dsc2_Relu_1_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 1152);
    _dsc2_Relu_1_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 1152);
    _dsc2_Relu_1_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _dsc2_Relu_1_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 1728);
    _dsc2_Relu_1_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 1728);
    _dsc3_Relu_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _dsc3_Relu_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 1824);
    _dsc3_Relu_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 1824);
    _dsc3_Relu_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _dsc3_Relu_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 1896);
    _dsc3_Relu_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 1896);
    _dsc3_Relu_1_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _dsc3_Relu_1_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 1992);
    _dsc3_Relu_1_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 1992);
    _dsc3_Relu_1_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _dsc3_Relu_1_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 2568);
    _dsc3_Relu_1_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 2568);
    _se_ReduceMean_output_0_Mul_scale_array.format |= AI_FMT_FLAG_CONST;
    _se_ReduceMean_output_0_Mul_scale_array.data = AI_PTR(g_network_weights_map[0] + 2664);
    _se_ReduceMean_output_0_Mul_scale_array.data_start = AI_PTR(g_network_weights_map[0] + 2664);
    _se_ReduceMean_output_0_Mul_bias_array.format |= AI_FMT_FLAG_CONST;
    _se_ReduceMean_output_0_Mul_bias_array.data = AI_PTR(g_network_weights_map[0] + 2760);
    _se_ReduceMean_output_0_Mul_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 2760);
    _se_Relu_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _se_Relu_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 2856);
    _se_Relu_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 2856);
    _se_Relu_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _se_Relu_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 3000);
    _se_Relu_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 3000);
    _se_fc2_Gemm_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _se_fc2_Gemm_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 3024);
    _se_fc2_Gemm_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 3024);
    _se_fc2_Gemm_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _se_fc2_Gemm_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 3168);
    _se_fc2_Gemm_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 3168);
    _classifier_classifier_1_Relu_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _classifier_classifier_1_Relu_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 3264);
    _classifier_classifier_1_Relu_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 3264);
    _classifier_classifier_1_Relu_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _classifier_classifier_1_Relu_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 4800);
    _classifier_classifier_1_Relu_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 4800);
    logits_QuantizeLinear_Input_weights_array.format |= AI_FMT_FLAG_CONST;
    logits_QuantizeLinear_Input_weights_array.data = AI_PTR(g_network_weights_map[0] + 5056);
    logits_QuantizeLinear_Input_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 5056);
    logits_QuantizeLinear_Input_bias_array.format |= AI_FMT_FLAG_CONST;
    logits_QuantizeLinear_Input_bias_array.data = AI_PTR(g_network_weights_map[0] + 5760);
    logits_QuantizeLinear_Input_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 5760);
    return true;
  }
  AI_ERROR_TRAP(net_ctx, INIT_FAILED, NETWORK_WEIGHTS);
  return false;
}


/**  PUBLIC APIs SECTION  *****************************************************/



AI_DEPRECATED
AI_API_ENTRY
ai_bool ai_network_get_info(
  ai_handle network, ai_network_report* report)
{
  ai_network* net_ctx = AI_NETWORK_ACQUIRE_CTX(network);

  if (report && net_ctx)
  {
    ai_network_report r = {
      .model_name        = AI_NETWORK_MODEL_NAME,
      .model_signature   = AI_NETWORK_MODEL_SIGNATURE,
      .model_datetime    = AI_TOOLS_DATE_TIME,
      
      .compile_datetime  = AI_TOOLS_COMPILE_TIME,
      
      .runtime_revision  = ai_platform_runtime_get_revision(),
      .runtime_version   = ai_platform_runtime_get_version(),

      .tool_revision     = AI_TOOLS_REVISION_ID,
      .tool_version      = {AI_TOOLS_VERSION_MAJOR, AI_TOOLS_VERSION_MINOR,
                            AI_TOOLS_VERSION_MICRO, 0x0},
      .tool_api_version  = AI_STRUCT_INIT,

      .api_version            = ai_platform_api_get_version(),
      .interface_api_version  = ai_platform_interface_api_get_version(),
      
      .n_macc            = 163615,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .params            = AI_STRUCT_INIT,
      .activations       = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0x9b96750f,
    };

    if (!ai_platform_api_get_network_report(network, &r)) return false;

    *report = r;
    return true;
  }
  return false;
}



AI_API_ENTRY
ai_bool ai_network_get_report(
  ai_handle network, ai_network_report* report)
{
  ai_network* net_ctx = AI_NETWORK_ACQUIRE_CTX(network);

  if (report && net_ctx)
  {
    ai_network_report r = {
      .model_name        = AI_NETWORK_MODEL_NAME,
      .model_signature   = AI_NETWORK_MODEL_SIGNATURE,
      .model_datetime    = AI_TOOLS_DATE_TIME,
      
      .compile_datetime  = AI_TOOLS_COMPILE_TIME,
      
      .runtime_revision  = ai_platform_runtime_get_revision(),
      .runtime_version   = ai_platform_runtime_get_version(),

      .tool_revision     = AI_TOOLS_REVISION_ID,
      .tool_version      = {AI_TOOLS_VERSION_MAJOR, AI_TOOLS_VERSION_MINOR,
                            AI_TOOLS_VERSION_MICRO, 0x0},
      .tool_api_version  = AI_STRUCT_INIT,

      .api_version            = ai_platform_api_get_version(),
      .interface_api_version  = ai_platform_interface_api_get_version(),
      
      .n_macc            = 163615,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .map_signature     = AI_MAGIC_SIGNATURE,
      .map_weights       = AI_STRUCT_INIT,
      .map_activations   = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0x9b96750f,
    };

    if (!ai_platform_api_get_network_report(network, &r)) return false;

    *report = r;
    return true;
  }
  return false;
}


AI_API_ENTRY
ai_error ai_network_get_error(ai_handle network)
{
  return ai_platform_network_get_error(network);
}


AI_API_ENTRY
ai_error ai_network_create(
  ai_handle* network, const ai_buffer* network_config)
{
  return ai_platform_network_create(
    network, network_config, 
    AI_CONTEXT_OBJ(&AI_NET_OBJ_INSTANCE),
    AI_TOOLS_API_VERSION_MAJOR, AI_TOOLS_API_VERSION_MINOR, AI_TOOLS_API_VERSION_MICRO);
}


AI_API_ENTRY
ai_error ai_network_create_and_init(
  ai_handle* network, const ai_handle activations[], const ai_handle weights[])
{
  ai_error err;
  ai_network_params params;

  err = ai_network_create(network, AI_NETWORK_DATA_CONFIG);
  if (err.type != AI_ERROR_NONE) {
    return err;
  }
  
  if (ai_network_data_params_get(&params) != true) {
    err = ai_network_get_error(*network);
    return err;
  }
#if defined(AI_NETWORK_DATA_ACTIVATIONS_COUNT)
  /* set the addresses of the activations buffers */
  for (ai_u16 idx=0; activations && idx<params.map_activations.size; idx++) {
    AI_BUFFER_ARRAY_ITEM_SET_ADDRESS(&params.map_activations, idx, activations[idx]);
  }
#endif
#if defined(AI_NETWORK_DATA_WEIGHTS_COUNT)
  /* set the addresses of the weight buffers */
  for (ai_u16 idx=0; weights && idx<params.map_weights.size; idx++) {
    AI_BUFFER_ARRAY_ITEM_SET_ADDRESS(&params.map_weights, idx, weights[idx]);
  }
#endif
  if (ai_network_init(*network, &params) != true) {
    err = ai_network_get_error(*network);
  }
  return err;
}


AI_API_ENTRY
ai_buffer* ai_network_inputs_get(ai_handle network, ai_u16 *n_buffer)
{
  if (network == AI_HANDLE_NULL) {
    network = (ai_handle)&AI_NET_OBJ_INSTANCE;
    AI_NETWORK_OBJ(network)->magic = AI_MAGIC_CONTEXT_TOKEN;
  }
  return ai_platform_inputs_get(network, n_buffer);
}


AI_API_ENTRY
ai_buffer* ai_network_outputs_get(ai_handle network, ai_u16 *n_buffer)
{
  if (network == AI_HANDLE_NULL) {
    network = (ai_handle)&AI_NET_OBJ_INSTANCE;
    AI_NETWORK_OBJ(network)->magic = AI_MAGIC_CONTEXT_TOKEN;
  }
  return ai_platform_outputs_get(network, n_buffer);
}


AI_API_ENTRY
ai_handle ai_network_destroy(ai_handle network)
{
  return ai_platform_network_destroy(network);
}


AI_API_ENTRY
ai_bool ai_network_init(
  ai_handle network, const ai_network_params* params)
{
  ai_network* net_ctx = AI_NETWORK_OBJ(ai_platform_network_init(network, params));
  ai_bool ok = true;

  if (!net_ctx) return false;
  ok &= network_configure_weights(net_ctx, params);
  ok &= network_configure_activations(net_ctx, params);

  ok &= ai_platform_network_post_init(network);

  return ok;
}


AI_API_ENTRY
ai_i32 ai_network_run(
  ai_handle network, const ai_buffer* input, ai_buffer* output)
{
  return ai_platform_network_process(network, input, output);
}


AI_API_ENTRY
ai_i32 ai_network_forward(ai_handle network, const ai_buffer* input)
{
  return ai_platform_network_process(network, input, NULL);
}



#undef AI_NETWORK_MODEL_SIGNATURE
#undef AI_NET_OBJ_INSTANCE
#undef AI_TOOLS_DATE_TIME
#undef AI_TOOLS_COMPILE_TIME

