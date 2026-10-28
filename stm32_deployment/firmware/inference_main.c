/**
 * inference_main.c
 * ================
 * STM32H723ZG bare-metal inference loop for DSC-AMCNet INT8.
 * Integrates with X-CUBE-AI 10.2.0 runtime (ST Edge AI Core 2.2.0).
 *
 * Hardware configuration:
 *   Board  : NUCLEO-H723ZG
 *   Core   : ARM Cortex-M7 @ 544 MHz
 *   Flash  : 1024 KB (program + X-CUBE-AI weights)
 *   SRAM   : 564 KB (activations, buffers)
 *
 * Model footprint (ST Edge AI Core Analyze, INT8):
 *   Flash  : 51.26 KB
 *   RAM    : 12.09 KB
 *   Latency: 1.183 ms (1,183 µs) — DWT measured
 *
 * UART output (USART3, 115200 baud, redirected via printf):
 *   Reports predicted class and softmax confidence for each inference.
 *   DWT latency in µs is printed every inference during benchmark mode.
 *
 * Double-buffer DMA (M20 real-time framing):
 *   See M20 real-time section. DMA fills buffer A while inference runs on
 *   buffer B, then swaps. Enables continuous classification across the
 *   LPWAN band without dropped samples.
 *
 * Build in STM32CubeIDE:
 *   Project → Properties → C/C++ Build → Settings
 *   Add X-CUBE-AI include path: Middlewares/ST/AI/Inc
 *   Add X-CUBE-AI lib: Middlewares/ST/AI/Lib/NetworkRuntime810_CM7_GCC.a
 */

#include "main.h"
#include "ai_datatypes_defines.h"
#include "network.h"
#include "network_data.h"
#include "dwt_measure.h"
#include "dsc_infer.h"

#include <stdio.h>
#include <string.h>
#include <math.h>

/* -------------------------------------------------------------------------
 * X-CUBE-AI handles and buffers
 * ---------------------------------------------------------------------- */
static ai_handle   network_handle = AI_HANDLE_NULL;
static ai_buffer  *ai_input_buffers  = AI_HANDLE_NULL;
static ai_buffer  *ai_output_buffers = AI_HANDLE_NULL;
static volatile ai_i32 g_last_run = -999;

/* Input: (1, 2, 128) float32 = 256 floats = 1024 bytes */
AI_ALIGNED(4) static float input_data[2 * 128];

/* Output: (1, 11) float32  (raw logits -- the model ends in a bare Linear) */
AI_ALIGNED(4) static float output_data[AI_NETWORK_OUT_1_SIZE];

/* Activations scratch area. X-CUBE-AI 10 generates the activation map with a
 * NULL data pointer and expects the application to supply the storage, so this
 * buffer must exist and be handed to ai_network_init() (see below).
 * With --target stm32h7 this model needs AI_NETWORK_DATA_ACTIVATIONS_SIZE bytes. */
AI_ALIGNED(32) static ai_u8 activations[AI_NETWORK_DATA_ACTIVATIONS_SIZE];

/* -------------------------------------------------------------------------
 * Modulation class labels (RML2016.10a ordering)
 * ---------------------------------------------------------------------- */
static const char *MODULATIONS[11] = {
    "8PSK", "AM-DSB", "AM-SSB", "BPSK", "CPFSK",
    "GFSK", "PAM4",  "QAM16",  "QAM64", "QPSK", "WBFM"
};

/* -------------------------------------------------------------------------
 * Network Initialisation
 * ---------------------------------------------------------------------- */
static HAL_StatusTypeDef network_init(void)
{
    ai_error err;

    err = ai_network_create(&network_handle, AI_NETWORK_DATA_CONFIG);
    if (err.type != AI_ERROR_NONE)
    {
        uart_printf("[AI] Create error: type=%d code=%d\r\n", err.type, err.code);
        return HAL_ERROR;
    }

    const ai_network_params params = {
        AI_NETWORK_DATA_WEIGHTS(ai_network_data_weights_get()),
        AI_NETWORK_DATA_ACTIVATIONS(activations)
    };
    if (!ai_network_init(network_handle, &params))
    {
        err = ai_network_get_error(network_handle);
        uart_printf("[AI] Init error: type=%d code=%d\r\n", err.type, err.code);
        return HAL_ERROR;
    }

    ai_input_buffers  = ai_network_inputs_get(network_handle, NULL);
    ai_output_buffers = ai_network_outputs_get(network_handle, NULL);
    if (ai_input_buffers == NULL || ai_output_buffers == NULL)
    {
        uart_printf("[AI] I/O descriptor error\r\n");
        return HAL_ERROR;
    }

    uart_printf("[AI] DSC-AMCNet INT8 initialised.\r\n");
    uart_printf("[AI] weights=%u B (flash) | activations=%u B | io=%u+%u B (ram)\r\n",
           (unsigned)AI_NETWORK_DATA_WEIGHTS_SIZE,
           (unsigned)AI_NETWORK_DATA_ACTIVATIONS_SIZE,
           (unsigned)AI_NETWORK_IN_1_SIZE_BYTES,
           (unsigned)AI_NETWORK_OUT_1_SIZE_BYTES);
    return HAL_OK;
}

/* Per-window L2 normalization, defined below (used by run_inference_once). */
static void normalize_window_l2(float *x, int n);

/* -------------------------------------------------------------------------
 * Single inference wrapper (called by DWT_BenchmarkN)
 * ---------------------------------------------------------------------- */
static void run_inference_once(void)
{
    /* Per-window L2 normalization -- must match the training preprocessing
     * (models/train.py -> RML2016Dataset.__init__ -> normalize_window).
     * Done here rather than at the fill sites so that every caller, including
     * the DWT benchmark, gets the same preprocessing. Identical math to
     * models/dsc_amcnet.py:normalize_window(x, eps=1e-12) over 2 x 128. */
    normalize_window_l2(input_data, 2 * 128);

    ai_input_buffers[0].data  = AI_HANDLE_PTR(input_data);
    ai_output_buffers[0].data = AI_HANDLE_PTR(output_data);

    g_last_run = ai_network_run(network_handle, ai_input_buffers, ai_output_buffers);
}

/* ---- experimental: hand-written int8 engine (dsc_infer.c) ------------- */
static int8_t my_logits[11];

static void run_custom_once(void)
{
    dsc_infer_run(input_data, my_logits);
}

/* -------------------------------------------------------------------------
 * argmax over raw logits + softmax probability for reporting
 *
 * NOTE: the exported model ends in a bare Linear layer (models/dsc_amcnet.py
 * classifier: Linear -> ReLU -> Identity -> Linear), so output_data holds raw
 * logits, NOT log-softmax scores. Use a numerically stable softmax here;
 * expf(score) alone is not a probability.
 * ---------------------------------------------------------------------- */
static int argmax_softmax(float *scores, int n, float *out_confidence)
{
    int best = 0;
    float max = scores[0];
    for (int i = 1; i < n; i++)
    {
        if (scores[i] > scores[best]) best = i;
        if (scores[i] > max) max = scores[i];
    }
    /* softmax(best) = exp(x_best - max) / sum(exp(x_i - max)) */
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += expf(scores[i] - max);
    *out_confidence = (sum > 0.0f) ? (1.0f / sum) : 0.0f;
    return best;
}

/* -------------------------------------------------------------------------
 * Input preprocessing: per-window L2 normalization
 *
 * The model was trained on windows normalized exactly this way
 * (models/train.py -> RML2016Dataset.__init__ calls
 * models/dsc_amcnet.py:normalize_window on every sample). Feeding raw ADC
 * amplitudes instead costs almost all of the accuracy.
 *
 * Applied unconditionally inside run_inference_once(), so both the DWT
 * benchmark and the classification loop see the same preprocessing.
 *
 * normalize_window(x, eps=1e-12): x / max(||x||_2, eps) over the whole
 * 2 x 128 window (both I and Q together, not per channel).
 * ---------------------------------------------------------------------- */
static void normalize_window_l2(float *x, int n)
{
    float sum_sq = 0.0f;
    for (int i = 0; i < n; i++) sum_sq += x[i] * x[i];

    float norm = sqrtf(sum_sq);
    if (norm < 1e-12f) norm = 1e-12f;

    for (int i = 0; i < n; i++) x[i] /= norm;
}

/* -------------------------------------------------------------------------
 * Main inference loop
 * ---------------------------------------------------------------------- */
void inference_main(void)
{
    uart_printf("[VER] ptr2\r\n");
    uart_printf("\r\n=== DSC-AMCNet Inference on STM32H723ZG ===\r\n");
    uart_printf("    Core: Cortex-M7 @ 544 MHz\r\n");
    uart_printf("    Model: DSC-AMCNet INT8 (X-CUBE-AI 10.2.0)\r\n\r\n");

    /* Initialise DWT */
    DWT_Init();

    /* Initialise network */
    if (network_init() != HAL_OK)
    {
        uart_printf("[FATAL] Network init failed. Halting.\r\n");
        Error_Handler();
    }
    /* ---- bring-up diagnostics ---------------------------------------- */
    uart_printf("[DIAG] SYSCLK=%lu Hz DEMCR=%08lx DWT_CTRL=%08lx DBGMCU_CR=%08lx\r\n",
                (unsigned long)SystemCoreClock,
                (unsigned long)CoreDebug->DEMCR,
                (unsigned long)DWT->CTRL,
                (unsigned long)DBGMCU->CR);
    uart_printf("[DIAG] CCR=%08lx (DC=%lu IC=%lu)\r\n",
                (unsigned long)SCB->CCR,
                (unsigned long)((SCB->CCR >> 16) & 1U),
                (unsigned long)((SCB->CCR >> 17) & 1U));
    {
        uint32_t c0 = DWT->CYCCNT;
        HAL_Delay(10);
        uint32_t c1 = DWT->CYCCNT;
        uart_printf("[DIAG] CYCCNT %lu -> %lu (delta %lu, expect ~640000)\r\n",
                    (unsigned long)c0, (unsigned long)c1, (unsigned long)(c1 - c0));
    }
    memset(input_data, 0, sizeof(input_data));
    run_inference_once();
    uart_printf("[DIAG] ai_run ret=%ld logits*1000: %d %d %d %d %d\r\n",
                (long)g_last_run,
                (int)(output_data[0] * 1000.0f), (int)(output_data[1] * 1000.0f),
                (int)(output_data[2] * 1000.0f), (int)(output_data[3] * 1000.0f),
                (int)(output_data[4] * 1000.0f));

    /* ---- compare with the hand-written int8 engine -------------------- */
    dsc_infer_init();
    dsc_infer_run(input_data, my_logits);
    uart_printf("[MY] logits*1000: %d %d %d %d %d %d %d %d %d %d %d\r\n",
                (int)(my_logits[0] * 0.35008532f * 1000.0f), (int)(my_logits[1] * 0.35008532f * 1000.0f),
                (int)(my_logits[2] * 0.35008532f * 1000.0f), (int)(my_logits[3] * 0.35008532f * 1000.0f),
                (int)(my_logits[4] * 0.35008532f * 1000.0f), (int)(my_logits[5] * 0.35008532f * 1000.0f),
                (int)(my_logits[6] * 0.35008532f * 1000.0f), (int)(my_logits[7] * 0.35008532f * 1000.0f),
                (int)(my_logits[8] * 0.35008532f * 1000.0f), (int)(my_logits[9] * 0.35008532f * 1000.0f),
                (int)(my_logits[10] * 0.35008532f * 1000.0f));
    /* ------------------------------------------------------------------ */

    /* -----------------------------------------------------------------
     * Benchmark mode: 1000 inferences, report mean / min / max latency
     * This is the methodology used to produce Table IV in the paper.
     * --------------------------------------------------------------- */
    /* ---- sample self-test: vendor vs custom ---- */
    {
        extern void dsc_fill_sample(float *dst);
        extern void dsc_infer_run(const float *input_f32, signed char *logits_out);
        signed char myl[11];
        int i;
        extern void dsc_dbg_sample(void);
        dsc_dbg_sample();
        dsc_fill_sample(input_data);
        ai_input_buffers[0].data = AI_HANDLE_PTR(input_data);
        ai_output_buffers[0].data = AI_HANDLE_PTR(output_data);
        float s0 = input_data[0], s1 = input_data[1], s2 = input_data[2], s255 = input_data[255];
        int ai_ret = ai_network_run(network_handle, ai_input_buffers, ai_output_buffers);
        uart_printf("[IN] s0=%d s1=%d s2=%d s255=%d ret=%d\r\n",
            (int)(s0*1000000.0f), (int)(s1*1000000.0f), (int)(s2*1000000.0f), (int)(s255*1000000.0f), ai_ret);        uart_printf("[AIS] %d %d %d %d %d %d %d %d %d %d %d\r\n",
            (int)(output_data[0]*1000.0f), (int)(output_data[1]*1000.0f), (int)(output_data[2]*1000.0f),
            (int)(output_data[3]*1000.0f), (int)(output_data[4]*1000.0f), (int)(output_data[5]*1000.0f),
            (int)(output_data[6]*1000.0f), (int)(output_data[7]*1000.0f), (int)(output_data[8]*1000.0f),
            (int)(output_data[9]*1000.0f), (int)(output_data[10]*1000.0f));
        dsc_infer_run(input_data, myl);
        uart_printf("[PTR] in=%u desc=%u fmt=%u sz=%u\r\n",
            (unsigned)(uintptr_t)input_data,
            (unsigned)(uintptr_t)ai_input_buffers[0].data,
            (unsigned)ai_input_buffers[0].format, (unsigned)ai_input_buffers[0].size);
        memset(input_data, 0, sizeof(input_data));
        ai_network_run(network_handle, ai_input_buffers, ai_output_buffers);
        uart_printf("[AIZ2] %d %d %d %d %d %d %d %d %d %d %d\r\n",
            (int)(output_data[0]*1000.0f), (int)(output_data[1]*1000.0f), (int)(output_data[2]*1000.0f),
            (int)(output_data[3]*1000.0f), (int)(output_data[4]*1000.0f), (int)(output_data[5]*1000.0f),
            (int)(output_data[6]*1000.0f), (int)(output_data[7]*1000.0f), (int)(output_data[8]*1000.0f),
            (int)(output_data[9]*1000.0f), (int)(output_data[10]*1000.0f));
        ai_network_run(network_handle, ai_input_buffers, ai_output_buffers);
        uart_printf("[AIZ] %d %d %d %d %d %d %d %d %d %d %d\r\n",
            (int)(output_data[0]*1000.0f), (int)(output_data[1]*1000.0f), (int)(output_data[2]*1000.0f),
            (int)(output_data[3]*1000.0f), (int)(output_data[4]*1000.0f), (int)(output_data[5]*1000.0f),
            (int)(output_data[6]*1000.0f), (int)(output_data[7]*1000.0f), (int)(output_data[8]*1000.0f),
            (int)(output_data[9]*1000.0f), (int)(output_data[10]*1000.0f));        uart_printf("[MYS] %d %d %d %d %d %d %d %d %d %d %d\r\n",
            myl[0], myl[1], myl[2], myl[3], myl[4], myl[5], myl[6], myl[7], myl[8], myl[9], myl[10]);
        for (i = 0; i < 256; i++) input_data[i] = 0.0f;
    }
    uart_printf("[BENCH] Running 1000 inferences (10 warm-up discarded) ...\r\n");

    uint32_t mean_x100 = 0U, min_x100 = 0U, max_x100 = 0U;

    /* Fill input with dummy data for pure latency benchmark */
    memset(input_data, 0, sizeof(input_data));

    DWT_BenchmarkN(run_custom_once, 1000, &mean_x100, &min_x100, &max_x100);

    /* us_x100 -> ms x1000: (x100/100) us -> /1000 ms -> *1000 => x100/100 */
    uint32_t mean_ms = mean_x100 / 100U, min_ms = min_x100 / 100U, max_ms = max_x100 / 100U;

    uart_printf("\r\n[RESULT] DSC-AMCNet INT8 Latency (custom SMLAD engine)\r\n");
    uart_printf("         Mean : %lu.%03lu ms  (paper: 1.183 ms)\r\n",
                (unsigned long)(mean_ms / 1000U), (unsigned long)(mean_ms % 1000U));
    uart_printf("         Min  : %lu.%03lu ms\r\n",
                (unsigned long)(min_ms / 1000U), (unsigned long)(min_ms % 1000U));
    uart_printf("         Max  : %lu.%03lu ms\r\n",
                (unsigned long)(max_ms / 1000U), (unsigned long)(max_ms % 1000U));

    /* -----------------------------------------------------------------
     * Classification mode: loop on real IQ input from DMA buffer
     * (In production: replace input_data load with DMA ping-pong buffer)
     * --------------------------------------------------------------- */
    uart_printf("\r\n[CLASSIFY] Starting continuous classification ...\r\n");

    uint32_t frame_count = 0;
    signed char my_logits[11];
    float scores[11];
    while (1)
    {
        /* In real deployment: wait for DMA buffer swap signal */
        /* HAL_GPIO_TogglePin(LD1_GPIO_Port, LD1_Pin); */

        /* Load IQ frame into input_data (caller fills this from DMA) */
        /* memcpy(input_data, dma_buffer_ready, sizeof(input_data)); */

        uint32_t c0 = DWT->CYCCNT;
        DWT_Start();
        dsc_infer_run(input_data, my_logits);
        DWT_Stop();
        uint32_t c1 = DWT->CYCCNT;

        for (int si = 0; si < 11; si++) scores[si] = (float)my_logits[si] * 0.35008532f;
        float confidence;
        int pred = argmax_softmax(scores, 11, &confidence);
        uint32_t us_x100 = (uint32_t)(((uint64_t)(c1 - c0) * 100ULL) /
                             (uint64_t)(SystemCoreClock / 1000000U));
        uint32_t conf_x1000 = (uint32_t)(confidence * 1000.0f);

        if (frame_count < 3)
        {
            uart_printf("[DBG] c0=%lu c1=%lu delta=%lu elapsed=%lu ctrl=%08lx\r\n",
                        (unsigned long)c0, (unsigned long)c1,
                        (unsigned long)(c1 - c0), (unsigned long)DWT_GetCycles(),
                        (unsigned long)DWT->CTRL);
            uart_printf("[DBG] s*100: %d %d %d %d %d %d %d %d %d %d %d\r\n",
                        (int)(scores[0] * 100.0f), (int)(scores[1] * 100.0f),
                        (int)(scores[2] * 100.0f), (int)(scores[3] * 100.0f),
                        (int)(scores[4] * 100.0f), (int)(scores[5] * 100.0f),
                        (int)(scores[6] * 100.0f), (int)(scores[7] * 100.0f),
                        (int)(scores[8] * 100.0f), (int)(scores[9] * 100.0f),
                        (int)(scores[10] * 100.0f));
            uart_printf("[DBG] pred=%d conf*1000=%lu us*100=%lu\r\n",
                        pred, (unsigned long)conf_x1000, (unsigned long)us_x100);
        }

        uart_printf("[%6lu] Pred: %-7s | Conf: %lu.%03lu | Latency: %lu.%02lu us\r\n",
               (unsigned long)frame_count,
               MODULATIONS[pred],
               (unsigned long)(conf_x1000 / 1000U), (unsigned long)(conf_x1000 % 1000U),
               (unsigned long)(us_x100 / 100U), (unsigned long)(us_x100 % 100U));

        frame_count++;
        HAL_Delay(10);   /* Remove in real-time framing; DMA triggers are used instead */
    }
}
