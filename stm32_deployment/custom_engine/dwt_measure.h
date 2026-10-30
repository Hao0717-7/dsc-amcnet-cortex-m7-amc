/**
 * dwt_measure.h
 * =============
 * Interface for the DWT (Data Watchpoint and Trace) cycle-accurate timing
 * helper implemented in dwt_measure.c.
 *
 * Copy this file to Core/Inc/ of the STM32CubeIDE project (see
 * stm32_deployment/README.md step 6).
 *
 * Typical use:
 *
 *     DWT_Init();                       // once, after SystemClock_Config()
 *     DWT_Start();
 *     ai_network_run(...);              // or run_inference_once()
 *     DWT_Stop();
 *     float us = DWT_GetMicroseconds(); // 1 us = HCLK_MHz cycles
 *
 * The counter is 32-bit and wraps every 2^32 cycles (~7.9 s at 544 MHz);
 * DWT_Stop() already handles the single-wrap case.
 */

#ifndef DSC_AMCNET_DWT_MEASURE_H
#define DSC_AMCNET_DWT_MEASURE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Enable CoreDebug trace and start the DWT cycle counter (CYCCNT). */
void DWT_Init(void);

/** Latch the current cycle count as the start of the measured region. */
void DWT_Start(void);

/** Latch the current cycle count and compute the elapsed cycles. */
void DWT_Stop(void);

/** Raw elapsed cycles from the last DWT_Start()/DWT_Stop() pair. */
uint32_t DWT_GetCycles(void);

/**
 * Elapsed time in microseconds.
 *
 * Derived from SystemCoreClock (the CPU core clock), so it is correct at any
 * core clock (544 MHz on the NUCLEO-H723ZG at VOS0). Note that HCLK here is
 * 272 MHz (HPRE = /2), so HAL_RCC_GetHCLKFreq() must NOT be used — doing so
 * reports 2x the real latency (bug fixed 2026-10-27).
 */
float DWT_GetMicroseconds(void);

/** printf a one-line timing report for the given label (USART3). */
void DWT_PrintResult(const char *label);

/**
 * Run inference_fn() N times and report the mean / min / max latency.
 *
 * The first 10 calls are warm-up and are not counted (paper Table IV
 * methodology). Any of the out_* pointers may be NULL.
 *
 * @param inference_fn  pointer to the wrapper that performs one inference
 * @param N             number of measured iterations (paper uses 1000)
 * @param out_mean_us_x100   mean latency in microseconds x100
 * @param out_min_us_x100    minimum latency in microseconds x100
 * @param out_max_us_x100    maximum latency in microseconds x100
 */
void DWT_BenchmarkN(void (*inference_fn)(void),
                    uint32_t N,
                    uint32_t *out_mean_us_x100,
                    uint32_t *out_min_us_x100,
                    uint32_t *out_max_us_x100);

#ifdef __cplusplus
}
#endif

#endif /* DSC_AMCNET_DWT_MEASURE_H */
