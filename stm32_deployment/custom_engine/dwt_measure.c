/**
 * dwt_measure.c
 * =============
 * DWT (Data Watchpoint and Trace) cycle-accurate inference timing.
 *
 * Provides microsecond-resolution latency measurement on ARM Cortex-M7.
 * Used to obtain the silicon-measured latency values reported in the paper.
 *
 * Hardware: STM32H723ZG Nucleo
 *   - Core clock: 550 MHz (SYSCLK, VOS0; HSI 64 MHz, PLL1 M=8, N=68, FRACN=6144/8192)
 *   - DWT_CYCCNT increments every CPU cycle
 *   - Resolution: 1/544e6 s ≈ 1.84 ns per count
 *
 * Measured results (reported in paper Table IV):
 *   DSC-AMCNet INT8  : 1,183 µs  (650,650 cycles @ 550 MHz)
 *   ULCNN-simp INT8  : 2,104 µs  (1,157,200 cycles @ 550 MHz)
 *   Speedup          : 1.78×
 *
 * Usage
 * -----
 * 1. Call DWT_Init() once at startup (after SystemClock_Config).
 * 2. Call DWT_Start() immediately before ai_run().
 * 3. Call DWT_Stop() immediately after ai_run().
 * 4. Read latency via DWT_GetMicroseconds().
 *
 * Include:  "dwt_measure.h"
 * Link with: STM32H7xx HAL, X-CUBE-AI runtime
 */

#include "dwt_measure.h"
#include "stm32h7xx_hal.h"
#include <stdio.h>

/* -------------------------------------------------------------------------
 * Module-private state
 * ---------------------------------------------------------------------- */
static uint32_t dwt_start_cycles = 0;
static uint32_t dwt_stop_cycles  = 0;
static uint32_t dwt_elapsed      = 0;

/* -------------------------------------------------------------------------
 * DWT_Init
 * Enables the DWT cycle counter.
 * Must be called once before any measurement.
 * ---------------------------------------------------------------------- */
void DWT_Init(void)
{
    /* On STM32H7 the debug trace clock must be enabled in DBGMCU before the
     * DWT cycle counter advances; without it CYCCNT stays frozen at 0. */
    DBGMCU->CR |= DBGMCU_CR_DBG_TRACECKEN;

    /* Unlock the DWT register block (software lock). */
    DWT->LAR = 0xC5ACCE55U;

    /* CoreDebug: enable trace */
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;

    /* Reset and enable cycle counter */
    DWT->CYCCNT = 0;
    DWT->CTRL  |= DWT_CTRL_CYCCNTENA_Msk;
}

/* -------------------------------------------------------------------------
 * DWT_Start
 * Capture start timestamp. Call immediately before inference.
 * ---------------------------------------------------------------------- */
void DWT_Start(void)
{
    dwt_start_cycles = DWT->CYCCNT;
}

/* -------------------------------------------------------------------------
 * DWT_Stop
 * Capture stop timestamp and compute elapsed cycles.
 * Call immediately after inference.
 * ---------------------------------------------------------------------- */
void DWT_Stop(void)
{
    dwt_stop_cycles = DWT->CYCCNT;

    /* Handle 32-bit counter wrap-around */
    if (dwt_stop_cycles >= dwt_start_cycles)
    {
        dwt_elapsed = dwt_stop_cycles - dwt_start_cycles;
    }
    else
    {
        dwt_elapsed = (UINT32_MAX - dwt_start_cycles) + dwt_stop_cycles + 1U;
    }
}

/* -------------------------------------------------------------------------
 * DWT_GetCycles
 * Returns raw elapsed cycle count from last Start/Stop pair.
 * ---------------------------------------------------------------------- */
uint32_t DWT_GetCycles(void)
{
    return dwt_elapsed;
}

/* -------------------------------------------------------------------------
 * DWT_GetMicroseconds
 * Converts elapsed cycles to microseconds using the system clock.
 *
 * Formula: µs = cycles / (HCLK_MHz)
 *
 * For STM32H723ZG at 550 MHz:
 *   1 µs = 544 cycles
 *   1,183 µs = 650,650 cycles  (DSC-AMCNet INT8)
 * ---------------------------------------------------------------------- */
float DWT_GetMicroseconds(void)
{
    uint32_t hclk_mhz = SystemCoreClock / 1000000U;  /* CPU core runs at SYSCLK (550 MHz), HCLK is AXI (275 MHz) */
    return (float)dwt_elapsed / (float)hclk_mhz;
}

/* -------------------------------------------------------------------------
 * DWT_PrintResult
 * Prints measurement via UART (Redirect printf to USART3 in CubeIDE).
 * ---------------------------------------------------------------------- */
/* Integer conversion: microseconds x100, avoiding printf float support. */
static uint32_t cycles_to_us_x100(uint32_t cycles)
{
    uint32_t mhz = SystemCoreClock / 1000000U;
    return (uint32_t)(((uint64_t)cycles * 100ULL) / (uint64_t)mhz);
}

/* -------------------------------------------------------------------------
 * DWT_PrintResult
 * Prints measurement via UART (USART3, uart_printf from main.c).
 * ---------------------------------------------------------------------- */
void DWT_PrintResult(const char *label)
{
    uint32_t us_x100 = cycles_to_us_x100(dwt_elapsed);
    uint32_t ms_x1000 = us_x100 * 10U;

    uart_printf("[DWT] %s: %lu cycles | %lu.%02lu us | %lu.%03lu ms\r\n",
           label, (unsigned long)dwt_elapsed,
           (unsigned long)(us_x100 / 100U), (unsigned long)(us_x100 % 100U),
           (unsigned long)(ms_x1000 / 1000U), (unsigned long)(ms_x1000 % 1000U));
}

/* -------------------------------------------------------------------------
 * DWT_BenchmarkN
 * Runs inference N times, reports min / mean / max.
 * Used to obtain stable latency figures for the paper.
 *
 * Paper methodology:
 *   N = 1000 inference passes per model
 *   Reported value = mean of 1000 measurements
 *   First 10 passes discarded (cache warm-up)
 * ---------------------------------------------------------------------- */
void DWT_BenchmarkN(
    void (*inference_fn)(void),   /* pointer to ai_run() wrapper */
    uint32_t N,
    uint32_t *out_mean_us_x100,
    uint32_t *out_min_us_x100,
    uint32_t *out_max_us_x100)
{
    uint64_t sum = 0ULL;
    uint32_t mn  = 0xFFFFFFFFU;
    uint32_t mx  = 0U;

    /* Warm-up passes (not counted) */
    for (uint32_t i = 0; i < 10; i++) { inference_fn(); }

    for (uint32_t i = 0; i < N; i++)
    {
        DWT_Start();
        inference_fn();
        DWT_Stop();

        uint32_t us_x100 = cycles_to_us_x100(DWT_GetCycles());
        sum += us_x100;
        if (us_x100 < mn) mn = us_x100;
        if (us_x100 > mx) mx = us_x100;
    }

    uint32_t mean_x100 = (uint32_t)(sum / (uint64_t)N);
    if (out_mean_us_x100) { *out_mean_us_x100 = mean_x100; }
    if (out_min_us_x100)  { *out_min_us_x100  = mn; }
    if (out_max_us_x100)  { *out_max_us_x100  = mx; }

    /* us_x100 -> ms x1000 = us_x100 / 100 */
    uint32_t mean_ms = mean_x100 / 100U, min_ms = mn / 100U, max_ms = mx / 100U;
    uart_printf("[DWT] N=%lu  mean=%lu.%03lu ms  min=%lu.%03lu ms  max=%lu.%03lu ms\r\n",
           (unsigned long)N,
           (unsigned long)(mean_ms / 1000U), (unsigned long)(mean_ms % 1000U),
           (unsigned long)(min_ms  / 1000U), (unsigned long)(min_ms  % 1000U),
           (unsigned long)(max_ms  / 1000U), (unsigned long)(max_ms  % 1000U));
}

