/**
 * retarget_keil.c
 * ===============
 * printf -> UART retargeting for Keil MDK (Arm Compiler 6).
 *
 * Why this file exists
 * --------------------
 * The CubeIDE / GCC flow redirects printf through __io_putchar() in the
 * CubeMX-generated Core/Src/syscalls.c. Keil MDK does not use syscalls.c;
 * its C library calls fputc() instead. This file provides that override so
 * [AI] / [DWT] / [RESULT] lines come out on the UART exactly as in the
 * original firmware.
 *
 * Usage
 * -----
 * 1. Copy this file into the CubeMX-generated project's Core/Src/ folder.
 * 2. Add it to the Keil project (Application/User/Core group).
 * 3. Keep "Use MicroLIB" UNCHECKED in Options for Target -> Target, because
 *    the full ARM C library is needed for floating-point printf ("%.3f").
 * 4. Set DSC_RETARGET_UART below to the handle CubeMX generated for your board:
 *      - NUCLEO-H723ZG        : USART3 on PD8/PD9 (ST-LINK Virtual COM) -> huart3
 *      - LXB723ZG-P1 (custom) : USART1 on PA9/PA10 (debug header)      -> huart1
 */

#include <stdio.h>
#include "main.h"

/* ---- choose the UART used for printf (change if your board differs) ---- */
#ifndef DSC_RETARGET_UART
#define DSC_RETARGET_UART huart3   /* NUCLEO: huart3 | custom board: huart1 */
#endif

extern UART_HandleTypeDef DSC_RETARGET_UART;

/**
 * @brief  Redirect the C library's character output to the UART.
 * @retval The character written (required by the ARM C library contract).
 */
int fputc(int ch, FILE *f)
{
    (void)f;
    HAL_UART_Transmit(&DSC_RETARGET_UART, (uint8_t *)&ch, 1U, HAL_MAX_DELAY);
    return ch;
}

/* Stubs the ARM C library may reference when printf is used without
 * semihosting. Harmless if never called; add them only if the linker
 * reports _sys_exit / _ttywrch as undefined. */
void _sys_exit(int x)
{
    (void)x;
    for (;;) { }
}

void _ttywrch(int ch)
{
    (void)ch;
}

/* The standard library's stdio initialization opens ":tt" through
 * _sys_open() at startup, which is a semihosting call (BKPT 0xAB) and
 * hard-faults when no debugger is attached. Intercept it via the ARM
 * linker's $Sub$$ mechanism and report "no stream". */
int $Sub$$_sys_open(const char *name, int openmode)
{
    (void)name;
    (void)openmode;
    return -1;
}
