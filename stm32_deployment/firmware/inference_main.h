/**
 * inference_main.h
 * ================
 * Interface for the DSC-AMCNet INT8 inference loop implemented in
 * inference_main.c.
 *
 * Copy this file to Core/Inc/ of the STM32CubeIDE project and call
 * inference_main() once from main(), after HAL_Init(), SystemClock_Config()
 * and MX_USART3_UART_Init(). The function does not return: it runs the
 * latency benchmark and then loops on continuous classification.
 */

#ifndef DSC_AMCNET_INFERENCE_MAIN_H
#define DSC_AMCNET_INFERENCE_MAIN_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Initialise the X-CUBE-AI network, run the 1000-iteration DWT benchmark,
 * then classify continuously and print results on USART3 (115200 8N1).
 *
 * Requires `printf` to be redirected to USART3 via __io_putchar() in
 * syscalls.c. Calls Error_Handler() and halts if network init fails.
 */
void inference_main(void);

#ifdef __cplusplus
}
#endif

#endif /* DSC_AMCNET_INFERENCE_MAIN_H */
