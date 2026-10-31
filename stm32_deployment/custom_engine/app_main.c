/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "inference_main.h"
#include "dsc_infer.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include <stdio.h>
#include <stdarg.h>
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/

UART_HandleTypeDef huart3;

/* USER CODE BEGIN PV */

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MPU_Config(void);
static void MX_GPIO_Init(void);
static void MX_USART3_UART_Init(void);
/* USER CODE BEGIN PFP */

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */
/* USER CODE END 0 */

/* ---------------------------------------------------------------------------
 * Minimal stdio-free formatter.
 *
 * The ARM C library's printf() pulls in stdio stream setup, which calls
 * _sys_open() during startup and kills the image before main() on this build.
 * This formatter writes straight to USART3 and covers the subset the firmware
 * uses: %d %i %u %x %c %s %f with flags '-' '0', width, precision and 'l'.
 * ------------------------------------------------------------------------- */
static void uart_putc(char c)
{
  HAL_UART_Transmit(&huart3, (uint8_t *)&c, 1U, HAL_MAX_DELAY);
}

static void uprint_ull(unsigned long long v, int width, char pad)
{
  char tmp[24];
  int n = 0;
  do { tmp[n++] = (char)('0' + (int)(v % 10ULL)); v /= 10ULL; } while (v != 0ULL && n < 23);
  while (n < width) { uart_putc(pad); width--; }
  while (n > 0) { uart_putc(tmp[--n]); }
}

static void uprint_hex(unsigned long v, int width, int zero)
{
  char tmp[16];
  int n = 0;
  do {
    int d = (int)(v & 0xFU);
    tmp[n++] = (char)(d < 10 ? ('0' + d) : ('a' + d - 10));
    v >>= 4;
  } while (v != 0UL && n < 15);
  char pad = zero ? '0' : ' ';
  while (n < width) { uart_putc(pad); width--; }
  while (n > 0) { uart_putc(tmp[--n]); }
}

static void uprint_float(double v, int prec)
{
  if (v < 0.0) { uart_putc('-'); v = -v; }
  if (prec < 0) { prec = 6; }
  if (prec > 9) { prec = 9; }

  double scale = 1.0;
  for (int i = 0; i < prec; i++) { scale *= 10.0; }

  double scaled = v * scale + 0.5;
  unsigned long long all = (unsigned long long)scaled;
  unsigned long long div = 1ULL;
  for (int i = 0; i < prec; i++) { div *= 10ULL; }

  unsigned long long ip = (prec > 0) ? (all / div) : all;
  unsigned long long fp = (prec > 0) ? (all % div) : 0ULL;

  uprint_ull(ip, 0, ' ');
  if (prec > 0)
  {
    uart_putc('.');
    char tmp[16];
    int n = 0;
    unsigned long long t = fp;
    do { tmp[n++] = (char)('0' + (int)(t % 10ULL)); t /= 10ULL; } while (t != 0ULL && n < 15);
    for (int i = n; i < prec; i++) { uart_putc('0'); }
    while (n > 0) { uart_putc(tmp[--n]); }
  }
}

void uart_printf(const char *fmt, ...)
{
  va_list ap;
  va_start(ap, fmt);
  for (const char *p = fmt; *p != '\0'; p++)
  {
    if (*p != '%') { uart_putc(*p); continue; }
    p++;
    if (*p == '%') { uart_putc('%'); continue; }

    int left = 0;
    int zero = 0;
    for (;;)
    {
      if (*p == '-') { left = 1; p++; }
      else if (*p == '0') { zero = 1; p++; }
      else if (*p == '+') { p++; }
      else { break; }
    }

    int width = 0;
    while (*p >= '0' && *p <= '9') { width = width * 10 + (*p - '0'); p++; }

    int prec = -1;
    if (*p == '.')
    {
      p++;
      prec = 0;
      while (*p >= '0' && *p <= '9') { prec = prec * 10 + (*p - '0'); p++; }
    }

    int lng = 0;
    while (*p == 'l') { lng++; p++; }

    char conv = *p;
    switch (conv)
    {
      case 'd':
      case 'i':
      {
        long v = lng ? va_arg(ap, long) : (long)va_arg(ap, int);
        char pad = left ? ' ' : (zero ? '0' : ' ');
        if (v < 0) { uart_putc('-'); uprint_ull((unsigned long long)(-v), width - 1, '0'); }
        else       { uprint_ull((unsigned long long)v, width, pad); }
        break;
      }
      case 'u':
      {
        unsigned long v = lng ? va_arg(ap, unsigned long) : (unsigned long)va_arg(ap, unsigned int);
        uprint_ull((unsigned long long)v, width, left ? ' ' : (zero ? '0' : ' '));
        break;
      }
      case 'x':
      {
        unsigned long v = lng ? va_arg(ap, unsigned long) : (unsigned long)va_arg(ap, unsigned int);
        uprint_hex(v, width, zero);
        break;
      }
      case 'c':
        uart_putc((char)va_arg(ap, int));
        break;
      case 's':
      {
        const char *s = va_arg(ap, const char *);
        int n = 0;
        while (s != NULL && s[n] != '\0') { uart_putc(s[n]); n++; }
        if (!left) { for (int i = n; i < width; i++) { uart_putc(' '); } }
        break;
      }
      case 'f':
      {
        double v = va_arg(ap, double);
        uprint_float(v, prec);
        break;
      }
      default:
        if (conv != '\0') { uart_putc('%'); uart_putc(conv); }
        break;
    }
  }
  va_end(ap);
}

/**
  * @brief  The application entry point.
  * @retval int
  */
/* -------------------------------------------------------------------------
 * Streaming mode ('S'): host sends 256 int16 LE values per window
 * (I[0..127] then Q[0..127], full-scale 32768).  The board does windowing,
 * L2 normalisation, quantisation and inference - i.e. the production path,
 * with UART standing in for the ADC/DMA front end.
 * ---------------------------------------------------------------------- */
static void normalize_l2(float *x, int n)
{
  float s = 0.0f;
  for (int i = 0; i < n; i++) s += x[i] * x[i];
  s = sqrtf(s);
  if (s < 1e-12f) s = 1e-12f;
  for (int i = 0; i < n; i++) x[i] /= s;
}

static void stream_loop(void);
void dsc_stream_entry(void) { stream_loop(); }

static void stream_loop(void)
{
  static const char *names[11] = {"8PSK","AM-DSB","AM-SSB","BPSK","CPFSK","GFSK",
                                  "PAM4","QAM16","QAM64","QPSK","WBFM"};
  uint8_t rb[512];
  float win[256];
  int8_t logits[11];
  float scores[11];

  dsc_infer_init();
  DWT_Init();
  uart_printf("[STREAM] switching to 921600, 256 x int16 LE per window\r\n");
  HAL_Delay(50);
  huart3.Init.BaudRate = 921600;
  HAL_UART_DeInit(&huart3);
  HAL_UART_Init(&huart3);

  for (;;) {
    if (HAL_UART_Receive(&huart3, rb, 512, 5000) != HAL_OK) continue;

    for (int i = 0; i < 256; i++) {
      int16_t v = (int16_t)((uint16_t)rb[2 * i] | ((uint16_t)rb[2 * i + 1] << 8));
      win[i] = (float)v * (1.0f / 32768.0f);
    }

    uint32_t c0 = DWT->CYCCNT;
    normalize_l2(win, 256);
    dsc_infer_run(win, logits);
    uint32_t c1 = DWT->CYCCNT;

    float mx = -1e9f;
    for (int i = 0; i < 11; i++) { scores[i] = (float)logits[i] * 0.35008532f; if (scores[i] > mx) mx = scores[i]; }
    float sum = 0.0f;
    for (int i = 0; i < 11; i++) { scores[i] = expf(scores[i] - mx); sum += scores[i]; }
    int best = 0; mx = scores[0];
    for (int i = 1; i < 11; i++) if (scores[i] > mx) { mx = scores[i]; best = i; }
    uint32_t conf = (uint32_t)((mx / sum) * 1000.0f + 0.5f);
    uint32_t us_x100 = (uint32_t)(((uint64_t)(c1 - c0) * 100ULL) /
                                  (uint64_t)(SystemCoreClock / 1000000U));
    uart_printf("C,%d,%lu,%lu,%s\r\n", best, (unsigned long)conf,
                (unsigned long)us_x100, names[best]);
  }
}

static void replay_loop(int first);
void dsc_replay_entry(void) { replay_loop(1); }

static void replay_serve(void)
{
  uint8_t rb[256];
  int8_t in8[256];
  int8_t out[11];
  if (HAL_UART_Receive(&huart3, rb, 256, 3000) != HAL_OK) return;
  for (int i = 0; i < 256; i++) in8[i] = (int8_t)rb[i];
  dsc_infer_run_q(in8, out);
  uart_printf("L");
  for (int i = 0; i < 11; i++) uart_printf(",%d", (int)out[i]);
  uart_printf("\r\n");
}

static void replay_loop(int first)
{
  uint8_t h;
  dsc_infer_init();
  uart_printf("[REPLAY] ready\r\n");
  if (first) replay_serve();
  for (;;) {
    if (HAL_UART_Receive(&huart3, &h, 1, 3000) != HAL_OK) continue;
    if (h != 'R') continue;
    replay_serve();
  }
}
int main(void)
{
  HAL_Init();
  SystemClock_Config();          /* 550 MHz: HSI 64 -> PLL1 M=8 N=68 + FRACN 6144/8192 P=1, VOS0 */
  SystemCoreClockUpdate();
  SCB_EnableICache();
  SCB_EnableDCache();

  MX_GPIO_Init();
  MX_USART3_UART_Init();         /* USART3 PD8/PD9, 115200 8N1 */

  /* Early sanity check: proves the UART path and the float formatter work
     before any model code runs. Kept intentionally - it is the first
     evidence line in every boot log. */
  uart_printf("\r\n[UART] formatter test: int=%d uint=%u long=%lu str=%s f3=%.3f f2=%.2f\r\n",
              -42, 7U, (unsigned long)1234567UL, "OK", 1.2345f, 3.14159f);

  {
    uint8_t h;
    if (HAL_UART_Receive(&huart3, &h, 1, 1500) == HAL_OK) {
      if (h == 'R') replay_loop(1);
      if (h == 'S') stream_loop();
    }
  }
  inference_main();
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Supply configuration update enable
  */
  HAL_PWREx_ConfigSupply(PWR_LDO_SUPPLY);

  /** Configure the main internal regulator output voltage
  */
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE0);

  while(!__HAL_PWR_GET_FLAG(PWR_FLAG_VOSRDY)) {}

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSI;
  RCC_OscInitStruct.PLL.PLLM = 8;
  RCC_OscInitStruct.PLL.PLLN = 68;     /* 8 MHz x (68 + 6144/8192) = 550 MHz */
  RCC_OscInitStruct.PLL.PLLP = 1;
  RCC_OscInitStruct.PLL.PLLQ = 4;
  RCC_OscInitStruct.PLL.PLLR = 2;
  RCC_OscInitStruct.PLL.PLLRGE = RCC_PLL1VCIRANGE_3;  /* VCI = 64/8 = 8 MHz -> 8-16 MHz range */
  RCC_OscInitStruct.PLL.PLLVCOSEL = RCC_PLL1VCOWIDE;
  RCC_OscInitStruct.PLL.PLLFRACN = 6144; /* fractional part: 8 MHz x 6144/8192 = 6 MHz */
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2
                              |RCC_CLOCKTYPE_D3PCLK1|RCC_CLOCKTYPE_D1PCLK1;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.SYSCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_HCLK_DIV2;
  RCC_ClkInitStruct.APB3CLKDivider = RCC_APB3_DIV2;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_APB1_DIV2;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_APB2_DIV2;
  RCC_ClkInitStruct.APB4CLKDivider = RCC_APB4_DIV2;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_4) != HAL_OK)
  {
    Error_Handler();
  }
}

/**
  * @brief USART3 Initialization Function
  * @param None
  * @retval None
  */
static void MX_USART3_UART_Init(void)
{

  /* USER CODE BEGIN USART3_Init 0 */

  /* USER CODE END USART3_Init 0 */

  /* USER CODE BEGIN USART3_Init 1 */

  /* USER CODE END USART3_Init 1 */
  huart3.Instance = USART3;
  huart3.Init.BaudRate = 115200;
  huart3.Init.WordLength = UART_WORDLENGTH_8B;
  huart3.Init.StopBits = UART_STOPBITS_1;
  huart3.Init.Parity = UART_PARITY_NONE;
  huart3.Init.Mode = UART_MODE_TX_RX;
  huart3.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart3.Init.OverSampling = UART_OVERSAMPLING_16;
  huart3.Init.OneBitSampling = UART_ONE_BIT_SAMPLE_DISABLE;
  huart3.Init.ClockPrescaler = UART_PRESCALER_DIV1;
  huart3.AdvancedInit.AdvFeatureInit = UART_ADVFEATURE_NO_INIT;
  if (HAL_UART_Init(&huart3) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_UARTEx_SetTxFifoThreshold(&huart3, UART_TXFIFO_THRESHOLD_1_8) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_UARTEx_SetRxFifoThreshold(&huart3, UART_RXFIFO_THRESHOLD_1_8) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_UARTEx_DisableFifoMode(&huart3) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN USART3_Init 2 */

  /* USER CODE END USART3_Init 2 */

}

/**
  * @brief GPIO Initialization Function
  * @param None
  * @retval None
  */
static void MX_GPIO_Init(void)
{
  /* USER CODE BEGIN MX_GPIO_Init_1 */

  /* USER CODE END MX_GPIO_Init_1 */

  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOC_CLK_ENABLE();
  __HAL_RCC_GPIOD_CLK_ENABLE();
  __HAL_RCC_GPIOA_CLK_ENABLE();

  /* USER CODE BEGIN MX_GPIO_Init_2 */

  /* USER CODE END MX_GPIO_Init_2 */
}

/* USER CODE BEGIN 4 */

/* USER CODE END 4 */

 /* MPU Configuration */

void MPU_Config(void)
{
  MPU_Region_InitTypeDef MPU_InitStruct = {0};

  /* Disables the MPU */
  HAL_MPU_Disable();

  /** Initializes and configures the Region and the memory to be protected
  */
  MPU_InitStruct.Enable = MPU_REGION_ENABLE;
  MPU_InitStruct.Number = MPU_REGION_NUMBER0;
  MPU_InitStruct.BaseAddress = 0x0;
  MPU_InitStruct.Size = MPU_REGION_SIZE_4GB;
  MPU_InitStruct.SubRegionDisable = 0x87;
  MPU_InitStruct.TypeExtField = MPU_TEX_LEVEL0;
  MPU_InitStruct.AccessPermission = MPU_REGION_NO_ACCESS;
  MPU_InitStruct.DisableExec = MPU_INSTRUCTION_ACCESS_DISABLE;
  MPU_InitStruct.IsShareable = MPU_ACCESS_SHAREABLE;
  MPU_InitStruct.IsCacheable = MPU_ACCESS_NOT_CACHEABLE;
  MPU_InitStruct.IsBufferable = MPU_ACCESS_NOT_BUFFERABLE;

  HAL_MPU_ConfigRegion(&MPU_InitStruct);
  /* Enables the MPU */
  HAL_MPU_Enable(MPU_PRIVILEGED_DEFAULT);

}

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* Blink LD3 (PB14) fast so a hang in Error_Handler is visible on the board. */
  {
    GPIO_InitTypeDef g = {0};
    __HAL_RCC_GPIOB_CLK_ENABLE();
    g.Pin = GPIO_PIN_14;
    g.Mode = GPIO_MODE_OUTPUT_PP;
    g.Pull = GPIO_NOPULL;
    g.Speed = GPIO_SPEED_FREQ_LOW;
    HAL_GPIO_Init(GPIOB, &g);
  }
  while (1)
  {
    HAL_GPIO_TogglePin(GPIOB, GPIO_PIN_14);
    for (volatile uint32_t i = 0; i < 800000U; i++)
    {
      __NOP();
    }
  }
  /* USER CODE END Error_Handler_Debug */
}
#ifdef USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */

