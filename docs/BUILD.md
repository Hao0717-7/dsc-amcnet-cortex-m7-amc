# STM32H723ZG 部署工程说明（可复现 / 可重建）

部署工程不在本仓库内，位于 `D:\DSC_AMCNet_Deployment`（CubeMX 生成 + 手工补丁）。本文记录它由什么组成、打了哪些补丁、如何重建。

## 1. 工程结构

```
D:\DSC_AMCNet_Deployment\
├── DSC_AMCNet_Deployment.ioc        # CubeMX 配置（USART3 PD8/PD9、MDK-ARM）
├── Src\
│   ├── main.c                       # ★ 手工补丁：544 MHz 时钟、BEACON、uart_printf、回放握手
│   ├── inference_main.c             # ★ 自研引擎的 benchmark / 分类主流程
│   ├── dsc_infer.c                  # ★ 自研 SMLAD INT8 引擎
│   ├── dwt_measure.c                # DWT 计时
│   └── retarget_keil.c              # $Sub$$_sys_open + fputc 重定向
├── Inc\
│   ├── dsc_infer.h / dsc_model_data.h   # 引擎接口 + 量化参数（权重在 .c 里）
│   ├── uart_printf.h / dwt_measure.h / inference_main.h
├── Middlewares\ST\AI\
│   ├── Inc\                         # X-CUBE-AI 运行时头文件（从 pack 拷贝）
│   ├── Lib\NetworkRuntime1020_CM7_Keil.lib
│   └── App\                         # network.c / network_data*.c（仅作对照，正式路径用自研引擎）
└── MDK-ARM\
    ├── DSC_AMCNet_Deployment.uvprojx  # ★ pCCUsed 必须为 6070000::V6.7::ARMCLANG
    └── startup_stm32h723xx.s          # ★ Stack_Size / Heap_Size = 0x2000
```

### 1.1 仓库内收口的真实源码（2026-10-30）

板端 0.905/0.914 ms 的固件（C1 内核优化后）与本文描述的部署工程同源；为可复现，已把当时实际编译的关键文件原样收回
`stm32_deployment/custom_engine/`：

| 文件 | 来源 | 说明 |
|---|---|---|
| `dwt_measure.c` / `dwt_measure.h` | `D:\DSC_AMCNet_Deployment` | 唯一能产生 0.905/0.914（及 C1 前 0.951/0.960）的计时实现：`uint32_t *_us_x100` 接口 + `SystemCoreClock` 按 544 MHz 换算。仓库 `stm32_deployment/firmware/dwt_measure.*` 是 X-CUBE-AI 时代旧版（`float*` + `HAL_RCC_GetHCLKFreq()` = HCLK 272 MHz），会把时间算成 2 倍，**两者不要混用** |
| `DSC_AMCNet_Deployment.uvprojx` | 同上 | Keil AC6 6.7 工程：`pCCUsed 6070000::V6.7::ARMCLANG`、Stack/Heap 各 0x2000，链接 `NetworkRuntime1020_CM7_Keil.lib`（X-CUBE-AI 10.2.0） |
| `DSC_AMCNet_Deployment.sct` | 同上 | 分散加载文件（默认分配，无特殊段） |
| `app_main.c` / `app_inference_main.c` | 与部署工程 `Src/main.c`、`Src/inference_main.c` 逐字一致 | uvprojx 按 `../Src`、`../Inc`、`../Drivers`、`../Middlewares/ST/AI` 相对路径引用，从仓库重建时需把这些文件放回部署工程对应位置 |

`DSC_AMCNet_Deployment.ioc` 已从旧的 192 MHz（PLL1 DIVM1=4 / DIVN1=12，VCO 输入 16 MHz）改为与 `main.c` 一致的 544 MHz（DIVM1=8 / DIVN1=68，VCO 输入 8 MHz；HCLK 272 MHz、APB 136 MHz）。注意 `.ioc` 没有 VOS0 / FLASH_LATENCY 条目，重新 Generate 后仍必须按 §2 恢复 `main.c` 里的补丁。

**CubeMX 重新生成后必查清单**（也可直接跑 `python stm32_deployment/tools/check_clock_config.py`，失败会指回本节）：

1. `.ioc` 侧：PLL1 源 = HSI、`DIVM1=8`、`DIVN1=68`、`DIVP1=1` → SYSCLK 544 MHz；HPRE `/2` → HCLK 272 MHz；APB1-4 `/2` → 136 MHz。
2. `main.c` 侧：`HAL_PWREx_ConfigSupply(PWR_LDO_SUPPLY)`、`__HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE0)`、等待 `PWR_FLAG_VOSRDY`、`FLASH_LATENCY_3`、`MX_USART3_UART_Init()` 仍在。
3. 计时文件：必须是 `custom_engine/dwt_measure.h` 的 `uint32_t *_us_x100` 接口（`firmware/dwt_measure.h` 是旧 `float*` 版，会把时间算成 2 倍）。
4. `uvprojx`：`<pCCUsed>6070000::V6.7::ARMCLANG</pCCUsed>`、`<uAC6>1</uAC6>`、Stack/Heap 各 `0x2000`。
5. 串口：NUCLEO-H723ZG 用 `USART3 PD8/PD9`（ST-LINK VCP → COM3）；`USART1 PA9/PA10` 是另一块自制核心板 LXB723ZG-P1 的调试排针。
6. 烧录镜像路径：`D:\DSC_AMCNet_Deployment\MDK-ARM\DSC_AMCNet_Deployment\DSC_AMCNet_Deployment.hex`（项目根那份 `firmware_DSC_AMCNet_INT8.hex` 是旧构建，不要用）。

## 2. CubeMX 重新生成会丢什么

`Project → Generate Code` 会**覆盖** `Src/main.c`，以下补丁必须重新打：

| 位置 | 补丁 |
|---|---|
| `SystemClock_Config()` | HSI 64 → PLL1 M=8 / N=68 / P=1 → 544 MHz，VOS0，HPRE/2，APB /2，FLASH_LATENCY_3 |
| `main()` | `SystemCoreClockUpdate()`、`SCB_EnableICache()`、`SCB_EnableDCache()`、`MX_USART3_UART_Init()` 后调 `inference_main()` |
| `main()` 顶部 | `uart_printf()` 实现（stdio-free 格式化）、`BEACON()` 闪灯、回放握手（可选） |
| `uvprojx` | `<pCCUsed>6070000::V6.7::ARMCLANG</pCCUsed>` + `<uAC6>1</uAC6>` 放在 `<Target>` 层 |
| `startup_stm32h723xx.s` | `Stack_Size 0x2000`、`Heap_Size 0x2000` |

**建议**：改 `.ioc` 后不要直接 Generate，先备份 `Src/main.c`、`Src/inference_main.c`、`Src/dsc_infer.c`，生成后恢复。

## 3. 模型代码（仅作对照）

`Middlewares/ST/AI/App/` 里的 `network.c` 等由 `stedgeai generate` 生成（float32 I/O，`--optimization time`），用于和自研引擎对比。正式推理路径不再调用它。

生成命令：

```
stedgeai generate --target stm32h7 --name network ^
  --model D:\dsc-amcnet-cortex-m7-amc\quantization\outputs\dscamcnet_int8.onnx ^
  --input-data-type float32 --output-data-type float32 ^
  --compression none --optimization time --output <dir>
```

## 4. 构建与烧录

```powershell
$uv = "D:\APP\Keil5\UV4\UV4.exe"
$proj = "D:\DSC_AMCNet_Deployment\MDK-ARM\DSC_AMCNet_Deployment.uvprojx"

& $uv -b $proj -j0 -o build.log     # 编译（exit 1 仅代表有 warning）
& $uv -f $proj -o flash.log         # 烧录（ST-Link，Verify OK）
```

串口 `COM3`，115200 8N1。

## 5. 验收清单

1. 串口出现 `[UART] formatter test: int=-42 ...`
2. `[AI] weights=5804 B (flash) | activations=4488 B | io=1024+44 B`
3. `[DWT] N=1000 mean=0.905 ms`（全零）与 `[RESULT-SAMPLE] Mean : 0.914 ms`
5. 回放模式：上电 1.5 s 内发 `R`，出现 `[REPLAY] ready`
