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
3. `[DWT] N=1000 mean=0.951 ms`（全零）与 `[RESULT-SAMPLE] Mean : 0.960 ms`
5. 回放模式：上电 1.5 s 内发 `R`，出现 `[REPLAY] ready`
