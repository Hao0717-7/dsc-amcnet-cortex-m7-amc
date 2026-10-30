# 自研 INT8 推理引擎（custom_engine）

目标：绕开 X-CUBE-AI 的标量 C 运行时（pack 版本 10.2.1 / ST Edge AI Core 2.2.0，运行时库 `NetworkRuntime1020_CM7_Keil.lib`，实测 1.827 ms），
在 NUCLEO-H723ZG（Cortex-M7 @544 MHz）上把 DSC-AMCNet INT8 单次推理压到 1.2 ms 以内。

## 结果

| 实现 | 单次推理 | 说明 |
|---|---|---|
| X-CUBE-AI 10.2.0（time 优化） | 1.827–1.843 ms | 标量 C 内核，约 6.1 cycles/MACC |
| **本引擎（清理调试探针后）** | **0.951 ms（全零）/ 0.960 ms（真实窗）** | N=1000；全零 min=max=0.951，真实窗 min 0.960 / max 0.962 |
| onnxruntime（PC, Python） | 11.17 ms | 仅作数值参考，非性能对比 |

比论文 1.183 ms 快 **18.8%**（0.960 / 1.183 = 0.812），相对 X-CUBE-AI 运行时快 **1.90×**。
逐层 cycle 剖析与失败的优化尝试见 `docs/P4_OPTIMIZATION.md`，最终数字与复测记录见
`stm32_deployment/results/measured_latest.json`。

数值：QPSK@10 dB 样本的 11 个 logits（int8 q）与 onnxruntime 逐位一致：
`55 31 0 48 30 34 51 64 60 60 34`（scale 0.35008532，zp 63）。

## 实现要点

- 量化域：全图激活 `zp = -128`（无符号域 u = q + 128），输入 zp = -4，SE 的 fc2 输出 zp = -5，logits zp = 63；权重/偏置 zp = 0。
- 通道两两打包成 32 位字，内联 `smlad` 做双 16 位乘加；每层用预计算 `sumw[oc]` 吸收输入零点偏移。
- requant 用 Q31 定点乘 + 移位（无逐元素 `lroundf`），四舍五入用 `rint` 语义。
- dw3 卷积的时间轴边界补零必须补 **q = -128**（实数 0），这是第一版数值漂移的根因。
- shuffle 为 `out[(r*4+g)] = in[(g*6+r)]`；SE 均值在未 shuffle 的 pool 张量上计算；GAP 在 shuffle 之后。
- 工作区只有三个静态缓冲：`A[24*128]`、`B[24*128]`、`C[24*16+64]`（层间乒乓），整个引擎 10,384 B code / 5,560 B RO（见 `docs/P5_RESOURCES.md`）。

## 文件

- `dsc_infer.c` / `dsc_infer.h`：引擎实现与接口（`dsc_infer_init` / `dsc_infer_run` / `dsc_infer_run_q`）。
- `dsc_model_data.h`：从 `quantization/outputs/dscamcnet_int8.onnx` 抽出的量化权重/偏置/scale（生成方式见 `stm32_deployment/X-CUBE-AI/`）。
- `test_sample.h`：QPSK@10 dB 参考样本（int8 q，输入 scale 0.00213504443，zp -4）。
- `dwt_measure.c` / `dwt_measure.h`：**板端实际使用**的 DWT 计时（`uint32_t *_us_x100` 接口 + `SystemCoreClock` 按 544 MHz 换算）。2026-10-30 从 `D:\DSC_AMCNet_Deployment` 原样收回仓库，只修正了头注释里一处与实现不符的旧描述。
- `DSC_AMCNet_Deployment.uvprojx` / `DSC_AMCNet_Deployment.sct`：产出 0.951/0.960 ms 的 Keil AC6 工程配置与分散加载文件；其中的相对路径（`../Src`、`../Inc`、`../Drivers`、`../Middlewares/ST/AI`）按部署工程布局书写，重建说明见 `docs/BUILD.md` §1.1。

⚠️ `stm32_deployment/firmware/dwt_measure.*` 是 **X-CUBE-AI 时代的旧版**（`float*` 接口，µs 换算用
`HAL_RCC_GetHCLKFreq()` = HCLK 272 MHz，会把时间算成 2 倍），只服务于 `firmware/inference_main.c`
那条旧路径。**自研引擎必须用本目录这两个文件**，两者不要混用。

## 工程文件的完整度（2026-10-30 补齐）

下表说明本目录文件与仓库外部署工程 `D:\DSC_AMCNet_Deployment` 的对应关系，以及 clone 后还差什么才能编译。

| 本目录文件 | 部署工程 | 说明 |
|---|---|---|
| `app_main.c` | `Src/main.c` | 逐字相同（main + 时钟 + USART3，HSI→544 MHz） |
| `app_inference_main.c` | `Src/inference_main.c` | 逐字相同（DWT benchmark + classify / replay / stream） |
| `dsc_infer.c` | `Src/dsc_infer.c` | 内容相同，仅头注释不同 |
| `dsc_infer.h` / `dsc_model_data.h` / `test_sample.h` | `Inc/`、`Src/` | 逐字相同 |
| `dwt_measure.c` / `dwt_measure.h` | `Src/`、`Inc/` | 逐字相同（头注释已修正为 SystemCoreClock） |
| `retarget_keil_h723.c` | `Src/retarget_keil.c` | 逐字相同（`$Sub$$_sys_open` 等重定向） |
| `startup_stm32h723xx.s` | `MDK-ARM/` | 逐字相同 |
| `main.h`、`stm32h7xx_hal_conf.h`、`stm32h7xx_it.h`、`stm32h7xx_it.c`、`stm32h7xx_hal_msp.c`、`system_stm32h7xx.c`、`uart_printf.h`、`inference_main.h` | `Inc/`、`Src/` | 工程胶水（CubeMX 生成或手写），2026-10-30 原样收回，逐字节相同 |
| `DSC_AMCNet_Deployment.uvprojx` / `.sct` / `.ioc` | `MDK-ARM/`、根 | Keil 工程、分散加载、CubeMX 配置（`.ioc` 已改为 HSI→544 MHz） |

仍然**不在仓库**里（由 ST 工具生成、体积大）：

- `Drivers/STM32H7xx_HAL_Driver` 与 `Drivers/CMSIS`（CubeMX 的 H7 包，约 10 MB 级）；
- `Middlewares/ST/AI`（X-CUBE-AI 运行时 `NetworkRuntime1020_CM7_Keil.lib` + 生成的 `network*.c`）；
- Keil 构建产物（`MDK-ARM/DSC_AMCNet_Deployment/` 下的 `.axf` / `.hex` / `.map` 等）；
- `uvoptx`、RTE、DebugConfig 等 IDE 本地状态。

重建步骤（细节见 `docs/BUILD.md` §1.1）：

1. 用 STM32CubeMX 打开 `.ioc` 生成 HAL 工程（已配置 HSI 64 → PLL1 M=8/N=68/P=1 → 544 MHz、USART3 PD8/PD9、VOS0、LDO）；
2. 把 `app_main.c` 重命名为 `Src/main.c`、`app_inference_main.c` 重命名为 `Src/inference_main.c`，其余 `.c` / `.h` 放入 `Src/` / `Inc/`；
3. 用 X-CUBE-AI CLI（命令见 `docs/BUILD.md`）生成并加入 AI 运行时与 `network*.c`；
4. 编译，期望启动打印 `[DWT] N=1000 mean=0.951 ms`。

## 集成

1. 将 `dsc_infer.c`、`dsc_infer.h`、`dsc_model_data.h`、`test_sample.h`、`dwt_measure.c`、`dwt_measure.h` 加入 Keil 工程（Application/AI 组）。
2. `main()` 里调用 `dsc_infer_init()`（一次性打包权重）。
3. 每次推理：`dsc_infer_run(input_f32_256, logits_int8)`；输出 logits 的 scale 为 0.35008532，zp 为 63。
4. 输入需与训练一致：整窗 L2 归一化后按 `q = round(x / 0.00213504443) - 4` 量化。
5. 计时：`DWT_Init()` → `DWT_BenchmarkN(inference_fn, 1000, &mean_x100, &min_x100, &max_x100)`
   （前 10 次为 warm-up 不计），打印 `[DWT] N=1000 mean=0.951 ms`。

## 注意

- 本引擎按 DSC-AMCNet 的精确结构硬编码（2×128 输入、3 级 DSC、SE、GAP、11 类）；换模型需重新生成权重与结构。
- 与 X-CUBE-AI 的 `network.c` 无关；当前 uvprojx 仍链接 X-CUBE-AI 运行时（供 banner / 对照用），不影响自研引擎的数值与计时。
