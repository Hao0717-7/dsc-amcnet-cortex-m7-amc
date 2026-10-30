# P1 基准测试规范与结果（2026-10-29）

## 1. 测量方法

| 项 | 设置 |
|---|---|
| 平台 | NUCLEO-H723ZG，STM32H723ZGT6，Cortex-M7 |
| 主频 | 544 MHz（HSI 64 MHz → PLL1 M=8 / N=68 / P=1，VOS0，HPRE/2） |
| 缓存 | I-Cache + D-Cache 开启（CCR = 0x00070200） |
| 编译器 | Keil ARM Compiler 6.7，`-O3` |
| 计时 | DWT CYCCNT（`DBGMCU->CR |= TRACECKEN`，`DWT->LAR = 0xC5ACCE55`） |
| 统计 | 10 次预热丢弃，连续 1000 次取 mean / min / max |
| 引擎 | 自研 SMLAD INT8 引擎（`Src/dsc_infer.c`），与 onnxruntime 逐样本一致 |

测的是**端到端单窗口推理**：`dsc_infer_run()` 从 float 输入开始（含输入量化），到 11 个 int8 logits 结束，不含串口打印。

## 2. 结果

| 输入 | Mean | Min | Max | 折算周期 @544 MHz |
|---|---|---|---|---|
| 全零窗口 | 0.951 ms | 0.951 ms | 0.951 ms | ≈517,300 |
| **真实 QPSK(+10 dB) 窗口** | **0.960 ms** | 0.960 ms | 0.962 ms | **≈522,200** |
| 论文（X-CUBE-AI，544 MHz） | 1.183 ms | — | — | 643,552 |
| X-CUBE-AI 10.2.1 实测（同板） | 1.827 ms | — | — | ≈995,000 |

真实数据比全零慢约 **0.9%**：ReLU 分支与缓存行为随数据变化，所以**对外报数用 0.960 ms**，全零的 0.951 ms 只作对照（清理固件调试探针后的最终值；清理前为 0.958/0.968 ms，优化前为 0.967/0.975 ms，见 `docs/P4_OPTIMIZATION.md`）。

即便如此，仍比论文的 1.183 ms 快 **18.8%**（0.960 / 1.183 = 0.812）。

## 3. 复现

固件启动后自动跑两轮 benchmark（先全零、后真实样本），串口 115200 输出：

```
[BENCH] Running 1000 inferences (10 warm-up discarded) ...
[DWT] N=1000  mean=0.951 ms  min=0.951 ms  max=0.951 ms
[RESULT] ... Mean : 0.951 ms
[RESULT-SAMPLE] same engine, real QPSK(+10 dB) window
         Mean : 0.960 ms
```

对应源码：`D:\DSC_AMCNet_Deployment\Src\inference_main.c`（两段 `DWT_BenchmarkN(run_custom_once, 1000, ...)`）。
