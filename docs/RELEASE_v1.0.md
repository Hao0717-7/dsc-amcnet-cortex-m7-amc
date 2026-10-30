# v1.0.0 — 端到端复现完成（2026-10-29）

## 亮点

- **0.951 ms**（全零输入）/ **0.960 ms**（真实 QPSK +10 dB 窗口）INT8 推理，
  544 MHz Cortex-M7，I/D Cache 开；论文 1.183 ms → **快 18.8%**。
- **59.20%** 全量 44,000 窗口准确率（全 SNR），与 onnxruntime 参考 59.21% 一致；
  +10 dB **89.36%**（论文 89.5%）。
- 逐位一致率 98.8%、argmax 一致率 99.98%；22 条 golden vectors 回归全过（±1 LSB）。
- UART 流式链路：板端完成 L2→量化→推理→softmax，计算 **979 µs**/窗口。

## 关键产物

| 内容 | 位置 |
|---|---|
| 自研 INT8 引擎 | `stm32_deployment/custom_engine/dsc_infer.c/.h`、`dsc_model_data.h` |
| 固件快照 | `stm32_deployment/custom_engine/app_main.c`、`app_inference_main.c` |
| Keil 工程（仓库外） | `D:\DSC_AMCNet_Deployment` |
| 固件 hex | `D:\DSC_AMCNet_Deployment\firmware_DSC_AMCNet_INT8.hex` |
| 实测结果 | `stm32_deployment/results/`（ref_accuracy、p5_accuracy_44k、p5_stream、golden_vectors） |
| 回归脚本 | `stm32_deployment/tools/golden_test.py` |

## 文档索引

- `docs/REPRODUCE.md` 端到端复现
- `docs/BUILD.md` 工程结构与补丁清单
- `docs/P1_BENCHMARK.md` 基准方法
- `docs/P4_OPTIMIZATION.md` 优化记录（含失败实验）
- `docs/P5_REALTIME.md` / `docs/P5_ACCURACY.md` / `docs/P5_RESOURCES.md`
- `docs/P5_BLOCKED.md` 未完成项与原因

## 已知限制

- 能耗未本地测量（无电流表），保留论文 0.091 mJ 作为文献值。
- Flash/RAM 79.01/47.29 KB，高于论文 51.26/12.09 KB（双推理栈 + HAL + 调试/双模式）。
- OTA 采集、ULCNN 基线未复现（缺射频硬件与基线权重，见 `docs/P5_BLOCKED.md`）。
