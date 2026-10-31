# P5 未完成项：原因、现状与可执行方法

## 1. OTA 采集（`ota_collection/collect_ota.py`）

- 需要：HackRF One 发射 + RTL-SDR 接收，以及对应天线/衰减器。
- 现状：本机 USB 设备扫描无 HackRF/RTL-SDR（只有 ST-Link 与虚拟串口），无法采集空口数据。
- 方法（有硬件时）：`python ota_collection/collect_ota.py --freq 433.92e6 --sample_rate 2e6 --n_windows 5000 --output ota_results.npz`，
  以噪声底为参考计算 SNR，再走与测试集相同的量化/推理流程。
- 未完成不等于失败：代码与流程在仓库内，缺的是射频前端。

## 2. ULCNN 基线对比

- 需要：ULCNN 的部署工程（C 源码 + Keil 工程）才能复测 2.104 ms / 44.26 KB / 1.78×；上游**只发布了 FP32 ONNX**，没有转换/板端 harness。
- 已做（2026-10-31）：用 Zenodo `21167587` 的 `ulcnn_fp32_fixed.onnx`（53,288 B，sha256 `7c493603…`）在同一个 44,000 窗 canonical split 上评估，并跑 `stedgeai analyze` 取模型足迹 → `docs/evidence/ulcnn_eval_log.txt`、`results/ulcnn_eval.json`：
  - 该 ONNX 声明 opset 13 却用旧式「axes 作为第二输入」的 `ReduceMean`，onnxruntime 直接判 INVALID_GRAPH；脚本里把 axes 改写成属性后即可加载（可复现）。
  - 上游**未定义输入预处理**：L2-per-window（DSC 的用法）→ 12.6%（接近随机）；raw → **48.76% overall / +10 dB 73.14%**（4 种变体里最好）。即发布模型的精度显著低于论文同表 85.80%。
  - FP32 足迹：params 8,395、MACC 455,819、weights 33,580 B、activations 49,280 B —— 与论文的 INT8 部署口径（44.26 KB Flash / 36.78 KB RAM）不是同一层，不能直接比较。
- 结论：2.104 ms / 1.78× / RAM 省 24.69 KB **仍不可复现**（缺部署工程）；并且发布的 ONNX 精度也不支持 85.80% 这一行，故只在本仓库以"文献引用 + 已核对精度/足迹"呈现。

## 3. 能耗（mJ/inference）

- 需要：µA 级电流表或 Joulescope，接 NUCLEO JP5（IDD）跳线。
- 现状：无测量硬件，不伪造数字；论文 0.091 mJ 仅作文献值。
- 方法见 `docs/P5_RESOURCES.md`（积分电流 × 3.3 V / 推理次数）。

## 4. 低功耗模式

- 现状：流式/回放模式用 `HAL_UART_Receive` 轮询，CPU 不进入 WFI；
  没有电流表也无法量化收益。
- 方法：UART 改中断/DMA + `__WFI()`，或把推理做成事件驱动；先在 64 MHz 下测，
  再回 550 MHz。需要电流数据才能证明值得做。

## 5. 进一步提速

- 当前 0.904 ms ≈ 3.04 cycles/MACC（497,216 cycles @550 MHz ÷ 163,615 MACC；C1 前 0.960 ms ≈ 3.19）；已试并回退：conv1x1 全时间打包（+45%）、
  tile 16+4 累加器（+76%）、双累加器（+76%）。
- 尚可尝试：SE/GAP 归约 SIMD 化、packed 权重常驻 DTCM、dw3 与 pw 流水、
  logits 层手写展开；预计收益 5–15%，风险中等。
