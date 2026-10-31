# P5 未完成项：原因、现状与可执行方法

## 1. OTA 采集（`ota_collection/collect_ota.py`）

- 需要：HackRF One 发射 + RTL-SDR 接收，以及对应天线/衰减器。
- 现状：本机 USB 设备扫描无 HackRF/RTL-SDR（只有 ST-Link 与虚拟串口），无法采集空口数据。
- 方法（有硬件时）：`python ota_collection/collect_ota.py --freq 433.92e6 --sample_rate 2e6 --n_windows 5000 --output ota_results.npz`，
  以噪声底为参考计算 SNR，再走与测试集相同的量化/推理流程。
- 未完成不等于失败：代码与流程在仓库内，缺的是射频前端。

## 2. ULCNN 基线对比

- 需要：ULCNN-simplified 的训练权重或 ONNX（仓库内没有），以及 PyTorch 训练环境
  （当前解释器没有 torch；数据集在，训练需要数小时 CPU）。
- 现状：README 中 ULCNN 的 2.104 ms / 49.25 KB / 87.55% 仍是论文值，未本地复测；
  本仓库的 0.904 ms 不能直接与论文 ULCNN 的 2.104 ms 做同机对比（后者是另一套运行时/口径）。
- 方法：clone ULCNN 结构 → 用同一 split 训练 → PTQ → 同一 550 MHz 板级基准脚本测量。

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
