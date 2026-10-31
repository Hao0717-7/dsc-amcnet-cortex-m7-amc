# 本仓库实测 vs 官方 README 数据对照

- 对照对象：官方发布包自带 README `_official/repo/DSC-AMCNet/README.md` 的
  “Key Results (Locked, Silicon-Measured)” 与 “Reproducing Paper Claims” 两节
- 实测时间：2026-10-29 22:42–22:53（CST），板子 COM3 在线，固件为 15:52 构建
- 结构化数据：`stm32_deployment/results/compare_official_readme.json`
- 原始证据：`stm32_deployment/results/evidence/serial_boot_run_2026-10-29.log`、
  `stm32_deployment/results/p0_replay_compare.json`

## 1. 主表（DSC-AMCNet INT8）

| 指标 | 官方 README | 本次实测 | 差异 / 说明 |
|---|---|---|---|
| 延迟（DWT、实板） | 1.183 ms | **0.895 ms**（全零基准）/ **0.904 ms**（真实窗，550 MHz，C1 内核） | **快 23.6%**（2026-10-31 起本机同为 550 MHz，可直接比较） |
| 连续分类延迟（串口） | 1.183 ms | **904.4 µs**（550 MHz；C1 时 914.5 µs，C1 前 960.63 µs） | 快 23.6%，与上面 DWT 基准自洽 |
| Flash | 51.26 KB | **78.23 KB**（ROM 80,104 B） | +26.97 KB。口径不同：本次是完整固件；模型权重只有 5.67 KB |
| RAM | 12.09 KB | **47.29 KB**（RW+ZI 48,424 B） | +35.2 KB。含 2×8 KB 栈堆 + 引擎工作区；官方口径未公开 |
| High-SNR 精度（≥10 dB） | 85.80% | FP32 **88.93%** / INT8 **88.44%**（n=11,039） | INT8 高 2.6 pp，但数据集划分与训练轮次不同，非严格同口径 |
| +10 dB 精度 | —（未给） | FP32 89.73% / INT8 89.36%（n=2,219） | 论文 Table II 常引用的口径 |
| 全 SNR 精度 | —（未给） | FP32 59.56% / INT8 59.21%（44,000 窗） | INT8 与 `results/ref_accuracy.json` 逐位一致 |
| PTQ 精度损失 | < 0.5 pp | **0.37 pp**（overall / +10 dB），0.49 pp（≥10 dB） | 符合官方声明 |
| 参数 / 权重 | 论文附录 4,961 | ONNX FP32 init 4,793 / INT8 init 4,869 | 统计口径不同（含/不含偏置、QDQ 常量） |
| 内核时钟 | 550 MHz | **550 MHz**（HSI 64 → PLL1 M=8 / N=68 + FRACN 6144/8192，`FLASH_LATENCY_4`；`CPU_FREQ_BOOST` 已使能） | 一致 |
| X-CUBE-AI | 10.2.0 / ST Edge AI 2.2.0 | banner：X-CUBE-AI **10.2.0** | 一致 |
| ULCNN 加速比 | 1.78× | **无法验证** | 仓库无 ULCNN 权重/代码 |
| OTA 精度 | 82.76% | **无法验证** | 本机无 HackRF / RTL-SDR |
| 板端 logits 一致性 | —（未给） | golden **22/22**；300 窗回放 argmax **300/300**、全等 297、MAD 0.0018 | 逐位级一致性证据 |

## 2. 板端本轮实跑记录

| 测试 | 结果 |
|---|---|
| 冷启动 benchmark（复位后自动跑 1000 次） | 全零 `0.895/0.895/0.895 ms`；真实 QPSK(+10 dB) `0.904/0.904/0.904 ms` |
| 连续分类 | `Pred: QAM16 | Conf: 0.645 | Latency: 960.x µs`，517 帧均值 960.63 µs |
| golden 22 条向量 | **22/22 PASS**（差距 ≤1 LSB） |
| 300 窗回放（板端 vs ORT） | n_ok 300、argmax 一致 300、logits 全等 297、板端 62.00% = 参考 62.00% |
| 20 窗实时流（921600） | 20/20、板端 65.0%、窗口内计算 972.2 µs（971.0–973.4）、往返 7.5 ms |

## 3. 结论

1. **延迟是明确占优项**：0.904 ms（真实窗，550 MHz）对官方 1.183 ms，快 23.6%；同频直接可比。
   差异主因是推理路径不同：本仓库用自研 SMLAD INT8 引擎，官方指标来自 X-CUBE-AI 运行时（本地同运行时实测 1.827 ms）。
2. **Flash/RAM 数字不能直接比**：官方 51.26 KB / 12.09 KB 未给统计口径；本次 78.23 KB / 47.29 KB 是完整固件，
   其中 X-CUBE-AI 运行时 + 生成网络约 31 KB、自研引擎约 12 KB、HAL/应用约 10 KB，模型权重仅 5.67 KB、激活 4.38 KB。
   若要同口径比较，应拿“去掉引擎与应用的纯 X-CUBE-AI 版本”或对方 map 文件。
3. **精度略高但不构成“复现超出”结论**：本仓库 split 非按类分层、训练为本地重训，官方权重（Zenodo）在本地只有 +
   10 dB 26% 精度（已记录在 `models/checkpoints/README.md`），因此 88.4% vs 85.8% 属不同训练产物，只能说明本地链路有效。
4. **一致性证据比官方 README 更细**：官方只给延迟/资源/精度三行；本仓库额外有 golden 22/22、44,000 窗逐位一致 98.80%、
   300 窗板端回放 argmax 100%、实时流 20/20。这些是复现“板端确实跑通”的直接证据。
5. **两项官方指标无法验证**：ULCNN 1.78× 加速与 OTA 82.76%，原因分别是缺 ULCNN 产物、缺射频硬件；不伪造数据。

## 4. 复现命令（本轮实际执行）

```bash
# 离线：INT8/FP32 44k 精度 + ONNX/资源清点（写 %TEMP%\\dsc_compare_official.json）
python stm32_deployment/tools/measure_vs_official.py

# 板端（COM3，115200 进入 replay）
python stm32_deployment/tools/golden_test.py --port COM3
python stm32_deployment/tools/p0_replay.py  --n 300 --port COM3

# 板端实时流（切 921600，跑完需复位）
python stm32_deployment/tools/live_stream.py --n 20 --port COM3

# 复位（Keil CLI，等价于按复位键；不重新编译）
"D:/APP/Keil5/UV4/UV4.exe" -f "D:/DSC_AMCNet_Deployment/MDK-ARM/DSC_AMCNet_Deployment.uvprojx"
```
