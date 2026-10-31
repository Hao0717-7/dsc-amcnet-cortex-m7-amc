# Release v1.0.7 — 一键验收 / 离线 CI / QAT·per-channel·ULCNN 对照（部署口径不变）

- Tag：`v1.0.7`（annotated，指向本提交）
- 日期：2026-10-31
- 上一版：`v1.0.6`（`16b1af2`，SYSCLK 550 MHz，0.895/0.904 ms）

本版**不改变**已发布的部署口径（550 MHz、0.895/0.904 ms、golden 22/22、44k 59.2091%），
只增加验收工具链和四个对照实验，用于把「能复现什么、不能复现什么」钉死。

## 工具链（A1–A3）

| 产物 | 作用 |
|---|---|
| `stm32_deployment/tools/verify_all.ps1` | 一键 8 阶段验收：preflight → host parity → build → flash+boot → golden → replay → stream → latency → report；汇总 `stm32_deployment/results/verify_last.json`（schema `dsc-amcnet-verify-v1`），任一阶段失败退出码 1 |
| `.github/workflows/ci.yml` | `offline` job（Ubuntu，无需板子）：时钟检查 + 22 条 golden host 逐位 + `compileall` + 文档巡检；`hardware` job 仅手动触发（self-hosted Windows + `HW_CI=true`） |
| `stm32_deployment/tools/check_docs_consistency.py` | 以 `measured_latest.json` 为权威基准扫描文档陈旧值与必需字符串；历史文档（PROJECT_MEMORY / P1_BENCHMARK / P4）豁免但仍检查「必含当前值」 |
| `stm32_deployment/tools/host_sim/compare_engine.py --golden …` | 数据集无关的 22 条 golden 校验（CI 用），判据 ±1 LSB + argmax 全等 |

44k 全量验收结果：host parity bit-exact **98.775%** / argmax **99.98%**；build `Code=60,844`、0 Error；
boot **0.895 / 0.904 ms**；golden **22/22**；replay 逐位 297/300、argmax 300/300；stream 20/20、**916.6 µs**。

## 对照实验（A4–A7，均未改变部署方案）

### A4 QAT（`quantization/qat_compare.py`）

| 方案 | overall | +10 dB | ≥10 dB |
|---|---|---|---|
| FP32 checkpoint | **59.5614%** | **89.7251%** | **88.9302%** |
| PTQ INT8（当前固件） | **59.2091%** | 89.3646% | 88.4410% |
| QAT fake-quant（3 epoch / 60k 窗） | 58.8205% | 89.0041% | 88.1692% |
| QAT INT8（torch `convert()`） | 不可行 | | |

`convert()` 在 SE 的逐通道缩放 `x = x * s.unsqueeze(2)` 处报
`NotImplementedError: Could not run 'aten::mul.out' … 'QuantizedCPU' backend`；
QAT→QDQ ONNX 导出也因 `aten::fused_moving_avg_obs_fake_quant` 无 opset-13 算子失败。
**结论：DSC-AMCNet 的 QAT 在这条工具链上没有可部署出口，维持 PTQ。**

### A5 per-channel 量化（`quantization/ptq_per_channel.py`）

per-tensor 59.2091% → per-channel 59.4000%（**+0.19 pp**），+10 dB +0.09 pp，
模型 27,989 B → 31,802 B（+3,813 B），argmax 一致率 94.84%。
**结论：未达到 0.2 pp 的换用门槛，维持 per-tensor。**

### A6 ULCNN 精度 / 足迹核对（`models/eval_ulcnn.py`）

上游只发布 `ulcnn_fp32_fixed.onnx`（53,288 B，sha256 `7c493603…`），没有部署工程。核对结果：

- 该 ONNX 声明 opset 13，却在 `ReduceMean` 里用旧式 axes 输入 → onnxruntime 判 `INVALID_GRAPH`；
  脚本内把 axes 改写成属性后可加载（可复现）。
- 上游未定义输入预处理：4 种变体里最好的是 raw → **48.76% overall / +10 dB 73.14%**；
  L2-per-window（DSC 的用法）仅 12.6%。
- FP32 足迹（`stedgeai analyze`）：params 8,395、MACC 455,819、weights 33,580 B、activations 49,280 B。

**结论：2.104 ms / 1.78× / RAM 省 24.69 KB 仍不可复现（无部署件）；且发布模型的精度也不支持论文同表的 85.80%。**

### A7 ITCM 热循环（`custom_engine/DSC_AMCNet_Deployment_itcm.sct`，已回退）

| 构建 | 布局 | 板端（全零 / 真实） |
|---|---|---|
| 基线 | Flash + AXI | **0.895 / 0.904 ms** |
| 对照：scatter 文件、无 ITCM | 同基线 | **0.895 / 0.904 ms** |
| A7-a 整个 `dsc_infer.o` 进 ITCM | 15,168 B 在 ITCM | 0.906 / 0.914 ms（慢 1.2%） |
| A7-b 仅代码进 ITCM | 9,464 B 在 ITCM | 0.906 / 0.914 ms（慢 1.2%） |

**结论：引擎是数据流型、Flash ART 已藏住取指延迟，ITCM 单端口反而增加冲突 → 回退，板端已恢复 0.895/0.904 ms（golden 22/22、replay 300 复验）。**

### 数据加载顺序修正

`datasets/prepare_dataset.py:load_raw()` 改为 canonical `sorted(key=(mod, snr))` 展平，
与 `split_indices.npz` 对齐（修正后 `snr==10: n=2,219`、`>=10: n=11,039`）。
训练/评估脚本原本就用 canonical 顺序，已发布的 59.2091% / 89.36% 等数字不受影响。

## 验证命令

```powershell
# 全量一键验收（需要板子 + COM3）
powershell -ExecutionPolicy Bypass -File stm32_deployment\tools\verify_all.ps1

# 无板子也能跑的检查
python stm32_deployment/tools/check_clock_config.py
python stm32_deployment/tools/check_docs_consistency.py
python stm32_deployment/tools/host_sim/compare_engine.py --golden stm32_deployment/results/golden_vectors.json
```

## 证据与产物

- 实验日志：`docs/evidence/qat_eval_log.txt`、`docs/evidence/ptq_per_channel_log.txt`、`docs/evidence/ulcnn_eval_log.txt`
- 结构化结果：`stm32_deployment/results/qat_compare.json`、`ptq_per_channel.json`、`ulcnn_eval.json`、`verify_last.json`
- `measured_latest.json` 新增 `a4_a7_experiments_2026_10_31` 段（含全部数字，文档巡检以此为准）

## 已知限制

- QAT 不可部署（SE 逐通道缩放）；per-channel 收益 <0.2 pp。
- ULCNN 无上游部署代码，2.104 ms / 1.78× 只能引用、不能复测；发布 ONNX 精度低于论文。
- OTA（缺 HackRF+RTL-SDR）、能耗（缺电流表）、ULCNN 上板对比仍未复现，见 README「未复现 / 未验证项」。
- 550 MHz 由 HSI 分频得到，HSI ±1% ⇒ 延迟数字带 ±1% 系统不确定度。
