# DSC-AMCNet: Deployment-Guided Automatic Modulation Classification on Bare-Metal MCU-Class IoT Devices

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21167587.svg)](https://doi.org/10.5281/zenodo.21167587)

Official code and artifacts for the paper:

> **"DSC-AMCNet: Deployment-Guided Automatic Modulation Classification on Bare-Metal MCU-Class IoT Devices"**  
> *IEEE Internet of Things Journal (under review), 2026*

---

## Overview

DSC-AMCNet is a lightweight neural network for **Automatic Modulation Classification (AMC)** designed for deployment on bare-metal embedded hardware. This repository provides everything needed to reproduce all experimental results reported in the paper, including:

- Model training and evaluation (RML2016.10a)
- Post-Training Quantization (PTQ) to INT8 with X-CUBE-AI-compatible backend
- Deployment on STM32H723ZG (ARM Cortex-M7, 550 MHz) with DWT cycle-accurate latency measurement
- Four-tier over-the-air (OTA) validation (HackRF One TX → RX loopback via SMA attenuator; live FM/AM broadcast reception)

---

## Key Results (论文报告值 / paper-reported)

| Model | Latency (ms) | Flash (KB) | RAM (KB) | INT8 Acc. @+10 dB | PTQ Degr. |
|---|---|---|---|---|---|
| DSC-AMCNet INT8 | **1.183** | **51.26** | **12.09** | **86.92%** | **0.09 pp** |
| ULCNN-simplified INT8 | 2.104 | 49.25 | 36.78 | 87.55% | 1.95 pp |
| **DSC vs ULCNN** | **1.78× faster** | — | **−24.69 KB** | within ±1.4 pp Wilson CI | **22× lower** |

> **引用时必须注明来源。** 上表是 **IEEE IoT-J 稿**（Zenodo `21167587`，v2.0.0）的报告值，不是本仓库实测。
> 同作者的 **TCCN 在审稿**（Zenodo `20579317`，v1.0.0，官方仓库 `_official/repo/DSC-AMCNet` 的 README）报告的是另一套数：85.80% High-SNR、Flash 51.26 KB、RAM 12.09 KB、ULCNN 44.26 KB / 36.78 KB、加速 1.78×、RAM 省 24.69 KB、OSNR 口径 550 MHz。**两稿数字不同，不要混用。**
> 能耗 0.091 mJ 为论文口径，本地无电流表、未复测。
>
> The table above lists the **IoT-J manuscript** (Zenodo `21167587`, v2.0.0) reported values, not this repository's reproduction. The companion **TCCN submission** (Zenodo `20579317`, v1.0.0, whose README is mirrored under `_official/repo/DSC-AMCNet/`) reports a different set (85.80% High-SNR, ULCNN 44.26 KB, 24.69 KB RAM saving, 550 MHz). Do not mix the two.

### 本仓库实测（2026-10，NUCLEO-H723ZG）

| 项 | 实测 | 说明 |
|---|---|---|
| 推理时延 | **0.895 ms**（全零）/ **0.904 ms**（真实 QPSK 窗口） | 自研 SMLAD INT8 引擎，**550 MHz**（HSI 64 → PLL1 M=8 / N=68 + FRACN 6144/8192，`FLASH_LATENCY_4`，`CPU_FREQ_BOOST` 已使能），I/D Cache 开；内核为 2026-10-30 C1 版（SMULL requant + dw3 无分支，544 MHz 时为 0.905/0.914 ms） |
| 对照 | X-CUBE-AI 10.2.1 同模型 1.827 ms | 官方运行时是标量 C 内核，已确认是上限 |
| 全量精度 | **59.20%**（44k locked test，全 SNR） | onnxruntime 参考 59.21%，逐位一致 98.8% |
| +10 dB 精度 | **89.36%** | 与论文 89.5% 一致 |
| Flash / RAM | 78.23 KB / 47.29 KB | 双推理栈 + HAL（C1 后 ROM 80,104 B，原 80,904 B）；组成见 `docs/P5_RESOURCES.md` |
| 能耗 | 未本地复测 | 缺电流测量硬件，保留论文 0.091 mJ 作为文献值 |

> 2026-10-31 时钟升到 550 MHz 后更新（0.895/0.904 ms、23.6% vs 论文、2.02× vs 运行时、ROM 78.23 KB；544 MHz 时的 C1 读数为 0.905/0.914 ms）。
>
> 历史上：2026-10-29 重新复核、2026-10-30 C1 内核优化后更新（0.905/0.914 ms、ROM 78.23 KB）：FP32/INT8 44k 精度重算与 `results/ref_accuracy.json` 完全一致；板端 golden 22/22、300 窗回放 argmax 300/300；资源数取 C1 后构建 map（22:33，Code 60,844 / RO 16,236 / RW 8,452 / ZI 39,972）。全部数值与来源见 `stm32_deployment/results/measured_latest.json`。

复现入口：`docs/REPRODUCE.md`；基准方法 `docs/P1_BENCHMARK.md`；优化记录 `docs/P4_OPTIMIZATION.md`；
实时链路 `docs/P5_REALTIME.md`；全量精度 `docs/P5_ACCURACY.md`；回归向量 `stm32_deployment/results/golden_vectors.json`。

---

## Repository Structure

```
DSC-AMCNet/
├── models/
│   ├── dsc_amcnet.py           # Model architecture (PyTorch)
│   ├── train.py                # Training script (RML2016.10a)
│   └── checkpoints/
│       └── README.md           # Download links (Zenodo)
├── datasets/
│   ├── prepare_dataset.py      # Dataset loading + locked split
│   └── README.md               # Dataset acquisition instructions
├── quantization/
│   ├── ptq_quantize.py         # PTQ pipeline (fbgemm → ONNX → X-CUBE-AI)
│   ├── validate_backend.py     # qnnpack vs fbgemm validation
│   └── qat_experiment.py       # QAT experiment (documented failure)
├── stm32_deployment/
│   ├── firmware/
│   │   ├── inference_main.c    # STM32 inference loop
│   │   └── dwt_measure.c       # DWT cycle-accurate timing
│   └── README.md               # STM32CubeIDE setup instructions
├── hardware_characterization/
│   ├── measure_latency.py      # Serial latency logger
│   └── analyze_flash_ram.py    # Flash/RAM parser (ST Edge AI Core)
├── ota_collection/
│   └── collect_ota.py          # OTA capture + noise-floor-referenced SNR
├── supplementary/
│   ├── appendix_a_architecture.md   # Layer-by-layer specification
│   ├── appendix_b_per_modulation.md # Per-class accuracy breakdown
│   └── appendix_c_qat_failure.md    # QAT failure analysis
├── requirements.txt
├── environment.yml
└── README.md
```

---

## Quick Start

### 1. Environment Setup

```bash
conda env create -f environment.yml
conda activate dsc-amcnet
```

Or with pip:
```bash
pip install -r requirements.txt
```

### 2. Dataset

Download RML2016.10a from [DeepSig](https://www.deepsig.ai/datasets) and place the file at:
```
datasets/RML2016.10a_dict.dat
```

See `datasets/README.md` for full instructions.

### 3. Train DSC-AMCNet

```bash
python models/train.py \
    --data datasets/RML2016.10a_dict.dat \
    --split datasets/split_indices.npz \
    --epochs 100 \
    --output models/checkpoints/dscamcnet_best.pt
```

Expected result: **89.73% FP32 accuracy** at +10 dB on the locked split
(measured locally, n = 2,219, with the genuine DeepSig I/Q pickle and the
archived `split_indices.npz`; the paper's Table II claim of 87.01% could not
be reproduced from the released artifacts). The INT8 export of the same
weights scores 89.36% at +10 dB. See `CHANGELOG.txt` for the provenance work
behind these numbers.

### 4. Quantize to INT8

```bash
python quantization/ptq_quantize.py \
    --checkpoint models/checkpoints/dscamcnet_best.pt \
    --data datasets/RML2016.10a_dict.dat \
    --split datasets/split_indices.npz \
    --output quantization/outputs/dscamcnet_int8.onnx
```

> **Critical:** Uses `fbgemm` backend (signed INT8, X-CUBE-AI compatible). See `quantization/validate_backend.py` for why `qnnpack` must NOT be used.

### 5. STM32 Deployment

See `stm32_deployment/README.md` for:
- X-CUBE-AI model import workflow
- STM32CubeIDE project configuration
- Flashing and running DWT latency measurements

### 6. OTA Evaluation

```bash
python ota_collection/collect_ota.py \
    --freq 433.92e6 \
    --sample_rate 2e6 \
    --n_windows 5000 \
    --output ota_results.npz
```

---

## Reproducing Paper Claims（对照官方 README 的完成状态）

官方仓库 README 列了 5 条 "Reproducing Paper Claims"。本表把每条映射到本仓库的脚本/产物，并如实标注状态：

| 官方 claim | 官方要求产物 | 本仓库状态 | 产物 / 证据 |
|---|---|---|---|
| FP32 85.80%（SNR≥10，`train.py` eval log） | Val acc ≥ 85.5% | ✅ 用 `models/eval_checkpoint.py` 加载仓库内 `.pt` 跑 44k：+10 dB **89.73%**、≥10 dB **88.93%**（n=11,039）、整体 59.56%；与 FP32 ONNX argmax 一致 44,000/44,000（max logit diff 3.8e-05）。权重是本地重训（Zenodo 官方 `.pt` 已损坏，见下），故不比 85.80% 这一具体数字 | `docs/evidence/fp32_eval_log.txt`、`models/checkpoints/`（权重 sha256 `f7a05c25…`） |
| INT8 Flash 51.26 / RAM 12.09 KB（ST Edge AI Core **analyze report**） | Analyze report | ⚠️ 口径不同：官方为模型+运行时估算；本仓库为完整可烧录固件 **78.23 / 47.29 KB**（map 实测），另有 analyze report 的模型足迹（weights 5,804 B / activations 7,488 B / ram 7,755 B / MACC 163,081） | `docs/evidence/stedgeai_analyze/dscamcnet_int8_analyze_report.txt`、`docs/P5_RESOURCES.md` |
| INT8 延迟 1.183 ms（`measure_latency.py` 串口 CSV） | Serial log / CSV | ✅ 同脚本、同判据实测 **0.895 ms（全零）/ 0.904 ms（真实窗）**，比论文快 **23.6%**（本机同为 550 MHz，可直接比较） | `docs/evidence/latency_dscamcnet_int8.csv`、`docs/P1_BENCHMARK.md` |
| OTA 82.76%（`collect_ota.py` `.npz`） | `.npz` 结果 | ❌ 未复现：本机无 HackRF One + RTL-SDR（脚本本身还缺整窗 L2 归一化） | 仅论文声明，无产物 |
| PTQ 掉点 <0.5%（`ptq_quantize.py` eval log） | Eval log | ✅ 实测整体 **0.35 pp**（59.5614%→59.2091%，26,207→26,052 / 44,000）；+10 dB 89.73%→89.36%（0.37 pp）；≥10 dB 0.49 pp（仍 < 0.5）。另：从 `.pt` 重跑 PTQ 可复现出同尺寸/同结构 ONNX，98 个 initializer 中 96 个逐字节相同，精度逐位一致（两处 scale 常量有 1e-7 级舍入差） | `docs/evidence/ptq_eval_log.txt`、`docs/evidence/ptq_regen_log.txt` |

补充状态：

| 项 | 状态 | 说明 |
|---|---|---|
| ULCNN-simplified 对照（2.104 ms / 1.78×） | ❌ 未复现 | 仓库无 ULCNN 部署代码；Zenodo 有 `ulcnn_fp32_fixed.onnx`（53,288 B）但需转 INT8 + 上板才能测 |
| 550 MHz | ✅ 本仓库 550 MHz | 固件时钟 HSI 64 MHz → PLL1 M=8 / N=68 + FRACN 6144/8192 → **550 MHz**（HCLK 275 / APB 137.5），`FLASH_LATENCY_4`；片内 `CPU_FREQ_BOOST` 已使能（否则 VOS0 上限 520 MHz），与官方口径一致 |
| 自研引擎 0.904 ms（优于论文） | ✅ | `stm32_deployment/custom_engine/`，**2.02×** vs X-CUBE-AI 运行时 1.827 ms |
| 板端全量精度 59.20% | ✅ | 44,000 窗，逐位一致 98.80%，argmax 一致 99.98% |
| 22 条 golden 回归向量 | ✅ | `golden_test.py`，±1 LSB，22/22 |
| X-CUBE-AI 版本 | ⚠️ 实测 pack **10.2.1** | 固件 banner 字符串硬编码 "10.2.0"（`Src/inference_main.c:202`），以 pack 版本为准 |
| PTQ 后端 | ✅ 等价实现 | 官方写 "fbgemm backend"（= signed INT8、zp 0）；本仓库用 `onnxruntime.quantization.quantize_static + QuantType.QInt8`，量化方案等价 |

### 参数口径说明（4,961 / 4,793 / 4,869）

仓库里出现三个"参数个数"，统计对象不同，**不能互相校验**：

| 数字 | 口径 | 来源 |
|---|---|---|
| **4,961** | PyTorch 可训练参数（含 6 个 BN 的 γ/β 共 336 个） | `models/dsc_amcnet.py`（论文 Table V 口径） |
| **4,793** | BN 折进相邻卷积后的权重+偏置元素数（4,625 原有 + 168 折叠偏置），按 4 B/参数折算 = 19,172 B = 18.72 KiB | `stedgeai analyze` 报告（`docs/evidence/stedgeai_analyze/`） |
| **4,869** | INT8 QDQ ONNX 的 98 个 initializer 元素总数 = 4,520 个 int8 权重 + 273 个 int32 偏置 + 38 个 per-tensor scale + 38 个 zero-point 元素 | `quantization/outputs/dscamcnet_int8.onnx`（93 节点） |

换算关系：`4,793 = 4,961 − 336(BN) + 168(折叠偏置)`；`4,869 = 4,793 + 76（38 个 scale + 38 个 zero-point）`。三者都对，只是统计层面不同。

### 工程一致性自动检查（三个脚本，均可离线跑）

```bash
python stm32_deployment/tools/check_clock_config.py        # .ioc + main.c 时钟/电源/串口
python stm32_deployment/tools/check_docs_consistency.py    # 文档口径 ↔ measured_latest.json
python stm32_deployment/tools/host_sim/compare_engine.py --golden stm32_deployment/results/golden_vectors.json
```

1. **时钟**：校验 `.ioc` 里的 PLL/分频（HSI 64 MHz → M=8、N=68 + FRACN 6144/8192 → **550 MHz**、HCLK 275、APB 137.5）与
   **CubeMX 不写进 .ioc 的四项**（`PWR_LDO_SUPPLY`、`PWR_REGULATOR_VOLTAGE_SCALE0`、`VOSRDY` 等待、`FLASH_LATENCY_4`，外加 `USART3`）。
2. **文档口径**：检查 JSON 内部三段（P2 / C1 / 基线）自洽（资源与 C1 后构建一致、550 MHz 必须快于 544 MHz），
   并要求 README / PROJECT_STATUS / REPORT.html 等当前口径文档含最新数字、不残留旧数字（历史文件豁免）。
3. **引擎数值**：用 zig 现场编译主机侧引擎，对提交进仓库的 22 条 golden 向量做 ±1 LSB + argmax 比对
   （**不需要 225 MB 数据集**，CI 也跑这个）。

本机全流程（含板子）用 `stm32_deployment/tools/verify_all.ps1`，见 `docs/BUILD.md` §4.1；
GitHub Actions 的离线 CI 在 `.github/workflows/ci.yml`（每次 push/PR 跑上面三件 + Python 语法检查）。

### 未复现 / 未验证项（如实声明）

下面三项**没有本地复现**，只有论文 / 官方 README 的报告值。本仓库任何位置都不把它们当成本仓库的实测结果，也不用它们计算"加速比"或"能耗指标"：

| 项 | 论文 / 官方值 | 未复现原因 | 本仓库能提供的 |
|---|---|---|---|
| OTA 过采样采集精度（`collect_ota.py`） | 82.76%（85.80%→82.76%，多 10× 样本） | 需要 HackRF One + RTL-SDR 采集真实空口 IQ；本机无射频硬件，且 `ota_collection/collect_ota.py` 尚未做整窗 L2 归一化 | `ota_collection/` 脚本框架 + `docs/P5_BLOCKED.md` 的缺口说明 |
| ULCNN-simplified 对照（2.104 ms / 1.78× / RAM 省 24.69 KB） | 2.104 ms，1.78× | 上游没有 ULCNN 部署代码；Zenodo 仅有 FP32 ONNX（`ulcnn_fp32_fixed.onnx`，53,288 B），要同口径对比需自行转 INT8、写引擎并上板 | 仅作文献引用；**不声称**本仓库的 1.90× 是"对 ULCNN 的 1.78×" |
| 能耗（0.091 mJ / inference） | 0.091 mJ | 需要电流表；NUCLEO-H723ZG 的 `JP4` 跳线可串入 mA 表（原理图 sheet 4 已核对），但本机无万用表 | 测量路径说明 + `docs/P5_BLOCKED.md`；**不伪造数字** |

同类"声明值 vs 实测值"还有：550 MHz（本仓库 HSI 64→PLL1 实测 **550 MHz**）、X-CUBE-AI 10.2.0（实测 pack **10.2.1**）——均以本仓库实测为准。详细对照见 [`docs/COMPARISON_vs_OFFICIAL_README.md`](docs/COMPARISON_vs_OFFICIAL_README.md)。

---

## Pre-trained Weights

两个 Zenodo 记录（同一作者、两篇稿件）都归档了权重、ONNX 与 locked split：

| 记录 | 版本 | 对应稿件 | 文件 |
|---|---|---|---|
| [`10.5281/zenodo.21167587`](https://doi.org/10.5281/zenodo.21167587) | v2.0.0 | IoT-J（本仓库 README 引用） | `dscamcnet_best.pt`、`dscamcnet_int8.onnx`、`ulcnn_fp32_fixed.onnx`、`split_indices.npz`、repo zip |
| [`10.5281/zenodo.20579317`](https://doi.org/10.5281/zenodo.20579317) | v1.0.0 | TCCN（`_official/repo/DSC-AMCNet`） | 同上 |

> ⚠️ **不要直接使用 Zenodo 的 `dscamcnet_best.pt`**：两个记录的该文件完全相同（37,397 B，md5 `56c214ad…` / sha256 `4827d8dc…`），在正确 I/Q 数据与 locked split 上 +10 dB 只有 **26.05%**——与论文 85.80% 不符（详见 [`models/checkpoints/README.md`](models/checkpoints/README.md) 的 provenance 段与 [`docs/P0_ACCURACY.md`](docs/P0_ACCURACY.md)）。本仓库的 **89.73%** 来自本地重训权重 `f7a05c25…`（36,437 B，已随仓库提供）。
> `dscamcnet_int8.onnx`（58,169 B）与 `ulcnn_fp32_fixed.onnx`（53,288 B）仍在 Zenodo 上，可用于对照实验（ULCNN 需自行转 INT8 + 上板，本仓库未复现）。

---

## Environment

| Component | Version |
|---|---|
| Python | 3.10+（开发/实测：3.14.5） |
| PyTorch | 2.x（仅训练与 `.pt` 评估需要；量化→上板链路不需要） |
| ONNX / ONNX Runtime | 1.16+ / 1.18+（实测 1.23.2 / 1.31.1） |
| STM32CubeIDE | 1.15.x（官方流程；本仓库实际用 Keil MDK 5.24a + AC6 6.7 编译，见 `docs/BUILD.md`） |
| X-CUBE-AI / ST Edge AI Core | 10.2.1 (pack) / 2.2.0 |
| Hardware | STM32H723ZG Nucleo；OTA 对照另需 HackRF One + RTL-SDR Blog V3（本仓库未做 OTA） |

---

## Citation

If you use this work, please cite:

```bibtex
@article{dscamcnet2026,
  title   = {{DSC-AMCNet}: Deployment-Guided Automatic Modulation
             Classification on Bare-Metal {MCU}-Class {IoT} Devices},
  author  = {Mishra, Suryaprakash},
  journal = {IEEE Internet of Things Journal},
  year    = {2026},
  note    = {Under review}
}
```

---

## License

This project is released under the [MIT License](LICENSE).
