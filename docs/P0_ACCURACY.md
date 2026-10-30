# P0 精度验收结果（2026-10-28）

## 1. 参考精度（onnxruntime，locked test split，44,000 样本）

- 模型：`quantization/outputs/dscamcnet_int8.onnx`（INT8 QDQ，batch=1）
- 预处理：per-window L2 归一化（与训练/固件一致），无 min-max 缩放
- 脚本：`stm32_deployment/tools/p0_accuracy.py`；结果：`stm32_deployment/results/ref_accuracy.json`

| 指标 | 值 |
|---|---|
| overall acc | **59.21%** |
| +10 dB acc | **89.36%**（论文 89.50/89.55） |
| +18 dB acc | 87.33% |

逐 SNR：-20 dB 9.05% / -16 dB 10.48% / -12 dB 15.70% / -8 dB 37.04% / -4 dB 63.48% / 0 dB 81.51% / +4 dB 86.89% / +8 dB 88.26% / +10 dB 89.36% / +14 dB 88.09% / +18 dB 87.33%。

## 2. 板端回放验证（自研 SMLAD 引擎）

- 固件：`D:\DSC_AMCNet_Deployment`，回放模式（UART：`R` + 256 字节 int8 → `L,q0..q10`）
- 脚本：`stm32_deployment/tools/p0_replay.py --n 5000 --flash --seed 2`
- 参考侧使用**反量化后的 q 输入**喂 ORT，保证与板端看到的数据完全一致（公平对比）
- 结果文件：`stm32_deployment/results/p0_replay_compare.json`

| 指标 | 值 |
|---|---|
| 成功回放样本 | 4,995 / 5,000 |
| 11 个 logits 全等 | **4,918 / 4,995 = 98.46%** |
| argmax 一致 | **4,989 / 4,995 = 99.88%** |
| 板端准确率 | **59.04%** |
| ORT 参考准确率（同批样本） | 59.12% |
| 平均绝对差 | **0.09 LSB**（q 域） |

与 44k 全量参考 59.21% 相差 0.17 pp，在 5,000 样本抽样误差（±0.7 pp, 95% CI）之内。

逐类（板端，5,000 样本）：8PSK 55.6%、AM-DSB 70.2%、AM-SSB 91.5%、BPSK 59.7%、CPFSK 62.9%、GFSK 63.3%、PAM4 68.6%、QAM16 50.7%、QAM64 55.1%、QPSK 53.1%、WBFM 20.9%。

## 3. 结论

- 自研引擎与 onnxruntime 在真实测试集上逐样本一致率 98.5%，argmax 一致率 99.9%，平均误差 0.09 LSB —— 剩余差异来自 Q31 定点舍入（ORT 用浮点/整型混合路径）与 1 LSB 量化边界，属量化容差内。
- 板端分类准确率与参考一致（Δ 0.08 pp），说明整条链路（L2 归一化 → int8 量化 → 自研内核 → logits）没有引入精度损失。
- 性能：544 MHz 下 **0.951 ms（全零）/ 0.960 ms（真实窗口）**（`[DWT] N=1000`，10 次预热；清理调试探针后的最终值，P0 阶段曾测得 0.966 ms），优于论文 1.183 ms，快 18.8%。

## 4. 复现方法

```powershell
# 参考精度
python stm32_deployment/tools/p0_accuracy.py

# 板端回放对比（固件需为回放版本，COM3）
python stm32_deployment/tools/p0_replay.py --n 5000 --flash --seed 2
```

固件要求：`Src/main.c` 中回放握手（上电 1.5 s 内收 `R` 进入回放，打印 `[REPLAY] ready`）；`Src/dsc_infer.c` 提供 `dsc_infer_run_q()`。
