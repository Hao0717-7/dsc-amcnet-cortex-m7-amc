# Release v1.0.8 — C 组实验（种子稳健性 / 引擎 / 低功耗）+ CI 修复

- Tag：`v1.0.8`（annotated，指向本提交）
- 日期：2026-10-31
- 上一版：`v1.0.7`（`f18b5a3`，工具链 A1–A3 + 实验 A4–A7）

## 本版做了什么

### C8 随机种子稳健性（新增证据）
同训练流程换 `--seed`（7 / 2024）重训，在 44k locked test 上用 `models/eval_checkpoint.py` 评估：

| seed | overall | +10 dB | ≥10 dB |
|---|---|---|---|
| 42（发布权重） | 59.5614% | 89.7251% | 88.9302% |
| 7 | 56.8295% | 85.7143% | 84.8809% |
| 2024 | 58.6432% | 88.9139% | 87.9609% |
| mean ± std (n=3) | 58.34 ± 1.39 pp | 88.12 ± 2.12 pp | 87.26 ± 2.11 pp |

- 只换种子，+10 dB 摆动 4.0 pp（上游声称 ±0.23 pp，本机大一个量级）→ 发布数字必须标注种子。
- 发布用的 seed 42 是三者最好，单点数字比均值高 1.2–1.7 pp（偏乐观）。固件仍用 seed 42，
  板端指标不变。
- 证据：`docs/evidence/seed_robustness_log.txt`、`docs/P4_OPTIMIZATION.md` §C8。

### C9 conv1x1 两输出通道共享输入字（失败，已回退）
Code 59,948（−896 B）、主机侧逐位一致；板端 **0.952 / 0.961 ms（慢 6.4%）** → 触发 5% 止损，
回退并复测基线 0.895 / 0.904 ms。

### C10 per-channel 上板（跳过）
A5 的 CPU 侧收益只有 +0.19 pp（< 0.2 pp 门槛）且模型 +3.8 KB；上板要重贴 X-CUBE-AI 表 +
`dsc_model_data.h` + 全回归，收益低于噪声 → 不做板端验证（记为备选方案）。

### C11 空闲 WFI（功能等价，未启用）
classify 循环忙等改 `__DSB(); __WFI()`（10 ms tick）：Code 60,892（+48 B），bench 仍
0.895/0.904 ms，classify 178 帧 / 2.5 s（14.0 ms 周期）、`904.49 us` 恒定、预测不变、
golden 22/22 PASS。功能等价但**无电流表无法量化省电**，且当前固件无长时空闲 → 回退，不默认启用。

## 修复与文档

- `PROJECT_STATUS.md:29` ITCM 行改成「比当时基准慢 1.2%（历史实验值）」，消除 CI 文档巡检的
  陈旧值误报（上次 v1.0.7 的两个 CI run 因此失败）。
- `models/eval_checkpoint.py`：修复传相对路径 checkpoint（`.tmp/seeds/...`）时
  `ckpt.relative_to(ROOT)` 抛 `ValueError` 的问题（`resolve()` + `disp()` 回退绝对路径）。
- `stm32_deployment/results/measured_latest.json` 新增 `c8_c9_c11_experiments_2026_10_31` 段。
- `docs/P4_OPTIMIZATION.md` 新增「C 组实验（2026-10-31）」：C8 表、C9/C10/C11 结论。
- `PROJECT_STATUS.md` §1 新增 C 组实验行。

## 验证

- 板端恢复基线并复测：`0.895 / 0.904 ms`、`golden 22/22`、`replay 300（exact 297 / argmax 300）`。
- `check_clock_config.py`：全 OK；`check_docs_consistency.py`：PASS。
- CI（离线四项）应在本次推送后由红转绿。

## 已知限制（与 v1.0.7 相同）

OTA（缺射频硬件）、ULCNN 部署（上游无工程）、能耗 0.091 mJ（无电流表）均未复现，已在
README「未复现 / 未验证项」如实声明；时钟已与上游对齐到 550 MHz（v1.0.6 起）。
