# Release v1.0.9 — C8 更正：用训练完成后的最终 checkpoint 重评

- Tag：`v1.0.9`（annotated，指向本提交）
- 日期：2026-10-31
- 上一版：`v1.0.8`（`921a7ed`，C 组实验 + CI 修复）
- 性质：**数据更正**（不是新实验）

## 为什么要更正

v1.0.8 里 seed 2024 的 checkpoint 是在训练还没跑完时被评估的（中途 best 快照），
另外 seed 7 的 checkpoint 当时也还有后续训练。后台训练完成后用最终权重重评，
两个种子的数字都变了：

| seed | v1.0.8 记录（中途快照） | **v1.0.9 更正（最终 checkpoint）** |
|---|---|---|
| 7 | 56.8295% / 85.7143% / 84.8809% | **57.6068% / 86.6156% / 85.9951%** |
| 2024 | 58.6432% / 88.9139% / 87.9609% | **59.8182% / 89.8152% / 89.4556%** |
| 42（发布权重，不变） | 59.5614% / 89.7251% / 88.9302% | **59.5614% / 89.7251% / 88.9302%** |
| mean ± std (n=3) | 58.34 ± 1.39 / 88.12 ± 2.12 | **59.00 ± 1.21 / 88.72 ± 1.82**（overall / +10 dB；≥10 dB 88.13 ± 1.86） |

（表格三列依次为 overall / +10 dB n=2,219 / ≥10 dB n=11,039。）

## 更正后的结论

- 只换种子，+10 dB 在 **86.62%–89.82%** 之间摆动（3.2 pp）、overall 摆动 2.2 pp，
  std 约 **1.2–1.9 pp** —— 仍比上游声称的 ±0.23 pp 大一个量级。
- 发布权重 seed 42 **不是最好**：seed 2024 略高（+0.26 pp overall、+0.09 pp @+10 dB、
  +0.53 pp @≥10 dB）；seed 42 相对三种子均值 +0.57 / +1.01 / +0.80 pp。单点发布略偏乐观，
  但在种子噪声范围内（test 集固定、三种子共用同一 split，无泄漏）。
- 板端不受影响：固件仍烧 seed 42 的 INT8 网络，0.895/0.904 ms、59.2045%、golden 22/22 不变。

## 文件改动

- `docs/evidence/seed_robustness_log.txt`：重写为最终版（含方法学说明：中途快照 → 最终重评）。
- `docs/P4_OPTIMIZATION.md` §C8：表格与结论更新。
- `stm32_deployment/results/measured_latest.json`：`c8_seed_robustness.fp32_44k` 更新为最终值。
- `PROJECT_STATUS.md` §1 C 组行：数字更新（86.62–89.82%，std 1.82 pp）。
- `docs/RELEASE_v1.0.8.md`：顶部追加「C8 数字已被 v1.0.9 更正」提示。

## 验证

- 两个最终 checkpoint：`.tmp/seeds/dscamcnet_seed7.pt`（11:23，36,493 B）、
  `.tmp/seeds/dscamcnet_seed2024.pt`（11:36，36,661 B），评估输出
  `.tmp/seeds/eval_seed{7,2024}.txt`。
- train.py 自报与 eval_checkpoint 一致：seed 7 = 86.62%、seed 2024 = 89.82%（+10 dB）。
- `check_docs_consistency.py`：PASS。
