# Release v1.0.5 — C1 引擎内核优化（0.905 / 0.914 ms）

- Tag：`v1.0.5`（annotated，指向本提交）
- 日期：2026-10-30
- 上一版：`v1.0.4`（`b295f3d`，torch 复算 + 可复现性闭环）

## 本版做了什么

自研 INT8 引擎（`stm32_deployment/custom_engine/dsc_infer.c`）的第三轮优化（C1），
两项改动都保持与原实现 **逐位一致**，只削减指令数：

1. **requant 改 SMULL + Q32 乘数**：`rq()` 由 `int64` 乘 + 舍入改为
   `smull` 内联汇编取 64 位积，返回 `hi + (lo >> 31)`；`q31()` 相应改为返回 `2 * lrint(m * 2^31)`。
   数学上 `floor((acc*mq31 + 2^30)/2^31) == floor((acc*2mq31 + 2^31)/2^32)`，
   各层乘数最大 ≈0.034，2× 后仍远不溢出 int32。
2. **`conv_dw3` 边界外提 + 内部无分支**：pair 0 与最后一对（左右补 q=−128）移出循环，
   内部循环去掉 `t0==0` / `t1+1>=len` 三元判断；MAC 求和顺序不变。

`conv1x1` **保持基线实现**：时间主序打包 tile4（+6 KB ZI）、双输出通道、t 双采样共享权重
三种重构全部实测变慢，已全部回退。

## 实测（NUCLEO-H723ZG，Cortex-M7 544 MHz，COM3）

| 项目 | C1 前 | C1 后 |
|---|---|---|
| 全零窗口（N=1000，10 次预热） | 0.951 ms | **0.905 ms**（min=max=0.905） |
| 真实 QPSK(+10 dB) 窗口 | 0.960 ms | **0.914 ms**（min=max=0.914） |
| 相对论文 1.183 ms | 快 18.8% | **快 22.7%** |
| 相对 X-CUBE-AI 运行时 1.827 ms | 1.90× | **2.00×** |
| 连续分类循环（串口） | 960.6 µs | **914.5 µs**（Conf 仍 0.645） |
| 流式 20 窗 compute 均值 | 972.6 µs | **926.7 µs**（925.2–929.8） |
| 固件 ROM / RAM | 80,904 B（79.01 KB）/ 48,424 B（47.29 KB） | **80,104 B（78.23 KB）** / 48,424 B（47.29 KB） |
| 构建 | Code 61,788 / RO 16,092 / RW 8,452 / ZI 39,972 | Code 60,844 / RO 16,236 / RW 8,452 / ZI 39,972（0 Error / 481 Warning，全量 rebuild） |

cycles：全零 ≈492,320、真实窗 ≈497,216（544 MHz）→ **≈3.01–3.04 cycles/MACC**（MACC 163,615）。

## 数值回归（未变）

- `golden_test.py`：**22/22 PASS**（±1 LSB + argmax）。
- `p0_replay.py --n 300`：n_ok **300**、逐位全等 **297/300**、argmax **300/300**、板端 acc **62.00%** = 参考、`mean_abs_diff_q` **0.001818**。
- 44,000 窗离线（INT8 ONNX / onnxruntime）：59.2091% / +10 dB 89.36%（与 `ref_accuracy.json` 逐位一致）。

## 证据与产物

- `stm32_deployment/results/evidence/serial_boot.log`：用新固件重抓（全量启动 + benchmark + classify）；
  `build.log`（0 Error / 481 Warning / 4 s）、`flash.log`（`Verify OK`，22:36:44）、`stream.log` 同步更新。
- `docs/evidence/latency_dscamcnet_int8.csv`：两行 → `2026-10-30T22:35:30.553..,1000,0.905,0.905,0.905` 与 `…,0.914,0.914,0.914`。
- `stm32_deployment/results/measured_latest.json`：新增 `c1_kernel_optimization_2026_10_30` 段（改动、前后延迟、资源、验收、证据清单）。
- `docs/REPORT.html` + `docs/REPORT_preview.png` + `stm32_deployment/results/figs/*.png`：全部以 C1 数字重生成。
- 文档口径统一到 0.905/0.914 ms、78.23/47.29 KB、22.7%、2.00×、3.04 cycles/MACC：
  `README.md`、`PROJECT_STATUS.md`、`docs/P0_ACCURACY.md`、`docs/P1_BENCHMARK.md`（加历史说明）、
  `docs/P4_OPTIMIZATION.md`（新增 C1 明细表）、`docs/P5_RESOURCES.md`、`docs/P5_BLOCKED.md`、
  `docs/COMPARISON_vs_OFFICIAL_README.md`、`docs/BUILD.md`、`docs/REPRODUCE.md`、
  `docs/PROJECT_MEMORY.md`（新增 §8.2）、`stm32_deployment/custom_engine/README.md`。

## 工具修复（顺带）

- `live_stream.py` 握手：原 40 次快速写 'S' 会在板子启动完成前写光；改 8 s 内循环发送（每次 50 ms 间隔）并排空串口。
- `gen_report.py`：map 正则原先跨不过 `(Code + RO Data + RW Data)` 括号，读不到真实 ROM/RW；改 `[^\n]*?` 并让表格数字动态生成（不再硬编码）。

## 已知限制

与 v1.0.4 相同：OTA（缺射频硬件）、ULCNN（上游无部署代码）、能耗 0.091 mJ（无电流表）未复现，
已在 README「未复现 / 未验证项」中如实声明。550 MHz（上游口径）与本地 544 MHz 的差异保留说明。
