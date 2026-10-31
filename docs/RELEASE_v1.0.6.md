# Release v1.0.6 — 时钟 550 MHz（与官方口径对齐）

- Tag：`v1.0.6`（annotated，指向本提交）
- 日期：2026-10-31
- 上一版：`v1.0.5`（`3e60bde`，C1 内核优化 0.905 / 0.914 ms @544 MHz）

## 本版做了什么

1. **SYSCLK 544 → 550 MHz**（官方 README 的时钟口径）：PLL1 源 HSI 64 MHz / M=8 = 8 MHz VCO 输入，
   **N=68 + FRACN=6144/8192**（0.75）→ 8 × 68.75 = **550 MHz**；P=1 / Q=4 / R=2；`FLASH_LATENCY_3 → 4`；
   HCLK 275 MHz、APB1-4 137.5 MHz。`.ioc`、`check_clock_config.py`、`gen_report.py` 同步改到 550 MHz 口径。
2. **`CPU_FREQ_BOOST` 实测已使能**：新固件读回 `SYSCFG->UR18` 的 `CPU_FREQ_BOOST` 位 = **1**（H72x/H73x 未使能时
   VOS0 上限为 520 MHz）。因此 550 MHz 在芯片规格内，**不需要烧写 option byte**（无需不可逆的 OTP 操作）。
3. 界面/文档：固件 banner 改为 `Core: Cortex-M7 @ 550 MHz`；README / PROJECT_STATUS / P4 / P0 / P5 系列 /
   COMPARISON / BUILD / REPRODUCE / PROJECT_MEMORY / custom_engine README / stm32_deployment README 全部同步。

## 实测（NUCLEO-H723ZG，Cortex-M7 550 MHz，COM3）

| 项目 | 544 MHz（v1.0.5） | **550 MHz（v1.0.6）** |
|---|---|---|
| 全零窗口（DWT N=1000，10 次预热） | 0.905 ms | **0.895 ms**（min=max） |
| 真实 QPSK(+10 dB) 窗口 | 0.914 ms | **0.904 ms**（min=max） |
| 相对论文 1.183 ms | 快 22.7% | **快 23.6%** |
| 相对 X-CUBE-AI 运行时 1.827 ms | 2.00× | **2.02×** |
| 连续分类循环（串口） | 914.5 µs | **904.4 µs** |
| 流式 20 窗 compute 均值 | 926.7 µs | **916.6 µs**（915.1–920.3） |
| 固件 ROM / RAM | 80,104 B（78.23 KB）/ 48,424 B | **不变** |
| 构建 | Code 60,844 / RO 16,236 / RW 8,452 / ZI 39,972 | **不变**（全量 rebuild 0 Error / 481 Warning，5 s） |

cycles：全零 ≈492,250、真实窗 ≈497,200（550 MHz）→ **≈3.01–3.04 cycles/MACC**（MACC 163,615），与 544 MHz 时相同
（内核指令数没变，只是主频 +1.1%）。

## 数值回归（未变）

- `golden_test.py`：**22/22 PASS**（±1 LSB + argmax）。
- `p0_replay.py --n 300`：n_ok **300**、逐位全等 **297/300**、argmax **300/300**、板端 acc **62.00%** = 参考、`mean_abs_diff_q` **0.001818**。
- 44,000 窗离线（INT8 ONNX / onnxruntime）：**59.2091% / +10 dB 89.36%**，与 `ref_accuracy.json` 逐位一致。
- 主机侧引擎替身（`stm32_deployment/tools/host_sim/`，44k 窗）：bit-exact 98.775%、argmax 99.9795%，与板端一致。

## 证据与产物

- `stm32_deployment/results/evidence/serial_boot.log`：550 MHz banner + `[DWT] mean=0.895` / `[RESULT-SAMPLE] Mean : 0.904`。
- `stm32_deployment/results/evidence/build.log`（全量 rebuild，0 Error / 481 Warning）、`flash.log`（`Verify OK`，08:16:34）。
- `stm32_deployment/results/evidence/golden.log`（22/22）、`stream.log`（20/20，916.6 µs）。
- `docs/evidence/latency_dscamcnet_int8.csv`：`1000,0.895,0.895,0.895` 与 `1000,0.904,0.904,0.904`。
- `stm32_deployment/results/measured_latest.json`：新增 `p2_clock_550mhz_2026_10_31`（改动、前后延迟、boost 读回、证据清单）。
- `docs/REPORT.html` + `docs/REPORT_preview.png` + `stm32_deployment/results/figs/*.png`：全部以 550 MHz 数字重生成。

## 口径与限制

- **HSI 精度 ±1%**：标称 550 MHz 的实际频点带 ±1% 系统不确定度（544 MHz HSI 方案同样如此，不是本版引入的）。
  因此 0.895/0.904 ms 的真实值区间约为 ±1%；与论文 1.183 ms 比较时两套时钟源不完全相同，结论（快 20%+）不受影响。
- `docs/P1_BENCHMARK.md`（P1 阶段）与 v1.0.0–v1.0.5 的 release notes 保留当时的 544 MHz 读数，并注明历史口径。
- 未复现项不变：OTA（缺射频硬件）、ULCNN（上游无部署代码）、能耗 0.091 mJ（无电流表；NUCLEO 的 `JP4` 可串表测量）。
