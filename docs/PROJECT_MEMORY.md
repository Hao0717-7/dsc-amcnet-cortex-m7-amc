# 项目记忆快照 · DSC-AMCNet on STM32H723

> 本文件是 **2026-10-29 18:16** 的记忆恢复记录：把「之前在仓库里做过的进度」按**目前最新、已核实**的证据重新固化下来，
> 供后续会话直接接手，不必再次全文通读。
>
> 事实来源：仓库 `D:\dsc-amcnet-cortex-m7-amc`（main = `f4e10cd`）+ 仓库外部署工程 `D:\DSC_AMCNet_Deployment` + `stm32_deployment/results/evidence/*` 串口/构建日志。
> 与 `PROJECT_STATUS.md`、README、docs/*.md 冲突之处，**以本文件为准**（冲突原因见 §9）。
> 本次只新增本文件，未修改任何既有文件；所有测量数字均来自真实构建与真实串口日志，没有推算或伪造。

---

## 1. 一句话与目标

复现论文 DSC-AMCNet：把 **4,961 参数**的调制识别网络用 **INT8** 跑在 **STM32H723ZGT6（Cortex-M7 @ 544 MHz）**上。
论文目标延迟 **1.183 ms**、`+10 dB` 准确率 **87.01%**。
当前状态：**数据 → 训练 → 量化 → 模型转 C → 编译烧录 → 板上实测，全链路打通**；
自研 SMLAD INT8 引擎板上实测 **0.914 ms（真实窗）/ 0.905 ms（全零）**（2026-10-30 C1 内核优化后；优化前 0.951/0.960），比论文的 1.183 ms 快 **22.7%**；
板端 44,000 窗回放准确率 **59.2045%**（参考实现 59.2091%），logits 逐位一致率 **98.80%**。

## 2. 硬件与时钟（真值）

| 项 | 值 |
|---|---|
| 芯片 | STM32H723ZGT6，LQFP144，Cortex-M7 + 双精度 FPU |
| 板子 | 自制核心板 LXB723ZG-P1（板载 25 MHz 晶振，但当前固件**没用 HSE**） |
| 当前固件时钟 | HSI 64 MHz → PLL1 `M=8 N=68 P=1` → SYSCLK **544 MHz**，VOS0；HPRE=/2 → HCLK 272 MHz（`Src/main.c:340-371`） |
| 串口 | USART1，PA9/PA10，115200 8N1（流式模式握手后切 921600） |
| 调试 | SWD PA13/PA14；DWT 计时需 DBGMCU `TRACECKEN` + `LAR` 解锁 |

⚠️ `.ioc` 里的时钟配置是错的（`DIVM1=4 DIVN1=12`、`SYSCLKFreq_VALUE=192000000`），
**用 CubeMX 重新 Generate 会把 544 MHz 打回 192 MHz 并覆盖 main.c**。不要 Generate（见 §9-4）。

## 3. 数据与划分

- 数据集：`datasets/RML2016.10a_dict.dat`，DeepSig 官方 I/Q pickle（latin1），字典 `key=(调制, SNR) → (1000, 2, 128) float32`，共 11 类 × 20 SNR = 220,000 窗。
- 类别序：`8PSK, AM-DSB, AM-SSB, BPSK, CPFSK, GFSK, PAM4, QAM16, QAM64, QPSK, WBFM`。
- 铺平顺序：`sorted(key=(k[0], k[1]))`。
- 划分：`datasets/split_indices.npz`（train 176,000 / test 44,000），**不可重新生成**：md5 `bb8bc1e7c41a58fad6f08a23ae7ef76d`，sha256 `69c8ed46…8d61b`；并集恰为 0..219999。
  - 注意：它**不是 stratified**（每 (mod,snr) cell 的 test 数 166–239，每类 3,923–4,122），但这是论文口径，必须沿用。
- 归一化：**每个窗口**用 256 点整体 L2 归一化（不是逐通道），训练/推理必须一致。
- 坑：桌面版 CSV 只有 I 通道（Q=0），用它训练只有 74.99%；必须用官方 I/Q pickle。实测当前 `.dat` 的 Q 通道 std 0.0032–0.0080，确为真 I/Q。
- `datasets/build_rml2016_from_csv.py` 默认 `--q-mode zero` 且**直接覆盖** `RML2016.10a_dict.dat` —— 不要跑默认参数。

## 4. 模型与量化

- `models/dsc_amcnet.py`：`DSCAMCNet(width=24, num_classes=11, shuffle_groups=4)`，输入 `(B,2,128)`，输出 `(B,11)` 裸 logits，**4,961 参数 / 50 个权重 key**。
- 权重文件：`models/checkpoints/dscamcnet_best.pt`（36,437 B，sha256 `f7a05c25672a0074cff30673c87acb097897fa96662264cb57f9a959a91baab6`）。
  - Zenodo 官方权重是坏的（+10 dB 仅 26.05%），当前权重是本地重训的；`models/checkpoints/README.md` 同时列了两套 SHA，以本地那行为准。
- 训练：FP32 全 SNR 59.56%，`+10 dB` 89.73%。
- PTQ：`quantization/ptq_quantize.py` → `quantization/outputs/dscamcnet_int8.onnx`（27,989 B，93 节点 / 68 个 QDQ），后端 signed int8 / IR7 / opset13。
- 量化参数（固件 `dsc_model_data.h`）：输入 scale `0.00213504443` zp `-4`；激活 zp `-128`；SE fc2 zp `-5`；logits scale `0.350085318` zp `+63`；权重 zp `0`。
- 板上精度：PTQ INT8 `+10 dB` 89.36%（FP32 89.73%，−0.37 pp）。
- 坑：必须 signed int8（qnnpack unsigned zp=128 不兼容）；转换固定 float32 I/O + `--no-inputs-allocation --no-outputs-allocation`；必须 `--optimization time`（报告文件里写成 balanced，与文档矛盾，以实际构建为准）。

## 5. 两条推理路径

| 路径 | 延迟 | 说明 |
|---|---|---|
| X-CUBE-AI 10.2.1 运行时 | 1.827–1.843 ms | 基线，只为证明能跑通；数值与 onnxruntime 不完全一致，**不作为精度主张** |
| 自研 SMLAD INT8 引擎（`custom_engine/dsc_infer.c`） | **0.905 / 0.914 ms** | 最终结论（C1 后）；通道两两打包 + 内联 SMLAD，Q32 乘数 + SMULL requant（bit-exact 等价 Q31），dw3 无分支，每层 sumw 吸收输入零点 |

引擎口径：163,615 MACC；P4 时 515,323 cycles @544 MHz ≈ 0.947 ms（实测 0.951），C1 后 ≈497k cycles ≈ 0.914 ms（实测 0.914）。
层序：`2→24→dw3→24→pool→dw3→24→pool→dw3→24→pool(16)→SE→shuffle4→GAP→FC24→64→FC64→11`。
工作区 `A[24*128] B[24*128] C[24*16+64]`，单栈最小约 41 KB Flash / 24 KB RAM。

## 6. 权威实测结果（最新，2026-10-29 15:52 构建 / 15:54 日志）

| 指标 | 值 | 证据 |
|---|---|---|
| 延迟（全零，N=1000） | **0.905 ms**（min=max=0.905） | `serial_boot.log` `[DWT] N=1000 mean=0.905`（C1 后） |
| 延迟（真实窗，N=1000） | **0.914 ms**（min=max=0.914） | `serial_boot.log` `[RESULT-SAMPLE]`（C1 后） |
| 44k 板端准确率 | **59.2045%**（26,050/44,000） | `results/p5_accuracy_44k.json` |
| 44k 参考准确率 | 59.2091%（26,052/44,000） | 同上 |
| logits 逐位一致 | **98.80%**；argmax 一致 99.98% | 同上 |
| `+10 dB` 准确率 | 89.36% | `docs/P5_ACCURACY.md`、`results/ref_accuracy.json` |
| golden 向量 | **22/22**（±1 LSB 且 argmax 等） | `evidence/golden.log` 末尾 `22/22 vectors within +/-1 LSB and argmax-equal` |
| 流式（UART int16） | 20/20 窗，board_acc 0.650，**972.6 µs/窗**，往返 7.5 ms | `evidence/stream.log` |
| 单栈资源（权威构建，C1 后） | Code 60,844 / RO 16,236 / RW 8,452 / ZI 39,972 → **ROM 80,104 B = 78.23 KB，RAM 48,424 B = 47.29 KB**；0 Error / 481 Warning（全量 rebuild） | 22:33 axf 的 map `Program Size`；C1 前 61,788/16,092 → 80,904 B |

补充：`p5_accuracy_full.json` 是旧的 300 窗小样本（board 55.33% vs ref 55.67%），不要引用其准确率；
`results/p0_replay_compare.json` 是 5,000 子集回放（n_ok 4,995，逐位 98.46%，argmax 99.88%，acc 59.04 vs 59.12），已被 44k 全量取代。
`golden_vectors.json` 22 条中有 2 条 argmax 不等于类索引（QAM64_10dB→QAM16、WBFM_0dB→AM-DSB），所以 22/22 证明的是「板端复现期望 logits」，**不是分类正确**。

## 7. 工程与产物位置

- 仓库（git，`main` = `f4e10cd`）：数据/训练/量化/文档/固件源码快照/工具/结果。
- **实际编译烧录的工程在仓库外**：`D:\DSC_AMCNet_Deployment`（Keil MDK-ARM，AC6 6.7）。
  - 当前产物：`MDK-ARM\DSC_AMCNet_Deployment.hex`（15:52，227,633 B）；`MDK-ARM\...axf` 同步。
  - ⚠️ 部署工程根目录的 `firmware_DSC_AMCNet_INT8.hex`（07:27，241,673 B）是**旧构建**，不要用它。
- 仓库内 `stm32_deployment/custom_engine/` 是固件源码（app_main.c、app_inference_main.c、dsc_infer.c/h、dsc_model_data.h、retarget_keil_h723.c、test_sample.h、.ioc）。
- 证据：`stm32_deployment/results/evidence/`（build/flash/golden/serial_boot/stream 5 份日志）。
- 一页式报告：`docs/REPORT.html`（+ `docs/REPORT_preview.png`、`docs/figs/*.png`、生成脚本 `stm32_deployment/tools/gen_report.py`）。

## 8. 证据链现状

**可信**：44k 59.2045%/98.80%、golden 22/22 与 15:27 烧录、15:52 构建、15:54 采集的那版固件一致；**0.951/0.960 ms、79.01/47.29 KB** 是该版读数，**C1 后**（2026-10-30 22:33 构建、22:36 烧录）为 **0.905/0.914 ms、78.23/47.29 KB、stream 926.7 µs**。

**已过期 / 不可引用（2026-10-29 18:5x 已按实测重生成/改写，见 §8.1）**：
- ~~`docs/REPORT.html`、`docs/figs/*.png`（15:28）是按旧日志（0.959/0.969）生成的~~ → 已用修正后的 `gen_report.py` 重生成（0.951/0.960、79.01/47.29 KB）。
- ~~README / RELEASE_v1.0 / P5_RESOURCES / P1 里的 85.36 KB、47.42 KB、0.968、18.2% 都是清理前口径~~ → 已统一为 79.01/47.29 KB、0.951/0.960、18.8%。
- `evidence/build.log` 只有 4 行、`0 Warning`、`Build Time Elapsed: 00:00:00` —— 是增量空构建，不能证明全量编译（全量实际 23 warnings）。
- `evidence/flash.log` 时间停在 15:27（烧的那一版），之后的 15:52 构建没有对应烧录记录。
- `serial_boot.log` 前 259 行的分类循环帧恒为 `Pred: QAM16 | Conf 0.645`（输入 DMA 拷贝被注释，见 §9-3），不能当分类证据。

## 8.1 2026-10-29 18:4x–18:5x 实测复核（本轮，改动已落盘）

离线复算（`stm32_deployment/tools/measure_results.py`，只读输入）：
- FP32 ONNX 44k **59.56%** / +10 dB **89.73%**；INT8 ONNX 44k **59.2091%** / **89.36%** —— 与 `results/ref_accuracy.json` **完全一致（diff=0）**。
- `p5_accuracy_44k.json` 自洽：26,050/44,000 = **59.2045%**。
- split：176,000/44,000，md5 `bb8bc1e7…`，**非 stratified**（每类 test 3,923–4,122，每 cell 166–239）。
- 数据集 sha256 `889ac64827e44224d2229acc870ca44e3b487e594d588de75372184c05a80b4a`（225,292,173 B），Q 通道 std 0.0032–0.0080（真 I/Q）。
- 15:52 构建 map：Code 61,788 / RO 16,092 / RW 8,452 / ZI 39,972 → **ROM 80,904 B（79.01 KB）/ RAM 48,424 B（47.29 KB）**。

板端实测（COM3，同一版 15:52 固件）：
- classify 循环连打 **257 帧**，`Latency` 恒 **960.58 µs**（min=max）。
- `golden_test.py`：**22/22 PASS**（21 条 logits 全等，PAM4_0dB 差 1 LSB），退出码 0。
- `p0_replay.py --n 300`：n_ok **300/300**，11 logits 全等 **297/300**，argmax 一致 **300/300**，板端 62.00% = 参考 62.00%，平均差 0.0018 LSB。
- 注意：进入 replay 模式后板子无法自行退出，需断电复位。

本轮落盘改动：`stm32_deployment/results/measured_latest.json`（全部实测与来源）、`docs/REPORT.html` + `stm32_deployment/results/figs/*.png` 重生成、`gen_report.py` 去掉 87,408/48,560 与 0.959/0.969 硬编码（资源改为读 map）、README/PROJECT_STATUS/RELEASE/REPRODUCE/P1/P4/P5_RESOURCES/P5_BLOCKED 数据统一。

## 8.2 C1 内核优化（2026-10-30 22:xx，已上板验收）

- 改动（`stm32_deployment/custom_engine/dsc_infer.c`，已与部署工程 `Src/dsc_infer.c` 同步）：
  1. `rq()`：int64 乘 + 舍入 → `smull` 内联汇编 + Q32 乘数，返回 `hi + (lo>>31)`；与旧 `floor((acc*mq31+2^30)/2^31)` 数学等价（bit-exact）。`q31()` 改为 `2*lrint(m*2^31)`（各层乘数 ≤0.034，不溢出）。
  2. `conv_dw3`：pair 0 与最后一对（边界补 q=−128）移出循环，内部循环无分支。
  3. `conv1x1` 保持原实现（三种重构均实测变慢并回退）；工作区 `A/B/C` 仍为 24×128 int8，无新增 ZI。
- 实测（COM3，544 MHz，`-O3`，Keil AC6 6.7 全量 rebuild 0 Error / 481 Warning）：
  - `[DWT] N=1000 mean=0.905 ms min=0.905 max=0.905`（全零）→ `[RESULT] Mean : 0.905 ms (paper: 1.183)`
  - `[DWT] N=1000 mean=0.914 ms min=0.914 max=0.914`（真实 QPSK +10 dB）→ `[RESULT-SAMPLE] Mean : 0.914 ms`
  - classify 循环 `Pred: QAM16 | Conf: 0.645 | Latency 914.5 us`（Conf 回到 0.645 = 数值逐位一致）
  - 相对 C1 前 0.951/0.960：**−4.8%**；相对论文 1.183：快 **22.7%**；相对 X-CUBE-AI 1.827 ms：**2.00×**
  - 资源：ROM 80,104 B（78.23 KB，−800 B）、RAM 48,424 B（47.29 KB，不变）；Code 60,844 / RO 16,236 / RW 8,452 / ZI 39,972
- 验收：`golden_test.py` 22/22 PASS；`p0_replay.py --n 300`：n_ok 300、exact_11_logits 297、argmax 300、acc 62.00%=参考、mean_abs_diff_q 0.001818；`live_stream.py --n 20`：20/20、board_acc 0.650、compute mean 926.7 µs（925.2–929.8）、rtt 7.5 ms。
- 证据：`stm32_deployment/results/evidence/serial_boot.log`（新固件全量 2,047 行）、`build.log`、`flash.log`（22:36:44）、`stream.log`；`docs/evidence/latency_dscamcnet_int8.csv` 已更新为 0.905/0.914。
- 工具修复：`live_stream.py` 握手改为 8 s 内循环发 'S'（原来 40 次快速发送会在板子启动前打完）；`gen_report.py` map 正则改为可跨 `(Code + RO Data + RW Data)`，表格数字改动态。
- 产物：`docs/REPORT.html`（113,548 B，C1 数字）、`docs/REPORT_preview.png`（332,540 B）、`stm32_deployment/results/figs/*.png` 全部重生成。

## 9. 可复现性缺口（2026-10-29 代码审计，只读）

1. **仓库里缺板上真正用的 `dwt_measure`** —— **已修复（2026-10-30）**，文件已收入 `stm32_deployment/custom_engine/`（见 §9.1）。历史问题（2026-10-29 审计）：
   部署工程 `Inc/dwt_measure.h:66-70` 签名是 `uint32_t *out_mean_us_x100/…`，`Src/dwt_measure.c:113` 用 `SystemCoreClock/1000000U`（544 MHz）；
   仓库 `stm32_deployment/firmware/dwt_measure.h:66-70` 还是旧版 `float *out_mean_us`，`firmware/dwt_measure.c:106` 用 `HAL_RCC_GetHCLKFreq()`（HCLK 272 MHz，会把 515,323 cycles 算成 1.894 ms）。
   仓库 `custom_engine/` 里**没有** dwt_measure.*，但 `app_inference_main.c:222/227` 传的是 `uint32_t*` → 按仓库原样编译类型不匹配，0.951/0.960 无法从当前 commit 重建。
2. **第二基准不是真实窗**：`custom_engine/test_sample.h` 只有 7 行，I 路 128 点有起伏，**Q 路 128 点全为 −12 常数**，P1/P4 却称其为 real QPSK(+10 dB)。
3. **连续分类帧是常数**：`firmware/inference_main.c:327` 的 DMA `memcpy` 被注释，`input_data` 永不更新。
4. **`.ioc` 与固件时钟冲突** —— **已修复（2026-10-30）**：`.ioc` 已改为 DIVM1=8 / DIVN1=68 → 544 MHz（HCLK 272、APB 136），见 §9.1。历史：`.ioc` 原为 192 MHz，代码是 544 MHz（见 §2）。
5. **工具开箱即错**：
   - `tools/golden_test.py:53`、`tools/live_stream.py:64` 用 `pathlib.Path(tempfile.gettempdir())` 但只 `from pathlib import Path` → 带 `--flash` 必 `NameError`；
   - `docs/REPRODUCE.md:30` 传 `--backend fbgemm`，但 `ptq_quantize.py` 没有该参数 → argparse 报错；
   - `tools/p0_replay.py:163` 写 `replay_compare.json`，文档/证据是 `p0_replay_compare.json`；
   - `tools/p5_full_accuracy.py:169` 的 `latency_ms: 0.979` 是硬编码；
   - 硬编码 `D:\APP\Keil5\UV4\UV4.exe`、部署工程路径、默认 `COM3`。
6. 其他已知 bug：`firmware/inference_main.c:258` 调用未定义的 `dsc_dbg_sample()`；`validate_backend.py:95` 的 `module.weight()` 应为 `.weight`；`qat_experiment.py` 解包 `load_splits()` 返回 4 值却写 6；`collect_ota.py` 未做整窗 L2；`hardware_characterization/analyze_flash_ram.py` 按论文 51.26/12.09 KB 期望值。

### 9.1 已修复项（2026-10-30，用户批准「你做吧」）

- `stm32_deployment/custom_engine/` 新增以下文件（从 `D:\DSC_AMCNet_Deployment` 原样收回；只 `dwt_measure.h` 头注释改过）：
  - `dwt_measure.h` 2,326 B，sha256 `5d58caa4aafd19b75fd30c3033e2cd04e3a341ef1a9af1f3d8ea7a3488a81b6c`（注释已改为 `SystemCoreClock`，并注明用 `HAL_RCC_GetHCLKFreq()` 会报 2 倍）
  - `dwt_measure.c` 6,829 B，sha256 `c279bc56640c64eea988e5e98dcea78d6e3d9e3220502436a2a0625e63640e8f`
  - `DSC_AMCNet_Deployment.uvprojx` 21,857 B，sha256 `9e1ed3985f1bc5a1201641f323689010badd03b16da0fe5d9b5ebed84fe10518`（相对路径 `../Src`、`../Drivers`、`../Middlewares/ST/AI`；`pCCUsed 6070000::V6.7::ARMCLANG`）
  - `DSC_AMCNet_Deployment.sct` 324 B，sha256 `01c1e02d2d087dfeb9931b4a3c8b4bb2fe5cad99f0757eab76b2216255fae923`
- `.ioc` 时钟改为 544 MHz：`DIVM1=8`、`DIVN1=68`、`VCOInput1=8 MHz`、`VCO1Output/SYSCLK/Cortex=544 MHz`、`HCLK=272 MHz`、`APB1/2/3/4=136 MHz`；`FDCAN/USBFreq=48 MHz` 属未启用/HSI48，保留不动。
- `stm32_deployment/custom_engine/README.md` 已重写为 0.951/0.960 ms、18.8%、1.90×、X-CUBE-AI 10.2.0，并指向本次收回的文件。
- `docs/BUILD.md` 增 §1.1 记录上述收口。
- 仍然保留：`stm32_deployment/firmware/dwt_measure.*` 旧版（已标注不可与 custom_engine 版混用）。

## 10. 文档口径（2026-10-29 18:5x 已统一到实测值）

- 延迟：**0.905 ms（全零）/ 0.914 ms（真实窗）**（C1 后）；历史值 0.951/0.960（C1 前）、0.958/0.968（清理前）、0.967/0.975（P4 前）只在标注为历史时保留。
- 资源：**80,104 B（78.23 KB）/ 48,424 B（47.29 KB）**（C1 后）；80,904（79.01 KB）为 C1 前，87,408/48,560 为更早固件。
- 参数：4,961（模型/论文口径）vs 4,793（`X-CUBE-AI/network_generate_report.txt`，差 168 = 7×24 偏置口径）；INT8 ONNX initializer 元素数复算为 4,869（第三种口径，勿混用）。
- 版本：**pack = X-CUBE-AI 10.2.1**（安装目录 `~\STM32Cube\Repository\Packs\STMicroelectronics\X-CUBE-AI\10.2.1`，ST Edge AI Core 2.2.0），运行时库文件名 `NetworkRuntime1020_CM7_Keil.lib`，固件 banner 字符串硬编码写着 “X-CUBE-AI 10.2.0” —— **以 pack 版本 10.2.1 为准**；优化等级：仓库内 `stm32_deployment/X-CUBE-AI/` 生成件是 balanced，板端实跑 time 版（激活 7,488 B，生成物未入库）。
- README 已把论文值表标题改为 “论文报告值 / paper-reported”，并注明 0.091 mJ 未本地复测。
- 缺失引用：`models/checkpoints/README.md:92` 引用的根 `CHANGELOG.txt` 已在提交 `9772864`（2026-10-20）删除，可从 `9772864^:CHANGELOG.txt` 恢复。

## 11. 待办与未决

- [x] **P1 一致性（2026-10-30 完成大半）**：新版 `dwt_measure.h/.c`（`uint32_t*_x100` + `SystemCoreClock`）已从部署工程收回 `stm32_deployment/custom_engine/`，`.ioc` 已改到 HSI M=8/N=68 → 544 MHz，`uvprojx`/`.sct` 一并入库（见 `docs/BUILD.md` §1.1）；仍欠 5 个脚本 bug 与 `REPORT_preview.png` 重生成。
- [x] **P2 口径统一**（2026-10-29 18:5x 完成）：以 0.951/0.960 ms、79.01/47.29 KB 为准，已重跑 `gen_report.py` 重建 `REPORT.html` + `figs/`，并同步 README / RELEASE_v1.0 / P5_RESOURCES / P1 / P4 / PROJECT_STATUS / REPRODUCE。
- [x] **P3 数据安全（一半）**：`RML2016.10a_dict.dat` 的 sha256 已记入 `stm32_deployment/results/measured_latest.json` 与 `datasets/README.md`；`build_rml2016_from_csv.py` 的防覆盖保护**未加**（仍默认写 canonical 路径）。
- [ ] **P4 可选**：恢复 `CHANGELOG.txt`；删 X-CUBE-AI 运行时省 Flash（需同步删 network*.c 与链接库）；per-channel 量化。
- [ ] **P5 阻塞（缺硬件/权重）**：OTA 采集（HackRF + RTL-SDR）、能耗实测（电流表）、ULCNN 对照（无权重）。
- [ ] 未 tracking 的 `demo/particle-animation.html`（9,124 B，与项目无关）——保留还是移入隔离区，等用户决定。
- [ ] 远端 `git@github.com:Hao0717-7/dsc-amcnet-cortex-m7-amc.git` 未推送：ssh 被沙箱禁止（`cannot create standard input pipe for ssh: Permission denied`）。

## 12. 复现路径（最短）

1. 打开仓库外 `D:\DSC_AMCNet_Deployment\MDK-ARM\DSC_AMCNet_Deployment.uvprojx`（Keil 5.24a + AC6 6.7），确认 `Device: STM32H723ZGTx`、时钟 544 MHz。
2. 全量 Rebuild（不要只看增量日志），烧录用 `DSC_AMCNet_Deployment.hex`。
3. 串口 115200 打开，按 `R` 进入重放：发 `R` + 256 B int8 量化输入 → 回 `L,<11 个 int8 logits>`；`golden_test.py` 对 22 条 ±1 LSB 校验。
4. 测延迟：无命令上电进入 inference_main，串口打印 `[DWT] N=1000 mean=…`。
5. 流式：发 `S`，切 921600，512 B int16 LE 平面 IQ → 回 `C,<cls>,<conf×1000>,<us×100>,<name>`。

> 记住三件事：①以 `f4e10cd` 之后的 15:52 构建 + 15:54 日志为最新事实；②论文值 ≠ 本仓库实测值，报告里必须分开写；③删/移任何文件前先问用户。
