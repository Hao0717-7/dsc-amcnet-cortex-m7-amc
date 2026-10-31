# P4 性能优化记录（2026-10-29；2026-10-30 追加 C1 内核优化；2026-10-31 时钟升到 550 MHz）

## 最终结果（当前：550 MHz，2026-10-31）

| 输入 | Mean | Min | Max |
|---|---|---|---|
| 全零窗口 | **0.895 ms** | 0.895 | 0.895 |
| 真实 QPSK(+10 dB) | **0.904 ms** | 0.904 | 0.904 |

按 550 MHz 折算：全零 ≈492,250 cycles、真实窗 ≈497,200 cycles → **≈3.01–3.04 cycles/MACC**（MACC 163,615）。
对比论文 1.183 ms：**0.904 / 1.183 = 0.764，快 23.6%**；相对 X-CUBE-AI 运行时 1.827 ms 为 **2.02×**。
（同一内核在 544 MHz 时为 0.905 / 0.914 ms —— 内核指令数没变，只是主频提高 1.1%。）

## C1 结果（2026-10-30，当时 544 MHz）

| 输入 | Mean | Min | Max |
|---|---|---|---|
| 全零窗口 | **0.905 ms** | 0.905 | 0.905 |
| 真实 QPSK(+10 dB) | **0.914 ms** | 0.914 | 0.914 |

对比 P4 收口基线（0.951 / 0.960 ms）：全零 −4.8%，真实数据 −4.8%。
对比论文 1.183 ms：**0.914 / 1.183 = 0.773，快 22.7%**；相对 X-CUBE-AI 运行时 1.827 ms 为 **2.00×**。（以上均为 544 MHz 下的 C1 数据；550 MHz 后的当前值见文首表。）

### C1 内核优化明细（2026-10-30）

| 改动 | 结果 | 结论 |
|---|---|---|
| `rq()`：int64 乘 + 舍入 → `smull` 内联汇编 + Q32 乘数（`hi + (lo>>31)`） | 0.919/0.928 → **0.905/0.914 ms** | 保留；与旧 Q31 实现 bit-exact（`floor((acc*mq31+2^30)/2^31) == floor((acc*2mq31+2^31)/2^32)`） |
| `conv_dw3`：pair 0 / 最后一对移出循环（补 q=−128），内部无三元判断 | 0.951/0.960 → **0.919/0.928 ms**（−3.4%） | 保留；MAC 顺序不变，Conf 仍 0.645 |
| `conv1x1` 时间主序打包 tile4（+6 KB ZI） | 0.910/0.919 ms | 回退（比 C1 最终版慢） |
| `conv1x1` 双输出通道共享载入 | 1.062/1.071 ms | 回退（−12%） |
| `conv1x1` t 双采样共享权重 | 0.947/0.956 ms | 回退（比 V3 慢 28 µs） |

验证：`golden_test.py` 22/22 PASS；`p0_replay.py --n 300`：exact_11_logits 297/300、argmax 300/300、acc 62.00%（=参考）；DWT 全零 min=max=0.905、真实窗 min=max=0.914。逐层 cycles 合计约 492k（全零）/497k（真实）→ **≈3.01–3.04 cycles/MACC**（MACC 163,615）。

## 逐层 cycle 剖析（DWT，544 MHz，真实样本；C1 后同表，ms 按 550 MHz 折算）

| 阶段 | cycles | 占比 |
|---|---|---|
| stem conv 2→24, T=128 | 50,648 | 9.8% |
| dw1 (24ch, T=128) | 67,660 | 13.1% |
| **pw1 (24→24, T=128)** | **162,527** | **31.5%** |
| pool1 | 12,525 | 2.4% |
| dw2 (T=64) | 34,064 | 6.6% |
| pw2 (T=64) | 81,284 | 15.8% |
| pool2 | 6,381 | 1.2% |
| dw3 (T=32) | 17,359 | 3.4% |
| pw3 (T=32) | 39,281 | 7.6% |
| pool3 | 1,747 | 0.3% |
| SE（含 expf） | 33,527 | 6.5% |
| shuffle | 3,055 | 0.6% |
| GAP | 3,262 | 0.6% |
| fc1 + fc2 | 4,006 | 0.8% |
| **合计** | **515,323** | 100% |

三个 1×1 卷积（pw1/pw2/pw3）合计 283k cycles，占 55%；三个 depthwise 合计 119k，占 23%。MACC 总量 163,615 → P4 时为 3.15 cycles/MACC；C1 后总 cycles ≈497k（真实窗）→ **3.04 cycles/MACC**（逐层表为 P4 时测量，C1 主要削减 rq 与 dw3）。

## 已做的尝试

| 尝试 | 结果 | 结论 |
|---|---|---|
| dw3 改「成对 SMLAD」：一次算 y[2i]、y[2i+1] | dw1 71.1k→67.7k，dw2 34.3k→34.1k，dw3 18.1k→17.4k；总 −4k cycles（约 0.8%） | 保留 |
| conv1x1 改为「先按时间打包全部输入，再按 oc 连续写」 | 162k→235k（变慢） | 回退：打包缓冲落在 AXI SRAM，D-Cache 流式读取不如栈上（DTCM）小块快 |
| conv1x1 tile 16 + 4 路累加器（连续写 16 字节） | 162k→286k（更慢） | 回退：寄存器压力/索引开销抵消了写连续性收益 |
| conv1x1 双累加器打断 SMLAD 依赖链 | 162k→286k（更慢） | 回退：armclang 在这段循环里已自行调度，手工拆分反而溢出 |

## 剩余瓶颈判断

1. **rq() 的 int64 乘法**：每个输出元素一次 64 位乘 + 舍入，pw1 有 3,072 个输出元素，估算占该层 30–40%。
2. **输出写跨步**：`out[oc*len+t]` 每 t 写 24 条不同 cache line，写分配开销大；但改成连续写又输给 DTCM 打包，需要更细的分块（如 oc 分块 8×t 分块 32）才可能兼得。
3. **SE 的 expf**：33.5k cycles 里大部分是 24 次 `expf` + 浮点乘法，可换 256 项查表或整数近似。

## C2 实验（2026-10-31，全部回退）

C1 定稿（544 MHz 下 0.905 / 0.914 ms）之后又试了三轮微优化，**三项都没有达到 2% 的止损阈值，全部回退**，最终保留 C1 基线。全部改动在主机侧 44,000 窗逐位校验（`stm32_deployment/tools/host_sim/`）中数值不变。

| 实验 | 改动 | 板端（全零 / 真实窗，544 MHz） | 结论 |
|---|---|---|---|
| C2a | `conv1x1()` 时间步×4 展开、4 个累加器、每步一次 32-bit 打包存储 | 0.902 / 0.913 ms（约 +0.3%） | 未达 2%，回退 |
| C2b | 在 C2a 之上把 SE 的 24 次 `expf` 换成 exp2 五次 Cephes 多项式 | 0.902 / 0.910 ms（累计约 +0.5%） | 未达 2%，回退 |
| C2c | 恢复单步、`conv1x1` 双累加器拆 SMLAD 依赖链 | 0.942 / 0.951 ms（**慢 4%**） | 指令数增加，立即回退 |

## P2：时钟 550 MHz（2026-10-31）

目标：与官方 README 的 550 MHz 口径对齐。改动（`custom_engine/app_main.c` + `.ioc`）：

- PLL1：HSI 64 MHz / M=8 = 8 MHz VCO 输入；**N=68 + FRACN=6144/8192**（0.75）→ 8 × 68.75 = **550 MHz**；P=1 / Q=4 / R=2。
- `FLASH_LATENCY_3 → 4`；HCLK 550/2 = **275 MHz**、APB1-4 = **137.5 MHz**（`.ioc` 已同步）。
- **`CPU_FREQ_BOOST` 实测已使能**：固件读 `SYSCFG->UR18 & SYSCFG_UR18_CPU_FREQ_BOOST = 1`（H72x/H73x 未使能时 VOS0 上限只有 520 MHz），因此 550 MHz 在规格内、**不需要烧 option byte**。

结果（同一 C1 内核、同一板、全量 rebuild 0 Error / 481 Warning、Code 60,844 B 不变）：

| 项 | 544 MHz | **550 MHz** |
|---|---|---|
| 全零窗口 N=1000 | 0.905 ms | **0.895 ms**（min=max） |
| 真实 QPSK(+10 dB) | 0.914 ms | **0.904 ms**（min=max） |
| classify 循环（串口） | 914.5 µs | **904.5 µs** |
| 流式 20 窗 compute | 926.7 µs | **916.6 µs**（标准差 2.3 µs） |
| vs 论文 1.183 ms | 22.7% | **23.6%** |
| vs X-CUBE-AI 1.827 ms | 2.00× | **2.02×** |
| ROM / RAM | 80,104 B / 48,424 B | **不变** |

数值回归不变：golden 22/22、replay 300（逐位 297、argmax 300、acc 62.00%）、INT8 ONNX 44k 59.2091% / +10 dB 89.36%。证据：`results/evidence/{serial_boot,build,flash,golden,stream}.log`、`docs/evidence/latency_dscamcnet_int8.csv`、`results/measured_latest.json` 的 `p2_clock_550mhz_2026_10_31`。

注意：HSI 本身 ±1% 精度，频率标称 550 MHz 的实际值是 544.5–555.5 MHz，因此延迟数字带 ±1% 系统不确定度（544 MHz HSI 方案同样如此）。

结论：本结构在 C1 之后已接近局部最优（`pw1` 仍占 31.5%）；要再明显提速需要换思路（把热循环搬进 ITCM、或把权重/激活改成 8-bit 打包 + 单次 4 通道 MAC），而不属于本轮目标。

## 主机侧数值回归（C2 起新增）

`dsc_infer.c` 支持 `-DDSC_HOST_SIM`（只替换两处 inline asm 与两个固件头文件，目标构建代码不变：Keil 仍 `Code=60844`、550 MHz 板端仍 0.895/0.904 ms），因此每次改引擎可以先在 PC 上跑 44,000 窗逐位对比，再决定是否值得上板：

```
python stm32_deployment/tools/host_sim/compare_engine.py      # 结果与板端一致
```

参考结果：bit-exact 98.775%、argmax 一致 99.9795%、engine acc 59.2045% / +10 dB 89.3646%。

## 量化方案对照（A4 / A5，2026-10-31）

在同一个 44,000 窗 locked test split、同一份 FP32 checkpoint 上比较三种量化路线（`quantization/qat_compare.py`、`quantization/ptq_per_channel.py`）：

| 方案 | overall | +10 dB | ≥10 dB | INT8 模型大小 | 结论 |
|---|---|---|---|---|---|
| FP32 checkpoint | **59.5614%** | **89.7251%** | **88.9302%** | — | 基线 |
| PTQ per-tensor（当前固件） | **59.2091%** | 89.3646% | 88.4410% | 27,989 B | 采用（−0.35 pp） |
| PTQ per-channel | 59.4000% | 89.4547% | 88.8124% | 31,802 B（+3,813 B） | **不采用**：+0.19 pp 未达 0.2 pp 门槛，额外 scale 纯属元数据开销 |
| QAT fake-quant（3 epoch / 60k 窗微调） | 58.8205% | 89.0041% | 88.1692% | — | **不采用**：低于 PTQ |
| QAT INT8（torch `convert()`） | n/a | n/a | n/a | — | 不可行（见下） |

- **QAT 失败原因（量化后端级证据）**：`convert()` 在 SE 的逐通道缩放 `x = x * s.unsqueeze(2)` 处报
  `NotImplementedError: Could not run 'aten::mul.out' with arguments from the 'QuantizedCPU' backend`；
  同时 QAT→QDQ ONNX 导出失败（`aten::fused_moving_avg_obs_fake_quant` 在 opset 13 无对应算子）。
  即 DSC-AMCNet 的 SE 注意力在 torch 静态量化路径上没有可部署出口 —— 这复现了论文附录 C「DSC-QAT 反而变差」的结论方向（本机 58.82% vs 论文 48.2%，差异来自微调子集/epoch 不同）。
- **per-channel**：2,271/44,000 窗 argmax 与 per-tensor 不同（5.16%），其中 per-channel 对 728 窗、per-tensor 对 644 窗；总体增益不显著，且权重只有约 4.5 KB，额外 scale 会让 INT8 模型 +3.8 KB —— 结论不变。
- **数据加载口径修正**（A4 过程中发现并修复）：`datasets/prepare_dataset.py:load_raw()` 此前按 pickle 字典插入顺序展平，而 `split_indices.npz` 按 canonical `sorted(key=(mod, snr))` 生成；修正后在 canonical 顺序下 `snr==10: n=2,219`、`>=10: n=11,039`（与文档一致）。训练脚本与 `tools/` 下的评估脚本本来就用 canonical 顺序，**已发布的 59.2091% / 89.36% 等数字不受影响**。
- 证据：`docs/evidence/qat_eval_log.txt`、`docs/evidence/ptq_per_channel_log.txt`、`results/qat_compare.json`、`results/ptq_per_channel.json`。

## A7：ITCM 热循环（2026-10-31，已回退）

把自研引擎搬进 64 KB ITCM（0x00000000）以减少取指延迟。用 3 次构建 + 板端 benchmark 做对照（`stm32_deployment/custom_engine/DSC_AMCNet_Deployment_itcm.sct` 保留为实验记录）：

| 构建 | 布局 | Program Size | 板端（全零 / 真实窗） |
|---|---|---|---|
| 基线（Target 对话框布局） | 全部 Flash + AXI SRAM | Code 60,844 / RO 16,236 | **0.895 / 0.904 ms** |
| 对照：scatter 文件、无 ITCM | 同基线布局 | Code 60,844 / RO 16,236（相同） | **0.895 / 0.904 ms** |
| A7-a：整个 `dsc_infer.o` 进 ITCM | 代码+常量 15,168 B 在 ITCM | Code 60,924 / RO 16,252 | 0.906 / 0.914 ms（**慢 1.2%**） |
| A7-b：仅 `dsc_infer.o` 代码进 ITCM | 代码 9,464 B 在 ITCM，常量留 Flash | Code 60,924 / RO 16,252 | 0.906 / 0.914 ms（**慢 1.2%**） |

- 对照构建证明 slowdown 不是 scatter 文件本身，而是 ITCM 放置；两种 ITCM 变体数值逐位一致（golden 22/22、replay 297/300 逐位），但都慢 11 µs。
- 结论：这个引擎是**数据流型**（权重/激活端口压力），不是取指瓶颈；Flash 的 ART 预取已经把取指延迟藏住，而 ITCM 单端口同时服务取指与自己内容的读访问，反而增加冲突。→ 回退，板端恢复 0.895/0.904 ms（golden 22/22、replay 300 已复验）。

## 复现

固件启动自动打印两轮 benchmark（全零 + 真实窗口）。逐层 `[PROF]` 探针仅用于调优，收口时已从固件移除，表 4 的数据为当时的测量结果。串口 115200（2026-10-31 起为 550 MHz）：

```
[DWT] N=1000  mean=0.895 ms  min=0.895 ms  max=0.895 ms
[RESULT] ... Mean : 0.895 ms
[RESULT-SAMPLE] ... Mean : 0.904 ms
```
