# P5-3 资源占用与能耗（能测的部分 + 测不了的部分）

## 实测 Flash / RAM（当前双模式固件）

数据来源：`D:\DSC_AMCNet_Deployment\MDK-ARM\DSC_AMCNet_Deployment\DSC_AMCNet_Deployment.map`
（2026-10-30 22:33 构建，C1 内核优化后、已移除固件调试探针）与同目录 Keil 构建报告。

> 2026-10-29 15:52 构建（C1 前）为 Code 61,788 / RO 16,092 / RW 8,452 / ZI 39,972 → ROM 80,904 B（79.01 KB）。C1 只改内核：Code −944 B、RO +144 B → ROM 80,104 B（78.23 KB），RAM 不变。

| 项 | 字节 | 说明 |
|---|---|---|
| Code | 60,844 | 含 HAL、自研引擎、X-CUBE-AI 运行时、串口格式化 |
| RO Data | 16,236 | 含 5,804 B X-CUBE-AI 权重、自研引擎权重表 |
| RW Data（初始值占 Flash） | 8,452 | 含 X-CUBE-AI 层描述符 |
| **Flash 合计（ROM）** | **80,104（78.23 KB）** | Code+RO+RW |
| **RAM 合计（RW+ZI）** | **48,424（47.29 KB）** | RW 8,452 + ZI 39,972，不含 16 MB 外部 SDRAM（未用） |

### 组成（按 map 符号归类，Code / RO）

| 子系统 | Code | RO | 备注 |
|---|---|---|---|
| X-CUBE-AI 运行时 + `network*.o` | 28,902 | 2,482 | `NetworkRuntime1020_CM7_Keil.lib`（标量 int8 内核） |
| 自研引擎 `dsc_infer.o` | 10,384 | 5,560 | 权重表在 RO，工作缓冲在 ZI |
| 应用层（main/inference_main/dwt/retarget） | 5,988 | 972 | 含串口 formatter、benchmark、回放/流式模式 |
| HAL + CMSIS + startup | 11,998 | 772 | 栈 0x2000 + 堆 0x2000 在 startup 的 ZI |
| 其他（libc、math 等） | 4,184 | 6,208 | |
| **合计** | **61,456 / 15,994**（map 总计 61,788 / 16,092，差异为未归类符号） | | 29 KB 的 X-CUBE-AI 运行时只作兼容，可整体删除 |

### 与论文口径对比

| 项 | 论文（Table II） | 本机实测 | 差异原因 |
|---|---|---|---|
| Flash | 51.26 KB | 78.23 KB | 我们同时链了 X-CUBE-AI 与自研引擎两套推理栈，且带 HAL/串口/双模式 |
| RAM | 12.09 KB | 47.29 KB | 栈+堆 16 KB、自研引擎工作缓冲、X-CUBE-AI 激活/IO 约 8.5 KB |

**只保留自研引擎、去掉 X-CUBE-AI 链路的估算**：
Flash ≈ 78.2 − 42.2（AI 运行时 28,902 + network*.o 与描述符 code/RO/RW/ZI）≈ **37–39 KB**，低于论文 51.26 KB；
RAM 去掉 8.5 KB AI 缓冲、把栈/堆从 16 KB 收到 4.5 KB 后 ≈ **26–28 KB**，仍高于论文，
差额主要是自研引擎的 3 个 24×128 int8 工作缓冲与 packed 权重副本。

> 注：论文的 12.09 KB 很可能不含栈/堆与 HAL 缓冲，且其固件只有单一推理栈。
> 本仓库没有为“最小固件”单独建 target，以上为按 map 组成的估算而非实测。

## 能耗：本机无法测量

- 论文给的是 0.091 mJ/inference（955.1 acc/mJ），需要测平均电流。
- 本机没有电流表/功率分析仪（USB 设备里没有 Joulescope/Yole 等），
  NUCLEO-H723ZG 的 IDD 跳线（JP5）也没有接测量设备，因此**无法给出可信的 mJ 数值**。
- 可复现的测量方法（留给有硬件的环境）：
  1. 拔掉 JP5 跳线，串入 µA 级电流表或 Joulescope；
  2. 固件循环跑同一窗口，示波器同步 DWT 起始/结束 GPIO；
  3. 积分一个推理周期的电流 → 电荷 → ×3.3 V 得 mJ；
  4. 分别记录 550 MHz 全速、WFI 空闲、以及关闭 I/D Cache 的对照。
- 不测量就不写数字：本仓库 README 只保留论文的 0.091 mJ 作为文献值，并标注“未本地复测”。

## 结论

- Flash/RAM 实测可用（78.23 KB / 47.29 KB），组成清楚；
- 若目标是贴近论文的 51/12 KB，需要做“单栈最小固件” target（去掉 X-CUBE-AI 与调试字符串、缩栈堆），
  估算可到 ~41 KB Flash / ~24 KB RAM，但 RAM 仍会高于论文口径；
- 能耗需要外部电流测量硬件，本次不伪造数据。
