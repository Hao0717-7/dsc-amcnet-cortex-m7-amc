# 项目状态与工作区地图

> 面向：接手这个仓库的人或 AI。读完这一页就能知道项目在做什么、做到哪、每个文件夹装的是什么、哪些坑已经踩过。
> 最后更新：2026-10-28 ｜ 分支 `main` ｜ HEAD `08ba0c6`

## 0. 一句话

复现一篇论文：把 4,961 参数的调制识别网络 DSC-AMCNet 部署到 STM32H723（Cortex-M7 544 MHz）上做 INT8 推理，目标延迟 1.183 ms、+10 dB 准确率 87.01%。**数据 → 训练 → 量化 → 模型转 C 代码 → 编译烧录 → 板上实测 已全部打通**；当前实测 **1.827–1.843 ms**，是论文的 1.54×，差距分析与下一步列在 §7、§7.1。

## 1. 当前进度

| 环节 | 状态 | 产物 | 实测指标 |
|---|---|---|---|
| 数据集构建 | ✅ 完成 | `datasets/RML2016.10a_dict.dat` | 220,000 条，11 类 × 20 SNR × 1,000 |
| 训练 | ✅ 完成 | `models/checkpoints/dscamcnet_best.pt` | **+10 dB 89.73%**，全 SNR 59.56% |
| PTQ 量化 | ✅ 完成 | `quantization/outputs/dscamcnet_int8.onnx` | **INT8 89.36%**，比 FP32 掉 0.37 pp |
| 后端兼容性校验 | ✅ 通过 | `validate_backend.py` 输出 | signed int8 / IR 7 / opset 13 / QDQ 68 节点 |
| ST Edge AI 转换 | ✅ 完成 | `stm32_deployment/X-CUBE-AI/*.c/.h` | 权重 5,804 B；仓库内为 balanced 版（激活 4,488 B），板端实跑 time 版（激活 7,488 B），MACC 163,615 |
| CubeMX 工程 + Keil 编译 | ✅ 完成 | `D:\DSC_AMCNet_Deployment`（生成物，未纳入本仓库） | Keil AC6 6.7，0 error / 9 warning |
| 烧录上板测延迟 | ✅ 完成 | `firmware_DSC_AMCNet_INT8.hex`（`D:\DSC_AMCNet_Deployment`） | **1.827–1.843 ms**（≈995k cycles @ 544 MHz），论文 1.183 ms |
| OTA 采集评估 | ⬜ 未做 | — | 需 HackRF + RTL-SDR |

## 2. 先看这八条，避免重复踩坑

1. **数据集是「官方 I/Q pickle 重建」的，不是桌面 CSV 直转的。** 桌面那批 CSV 只有 I 通道（128 列实数 + 1 列 SNR），Q 通道缺失；用 Q=0 的数据训练只能到 74.99%。真正的 I/Q 来自 DeepSig 官方 `RML2016.10a_dict.pkl`（现在已固化进 `datasets/RML2016.10a_dict.dat`）。
2. **`datasets/split_indices.npz` 不能重新生成。** 它的 MD5 与 Zenodo 发布件完全一致（`bb8bc1e7c41a58fad6f08a23ae7ef76d`），176,000 / 44,000 的划分是论文精度口径的一部分。
3. **Zenodo 上发布的官方权重是坏的，不要用。** 官方 `dscamcnet_best.pt`（37,397 B）在官方自己的数据上 +10 dB 只有 26.05%、在本地数据上 15.35%，11 类随机基线约 9.1%。当前权重是本地用官方数据按论文协议重训的。
4. **必须 signed int8，权重 zero-point = 0。** PyTorch qnnpack 后端的 unsigned int8（zero-point 128）与 X-CUBE-AI 不兼容，上板会静默出错。用 `validate_backend.py` 检查。
5. **训练和推理都必须 per-window L2 归一化。** 归一化覆盖整个 2×128 窗口（256 个数一起求范数），不是按 I/Q 分通道。固件里漏掉这一步，板上精度会掉到接近随机。
6. **模型转换固定用 float32 I/O + 不分配 I/O 缓冲**（命令见 §6）。ONNX 图本身是 int8 I/O，但仓库固件按 float 收发、自己管理缓冲区，所以生成时必须加 `--input-data-type float32 --output-data-type float32 --no-inputs-allocation --no-outputs-allocation`。
7. **论文数字与实测不一致，以实测为准。** 论文写 87.01%（官方 Zenodo 元数据写 85.80%，复现不出来）；又写 51.26 KB Flash / 12.09 KB RAM / ≈1.2M MACC，而转换报告是 5.67 KiB 权重、4.38 KiB 激活、163,615 MACC（51.26 KB 应该是含 AI 运行时库的估算）。
8. **USART 引脚文档是错的。** STM32H723 的 PA8 没有 USART3 复用；自制板调试排针上的 TXD/RXD 实际是 USART1（PA9/PA10），CubeMX 以引脚分配图为准。

## 3. 工作区目录地图

```
D:\dsc-amcnet-cortex-m7-amc\
├── README.md                 原始项目说明（复现表、6 步流程）
├── PROJECT_STATUS.md         本文件
├── environment.yml           conda 环境定义（python 3.10 + pytorch 等）
├── requirements.txt          pip 依赖清单
├── activate_env.ps1          ⚠ 旧的自建环境脚本，依赖的 .pylibs 已在清理时删除
├── .gitignore                忽略数据/权重/临时目录
│
├── datasets/                 【数据】
│   ├── RML2016.10a_dict.dat        标准数据集：{(调制, SNR): (1000, 2, 128) float32}，220,000 条，215 MB
│   ├── full_2k/
│   │   ├── RML2016.10a_2k_dict.dat 完整版：每组 2,000 条，共 440,000 条，430 MB
│   │   └── selection_manifest.json 标准版抽样记录（seed=42，每组抽中的源行号）
│   ├── split_indices.npz           train_idx 176,000 / test_idx 44,000（官方锁定划分）
│   ├── prepare_dataset.py          数据加载与划分校验脚本
│   ├── build_rml2016_from_csv.py   CSV→pickle 转换脚本（历史路径，当前数据集已改为官方 I/Q）
│   └── README.md                   数据集格式说明与来源记录
│
├── models/                   【模型与训练】
│   ├── dsc_amcnet.py               网络定义：DSC 深度可分离卷积 + SE + shuffle，4,961 参数
│   ├── train.py                    论文协议训练脚本（Adam 1e-3、batch 512、早停 15、ReduceLROnPlateau）
│   └── checkpoints/
│       ├── dscamcnet_best.pt       ★ 当前部署权重：+10 dB 89.73%
│       ├── dscamcnet_best.prev-89.00.pt  同协议独立跑的另一版（89.00%），仅对照
│       └── README.md               权重 provenance、SHA256、官方权重失效说明
│
├── quantization/             【量化】
│   ├── ptq_quantize.py             FP32 ONNX 导出 + 静态 PTQ（QDQ / QInt8 / 1,024 校准样本）
│   ├── validate_backend.py         检查 signed int8、IR 版本、X-CUBE-AI 兼容性与精度差
│   ├── qat_experiment.py           QAT 失败实验（论文附录 C）
│   └── outputs/
│       ├── dscamcnet_fp32.onnx     FP32 图：ir 7 / opset 13 / 34 节点
│       └── dscamcnet_int8.onnx     ★ INT8 图：93 节点 / 68 个 QDQ，部署入口
│
├── stm32_deployment/         【上板】
│   ├── README.md                   9 步上板指南（已按实测更新：转换命令、时钟、USART、缓冲、输出）
│   ├── firmware/
│   │   ├── inference_main.c        ★ 推理主流程（L2 归一化、activations 缓冲、softmax、UART 打印）
│   │   ├── inference_main.h        main() 调用入口声明
│   │   ├── dwt_measure.c           DWT 周期计数与 1,000 次基准测试
│   │   └── dwt_measure.h           DWT 接口声明
│   └── X-CUBE-AI/                  ★ ST Edge AI 10.2.1 生成的模型 C 代码（未提交，建议 git add）
│       ├── network.c               模型代码 + 权重（81.8 KB）
│       ├── network_data.c          权重/激活缓冲映射表
│       ├── network_data_params.c   AI_NETWORK_DATA_* 尺寸宏
│       ├── network.h / network_data.h / network_data_params.h / network_config.h
│       └── network_generate_report.txt  逐层报告（MACC、内存、输入输出量化参数）
│
├── hardware_characterization/ 【测量】
│   ├── measure_latency.py          串口读取板上 DWT 输出并写 CSV
│   └── analyze_flash_ram.py        根据编译产物分析 Flash/RAM 占用
│
├── ota_collection/           【OTA 采集】collect_ota.py（RTL-SDR 实时采集，需硬件）
├── supplementary/            【论文附录原文】架构、分调制精度、QAT 失败分析三篇 md
│
└── _official/                【provenance 取证脚本与证据】
    ├── train_fresh.py          ★ 产出当前 89.73% 权重的重训脚本（已补随机种子）
    ├── eval_onnx.py / eval_ckpt.py / epoch_map.py     评估与逐轮曲线工具
    ├── compare_sources.py / diag_sources.py / inspect_keys.py 等  数据来源比对脚本
    ├── epoch_curve.csv        100 轮逐轮 +10 dB 精度曲线（epoch,test_all_snr_pct,test_plus10db_pct）
    └── repo/DSC-AMCNet/       Zenodo 官方发布包快照（含官方训练脚本/文档，用于比对）
```

## 4. 关键产物与精确约定

### 数据集
- 文件是 `pickle` 字典（`encoding="latin1"` 读取），不是 torch 文件，也不是 npz。
- key：`(调制名 str, SNR int)`；value：`numpy.ndarray (1000, 2, 128) float32`。
- 铺平顺序：`sorted(keys, key=lambda k: (k[0], k[1]))` → 按调制名字母序、组内 SNR 升序、组内 1,000 条。`split_indices.npz` 的下标就是这套顺序下的 0…219,999，读数据时顺序必须一致。
- 类别索引（0…10）：`8PSK, AM-DSB, AM-SSB, BPSK, CPFSK, GFSK, PAM4, QAM16, QAM64, QPSK, WBFM`。

### 模型
- 输入 `(batch, 2, 128)`，输出 `(batch, 11)` 的 **原始 logits**（末尾是裸 Linear，没有 softmax）。
- 4,961 个可训练参数；state_dict 50 个 key。
- 当前权重 SHA256：`F7A05C25672A0074CFF30673C87ACB097897FA96662264CB57F9A959A91BAAB6`，36,437 B。
- 评估口径：`+10 dB` 测试子集（`snr == 10`，n = 2,219）。`validate_backend.py` 内部用的是 n = 2,210 的分箱，两者差 9 条，数字略有出入属正常。

### ONNX / 生成的 C 代码
- ONNX：IR 7、opset 13、输入名 `input`、输出名 `logits`、固定 batch 1。
- INT8 图：输入 scale 0.002135044 / zero-point −4；输出 scale 0.350085318 / zero-point 63；68 个 QDQ 节点。
- 生成的 C 宏：`AI_NETWORK_IN_1_SIZE = 256`（1024 B float）、`AI_NETWORK_OUT_1_SIZE = 11`（44 B）、`AI_NETWORK_DATA_WEIGHTS_SIZE = 5804`、`AI_NETWORK_DATA_ACTIVATIONS_SIZE = 4488`。
- X-CUBE-AI 10 生成的 `g_network_data_map_activations` 数据指针为 `NULL`，**应用必须自己提供 activations 缓冲**（固件里 `AI_ALIGNED(4) static ai_u8 activations[...]`）。

### 目标硬件（自制核心板 LXB723ZG-P1）
- MCU：STM32H723ZGT6（LQFP144）；晶振 **25 MHz**（不是 Nucleo 的 8 MHz）；无板载 ST-LINK、无 USB 转串口。
- 时钟：VOS0，HSE 25 MHz → PLL1 **M=25 / N=544 / P=1** → 544 MHz。（若改用 Nucleo：8 MHz → M=2 / N=136 / P=1。550 MHz 需要额外开 `CPU_FREQ_BOOST`，代价是 ITCM/DTCM 的 ECC 失效。）
- 串口：USART1，PA9 = TX / PA10 = RX，115200 8N1，已引到调试排针 `3V3 RST TXD RXD`。
- 调试：SWD，PA13 = SWDIO / PA14 = SWCLK，外接 ST-LINK。

## 5. 关键里程碑

| 日期 | 事件 |
|---|---|
| 2026-10-16 | CSV→pickle 数据集构建；首次训练仅 74.99% |
| 2026-10-16 | 查到官方权重失效、CSV 缺 Q；改用官方 I/Q pickle 重训 → **89.73%**；PTQ → 89.36% |
| 2026-10-17 | 补齐固件头文件、修 L2 归一化、activations 缓冲；ST Edge AI 转换出 C 代码 |
| 2026-10-19 | 清理工作区（1.50 GB → 649 MB），删除生成文档 |
| 2026-10-22 | 装好 Keil DFP（STM32H7 4.1.3）；CubeMX CLI 生成 NUCLEO-H723ZG 工程并手工接入 AI 模型 |
| 2026-10-27 | 定位并修复 stdio/半主机启动崩溃、DWT 不计数；544 MHz 首次跑通，实测 2.55 ms（balanced） |
| 2026-10-28 | 换 `--optimization time` 重生成（激活 7,488 B），实测 **1.827 ms**；ITCM/DTCM 对照无收益，回退默认布局 |

## 6. 复现命令

> 注意：清理时删掉了自建的 `.pylibs`，所以现在直接跑 Python 会缺依赖。先按 `environment.yml`（conda）或 `requirements.txt`（pip）重建环境。`.pylibs` 方案当时是因为本机 pip 在沙箱里无法安装。

```powershell
# 数据校验（期望：220,000 条 / 11 类 / -20..18 dB / 176,000:44,000）
python datasets/prepare_dataset.py datasets/RML2016.10a_dict.dat datasets/split_indices.npz

# 训练（论文协议，约 10.6 s/epoch，100 epoch ≈ 18 分钟）
python models/train.py --data datasets/RML2016.10a_dict.dat `
    --split datasets/split_indices.npz --epochs 100 `
    --output models/checkpoints/dscamcnet_best.pt

# 量化 + 后端校验
python quantization/ptq_quantize.py --checkpoint models/checkpoints/dscamcnet_best.pt `
    --data datasets/RML2016.10a_dict.dat --split datasets/split_indices.npz `
    --output quantization/outputs/dscamcnet_int8.onnx
python quantization/validate_backend.py --checkpoint models/checkpoints/dscamcnet_best.pt `
    --data datasets/RML2016.10a_dict.dat --split datasets/split_indices.npz

# ONNX 精度评估（+10 dB / 全 SNR）
python _official/eval_onnx.py --onnx quantization/outputs/dscamcnet_int8.onnx `
    --data datasets/RML2016.10a_dict.dat --split datasets/split_indices.npz
```

重新生成模型 C 代码（本机已装 X-CUBE-AI 10.2.1）：

```powershell
$ste = "$env:USERPROFILE\STM32Cube\Repository\Packs\STMicroelectronics\X-CUBE-AI\10.2.1\Utilities\windows\stedgeai.exe"
& $ste generate --target stm32h7 --name network `
    --model quantization\outputs\dscamcnet_int8.onnx `
    --input-data-type float32 --output-data-type float32 `
    --compression none --no-inputs-allocation --no-outputs-allocation `
    --optimization time `
    --output st_ai_gen
# 生成 7 个文件，拷进 stm32_deployment/X-CUBE-AI/
```

> `--optimization time`（2026-10-28 加入）把 activations 从 4,488 B 提到 7,488 B，板端从 2.55 ms 降到 1.83 ms；若重新生成时漏掉这个参数，会退回 balanced 版本。

## 7. 下一步 Todo（按顺序）

> 2026-10-28 更新：上板链路已通，以下从「搭建」改为「缩小 1.54× 差距」。

1. **int8 I/O 对照**：用 `--input-data-type int8 --output-data-type int8` 重新生成，固件直接喂 int8（scale 0.002135044 / zp −4），去掉每次推理的 256 点 float→int8 与 11 点 int8→float 转换，测差值（预期 ~1%）。
2. **量化图对照**：把仓库里论文原版 `stm32_deployment/X-CUBE-AI/`（4,488 B 激活那版）换回工程实测一次，确认差距在图还是在生成参数。
3. **per-channel 量化实验**：改 `ptq_quantize.py` 的 `per_channel=True`，看 QDQ 节点数与 MACC 路径是否变化；若内核效率提升明显，再评估精度回退。
4. **官方 on-target 验证**：`stedgeai validate --mode target --desc serial:COM3:115200`，用真实测试集在板上核对逐样本精度（当前板端只喂了全零）。
5. OTA：需要 HackRF + RTL-SDR，用 `ota_collection/collect_ota.py`（需 pyrtlsdr/onnxruntime）。

## 7.1 上板联调记录（2026-10-28）

**硬件**：NUCLEO-H723ZG（MB1364E，STLINK-V3E 板载）；HSE 走 ST-LINK MCO 的 8 MHz（板上 25 MHz 晶振 X3 标 DNF）；VCP = USART3 PD8/PD9，枚举为 `COM3`；LED：LD1 绿 PB0、LD2 黄 PE1、LD3 红 PB14。

**工具链**：CubeMX 6.18.1 CLI 生成工程（`-q script`：`config load` → `project generate`）；Keil µVision 5.24a + **ARM Compiler 6.7**（`pCCUsed 6070000::V6.7::ARMCLANG`，必须放在 `<Target>` 层）；ST-LINK 烧录。

**时钟**：VOS0、HSE bypass 8 MHz、PLL1 **M=2 / N=136 / P=1 / Q=4 / R=2** → SYSCLK 544 MHz；HPRE /2（AXI=272 MHz）、APB1-4 /2、FLASH_LATENCY_3。HPRE 设为 /1 时 CubeMX 会算出非法的 FLASH_LATENCY_0，故 HCLK 必须 /2。

**踩过的四个坑**（都已修）：
1. **链接 printf 会在启动阶段崩**：AC6 的 stdio 初始化走 `_sys_open`/半主机 `BKPT 0xAB`，未接调试器即 HardFault，程序到不了 main。→ 固件改用自写 `uart_printf`（`Src/main.c` + `Inc/uart_printf.h`），全程不链接 stdio。
2. **浮点 printf 打不出来**：AC6 的 `%f` 依赖完整格式化路径，串口只出 0.000。→ 时延/置信度全部改成整数定点打印（`us_x100`、`conf_x1000`）。
3. **DWT 不计数**：H7 需要 `DBGMCU->CR |= DBGMCU_CR_DBG_TRACECKEN` 且 `DWT->LAR = 0xC5ACCE55` 解锁；否则 CYCCNT 恒 0。
4. **µs 换算用错时钟**：CPU 544 MHz 而 HCLK 272 MHz，`dwt_measure.c` 的 µs 换算改用 `SystemCoreClock`。

**实测**（`[DWT] N=1000`，输入全零）：

| 布局 | mean |
|---|---|
| 默认（Flash + AXI SRAM） | **1.827 ms**（995,505 cycles） |
| 全部 RO 搬 ITCM | 1.842 ms |
| RW/ZI 搬 DTCM | 1.835 ms |
| balanced 生成（4,488 B 激活） | 2.55 ms |
| 论文 | 1.183 ms（643,552 cycles） |

结论：内存布局 ±0.7%，瓶颈在 X-CUBE-AI 10.2 内核（≈6.1 cycles/MACC，论文 ≈3.9）。

**与官方流程的差异**：CLI 生成 + 手工集成（非 CubeMX 中间件向导）；`--optimization time`（官方默认 balanced）；float32 I/O（官方默认 int8）；Keil AC6（官方示例 CubeIDE GCC）；未做 on-target validation。激活缓冲已按官方模板对齐到 `AI_ALIGNED(32)`，实测无变化。

## 8. 环境与工具现状

| 项 | 状态 |
|---|---|
| Python | 3.14.5，依赖包已删（需按 requirements.txt 重装） |
| torch / onnx / onnxruntime | 曾用版本：2.14.1+cpu / 1.23.2 / 1.24.1 |
| STM32CubeMX | ✅ 6.18.1（`D:\APP\STM32CubeMX\STM32CubeMX\STM32CubeMX.exe`） |
| X-CUBE-AI | ✅ 10.2.1（`%USERPROFILE%\STM32Cube\Repository\Packs\STMicroelectronics\X-CUBE-AI\10.2.1`） |
| ST Edge AI Core | ✅ 2.2.0（随 X-CUBE-AI 10.2.1 提供） |
| Keil MDK | ✅ µVision 5.24a + ARM Compiler 6.7（`D:\APP\Keil5`），STM32H7xx_DFP 4.1.3 手工安装 |
| STM32CubeIDE / arm-none-eabi-gcc | ❌ 未安装（最终没走 CubeIDE 路线） |
| 部署工程 | `D:\DSC_AMCNet_Deployment`（CubeMX 生成 + 手工集成 AI 与固件，不在本仓库内） |
| 硬件 | NUCLEO-H723ZG（板载 STLINK-V3E + VCP），COM3；OTA 还需 HackRF + RTL-SDR |

## 9. 已知限制

- 板端 benchmark 输入是全零，只验证时延，未验证精度；精度核对需跑 `stedgeai validate --mode target` 或用真实样本回放。
- 部署工程 `D:\DSC_AMCNet_Deployment` 在仓库之外，`.ioc` 中没有 X-CUBE-AI 中间件条目：在 CubeMX 里重新 Generate 会覆盖 `main.c` 的 544 MHz 时钟与 `inference_main()` 调用，也会丢 AI 集成，改配置后要重新打补丁。
- `.tmp/` 是联调期临时目录（诊断脚本、串口日志、截图），已加入 `.gitignore`，不入库。
- 0.2 ms 级的差距尚未解释：已排除内存布局、激活对齐、优化等级、编译工具链；剩余怀疑为 int8 I/O 转换与量化图差异（见 §7）。
- 当前 HEAD 之上没有未提交的源码改动；本次提交只涉及 `PROJECT_STATUS.md` 与 `.gitignore`。
