# 项目状态与工作区地图

> 面向：接手这个仓库的人或 AI。读完这一页就能知道项目在做什么、做到哪、每个文件夹装的是什么、哪些坑已经踩过。
> 最后更新：2026-10-19 ｜ 分支 `main` ｜ HEAD `a85dbaa`

## 0. 一句话

复现一篇论文：把 4,961 参数的调制识别网络 DSC-AMCNet 部署到 STM32H723（Cortex-M7 544 MHz）上做 INT8 推理，目标延迟 1.183 ms、+10 dB 准确率 87.01%。**数据 → 训练 → 量化 → 模型转 C 代码 已完成**，剩 CubeIDE 编译烧录和 OTA 采集两段。

## 1. 当前进度

| 环节 | 状态 | 产物 | 实测指标 |
|---|---|---|---|
| 数据集构建 | ✅ 完成 | `datasets/RML2016.10a_dict.dat` | 220,000 条，11 类 × 20 SNR × 1,000 |
| 训练 | ✅ 完成 | `models/checkpoints/dscamcnet_best.pt` | **+10 dB 89.73%**，全 SNR 59.56% |
| PTQ 量化 | ✅ 完成 | `quantization/outputs/dscamcnet_int8.onnx` | **INT8 89.36%**，比 FP32 掉 0.37 pp |
| 后端兼容性校验 | ✅ 通过 | `validate_backend.py` 输出 | signed int8 / IR 7 / opset 13 / QDQ 68 节点 |
| ST Edge AI 转换 | ✅ 完成 | `stm32_deployment/X-CUBE-AI/*.c/.h` | 权重 5,804 B、激活 4,488 B、MACC 163,615 |
| CubeIDE 工程与编译 | ⬜ 未做 | — | 本机缺 arm-none-eabi-gcc |
| 烧录上板测延迟 | ⬜ 未做 | — | 目标 mean = 1.183 ms（= 643,552 cycles @ 544 MHz） |
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
    --output st_ai_gen
# 生成 7 个文件，拷进 stm32_deployment/X-CUBE-AI/
```

## 7. 下一步 Todo（按顺序）

1. 安装 STM32CubeIDE 2.1.1（自带 arm-none-eabi-gcc，用于编译和烧录）。
2. 新建工程：MCU 选 `STM32H723ZGTx`；RCC HSE = Crystal（25 MHz）、VOS0、PLL1 25/544/1；SYS → Debug = Serial Wire；USART1 115200（PA9/PA10）；生成代码。
3. 导入模型：`stm32_deployment/X-CUBE-AI/` 的 3 个 `.c` 放进工程的 `X-CUBE-AI/App/`，头文件进 `Middlewares/ST/AI/Inc/`；从 X-CUBE-AI 包复制 `Middlewares/ST/AI/Inc/*.h` 与 `Lib/GCC/STM32H7/NetworkRuntime1020_CM7_GCC.a` 并加入链接。
4. 把 `stm32_deployment/firmware/` 的 4 个文件加进工程，`main.c` 在 `MX_USART1_UART_Init()` 之后调用 `inference_main()`；`syscalls.c` 里用 `__io_putchar` 把 printf 重定向到 USART1。
5. 编译烧录，串口 115200 预期看到 `[AI] weights=5804 B ...`、`[DWT] N=1000 mean=1.183 ms`。
6. 记录延迟：`python hardware_characterization/measure_latency.py --port COMx --baud 115200 --output results/latency_log.csv`（需装 pyserial）。
7. OTA：需要 HackRF + RTL-SDR，用 `ota_collection/collect_ota.py`（需 pyrtlsdr/onnxruntime）。

## 8. 环境与工具现状

| 项 | 状态 |
|---|---|
| Python | 3.14.5，依赖包已删（需按 requirements.txt 重装） |
| torch / onnx / onnxruntime | 曾用版本：2.14.1+cpu / 1.23.2 / 1.24.1 |
| STM32CubeMX | ✅ 6.18.1（`D:\APP\STM32CubeMX\STM32CubeMX\STM32CubeMX.exe`） |
| X-CUBE-AI | ✅ 10.2.1（`%USERPROFILE%\STM32Cube\Repository\Packs\STMicroelectronics\X-CUBE-AI\10.2.1`） |
| ST Edge AI Core | ✅ 2.2.0（随 X-CUBE-AI 10.2.1 提供） |
| STM32CubeIDE / arm-none-eabi-gcc | ❌ 未安装（当前唯一硬阻塞） |
| 硬件 | 自制 LXB723ZG-P1 核心板 + 外接 ST-LINK；OTA 还需 RF 前端 |

## 9. 已知限制

- 沙箱环境（workspace-write）下，`models/`、`quantization/`、`.git` 等原有目录对 shell 只读，写入需要提权；`datasets/` 可写。新目录可直接创建。
- `git` 在本会话内不可执行（`.git/index.lock` 被拒绝），提交需在普通终端手动完成。
- 当前未提交改动：`.gitignore`（M）、`CHANGELOG.txt`（D）、`stm32_deployment/README.md`（M）、`stm32_deployment/firmware/inference_main.c`（M）、`stm32_deployment/X-CUBE-AI/`（未跟踪，建议纳入版本控制）。
