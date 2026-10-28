# 端到端复现指南（论文 → 板端）

本仓库复现 DSC-AMCNet 的 INT8 部署：数据 → 训练 → 量化 → 模型 C 代码 → Keil 工程 → 烧录 → 精度/时延验收。

## 0. 环境

| 工具 | 版本 | 说明 |
|---|---|---|
| Python | 3.14（3.10+ 亦可） | `pip install -r requirements.txt`；P0/P1 脚本另需 `onnxruntime`、`pyserial` |
| PyTorch / ONNX | 2.x / 1.2x | 训练与导出 |
| STM32CubeMX | 6.18.1 | 生成 MDK 工程 |
| X-CUBE-AI | 10.2.1 pack | 提供 `stedgeai` 与运行时库 |
| Keil MDK | 5.24a + ARM Compiler 6.7 | 编译/烧录（DFP `Keil.STM32H7xx_DFP 4.1.3` 需手工安装） |
| 硬件 | NUCLEO-H723ZG | 板载 STLINK-V3E，VCP = COM3 |

## 1. 数据与训练

```powershell
python models/train.py --data datasets/RML2016.10a_dict.dat `
    --split datasets/split_indices.npz --epochs 100 `
    --output models/checkpoints/dscamcnet_best.pt
```
期望：+10 dB FP32 89.73%（论文 87.01%，不可复现，README 已说明）。

## 2. 量化（必须 signed int8）

```powershell
python quantization/ptq_quantize.py --checkpoint models/checkpoints/dscamcnet_best.pt `
    --data datasets/RML2016.10a_dict.dat --split datasets/split_indices.npz `
    --backend fbgemm --output quantization/outputs/dscamcnet_int8.onnx
```
`qnnpack` 会得到 unsigned 量化（zero-point 128），与 X-CUBE-AI/自研引擎都不兼容。

## 3. 模型 C 代码（可选，仅作对照）

```powershell
$ste = "$env:USERPROFILE\STM32Cube\Repository\Packs\STMicroelectronics\X-CUBE-AI\10.2.1\Utilities\windows\stedgeai.exe"
& $ste generate --target stm32h7 --name network `
  --model quantization\outputs\dscamcnet_int8.onnx `
  --input-data-type float32 --output-data-type float32 `
  --compression none --optimization time --output st_ai_gen
```
本项目的正式推理路径用自研引擎（`stm32_deployment/custom_engine/dsc_infer.c`），X-CUBE-AI 生成物只用于对比。

## 4. 固件

- 部署工程：`D:\DSC_AMCNet_Deployment`（结构见 `docs/BUILD.md`）
- 补丁源码快照：`stm32_deployment/custom_engine/`（`app_main.c`、`app_inference_main.c`、`dsc_infer.c`、`retarget_keil_h723.c`、`startup_stm32h723xx.s`）
- 一键构建+烧录+抓串口：`stm32_deployment/tools/build_flash.ps1`

预期输出：

```
[AI] weights=5804 B (flash) | activations=7488 B | io=1024+44 B
[DIAG] SYSCLK=544000000 Hz ... CCR=00070200
[RESULT] ... Mean : 0.958 ms
[RESULT-SAMPLE] ... Mean : 0.968 ms
```

## 5. 精度验收（P0）

```powershell
python stm32_deployment/tools/p0_accuracy.py              # ORT 44k 参考
python stm32_deployment/tools/p0_replay.py --n 5000 --flash
```
参考：overall 59.21%，+10 dB 89.36%。
板端回放：98.46% 样本 11 logits 全等，argmax 一致 99.88%，准确率 59.04%。

## 6. 结果汇总（与论文对照）

| 指标 | 论文 | 本仓库 |
|---|---|---|
| 参数量 | 4,961 | 4,961 |
| INT8 +10 dB 精度 | 86.92% | 89.36%（重训后；论文 FP32 87.01% 不可复现） |
| Latency | 1.183 ms | **0.958 ms（全零）/ 0.968 ms（真实样本）** |
| 运行时 | X-CUBE-AI（未注明版本） | 自研 SMLAD 引擎（X-CUBE-AI 10.2.1 实测 1.827 ms） |

## 7. 许可与引用

- 本仓库代码：MIT（见 `LICENSE`）。
- 数据集：RML2016.10a（DeepSig / O'Shea et al.）。
- 预训练权重：Zenodo DOI `10.5281/zenodo.21167587`（官方发布权重在本地数据上仅 26%，见 `PROJECT_STATUS.md` 说明）。
- 论文：DSC-AMCNet（IEEE IoT-J 投稿复现包）。

## 8. 已知限制

- 板端精度验收用了 5,000 样本子集（非全 44k），置信区间 ±0.7 pp。
- `D:\DSC_AMCNet_Deployment` 不在仓库内；CubeMX 重新生成会覆盖 `Src/main.c`，补丁清单见 `docs/BUILD.md`。
- OTA 链路未验证（需 HackRF + RTL-SDR）。
