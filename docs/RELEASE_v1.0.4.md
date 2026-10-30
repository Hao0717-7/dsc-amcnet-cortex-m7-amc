# Release v1.0.4 — torch 复算与可复现性闭环

- Tag：`v1.0.4`（annotated，指向 `b295f3d`）
- 日期：2026-10-30
- 上一版：`v1.0.3`（`cef3994`，工程胶水文件 + 未复现声明）

## 本版做了什么

1. **PyTorch 侧复算（官方格式 eval log）**：新增 `models/eval_checkpoint.py`，用本地环境（`D:\APP\Pytorch`，
   `torch 2.15.0+cpu`）加载仓库内 `models/checkpoints/dscamcnet_best.pt`，在 44,000 窗 locked test 上复算：
   overall **59.5614%**、+10 dB **89.7251%**（n=2,219）、≥10 dB **88.9302%**（n=11,039），参数 **4,961**；
   与 FP32 ONNX argmax 一致 **44,000/44,000**（max |logit diff| 3.8e-05）。
   产物：`docs/evidence/fp32_eval_log.txt`（逐 SNR 表 + 11×11 混淆矩阵）。
2. **PTQ 可复现性校验**：新增 `quantization/compare_ptq_regen.py`，从 `.pt` 重跑 `ptq_quantize.py`：
   FP32 22/22 initializer 逐字节相同；INT8 98 个 initializer **96/98 逐字节相同**
   （仅 `/se/fc2/Gemm_output_0_scale`、`logits_scale` 两个 scale 常量有舍入差）；
   INT8 精度 **59.2091% / +10 dB 89.3646% / ≥10 dB 88.4410%**，与 shipped 完全一致。
   产物：`docs/evidence/ptq_regen_log.txt`。
3. **参数口径说明**（README 新增一节）：4,961（PyTorch 可训练，含 6 个 BN 的 γ/β 336） →
   4,793（BN 折叠后权重+偏置，= 4,625 + 168，对应 stedgeai analyze 的 `params 4,793 items`） →
   4,869（INT8 QDQ ONNX 98 个 initializer 元素 = int8 4,547 + int32 284 + float32 38）。
4. **工程一致性自动检查**：新增 `stm32_deployment/tools/check_clock_config.py`，一条命令校验
   `.ioc` 的 PLL/分频（HSI 64 → M=8/N=68/P=1 → 544 MHz、HCLK 272、APB 136）与 CubeMX **不写进 .ioc**
   的 `PWR_LDO_SUPPLY` / VOS0 / VOSRDY 等待 / `FLASH_LATENCY_3` / `USART3`；当前输出全部 OK，exit 0。
   `docs/BUILD.md` §1.1 增加「CubeMX 重新生成后必查清单」6 条；`custom_engine/README.md` 重建步骤指向该脚本。
5. **`CHANGELOG.txt` 恢复**：从提交 `9772864^` 原样恢复 2026-10-16 的历史记录（276 行），并加注「历史文件、
   状态以 PROJECT_STATUS.md 为准」。
6. **`docs/RELEASE_v1.0.3.md`**：补 v1.0.3 的发布说明（含 GitHub Release 网页粘贴步骤；本机 `gh` 未登录）。

## README 状态表更新

- Reproducing 表 FP32 行由 ⚠️ 改为 ✅（引用 `docs/evidence/fp32_eval_log.txt`，注明权重为本地重训、
  Zenodo 官方 `.pt` 已损坏，故不与 85.80% 这个具体数字比）。
- PTQ 行补充「从 `.pt` 重跑 PTQ 可复现同结构 ONNX、96/98 initializer 逐字节相同、精度逐位一致」。

## 验证

```
D:\APP\Pytorch\Scripts\python.exe models\eval_checkpoint.py --data datasets\RML2016.10a_dict.dat \
  --split datasets\split_indices.npz --fp32-onnx quantization\outputs\dscamcnet_fp32.onnx \
  --output docs\evidence\fp32_eval_log.txt
D:\APP\Pytorch\Scripts\python.exe quantization\compare_ptq_regen.py
python stm32_deployment\tools\check_clock_config.py
```

## 已知限制

与 v1.0.3 相同：OTA（缺射频硬件）、ULCNN（上游无部署代码）、能耗 0.091 mJ（无电流表）未复现，
已在 README「未复现 / 未验证项」中如实声明。
