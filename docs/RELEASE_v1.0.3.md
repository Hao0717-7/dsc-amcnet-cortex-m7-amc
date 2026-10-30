# Release v1.0.3 — 可复现的构建文件 + 如实声明

- Tag：`v1.0.3`（annotated，指向 `cef3994`）
- 日期：2026-10-30
- 上一版：`v1.0.2`（`5688ab1`，重生成 `docs/REPORT_preview.png` 到实测口径）
- 仓库：https://github.com/Hao0717-7/dsc-amcnet-cortex-m7-amc

## 本版做了什么

1. **把部署工程的 8 个工程胶水文件收进仓库**（`stm32_deployment/custom_engine/`，与 `D:\DSC_AMCNet_Deployment`
   逐字节相同）：`main.h`、`stm32h7xx_hal_conf.h`、`stm32h7xx_it.h/.c`、`stm32h7xx_hal_msp.c`、
   `system_stm32h7xx.c`、`uart_printf.h`、`inference_main.h`。加上 v1.0.1 收进来的
   `dwt_measure.c/.h`、`uvprojx`、`.sct`、`.ioc`，仓库里已有「0.951/0.960 ms 那份固件」的全部手写源码；
   仍然不含 ST 生成/体积大的部分（HAL/CMSIS、`Middlewares/ST/AI` 运行时、Keil 构建产物）。
2. **`stm32_deployment/custom_engine/README.md` 增加「工程文件的完整度」**：与部署工程逐文件对应表、
   仍缺清单、四步重建流程。
3. **`README.md` 增加「未复现 / 未验证项（如实声明）」**：OTA 82.76%（缺 HackRF+RTL-SDR）、
   ULCNN 2.104 ms / 1.78×（上游无部署代码）、能耗 0.091 mJ（需要电流表，可串 NUCLEO 的 JP4）；
   并说明 550 MHz 与 X-CUBE-AI 10.2.0 也只是论文/上游声明值（本仓库实测 544 MHz、pack 10.2.1）。

## 关键数字（本仓库实测，非论文值）

| 项目 | 数值 | 证据 |
|---|---|---|
| 自研 INT8 引擎延迟 | **0.951 ms（全零）/ 0.960 ms（真实窗）**，N=1000 | `docs/evidence/serial_boot_run_2026-10-29.log` |
| 与论文 1.183 ms 对比 | 快 **18.8%**（0.960 / 1.183 = 0.812） | 同上 |
| X-CUBE-AI 运行时（对照） | 1.827–1.843 ms | `PROJECT_STATUS.md` §7.1 |
| 44,000 窗板端准确率 | **59.2045%**（参考 59.2091%，差 2 窗） | `stm32_deployment/results/p5_accuracy_44k.json` |
| logits 逐位一致 | 98.80%（argmax 一致 99.98%） | 同上 |
| golden vectors | 22/22（±1 LSB + argmax） | `docs/evidence/golden.log` |
| 实时流式 | 20/20、972.6 µs/窗、往返 7.5 ms | `docs/evidence/stream.log` |
| 固件占用 | Flash 79.01 KB / RAM 47.29 KB（ROM 80,904 B、RW+ZI 48,424 B） | Keil map，`docs/P5_RESOURCES.md` |
| 精度 | FP32 59.56% / +10 dB 89.73%；INT8 59.21% / 89.36%（掉点 0.37 pp） | `stm32_deployment/results/ref_accuracy.json` |

## 验证

- `python stm32_deployment/tools/check_clock_config.py`：.ioc 分频（HSI → 544/272/136 MHz）与
  `main.c` 里的 VOS0/LDO/VOSRDY/FLASH_LATENCY_3/USART3 全部 OK（该脚本在 v1.0.3 之后的 main 上加入）。
- 全量离线复算：INT8 ONNX 44k 与 `ref_accuracy.json` 逐位一致（diff = 0.0）；FP32 ONNX 与上述 FP32 数字一致。
- 上板复测（2026-10-29 22:42–22:53，COM3）：冷启动 benchmark、golden 22/22、回放 300 窗、
  流式 20/20 全部通过，数字见上表。

## 已知限制（如实声明）

- OTA 采集未复现：缺 HackRF + RTL-SDR 硬件；`ota_collection/collect_ota.py` 也未做整窗 L2 归一化。
- ULCNN 对照未复现：上游只提供 FP32 ONNX（Zenodo，53,288 B），没有部署代码/权重对应的固件。
- 能耗 0.091 mJ 未测：需要 mA 档万用表串 NUCLEO-H723ZG 的 JP4（可测，但本机没有表）。
- 仓库内不含 HAL/CMSIS 与 X-CUBE-AI 运行时库，按 `docs/BUILD.md` §1.1 的重建流程补齐后才可编译。
- 本 tag 之后 main 上还有一批改进（torch 复算 FP32 eval log、PTQ 可复现性校验、
  参数口径脚注、`check_clock_config.py`、`CHANGELOG.txt` 恢复），见随后的 v1.0.4。

## 在 GitHub 上发布

本机 `gh` 未登录（`C:\Users\Hao\gh\bin\gh.exe`），两种方式任选：

```powershell
# 方式一：登录后一条命令
gh auth login
gh release create v1.0.3 --title "v1.0.3: reproducible build files + honest declarations" `
  --notes-file docs/RELEASE_v1.0.3.md
```

方式二：网页 https://github.com/Hao0717-7/dsc-amcnet-cortex-m7-amc/releases/new?tag=v1.0.3
→ 标题填上面的 title，正文粘贴本文件内容。
