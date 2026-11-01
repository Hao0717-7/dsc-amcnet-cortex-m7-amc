# 仓库目录与阅读索引

> 仓库：`D:\dsc-amcnet-cortex-m7-amc`；本文生成于 2026-11-01。
> 文件数口径：`git ls-files` 共 **186 个跟踪文件**；被 `.gitignore` 忽略的
> 数据/ONNX/日志/缓存不计入，见 §5。
>
> 标记：`[源码]` 手写代码或脚本 · `[生成]` 由脚本/工具产出 ·
> `[证据]` 测量结果、日志、数据快照 · `[第三方]` 外部官方包镜像 ·
> `[忽略]` 本地产物，clone 后不存在。

## 1. 目录树

```text
dsc-amcnet-cortex-m7-amc/            # 仓库根；186 个跟踪文件
├── README.md                        # [文档] 总览、实测数字、论文对照、未复现项
├── PROJECT_STATUS.md                # [文档] 进度、八条坑、产物精确约定、里程碑
├── CHANGELOG.txt                    # [归档] 2026-10-16 旧改动记录（自述已过期）
├── LICENSE                          # [文档] 许可证
├── activate_env.ps1                 # [源码] 激活 conda 环境
├── environment.yml                  # [源码] conda 环境定义
├── requirements.txt                 # [源码] pip 依赖
├── STM32H723ZG MCU/                 # [文档] 芯片数据手册/参考手册 PDF（2 份）
├── .github/workflows/ci.yml         # [源码] 离线 CI + 手工 hardware job
├── docs/                            # [文档] 21 个文件 + evidence/
│   ├── PROJECT_MEMORY.md            # 记忆快照；冲突时以它为准
│   ├── BUILD.md                     # 部署工程重建 + 8 阶段验收 + 离线 CI
│   ├── REPRODUCE.md                 # 端到端复现命令（数据到板端）
│   ├── P0_ACCURACY.md               # 参考精度 + 5,000 窗回放对比
│   ├── P1_BENCHMARK.md              # P1 基准方法（544 MHz 历史读数）
│   ├── P4_OPTIMIZATION.md           # 550 MHz 时延、逐层 cycle、C1-C11
│   ├── P5_ACCURACY.md               # 44k 板端精度与逐位一致率
│   ├── P5_REALTIME.md               # UART 流式链路
│   ├── P5_RESOURCES.md              # 固件 Flash/RAM 组成、能耗缺口
│   ├── P5_BLOCKED.md                # OTA/ULCNN/能耗为何未复现
│   ├── COMPARISON_vs_OFFICIAL_README.md  # 官方 README 对照（已更新到 550 MHz 口径）
│   ├── RELEASE_v1.0.md … v1.0.9.md  # [归档] 8 份发布说明（v1.0.1 / v1.0.2 无单独文件，见 v1.0.3）
│   ├── REPORT.html                  # [生成] 一页式证据报告
│   ├── REPORT_preview.png           # [生成] 报告截图
│   └── evidence/                    # [证据] 11 个条目（10 个文件 + stedgeai_analyze/）
│       ├── fp32_eval_log.txt、ptq_eval_log.txt、ptq_regen_log.txt
│       ├── ptq_per_channel_log.txt、qat_eval_log.txt
│       ├── seed_robustness_log.txt、ulcnn_eval_log.txt
│       ├── latency_dscamcnet_int8.csv     # DWT 时延原始 CSV
│       ├── mcu_golden_2026-10-30.png、mcu_live_bench_2026-10-30.png
│       └── stedgeai_analyze/        # analyze 报告 + ulcnn 子目录
├── completion_flow/                 # 交付物目录（本文件所在）
│   ├── PROJECT_MAP.md               # 3 张流程图 + 关键数字速查
│   ├── DIRECTORY.md                 # 本文件：目录与索引
│   └── figs/                        # [生成] 图 A/B/C 的 PNG 渲染（Mermaid 源在正文里）
│       ├── project_flow_a.png       # 端到端流水线（图 A）
│       ├── project_flow_b.png       # 引擎 int8 数据流（图 B）
│       └── project_flow_c.png       # 验收与 CI（图 C）
├── datasets/                        # 数据脚本与锁定划分
│   ├── prepare_dataset.py           # [源码] 加载 pickle + canonical 排序
│   ├── build_rml2016_from_csv.py    # [源码] 由 CSV 重建 pickle
│   ├── README.md                    # [文档] 数据获取与校验说明
│   ├── split_indices.npz            # [证据] 锁定 split，md5 bb8bc1e7
│   ├── full_2k/selection_manifest.json  # [生成] 2k 子集清单
│   └── RML2016.10a_dict.dat         # [忽略] 225 MB 原始数据
├── models/                          # 模型、训练与评估
│   ├── dsc_amcnet.py                # [源码] 网络结构（4,961 参数）
│   ├── train.py                     # [源码] 训练（seed42）
│   ├── eval_checkpoint.py           # [源码] FP32 checkpoint 44k 评估
│   ├── eval_ulcnn.py                # [源码] ULCNN ONNX 对照评估
│   └── checkpoints/                 # README.md + dscamcnet_best.pt [生成]
├── quantization/                    # [源码] 7 个脚本
│   ├── ptq_quantize.py              # QDQ 静态量化主流程
│   ├── compare_ptq_regen.py、ptq_per_channel.py、qat_experiment.py
│   ├── qat_compare.py、ptq_eval_log.py、validate_backend.py
│   └── outputs/                     # [忽略] ONNX 生成物（git-ignored）
├── hardware_characterization/       # [源码] measure_latency.py、analyze_flash_ram.py
├── ota_collection/collect_ota.py    # [源码] OTA 采集（缺射频硬件，未跑）
├── supplementary/                   # [文档] 论文附录 A/B/C
├── stm32_deployment/                # 部署收口
│   ├── README.md                    # [归档] 旧 CubeIDE 基线指南（1.183 ms）
│   ├── X-CUBE-AI/                   # [生成] stedgeai 生成的 9 个文件
│   ├── custom_engine/               # [源码] 自研引擎 + Keil 工程（23 个文件）
│   │   ├── dsc_infer.c / dsc_infer.h    # 引擎实现与接口
│   │   ├── dsc_model_data.h         # [生成] 从 INT8 ONNX 抽出的常量
│   │   ├── test_sample.h            # [证据] QPSK@+10 dB 参考样本
│   │   ├── app_main.c、app_inference_main.c、dwt_measure.c
│   │   ├── DSC_AMCNet_Deployment.uvprojx / .sct / .ioc
│   │   └── 其余 HAL 胶水、startup、retarget
│   ├── firmware/                    # [归档] X-CUBE-AI 时代旧固件（5 个）
│   ├── results/                     # [证据] 14 个 JSON + evidence/ + figs/
│   │   ├── measured_latest.json     # 权威数字汇总（只取 p2 段）
│   │   ├── ref_accuracy.json        # INT8 ORT 44k 逐 SNR/逐类
│   │   ├── p5_accuracy_44k.json     # 板端 44k（latency_ms 字段已过期）
│   │   ├── golden_vectors.json      # 22 条回归向量
│   │   ├── verify_last.json         # 一键验收 0-8 阶段结果
│   │   ├── host_parity_last.json    # 44k 主机逐位 98.775%
│   │   ├── p0_replay_compare.json   # [忽略] 本地 300 窗回放结果
│   │   ├── p5_accuracy_full.json、p5_stream.json  # [归档] 旧口径
│   │   ├── qat_compare.json、ptq_per_channel.json、ulcnn_eval.json
│   │   ├── compare_official_readme.json、flash_ram_report.json
│   │   ├── evidence/                # 8 份阶段日志（部分被 gitignore）
│   │   └── figs/                    # [生成] latency/per_snr/resources 图
│   └── tools/                       # [源码] 12 个脚本 + host_sim/
│       ├── verify_all.ps1           # 一键验收 0-8 阶段
│       ├── check_clock_config.py、check_docs_consistency.py
│       ├── p0_accuracy.py、p0_replay.py、p5_full_accuracy.py
│       ├── golden_test.py、live_stream.py、measure_results.py
│       ├── gen_report.py、build_flash.ps1、measure_vs_official.py
│       └── host_sim/                # compare_engine.py + host_engine.c
├── _official/                       # [第三方] Zenodo 官方仓库镜像（40 个跟踪文件）
├── st_ai_ws/                        # [忽略] stedgeai 工作区
└── .tmp/、__pycache__/              # [忽略] 临时文件与字节码
```

> 上面未展开的细节：`docs/evidence/stedgeai_analyze/` 内含
> `dscamcnet_int8_analyze_report.txt` 与 `ulcnn/ulcnn_fp32_analyze_report.txt`；
> `stm32_deployment/results/figs/` 三个 PNG 由 `gen_report.py` 生成；
> `_official/` 是 Zenodo 官方发布包镜像，只读参考，不参与构建。

## 2. 文档索引

| 文件 | 一句话用途 | 建议顺序 |
|---|---|---|
| README.md | 总览、实测 6 个数、论文对照、未复现声明 | P0 必读 |
| PROJECT_STATUS.md | 进度、八条坑、产物约定、里程碑、限制 | P0 必读 |
| docs/PROJECT_MEMORY.md | 最新记忆快照；与其他文档冲突以它为准 | P0 必读 |
| completion_flow/PROJECT_MAP.md | 3 张流程图 + 关键数字速查 | P0 必读 |
| completion_flow/DIRECTORY.md | 目录树、索引、证据清单（本文件） | P0 必读 |
| docs/P4_OPTIMIZATION.md | 550 MHz 时延、逐层 cycle、C1-C11 实验 | 深挖 |
| docs/REPRODUCE.md | 数据→训练→量化→板端的复现命令 | 深挖 |
| docs/BUILD.md | 部署工程重建、8 阶段验收、离线 CI | 深挖 |
| docs/P0_ACCURACY.md | 参考精度 + 5,000 窗回放对比 | 深挖 |
| docs/P5_ACCURACY.md | 44k 板端精度、逐位一致率 | 深挖 |
| docs/P5_REALTIME.md | UART 流式链路（550 MHz 20 窗 + 544 MHz 300 窗历史） | 深挖 |
| docs/P5_RESOURCES.md | 固件 Flash/RAM 组成、能耗缺口 | 深挖 |
| docs/P5_BLOCKED.md | OTA/ULCNN/能耗为何未复现 | 深挖 |
| docs/REPORT.html | 一页式证据报告（脚本生成） | 深挖 |
| docs/RELEASE_v1.0.9.md | 最新发布：C8 种子稳健性更正 | 深挖 |
| docs/RELEASE_v1.0.md | 首个端到端复现发布（0.951/0.960 期） | 归档 |
| docs/RELEASE_v1.0.3–v1.0.8.md | 历史发布说明（6 份） | 归档 |
| docs/P1_BENCHMARK.md | P1 基准方法（544 MHz 历史读数） | 归档 |
| docs/COMPARISON_vs_OFFICIAL_README.md | 与官方 README 逐项对照（已更新 550 MHz） | 归档 |
| CHANGELOG.txt | 2026-10-16 旧改动记录 | 归档 |
| stm32_deployment/README.md | 旧 CubeIDE + X-CUBE-AI 基线指南 | 归档 |
| stm32_deployment/custom_engine/README.md | 引擎实现、重建、主机侧校验 | 深挖 |
| stm32_deployment/tools/host_sim/README.md | 主机侧引擎逐位比对工具 | 深挖 |
| datasets/README.md | 数据获取、canonical 排序、锁定 split | 深挖 |
| models/checkpoints/README.md | 权重 Zenodo 下载与校验 | 深挖 |
| supplementary/appendix_a_architecture.md | 论文附录 A：层表与 4,961 参数 | 深挖 |
| supplementary/appendix_b_per_modulation.md | 论文附录 B：逐类精度 | 深挖 |
| supplementary/appendix_c_qat_failure.md | 论文附录 C：QAT 失败分析 | 深挖 |
| _official/repo/DSC-AMCNet/README.md | 官方发布包 README 原文（第三方） | 归档 |

## 3. results / evidence 清单

### 3.1 权威来源（引用当前数字只看这些）

> 下表未写前缀的行：`results/...` 指 `stm32_deployment/results/...`；
> `docs/evidence/...` 指仓库内 `docs/evidence/...`。

| 文件 | 内容 | 备注 |
|---|---|---|
| results/measured_latest.json | 全部实测数字汇总 | 只取 p2 段；早段已过期 |
| results/ref_accuracy.json | INT8 ORT 44k 逐 SNR/逐类 | 59.2091% 的出处 |
| results/p5_accuracy_44k.json | 板端 44k 精度 | 时延字段 0.979 已过期 |
| results/golden_vectors.json | 22 条 golden 向量 | golden 回归基准 |
| results/verify_last.json | 0-8 阶段验收结果 | all_passed=true |
| results/host_parity_last.json | 44k 主机逐位对比 | 98.775% / argmax 99.98% |
| results/evidence/serial_boot.log | 550 MHz banner + DWT | 0.895/0.904 的原始日志 |
| results/evidence/build.log | 全量构建日志 | 481 warning，0 Error |
| results/evidence/flash.log | 烧录日志 | Verify OK |
| results/evidence/golden.log | 22 条向量回归 | 22/22 |
| results/evidence/stream.log | 20 窗流式 | 916.6 us 均值 |
| docs/evidence/latency_dscamcnet_int8.csv | DWT 时延 CSV | 0.895/0.904 原始数据 |
| docs/evidence/*_log.txt | 各阶段评估日志 | fp32/ptq/qat/seed/ulcnn |
| docs/evidence/stedgeai_analyze/*.txt | ST Edge AI analyze 报告 | 工具口径足迹 |
| results/qat_compare.json、ptq_per_channel.json | A4/A5 量化对照 | 实验记录 |
| results/ulcnn_eval.json | ULCNN 复测 | 未复现项证据 |

### 3.2 历史 / 生成物（勿当当前值）

| 文件 | 内容 | 状态 |
|---|---|---|
| measured_latest.json 的 evidence_logs 等段 | 0.951/0.960 ms、79.01 KB | 历史，被 p2 段覆盖 |
| results/p5_accuracy_full.json | 300 窗小样本 board 55.33% | 历史，勿引准确率 |
| results/p5_stream.json | 300 窗 979 us | 历史（544 MHz 期） |
| results/evidence/serial_boot_run_2026-10-29.log | 旧串口日志 | 历史 |
| results/evidence/host_parity.log 等 | 主机/回放/时延阶段日志 | 本地产物，gitignore |
| results/figs/*.png | latency/per_snr/resources | 生成物 |
| docs/REPORT.html、REPORT_preview.png | 一页式报告与截图 | 生成物 |
| results/compare_official_readme.json | 官方对照结构化结果 | 本地产物，gitignore |

> 被 `.gitignore` 忽略、clone 后需重跑才能得到的日志：`host_parity.log`、
> `replay.log`、`latency_stage.log`、`latency_stage.err`、`flash_latency.log`、
> `gen_report.log`、`serial_raw.log`。

## 4. 5 分钟上手路径

按下面四步读，先建立「当前数字是什么、从哪来」，再决定是否深挖：

1. **README.md 的「本仓库实测」表**（约 30 秒）：记住 6 个数——0.895/0.904 ms、
   1.827 ms、59.20%、78.23/47.29 KB、22/22。论文报告值在上一张表，不要混用。
2. **PROJECT_STATUS.md §0 / §1 / §2 / §9**：一句话目标、进度、八条坑、已知限制。
   其中 §2 能避免重复踩坑（数据顺序、量化域、串口、Keil 警告等）。
3. **docs/P4_OPTIMIZATION.md §最终结果 + §逐层 cycle 剖析**：理解 0.904 ms 由
   哪些层构成、23.6% 与 2.02× 怎么算，以及哪些优化失败并回退。
4. **stm32_deployment/results/measured_latest.json**：只看四个键——
   `p2_clock_550mhz_2026_10_31`（当前时延/流式/golden/资源）、`model`（ONNX
   93 节点、4,869 初始化元素）、`accuracy_locked_44k`（FP32/INT8 四舍五入值）、
   `split`（176,000/44,000）。其余顶层键是历史段落或实验附录。

想动手：先读 `docs/REPRODUCE.md`（无板子也能跑数据/训练/量化/ORT 评估），
有板子和 Keil 部署工程时直接跑 `stm32_deployment/tools/verify_all.ps1`。

### 数字新旧速判

- **当前**：0.895 / 0.904 ms、550 MHz、78.23 / 47.29 KB、59.2091%、93 节点。
- **历史**：0.951 / 0.960（C1 前）、0.905 / 0.914（544 MHz）、0.967 / 0.975、
  0.979 ms 流式、972.2 us、79.01 KB、1,183 ms 基线指南。
- 见到「0.979」「79.01」「94 节点」基本可判定为旧口径或误写。

## 5. 被忽略的生成物（clone 后不存在）

| 路径 | 说明 |
|---|---|
| datasets/RML2016.10a_dict.dat | 225 MB DeepSig 原始 pickle |
| datasets/full_2k/RML2016.10a_2k_dict.dat | 2k 子集数据 |
| datasets/__pycache__/、models/__pycache__/ 等 | Python 字节码 |
| models/checkpoints/dscamcnet_best.prev-89.00.pt | 旧版 checkpoint |
| quantization/outputs/*.onnx | FP32/INT8 ONNX 生成物 |
| st_ai_ws/、.tmp/ | stedgeai 工作区、临时文件 |
| stm32_deployment/results/compare_official_readme.json | 官方对照结果 |
| stm32_deployment/results/flash_ram_report.json | 构建 map 解析结果 |
| stm32_deployment/results/p0_replay_compare.json | 本地回放对比结果 |
| docs/evidence/stedgeai_analyze/ulcnn/ulcnn_fp32_c_info.json | analyze 中间文件 |

需要重建时：`datasets/prepare_dataset.py`（数据）、`models/train.py`（权重）、
`quantization/ptq_quantize.py`（ONNX）、`stm32_deployment/tools/gen_report.py`
（报告与图）。
