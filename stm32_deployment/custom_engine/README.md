# 自研 INT8 推理引擎（custom_engine）

目标：绕开 X-CUBE-AI 10.2.1 的标量 C 运行时（无反汇编级 DSP 指令，实测 1.827 ms），
在 NUCLEO-H723ZG（Cortex-M7 @544 MHz）上把 DSC-AMCNet INT8 单次推理压到 1.2 ms 以内。

## 结果

| 实现 | 单次推理 | 说明 |
|---|---|---|
| X-CUBE-AI 10.2.1（time 优化） | 1.827–1.845 ms | 标量 C 内核，约 6.1 cycles/MACC |
| **本引擎（clean 版）** | **0.966 ms（min 0.965 / max 0.967，N=1000）** | 内联 SMLAD + Q31 定点 requant |
| onnxruntime（PC, Python） | 11.17 ms | 仅作数值参考，非性能对比 |

数值：QPSK@10 dB 样本的 11 个 logits（int8 q）与 onnxruntime 逐位一致：
`55 31 0 48 30 34 51 64 60 60 34`（scale 0.35008532，zp 63）。

## 实现要点

- 量化域：全图激活 `zp = -128`（无符号域 u = q + 128），输入 zp = -4，SE 的 fc2 输出 zp = -5，logits zp = 63；权重/偏置 zp = 0。
- 通道两两打包成 32 位字，内联 `smlad` 做双 16 位乘加；每层用预计算 `sumw[oc]` 吸收输入零点偏移。
- requant 用 Q31 定点乘 + 移位（无逐元素 `lroundf`），四舍五入用 `rint` 语义。
- dw3 卷积的时间轴边界补零必须补 **q = -128**（实数 0），这是第一版数值漂移的根因。
- shuffle 为 `out[(r*4+g)] = in[(g*6+r)]`；SE 均值在未 shuffle 的 pool 张量上计算；GAP 在 shuffle 之后。

## 文件

- `dsc_infer.c` / `dsc_infer.h`：引擎实现与接口（`dsc_infer_init` / `dsc_infer_run`）。
- `dsc_model_data.h`：从 `quantization/outputs/dscamcnet_int8.onnx` 抽出的量化权重/偏置/scale（生成方式见 `stm32_deployment/X-CUBE-AI/`）。
- `test_sample.h`：QPSK@10 dB 参考样本（int8 q，输入 scale 0.00213504443，zp -4）。

## 集成

1. 将 `dsc_infer.c`、`dsc_infer.h`、`dsc_model_data.h`、`test_sample.h` 加入 Keil 工程（Application/AI 组）。
2. `main()` 里调用 `dsc_infer_init()`（一次性打包权重）。
3. 每次推理：`dsc_infer_run(input_f32_256, logits_int8)`；输出 logits 的 scale 为 0.35008532，zp 为 63。
4. 输入需与训练一致：整窗 L2 归一化后按 `q = round(x / 0.00213504443) - 4` 量化。

## 注意

- 本引擎按 DSC-AMCNet 的精确结构硬编码（2×128 输入、3 级 DSC、SE、GAP、11 类）；换模型需重新生成权重与结构。
- 与 X-CUBE-AI 的 `network.c` 无关，工程里两者可并存（后者仅用于对照）。
