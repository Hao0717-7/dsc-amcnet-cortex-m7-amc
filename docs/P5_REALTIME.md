# P5-1 实时数据链路（UART 流式）

目标：把「外部 IQ → 窗口化 → L2 归一化 → 量化 → 推理 → 分类」整条链路放到板上跑，
用 UART 代替真实 ADC/DMA 前端，验证端到端通路与单窗口时延。

## 数据流

```
host (PC)                    board (STM32H723, 544 MHz)
 512 B / window   ────────►  USART3 RX (921600 8N1)
                              ├─ int16 LE → float (÷32768)
                              ├─ 窗口 L2 归一化（256 点）
                              ├─ 量化 q = round(x / 0.00213504443) - 4
                              ├─ dsc_infer_run(q) → 11 logits（SMLAD 引擎）
                              └─ softmax → 类别/置信度
 "C,<cls>,<conf‰>,<us×100>,<name>"  ◄──────── USART3 TX
```

## 协议

| 方向 | 内容 |
|---|---|
| PC → 板 | `'S'`（115200）→ 板回 `[STREAM] switching to 921600...` 并切波特率 |
| PC → 板 | 每窗口 512 B：`I[0..127]` 然后 `Q[0..127]`，小端 int16，满量程 32768 |
| 板 → PC | `C,<cls>,<conf_x1000>,<us_x100>,<name>\r\n` |

进入方式有两种：上电 1.5 s 内收到 `'S'`，或在分类循环中随时收到 `'S'`（循环每 10 ms 轮询一次）。
`'R'` 仍用于 P0 的 int8 回放（`L,q0..q10`）。

## 固件改动（`D:\DSC_AMCNet_Deployment`）

- `Src/main.c`
  - `normalize_l2()`：256 点窗口 L2 归一化（与训练/评测一致）。
  - `stream_loop()`：切 921600 → 收 512 B → 反量化成 float → 归一化 → `dsc_infer_run()` →
    softmax → 回 `C,...`；`compute` 用 DWT 计时（L2+量化+推理）。
  - `dsc_stream_entry()` / `dsc_replay_entry()`：暴露给分类循环的入口。
- `Src/inference_main.c`
  - 分类循环顶部轮询 `HAL_UART_Receive(&huart3, &cmd, 1, 0)`，收到 `'S'`/`'R'` 切模式。
- 构建：`Code=67484 RO=16884 RW=8456 ZI=40104`，0 error。

## PC 端工具

`stm32_deployment/tools/live_stream.py`

```powershell
python stm32_deployment\tools\live_stream.py --n 300 --port COM3
```

随机抽 locked test split 的窗口，发送 int16 平面 IQ，统计板端分类准确率、板端计算时延、往返时延。

## 实测（300 窗口，全 SNR 混合）

| 指标 | 值 |
|---|---|
| 板端分类准确率 | 62.3%（同分布全 SNR 参考 59.2%） |
| 板端计算时延（L2+量化+推理） | **979.0 µs**（min 974.8 / max 984.6） |
| 往返时延 | 7.4 ms |
| UART 理论下限（512 B @921600） | 5.6 ms |

结果文件：`stm32_deployment/results/p5_stream.json`

## 说明 / 后续

- 往返 7.4 ms 中约 5.6 ms 是 UART 传输，**计算只占 0.98 ms**。真实部署应换成
  ADC+DMA（或 SPI/I2S 前端）双缓冲，把采集与推理流水起来，理论上可做到 ~1 ms/窗口。
- 板端对原始 int16 输入做归一化+量化，等价于 PC 端 `l2n → quant` 的前处理，
  所以 62.3% 与参考 59.2% 的小差异来自 300 样本抽样波动，不是链路问题。
- 该模式与 P0 回放模式并存，互不影响；不需要时可删掉 `stream_loop()` 与轮询分支。
- 全量精度验证见 `docs/P5_ACCURACY.md`（44k 窗口，板端与参考一致）。
