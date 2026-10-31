# STM32 Deployment

Step-by-step instructions to deploy DSC-AMCNet INT8 on the NUCLEO-H723ZG
using STM32CubeIDE 2.1.1 and X-CUBE-AI 10.2.0 (ST Edge AI Core 2.2.0).

---

## Prerequisites

| Tool | Version | Notes |
|---|---|---|
| STM32CubeIDE | 2.1.1 | Available at st.com |
| X-CUBE-AI | 10.2.0 | Install via CubeMX Embedded Software |
| ST Edge AI Core | 2.2.0 | CLI: `stedgeai` |
| STM32CubeMX | 6.11.x | Bundled with CubeIDE |
| Arm GCC | 12.3 | Bundled with CubeIDE |

**Board:** NUCLEO-H723ZG (STM32H723ZG, Cortex-M7, **550 MHz**（2026-10-31 起；此前 544 MHz）, 1 MB Flash, 564 KB RAM)

> 2026-10-31 更新：当前固件用 **HSI 64 MHz → PLL1 M=8 / N=68 + FRACN 6144/8192 → 550 MHz**、`FLASH_LATENCY_4`，片内 `CPU_FREQ_BOOST` 已使能。下文 Step 4 里 HSE 晶振 / 544 MHz 的步骤是旧版说明，仅作历史参考；以 `docs/BUILD.md` §1.1 与仓库内 `custom_engine/app_main.c` 为准。

---

## Step 1: Verify the ONNX Model

Before importing into X-CUBE-AI, confirm the ONNX was generated with the
`fbgemm` backend (signed INT8):

```bash
python quantization/validate_backend.py \
    --checkpoint models/checkpoints/dscamcnet_best.pt \
    --data datasets/RML2016.10a_dict.dat \
    --split datasets/split_indices.npz
```

Expected output:
```
fbgemm  → zero-point=0, dtype=torch.qint8   ← COMPATIBLE ✓
qnnpack → zero-point=128, dtype=torch.quint8 ← INCOMPATIBLE ✗
```

> **Platform note.** The comparison above needs a PyTorch build with the
> `fbgemm`/`qnnpack` quantized kernels. Hosts that only register `onednn`
> (e.g. the Windows CPU wheel) raise `NotImplementedError: Could not run
> 'quantized::conv1d' ...`. On those hosts the script skips the eager-mode
> path and validates the exported artifact instead — signed int8 weights
> (zero-point 0), int8 zero-point tensors, IR ≤ 8, and the INT8-vs-FP32
> accuracy delta:
>
> ```
> Weight zero-points : [0] -> signed int8 (0) ✓
> Q/DQ zero-point dtype(s): ['int8'] -> all int8 ✓
> IR version <= 8 (X-CUBE-AI 10.2): yes ✓
>   +10 dB accuracy (n=2210): FP32 89.50%  INT8 89.55%  Δ +0.05 pp
> VERDICT: signed-int8 QDQ artifact -> X-CUBE-AI COMPATIBLE ✓
> ```
>
> (Values from the retrained weights, 2026-10-16. Measured on the locked test
> split the same pair scores FP32 89.73% / INT8 89.36%; the script's internal
> +10 dB bin is 9 samples narrower.)

---

## Step 2: Analyze and generate the C code with ST Edge AI Core CLI

`stedgeai` ships inside the X-CUBE-AI pack. With X-CUBE-AI **10.2.1** installed
through CubeMX it lives at
`%USERPROFILE%\STM32Cube\Repository\Packs\STMicroelectronics\X-CUBE-AI\10.2.1\Utilities\windows\stedgeai.exe`.

```powershell
$ste = "$env:USERPROFILE\STM32Cube\Repository\Packs\STMicroelectronics\X-CUBE-AI\10.2.1\Utilities\windows\stedgeai.exe"
& $ste generate --target stm32h7 --name network `
    --model quantization\outputs\dscamcnet_int8.onnx `
    --input-data-type float32 --output-data-type float32 `
    --compression none --no-inputs-allocation --no-outputs-allocation `
    --output st_ai_gen
```

Measured output for this model (2026-10-17, X-CUBE-AI 10.2.1 / ST Edge AI Core
2.2.0, `stedgeai generate` report):

```
params #           : 4,793 items (18.72 KiB)
input 1/1          : 'input', f32(1x2x128), 1024 Bytes, user
output 1/1         : 'logits_QuantizeLin..conversion', f32(1x11), 44 Bytes, user
macc               : 163,615
weights (ro)       : 5,804 B (5.67 KiB) (1 segment)
activations (rw)   : 4,488 B (4.38 KiB) (1 segment)
ram (total)        : 5,556 B (5.43 KiB) = 4,488 + 1,024 + 44
```

Why these flags:

- `--input-data-type float32 --output-data-type float32` — the ONNX I/O is int8
  (`input` scale 0.002135044 / zero-point −4; `logits` scale 0.350085318 /
  zero-point 63). `firmware/inference_main.c` feeds and reads `float` buffers,
  so the float32 I/O variant is required; otherwise the generated macros are
  int8 and the firmware has to quantize by hand.
- `--no-inputs-allocation --no-outputs-allocation` — the firmware owns
  `input_data[2*128]` / `output_data[11]` and points the `ai_buffer` structs at
  them (`inference_main.c`, `run_inference_once`). With the default
  (allocate-inside-activations) those external pointers are ignored.
- `--name network` — the firmware includes `network.h` / `network_data.h` and
  uses the `AI_NETWORK_*` macros.

The committed reference copies of the generated sources are in
`stm32_deployment/X-CUBE-AI/` (`network.c`, `network_data.c`,
`network_data_params.c`, `network.h`, `network_data.h`,
`network_data_params.h`, `network_config.h`, plus the generation report).

> **Paper Table II numbers vs. what the tool prints.** Table II lists 51.26 KB
> Flash / 12.09 KB RAM / ≈1.2M MACC. The converter reports 5.67 KiB of weights,
> 4.38 KiB of activations and 163,615 MACC. The Flash figure in the paper
> includes the statically linked AI runtime (tens of KB) on top of the weights
> and a different build/compression configuration, so the converter's own
> numbers are the ones to trust when checking the artifact; do not gate on the
> paper's 51.26 KB. (See `network_generate_report.txt` in
> `stm32_deployment/X-CUBE-AI/`.)

If Flash or RAM differs wildly from the numbers above, recheck the ONNX (wrong
backend or architecture variant).


---

## Step 3: Create STM32CubeIDE Project

1. Open STM32CubeIDE → File → New → STM32 Project
2. Select board `NUCLEO-H723ZG` — or, for a custom H723ZGT6 board, switch to
   **MCU Selector** and pick `STM32H723ZGTx` (LQFP144). The HSE crystal differs
   between the two, which changes the PLL settings (step 4).
3. Project name: `DSC_AMCNet_Deployment`
4. Language: C, Executable

---

## Step 4: Configure Clock (544 MHz) and UART

In CubeMX (double-click the `.ioc` file):

1. **RCC:** HSE = Crystal/Ceramic Resonator, VOS0, SYSCLK = 544 MHz
   (paper Table I). PLL1 depends on the crystal:

   | Board | HSE | PLL1 M / N / P | ref1_ck |
   |---|---|---|---|
   | NUCLEO-H723ZG | 8 MHz | 2 / 136 / 1 | 4 MHz |
   | custom 25 MHz board (e.g. LXB723ZG-P1) | 25 MHz | 25 / 544 / 1 | 1 MHz |

   VOS0 is mandatory above 400 MHz. Keep `CPU_FREQ_BOOST` off unless you target
   the 550 MHz absolute maximum; it disables ECC on ITCM/DTCM.
2. **UART:** Enable one USART in Asynchronous mode, 115200 8N1.
   - The pin pair quoted in the original release notes (`PA8 = USART3_TX`,
     `PB11 = USART3_RX`) is **wrong for STM32H723**: PA8 has no USART3
     alternate function on this part (PA8 = MCO1 / TIM1_CH1 / I2C3_SCL /
     USART1_CK / UART7_RX …). Use the pin pair CubeMX offers for USART3
     (PB10/PB11, PC10/PC11, PD8/PD9 or PB12/PB13/PB14) — or simply USART1 on
     PA9/PA10, which is what the LXB723ZG-P1 debug header exposes as TXD/RXD.
   - Redirect `printf` to that UART: add `__io_putchar()` to `syscalls.c`
     (`HAL_UART_Transmit(&huart1, ...)` for USART1).
3. **Generate code** (Project → Generate Code)


---

## Step 5: Import X-CUBE-AI Model

Two equivalent routes. The CLI route (step 2) is what was used to produce the
committed sources in `stm32_deployment/X-CUBE-AI/`; the GUI route additionally
sets up the middleware tree and the runtime library for you.

**GUI (CubeMX):**

1. In CubeMX: **Additional Software → X-CUBE-AI**
2. Enable X-CUBE-AI, add Network:
   - Model file: `quantization/outputs/dscamcnet_int8.onnx`
   - Network name: `network`
   - Compression: None (already INT8)
   - In the X-CUBE-AI advanced settings, disable input/output buffer allocation
     so the firmware keeps owning `input_data` / `output_data`
3. Click **Analyze** — expect ≈5.67 KiB weights / ≈4.38 KiB activations
   (163,615 MACC), not the paper's 51.26 KB / 12.09 KB
4. Click **Generate Code**

**CLI (already run, reproducible):** the `stedgeai generate` command in step 2
writes seven files: `network.c`, `network_data.c`, `network_data_params.c`,
`network.h`, `network_data.h`, `network_data_params.h`, `network_config.h`.
Copies of all of them plus `network_generate_report.txt` live in
`stm32_deployment/X-CUBE-AI/`.

**What the firmware needs from the pack.** The X-CUBE-AI runtime is *not*
generated: copy the middleware headers and the CM7 GCC runtime library into the
project (CubeMX does this automatically on the GUI route):

- headers: `<pack>/Middlewares/ST/AI/Inc/*.h` → project
  `Middlewares/ST/AI/Inc/` (at minimum `ai_platform.h`,
  `ai_datatypes_defines.h`, `ai_datatypes_format.h`, `core_*.h`)
- runtime: `<pack>/Middlewares/ST/AI/Lib/GCC/STM32H7/NetworkRuntime1020_CM7_GCC.a`
  → link it in the build

After this step the project tree contains the classic layout:

```
Middlewares/ST/AI/
├── Inc/
│   ├── ai_datatypes_defines.h
│   ├── ai_platform.h
│   ├── network.h
│   ├── network_data.h
│   └── network_data_params.h
└── Lib/NetworkRuntime1020_CM7_GCC.a
X-CUBE-AI/App/
├── network.c
├── network_data.c
└── network_data_params.c
```

> **Activations buffer.** X-CUBE-AI 10 emits `g_network_data_map_activations`
> with a `NULL` data pointer, i.e. the application must supply the scratch
> memory. `firmware/inference_main.c` therefore declares
> `AI_ALIGNED(4) static ai_u8 activations[AI_NETWORK_DATA_ACTIVATIONS_SIZE]`
> and passes `AI_NETWORK_DATA_ACTIVATIONS(activations)` to
> `ai_network_init()`; passing `NULL` there (as the original release firmware
> did) fails at init.


---

## Step 6: Add DSC-AMCNet Firmware Files

Copy the firmware sources into the project (both `.c` **and** both `.h` files —
the release zip was missing the headers):

```
Core/Src/
├── inference_main.c   ← from stm32_deployment/firmware/
├── dwt_measure.c      ← from stm32_deployment/firmware/
Core/Inc/
├── inference_main.h   ← from stm32_deployment/firmware/
├── dwt_measure.h      ← from stm32_deployment/firmware/
```

In `Core/Src/main.c`, after HAL_Init / SystemClock_Config / the UART init:
```c
#include "inference_main.h"
// ...
int main(void) {
    HAL_Init();
    SystemClock_Config();   // ensures 544 MHz
    MX_USART1_UART_Init();  // whichever UART you configured in step 4
    inference_main();       // does not return
}
```

---

## Step 7: Build and Flash

```
Project → Build Project   (Ctrl+B)
Run → Debug               (F11) or Run → Run (Ctrl+F11)
```

Flash should be under 160 KB total (model + firmware + HAL).

---

## Step 8: Read Latency Output

Connect the UART (115200 8N1) via USB-UART or STLink VCP:

```
=== DSC-AMCNet Inference on STM32H723ZG ===
    Core: Cortex-M7 @ 544 MHz
    Model: DSC-AMCNet INT8 (X-CUBE-AI 10.2.1)

[AI] DSC-AMCNet INT8 initialised.
[AI] weights=5804 B (flash) | activations=4488 B | io=1024+44 B (ram)

[BENCH] Running 1000 inferences (10 warm-up discarded) ...
[DWT] N=1000  mean=1.183 ms  min=1.181 ms  max=1.187 ms

[RESULT] DSC-AMCNet INT8 Latency
         Mean : 1.183 ms  (paper: 1.183 ms)  ✓
```

The `weights=` / `activations=` line prints
`AI_NETWORK_DATA_WEIGHTS_SIZE` / `AI_NETWORK_DATA_ACTIVATIONS_SIZE` from the
generated `network_data_params.h`, so it is the authoritative check that the
`.onnx` in the project is the one described above.

---

## Step 9: Record Latency with Python Logger

Use `hardware_characterization/measure_latency.py` to log results to file:

```bash
python hardware_characterization/measure_latency.py \
    --port COM7 \
    --baud 115200 \
    --output results/latency_log.csv
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Random classification output | Wrong ONNX backend (qnnpack) | Re-export with fbgemm |
| Flash overflow | Wrong ONNX (FP32 instead of INT8) | Re-run ptq_quantize.py |
| Latency >> 1.183 ms | Clock not 544 MHz | Check SystemClock_Config (M=2, N=136, P=1) |
| DWT reads 0 | CoreDebug TRCENA not set | Ensure DWT_Init() called first |
| No UART output | Printf not redirected | Add `__io_putchar` to syscalls.c |
