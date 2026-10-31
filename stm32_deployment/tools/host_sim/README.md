# Host-side bit-parity harness for the custom INT8 engine

Runs the **production** engine source (`stm32_deployment/custom_engine/dsc_infer.c`)
on a PC and compares its 11 q-domain logits with the shipped QDQ ONNX under
onnxruntime, window by window, over the whole locked test split.

Why: the engine's quantized arithmetic is plain integer C. Only three things are
MCU-specific — the `SMLAD`/`SMULL` inline asm and the `main.h`/`test_sample.h`
firmware headers. Compiling with `-DDSC_HOST_SIM` swaps just those for portable
stand-ins (same two-lane 16-bit MAC math), so a PC can validate a change in
seconds before spending a build/flash/bench cycle on the board.

## Usage

```bash
pip install ziglang            # any C compiler works; zig is self-contained
python stm32_deployment/tools/host_sim/compare_engine.py            # all 44,000 windows
python stm32_deployment/tools/host_sim/compare_engine.py --limit 4000
python stm32_deployment/tools/host_sim/compare_engine.py --engine path/to/engine.exe
```

The script builds `.tmp/host_sim/engine.exe` when needed (`--engine` skips that).

## Reference numbers (v1.0.5 engine, 44,000 locked windows)

| metric | host harness | board (`p5_accuracy_44k.json`) |
|---|---|---|
| windows with all 11 logits bit-equal to onnxruntime | **98.775%** | 98.80% |
| argmax agreement | **99.9795%** | 99.98% |
| engine accuracy | **59.2045%** | 59.2045% |
| +10 dB accuracy | **89.3646%** | 89.36% |
| max abs logit difference (q units) | 1 | 1 |

The tiny `bit-exact` gap matches the board exactly, i.e. the host build is a
faithful stand-in for the MCU engine. Runtime: ~0.7 s for the engine on 44k
windows, ~3 s for onnxruntime.

## Notes

- `-DDSC_HOST_SIM` only guards: the two firmware includes, `dsc_fill_sample()`
  (needs `test_sample.h`), and the two inline-asm helpers. The target build
  (`DSC_HOST_SIM` undefined) is byte-identical in behaviour: the Keil build
  still reports `Code=60844` and 0.905 / 0.914 ms.
- Any numeric change to the engine should keep the three "agreement" rows at or
  above the reference values above; the board run (`golden_test.py` 22/22,
  `p0_replay.py --n 300`) is still the final word.
