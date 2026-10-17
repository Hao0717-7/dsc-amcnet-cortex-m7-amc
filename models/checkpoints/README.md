# Pre-trained Checkpoints

Pre-trained model weights and the locked dataset split are archived on Zenodo
to ensure long-term, citable availability independent of this repository.

## Download

**Zenodo Archive:** https://doi.org/10.5281/zenodo.20579317

Download and place files as follows:

```
models/checkpoints/
├── dscamcnet_best.pt          ← DSC-AMCNet FP32 weights (4,961 params)
├── dscamcnet_int8.onnx        ← INT8 ONNX (QDQ, signed INT8; opset 13, IR 8)
└── ulcnn_fp32_fixed.onnx      ← ULCNN-simplified baseline (FP32 ONNX)
datasets/
└── split_indices.npz          ← Locked 80/20 train/test split
```

## Integrity Verification

After downloading, verify SHA-256 checksums:

| File | SHA-256 |
|------|---------|
| `dscamcnet_best.pt` | `4827d8dc549866390fe22914c6ba3f6be03e4330dc9e0f0aa750013e8160a06e` |
| `dscamcnet_int8.onnx` | `6d3520a3081aa07d47713b4253d30a09f07e754fe8db0e5ddc9e7327ce5e78dd` |
| `ulcnn_fp32_fixed.onnx` | `7c4936039fb0df17e7fa15322a9d5c4cd81847cea5fa7b54a39bd79e5b22a8e9` |
| `split_indices.npz` | `69c8ed46e301f374f0beecb3919495a55e7fff73bebbad7e156a7f97d2e8d61b` |

Verify on Linux/macOS:
```bash
sha256sum models/checkpoints/dscamcnet_best.pt
sha256sum datasets/split_indices.npz
```

Verify on Windows (PowerShell):
```powershell
Get-FileHash models\checkpoints\dscamcnet_best.pt -Algorithm SHA256
Get-FileHash datasets\split_indices.npz -Algorithm SHA256
```

## Notes

- The `split_indices.npz` is **locked** — it defines the exact 80/20 train/test partition
  used in all paper experiments. Using a different split will produce different accuracy
  numbers. This is expected.
- The INT8 ONNX carries signed INT8 (QDQ; zero-point 0). Unsigned INT8
  (qnnpack-style, zero-point 128) is incompatible with X-CUBE-AI and fails
  silently on device. See `quantization/validate_backend.py`.

## Local provenance (2026-10-16) — read before trusting the Zenodo weights

The published `dscamcnet_best.pt` (SHA-256 `4827d8dc…`, 37,397 B) could **not**
be reproduced against either available copy of the dataset:

| Evaluation data | Official Zenodo checkpoint | Locally retrained checkpoint |
|---|---|---|
| CSV-derived canonical file (Q = 0) | 15.35% @ +10 dB | 73.86% @ +10 dB |
| Genuine DeepSig pickle, locked split | 26.05% @ +10 dB | **89.73% @ +10 dB** |

11-class chance level is ~9.1%, so the published checkpoint behaves as if it
was trained on a differently ordered/aligned array.

The checkpoint now in this directory was retrained from scratch with
`models/train.py` on the genuine DeepSig I/Q pickle and the archived
`split_indices.npz`:

- `dscamcnet_best.pt` — 36,437 B, SHA-256
  `F7A05C25672A0074CFF30673C87ACB097897FA96662264CB57F9A959A91BAAB6`,
  **+10 dB 89.73% / all-SNR 59.56%** on the locked test split (2,219 samples
  in the +10 dB bin), best validation epoch 87 of 100, 4,961 parameters as in
  the paper.
- `quantization/outputs/dscamcnet_int8.onnx` — 27,989 B, +10 dB 89.36%
  (PTQ degradation 0.37 pp). The Zenodo artifact of the same name (58,169 B,
  SHA-256 `6d3520a3…`) does not match this file either.
- `dscamcnet_best.prev-89.00.pt` — earlier independent run of the same
  protocol (+10 dB 89.00%, all-SNR 59.36%). The 0.73 pp gap to
  `dscamcnet_best.pt` is ordinary run-to-run scatter, not a protocol change:
  `_official/train_fresh.py` did not seed the PyTorch RNG when these two runs
  were made, so each started from a different random initialisation and saw a
  different batch order. Independently, the +10 dB bin holds only 2,219
  samples (one sample = 0.045 pp, binomial 1σ ≈ 0.6 pp) and the two snapshots
  sit at different best-validation epochs (98 vs 87), where the accuracy curve
  already plateaus around 88–90%.
- The old CSV-era baseline (74.99%, trained on the CSV-derived data with
  Q = 0) has been deleted — it must not be deployed.

The CSV export on the Desktop (`RadioML2016.10a/*.csv`, 129 fields per row)
contains the I channel only; the Q channel does not exist in it. See the root
`CHANGELOG.txt` section "数据 provenance 更正与重训" for the full evidence chain.

