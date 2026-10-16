# Datasets

## RML2016.10a

**Source:** DeepSig Open Data  
**URL:** https://www.deepsig.ai/datasets  
**File:** `RML2016.10a_dict.dat` (pickled Python dict, ~215 MB for the
1,000-per-cell build used here)

### Built from the per-modulation CSV export

This copy was **not** downloaded as the original pickle. It was rebuilt from a
per-modulation CSV export (`<MOD>.csv`, 11 files) with
[`build_rml2016_from_csv.py`](build_rml2016_from_csv.py):

```bash
python datasets/build_rml2016_from_csv.py \
    --src "C:/Users/Hao/Desktop/RadioML2016.10a" \
    --out datasets
```

| Artifact | Path | Shape | Count |
|---|---|---|---|
| Canonical | `datasets/RML2016.10a_dict.dat` | `{(mod, snr): (1000, 2, 128) float32}` | 220,000 |
| Full source | `datasets/full_2k/RML2016.10a_2k_dict.dat` | `{(mod, snr): (2000, 2, 128) float32}` | 440,000 |
| Selection audit | `datasets/full_2k/selection_manifest.json` | source row indices for the canonical 1,000/cell | — |

### Source CSV layout

Each `<MOD>.csv` is 40,000 rows × 129 columns (comma separated):

* columns `0..127` — the sample payload, 128 float values;
* column `128` — SNR label in dB, one of −20, −18, …, +18 (2,000 rows per SNR,
  20 SNR values, 11 modulations → 440,000 rows total).

**Open item — the payload holds only one real component.** Statistical
inspection of the payload shows it behaves like *one real time series per row*,
not like an interleaved I/Q pair: at +18 dB the lag-1 autocorrelation is
0.96–1.00 for every modulation and collapses to ≈0.02 at −20 dB, and even/odd
columns are 0.96 correlated at high SNR, which is impossible for the I and Q
halves of a PSK/QAM window. The export therefore carries 128 values where the
canonical file needs 128 (I) + 128 (Q).

Until the export convention is confirmed, the build fills the (2, 128) window as
**I = payload, Q = 0** (`--q-mode zero`, the default). `--q-mode copy` and
`--q-mode hilbert` exist for re-generating the canonical file if the payload
turns out to be one channel of a complex window (Hilbert gives the analytic
quadrature). Only the canonical 1,000/cell file changes with that flag; the
440,000-row source is the same rows either way, so re-running is cheap.

### Sampling rule for the canonical file

The 1,000 samples per `(mod, snr)` cell are drawn **uniformly at random**
(without replacement, `SEED = 42`) from the 2,000 available in that cell — not
the first half. `full_2k/selection_manifest.json` records the exact source row
indices, so the subset can be audited or rebuilt.

The locked `split_indices.npz` still applies: its `train_idx`/`test_idx` count
176,000 / 44,000 and index valid cells in the canonical (mod, snr) ordering.
Note that this rebuild draws a different 1,000/cell subset than the original
DeepSig pickle, so per-sample accuracies can differ slightly from the paper;
the class/SNR structure is identical.

### Contents

| Property | Value |
|---|---|
| Modulation classes | 11 (8PSK, AM-DSB, AM-SSB, BPSK, CPFSK, GFSK, PAM4, QAM16, QAM64, QPSK, WBFM) |
| SNR range | −20 to +18 dB (step 2 dB) |
| Samples per class/SNR | 1,000 |
| Total samples | 220,000 |
| Sample format | Complex IQ, 128 samples → shape (2, 128) float32 |

### Acquisition

1. Download `RML2016.10a_dict.dat` from [DeepSig](https://www.deepsig.ai/datasets).
2. Place it in this directory:
   ```
   datasets/RML2016.10a_dict.dat
   ```
3. Verify:
   ```bash
   python datasets/prepare_dataset.py \
       datasets/RML2016.10a_dict.dat \
       datasets/split_indices.npz
   ```
   Expected output:
   ```
   Total samples : 220,000
   Unique classes: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
   SNR range     : -20 to 18 dB
   Locked split (80/20 stratified, SEED = 42):
     Train : 176,000
     Test  :  44,000
   ```
   (Validation during training is carved from the train indices only;
   the locked test set is never used for model selection.)

## Dataset Split (split_indices.npz)

The train/val/test split is **locked** and archived on Zenodo:

> **DOI:** https://doi.org/10.5281/zenodo.20579317

Download `split_indices.npz` from Zenodo and place it in this directory.

The split was generated with a fixed random seed (seed = 42) and stratified
sampling to ensure equal class/SNR representation across splits. It must not
be regenerated — doing so will produce different accuracy numbers than those
reported in the paper.

### Why is the split locked?

Without a fixed split, two researchers training the same model may evaluate
on different samples, making results incomparable. Archiving the exact index
arrays on Zenodo allows anyone to independently verify all claims in the paper
using precisely the test set the authors used.

## RML2018.01a (cross-dataset generalization, paper Section X)

RML2018.01a (24 modulation classes, 1,024-sample windows) **is used** for
cross-dataset generalization: DSC-AMCNet-24 and LiteAMCNet-24 are trained
from scratch on it (no transfer learning), changing only the output layer
and, for DSC-AMCNet-24, adding a fourth MaxPool1d stage (paper Table IX:
+0.01 pp PTQ degradation at 24 classes; 48.28 KB Flash / 31.70 KB RAM).
Download it separately from DeepSig; note its much larger size (~20 GB).
