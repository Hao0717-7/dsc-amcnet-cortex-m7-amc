"""
build_rml2016_from_csv.py
=========================
Build RML2016.10a-format pickle files from the per-modulation CSV export.

Source layout (verified against the actual files, one CSV per modulation)
-------------------------------------------------------------------------
    <MOD>.csv : 40,000 rows x 129 columns, comma separated

      columns 0..127  -> 128 float values, the sample payload
      column  128     -> SNR label in dB, one of -20,-18,...,+18

    Each modulation therefore holds 20 SNR cells x 2,000 rows = 40,000 rows,
    and all 11 modulations hold 11 x 20 x 2,000 = 440,000 rows.

Target layout (datasets/README.md)
----------------------------------
    {(mod, snr): ndarray (N, 2, 128) float32}, row 0 = I, row 1 = Q,
    canonical N = 1,000 per cell (220,000 samples), full N = 2,000 per cell
    (440,000 samples).

Open question -- how the 128 payload values map onto (2, 128)
-------------------------------------------------------------
The payload is 128 values, not the 128(I) + 128(Q) = 256 values that the
canonical file holds, so the quadrature half is not present in the export.
Statistical inspection of the payload (see datasets/README.md, section
"Source CSV layout") shows it behaves like ONE real time series per row:

    * at +18 dB the lag-1 autocorrelation is 0.96...1.00 for every modulation
      and collapses to ~0.02 at -20 dB, i.e. the payload is a smooth
      oversampled waveform plus noise, not an interleaved I/Q pair
      (an interleaved reading would need even/odd columns to be uncorrelated
      for PSK/QAM at high SNR);
    * the payload spectrum is flat out to Nyquist, the signature of a single
      sampled channel rather than a half-empty I/Q pair.

Because the Q component cannot be recovered from the export, `--q-mode`
selects how row 1 of the (2, 128) window is filled. The default, `zero`,
never invents data: the payload becomes the I channel and Q stays 0. The
canonical file can be regenerated at any time with another mode once the
export convention is confirmed, so picking a mode now is not destructive.

Outputs
-------
1. datasets/RML2016.10a_dict.dat
   pickled dict {(mod, snr): ndarray (1000, 2, 128) float32}
   11 classes x 20 SNR x 1000 = 220,000 samples.
   The 1,000 samples per (mod, snr) cell are drawn UNIFORMLY AT RANDOM
   (without replacement) from the 2,000 available, with a fixed seed, so the
   subset is reproducible and is not the trivial "first half".

2. datasets/full_2k/RML2016.10a_2k_dict.dat
   pickled dict {(mod, snr): ndarray (2000, 2, 128) float32}
   11 x 20 x 2000 = 440,000 samples -- the complete source, nothing dropped.

3. datasets/full_2k/selection_manifest.json
   the exact source row indices selected for the canonical file, so the
   220,000-sample subset can be reconstructed/audited later.

Usage
-----
    python datasets/build_rml2016_from_csv.py \
        --src "C:/Users/Hao/Desktop/RadioML2016.10a" \
        --out datasets
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

# Canonical ordering used by models/train.py (sorted by (mod, snr)) and by
# datasets/prepare_dataset.py (MODULATIONS list).
MODULATIONS = [
    "8PSK", "AM-DSB", "AM-SSB", "BPSK", "CPFSK",
    "GFSK", "PAM4", "QAM16", "QAM64", "QPSK", "WBFM",
]

SNR_VALUES = list(range(-20, 20, 2))        # -20, -18, ..., +18 dB
SAMPLES_PER_CELL_FULL = 2000                # source rows per (mod, snr) cell
SAMPLES_PER_CELL_CANON = 1000               # canonical RML2016.10a cell size
SEED = 42

WINDOW = 128                                # time samples per window
N_PAYLOAD_COLS = WINDOW                     # columns 0..127 payload
N_COLS = N_PAYLOAD_COLS + 1                 # + 1 SNR column = 129

Q_MODES = ("zero", "copy", "hilbert")


def read_csv(path: Path) -> np.ndarray:
    """Read a 129-column CSV into a ((N, 129)) float32 array.

    A vectorised split + list comprehension over ~5.2M tokens per file is
    several times faster than np.loadtxt and, unlike np.fromstring with
    sep=',', it actually parses this export to the end.
    """
    text = path.read_bytes().split()
    flat = np.array([float(t) for tok in text for t in tok.split(b",")],
                    dtype=np.float32)
    if flat.size % N_COLS != 0:
        raise ValueError(
            f"{path.name}: {flat.size} values is not a multiple of {N_COLS} "
            f"columns (remainder {flat.size % N_COLS})"
        )
    return flat.reshape(-1, N_COLS)


def parse_csv(path: Path, q_mode: str = "zero") -> tuple[np.ndarray, np.ndarray]:
    """
    Load one modulation CSV.

    Returns
    -------
    iq  : np.ndarray, shape (N, 2, WINDOW), float32
    snr : np.ndarray, shape (N,), float32
    """
    table = read_csv(path)
    payload = table[:, :N_PAYLOAD_COLS]
    snr = table[:, N_PAYLOAD_COLS]

    # Last column must look like an SNR label on the canonical grid.
    bad = ~np.isin(snr, np.asarray(SNR_VALUES, dtype=np.float32))
    if bad.any():
        vals = np.unique(snr[bad])[:5]
        raise ValueError(
            f"{path.name}: column {N_PAYLOAD_COLS} is not an SNR label; "
            f"unexpected values {vals.tolist()}"
        )

    iq = np.empty((payload.shape[0], 2, WINDOW), dtype=np.float32)
    iq[:, 0, :] = payload
    if q_mode == "zero":
        iq[:, 1, :] = 0.0
    elif q_mode == "copy":
        iq[:, 1, :] = payload
    elif q_mode == "hilbert":
        # Analytic-signal quadrature: Q = Hilbert transform of I.
        spectrum = np.fft.fft(payload.astype(np.float64), axis=1)
        h = np.zeros(WINDOW, dtype=np.float64)
        h[0] = 1.0
        h[1:WINDOW // 2] = 2.0
        h[WINDOW // 2] = 1.0
        iq[:, 1, :] = np.imag(np.fft.ifft(spectrum * h, axis=1)).astype(np.float32)
    else:  # pragma: no cover - argparse restricts the choices
        raise ValueError(f"unknown q_mode {q_mode!r}")
    return iq, snr


def cells_for(iq: np.ndarray, snr: np.ndarray, name: str) -> dict[int, np.ndarray]:
    """Split one modulation's rows into per-SNR cells of 2,000 rows each."""
    cells: dict[int, np.ndarray] = {}
    for s in SNR_VALUES:
        idx = np.flatnonzero(snr == np.float32(s))
        if len(idx) != SAMPLES_PER_CELL_FULL:
            raise ValueError(
                f"{name} @ {s} dB: expected {SAMPLES_PER_CELL_FULL} rows, "
                f"found {len(idx)}"
            )
        cells[s] = iq[idx]
    return cells


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="folder holding <MOD>.csv")
    ap.add_argument("--out", default="datasets", help="output datasets folder")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--q-mode", choices=Q_MODES, default="zero",
                    help="how to fill the Q channel from a payload that "
                         "contains only one real component (default: zero)")
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    full_dir = out / "full_2k"
    full_dir.mkdir(parents=True, exist_ok=True)

    if not src.is_dir():
        print(f"ERROR: source folder not found: {src}", file=sys.stderr)
        return 1

    rng = np.random.default_rng(args.seed)

    full: dict[tuple[str, int], np.ndarray] = {}
    canon: dict[tuple[str, int], np.ndarray] = {}
    manifest: dict[str, list[int]] = {}

    t0 = time.time()
    for mod in MODULATIONS:
        csv_path = src / f"{mod}.csv"
        if not csv_path.is_file():
            print(f"ERROR: missing {csv_path}", file=sys.stderr)
            return 1

        t = time.time()
        iq, snr = parse_csv(csv_path, q_mode=args.q_mode)
        cells = cells_for(iq, snr, mod)

        kept_rows = np.empty((len(SNR_VALUES), SAMPLES_PER_CELL_CANON),
                             dtype=np.int64)
        for ci, s in enumerate(SNR_VALUES):
            cell = cells[s]                                   # (2000, 2, 128)
            # Random (not positional) draw of 1,000 of the 2,000 rows.
            picked = np.sort(rng.choice(SAMPLES_PER_CELL_FULL,
                                        size=SAMPLES_PER_CELL_CANON,
                                        replace=False))
            kept_rows[ci] = picked
            canon[(mod, s)] = cell[picked]
            full[(mod, s)] = cell

        manifest[mod] = kept_rows.reshape(-1).tolist()
        print(f"  {mod:<7} rows={iq.shape[0]:,}  cells=20x{SAMPLES_PER_CELL_FULL}  "
              f"({time.time() - t:.1f}s)", flush=True)

    # ---- integrity checks before writing -----------------------------------
    n_full = sum(v.shape[0] for v in full.values())
    n_canon = sum(v.shape[0] for v in canon.values())
    assert len(full) == 11 * 20 and len(canon) == 11 * 20
    assert n_full == 440_000, f"full sample count {n_full} != 440,000"
    assert n_canon == 220_000, f"canonical sample count {n_canon} != 220,000"
    for k, v in canon.items():
        assert v.shape == (SAMPLES_PER_CELL_CANON, 2, WINDOW), (k, v.shape)
        assert v.dtype == np.float32, (k, v.dtype)
    print(f"\nIntegrity OK: full={n_full:,}  canonical={n_canon:,}")

    # ---- write -------------------------------------------------------------
    canon_path = out / "RML2016.10a_dict.dat"
    full_path = full_dir / "RML2016.10a_2k_dict.dat"

    for path, obj in ((full_path, full), (canon_path, canon)):
        t = time.time()
        with open(path, "wb") as f:
            pickle.dump(obj, f, protocol=4)
        print(f"wrote {path}  ({path.stat().st_size / 1e6:.1f} MB, "
              f"{time.time() - t:.1f}s)")

    man_path = full_dir / "selection_manifest.json"
    with open(man_path, "w", encoding="utf-8") as f:
        json.dump({
            "note": "Source row index (within each modulation CSV) selected "
                    "for datasets/RML2016.10a_dict.dat. Rows are grouped per "
                    "SNR cell in SNR_VALUES order, 1000 indices per cell.",
            "seed": args.seed,
            "q_mode": args.q_mode,
            "source_csv_columns": {
                "payload": "0..127",
                "snr": 128,
            },
            "modulations": MODULATIONS,
            "snr_values": SNR_VALUES,
            "rows_per_cell": SAMPLES_PER_CELL_FULL,
            "selected_per_cell": SAMPLES_PER_CELL_CANON,
            "selected_rows": manifest,
        }, f)
    print(f"wrote {man_path}")

    print(f"\nTotal elapsed {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
