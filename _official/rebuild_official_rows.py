"""Rebuild the official 220k dataset exactly from the CSV rows (I channel).

The CSV payload IS the official I channel (rounded in the export).  For each
(mod, snr) cell we locate the 1000 rows that match the official pickle, in the
official pickle's own row order, and verify the reconstruction numerically.
"""
import os
import pickle
import sys

import numpy as np

OFF = r'_official\radio\RML2016.10a_dict_optimized.pkl'
CSV_DIR = r'C:\Users\Hao\Desktop\RadioML2016.10a'
MODS = ["8PSK", "AM-DSB", "AM-SSB", "BPSK", "CPFSK",
        "GFSK", "PAM4", "QAM16", "QAM64", "QPSK", "WBFM"]
SNRS = list(range(-20, 20, 2))


def load_csv(mod):
    flat = np.array(
        [float(t) for tok in open(os.path.join(CSV_DIR, mod + '.csv'), 'rb')
         .read().split() for t in tok.split(b',')], dtype=np.float32)
    table = flat.reshape(-1, 129)
    return table[:, :128], table[:, 128]


def match_rows(official_rows, csv_rows, decimals=5):
    """Return csv indices matching each official row (dict on quantised keys)."""
    key = np.round(csv_rows.astype(np.float64), decimals)
    store = {}
    for i, row in enumerate(key):
        store.setdefault(tuple(row.tolist()), []).append(i)
    out = []
    for row in official_rows:
        k = tuple(np.round(row.astype(np.float64), decimals).tolist())
        if k not in store or not store[k]:
            return None
        out.append(store[k].pop())
    return out


def main():
    with open(OFF, 'rb') as f:
        off = pickle.load(f, encoding='latin1')

    print('cell                     exact_match  max_abs_diff  rows_found')
    worst = 0.0
    total_ok = 0
    for mod in MODS:
        pay, snr = load_csv(mod)
        for s in SNRS:
            o = off[(mod, s)][:, 0, :]
            c = pay[snr == np.float32(s)]
            idx = match_rows(o, c)
            if idx is None:
                print(f'  {mod:<7} {s:+3d} dB  NO MATCH')
                continue
            rec = c[idx]
            diff = float(np.abs(rec - o).max())
            worst = max(worst, diff)
            total_ok += 1
            if s in (-20, 10):
                print(f'  {mod:<7} {s:+3d} dB  {rec.shape}  {diff:.3e}  '
                      f'{len(set(idx))}/1000')
    print(f'\ncells reconstructed: {total_ok}/220   worst |diff| = {worst:.3e}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
