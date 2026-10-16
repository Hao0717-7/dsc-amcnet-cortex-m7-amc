"""Compare the official RadioML 2016.10a pickle against the local CSV rebuild.

Answers three questions:
  1. Which keys does the official split_indices.npz use?
  2. Is the official dataset's per-cell sample content the same as the CSV
     export's rows (exact float32 hash match), and how much do they overlap?
  3. Which flat ordering makes the locked split consistent (class-stratified)?
"""
import hashlib
import os
import pickle
import sys

import numpy as np

OFF_PKL = r'_official\radio\RML2016.10a_dict_optimized.pkl'
CSV_DIR = r'C:\Users\Hao\Desktop\RadioML2016.10a'
FULL_DAT = r'datasets\full_2k\RML2016.10a_2k_dict.dat'

MODS = ["8PSK", "AM-DSB", "AM-SSB", "BPSK", "CPFSK",
        "GFSK", "PAM4", "QAM16", "QAM64", "QPSK", "WBFM"]
SNRS = list(range(-20, 20, 2))


def row_hashes(a):
    """frozenset-friendly set of md5 hashes of each (2,128) row."""
    b = np.ascontiguousarray(a, dtype=np.float32).tobytes()
    row = 2 * 128 * 4
    return [hashlib.md5(b[i:i + row]).hexdigest() for i in range(0, len(b), row)]


def load_official():
    with open(OFF_PKL, 'rb') as f:
        return pickle.load(f, encoding='latin1')


def load_csv_cells():
    """Rebuild the (mod, snr) -> (2000, 2, 128) cells straight from the CSVs."""
    cells = {}
    for mod in MODS:
        flat = np.array(
            [float(t) for tok in open(os.path.join(CSV_DIR, mod + '.csv'), 'rb')
             .read().split() for t in tok.split(b',')], dtype=np.float32)
        table = flat.reshape(-1, 129)
        payload = table[:, :128]
        snr = table[:, 128]
        iq = np.zeros((len(payload), 2, 128), dtype=np.float32)
        iq[:, 0, :] = payload
        for s in SNRS:
            idx = np.flatnonzero(snr == np.float32(s))
            cells[(mod, s)] = iq[idx]
    return cells


def main():
    z = np.load(r'datasets\split_indices.npz')
    print('local split keys :', list(z.keys()))
    z2 = np.load(r'_official\split_indices.npz')
    print('zenodo split keys:', list(z2.keys()))
    for k in z2.keys():
        print(f'   {k}: shape {z2[k].shape} dtype {z2[k].dtype}')

    off = load_official()
    print('\nCSV cell hashes vs official (per modulation x SNR):')
    csv_cells = load_csv_cells()
    total_off = total_hit = 0
    for mod in MODS:
        hits = []
        for s in SNRS:
            oh = set(row_hashes(off[(mod, s)]))
            ch = row_hashes(csv_cells[(mod, s)])
            n_hit = sum(1 for h in ch if h in oh)
            hits.append(n_hit)
            total_off += len(oh)
            total_hit += n_hit
        print(f'  {mod:<7} official rows matched in CSV 2000: '
              f'min={min(hits):4d} max={max(hits):4d} mean={np.mean(hits):6.1f}')
    print(f'\n  overall CSV rows whose exact bits appear in official: '
          f'{total_hit:,} / {20*11*2000:,} = {100*total_hit/(20*11*2000):.2f}%')

    # exact-order test: are the official rows the FIRST k CSV rows?
    print('\nCSV-vs-official same-position exact match (first 1000 rows):')
    for mod in ['8PSK', 'WBFM', 'QPSK']:
        for s in [-20, 0, 10, 18]:
            a = off[(mod, s)]
            b = csv_cells[(mod, s)][:1000]
            eq = int((a == b).all(axis=(1, 2)).sum())
            print(f'  {mod:<7} {s:+3d} dB: identical rows at same position '
                  f'{eq:4d}/1000')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
