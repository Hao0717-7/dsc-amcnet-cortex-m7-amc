"""Inspect the official pickle key order, cell shapes, and Q-channel structure.

Also checks whether matching official I rows back into the CSV by their I values
is unambiguous (needed to recover the paired Q samples).
"""
import os
import pickle
from collections import Counter

import numpy as np

OFF = r'_official\radio\RML2016.10a_dict_optimized.pkl'
CSV_DIR = r'C:\Users\Hao\Desktop\RadioML2016.10a'
MODS = ["8PSK", "AM-DSB", "AM-SSB", "BPSK", "CPFSK",
        "GFSK", "PAM4", "QAM16", "QAM64", "QPSK", "WBFM"]
SNRS = list(range(-20, 20, 2))


def main():
    with open(OFF, 'rb') as f:
        off = pickle.load(f, encoding='latin1')
    keys = list(off.keys())
    print('official key order (idx: mod/snr):')
    for i in range(0, 220, 5):
        chunk = '  '.join(f'{k[0]}/{k[1]:+d}' for k in keys[i:i + 5])
        print(f'  {i:3d}: {chunk}')

    print('\ndistinct (mod,snr) keys:', len(set(keys)))
    modseq = [k[0] for k in keys]
    print('modulation run-length structure (first 40 runs):')
    runs = []
    for m in modseq:
        if runs and runs[-1][0] == m:
            runs[-1][1] += 1
        else:
            runs.append([m, 1])
    print('  ', ' '.join(f'{m}x{n}' for m, n in runs[:40]))
    print('  total runs:', len(runs))

    # is SNR contiguous / grouped per modulation?
    for m in MODS:
        idx = [i for i, k in enumerate(keys) if k[0] == m]
        print(f'  {m:<7} at positions {idx[0]:3d}..{idx[-1]:3d}  '
              f'contiguous={idx == list(range(idx[0], idx[-1] + 1))}')

    # Uniqueness of I rows inside each CSV cell (for Q recovery).
    print('\nI-row uniqueness within CSV cells (quantised to 5 decimals):')
    for mod in ['8PSK', 'QPSK', 'WBFM']:
        flat = np.array(
            [float(t) for tok in open(os.path.join(CSV_DIR, mod + '.csv'), 'rb')
             .read().split() for t in tok.split(b',')], dtype=np.float32)
        table = flat.reshape(-1, 129)
        pay = table[:, :128]
        snr = table[:, 128]
        for s in (-20, 10):
            c = pay[snr == np.float32(s)]
            key = np.round(c.astype(np.float64), 5)
            seen = Counter(map(tuple, key.tolist()))
            dup = sum(v - 1 for v in seen.values() if v > 1)
            print(f'  {mod:<7} {s:+3d} dB: {len(c)} rows, {len(seen)} distinct '
                  f'quantised I rows, duplicates={dup}')

    # Q magnitude vs I
    print('\nofficial |Q| vs |I| statistics (8PSK):')
    for s in (-20, 0, 10, 18):
        a = off[('8PSK', s)]
        print(f'  {s:+3d} dB: |I| mean {np.abs(a[:,0,:]).mean():.5f}  '
              f'|Q| mean {np.abs(a[:,1,:]).mean():.5f}  '
              f'corr(I,Q) {np.corrcoef(a[:,0,:].ravel(), a[:,1,:].ravel())[0,1]:+.4f}  '
              f'mean|I+jQ| {np.abs(a[:,0,:]+1j*a[:,1,:]).mean():.5f}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
