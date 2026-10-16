"""Statistical comparison: official RadioML pickle vs local CSV-derived data."""
import os
import pickle

import numpy as np

OFF = r'_official\radio\RML2016.10a_dict_optimized.pkl'
CSV = os.path.join(r'C:\Users\Hao\Desktop\RadioML2016.10a',
                   '8PSK.csv').replace('8PSK', '{M}')
FULL = r'datasets\full_2k\RML2016.10a_2k_dict.dat'


def stats(name, a):
    a = np.asarray(a, dtype=np.float64)
    print(f'  {name:38s} shape={a.shape} '
          f'min={a.min():+.4f} max={a.max():+.4f} '
          f'mean={a.mean():+.4f} std={a.std():.4f} '
          f'rms={np.sqrt((a**2).mean()):.4f} absmax={np.abs(a).max():.4f}')


def csv_cell(mod, s):
    flat = np.array(
        [float(t) for tok in open(CSV.replace('{M}', mod), 'rb')
         .read().split() for t in tok.split(b',')], dtype=np.float32)
    table = flat.reshape(-1, 129)
    payload = table[:, :128]
    snr = table[:, 128]
    return payload[np.flatnonzero(snr == np.float32(s))]


def main():
    with open(OFF, 'rb') as f:
        off = pickle.load(f, encoding='latin1')
    with open(FULL, 'rb') as f:
        full = pickle.load(f, encoding='latin1')

    print('== official pickle (220k) ==')
    for m, s in [('8PSK', -20), ('8PSK', 0), ('8PSK', 10), ('8PSK', 18)]:
        stats(f'official {m} {s:+d}dB I', off[(m, s)][:, 0, :])
    print('== official pickle Q channel ==')
    for m, s in [('8PSK', -20), ('8PSK', 10)]:
        stats(f'official {m} {s:+d}dB Q', off[(m, s)][:, 1, :])

    print('\n== CSV export payload (2000 rows/cell) ==')
    for m, s in [('8PSK', -20), ('8PSK', 0), ('8PSK', 10), ('8PSK', 18)]:
        stats(f'csv {m} {s:+d}dB payload', csv_cell(m, s)[:200])

    print('\n== local 2k rebuild ==')
    for m, s in [('8PSK', -20), ('8PSK', 10)]:
        stats(f'local2k {m} {s:+d}dB I', full[(m, s)][:, 0, :])
        stats(f'local2k {m} {s:+d}dB Q', full[(m, s)][:, 1, :])

    # correlation between official and CSV at the same cell (pooled row-major)
    print('\n== pooled correlation official vs CSV payload, 8PSK ==')
    for s in [-20, 0, 10, 18]:
        o = off[('8PSK', s)][:, 0, :].ravel()
        c = csv_cell('8PSK', s)[:1000].ravel()
        print(f'  {s:+3d} dB: corr(flat 128000) = {np.corrcoef(o, c)[0,1]:+.4f}')

    # are the official 1000 rows a subset of the 2000 CSV rows, up to float
    # rounding?  compare sorted row l2 norms
    print('\n== row norm distribution (sorted), 8PSK +10 dB ==')
    on = np.sort(np.linalg.norm(off[('8PSK', 10)][:, 0, :], axis=1))
    cn = np.sort(np.linalg.norm(csv_cell('8PSK', 10)[:, :], axis=1))
    print('  official n=1000:', np.array2string(on[[0, 250, 500, 750, 999]],
                                                precision=4))
    print('  csv      n=2000:', np.array2string(cn[[0, 500, 1000, 1500, 1999]],
                                                precision=4))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
