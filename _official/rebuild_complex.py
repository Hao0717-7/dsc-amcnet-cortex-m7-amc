"""Rebuild RML2016.10a with the missing Q channel recovered from DeepSig's file.

The per-modulation CSV export in `C:\\Users\\Hao\\Desktop\\RadioML2016.10a` contains
only the in-phase channel of the official DeepSig release: each row holds the
128 I samples of one window plus the SNR in the last column.  The official
`RML2016.10a_dict_optimized.pkl` (downloaded from DeepSig) holds the matching
complex windows.  Every official I-channel row matches exactly one CSV row
(rounded to 5 decimals), so the paired Q row can be pulled back out and the
full complex (2, 128) window reconstructed without any loss.

Layout and ordering are identical to the CSV-built files, so the locked
`split_indices.npz` indexes this file exactly as it indexes the old ones.
"""
import argparse
import json
import os
import pickle
import time

import numpy as np

CSV_DIR = r'C:\Users\Hao\Desktop\RadioML2016.10a'
OFFICIAL = r'_official\radio\RML2016.10a_dict_optimized.pkl'
MODS = ["8PSK", "AM-DSB", "AM-SSB", "BPSK", "CPFSK",
        "GFSK", "PAM4", "QAM16", "QAM64", "QPSK", "WBFM"]
SNRS = list(range(-20, 20, 2))
WINDOW = 128
N_COLS = 129


def read_csv(mod: str) -> np.ndarray:
    """Return (n_rows, 129) float32: 128 payload columns + SNR."""
    path = os.path.join(CSV_DIR, mod + '.csv')
    flat = np.array([float(t) for tok in open(path, 'rb').read().split()
                     for t in tok.split(b',')], dtype=np.float32)
    return flat.reshape(-1, N_COLS)


def build(mods, out_path, manifest_path=None, q_mode='recover'):
    t0 = time.time()
    with open(OFFICIAL, 'rb') as f:
        off = pickle.load(f, encoding='latin1')
    print(f'official pickle: {len(off)} cells ({time.time()-t0:.0f}s)')

    data = {}
    manifest = {'seed': None, 'q_mode': q_mode, 'cells': {}}
    total_rows = 0

    for mod in mods:
        table = read_csv(mod)
        pay = table[:, :WINDOW]
        snr = table[:, WINDOW]
        for s in SNRS:
            rows = np.flatnonzero(snr == np.float32(s))
            if len(rows) != 2000:
                raise SystemExit(f'{mod} {s:+d} dB: expected 2000 rows, '
                                 f'found {len(rows)}')
            csv_i = pay[rows]                       # (2000, 128)
            csv_q_idx = None
            if q_mode == 'recover':
                # index CSV rows by their rounded I payload
                store = {}
                for i, row in enumerate(csv_i):
                    store.setdefault(tuple(np.round(row, 5).tolist()),
                                     []).append(rows[i])
                off_i = off[(mod, s)][:, 0, :]      # official I rows (1000, 128)
                csv_q_idx = np.empty(len(off_i), np.int64)
                misses = 0
                for j, row in enumerate(off_i):
                    key = tuple(np.round(row, 5).tolist())
                    hit = store.get(key)
                    if not hit:
                        misses += 1
                        csv_q_idx[j] = rows[0]
                    else:
                        csv_q_idx[j] = hit.pop()
                if misses:
                    raise SystemExit(f'{mod} {s:+d} dB: {misses} I rows had no '
                                     f'CSV match — Q recovery incomplete')
                arr = np.stack([off_i, pay[csv_q_idx]], axis=1)   # (1000, 2, 128)
            elif q_mode == 'zero':
                arr = np.stack([off[(mod, s)][:, 0, :],
                                np.zeros_like(off[(mod, s)][:, 0, :])], axis=1)
            elif q_mode == 'copy':
                arr = np.stack([off[(mod, s)][:, 0, :],
                                off[(mod, s)][:, 0, :].copy()], axis=1)
            else:
                raise SystemExit(f'unknown q_mode {q_mode!r}')

            arr = np.ascontiguousarray(arr, dtype=np.float32)
            if arr.shape[1] != 2 or arr.shape[2] != WINDOW:
                raise SystemExit(f'{mod} {s:+d} dB: bad shape {arr.shape}')
            if not np.isfinite(arr).all():
                raise SystemExit(f'{mod} {s:+d} dB: non-finite values')
            data[(mod, s)] = arr
            manifest['cells'][f'{mod}|{s}'] = {
                'shape': list(arr.shape),
                'q_source': ('csv_paired_rows' if q_mode == 'recover' else q_mode),
                'q_nonzero_fraction': float(np.count_nonzero(arr[:, 1, :])
                                            / arr[:, 1, :].size),
            }
            total_rows += len(arr)
        print(f'  {mod:7s} done  ({time.time()-t0:.0f}s)')

    with open(out_path, 'wb') as f:
        pickle.dump(data, f, protocol=4)
    size_mb = os.path.getsize(out_path) / 2**20
    print(f'\nwrote {out_path}  ({size_mb:.1f} MB, {total_rows:,} windows)')

    if manifest_path:
        manifest['total_windows'] = total_rows
        manifest['modulations'] = list(mods)
        manifest['snr_values'] = SNRS
        manifest['source_csv_dir'] = CSV_DIR
        manifest['official_source'] = OFFICIAL
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=1)
        print(f'wrote {manifest_path}')
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=r'_rebuilt\RML2016.10a_complex_dict.dat')
    ap.add_argument('--manifest', default=r'_rebuilt\recovery_manifest.json')
    ap.add_argument('--q-mode', default='recover',
                    choices=['recover', 'zero', 'copy'])
    ap.add_argument('--mods', nargs='*', default=MODS)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    return build(args.mods, args.out, args.manifest, args.q_mode)


if __name__ == '__main__':
    raise SystemExit(main())
