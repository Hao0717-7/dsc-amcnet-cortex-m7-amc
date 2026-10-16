"""Test whether the CSV rows are rounded copies of the official rows.

The CSV export carries ~8-9 significant digits of the original float32 values,
so a bit-exact hash match fails.  Quantise both sides to a decimal grid and
look for matches, then report the maximum absolute discrepancy.
"""
import os
import pickle

import numpy as np

OFF = r'_official\radio\RML2016.10a_dict_optimized.pkl'
CSV_DIR = r'C:\Users\Hao\Desktop\RadioML2016.10a'
MODS = ["8PSK", "AM-DSB", "AM-SSB", "BPSK", "CPFSK",
        "GFSK", "PAM4", "QAM16", "QAM64", "QPSK", "WBFM"]
SNRS = list(range(-20, 20, 2))


def csv_payload(mod):
    flat = np.array(
        [float(t) for tok in open(os.path.join(CSV_DIR, mod + '.csv'), 'rb')
         .read().split() for t in tok.split(b',')], dtype=np.float32)
    table = flat.reshape(-1, 129)
    return table[:, :128], table[:, 128]


def quant_key(a, decimals):
    q = np.round(a.astype(np.float64), decimals)
    return set(map(tuple, q.tolist()))


def main():
    with open(OFF, 'rb') as f:
        off = pickle.load(f, encoding='latin1')

    for mod in ['8PSK', 'QPSK', 'WBFM']:
        pay, snr = csv_payload(mod)
        print(f'\n=== {mod} ===')
        for s in [-20, 0, 10, 18]:
            o = off[(mod, s)][:, 0, :].astype(np.float64)      # (1000,128)
            c = pay[snr == np.float32(s)].astype(np.float64)   # (2000,128)
            for dec in (5, 6, 7):
                ok = quant_key(o, dec)
                ch = quant_key(c, dec)
                n = len(ok & ch)
                if dec == 5:
                    first = n
                last = n
            # nearest-neighbour discrepancy: for 20 official rows find the
            # closest CSV row (row-major l2) and report the distance
            d = np.linalg.norm(o[:20, None, :] - c[None, :1000, :], axis=2)
            nn = d.min(axis=1)
            print(f'  {s:+3d} dB: quantised-set overlap d5={first}/1000 '
                  f'd7={last}/1000 | nearest-row l2: min={nn.min():.2e} '
                  f'med={np.median(nn):.2e} max={nn.max():.2e} '
                  f'| typical |value|={np.abs(c).mean():.2e}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
