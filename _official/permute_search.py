"""Brute-force the index->sample mapping the released checkpoint was trained on.

The released dscamcnet_best.pt matches the workspace architecture exactly
(4,961 params, 50 keys, strict load) but scores ~15% on datasets/RML2016.10a_dict.dat
and ~18% on the official pickle in sorted order.  The locked split_indices.npz has a
220,000-wide index space, so it must point into *some* 220,000-sample arrangement.
This sweeps the plausible arrangements and reports +10 dB accuracy of the locked
split's test indices under each one; the right arrangement should jump to ~80%+.
"""
import itertools
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, 'models')
from dsc_amcnet import build_dsc_amcnet, normalize_window  # noqa: E402

MODS = ["8PSK", "AM-DSB", "AM-SSB", "BPSK", "CPFSK",
        "GFSK", "PAM4", "QAM16", "QAM64", "QPSK", "WBFM"]
CI = {m: i for i, m in enumerate(MODS)}


def load_ckpt(path):
    sd = torch.load(path, map_location='cpu')
    if isinstance(sd, dict) and 'state_dict' in sd:
        sd = sd['state_dict']
    m = build_dsc_amcnet()
    m.load_state_dict(sd, strict=False)
    m.eval()
    return m


@torch.no_grad()
def score(model, X, y, idx, bs=4096):
    c = 0
    for i in range(0, len(idx), bs):
        b = idx[i:i + bs]
        p = model(normalize_window(torch.from_numpy(X[b])).float()).argmax(1)
        c += int((p == torch.from_numpy(y[b])).sum())
    return 100 * c / len(idx)


def from_pickle(path, order):
    with open(path, 'rb') as f:
        d = pickle.load(f, encoding='latin1')
    ks = list(d.keys())
    if order == 'sorted':
        ks = sorted(ks, key=lambda k: (k[0], k[1]))
    elif order == 'sorted_snr':
        ks = sorted(ks, key=lambda k: (k[1], k[0]))
    elif order == 'reversed':
        ks = list(reversed(ks))
    X = np.concatenate([d[k] for k in ks]).astype(np.float32)
    y = np.concatenate([np.full(len(d[k]), CI[k[0]], np.int64) for k in ks])
    snr = np.concatenate([np.full(len(d[k]), k[1], np.int64) for k in ks])
    return X, y, snr


def subsample(X, y, snr, per_cell):
    """Keep the first `per_cell` rows of every contiguous 2000/1000-row cell."""
    uniq_snr = sorted(set(snr.tolist()))
    out = []
    n_cell = len(X) // (11 * len(uniq_snr))
    for c in range(11):
        for s in uniq_snr:
            m = np.flatnonzero((y == c) & (snr == s))
            out.append(m[:per_cell])
    sel = np.concatenate(out)
    return X[sel], y[sel], snr[sel]


def main():
    off = load_ckpt(r'_official\dscamcnet_best.pt')
    loc = load_ckpt(r'models\checkpoints\dscamcnet_best.pt')
    z = np.load(r'datasets\split_indices.npz')
    tr, te = z['train_idx'], z['test_idx']

    cands = []
    for path, tag in ((r'datasets\RML2016.10a_dict.dat', 'workspace 1k'),
                      (r'datasets\full_2k\RML2016.10a_2k_dict.dat', 'workspace 2k'),
                      (r'_official\radio\RML2016.10a_dict_optimized.pkl', 'official pkl')):
        for order in ('sorted', 'insert', 'sorted_snr', 'reversed'):
            try:
                X, y, snr = from_pickle(path, order)
            except Exception as e:  # noqa: BLE001
                print(f'  skip {tag}/{order}: {type(e).__name__}: {e}')
                continue
            cands.append((f'{tag}|{order}|{len(X)}', X, y, snr))

    results = []
    for tag, X, y, snr in cands:
        if len(X) < 220000:
            continue
        variants = [('as-is', X, y, snr)]
        if len(X) >= 440000:
            Xs, ys, ss = subsample(X, y, snr, 1000)
            variants.append(('first-1k', Xs, ys, ss))
        for vtag, Xv, yv, snrv in variants:
            hi = te[snrv[te] == 10]
            a_off = score(off, Xv, yv, te)
            a_off_hi = score(off, Xv, yv, hi)
            a_loc = score(loc, Xv, yv, te)
            print(f'{tag:<28} {vtag:<9} N={len(Xv):>6}  '
                  f'official={a_off:6.2f}% (hi {a_off_hi:6.2f}%)  local={a_loc:6.2f}%',
                  flush=True)
            results.append((a_off_hi, tag, vtag))
    results.sort(reverse=True)
    print('\nranking by official +10dB accuracy:')
    for a, tag, vtag in results[:6]:
        print(f'  {a:6.2f}%  {tag} [{vtag}]')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
