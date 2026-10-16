"""Evaluate the official-pickle retrain checkpoint on the +10 dB test bin and all-SNR test set.

Uses the same flat loading / split / normalization as _official/train_fresh.py so the
numbers are directly comparable with the training log.
"""
import argparse
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, 'models')
from dsc_amcnet import build_dsc_amcnet, normalize_window  # noqa: E402

MODS = ["8PSK", "AM-DSB", "AM-SSB", "BPSK", "CPFSK",
        "GFSK", "PAM4", "QAM16", "QAM64", "QPSK", "WBFM"]
CI = {m: i for i, m in enumerate(MODS)}


def load(path):
    with open(path, 'rb') as f:
        d = pickle.load(f, encoding='latin1')
    keys = sorted(d.keys(), key=lambda k: (k[0], k[1]))
    X = np.concatenate([d[k] for k in keys]).astype(np.float32)
    y = np.concatenate([np.full(len(d[k]), CI[k[0]], np.int64) for k in keys])
    snr = np.concatenate([np.full(len(d[k]), k[1], np.int64) for k in keys])
    return X, y, snr


@torch.no_grad()
def probs(model, X, idx, bs=4096):
    out = []
    for i in range(0, len(idx), bs):
        b = idx[i:i + bs]
        out.append(model(normalize_window(torch.from_numpy(X[b])).float()))
    return torch.cat(out)


@torch.no_grad()
def acc(model, X, y, idx, bs=4096):
    p = probs(model, X, idx, bs).argmax(1).numpy()
    return 100 * float((p == y[idx]).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--data', default=r'_official\radio\RML2016.10a_dict_optimized.pkl')
    ap.add_argument('--split', default=r'datasets\split_indices.npz')
    ap.add_argument('--per-snr', action='store_true')
    ap.add_argument('--per-class', action='store_true')
    args = ap.parse_args()

    X, y, snr = load(args.data)
    z = np.load(args.split)
    te = z['test_idx']
    sd = torch.load(args.checkpoint, map_location='cpu')
    if isinstance(sd, dict) and 'state_dict' in sd:
        sd = sd['state_dict']
    m = build_dsc_amcnet()
    m.load_state_dict(sd, strict=True)
    m.eval()

    hi = te[snr[te] == 10]
    print(f'checkpoint      : {args.checkpoint}')
    print(f'test (all SNR)  : {acc(m, X, y, te):6.2f}%   n={len(te):,}')
    print(f'test (+10 dB)   : {acc(m, X, y, hi):6.2f}%   n={len(hi):,}   '
          f'[paper target 85.80%]')

    hi_all = np.flatnonzero(snr == 10)
    print(f'ALL samples +10dB: {acc(m, X, y, hi_all):6.2f}%   n={len(hi_all):,}')

    if args.per_snr:
        print('\nper-SNR test accuracy:')
        for s in sorted(set(snr[te].tolist())):
            idx = te[snr[te] == s]
            print(f'  {s:+4d} dB : {acc(m, X, y, idx):6.2f}%  (n={len(idx):,})')

    if args.per_class:
        logits = probs(m, X, te)
        pred = logits.argmax(1).numpy()
        print('\nper-class test accuracy (all SNR):')
        for c, name in enumerate(MODS):
            mask = y[te] == c
            print(f'  {c:2d} {name:<7}: {100*float((pred[mask]==c).mean()):6.2f}%  '
                  f'(n={int(mask.sum()):,})')
        print('\nper-class test accuracy (+10 dB):')
        idx = np.flatnonzero(snr[te] == 10)
        pos = np.searchsorted(te, te[idx])
        pred_hi = pred[pos]
        for c, name in enumerate(MODS):
            mask = y[te[idx]] == c
            print(f'  {c:2d} {name:<7}: {100*float((pred_hi[mask]==c).mean()):6.2f}%  '
                  f'(n={int(mask.sum()):,})')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
