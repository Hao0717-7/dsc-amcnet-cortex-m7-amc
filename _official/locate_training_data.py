"""Locate the data/order the released official checkpoint was actually trained on.

Scores the official checkpoint (and, for reference, the local one) on every
plausible (data file x ordering x channel-source) combination, so we can tell
which dataset the released weights belong to.

Orderings tested
  sorted  : keys sorted by (mod, snr)   <- what models/train.py uses
  insert  : raw pickle insertion order
Cell orders tested
  as-is, reversed, snr-major
Channel content
  real    : both channels as stored
  zeroQ   : channel 1 zeroed (the workspace CSV export has no Q)
"""
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, 'models')
from dsc_amcnet import build_dsc_amcnet, normalize_window  # noqa: E402

MODS = ["8PSK", "AM-DSB", "AM-SSB", "BPSK", "CPFSK",
        "GFSK", "PAM4", "QAM16", "QAM64", "QPSK", "WBFM"]
CI = {m: i for i, m in enumerate(MODS)}


def cells(path):
    with open(path, 'rb') as f:
        d = pickle.load(f, encoding='latin1')
    return d


def order_keys(d, mode):
    ks = list(d.keys())
    if mode == 'sorted':
        return sorted(ks, key=lambda k: (k[0], k[1]))
    if mode == 'snr':
        return sorted(ks, key=lambda k: (k[1], k[0]))
    if mode == 'reversed':
        return list(reversed(ks))
    return ks


def flat(d, mode, zero_q=False, limit_cells=None):
    ks = order_keys(d, mode)
    if limit_cells:
        ks = ks[:limit_cells]
    X = np.concatenate([d[k] for k in ks]).astype(np.float32)
    if zero_q:
        X[:, 1, :] = 0.0
    y = np.concatenate([np.full(len(d[k]), CI[k[0]], np.int64) for k in ks])
    snr = np.concatenate([np.full(len(d[k]), k[1], np.int64) for k in ks])
    return X, y, snr


@torch.no_grad()
def score(model, X, y, bs=4096):
    c = 0
    for i in range(0, len(X), bs):
        p = model(normalize_window(torch.from_numpy(X[i:i + bs])).float()).argmax(1)
        c += int((p == torch.from_numpy(y[i:i + bs])).sum())
    return 100 * c / len(X)


def load_ckpt(path):
    sd = torch.load(path, map_location='cpu')
    if isinstance(sd, dict) and 'state_dict' in sd:
        sd = sd['state_dict']
    m = build_dsc_amcnet()
    m.load_state_dict(sd, strict=False)
    m.eval()
    return m


def main():
    off = load_ckpt(r'_official\dscamcnet_best.pt')
    loc = load_ckpt(r'models\checkpoints\dscamcnet_best.pt')

    for path, tag in ((r'datasets\RML2016.10a_dict.dat', 'official pkl'),
                      (r'datasets\RML2016.10a_dict.dat', 'workspace dat')):
        d = cells(path)
        print(f'\n================ {tag}  ({len(d)} cells) ================')
        first = next(iter(d.values()))
        print(f'  cell shape {first.shape} dtype {first.dtype}')
        for mode in ('sorted', 'insert', 'snr', 'reversed'):
            ks = order_keys(d, mode)
            # sample a spread of cells, 100 rows each, to keep it fast
            pick = np.linspace(0, len(ks) - 1, 22).astype(int)
            X = np.concatenate([d[ks[i]][:100] for i in pick]).astype(np.float32)
            y = np.concatenate([np.full(100, CI[ks[i][0]], np.int64) for i in pick])
            for zq in (False, True):
                Xz = X.copy()
                if zq:
                    Xz[:, 1, :] = 0.0
                print(f'  order={mode:<9} zeroQ={str(zq):<5} '
                      f'official={score(off, Xz, y):6.2f}%  local={score(loc, Xz, y):6.2f}%')
            if mode == 'insert':
                print(f'     first 3 keys insertion order: {ks[:3]}')
                print(f'     first 3 keys sorted    order: {order_keys(d, "sorted")[:3]}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
