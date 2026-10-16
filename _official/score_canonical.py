"""Score the released official checkpoint on the workspace dataset in train.py's order.

train.py loads the pickle and flattens it in *sorted* key order
(`keys = sorted(d.keys(), key=lambda k: (k[0], k[1]))`, i.e. class-major, then
SNR ascending), which is also the order the CSV-built workspace dataset was
written in.  This script evaluates the released checkpoint on that data, plus
a load_state_dict strict check and a sanity check on logit statistics.
"""
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, 'models')
from dsc_amcnet import build_dsc_amcnet, normalize_window, channel_shuffle  # noqa: E402

MODS = ["8PSK", "AM-DSB", "AM-SSB", "BPSK", "CPFSK",
        "GFSK", "PAM4", "QAM16", "QAM64", "QPSK", "WBFM"]


def flatten(path, order='sorted'):
    with open(path, 'rb') as f:
        d = pickle.load(f, encoding='latin1')
    keys = (sorted(d.keys(), key=lambda k: (k[0], k[1])) if order == 'sorted'
            else list(d.keys()))
    ci = {m: i for i, m in enumerate(MODS)}
    X = np.concatenate([d[k] for k in keys]).astype(np.float32)
    y = np.concatenate([np.full(len(d[k]), ci[k[0]], np.int64) for k in keys])
    snr = np.concatenate([np.full(len(d[k]), k[1], np.int64) for k in keys])
    return X, y, snr


@torch.no_grad()
def acc(model, X, y, idx, bs=4096):
    c = t = 0
    for i in range(0, len(idx), bs):
        b = idx[i:i + bs]
        p = model(normalize_window(torch.from_numpy(X[b])).float()).argmax(1)
        c += int((p == torch.from_numpy(y[b])).sum())
        t += len(b)
    return 100 * c / t


def main():
    X, y, snr = flatten(r'datasets\RML2016.10a_dict.dat', 'sorted')
    print('workspace dataset (sorted order):', X.shape)
    z = np.load(r'datasets\split_indices.npz')
    tr, te = z['train_idx'], z['test_idx']
    hi_tr = tr[snr[tr] == 10]
    hi_te = te[snr[te] == 10]

    for tag, ck in (('official', r'_official\dscamcnet_best.pt'),
                    ('local   ', r'models\checkpoints\dscamcnet_best.pt')):
        sd = torch.load(ck, map_location='cpu')
        if isinstance(sd, dict) and 'state_dict' in sd:
            sd = sd['state_dict']
        m = build_dsc_amcnet()
        missing, unexpected = m.load_state_dict(sd, strict=False)
        m.eval()
        print(f'\n{tag} checkpoint: missing={len(missing)} unexpected={len(unexpected)}')
        if missing:
            print('   missing keys:', missing[:8])
        if unexpected:
            print('   unexpected keys:', unexpected[:8])
        print(f'  +10dB train-split {acc(m, X, y, hi_tr):6.2f}%   '
              f'+10dB test-split {acc(m, X, y, hi_te):6.2f}%   '
              f'all-SNR test {acc(m, X, y, te):6.2f}%')
        xb = normalize_window(torch.from_numpy(X[hi_te[:256]])).float()
        lo = m(xb)
        print(f'  logit stats: mean={lo.mean():+.3f} std={lo.std():.3f} '
              f'min={lo.min():+.3f} max={lo.max():+.3f}')

    # fully random baseline for reference
    torch.manual_seed(0)
    rnd = build_dsc_amcnet()
    rnd.eval()
    print(f'\nrandom-init baseline: +10dB test {acc(rnd, X, y, hi_te):6.2f}%')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
