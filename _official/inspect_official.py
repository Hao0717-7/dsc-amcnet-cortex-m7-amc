"""Unpack and inspect the official RadioML 2016.10a distribution (Zenodo 18397070).

Extracts _official/RML2016.10a.tar.bz2, locates the pickled dict, reports the
canonical structure, and (if available) evaluates both the released official
checkpoint and the locally trained checkpoint at +10 dB.
"""
import os
import pickle
import sys
import tarfile
import time

import numpy as np
import torch

sys.path.insert(0, 'models')
from dsc_amcnet import build_dsc_amcnet, normalize_window

TAR = r'_official\RML2016.10a.tar.bz2'
DEST = r'_official\radio'


def extract():
    if not os.path.exists(TAR):
        print('missing', TAR)
        return None
    t = time.time()
    with tarfile.open(TAR, 'r:bz2') as tf:
        tf.extractall(DEST)
    print(f'extracted {TAR} -> {DEST} in {time.time()-t:.0f}s')
    found = []
    for root, _d, files in os.walk(DEST):
        for f in files:
            p = os.path.join(root, f)
            found.append((os.path.getsize(p), p))
    for size, p in sorted(found, reverse=True):
        print(f'  {size:>12,}  {p}')
    return found


def load_dict(path):
    t = time.time()
    with open(path, 'rb') as f:
        d = pickle.load(f, encoding='latin1')
    print(f'loaded {path}: {len(d)} keys in {time.time()-t:.0f}s')
    keys = list(d.keys())
    print('  first 5 keys:', keys[:5])
    print('  last 5 keys :', keys[-5:])
    print('  snr type    :', type(keys[0][1]).__name__)
    mods = sorted({k[0] for k in keys})
    print('  modulations :', mods)
    shapes = {d[k].shape for k in keys}
    dtypes = {d[k].dtype for k in keys}
    print('  shapes      :', shapes, ' dtypes:', dtypes)
    return d


@torch.no_grad()
def acc(model, X, y, bs=4096):
    model.eval()
    correct = 0
    for i in range(0, len(y), bs):
        xb = normalize_window(torch.from_numpy(X[i:i + bs])).float()
        pred = model(xb).argmax(1).numpy()
        correct += int((pred == y[i:i + bs]).sum())
    return correct / len(y)


def main():
    if not os.path.isdir(DEST):
        extract()

    pkl = None
    for root, _d, files in os.walk(DEST):
        for f in files:
            if f.endswith(('.pkl', '.dat', '.p')):
                pkl = os.path.join(root, f)
    if pkl is None:
        # maybe the tar holds a .tar of arrays
        for root, _d, files in os.walk(DEST):
            for f in files:
                print('  other file:', os.path.join(root, f))
        return 1

    d = load_dict(pkl)

    # canonical flat layout, matching models/train.py sorted-key order
    keys = sorted(d.keys(), key=lambda k: (k[0], str(k[1])))
    mods = sorted({k[0] for k in keys})
    ci = {m: i for i, m in enumerate(mods)}
    X = np.concatenate([d[k] for k in keys]).astype(np.float32)
    y = np.concatenate([np.full(len(d[k]), ci[k[0]], np.int64) for k in keys])
    snr = np.concatenate([np.full(len(d[k]), int(k[1]), np.int64) for k in keys])
    print('\nflat X', X.shape, 'y', y.shape, 'snr', snr.shape)

    z = np.load(r'datasets\split_indices.npz')
    tr, te = z['train_idx'], z['test_idx']
    print('split keys:', list(z.keys()), 'train', tr.shape, 'test', te.shape)
    print('index range fits:', tr.max() < len(X), te.max() < len(X))

    hi = te[snr[te] == 10]
    print('+10 dB test bin size:', len(hi))

    for tag, ck in (('official', r'_official\dscamcnet_best.pt'),
                    ('local   ', r'models\checkpoints\dscamcnet_best.pt')):
        sd = torch.load(ck, map_location='cpu')
        if isinstance(sd, dict) and 'state_dict' in sd:
            sd = sd['state_dict']
        m = build_dsc_amcnet()
        try:
            m.load_state_dict(sd)
        except Exception as e:
            print(tag, 'LOAD FAIL', e)
            continue
        print(f'{tag}: all-SNR={acc(m, X[te], y[te])*100:.2f}%  '
              f'+10dB={acc(m, X[hi], y[hi])*100:.2f}%  '
              f'train-split={acc(m, X[tr[:20000]], y[tr[:20000]])*100:.2f}%')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
