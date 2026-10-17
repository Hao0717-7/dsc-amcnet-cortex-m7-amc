"""Quick training-budget probe: does the locked split fit the official data?

Runs a short training run (few epochs) on a given dataset+split pair and prints
train accuracy per epoch.  A correct (data, split) pairing reaches high training
accuracy within a couple of epochs; a mismatched pairing stalls near chance.
"""
import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, 'models')
from dsc_amcnet import build_dsc_amcnet, normalize_window  # noqa: E402

MODS = ["8PSK", "AM-DSB", "AM-SSB", "BPSK", "CPFSK",
        "GFSK", "PAM4", "QAM16", "QAM64", "QPSK", "WBFM"]


class DS(Dataset):
    def __init__(self, X, y, l2=True):
        self.X = normalize_window(torch.from_numpy(X).float()) if l2 \
            else torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def load_official():
    with open(r'datasets\RML2016.10a_dict.dat', 'rb') as f:
        d = pickle.load(f, encoding='latin1')
    ci = {m: i for i, m in enumerate(MODS)}
    X, y, snr = [], [], []
    for (m, s), a in d.items():
        X.append(a)
        y.append(np.full(len(a), ci[m], np.int64))
        snr.append(np.full(len(a), int(s), np.int64))
    return (np.concatenate(X).astype(np.float32),
            np.concatenate(y), np.concatenate(snr))


def load_local(path):
    with open(path, 'rb') as f:
        d = pickle.load(f, encoding='latin1')
    ci = {m: i for i, m in enumerate(MODS)}
    X, y, snr = [], [], []
    for (m, s), a in d.items():
        X.append(a)
        y.append(np.full(len(a), ci[m], np.int64))
        snr.append(np.full(len(a), int(s), np.int64))
    return (np.concatenate(X).astype(np.float32),
            np.concatenate(y), np.concatenate(snr))


@torch.no_grad()
def acc(model, loader):
    model.eval()
    c = t = 0
    for xb, yb in loader:
        p = model(xb).argmax(1)
        c += int((p == yb).sum())
        t += len(yb)
    return c / t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='official')
    ap.add_argument('--split', default=r'datasets\split_indices.npz')
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--bs', type=int, default=512)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--limit', type=int, default=0,
                    help='use only the first N train samples (0 = all)')
    args = ap.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)

    if args.data == 'official':
        X, y, snr = load_official()
        print('data: official pickle', X.shape)
    elif args.data == 'csv2k':
        X, y, snr = load_local(r'datasets\full_2k\RML2016.10a_2k_dict.dat')
        print('data: local 2k rebuild', X.shape)
    else:
        X, y, snr = load_local(args.data)
        print('data:', args.data, X.shape)

    z = np.load(args.split)
    tr, te = z['train_idx'], z['test_idx']
    print(f'split: max train {tr.max()} max test {te.max()} (data N={len(y)})')
    if args.limit:
        tr = tr[:args.limit]

    tr_loader = DataLoader(DS(X[tr], y[tr]), batch_size=args.bs, shuffle=True)
    val_idx = te[:20000]
    val_loader = DataLoader(DS(X[val_idx], y[val_idx]), batch_size=args.bs)

    model = build_dsc_amcnet()
    print('params:', sum(p.numel() for p in model.parameters() if p.requires_grad))
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        tc = tt = 0
        for xb, yb in tr_loader:
            opt.zero_grad()
            out = model(xb)
            loss = crit(out, yb)
            loss.backward()
            opt.step()
            tc += int((out.argmax(1) == yb).sum())
            tt += len(yb)
        print(f'epoch {ep}: train_acc={tc/tt*100:.2f}%  '
              f'val_acc={acc(model, val_loader)*100:.2f}%  ({time.time()-t0:.0f}s)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
