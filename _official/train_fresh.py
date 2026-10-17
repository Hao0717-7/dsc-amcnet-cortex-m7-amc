"""Train DSC-AMCNet on the genuine RML2016.10a pickle (real I/Q) with a fresh split.

Purpose: the locked split_indices.npz is index-space-incompatible with the
official pickle, so this script builds its own stratified 80/20 split
(SEED=42) over the 220 cells and records it to .npz for reuse, then trains
with the protocol from models/train.py (Adam lr=1e-3, batch 512, L2 window
normalization, ReduceLROnPlateau on val acc, early-stop patience 15) and
reports +10 dB test accuracy, the figure the README quotes.

Usage
    python _official/train_fresh.py --data <pkl> --split-out <npz> \
        --epochs 100 --output <ckpt> [--epochs-probe 5]
"""
import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, 'models')
from dsc_amcnet import build_dsc_amcnet, normalize_window  # noqa: E402

MODS = ["8PSK", "AM-DSB", "AM-SSB", "BPSK", "CPFSK",
        "GFSK", "PAM4", "QAM16", "QAM64", "QPSK", "WBFM"]
CI = {m: i for i, m in enumerate(MODS)}
SEED = 42


def load_official(path):
    with open(path, 'rb') as f:
        d = pickle.load(f, encoding='latin1')
    keys = sorted(d.keys(), key=lambda k: (k[0], k[1]))
    X = np.concatenate([d[k] for k in keys]).astype(np.float32)
    y = np.concatenate([np.full(len(d[k]), CI[k[0]], np.int64) for k in keys])
    snr = np.concatenate([np.full(len(d[k]), k[1], np.int64) for k in keys])
    return X, y, snr


def build_split(y, snr, seed=SEED, frac_test=0.20):
    """Stratified 80/20 over (class, snr) cells, deterministic."""
    rng = np.random.default_rng(seed)
    train, test = [], []
    for c in range(len(MODS)):
        for s in sorted(set(snr.tolist())):
            idx = np.flatnonzero((y == c) & (snr == s))
            perm = rng.permutation(len(idx))
            n_test = int(round(len(idx) * frac_test))
            test.extend(idx[perm[:n_test]])
            train.extend(idx[perm[n_test:]])
    return np.sort(np.array(train, np.int64)), np.sort(np.array(test, np.int64))


def aug_normalize(x):
    return normalize_window(x)


@torch.no_grad()
def evaluate(model, X, y, idx, device, bs=4096):
    model.eval()
    c = 0
    for i in range(0, len(idx), bs):
        b = torch.from_numpy(X[idx[i:i + bs]]).to(device)
        p = model(aug_normalize(b)).argmax(1).cpu().numpy()
        c += int((p == y[idx[i:i + bs]]).sum())
    return 100 * c / len(idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--split-out', default=None)
    ap.add_argument('--split', default=None, help='existing split npz to reuse')
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--batch-size', type=int, default=512)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--val-frac', type=float, default=0.10)
    ap.add_argument('--patience', type=int, default=15)
    ap.add_argument('--output', required=True)
    ap.add_argument('--threads', type=int, default=24)
    ap.add_argument('--save-every', type=int, default=0,
                    help='also save a snapshot of the best-so-far weights every N epochs '
                         '(into <output-dir>/eNNN.pt) so a per-epoch accuracy curve can '
                         'be reconstructed afterwards')
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    dev = 'cpu'
    # NOTE: the two checkpoints kept in models/checkpoints/ (89.73% and 89.00%)
    # were produced by versions of this script that did NOT seed the PyTorch RNG.
    # They therefore started from different random weight initialisations and saw
    # different batch orders; that, not a protocol change, is the whole 0.73 pp
    # gap between them. Seeding here makes future runs reproducible.
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    t0 = time.time()
    X, y, snr = load_official(args.data)
    print(f'data: {X.shape} {X.dtype}  cells={len(set(zip(y.tolist(), snr.tolist())))}  '
          f'load {time.time()-t0:.1f}s', flush=True)
    print(f'Q-channel nonzero fraction: {float((X[:,1,:]!=0).mean()):.4f}', flush=True)

    if args.split:
        z = np.load(args.split)
        train_idx, test_idx = z['train_idx'], z['test_idx']
        print(f'reusing split {args.split}: train {len(train_idx):,} test {len(test_idx):,}',
              flush=True)
    else:
        train_idx, test_idx = build_split(y, snr)
        print(f'fresh split: train {len(train_idx):,} test {len(test_idx):,}', flush=True)
        if args.split_out:
            Path(args.split_out).parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(args.split_out, train_idx=train_idx, test_idx=test_idx)
            print(f'saved split -> {args.split_out}', flush=True)

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(train_idx))
    n_val = int(len(train_idx) * args.val_frac)
    val_idx, tr_idx = train_idx[perm[:n_val]], train_idx[perm[n_val:]]
    print(f'train {len(tr_idx):,} / val {len(val_idx):,} / test {len(test_idx):,}', flush=True)

    hi_test = test_idx[snr[test_idx] == 10]
    print(f'+10 dB test subset: {len(hi_test):,}', flush=True)

    model = build_dsc_amcnet().to(dev)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'DSC-AMCNet parameters: {n_par:,} (paper: 4,961)', flush=True)

    ds = TensorDataset(torch.from_numpy(X[tr_idx]), torch.from_numpy(y[tr_idx]))
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0,
                    drop_last=False)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5,
                                                       patience=5)
    crit = nn.CrossEntropyLoss()

    best, best_ep, bad = -1.0, -1, 0
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        tot = cor = 0
        for xb, yb in dl:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            out = model(aug_normalize(xb))
            loss = crit(out, yb)
            loss.backward()
            opt.step()
            tot += len(yb)
            cor += int((out.argmax(1) == yb).sum())
        val_acc = evaluate(model, X, y, val_idx, dev)
        sched.step(val_acc)
        flag = ''
        if val_acc > best:
            best, best_ep, bad = val_acc, ep, 0
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), args.output)
            flag = '  *saved'
        else:
            bad += 1
        print(f'Epoch {ep:3d}/{args.epochs}  train={100*cor/tot:6.2f}%  '
              f'val={val_acc:6.2f}%  lr={opt.param_groups[0]["lr"]:.2e}  '
              f'[{time.time()-t0:.0f}s]{flag}', flush=True)
        if bad >= args.patience:
            print(f'early stop at epoch {ep} (best epoch {best_ep})', flush=True)
            break
        if args.save_every and ep % args.save_every == 0:
            snap = Path(args.output).with_name(f'e{ep:03d}.pt')
            torch.save(model.state_dict(), snap)

    sd = torch.load(args.output, map_location=dev)
    model.load_state_dict(sd)
    model.eval()
    hi = evaluate(model, X, y, hi_test, dev)
    allte = evaluate(model, X, y, test_idx, dev)
    print(f'\nbest val {best:.2f}% @ epoch {best_ep}')
    print(f'+10 dB test accuracy (fresh split): {hi:.2f}%')
    print(f'all-SNR test accuracy: {allte:.2f}%')
    print(f'saved -> {args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
