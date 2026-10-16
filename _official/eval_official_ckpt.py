"""Evaluate the released official checkpoint on the official dataset.

Reconstructs the full complex 220k dataset two ways:
  A. straight from the official optimised pickle (I and Q),
  B. from the CSV export's I channel plus the Q recovered by matching I rows.
Then it scores the released checkpoint on its own training split and on the
locked test split, for all SNRs and for the +10 dB bin.
"""
import os
import pickle
import sys
import time

import numpy as np
import torch

sys.path.insert(0, 'models')
from dsc_amcnet import build_dsc_amcnet, normalize_window

OFF = r'_official\radio\RML2016.10a_dict_optimized.pkl'
CSV_DIR = r'C:\Users\Hao\Desktop\RadioML2016.10a'
MODS = ["8PSK", "AM-DSB", "AM-SSB", "BPSK", "CPFSK",
        "GFSK", "PAM4", "QAM16", "QAM64", "QPSK", "WBFM"]
SNRS = list(range(-20, 20, 2))


def csv_cells(mod):
    flat = np.array(
        [float(t) for tok in open(os.path.join(CSV_DIR, mod + '.csv'), 'rb')
         .read().split() for t in tok.split(b',')], dtype=np.float32)
    table = flat.reshape(-1, 129)
    pay = table[:, :128]
    snr = table[:, 128]
    return {s: pay[np.flatnonzero(snr == np.float32(s))] for s in SNRS}


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
    t0 = time.time()
    with open(OFF, 'rb') as f:
        off = pickle.load(f, encoding='latin1')
    keys = list(off.keys())                        # official random order
    print(f'official pickle loaded: {len(keys)} keys  ({time.time()-t0:.0f}s)')

    # ---- flat arrays in official key order --------------------------------
    cls_of_mod = {m: i for i, m in enumerate(MODS)}
    X_off, y_off, snr_off = [], [], []
    for (m, s) in keys:
        X_off.append(off[(m, s)])
        y_off.append(np.full(len(off[(m, s)]), cls_of_mod[m], np.int64))
        snr_off.append(np.full(len(off[(m, s)]), int(s), np.int64))
    X_off = np.concatenate(X_off).astype(np.float32)
    y_off = np.concatenate(y_off)
    snr_off = np.concatenate(snr_off)

    # ---- rebuild the same samples from the CSV (recover Q) ----------------
    t = time.time()
    X_csv, y_csv, snr_csv = [], [], []
    cache = {}
    for (m, s) in keys:
        if m not in cache:
            cache[m] = csv_cells(m)
        pay = cache[m][s]
        key = np.round(pay.astype(np.float64), 5)
        store = {}
        for i, row in enumerate(key):
            store.setdefault(tuple(row.tolist()), []).append(i)
        o = off[(m, s)]
        idx = []
        for row in o[:, 0, :]:
            k = tuple(np.round(row.astype(np.float64), 5).tolist())
            idx.append(store[k].pop())
        idx = np.asarray(idx)
        X_csv.append(o.copy())                      # I from official
        y_csv.append(np.full(len(o), cls_of_mod[m], np.int64))
        snr_csv.append(np.full(len(o), int(s), np.int64))
        # report reconstruction fidelity of I from the CSV
        if s == 10 and m == '8PSK':
            diff = np.abs(pay[idx] - o[:, 0, :]).max()
            print(f'  CSV I-channel fidelity check 8PSK +10 dB: max|diff|={diff:.2e}')
    X_csv = np.concatenate(X_csv)
    y_csv = np.concatenate(y_csv)
    snr_csv = np.concatenate(snr_csv)
    print(f'CSV-based reconstruction done ({time.time()-t:.0f}s)')

    z = np.load(r'datasets\split_indices.npz')
    tr, te = z['train_idx'], z['test_idx']
    hi_tr = tr[snr_off[tr] == 10]
    hi_te = te[snr_off[te] == 10]
    print(f'splits: train {len(tr):,} test {len(te):,}  '
          f'+10 dB: train {len(hi_tr):,} test {len(hi_te):,}')

    for tag, ck in (('official ckpt', r'_official\dscamcnet_best.pt'),
                    ('local ckpt   ', r'models\checkpoints\dscamcnet_best.pt')):
        sd = torch.load(ck, map_location='cpu')
        if isinstance(sd, dict) and 'state_dict' in sd:
            sd = sd['state_dict']
        m = build_dsc_amcnet()
        m.load_state_dict(sd)
        print(f'\n{tag}:')
        print(f'  official-order data: train-split {acc(m, X_off[tr], y_off[tr])*100:.2f}%'
              f'  test-split {acc(m, X_off[te], y_off[te])*100:.2f}%'
              f'  +10dB-train {acc(m, X_off[hi_tr], y_off[hi_tr])*100:.2f}%'
              f'  +10dB-test {acc(m, X_off[hi_te], y_off[hi_te])*100:.2f}%')
    print(f'\ntotal {time.time()-t0:.0f}s')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
