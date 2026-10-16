"""Provenance diagnostics: what is actually in each candidate data source?

Answers four questions with hard evidence:
  1. Does the official DeepSig pickle have Q == I, or a real quadrature channel?
  2. Does each CSV row's second half equal its first half (I repeated) or hold Q?
  3. Does the rebuild reproduce the official pickle exactly?
  4. What does the released checkpoint score on each data variant?
"""
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, 'models')
from dsc_amcnet import build_dsc_amcnet, normalize_window  # noqa: E402

MODS = ["8PSK", "AM-DSB", "AM-SSB", "BPSK", "CPFSK",
        "GFSK", "PAM4", "QAM16", "QAM64", "QPSK", "WBFM"]
CSV_DIR = r'C:\Users\Hao\Desktop\RadioML2016.10a'
OFF = r'_official\radio\RML2016.10a_dict_optimized.pkl'


def csv_table(mod):
    flat = np.array([float(t) for tok in open(f'{CSV_DIR}\\{mod}.csv', 'rb')
                     .read().split() for t in tok.split(b',')], dtype=np.float32)
    return flat.reshape(-1, 129)


def main():
    with open(OFF, 'rb') as f:
        off = pickle.load(f, encoding='latin1')
    print('=== 1. official pickle channel structure ===')
    for k in [('8PSK', 10), ('WBFM', -20), ('QAM64', 0)]:
        a = off[k]
        d = float(np.abs(a[:, 0, :] - a[:, 1, :]).max())
        print(f'  {k}: shape {a.shape}  max|ch0-ch1|={d:.3e}  '
              f'ch0 std={a[:,0,:].std():.4f} ch1 std={a[:,1,:].std():.4f}  '
              f'corr={np.corrcoef(a[:,0,:].ravel(), a[:,1,:].ravel())[0,1]:+.4f}')
    k = ('8PSK', 10)
    print('  official 8PSK +10dB row0 ch0[:6]:', off[k][0, 0, :6])
    print('  official 8PSK +10dB row0 ch1[:6]:', off[k][0, 1, :6])

    print('\n=== 2. CSV row internal structure ===')
    t = csv_table('8PSK')
    m = t[:, 128] == np.float32(10)
    rows = t[m]
    print(f'  8PSK +10 dB rows: {len(rows)}')
    print('  row0 first 6 cols :', rows[0, :6])
    print('  row0 cols 64..69  :', rows[0, 64:70])
    print('  row0 last 6 cols  :', rows[0, 122:128])
    print(f'  max|col[0:64]-col[64:128]| over {len(rows)} rows: '
          f'{np.abs(rows[:, :64] - rows[:, 64:128]).max():.3e}')
    print(f'  max|col[:128] - official I rows| (row-wise, same order): '
          f'{np.abs(rows[:1000, :128] - off[k][:, 0, :]).max():.3e}')
    print(f'  rows count per SNR for all 20 bins: '
          f'{np.bincount(np.searchsorted(np.arange(-20,20,2), t[:,128]))}')

    print('\n=== 3. rebuild vs official ===')
    with open(r'_rebuilt\RML2016.10a_complex_dict.dat', 'rb') as f:
        rb = pickle.load(f, encoding='latin1')
    dmax = max(float(np.abs(rb[kk][:, 0, :] - off[kk][:, 0, :]).max()) for kk in off)
    print(f'  max|rebuild ch0 - official ch0| over all cells: {dmax:.3e}')
    same_as_copy = all(np.array_equal(rb[kk][:, 0, :], rb[kk][:, 1, :]) for kk in off)
    print(f'  rebuild ch1 == ch0 in every cell: {same_as_copy}')
    print(f'  8PSK +10dB rebuild row0 ch1[:6]: {rb[k][0,1,:6]}')
    print(f'  8PSK +10dB official row0 ch1[:6]: {off[k][0,1,:6]}')

    print('\n=== 4. checkpoint scores by data variant (official order) ===')
    ci = {m: i for i, m in enumerate(MODS)}
    keys = list(off.keys())
    y = np.concatenate([np.full(len(off[kk]), ci[kk[0]], np.int64) for kk in keys])
    snr = np.concatenate([np.full(len(off[kk]), kk[1], np.int64) for kk in keys])
    X_off = np.concatenate([off[kk] for kk in keys]).astype(np.float32)
    X_rb = np.concatenate([rb[kk] for kk in keys]).astype(np.float32)
    X_i0 = np.concatenate([np.stack([off[kk][:, 0, :],
                                     np.zeros_like(off[kk][:, 0, :])], 1)
                           for kk in keys]).astype(np.float32)
    z = np.load(r'datasets\split_indices.npz')
    tr, te = z['train_idx'], z['test_idx']
    hi = te[snr[te] == 10]

    sd = torch.load(r'_official\dscamcnet_best.pt', map_location='cpu')
    if isinstance(sd, dict) and 'state_dict' in sd:
        sd = sd['state_dict']
    model = build_dsc_amcnet()
    model.load_state_dict(sd)
    model.eval()

    @torch.no_grad()
    def acc(X, idx):
        c = t = 0
        for i in range(0, len(idx), 4096):
            b = idx[i:i + 4096]
            p = model(normalize_window(torch.from_numpy(X[b])).float()).argmax(1)
            c += int((p == torch.from_numpy(y[b])).sum())
            t += len(b)
        return 100 * c / t

    for tag, X in (('official full complex', X_off),
                   ('rebuild (recovered Q)', X_rb),
                   ('official I, Q=0      ', X_i0)):
        print(f'  {tag}: +10dB test {acc(X, hi):6.2f}%   all-SNR test {acc(X, te):6.2f}%')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
