"""Recover the cell ordering that the locked split was generated against.

The locked split is a stratified 80/20 draw over 220,000 samples. If the flat
index space is 220 consecutive (mod, snr) cells of 1,000 rows each, then a
stratified split visits every chunk with nearly the same occupancy (800 train
and 200 test in each), whatever the chunk order. That gives an unambiguous way
to recover the cell order: score candidate orderings by how uniform the
per-1000-chunk occupancy of split["train_idx"] is.
"""
import itertools
import os
import pickle

import numpy as np

MODS = ["8PSK", "AM-DSB", "AM-SSB", "BPSK", "CPFSK",
        "GFSK", "PAM4", "QAM16", "QAM64", "QPSK", "WBFM"]
SNRS = list(range(-20, 20, 2))
OFF = r'_official\radio\RML2016.10a_dict_optimized.pkl'
CSV_DIR = r'C:\Users\Hao\Desktop\RadioML2016.10a'


def load_official():
    with open(OFF, 'rb') as f:
        return pickle.load(f, encoding='latin1')


def load_csv():
    cells = {}
    for mod in MODS:
        flat = np.array(
            [float(t) for tok in open(os.path.join(CSV_DIR, mod + '.csv'), 'rb')
             .read().split() for t in tok.split(b',')], dtype=np.float32)
        table = flat.reshape(-1, 129)
        iq = np.zeros((len(table), 2, 128), dtype=np.float32)
        iq[:, 0, :] = table[:, :128]
        for s in SNRS:
            cells[(mod, s)] = iq[np.flatnonzero(table[:, 128] == np.float32(s))]
    return cells


def candidate_orders(keys_official):
    """Named candidate cell orderings (each a list of 220 (mod, snr) keys)."""
    names = {}
    names['official_insertion'] = list(keys_official)
    names['sort_str_snr'] = sorted(keys_official, key=lambda k: (k[0], str(k[1])))
    names['sort_str_snr_neg0'] = sorted(
        keys_official, key=lambda k: (k[0], ('0' + str(k[1])) if k[1] >= 0 else str(k[1])))
    names['sort_abs_snr_then_sign'] = sorted(
        keys_official, key=lambda k: (k[0], abs(k[1]), k[1]))
    names['sort_str_padded'] = sorted(
        keys_official, key=lambda k: (k[0], f'{k[1]:+03d}'))
    names['sort_rev_str'] = sorted(
        keys_official, key=lambda k: (k[0], str(k[1])), reverse=True)
    names['sort_col_major'] = sorted(
        keys_official, key=lambda k: (str(k[1]), k[0]))
    names['sort_col_major_neg0'] = sorted(
        keys_official,
        key=lambda k: (('0' + str(k[1])) if k[1] >= 0 else str(k[1]), k[0]))
    names['group_str_snr_by_mod'] = [k for m in MODS
                                     for k in sorted(
                                         [x for x in keys_official if x[0] == m],
                                         key=lambda k: str(k[1]))]
    names['mods_perm'] = []
    return names


def occupancy(train_idx, n_chunks=220):
    c = np.bincount(train_idx // 1000, minlength=n_chunks)[:n_chunks]
    return c


def score(c):
    return float(np.std(c)) / max(1e-9, float(np.mean(c)))


def main():
    z = np.load(r'datasets\split_indices.npz')
    tr = z['train_idx']
    print('train_idx per 1000-chunk (first 12, assumes some order):',
          occupancy(tr)[:12])
    off = load_official()
    keys = list(off.keys())
    cands = candidate_orders(keys)

    print('\n== candidate orderings (raw occupancy scores) ==')
    for name, order in cands.items():
        if len(order) != 220:
            continue
        # order does not change the split; what changes is which data chunk a
        # fixed index range points at.  Occupancy of the index space is fixed,
        # so the discriminating statistic must use data, not indices.
        print(f'  {name}')

    # Data-driven: for a candidate order, check whether each 1000-chunk of the
    # concatenated data holds one modulation and the modulation label agrees
    # with the split's class counts.
    print('\n== chunk purity: is each 1000-sample chunk single-modulation? ==')
    for name, order in cands.items():
        if len(order) != 220:
            continue
        vals = []
        for chunk in range(0, 220):
            sample = order[chunk]
            vals.append(sample[0])
        # purity is trivially 1 by construction; instead check the data itself
        # by looking at whether the CSV rebuild's cell boundaries align
        print(f'  {name}: first 3 chunks {[ (k[0], k[1]) for k in order[:3] ]}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
