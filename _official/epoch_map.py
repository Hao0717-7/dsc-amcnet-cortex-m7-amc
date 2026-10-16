"""Build the epoch -> accuracy table from per-epoch snapshots.

Evaluates every snapshot in `--dir` (files named eNNN.pt) on the locked test
split and prints the +10 dB test accuracy (the README metric) plus the all-SNR
test accuracy, so the training curve can be read off after a single training
run.

    python _official/epoch_map.py --dir _official/epochmap --data datasets/RML2016.10a_dict.dat
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "models"))
from dsc_amcnet import build_dsc_amcnet, normalize_window  # noqa: E402
from train import load_rml2016, load_split  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--data", default="datasets/RML2016.10a_dict.dat")
    ap.add_argument("--split", default="datasets/split_indices.npz")
    ap.add_argument("--csv")
    args = ap.parse_args()

    X, y, snr = load_rml2016(args.data)
    _, te = load_split(args.split)
    hi = te[snr[te] == 10]

    def norm(idx):
        return normalize_window(torch.from_numpy(X[idx]).float())

    Xte, yte = norm(te), y[te]
    Xhi, yhi = norm(hi), y[hi]
    print(f"test {len(te):,} ({len(hi):,} at +10 dB)", flush=True)

    snaps = sorted(Path(args.dir).glob("e*.pt"),
                   key=lambda p: int(re.search(r"(\d+)", p.stem).group(1)))
    model = build_dsc_amcnet()
    rows = []
    with torch.no_grad():
        for p in snaps:
            ep = int(re.search(r"(\d+)", p.stem).group(1))
            model.load_state_dict(torch.load(p, map_location="cpu"), strict=True)
            model.eval()
            pred_te = model(Xte).argmax(1).numpy()
            pred_hi = model(Xhi).argmax(1).numpy()
            a_all = 100 * float((pred_te == yte).mean())
            a_hi = 100 * float((pred_hi == yhi).mean())
            rows.append((ep, a_all, a_hi))
            print(f"epoch {ep:3d}  all-SNR {a_all:6.2f}%   +10 dB {a_hi:6.2f}%", flush=True)

    if args.csv:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", encoding="utf-8") as f:
            f.write("epoch,test_all_snr_pct,test_plus10db_pct\n")
            for ep, a_all, a_hi in rows:
                f.write(f"{ep},{a_all:.4f},{a_hi:.4f}\n")
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
