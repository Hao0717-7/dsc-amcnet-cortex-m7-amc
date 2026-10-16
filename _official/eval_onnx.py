"""Evaluate an exported FP32/INT8 ONNX artifact on the locked test split,
using exactly the same window normalization as train.py (per-window L2).

    python _official/eval_onnx.py --onnx quantization/outputs/dscamcnet_int8.onnx \
        --data datasets/RML2016.10a_dict.dat --split datasets/split_indices.npz
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "models"))
from dsc_amcnet import normalize_window  # noqa: E402
from train import load_rml2016, load_split  # noqa: E402

import torch  # noqa: E402
import onnxruntime as ort  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", required=True)
    args = ap.parse_args()

    X, y, snr = load_rml2016(args.data)
    _, test_idx = load_split(args.split)
    print(f"samples {len(X):,}  test {len(test_idx):,}")

    xt = normalize_window(torch.from_numpy(X[test_idx]).float()).numpy()
    yt = y[test_idx]
    st = snr[test_idx]

    so = ort.SessionOptions()
    so.intra_op_num_threads = 24
    sess = ort.InferenceSession(args.onnx, so, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    # the exported graph has a fixed batch size of 1 (deployment shape)
    out = [sess.run(None, {iname: xt[i:i + 1]})[0] for i in range(len(xt))]
    pred = np.concatenate(out).argmax(1)

    hi = st == 10
    print(f"test (all SNR)  : {100.0 * (pred == yt).mean():6.2f}%   n={len(yt):,}")
    print(f"test (+10 dB)   : {100.0 * (pred[hi] == yt[hi]).mean():6.2f}%   "
          f"n={int(hi.sum()):,}")
    print(f"test (SNR>=10)  : {100.0 * (pred[st >= 10] == yt[st >= 10]).mean():6.2f}%   "
          f"n={int((st >= 10).sum()):,}")
    # writing next to the ONNX artifact may be denied by the file sandbox
    try:
        np.save(Path(args.onnx).with_suffix(".pred.npy"), pred)
    except OSError as exc:
        print(f"(predictions not saved: {exc})")

if __name__ == "__main__":
    main()
