"""P0: replay samples to the board custom engine and compare with ORT."""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import serial

ROOT = Path(r"D:\dsc-amcnet-cortex-m7-amc")
DATA = ROOT / "datasets" / "RML2016.10a_dict.dat"
SPLIT = ROOT / "datasets" / "split_indices.npz"
MODEL = ROOT / "quantization" / "outputs" / "dscamcnet_int8.onnx"
OUT = ROOT / ".tmp" / "p0"
A_IN = 0.00213504443

CLASSES = ["8PSK", "AM-DSB", "AM-SSB", "BPSK", "CPFSK", "GFSK",
           "PAM4", "QAM16", "QAM64", "QPSK", "WBFM"]


def load_rml(path):
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="latin1")
    keys = sorted(d.keys(), key=lambda k: (k[0], k[1]))
    classes = sorted({k[0] for k in keys})
    c2i = {c: i for i, c in enumerate(classes)}
    X, y, snr = [], [], []
    for (mod, s) in keys:
        arr = d[(mod, s)]
        X.append(arr)
        y.extend([c2i[mod]] * len(arr))
        snr.extend([s] * len(arr))
    return (np.concatenate(X).astype(np.float32),
            np.asarray(y, np.int64), np.asarray(snr, np.int64))


def l2n(x):
    flat = x.reshape(len(x), -1)
    n = np.maximum(np.linalg.norm(flat, axis=1, keepdims=True), 1e-12)
    return (flat / n).reshape(x.shape)


def quant(z):
    q = np.rint(z / A_IN).astype(np.int32) - 4
    return np.clip(q, -128, 127).astype(np.int8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--port", default="COM3")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sp = np.load(SPLIT)
    test_idx = sp["test_idx"]
    X, y, snr = load_rml(DATA)
    rng = np.random.default_rng(args.seed)
    sel = rng.choice(len(test_idx), size=args.n, replace=False)
    idx = test_idx[sel]
    Xs = l2n(X[idx]); ys = y[idx]; ss = snr[idx]
    qs = quant(Xs)

    sess = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    ref = np.array([sess.run(None, {iname: Xs[i:i + 1]})[0][0] for i in range(len(Xs))])
    ref_q = np.clip(np.rint(ref / 0.35008532) + 63, -128, 127).astype(np.int8)

    ser = serial.Serial(args.port, 115200, timeout=3)
    board = np.zeros((len(Xs), 11), np.int8)
    got = 0
    for i in range(len(Xs)):
        ser.write(b"R" + qs[i].tobytes())
        line = b""
        while not line.startswith(b"L,"):
            line = ser.readline().strip()
            if not line:
                break
        if line.startswith(b"L,"):
            vals = [int(v) for v in line[2:].split(b",")]
            if len(vals) == 11:
                board[i] = np.array(vals, np.int8)
                got += 1
    ser.close()

    exact = int((board[:got] == ref_q[:got]).all(axis=1).sum())
    argmax_agree = int((board[:got].argmax(1) == ref_q[:got].argmax(1)).sum())
    acc_board = float((board[:got].argmax(1) == ys[:got]).mean())
    acc_ref = float((ref_q[:got].argmax(1) == ys[:got]).mean())
    mad = float(np.abs(board[:got].astype(np.int32) - ref_q[:got].astype(np.int32)).mean())

    res = {
        "n_requested": int(len(Xs)), "n_ok": got,
        "exact_11_logits": exact, "argmax_agree": argmax_agree,
        "acc_board": acc_board, "acc_ref": acc_ref, "mean_abs_diff_q": mad,
        "per_class_acc_board": {CLASSES[c]: float((board[:got].argmax(1)[ys[:got] == c] == c).mean())
                                 for c in range(11) if (ys[:got] == c).any()},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "replay_compare.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))
    print("saved:", OUT / "replay_compare.json")


if __name__ == "__main__":
    sys.exit(main())
