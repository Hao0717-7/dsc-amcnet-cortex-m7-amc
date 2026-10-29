"""P0: reference accuracy of the INT8 QDQ model on the locked test split."""
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path(r"D:\dsc-amcnet-cortex-m7-amc")
DATA = ROOT / "datasets" / "RML2016.10a_dict.dat"
SPLIT = ROOT / "datasets" / "split_indices.npz"
MODEL = ROOT / "quantization" / "outputs" / "dscamcnet_int8.onnx"
OUT = ROOT / "stm32_deployment" / "results"
OUT.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

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
    n = np.linalg.norm(flat, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return (flat / n).reshape(x.shape)


def main():
    t0 = time.time()
    sp = np.load(SPLIT)
    test_idx = sp["test_idx"]
    print(f"test_idx: {len(test_idx)}", flush=True)
    X, y, snr = load_rml(DATA)
    print(f"dataset: {X.shape} in {time.time()-t0:.1f}s", flush=True)
    Xt = l2n(X[test_idx]); yt = y[test_idx]; st = snr[test_idx]

    sess = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    print("input:", iname, sess.get_inputs()[0].shape, flush=True)

    preds = np.empty(len(Xt), np.int64)
    t1 = time.time()
    for i in range(len(Xt)):
        out = sess.run(None, {iname: Xt[i:i + 1]})[0]
        preds[i] = int(out.argmax())
        if i % 2000 == 0:
            print(f"  {i}/{len(Xt)}", flush=True)
    acc = (preds == yt).mean()
    print(f"overall acc: {acc:.4f}  ({time.time()-t1:.1f}s)", flush=True)

    per_snr = {}
    for s in sorted(set(st.tolist())):
        m = st == s
        per_snr[int(s)] = float((preds[m] == yt[m]).mean())
    per_cls = {}
    for c in range(len(CLASSES)):
        m = yt == c
        per_cls[CLASSES[c]] = float((preds[m] == yt[m]).mean())
    cm = np.zeros((len(CLASSES), len(CLASSES)), np.int64)
    for t, p in zip(yt, preds):
        cm[t, p] += 1

    res = {
        "n": int(len(Xt)),
        "acc": float(acc),
        "per_snr": per_snr,
        "per_class": per_cls,
        "confusion": cm.tolist(),
        "classes": CLASSES,
        "seconds": round(time.time() - t0, 1),
    }
    (OUT / "ref_accuracy.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("per_snr:", {k: round(v, 4) for k, v in per_snr.items()}, flush=True)
    print("saved:", OUT / "ref_accuracy.json", flush=True)


if __name__ == "__main__":
    sys.exit(main())
