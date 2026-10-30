"""One-shot offline re-measurement of the repo results, intended for a
side-by-side comparison against the numbers printed in the *official* README
(`_official/repo/DSC-AMCNet/README.md`, "Key Results (Locked, Silicon-Measured)").

What it measures (no board needed):
  * dataset / split facts
  * INT8 ONNX and FP32 ONNX accuracy on the locked 44,000-window test split:
    overall, SNR == +10 dB, SNR >= +10 dB ("high SNR"), per-SNR
  * ONNX graph size / initializer element count
  * resource totals parsed from the Keil map of the flashed firmware

Writes a JSON summary to %TEMP%\\dsc_compare_official.json (and prints it).
No files inside the repository are modified.
"""
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

ROOT = Path(r"D:\dsc-amcnet-cortex-m7-amc")
sys.path.insert(0, str(ROOT / "stm32_deployment" / "tools"))
import p0_replay as R  # noqa: E402  (load_rml / l2n / quant)

FP32 = ROOT / "quantization" / "outputs" / "dscamcnet_fp32.onnx"
INT8 = ROOT / "quantization" / "outputs" / "dscamcnet_int8.onnx"
MAP = Path(r"D:\DSC_AMCNet_Deployment\MDK-ARM\DSC_AMCNet_Deployment\DSC_AMCNet_Deployment.map")
CLASSES = R.CLASSES


def model_facts(path):
    m = onnx.load(str(path))
    elems = 0
    for i in m.graph.initializer:
        n = 1
        for d in i.dims:
            n *= int(d)
        elems += n
    return {
        "file": str(path.relative_to(ROOT)),
        "bytes": path.stat().st_size,
        "nodes": len(m.graph.node),
        "initializer_elems": elems,
        "opsets": [int(o.version) for o in m.opset_import],
    }


def acc_stats(pred, y, snr):
    pred = np.asarray(pred)
    out = {"overall": float((pred == y).mean())}
    for name, mask in (("snr_eq_10", snr == 10), ("snr_ge_10", snr >= 10)):
        out[name] = {"n": int(mask.sum()), "acc": float((pred[mask] == y[mask]).mean())}
    out["per_snr"] = {
        str(int(s)): {"n": int((snr == s).sum()), "acc": float((pred[snr == s] == y[snr == s]).mean())}
        for s in sorted(set(int(v) for v in snr))
    }
    return out


def parse_map():
    if not MAP.exists():
        return None
    txt = MAP.read_text(encoding="utf-8", errors="replace")
    out = {}
    m = re.search(r"Total RO Size\s*\(code \+ ro data\)\s*(\d+)\s*\(([\d.]+)kB\)", txt)
    if m:
        out["rom"] = {"bytes": int(m.group(1)), "kb": float(m.group(2))}
    m = re.search(r"Total RW Size\s*\(rw data \+ zi data\)\s*(\d+)\s*\(([\d.]+)kB\)", txt)
    if m:
        out["ram"] = {"bytes": int(m.group(1)), "kb": float(m.group(2))}
    for key, label in (("code", "Code"), ("ro", "RO Data"), ("rw", "RW Data"), ("zi", "ZI Data")):
        m = re.search(r"^\s*%s\s+(\d+)" % label.replace(" ", r"\s+"), txt, re.M)
        if m:
            out[key] = int(m.group(1))
    return out


def main():
    t0 = time.time()
    sp = np.load(ROOT / "datasets" / "split_indices.npz")
    test_idx = sp["test_idx"]
    train_idx = sp["train_idx"]
    X, y, snr = R.load_rml(ROOT / "datasets" / "RML2016.10a_dict.dat")
    Xt = R.l2n(X[test_idx])
    yt = y[test_idx]
    st = snr[test_idx]
    qt = R.quant(Xt)
    Xdq = (qt.astype(np.float32) + 4.0) * R.A_IN

    sess_f = ort.InferenceSession(str(FP32), providers=["CPUExecutionProvider"])
    sess_q = ort.InferenceSession(str(INT8), providers=["CPUExecutionProvider"])
    name_f = sess_f.get_inputs()[0].name
    name_q = sess_q.get_inputs()[0].name

    def run(sess, name, arr):
        pred = np.empty(len(arr), np.int64)
        for i in range(len(arr)):
            logits = sess.run(None, {name: arr[i:i + 1].astype(np.float32)})[0]
            pred[i] = int(np.asarray(logits).argmax(1)[0])
        return pred

    t1 = time.time()
    pf = run(sess_f, name_f, Xt)
    pq = run(sess_q, name_q, Xdq)

    out = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
        },
        "dataset": {
            "windows": int(len(X)),
            "bytes": (ROOT / "datasets" / "RML2016.10a_dict.dat").stat().st_size,
            "split_train": int(len(train_idx)),
            "split_test": int(len(test_idx)),
            "split_union_ok": bool(set(range(len(X))) == set(train_idx.tolist()) | set(test_idx.tolist())),
        },
        "onnx": {"fp32": model_facts(FP32), "int8": model_facts(INT8)},
        "accuracy_fp32": acc_stats(pf, yt, st),
        "accuracy_int8": acc_stats(pq, yt, st),
        "accuracy_int8_bit_identical_to_ref_json": None,
        "resources_map": parse_map(),
        "seconds": round(time.time() - t0, 1),
    }
    ref = ROOT / "stm32_deployment" / "results" / "ref_accuracy.json"
    if ref.exists():
        r = json.loads(ref.read_text(encoding="utf-8"))
        ref_acc = float(r["acc"])
        out["ref_accuracy_json_acc"] = ref_acc
        out["accuracy_int8_bit_identical_to_ref_json"] = abs(
            ref_acc - out["accuracy_int8"]["overall"]) < 1e-12
    dst = Path(os.environ.get("TEMP", ".")) / "dsc_compare_official.json"
    dst.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print("saved:", dst)


if __name__ == "__main__":
    main()
