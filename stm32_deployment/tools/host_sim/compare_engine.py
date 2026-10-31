"""
compare_engine.py
=================
Host-side numerical parity check for the custom INT8 engine.

The production file `stm32_deployment/custom_engine/dsc_infer.c` compiles on a
PC with `-DDSC_HOST_SIM` (only the SMLAD/SMULL intrinsics and the two firmware
headers are stubbed; the quantized arithmetic is the same C code that runs on
the MCU).  This script:

1. builds that host binary (needs `pip install ziglang`, or pass --engine),
2. runs it on every window of the locked 44,000-window test split,
3. compares the engine's 11 q-domain logits with the shipped QDQ ONNX under
   onnxruntime (bit-exact fraction, argmax agreement, accuracy).

Reference numbers (2026-10-30, v1.0.5 firmware): 98.80% of windows bit-exact,
99.98% argmax agreement on the board; this host check reproduces the same
numbers up to the "board vs host" comparison being exact by construction.

Usage
-----
    python stm32_deployment/tools/host_sim/compare_engine.py
    python stm32_deployment/tools/host_sim/compare_engine.py --limit 4000
    python stm32_deployment/tools/host_sim/compare_engine.py --engine path/to/engine.exe
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[3]
ENGINE_SRC = ROOT / "stm32_deployment/custom_engine/dsc_infer.c"
HOST_DIR = ROOT / "stm32_deployment/tools/host_sim"
WORK_DIR = ROOT / ".tmp/host_sim"
DEFAULT_ENGINE = WORK_DIR / "engine.exe"

A_IN = 0.00213504443
A_LOGITS = 0.350085318
IN_ZP = -4
OUT_ZP = 63


def build_engine(out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "ziglang", "cc", "-O2", "-DDSC_HOST_SIM",
        f"-I{HOST_DIR}", f"-I{ROOT / 'stm32_deployment/custom_engine'}",
        str(HOST_DIR / "host_engine.c"), str(ENGINE_SRC), "-o", str(out),
    ]
    print("[build]", " ".join(cmd[2:6]), "...")
    subprocess.run(cmd, check=True)
    return out


def load_dataset(data_path: Path, split_path: Path):
    """Flat-order test split, per-window L2 normalised (same as training/eval)."""
    import pickle

    classes = ["8PSK", "AM-DSB", "AM-SSB", "BPSK", "CPFSK",
               "GFSK", "PAM4", "QAM16", "QAM64", "QPSK", "WBFM"]
    with open(data_path, "rb") as f:
        d = pickle.load(f, encoding="latin1")
    keys = sorted(d.keys(), key=lambda k: (k[0], k[1]))
    n_total = sum(d[k].shape[0] for k in keys)
    X = np.empty((n_total, 2, 128), dtype=np.float32)
    y = np.empty(n_total, dtype=np.int64)
    snr = np.empty(n_total, dtype=np.int64)
    off = 0
    for mod, s in keys:
        arr = np.asarray(d[(mod, s)], dtype=np.float32)
        m = arr.shape[0]
        X[off:off + m] = arr
        y[off:off + m] = classes.index(mod)
        snr[off:off + m] = s
        off += m
    test_idx = np.load(split_path)["test_idx"]
    X = X[test_idx]
    flat = X.reshape(len(X), -1)
    norm = np.linalg.norm(flat, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    X = (flat / norm).reshape(X.shape).astype(np.float32)
    return X, y[test_idx], snr[test_idx]


def run_engine(engine: Path, q: np.ndarray) -> tuple[np.ndarray, float]:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    in_bin, out_bin = WORK_DIR / "in.bin", WORK_DIR / "out.bin"
    q.astype(np.int8).tofile(in_bin)
    t0 = time.time()
    subprocess.run([str(engine), str(in_bin), str(out_bin), str(len(q))], check=True)
    dt = time.time() - t0
    out = np.fromfile(out_bin, dtype=np.int8).reshape(-1, 11)
    if len(out) != len(q):
        raise RuntimeError(f"engine wrote {len(out)} windows, expected {len(q)}")
    return out, dt


def run_ort(int8_onnx: Path, x: np.ndarray) -> tuple[np.ndarray, float]:
    sess = ort.InferenceSession(str(int8_onnx), providers=["CPUExecutionProvider"])
    inp, out = sess.get_inputs()[0].name, sess.get_outputs()[0].name
    batch = sess.get_inputs()[0].shape[0]
    batch = batch if isinstance(batch, int) and batch > 0 else 1
    logits = np.empty((len(x), 11), dtype=np.float32)
    t0 = time.time()
    for i in range(0, len(x), batch):
        logits[i:i + batch] = sess.run([out], {inp: x[i:i + batch]})[0]
    return logits, time.time() - t0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "datasets/RML2016.10a_dict.dat"))
    ap.add_argument("--split", default=str(ROOT / "datasets/split_indices.npz"))
    ap.add_argument("--int8", default=str(ROOT / "quantization/outputs/dscamcnet_int8.onnx"))
    ap.add_argument("--engine", default="", help="prebuilt host engine (otherwise build)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N test windows")
    ap.add_argument("--json", default="", help="write the summary JSON here")
    args = ap.parse_args()

    engine = Path(args.engine) if args.engine else (DEFAULT_ENGINE if DEFAULT_ENGINE.exists() else build_engine(DEFAULT_ENGINE))
    if not engine.exists():
        engine = build_engine(DEFAULT_ENGINE)

    X, y, snr = load_dataset(Path(args.data), Path(args.split))
    if args.limit:
        X, y, snr = X[: args.limit], y[: args.limit], snr[: args.limit]

    q_in = np.clip(np.rint(X.astype(np.float64) / A_IN) + IN_ZP, -128, 127).astype(np.int8)
    logits_ref, t_ort = run_ort(Path(args.int8), X)
    q_ref = np.clip(np.rint(logits_ref.astype(np.float64) / A_LOGITS) + OUT_ZP, -128, 127).astype(np.int8)
    q_eng, t_eng = run_engine(engine, q_in)

    pred_ref, pred_eng = logits_ref.argmax(1), q_eng.argmax(1)
    dq = np.abs(q_eng.astype(np.int16) - q_ref.astype(np.int16))
    summary = {
        "n_windows": int(len(X)),
        "bit_exact_fraction": float((dq.max(axis=1) == 0).mean()),
        "argmax_agree": float((pred_ref == pred_eng).mean()),
        "mean_abs_dq": float(dq.mean()),
        "max_abs_dq": int(dq.max()),
        "accuracy_onnxruntime": float((pred_ref == y).mean()),
        "accuracy_engine": float((pred_eng == y).mean()),
        "accuracy_plus10db_engine": float((pred_eng[snr == 10] == y[snr == 10]).mean()),
        "seconds_onnxruntime": round(t_ort, 2),
        "seconds_engine": round(t_eng, 2),
    }
    print(json.dumps(summary, indent=2))
    if args.json:
        Path(args.json).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # The board measured 98.80% bit-exact / 99.98% argmax over the same 44k windows.
    ok = summary["argmax_agree"] > 0.99 and summary["bit_exact_fraction"] > 0.95
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
