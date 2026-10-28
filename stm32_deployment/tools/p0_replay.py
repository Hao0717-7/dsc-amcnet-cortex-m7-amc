"""P0: replay samples to the board custom engine and compare with ORT.

Protocol (firmware `Src/main.c`):
  boot: wait <=1.5 s for 'R' -> print "[REPLAY] ready" -> wait 256 bytes -> reply "L,q0..q10"
  loop: wait 'R' -> wait 256 bytes -> reply line
"""
import argparse
import json
import pickle
import subprocess
import sys
import time
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
UV4 = Path(r"D:\APP\Keil5\UV4\UV4.exe")
PROJ = Path(r"D:\DSC_AMCNet_Deployment\MDK-ARM\DSC_AMCNet_Deployment.uvprojx")

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


def estimate_ref(sess, Xs):
    iname = sess.get_inputs()[0].name
    ref = np.array([sess.run(None, {iname: Xs[i:i + 1]})[0][0] for i in range(len(Xs))])
    return np.clip(np.rint(ref / 0.35008532) + 63, -128, 127).astype(np.int8)


def flash_board(log=Path(r"D:\dsc-amcnet-cortex-m7-amc\.tmp\p0_flash.log")):
    log.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run([str(UV4), "-f", str(PROJ), "-o", str(log)],
                       capture_output=True, text=True)
    return r.returncode


def read_one_line(ser, timeout_s):
    """Read until a complete line arrives; returns str without newline."""
    end = time.time() + timeout_s
    buf = b""
    while time.time() < end:
        b = ser.read(1)
        if not b:
            continue
        if b in (b"\r", b"\n"):
            if buf:
                return buf.decode("ascii", "replace")
            continue
        buf += b
        if len(buf) > 200:
            return buf.decode("ascii", "replace")
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--port", default="COM3")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--flash", action="store_true",
                    help="flash the board first (resets it into the replay window)")
    ap.add_argument("--boot-timeout", type=float, default=4.0)
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
    # fair reference: the board only sees the quantised window, so feed ORT the
    # dequantised q values instead of the original float window
    Xdq = (qs.astype(np.float32) + 4.0) * A_IN
    ref_q = estimate_ref(sess, Xdq)

    ser = serial.Serial(args.port, 115200, timeout=0.5)
    ser.reset_input_buffer()

    board = np.full((len(Xs), 11), 127, np.int8)
    got = 0

    if args.flash:
        rc = flash_board()
        print(f"[replay] flash rc={rc}", flush=True)

    first = True
    for i in range(len(Xs)):
        deadline = time.time() + (args.boot_timeout if first else 2.0)
        ok = False
        while time.time() < deadline and not ok:
            ser.write(b"R" + qs[i].tobytes())
            line = read_one_line(ser, 0.8)
            if line.startswith("L,"):
                vals = [int(v) for v in line[2:].split(",")]
                if len(vals) == 11:
                    board[i] = np.array(vals, np.int8)
                    got += 1
                    ok = True
            elif line.startswith("[REPLAY]"):
                # board is in replay mode now; the previous sample was consumed as
                # the boot sample, so resend this one on the next loop turn
                continue
        first = False

    ser.close()

    n = got
    exact = int((board[:n] == ref_q[:n]).all(axis=1).sum())
    argmax_agree = int((board[:n].argmax(1) == ref_q[:n].argmax(1)).sum())
    acc_board = float((board[:n].argmax(1) == ys[:n]).mean())
    acc_ref = float((ref_q[:n].argmax(1) == ys[:n]).mean())
    mad = float(np.abs(board[:n].astype(np.int32) - ref_q[:n].astype(np.int32)).mean())

    res = {
        "n_requested": int(len(Xs)), "n_ok": got,
        "exact_11_logits": exact, "argmax_agree": argmax_agree,
        "acc_board": acc_board, "acc_ref": acc_ref, "mean_abs_diff_q": mad,
        "per_class_acc_board": {CLASSES[c]: float((board[:n].argmax(1)[ys[:n] == c] == c).mean())
                                 for c in range(11) if (ys[:n] == c).any()},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "replay_compare.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))
    print("saved:", OUT / "replay_compare.json")


if __name__ == "__main__":
    sys.exit(main())
