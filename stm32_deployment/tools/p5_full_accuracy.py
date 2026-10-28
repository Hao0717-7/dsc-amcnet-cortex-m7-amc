"""P5-2: full locked-test-split accuracy on the board (custom engine, replay mode).

Protocol (firmware `Src/main.c` replay_loop / replay_serve):
    host -> 'R' + 256 x int8 (quantised window)
    board -> "L,q0,...,q10"

Run:  python p5_full_accuracy.py --flash
"""
import argparse
import json
import pickle
import subprocess
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import serial

ROOT = Path(r"D:\dsc-amcnet-cortex-m7-amc")
DATA = ROOT / "datasets" / "RML2016.10a_dict.dat"
SPLIT = ROOT / "datasets" / "split_indices.npz"
MODEL = ROOT / "quantization" / "outputs" / "dscamcnet_int8.onnx"
OUT = ROOT / "stm32_deployment" / "results"
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


def read_line(ser, timeout_s):
    end = time.time() + timeout_s
    buf = bytearray()
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
            break
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="COM3")
    ap.add_argument("--flash", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="0 = all 44k")
    ap.add_argument("--out", default=str(OUT / "p5_accuracy_full.json"))
    args = ap.parse_args()

    X, y, snr = load_rml(DATA)
    test_idx = np.load(SPLIT)["test_idx"]
    if args.limit:
        test_idx = test_idx[: args.limit]
    Xs = l2n(X[test_idx]); ys = y[test_idx]; ss = snr[test_idx]
    qs = quant(Xs)
    print(f"samples: {len(test_idx)}  (all-SNR mix)")

    sess = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    ref = np.empty((len(qs), 11), np.int8)
    t0 = time.time()
    x_all = (qs.astype(np.float32) + 4.0) * A_IN        # fixed batch=1 model
    for i in range(len(qs)):
        out = sess.run(None, {iname: x_all[i:i + 1]})[0]
        ref[i] = np.clip(np.rint(out[0] / 0.35008532) + 63, -128, 127)
    print(f"ORT reference done in {time.time() - t0:.1f}s")

    if args.flash:
        r = subprocess.run([str(UV4), "-f", str(PROJ), "-o",
                            str(ROOT / ".tmp" / "p5_flash.log")], capture_output=True)
        print("flash rc:", r.returncode)

    ser = serial.Serial(args.port, 115200, timeout=0.5)
    ser.reset_input_buffer()
    # wait for the classification loop (it polls for 'R'); UART RX is polled, so
    # bytes sent during the boot benchmarks would be lost
    if args.flash:
        deadline = time.time() + 25.0
        while time.time() < deadline:
            line = read_line(ser, 0.5)
            if "CLASSIFY" in line:
                print("board:", line)
                break
        ser.reset_input_buffer()

    board = np.full((len(qs), 11), 127, np.int8)
    ok = 0
    t0 = time.time()
    for i in range(len(qs)):
        for attempt in range(2):
            ser.write(b"R" + qs[i].tobytes())
            line = read_line(ser, 3.0)
            if line.startswith("L,"):
                vals = line[2:].split(",")
                if len(vals) == 11:
                    board[i] = np.array([int(v) for v in vals], np.int8)
                    ok += 1
                    break
        if (i + 1) % 1000 == 0:
            el = time.time() - t0
            eta = el / (i + 1) * (len(qs) - i - 1)
            print(f"{i + 1}/{len(qs)}  ok={ok}  {el:.0f}s elapsed, eta {eta / 60:.1f} min", flush=True)

    ser.close()
    print(f"received {ok}/{len(qs)} replies")

    valid = np.any(board != 127, axis=1) | (board == 127).all(axis=1)
    pred_b = board.argmax(1)
    pred_r = ref.argmax(1)
    match = (board == ref).all(1)
    acc_b = (pred_b == ys).mean()
    acc_r = (pred_r == ys).mean()

    per_snr = {}
    for s in sorted(set(ss.tolist())):
        m = ss == s
        per_snr[str(s)] = {"acc_board": float((pred_b[m] == ys[m]).mean()),
                           "acc_ref": float((pred_r[m] == ys[m]).mean()),
                           "n": int(m.sum())}

    conf = np.zeros((11, 11), np.int64)
    for t, p in zip(ys, pred_b):
        conf[t, p] += 1

    res = {
        "n": int(len(qs)),
        "board_acc": float(acc_b),
        "ref_acc": float(acc_r),
        "bit_exact_match_rate": float(match.mean()),
        "argmax_match_rate": float((pred_b == pred_r).mean()),
        "per_snr": per_snr,
        "confusion": conf.tolist(),
        "classes": CLASSES,
        "latency_ms": 0.979,
        "port": args.port,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("n", "board_acc", "ref_acc",
                                          "bit_exact_match_rate", "argmax_match_rate")}, indent=1))
    print("written:", args.out)


if __name__ == "__main__":
    main()
