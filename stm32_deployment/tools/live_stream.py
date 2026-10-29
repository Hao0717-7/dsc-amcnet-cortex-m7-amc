"""Stream IQ windows to the board (UART) and print live classification results.

Protocol (firmware `Src/main.c`, stream mode):
    host -> 'S' at 115200
    board: "[STREAM] switching to 921600, 256 x int16 LE per window"
    then 256 x int16 LE per window (I[0..127], Q[0..127], full scale 32768)
    board -> "C,<class>,<conf_x1000>,<us_x100>,<name>"
"""
import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import serial

ROOT = Path(r"D:\dsc-amcnet-cortex-m7-amc")
DATA = ROOT / "datasets" / "RML2016.10a_dict.dat"
SPLIT = ROOT / "datasets" / "split_indices.npz"

CLASSES = ["8PSK", "AM-DSB", "AM-SSB", "BPSK", "CPFSK", "GFSK",
           "PAM4", "QAM16", "QAM64", "QPSK", "WBFM"]


def load_rml(path):
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="latin1")
    keys = sorted(d.keys(), key=lambda k: (k[0], k[1]))
    c2i = {c: i for i, c in enumerate(sorted({k[0] for k in keys}))}
    X, y = [], []
    for (mod, s) in keys:
        a = d[(mod, s)]
        X.append(a)
        y.extend([c2i[mod]] * len(a))
    return np.concatenate(X).astype(np.float32), np.asarray(y)


def to_i16(win):
    """float (2,128) -> planar int16 (I then Q), any constant scale is fine
    because the firmware L2-normalises the window."""
    v = np.clip(win * 20000.0, -32767, 32767).astype(np.int16)
    return v.reshape(-1).tobytes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--port", default="COM3")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--flash", action="store_true", help="flash first so 'S' lands in the 1.5 s boot window")
    args = ap.parse_args()

    X, y = load_rml(DATA)
    test_idx = np.load(SPLIT)["test_idx"]
    rng = np.random.default_rng(args.seed)
    idx = test_idx[rng.choice(len(test_idx), size=args.n, replace=False)]

    ser = serial.Serial(args.port, 115200, timeout=1)
    ser.reset_input_buffer()
    if args.flash:
        import subprocess
        subprocess.run([r"D:\APP\Keil5\UV4\UV4.exe", "-f",
                        r"D:\DSC_AMCNet_Deployment\MDK-ARM\DSC_AMCNet_Deployment.uvprojx",
                        "-o", str(pathlib.Path(tempfile.gettempdir()) / "dsc_stream_flash.log")],
                       capture_output=True)
    line = ""
    for _ in range(40):
        ser.write(b"S")
        got = ser.readline().decode("ascii", "replace").strip()
        if got:
            print("board:", got)
        if "STREAM" in got:
            line = got
            break
    if not line:
        print("no STREAM handshake")
        ser.close()
        return
    ser.baudrate = 921600
    time.sleep(0.2)
    ser.reset_input_buffer()

    ok = 0
    compute_us = []
    rtt_ms = []
    preds, labels = [], []
    for k in range(args.n):
        t0 = time.time()
        ser.write(to_i16(X[idx[k]]))
        r = ser.readline().decode("ascii", "replace").strip()
        dt = (time.time() - t0) * 1000.0
        if not r.startswith("C,"):
            print("unexpected:", r)
            continue
        _, cls, conf, us, name = r.split(",")
        cls = int(cls); conf = int(conf); us = int(us)
        ok += 1
        compute_us.append(us)
        rtt_ms.append(dt)
        preds.append(cls)
        labels.append(int(y[idx[k]]))
        if k < 10:
            print(f"[{k:3d}] {name:<7} conf={conf/1000:.3f} compute={us/100:.1f}us rtt={dt:6.1f}ms")

    if ok:
        preds = np.array(preds); labels = np.array(labels)
        print(f"\nwindows={ok}/{args.n}  board_acc={(preds == labels).mean():.3f}")
        print(f"compute  mean={np.mean(compute_us)/100:.1f}us  min={np.min(compute_us)/100:.1f}  max={np.max(compute_us)/100:.1f}")
        print(f"round-trip mean={np.mean(rtt_ms):.1f}ms (UART 921600: 512 B payload ~= 5.6 ms)")
    ser.close()


if __name__ == "__main__":
    main()
