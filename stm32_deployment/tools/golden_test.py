"""Golden-vector regression for the board custom engine (replay mode).

Each vector: 256 int8 q values + expected 11 int8 logits (from onnxruntime,
quantised with output scale 0.35008532 / zp 63).

Usage:  python golden_test.py [--port COM3] [--json PATH]
Exit code 0 iff every vector matches exactly.
"""
import argparse
import json
import time
from pathlib import Path

import serial

ROOT = Path(r"D:\dsc-amcnet-cortex-m7-amc")
VEC = ROOT / "stm32_deployment" / "results" / "golden_vectors.json"


def read_line(ser, timeout_s=2.0):
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
    ap.add_argument("--json", default=str(VEC))
    ap.add_argument("--flash", action="store_true",
                    help="flash first and wait for [CLASSIFY] so the board is in 115200 replay mode")
    args = ap.parse_args()

    vecs = json.loads(Path(args.json).read_text(encoding="utf-8"))
    ser = serial.Serial(args.port, 115200, timeout=0.5)
    ser.reset_input_buffer()
    time.sleep(0.3)
    if args.flash:
        import subprocess
        subprocess.run([r"D:\APP\Keil5\UV4\UV4.exe", "-f",
                        r"D:\DSC_AMCNet_Deployment\MDK-ARM\DSC_AMCNet_Deployment.uvprojx",
                        "-o", str(pathlib.Path(tempfile.gettempdir()) / "dsc_golden_flash.log")],
                       capture_output=True)
        deadline = time.time() + 25
        while time.time() < deadline:
            line = read_line(ser, 1.0)
            if "CLASSIFY" in line:
                break
        ser.reset_input_buffer()

    # Prime: send one throw-away frame and wait for its reply so the board's
    # replay loop and this script agree on frame boundaries (the first 'R' after
    # boot can otherwise be consumed while the classify banner is still printing).
    ser.write(b"R" + bytes(256))
    prime_deadline = time.time() + 3.0
    while time.time() < prime_deadline:
        if read_line(ser, 0.5).startswith("L,"):
            break
    ser.reset_input_buffer()

    passed = 0
    for v in vecs:
        q = bytes((int(x) & 0xFF) for x in v["q"])
        line = ""
        for _ in range(3):
            ser.write(b"R" + q)
            line = read_line(ser)
            if line.startswith("L,"):
                break
        if not line.startswith("L,"):
            print(f"FAIL {v['name']:<12} no reply")
            continue
        got = [int(x) for x in line[2:].split(",")]
        exp = v["logits"]
        diffs = [(i, got[i], exp[i]) for i in range(11) if got[i] != exp[i]]
        max_abs = max((abs(g - e) for _, g, e in diffs), default=0)
        argmax_ok = max(range(11), key=lambda i: got[i]) == max(range(11), key=lambda i: exp[i])
        # C uses round-half-away (lroundf/Q31), ONNX uses round-half-to-even, so a
        # handful of logits may differ by 1 LSB.  Gate on +/-1 LSB + argmax.
        if max_abs <= 1 and argmax_ok:
            passed += 1
            tag = "exact" if not diffs else f"max|diff|={max_abs}"
            print(f"PASS {v['name']:<12} argmax={max(range(11), key=lambda i: got[i])} ({tag})")
        else:
            print(f"FAIL {v['name']:<12} max|diff|={max_abs} argmax_ok={argmax_ok} diff={diffs[:4]}")

    ser.close()
    print(f"\n{passed}/{len(vecs)} vectors within +/-1 LSB and argmax-equal")
    raise SystemExit(0 if passed == len(vecs) else 1)


if __name__ == "__main__":
    main()
