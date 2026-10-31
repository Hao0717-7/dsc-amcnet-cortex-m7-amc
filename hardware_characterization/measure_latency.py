"""
measure_latency.py
==================
Reads DWT latency reports from the STM32H723ZG over UART and logs them to CSV.

Parses the benchmark output produced by DWT_BenchmarkN() in dwt_measure.c:

    [DWT] N=1000  mean=0.951 ms  min=0.951 ms  max=0.951 ms   (all-zero input)
    [DWT] N=1000  mean=0.960 ms  min=0.960 ms  max=0.962 ms   (real window)

The board only prints those lines once, after reset (main() runs the benchmark
before entering the classify loop). Trigger a reset/reflash while this script
is listening, e.g.:

    "D:/APP/Keil5/UV4/UV4.exe" -f "D:/DSC_AMCNet_Deployment/MDK-ARM/DSC_AMCNet_Deployment.uvprojx" -o flash.log

Usage
-----
    python hardware_characterization/measure_latency.py \\
        --port COM3 --baud 115200 \\
        --output docs/evidence/latency_dscamcnet_int8.csv

Note on encoding: the serial stream is UTF-8, but the Windows console may be
GBK. Lines are sanitized before printing so that non-ASCII bytes never raise
UnicodeEncodeError; the CSV only contains ASCII numbers.

Paper result (for reference, different hardware/clock):
    DSC-AMCNet INT8 : mean = 1.183 ms @ 550 MHz (paper) / measured here 0.895 / 0.904 ms @ 550 MHz
                      (C1 kernel at 544 MHz was 0.905 / 0.914 ms)
    ULCNN-simp INT8 : mean = 2.104 ms @ 550 MHz
"""

import argparse
import csv
import re
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    import serial
except ImportError:
    raise ImportError("Install pyserial: pip install pyserial")


# DWT output pattern from DWT_BenchmarkN()
DWT_PATTERN = re.compile(
    r"\[DWT\]\s+N=(\d+)\s+mean=([\d.]+)\s*ms\s+min=([\d.]+)\s*ms\s+max=([\d.]+)\s*ms"
)


def safe_print(text: str) -> None:
    """Print text without ever raising UnicodeEncodeError on a GBK console."""
    enc = sys.stdout.encoding or "utf-8"
    try:
        print(text.encode(enc, errors="replace").decode(enc, errors="replace"))
    except Exception:
        print(text.encode("ascii", errors="replace").decode("ascii"))


def parse_dwt_line(line: str):
    """Return (n, mean_ms, min_ms, max_ms) or None if the line doesn't match."""
    m = DWT_PATTERN.search(line)
    if m:
        return int(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4))
    return None


def read_and_log(port: str, baud: int, output: str, timeout: float = 60.0):
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    safe_print(f"Connecting to {port} @ {baud} baud ...")
    with serial.Serial(port, baud, timeout=1) as ser:
        safe_print(f"Connected. Logging to {output_path}")
        safe_print("Waiting for DWT output ... trigger a reset/reflash now (Ctrl+C to stop)\n")

        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "n_inferences", "mean_ms", "min_ms", "max_ms"])

            start = time.time()
            while time.time() - start < timeout:
                try:
                    raw = ser.readline()
                    line = raw.decode("utf-8", errors="replace").strip()
                except Exception:
                    continue

                if line:
                    safe_print(f"  {line}")

                result = parse_dwt_line(line)
                if result:
                    n, mean_ms, min_ms, max_ms = result
                    ts = datetime.now().isoformat()
                    writer.writerow([ts, n, mean_ms, min_ms, max_ms])
                    f.flush()
                    safe_print(f"\n  Recorded: mean={mean_ms:.3f} ms | min={min_ms:.3f} ms | max={max_ms:.3f} ms\n")

    safe_print(f"\nLogging complete. Results saved to {output_path}")
    summarize(output_path)


def summarize(csv_path: Path):
    import statistics
    means = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            means.append(float(row["mean_ms"]))

    if not means:
        safe_print("No DWT measurements recorded.")
        return

    safe_print("\n" + "=" * 50)
    safe_print("LATENCY SUMMARY")
    safe_print("=" * 50)
    safe_print(f"  Measurements   : {len(means)}")
    safe_print(f"  Overall mean   : {statistics.mean(means):.3f} ms")
    if len(means) > 1:
        safe_print(f"  Std dev        : {statistics.stdev(means):.4f} ms")
    safe_print("\n  Paper reference : 1.183 ms (DSC-AMCNet INT8 @ 550 MHz)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="STM32 DWT Latency Logger")
    parser.add_argument("--port", required=True, help="Serial port (e.g. COM3 or /dev/ttyACM0)")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--output", default="docs/evidence/latency_dscamcnet_int8.csv")
    parser.add_argument("--timeout", type=float, default=120.0, help="Stop after this many seconds")
    args = parser.parse_args()
    read_and_log(args.port, args.baud, args.output, args.timeout)
