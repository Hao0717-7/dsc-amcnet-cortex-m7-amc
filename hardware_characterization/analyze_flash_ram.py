"""
analyze_flash_ram.py
====================
Parse ST Edge AI Core (`stedgeai analyze`) reports and/or Keil MDK map files
and report the model footprint separately from the full-firmware footprint.

Why two numbers?
----------------
`stedgeai analyze` only knows the *model*: weights (ro), activations (rw) and
the ram total it needs. It says nothing about HAL, printf, stacks or the
X-CUBE-AI runtime, so it can never equal the "full firmware" figures that the
paper quotes (51.26 KB Flash / 12.09 KB RAM). The full-firmware numbers must be
read from the linker map of the actually flashed image.

This script therefore reports:

  model (stedgeai analyze) : weights / activations / ram total / MACC / params
  firmware (Keil map)      : Code + RO + RW (ROM) and RW + ZI (RAM)
  paper reference          : printed for context only, *not* used as a pass/fail test

Usage
-----
    # model only (parses an existing analyze report)
    python hardware_characterization/analyze_flash_ram.py \\
        --log docs/evidence/stedgeai_analyze/dscamcnet_int8_analyze_report.txt \\
        --output results/flash_ram_report.json

    # full firmware footprint from the flashed image's map
    python hardware_characterization/analyze_flash_ram.py \\
        --map "D:/DSC_AMCNet_Deployment/MDK-ARM/DSC_AMCNet_Deployment/DSC_AMCNet_Deployment.map" \\
        --output results/flash_ram_report.json

    # run analyze yourself (stedgeai must be on PATH, or pass --stedgeai)
    python hardware_characterization/analyze_flash_ram.py \\
        --onnx quantization/outputs/dscamcnet_int8.onnx --output results/flash_ram_report.json

Paper reference values (do NOT treat as acceptance criteria):
    DSC-AMCNet INT8 : Flash = 51.26 KB, RAM = 12.09 KB   (full firmware, paper)
    ULCNN-simp INT8 : Flash = 49.25 KB, RAM = 36.78 KB
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

# Paper-reported values, printed for context only. Different scope than the
# model-only analyze output, so they are never used as pass/fail thresholds.
PAPER = {
    "dscamcnet": {"flash_kb": 51.26, "ram_kb": 12.09},
    "ulcnn": {"flash_kb": 49.25, "ram_kb": 36.78},
}

# --- stedgeai analyze output patterns -------------------------------------
ANALYZE_WEIGHTS = re.compile(
    r"weights\s*\(ro\)\s*:\s*([\d,]+)\s*B\s*\(([\d.]+)\s*KiB\)", re.IGNORECASE)
ANALYZE_ACT = re.compile(
    r"activations\s*\(rw\)\s*:\s*([\d,]+)\s*B\s*\(([\d.]+)\s*KiB\)", re.IGNORECASE)
ANALYZE_RAM = re.compile(
    r"ram\s*\(total\)\s*:\s*([\d,]+)\s*B\s*\(([\d.]+)\s*KiB\)", re.IGNORECASE)
ANALYZE_MACC = re.compile(r"macc\s*:\s*([\d,]+)", re.IGNORECASE)
ANALYZE_PARAMS = re.compile(r"params\s*#\s*:\s*([\d,]+)\s*items", re.IGNORECASE)

# --- Keil map patterns -----------------------------------------------------
MAP_ROM = re.compile(
    r"Total\s+ROM\s+Size\s*\(Code\s*\+\s*RO\s+Data\s*\+\s*RW\s+Data\)\s*([\d]+)\s*\(\s*([\d.]+)\s*kB\s*\)",
    re.IGNORECASE)
MAP_RAM = re.compile(
    r"Total\s+RW\s+Size\s*\(RW\s+Data\s*\+\s*ZI\s+Data\)\s*([\d]+)\s*\(\s*([\d.]+)\s*kB\s*\)",
    re.IGNORECASE)
MAP_RO = re.compile(
    r"Total\s+RO\s+Size\s*\(Code\s*\+\s*RO\s+Data\)\s*([\d]+)\s*\(\s*([\d.]+)\s*kB\s*\)",
    re.IGNORECASE)


def _num(s: str) -> int:
    return int(s.replace(",", ""))


def parse_analyze(text: str) -> dict:
    out = {}
    for key, pat in (
        ("weights", ANALYZE_WEIGHTS),
        ("activations", ANALYZE_ACT),
        ("ram_total", ANALYZE_RAM),
    ):
        m = pat.search(text)
        if m:
            out[f"{key}_bytes"] = _num(m.group(1))
            out[f"{key}_kb"] = float(m.group(2))
    m = ANALYZE_MACC.search(text)
    if m:
        out["macc"] = _num(m.group(1))
    m = ANALYZE_PARAMS.search(text)
    if m:
        out["params"] = _num(m.group(1))
    if "weights_bytes" not in out and "activations_bytes" not in out:
        raise ValueError("Could not parse weights/activations from the analyze output.")
    return out


def parse_map(text: str) -> dict:
    out = {}
    m = MAP_ROM.search(text)
    if m:
        out["rom_bytes"], out["rom_kb"] = _num(m.group(1)), float(m.group(2))
    m = MAP_RAM.search(text)
    if m:
        out["ram_bytes"], out["ram_kb"] = _num(m.group(1)), float(m.group(2))
    m = MAP_RO.search(text)
    if m:
        out["ro_bytes"], out["ro_kb"] = _num(m.group(1)), float(m.group(2))
    if not out:
        raise ValueError("Could not parse Total ROM/RW Size from the map file.")
    return out


def run_stedgeai(onnx_path: str, target: str, exe: str = "stedgeai") -> str:
    cmd = [exe, "analyze", "--model", str(onnx_path), "--target", target,
           "--type", "onnx", "--no-inputs-allocation", "--no-outputs-allocation",
           "--optimization", "time"]
    print(f"Running: {' '.join(cmd)}")
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except FileNotFoundError:
        print("ERROR: 'stedgeai' not found. Pass --stedgeai PATH to the executable.")
        sys.exit(1)
    except subprocess.TimeoutExpired:
        print("ERROR: stedgeai timed out after 300 s.")
        sys.exit(1)
    if res.returncode != 0:
        print(res.stdout[-2000:])
        print(res.stderr[-2000:])
        print(f"ERROR: stedgeai exited with {res.returncode}.")
        sys.exit(1)
    return res.stdout + res.stderr


def main(args) -> int:
    report = {"model": args.model, "paper_reference": PAPER.get(args.model)}

    if args.onnx:
        report["source"] = str(args.onnx)
        model = parse_analyze(run_stedgeai(args.onnx, args.target, args.stedgeai))
        report["stedgeai_analyze"] = model
    elif args.log:
        report["source"] = str(args.log)
        report["stedgeai_analyze"] = parse_analyze(
            Path(args.log).read_text(encoding="utf-8", errors="replace"))

    if args.map:
        report["map"] = str(args.map)
        report["firmware"] = parse_map(
            Path(args.map).read_text(encoding="utf-8", errors="replace"))

    if "stedgeai_analyze" not in report and "firmware" not in report:
        print("ERROR: nothing to parse — pass --onnx, --log or --map.")
        return 2

    print("\n--- Model footprint (stedgeai analyze) ---")
    a = report.get("stedgeai_analyze")
    if a:
        print(f"  weights      : {a['weights_bytes']:,} B ({a['weights_kb']:.2f} KiB)")
        print(f"  activations  : {a['activations_bytes']:,} B ({a['activations_kb']:.2f} KiB)")
        print(f"  ram (total)  : {a['ram_total_bytes']:,} B ({a['ram_total_kb']:.2f} KiB)")
        if "macc" in a:
            print(f"  MACC         : {a['macc']:,}")
        if "params" in a:
            print(f"  params       : {a['params']:,} items")
    else:
        print("  (not parsed)")

    print("\n--- Full firmware footprint (Keil map) ---")
    fw = report.get("firmware")
    if fw:
        print(f"  ROM (Code+RO+RW) : {fw['rom_bytes']:,} B ({fw['rom_kb']:.2f} KiB)")
        print(f"  RAM (RW+ZI)      : {fw['ram_bytes']:,} B ({fw['ram_kb']:.2f} KiB)")
    else:
        print("  (not parsed — pass --map to include it)")

    pap = report.get("paper_reference")
    if pap:
        print("\n--- Paper reference (context only, different scope) ---")
        print(f"  paper full firmware: Flash {pap['flash_kb']:.2f} KB / RAM {pap['ram_kb']:.2f} KB")
        print("  NOTE: stedgeai analyze only covers the model (weights+activations);")
        print("        HAL/printf/stacks/runtime are not included, so a direct")
        print("        equality check against the paper numbers is not meaningful.")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport saved to {output_path}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse ST Edge AI / Keil map Flash-RAM footprints")
    parser.add_argument("--onnx", help="ONNX model path (runs stedgeai analyze)")
    parser.add_argument("--log", help="Existing stedgeai analyze report file")
    parser.add_argument("--map", help="Keil .map file of the flashed image")
    parser.add_argument("--target", default="stm32")
    parser.add_argument("--stedgeai", default="stedgeai", help="Path to stedgeai executable")
    parser.add_argument("--model", default="dscamcnet", choices=["dscamcnet", "ulcnn"])
    parser.add_argument("--output", default="stm32_deployment/results/flash_ram_report.json")
    sys.exit(main(parser.parse_args()))
