"""
eval_ulcnn.py
=============
A6: evaluate the upstream ULCNN FP32 ONNX (the comparison model in the paper's
table, reported at 2.104 ms / 1.78x speed-up on the MCU) on the same locked
44,000-window test split as DSC-AMCNet.

Why this script exists
----------------------
The paper's ULCNN row is a *deployment* claim (2.104 ms on STM32H723ZG,
44.26 KB Flash, 36.78 KB RAM).  Upstream only publishes ``ulcnn_fp32_fixed.onnx``
(53,288 B) - no X-CUBE-AI/Keil project, no board harness - so the latency and
RAM numbers cannot be reproduced from the released artefacts.  What can be
checked is the accuracy side and the model footprint; this script does both.

Two gotchas found while writing it (both handled here, both documented):
1. the upstream ONNX declares opset 13 but uses the legacy "axes as second
   input" ReduceMean form that onnxruntime >= 1.18 rejects; the graph is
   repaired in memory (axes -> attribute) and saved next to the original;
2. upstream does not say how inputs are preprocessed.  L2-per-window
   normalisation (what DSC-AMCNet uses) drives this model to near-chance, so
   four plausible variants are swept and the best one is reported explicitly.

Outputs
-------
docs/evidence/ulcnn_eval_log.txt
stm32_deployment/results/ulcnn_eval.json

Usage:  python models/eval_ulcnn.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.prepare_dataset import load_raw, load_splits  # noqa: E402

CLASSES = ["8PSK", "AM-DSB", "AM-SSB", "BPSK", "CPFSK", "GFSK",
           "PAM4", "QAM16", "QAM64", "QPSK", "WBFM"]


def acc(pred: np.ndarray, y: np.ndarray) -> float:
    return float((pred == y).mean())


def variant_inputs(x: np.ndarray) -> dict[str, np.ndarray]:
    f = x.reshape(len(x), -1).astype(np.float64)
    out: dict[str, np.ndarray] = {}
    out["raw"] = x.astype(np.float32)
    norm = np.linalg.norm(f, axis=1, keepdims=True)
    norm[norm < 1e-12] = 1e-12
    out["l2-window"] = (f / norm).reshape(x.shape).astype(np.float32)
    maxabs = np.abs(f).max(1, keepdims=True)
    maxabs[maxabs < 1e-12] = 1e-12
    out["maxabs-window"] = (f / maxabs).reshape(x.shape).astype(np.float32)
    sd = f.std(1, keepdims=True) + 1e-12
    out["zscore-window"] = ((f - f.mean(1, keepdims=True)) / sd).reshape(x.shape).astype(np.float32)
    return out


def repair_axes_input_model(src: Path, dst: Path) -> list[str]:
    """Convert the legacy (opset<=12) ``axes``-as-input form to opset-13 attributes.

    ``ulcnn_fp32_fixed.onnx`` declares opset 13 but stores ReduceMean axes in a
    second int64 input, which onnxruntime rejects as INVALID_GRAPH.  The graph
    is otherwise valid, so rewriting one node is enough.
    """
    import onnx
    from onnx import helper, numpy_helper

    m = onnx.load(str(src))
    inits = {i.name: i for i in m.graph.initializer}
    patched: list[str] = []
    for node in m.graph.node:
        if node.op_type not in ("ReduceMean", "ReduceSum", "Squeeze", "Unsqueeze"):
            continue
        if len(node.input) != 2 or node.input[1] not in inits:
            continue
        axes_name = node.input[1]
        axes = [int(v) for v in numpy_helper.to_array(inits[axes_name]).reshape(-1)]
        if node.op_type in ("Squeeze", "Unsqueeze"):
            continue
        node.input.pop()
        for a in list(node.attribute):
            # opset-13 ReduceMean only has ``axes`` + ``keepdims``;
            # ``noop_with_empty_axes`` first appeared in opset 18.
            if a.name in ("axes", "noop_with_empty_axes"):
                node.attribute.remove(a)
        node.attribute.append(helper.make_attribute("axes", axes))
        patched.append(f"{node.op_type}({node.name or 'unnamed'}) axes={axes}")
        m.graph.initializer.remove(inits[axes_name])
    onnx.save(m, str(dst))
    return patched


def parse_analyze_report(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8", errors="replace")
    footprint: dict[str, object] = {}
    m = re.search(r"params #\s*:\s*([\d,]+)\s*items\s*\(([\d.]+)\s*KiB\)", text)
    if m:
        footprint["params"] = int(m.group(1).replace(",", ""))
        footprint["params_kib"] = float(m.group(2))
    m = re.search(r"macc\s*:\s*([\d,]+)", text)
    if m:
        footprint["macc"] = int(m.group(1).replace(",", ""))
    for key, pat in (("weights_b", r"weights \(ro\)\s*:\s*([\d,]+)\s*B"),
                     ("activations_b", r"activations \(rw\)\s*:\s*([\d,]+)\s*B"),
                     ("ram_total_b", r"ram \(total\)\s*:\s*([\d,]+)\s*B")):
        m = re.search(pat, text)
        if m:
            footprint[key] = int(m.group(1).replace(",", ""))
    return footprint


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default=".tmp/ulcnn/ulcnn_fp32_fixed.onnx")
    ap.add_argument("--data", default="datasets/RML2016.10a_dict.dat")
    ap.add_argument("--split", default="datasets/split_indices.npz")
    ap.add_argument("--output", default="docs/evidence/ulcnn_eval_log.txt")
    ap.add_argument("--json-output", default="stm32_deployment/results/ulcnn_eval.json")
    ap.add_argument("--stedgeai", default=str(
        Path.home() / "STM32Cube/Repository/Packs/STMicroelectronics/X-CUBE-AI/"
        "10.2.1/Utilities/windows/stedgeai.exe"))
    ap.add_argument("--analyze-out", default="docs/evidence/stedgeai_analyze/ulcnn")
    args = ap.parse_args()

    import onnx
    import onnxruntime as ort

    t0 = time.time()
    lines: list[str] = []

    def log(msg: str = "") -> None:
        print(msg)
        lines.append(msg)

    onnx_path = Path(args.onnx)
    log("=" * 78)
    log("A6: upstream ULCNN FP32 ONNX on the locked 44,000-window test split")
    log("=" * 78)
    if not onnx_path.exists():
        log(f"ERROR: {onnx_path} not found - download it from Zenodo record 21167587:")
        log("  curl -x http://127.0.0.1:7897 -o .tmp/ulcnn/ulcnn_fp32_fixed.onnx")
        log("       https://zenodo.org/api/records/21167587/files/ulcnn_fp32_fixed.onnx/content")
        return 2

    m = onnx.load(str(onnx_path))
    ops = [n.op_type for n in m.graph.node]
    init_elems = int(sum(np.prod(t.dims) for t in m.graph.initializer))
    log(f"model: {onnx_path} ({onnx_path.stat().st_size:,} B)")
    log(f"  nodes={len(m.graph.node)} initializer_elements={init_elems:,} "
        f"opset={[o.version for o in m.opset_import]}")
    log(f"  sha256={hashlib.sha256(onnx_path.read_bytes()).hexdigest()}")
    log(f"  op types: {', '.join(sorted(set(ops)))}")
    inp = m.graph.input[0]
    dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
    log(f"  input {inp.name} {dims}; output {m.graph.output[0].name}")

    _, _, test_X, test_y = load_splits(args.data, args.split)
    _, _, snr = load_raw(args.data)
    sp = np.load(args.split)
    snr_test = snr[sp["test_idx"]]
    log(f"\ntest set: {len(test_X):,} windows (canonical sorted split)")
    log("upstream does not document input preprocessing -> sweeping 4 variants")

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    try:
        sess = ort.InferenceSession(str(onnx_path), sess_options=so,
                                    providers=["CPUExecutionProvider"])
    except Exception as exc:  # noqa: BLE001 - upstream uses legacy axes input
        log("")
        log(f"ORT cannot load the upstream file as-is: {type(exc).__name__}")
        log(f"  {str(exc).splitlines()[-1][:400]}")
        repaired = onnx_path.with_name(onnx_path.stem + "_opset13attr.onnx")
        patched = repair_axes_input_model(onnx_path, repaired)
        log(f"  repaired -> {repaired} ({repaired.stat().st_size:,} B)")
        log(f"  patches: {'; '.join(patched) if patched else 'none (unexpected)'}")
        if not patched:
            raise
        sess = ort.InferenceSession(str(repaired), sess_options=so,
                                    providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name

    variants = variant_inputs(np.asarray(test_X, dtype=np.float32))
    results: dict[str, dict[str, float]] = {}
    best_name, best_pred = "", None
    log(f"\n{'variant':<16} | {'overall':>8} | {'+10 dB':>7} | {'>=10 dB':>8}")
    log("  " + "-" * 48)
    for name, data in variants.items():
        preds = [np.asarray(sess.run(None, {iname: data[i:i + 1]})[0]).reshape(-1).argmax()
                 for i in range(len(data))]
        pred = np.asarray(preds)
        overall = acc(pred, test_y)
        hi = acc(pred[snr_test == 10], test_y[snr_test == 10])
        ge = acc(pred[snr_test >= 10], test_y[snr_test >= 10])
        results[name] = {"overall": overall, "plus10db": hi, "ge10db": ge}
        log(f"  {name:<16} | {overall * 100:>7.4f}% | {hi * 100:>6.2f}% | {ge * 100:>7.2f}%")
        if best_pred is None or overall > results[best_name]["overall"]:
            best_name, best_pred = name, pred
    pred = best_pred
    log(f"\nbest variant: {best_name} "
        f"(overall {results[best_name]['overall'] * 100:.4f}%, "
        f"+10 dB {results[best_name]['plus10db'] * 100:.4f}%)")
    log("NOTE: DSC-AMCNet's deployed pipeline uses L2-per-window; feeding that to")
    log("      the released ULCNN drives it to near-chance, so the released model")
    log("      cannot be the one behind the paper's shared-preprocessing table row.")

    per_snr: dict[str, dict[str, float]] = {}
    log("\nper-SNR accuracy (best variant)")
    log(f"  {'SNR':>7} | {'n':>6} | {'ULCNN':>8}")
    for s in sorted(set(int(v) for v in snr_test)):
        mask = snr_test == s
        a = acc(pred[mask], test_y[mask])
        per_snr[str(s)] = {"n": int(mask.sum()), "accuracy": a}
        log(f"  {s:>+7d} | {int(mask.sum()):>6d} | {a * 100:>7.2f}%")

    per_class: dict[str, dict[str, float]] = {}
    log("\nper-class accuracy (best variant)")
    log(f"  {'class':<8} | {'n':>6} | {'acc':>8}")
    for c, name in enumerate(CLASSES):
        mask = test_y == c
        a = acc(pred[mask], test_y[mask])
        per_class[name] = {"n": int(mask.sum()), "accuracy": a}
        log(f"  {name:<8} | {int(mask.sum()):>6d} | {a * 100:>7.2f}%")

    footprint: dict[str, object] = {}
    analyze_report = ""
    stedgeai = Path(args.stedgeai)
    if stedgeai.exists():
        outdir = Path(args.analyze_out)
        outdir.mkdir(parents=True, exist_ok=True)
        cmd = [str(stedgeai), "analyze", "--model", str(onnx_path),
               "--target", "stm32", "--type", "onnx", "--name", "ulcnn_fp32",
               "--no-inputs-allocation", "--no-outputs-allocation",
               "--optimization", "time", "--output", str(outdir)]
        log(f"\nstedgeai analyze -> {outdir}")
        try:
            subprocess.run(cmd, capture_output=True, text=True,
                           timeout=300, stdin=subprocess.DEVNULL)
        except Exception as exc:  # noqa: BLE001
            log(f"  analyze failed: {type(exc).__name__}: {exc}")
        rep = outdir / "ulcnn_fp32_analyze_report.txt"
        if rep.exists():
            analyze_report = str(rep)
            footprint = parse_analyze_report(rep)
            log(f"  params={footprint.get('params')} macc={footprint.get('macc')} "
                f"weights={footprint.get('weights_b')} B "
                f"activations={footprint.get('activations_b')} B "
                f"ram={footprint.get('ram_total_b')} B")
            log("  (footprint is FP32/float; the paper row's 44.26 KB / 36.78 KB is")
            log("   their quantised deployment, which has no released artefacts)")
    else:
        log(f"\nstedgeai not found at {stedgeai} - footprint analysis skipped")

    log("\nREPRODUCIBILITY NOTE")
    log("  Paper row: ULCNN-simplified 2.104 ms / 44.26 KB Flash / 36.78 KB RAM / 1.78x,")
    log("  High-SNR 85.80%.  Upstream releases only this FP32 ONNX (no C project,")
    log("  no board harness), so latency and RAM cannot be reproduced.  Accuracy")
    log(f"  also does not reach 85.80%: best over 4 preprocessing variants is")
    log(f"  {results[best_name]['overall'] * 100:.4f}% overall / "
        f"{results[best_name]['plus10db'] * 100:.4f}% @+10 dB ({best_name} inputs).")
    log("  -> the '1.78x speed-up' row is not verifiable from the released artefacts;")
    log("     it stays in README's 未复现/未验证 table.")
    log(f"\nElapsed: {time.time() - t0:.1f} s")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text("\n".join(lines) + "\n", encoding="utf-8")
    Path(args.json_output).write_text(json.dumps({
        "schema": "dsc-amcnet-ulcnn-eval-v1",
        "onnx": str(onnx_path),
        "onnx_bytes": onnx_path.stat().st_size,
        "onnx_sha256": hashlib.sha256(onnx_path.read_bytes()).hexdigest(),
        "nodes": len(m.graph.node),
        "initializer_elements": init_elems,
        "variants": results,
        "best_variant": best_name,
        "overall": results[best_name]["overall"],
        "plus10db": results[best_name]["plus10db"],
        "ge10db": results[best_name]["ge10db"],
        "per_snr": per_snr,
        "per_class": per_class,
        "footprint_float": footprint,
        "analyze_report": analyze_report,
        "note": ("upstream releases FP32 ONNX only (legacy axes-input graph, repaired); "
                 "no preprocessing documented; board latency/RAM row not reproducible; "
                 "best accuracy over 4 variants stays below the paper's 85.80%"),
    }, indent=2), encoding="utf-8")
    print(f"\nwritten: {args.output} | {args.json_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
