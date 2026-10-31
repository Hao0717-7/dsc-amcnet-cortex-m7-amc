"""
ptq_per_channel.py
==================
A5: per-channel (per-output-channel) weight quantization vs the shipped
per-tensor PTQ, evaluated on the 44,000 locked test windows with onnxruntime.

onnxruntime.quantization.quantize_static(..., per_channel=True) stores one
scale per output channel instead of one per tensor.  For DSC-AMCNet the
weights are tiny (4.5 KB int8), so the extra scale metadata is a real cost on
an MCU; this script measures both accuracy and file-size impact before any
decision to switch the deployment.

Outputs
-------
docs/evidence/ptq_per_channel_log.txt      human-readable comparison
stm32_deployment/results/ptq_per_channel.json  machine-readable summary
.tmp/per_channel/dscamcnet_int8_per_channel.onnx  the candidate model

Usage:  python quantization/ptq_per_channel.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.prepare_dataset import load_raw, load_splits  # noqa: E402
from quantization.ptq_quantize import RMLCalibReader  # noqa: E402


def normalize_numpy(x: np.ndarray) -> np.ndarray:
    flat = x.reshape(x.shape[0], -1).astype(np.float64)
    norm = np.linalg.norm(flat, axis=1, keepdims=True)
    norm[norm < 1e-12] = 1e-12
    return (flat / norm).reshape(x.shape).astype(np.float32)


def make_session(path: Path):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), sess_options=so,
                                providers=["CPUExecutionProvider"])


def eval_onnx(sess, X: np.ndarray, batch: int = 256) -> np.ndarray:
    iname = sess.get_inputs()[0].name
    shape = sess.get_inputs()[0].shape
    fixed = shape is not None and isinstance(shape[0], int) and shape[0] > 0
    b = 1 if fixed else batch
    preds = []
    for i in range(0, len(X), b):
        xb = X[i:i + b]
        out = sess.run(None, {iname: xb})[0]
        preds.append(np.asarray(out).reshape(xb.shape[0], -1).argmax(1))
    return np.concatenate(preds)


def acc(pred: np.ndarray, y: np.ndarray) -> float:
    return float((pred == y).mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp32", default="quantization/outputs/dscamcnet_fp32.onnx")
    ap.add_argument("--ptq", default="quantization/outputs/dscamcnet_int8.onnx")
    ap.add_argument("--data", default="datasets/RML2016.10a_dict.dat")
    ap.add_argument("--split", default="datasets/split_indices.npz")
    ap.add_argument("--calib", type=int, default=1024)
    ap.add_argument("--out-onnx", default=".tmp/per_channel/dscamcnet_int8_per_channel.onnx")
    ap.add_argument("--output", default="docs/evidence/ptq_per_channel_log.txt")
    ap.add_argument("--json-output", default="stm32_deployment/results/ptq_per_channel.json")
    args = ap.parse_args()

    from onnxruntime.quantization import QuantFormat, QuantType, quantize_static

    t0 = time.time()
    lines: list[str] = []

    def log(msg: str = "") -> None:
        print(msg)
        lines.append(msg)

    log("=" * 78)
    log("A5: per-channel vs per-tensor INT8 PTQ - DSC-AMCNet (44,000 locked test)")
    log("=" * 78)

    train_X, train_y, test_X, test_y = load_splits(args.data, args.split)
    _, _, snr = load_raw(args.data)
    sp = np.load(args.split)
    snr_test = snr[sp["test_idx"]]
    test_X = normalize_numpy(test_X)
    log(f"test={len(test_X):,} (+10 dB n={int((snr_test == 10).sum()):,}, "
        f">=10 dB n={int((snr_test >= 10).sum()):,})")

    out = Path(args.out_onnx)
    out.parent.mkdir(parents=True, exist_ok=True)
    reader = RMLCalibReader(args.data, args.split, n_calib=args.calib)
    log(f"quantizing with per_channel=True ({args.calib} calibration windows) ...")
    quantize_static(
        args.fp32,
        str(out),
        reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
    )

    def run_and_score(path: Path) -> tuple[np.ndarray, float, float, float, int]:
        sess = make_session(path)
        pred = eval_onnx(sess, test_X)
        return (pred,
                acc(pred, test_y),
                acc(pred[snr_test == 10], test_y[snr_test == 10]),
                acc(pred[snr_test >= 10], test_y[snr_test >= 10]),
                path.stat().st_size)

    log("\nevaluating ...")
    pt = Path(args.ptq)
    base_pred, base_acc, base_hi, base_ge, base_bytes = run_and_score(pt)
    pc_pred, pc_acc, pc_hi, pc_ge, pc_bytes = run_and_score(out)
    log(f"  per-tensor  (shipped): {base_acc * 100:.4f}% | +10 dB {base_hi * 100:.4f}% | "
        f">=10 dB {base_ge * 100:.4f}% | {base_bytes:,} B")
    log(f"  per-channel          : {pc_acc * 100:.4f}% | +10 dB {pc_hi * 100:.4f}% | "
        f">=10 dB {pc_ge * 100:.4f}% | {pc_bytes:,} B  ({(pc_bytes - base_bytes):+,} B)")
    log(f"  delta                : {(pc_acc - base_acc) * 100:+.4f} pp overall | "
        f"{(pc_hi - base_hi) * 100:+.4f} pp @+10 dB | {(pc_ge - base_ge) * 100:+.4f} pp @>=10 dB")
    agree = float((pc_pred == base_pred).mean())
    log(f"  argmax agreement per-channel vs per-tensor: {agree * 100:.4f}%")

    log("\nper-SNR accuracy")
    log(f"  {'SNR':>7} | {'n':>6} | {'per-tensor':>10} | {'per-channel':>11} | {'delta':>8}")
    log("  " + "-" * 54)
    per_snr: dict[str, dict[str, float]] = {}
    for s in sorted(set(int(v) for v in snr_test)):
        m = snr_test == s
        a, b = acc(base_pred[m], test_y[m]), acc(pc_pred[m], test_y[m])
        per_snr[str(s)] = {"n": int(m.sum()), "per_tensor": a, "per_channel": b}
        log(f"  {s:>+7d} | {int(m.sum()):>6d} | {a * 100:>9.2f}% | {b * 100:>10.2f}% | "
            f"{(b - a) * 100:>+7.2f} pp")

    # where do the two models disagree?
    diff = pc_pred != base_pred
    log(f"\n  frames with different argmax: {int(diff.sum()):,} / {len(diff):,} "
        f"({diff.mean() * 100:.3f}%)")
    correct_pc = int((pc_pred[diff] == test_y[diff]).sum())
    correct_base = int((base_pred[diff] == test_y[diff]).sum())
    log(f"  on those frames: per-channel right {correct_pc}, per-tensor right {correct_base}")

    delta_pp = (pc_acc - base_acc) * 100
    verdict = ("per-channel gives no accuracy gain -> keep per-tensor "
               "(weights are only ~4.5 KB; extra scales are pure metadata cost)")
    if delta_pp >= 0.2:
        verdict = ("per-channel gains >= 0.2 pp -> candidate for the next firmware "
                   "(needs X-CUBE-AI re-conversion + board re-measurement)")
    elif delta_pp <= -0.2:
        verdict = "per-channel is worse -> keep per-tensor"
    log(f"\nVERDICT: {verdict}")
    log(f"Elapsed: {time.time() - t0:.1f} s | model: {out}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text("\n".join(lines) + "\n", encoding="utf-8")
    Path(args.json_output).write_text(json.dumps({
        "schema": "dsc-amcnet-ptq-per-channel-v1",
        "calib_windows": args.calib,
        "per_tensor": {"model": str(pt), "bytes": base_bytes, "overall": base_acc,
                       "plus10db": base_hi, "ge10db": base_ge},
        "per_channel": {"model": str(out), "bytes": pc_bytes, "overall": pc_acc,
                        "plus10db": pc_hi, "ge10db": pc_ge},
        "delta_pp": {"overall": delta_pp, "plus10db": (pc_hi - base_hi) * 100,
                     "ge10db": (pc_ge - base_ge) * 100},
        "argmax_agreement": agree,
        "verdict": verdict,
        "per_snr": per_snr,
    }, indent=2), encoding="utf-8")
    print(f"\nwritten: {args.output} | {args.json_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
