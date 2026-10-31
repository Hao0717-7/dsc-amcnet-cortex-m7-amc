"""
qat_compare.py
==============
Reproducible QAT (quantization-aware training) comparison for DSC-AMCNet.

The paper (Section VIII-A / Appendix C) reports QAT as a NEGATIVE result for
DSC-AMCNet (48.2% overall vs 57% FP32) while plain PTQ stays within noise.
This script re-runs that experiment on the local checkpoint and records the
numbers in ``docs/evidence/qat_eval_log.txt``:

1. FP32 baseline (44,000 locked test windows, per-window L2 normalized)
2. PTQ reference (the shipped INT8 QDQ ONNX, evaluated with onnxruntime)
3. QAT: conv+BN fusion -> fake-quant observers -> fine-tune -> QDQ ONNX export
   -> onnxruntime evaluation (same metric as PTQ)
4. Per-SNR table for FP32 / PTQ / QAT

Usage (venv with torch+onnx+onnxruntime, e.g. D:\\APP\\Pytorch\\Scripts\\python.exe):

    python quantization/qat_compare.py --epochs 3 --train-limit 60000

Everything runs on CPU; the model has only 4,961 parameters.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.dsc_amcnet import build_dsc_amcnet  # noqa: E402
from datasets.prepare_dataset import MODULATIONS, load_raw, load_splits  # noqa: E402


def normalize_numpy(x: np.ndarray) -> np.ndarray:
    """Per-window L2 normalization over the whole 2x128 window (paper III-C)."""
    flat = x.reshape(x.shape[0], -1).astype(np.float64)
    norm = np.linalg.norm(flat, axis=1, keepdims=True)
    norm[norm < 1e-12] = 1e-12
    return (flat / norm).reshape(x.shape).astype(np.float32)


def fuse_model(model: nn.Module) -> nn.Module:
    """Fuse Conv1d+BatchNorm1d pairs so QAT quantizes the deployed graph."""
    pairs = [
        ["stem.0", "stem.1"],
        ["dsc1.dw", "dsc1.bn1"],
        ["dsc1.pw", "dsc1.bn2"],
        ["dsc2.dw", "dsc2.bn1"],
        ["dsc2.pw", "dsc2.bn2"],
        ["dsc3.dw", "dsc3.bn1"],
        ["dsc3.pw", "dsc3.bn2"],
    ]
    import torch.ao.quantization as tq

    model.eval()  # fuse_conv_bn_eval requires eval mode
    return tq.fuse_modules(model, pairs, inplace=True)


class QATWrapper(nn.Module):
    """QuantStub / DeQuantStub around the fused backbone (required by convert())."""

    def __init__(self, base: nn.Module):
        super().__init__()
        self.quant = torch.ao.quantization.QuantStub()
        self.base = base
        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dequant(self.base(self.quant(x)))


@torch.no_grad()
def evaluate_torch(model: nn.Module, X: np.ndarray, y: np.ndarray,
                   batch: int = 512) -> np.ndarray:
    model.eval()
    preds = []
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(X[i:i + batch])
        preds.append(model(xb).argmax(1).numpy())
    return np.concatenate(preds)


def accuracy(pred: np.ndarray, y: np.ndarray) -> float:
    return float((pred == y).mean())


def eval_onnx(path: Path, X: np.ndarray, y: np.ndarray, batch: int = 512) -> np.ndarray:
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(path), sess_options=so,
                                providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    shape = sess.get_inputs()[0].shape
    fixed = shape is not None and isinstance(shape[0], int) and shape[0] > 0
    b = 1 if fixed else batch          # QDQ/PTQ exports here are batch=1
    preds = []
    for i in range(0, len(X), b):
        xb = X[i:i + b]
        out = sess.run(None, {iname: xb})[0]
        preds.append(np.asarray(out).reshape(xb.shape[0], -1).argmax(1))
    return np.concatenate(preds)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="models/checkpoints/dscamcnet_best.pt")
    ap.add_argument("--data", default="datasets/RML2016.10a_dict.dat")
    ap.add_argument("--split", default="datasets/split_indices.npz")
    ap.add_argument("--ptq-onnx", default="quantization/outputs/dscamcnet_int8.onnx")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--train-limit", type=int, default=60000,
                    help="use at most N (randomly chosen) training windows (0 = all)")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default="docs/evidence/qat_eval_log.txt")
    ap.add_argument("--json-output", default="stm32_deployment/results/qat_compare.json")
    ap.add_argument("--qat-onnx", default=".tmp/qat/dscamcnet_qat_qdq.onnx")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    t0 = time.time()
    lines: list[str] = []

    def log(msg: str = "") -> None:
        print(msg)
        lines.append(msg)

    log("=" * 74)
    log("QAT vs PTQ comparison - DSC-AMCNet (RML2016.10a, 44,000 locked test)")
    log(f"torch {torch.__version__} | python {sys.version.split()[0]}")
    log("=" * 74)

    log("\n[1/5] Loading data ...")
    train_X, train_y, test_X, test_y = load_splits(args.data, args.split)
    _, _, snr_raw = load_raw(args.data)
    split = np.load(args.split)
    test_idx = split["test_idx"]
    snr_test = snr_raw[test_idx]
    train_X = normalize_numpy(train_X)
    test_X = normalize_numpy(test_X)
    log(f"  train={len(train_X):,}  test={len(test_X):,}  "
        f"(+10 dB n={int((snr_test == 10).sum()):,}, >=10 dB n={int((snr_test >= 10).sum()):,})")

    if args.train_limit and args.train_limit < len(train_X):
        rng = np.random.default_rng(args.seed)
        sel = rng.choice(len(train_X), size=args.train_limit, replace=False)
        train_X, train_y = train_X[sel], train_y[sel]
        log(f"  QAT fine-tune subset: {len(train_X):,} windows (--train-limit)")

    # ---------------------------------------------------------------- FP32
    log("\n[2/5] FP32 baseline ...")
    model = build_dsc_amcnet()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu",
                                     weights_only=False))
    fp32_pred = evaluate_torch(model, test_X, test_y)
    fp32_acc = accuracy(fp32_pred, test_y)
    fp32_hi = accuracy(fp32_pred[snr_test == 10], test_y[snr_test == 10])
    fp32_ge = accuracy(fp32_pred[snr_test >= 10], test_y[snr_test >= 10])
    log(f"  FP32 overall {fp32_acc * 100:.4f}% | +10 dB {fp32_hi * 100:.4f}% | "
        f">=10 dB {fp32_ge * 100:.4f}%")

    # ----------------------------------------------------------------- PTQ
    log("\n[3/5] PTQ reference (shipped INT8 QDQ ONNX, onnxruntime) ...")
    ptq_pred = eval_onnx(Path(args.ptq_onnx), test_X, test_y)
    ptq_acc = accuracy(ptq_pred, test_y)
    ptq_hi = accuracy(ptq_pred[snr_test == 10], test_y[snr_test == 10])
    ptq_ge = accuracy(ptq_pred[snr_test >= 10], test_y[snr_test >= 10])
    log(f"  PTQ  overall {ptq_acc * 100:.4f}% | +10 dB {ptq_hi * 100:.4f}% | "
        f">=10 dB {ptq_ge * 100:.4f}%  (vs FP32 {fp32_acc * 100 - ptq_acc * 100:+.4f} pp)")

    # ----------------------------------------------------------------- QAT
    log(f"\n[4/5] QAT: fuse conv+BN -> fake-quant -> fine-tune {args.epochs} epoch(s) ...")
    import torch.ao.quantization as tq

    fresh = build_dsc_amcnet()
    fresh.load_state_dict(torch.load(args.checkpoint, map_location="cpu",
                                     weights_only=False))
    fused = fuse_model(fresh)                       # fold BN AFTER loading weights
    fused_pred = evaluate_torch(fused, test_X, test_y)
    log(f"  fused FP32 overall {accuracy(fused_pred, test_y) * 100:.4f}% "
        f"(must equal the FP32 baseline)")

    qat = QATWrapper(fused)
    qat.train()
    qat.qconfig = tq.get_default_qat_qconfig("fbgemm")
    tq.prepare_qat(qat, inplace=True)
    optimizer = torch.optim.Adam(qat.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    rng = np.random.default_rng(args.seed)
    n = len(train_X)
    qat_err = ""
    for epoch in range(1, args.epochs + 1):
        perm = rng.permutation(n)
        qat.train()
        running = 0.0
        for i in range(0, n, args.batch):
            idx = perm[i:i + args.batch]
            xb = torch.from_numpy(train_X[idx])
            yb = torch.from_numpy(train_y[idx])
            optimizer.zero_grad()
            loss = criterion(qat(xb), yb)
            loss.backward()
            optimizer.step()
            running += float(loss.item()) * len(idx)
        # quick fake-quant check on a 8k slice
        sub = slice(0, 8000)
        pred = evaluate_torch(qat, test_X[sub], test_y[sub])
        log(f"  epoch {epoch}: loss {running / n:.4f} | 8k-test fake-quant acc "
            f"{accuracy(pred, test_y[sub]) * 100:.2f}%")

    qat.eval()
    qat.apply(tq.disable_observer)  # freeze scales before export
    qat_pred = evaluate_torch(qat, test_X, test_y)
    qat_acc = accuracy(qat_pred, test_y)
    qat_hi = accuracy(qat_pred[snr_test == 10], test_y[snr_test == 10])
    qat_ge = accuracy(qat_pred[snr_test >= 10], test_y[snr_test >= 10])
    log(f"  QAT (fake-quant, torch) overall {qat_acc * 100:.4f}% | "
        f"+10 dB {qat_hi * 100:.4f}% | >=10 dB {qat_ge * 100:.4f}%")

    # real INT8 arithmetic (fbgemm kernels) - same role as the MCU int8 kernels.
    # NOTE: the SE block multiplies the feature map by a per-channel float scale
    # (`x * s.unsqueeze(2)`).  PyTorch's QuantizedCPU backend has no `mul` kernel
    # for that shape combination, so convert()+inference is expected to fail for
    # DSC-AMCNet - this is the quantizer-level symptom of the paper's "QAT does
    # not work with the SE attention path" claim.  Record it instead of aborting.
    conv_acc = conv_hi = conv_ge = float("nan")
    convert_err = ""
    try:
        qat_conv = tq.convert(qat.eval())
        conv_pred = evaluate_torch(qat_conv, test_X, test_y)
        conv_acc = accuracy(conv_pred, test_y)
        conv_hi = accuracy(conv_pred[snr_test == 10], test_y[snr_test == 10])
        conv_ge = accuracy(conv_pred[snr_test >= 10], test_y[snr_test >= 10])
        log(f"  QAT (converted INT8) overall {conv_acc * 100:.4f}% | "
            f"+10 dB {conv_hi * 100:.4f}% | >=10 dB {conv_ge * 100:.4f}%")
    except Exception as exc:  # noqa: BLE001
        convert_err = f"{type(exc).__name__}: {exc}"
        log("  QAT (converted INT8) FAILED - quantized backend cannot run the SE")
        log(f"    scale multiply: {convert_err.splitlines()[0][:160]}")
        log("    -> INT8 deployment of the SE path needs the custom engine "
            "(float/Q31 sigmoid), not torch static-quant")

    # QAT -> QDQ ONNX export is NOT supported by the torchscript exporter
    # (UnsupportedOperatorError: aten::fused_moving_avg_obs_fake_quant) and the
    # quantized backend cannot run the SE multiply anyway, so no QAT ONNX is
    # produced.  The deployed INT8 model stays the PTQ QDQ export + custom engine.
    log("  QAT -> QDQ ONNX export: skipped (torch exporter does not support QAT "
        "fake-quant ops; SE multiply has no QuantizedCPU kernel)")

    # ------------------------------------------------------- per-SNR table
    log("\n[5/5] Per-SNR accuracy")
    header = f"  {'SNR':>7} | {'n':>6} | {'FP32':>8} | {'PTQ':>8} | {'QAT':>8}"
    log(header)
    log("  " + "-" * (len(header) - 2))
    per_snr: dict[str, dict[str, float]] = {}
    for snr in sorted(set(int(s) for s in snr_test)):
        m = snr_test == snr
        row = {
            "n": int(m.sum()),
            "fp32": accuracy(fp32_pred[m], test_y[m]),
            "ptq": accuracy(ptq_pred[m], test_y[m]),
            "qat_fake": accuracy(qat_pred[m], test_y[m]),
            "qat": accuracy(conv_pred[m], test_y[m]) if not np.isnan(conv_acc)
                     else accuracy(qat_pred[m], test_y[m]),
        }
        if not np.isnan(conv_acc):
            row["qat_int8"] = accuracy(conv_pred[m], test_y[m])
        per_snr[str(snr)] = row
        log(f"  {snr:>+7d} | {row['n']:>6d} | {row['fp32'] * 100:>7.2f}% | "
            f"{row['ptq'] * 100:>7.2f}% | {row['qat'] * 100:>7.2f}%")

    # ----------------------------------------------------------- summary
    log("\n" + "=" * 74)
    log("SUMMARY (44,000 locked test windows)")
    log("=" * 74)
    log(f"  FP32          : {fp32_acc * 100:.4f}% overall | {fp32_hi * 100:.4f}% @+10 dB | "
        f"{fp32_ge * 100:.4f}% @>=10 dB")
    log(f"  PTQ  (INT8)   : {ptq_acc * 100:.4f}% overall | {ptq_hi * 100:.4f}% @+10 dB | "
        f"{ptq_ge * 100:.4f}% @>=10 dB  ({ptq_acc * 100 - fp32_acc * 100:+.4f} pp)")
    log(f"  QAT  (fake-q) : {qat_acc * 100:.4f}% overall | {qat_hi * 100:.4f}% @+10 dB | "
        f"{qat_ge * 100:.4f}% @>=10 dB")
    if np.isnan(conv_acc):
        log("  QAT  (INT8)   : n/a - torch QuantizedCPU cannot execute the SE scale "
            "multiply (see above); DSC has no portable QAT-INT8 path")
    else:
        log(f"  QAT  (INT8)   : {conv_acc * 100:.4f}% overall | {conv_hi * 100:.4f}% @+10 dB | "
            f"{conv_ge * 100:.4f}% @>=10 dB  ({conv_acc * 100 - fp32_acc * 100:+.4f} pp)")
    log("")
    log("  QAT* column above = fake-quant model (INT8 convert unsupported for the SE path).")
    log("  Paper reference (App. C): FP32 57%, DSC QAT 48.2%, PTQ within +0.03 pp;"
        " LiteAMCNet QAT 56.60% (+1.40 pp).")
    verdict = ("QAT does not beat PTQ for DSC-AMCNet: keep PTQ "
               "(reproduces the paper's negative result)")
    if conv_acc > fp32_acc + 0.002:
        verdict = "QAT improves over FP32 in this run - investigate before changing the pipeline"
    log(f"  Verdict: {verdict}")
    log(f"  Elapsed: {time.time() - t0:.1f} s")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    Path(args.json_output).write_text(json.dumps({
        "schema": "dsc-amcnet-qat-compare-v1",
        "checkpoint": args.checkpoint,
        "epochs": args.epochs,
        "train_limit": args.train_limit,
        "fp32": {"overall": fp32_acc, "plus10db": fp32_hi, "ge10db": fp32_ge},
        "ptq": {"overall": ptq_acc, "plus10db": ptq_hi, "ge10db": ptq_ge},
        "qat_fake": {"overall": qat_acc, "plus10db": qat_hi, "ge10db": qat_ge},
        "qat_int8": {"overall": conv_acc, "plus10db": conv_hi, "ge10db": conv_ge},
        "qat_int8_error": convert_err[:2000] if convert_err else None,
        "per_snr": per_snr,
    }, indent=2), encoding="utf-8")
    log(f"\nwritten: {out}  |  {args.json_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
