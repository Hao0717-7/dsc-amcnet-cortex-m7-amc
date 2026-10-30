"""
eval_checkpoint.py
==================
FP32 accuracy evaluation of the official PyTorch checkpoint
(``models/checkpoints/dscamcnet_best.pt``) on the locked 44,000-window
RML2016.10a test split -- the artefact required by the upstream
"Reproducing Paper Claims" table (``train.py`` eval log).

This does **not** train anything: it loads the shipped weights, runs the
paper evaluation protocol (per-window L2 normalization, batch 512) and
writes a log to ``docs/evidence/fp32_eval_log.txt``.  It also evaluates the
exported FP32 ONNX (``quantization/outputs/dscamcnet_fp32.onnx``) on the same
windows and reports the agreement, so the two paths can be cross-checked.

Run with the PyTorch environment:
    D:\\APP\\Pytorch\\Scripts\\python.exe models/eval_checkpoint.py

Output columns are the paper convention: ``+10 dB`` = SNR exactly 10 dB,
``>=10 dB`` = SNR in [10, 18] dB.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "models"))

from dsc_amcnet import build_dsc_amcnet, normalize_window  # noqa: E402
from train import load_rml2016, load_split  # noqa: E402

CLASSES = ["8PSK", "AM-DSB", "AM-SSB", "BPSK", "CPFSK",
           "GFSK", "PAM4", "QAM16", "QAM64", "QPSK", "WBFM"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@torch.no_grad()
def eval_torch(model, X: np.ndarray, y: np.ndarray, batch: int = 512):
    """Returns (logits [N,11] int8-not, argmax [N])."""
    model.eval()
    logits = []
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(X[i:i + batch]).float()
        logits.append(model(xb).numpy())
    logits = np.concatenate(logits, axis=0)
    return logits, logits.argmax(axis=1)


def eval_onnx(onnx_path: Path, X: np.ndarray):
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    logits = np.empty((len(X), 11), dtype=np.float32)
    for i in range(len(X)):
        logits[i] = sess.run(None, {iname: X[i:i + 1]})[0][0]
    return logits, logits.argmax(axis=1)


def acc(y, p):
    return float((p == y).mean())


def main() -> int:
    ap = argparse.ArgumentParser(description="FP32 eval of the official checkpoint")
    ap.add_argument("--checkpoint", default=str(ROOT / "models/checkpoints/dscamcnet_best.pt"))
    ap.add_argument("--data", default=str(ROOT / "datasets/RML2016.10a_dict.dat"))
    ap.add_argument("--split", default=str(ROOT / "datasets/split_indices.npz"))
    ap.add_argument("--fp32-onnx", default=str(ROOT / "quantization/outputs/dscamcnet_fp32.onnx"))
    ap.add_argument("--output", default=str(ROOT / "docs/evidence/fp32_eval_log.txt"))
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--no-onnx", action="store_true", help="skip the ONNX cross-check")
    args = ap.parse_args()

    ckpt, data, split = Path(args.checkpoint), Path(args.data), Path(args.split)
    lines: list[str] = []
    def say(s: str = "") -> None:
        print(s)
        lines.append(s)

    say("=" * 74)
    say("DSC-AMCNet — FP32 checkpoint evaluation (official protocol)")
    say("=" * 74)
    say(f"generated      : {datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds')}")
    say(f"torch          : {torch.__version__}  (device=cpu, no training performed)")
    say(f"python         : {sys.version.split()[0]}")
    say(f"checkpoint     : {ckpt.relative_to(ROOT)}  sha256={sha256(ckpt)}  {ckpt.stat().st_size:,} B")
    say(f"dataset        : {data.relative_to(ROOT)}  sha256={sha256(data)}  {data.stat().st_size:,} B")
    say(f"split          : {split.relative_to(ROOT)}  md5={md5(split)}")
    say("")

    X, y, snr = load_rml2016(str(data))
    train_idx, test_idx = load_split(str(split))
    say(f"dataset windows: {len(y):,}   (train {len(train_idx):,} / test {len(test_idx):,})")
    assert sorted(np.concatenate([train_idx, test_idx]).tolist()) == list(range(len(y)))

    model = build_dsc_amcnet()
    n_params = sum(p.numel() for p in model.parameters())
    sd = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    model.load_state_dict(sd, strict=True)
    say(f"parameters     : {n_params:,} (paper: 4,961)")
    say("")

    Xte = normalize_window(torch.from_numpy(X[test_idx]).float()).numpy()
    yte, snr_te = y[test_idx], snr[test_idx]
    logits_t, pred_t = eval_torch(model, Xte, yte, args.batch)
    say(f"test windows   : {len(yte):,} (locked split, L2-normalized per window)")
    say("")
    say("[PyTorch checkpoint]")
    say(f"  overall accuracy : {acc(yte, pred_t) * 100:.4f}%  ({int((pred_t == yte).sum()):,}/{len(yte):,})")
    say(f"  +10 dB           : {acc(yte[snr_te == 10], pred_t[snr_te == 10]) * 100:.4f}%  (n={(snr_te == 10).sum():,})")
    say(f"  >=10 dB          : {acc(yte[snr_te >= 10], pred_t[snr_te >= 10]) * 100:.4f}%  (n={(snr_te >= 10).sum():,})")
    say("")
    say("  per-SNR accuracy:")
    for s in range(-20, 19, 2):
        m = snr_te == s
        if m.any():
            say(f"    SNR {s:+3d} dB : {acc(yte[m], pred_t[m]) * 100:6.2f}%  (n={int(m.sum()):,})")
    say("")
    cm = np.zeros((11, 11), dtype=int)
    for t, p in zip(yte, pred_t):
        cm[t, p] += 1
    say("  confusion matrix (rows=true, cols=pred):")
    say("    " + " " * 8 + " ".join(f"{c:>7s}" for c in CLASSES))
    for i, c in enumerate(CLASSES):
        say(f"    {c:>8s}" + "".join(f"{v:7d}" for v in cm[i]))
    say("")

    if not args.no_onnx and Path(args.fp32_onnx).exists():
        fp32 = Path(args.fp32_onnx)
        say(f"[FP32 ONNX] {fp32.relative_to(ROOT)}  {fp32.stat().st_size:,} B  "
            f"sha256={sha256(fp32)}")
        logits_o, pred_o = eval_onnx(fp32, Xte)
        say(f"  overall accuracy : {acc(yte, pred_o) * 100:.4f}%")
        say(f"  +10 dB           : {acc(yte[snr_te == 10], pred_o[snr_te == 10]) * 100:.4f}%")
        say(f"  >=10 dB          : {acc(yte[snr_te >= 10], pred_o[snr_te >= 10]) * 100:.4f}%")
        agree = int((pred_t == pred_o).sum())
        say(f"  argmax agreement : {agree:,}/{len(yte):,} = {agree / len(yte) * 100:.4f}%")
        say(f"  max |logit diff| : {np.abs(logits_t - logits_o).max():.3e}")
    else:
        say("[FP32 ONNX] skipped")
    say("")
    say("protocol: RML2016.10a locked split md5 bb8bc1e7c41a58fad6f08a23ae7ef76d,")
    say("          per-window L2 normalization over the full 2x128 window,")
    say("          batch=512, no augmentation, weights = local retrain f7a05c25…")
    say("=" * 74)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwritten: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
