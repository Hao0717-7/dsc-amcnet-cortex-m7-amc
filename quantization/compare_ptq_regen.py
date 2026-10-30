"""
compare_ptq_regen.py
====================
Cross-check that the shipped INT8 artefacts (``quantization/outputs/``) can be
reproduced from the checkpoint by re-running ``ptq_quantize.py``.

Compares, for both FP32 and INT8 ONNX:
  * file size + sha256
  * opset / IR version, node count, initializer count
  * per-initializer byte equality (weights, biases, scales, zero-points)
  * INT8 accuracy on the locked 44,000-window test split vs the shipped model

Run (needs onnx + onnxruntime + numpy):
    D:\\APP\\Pytorch\\Scripts\\python.exe quantization/compare_ptq_regen.py
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "models"))
from train import load_rml2016, load_split  # noqa: E402
from dsc_amcnet import normalize_window  # noqa: E402
import torch  # noqa: E402  (only for normalize_window; CPU, no training)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def summarize(path: Path) -> dict:
    m = onnx.load(str(path), load_external_data=False)
    return {
        "path": path,
        "size": path.stat().st_size,
        "sha256": sha256(path),
        "ir_version": m.ir_version,
        "opset": [(o.domain or "ai.onnx", o.version) for o in m.opset_import],
        "nodes": len(m.graph.node),
        "initializers": len(m.graph.initializer),
    }


def compare_init(a: Path, b: Path) -> tuple[int, int, list[str]]:
    """Returns (#equal, #total, names differing)."""
    ma = onnx.load(str(a), load_external_data=False)
    mb = onnx.load(str(b), load_external_data=False)
    ta = {t.name: onnx.numpy_helper.to_array(t) for t in ma.graph.initializer}
    tb = {t.name: onnx.numpy_helper.to_array(t) for t in mb.graph.initializer}
    names = sorted(set(ta) | set(tb))
    equal, diff = 0, []
    for n in names:
        if n in ta and n in tb and ta[n].shape == tb[n].shape and np.array_equal(ta[n], tb[n]):
            equal += 1
        else:
            diff.append(n)
    return equal, len(names), diff


def eval_int8(path: Path, Xte: np.ndarray) -> dict:
    import onnxruntime as ort
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    logits = np.empty((len(Xte), 11), dtype=np.float32)
    for i in range(len(Xte)):
        logits[i] = sess.run(None, {iname: Xte[i:i + 1]})[0][0]
    return {"logits": logits, "pred": logits.argmax(1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shipped-int8", default=str(ROOT / "quantization/outputs/dscamcnet_int8.onnx"))
    ap.add_argument("--shipped-fp32", default=str(ROOT / "quantization/outputs/dscamcnet_fp32.onnx"))
    ap.add_argument("--regen-dir", default=str(ROOT / ".tmp/ptq_regen"))
    ap.add_argument("--data", default=str(ROOT / "datasets/RML2016.10a_dict.dat"))
    ap.add_argument("--split", default=str(ROOT / "datasets/split_indices.npz"))
    ap.add_argument("--output", default=str(ROOT / "docs/evidence/ptq_regen_log.txt"))
    args = ap.parse_args()

    regen = Path(args.regen_dir)
    pairs = [("FP32", Path(args.shipped_fp32), regen / "dscamcnet_fp32.onnx"),
             ("INT8", Path(args.shipped_int8), regen / "dscamcnet_int8.onnx")]

    lines: list[str] = []
    def say(s: str = "") -> None:
        print(s)
        lines.append(s)

    say("=" * 74)
    say("PTQ reproducibility check — regenerate INT8 ONNX from the checkpoint")
    say("=" * 74)
    say(f"generated     : {datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds')}")
    say(f"command       : ptq_quantize.py --checkpoint models/checkpoints/dscamcnet_best.pt")
    say(f"                (QDQ, signed INT8, 1,024 calibration windows, seed 42)")
    say("")
    for label, shipped, regen_path in pairs:
        if not regen_path.exists():
            say(f"[{label}] regenerated file missing: {regen_path}")
            continue
        a, b = summarize(shipped), summarize(regen_path)
        say(f"[{label}] shipped     : {a['size']:,} B  sha256={a['sha256']}")
        say(f"[{label}] regenerated : {b['size']:,} B  sha256={b['sha256']}")
        say(f"[{label}] identical   : {'YES' if a['sha256'] == b['sha256'] else 'NO'}  "
            f"(IR {a['ir_version']}/{b['ir_version']}, opset {a['opset']}/{b['opset']}, "
            f"nodes {a['nodes']}/{b['nodes']}, initializers {a['initializers']}/{b['initializers']})")
        eq, total, diff = compare_init(shipped, regen_path)
        say(f"[{label}] initializers equal: {eq}/{total}"
            + ("" if not diff else f"   differing: {diff[:6]}"))
        say("")

    say("[INT8 accuracy] locked 44,000-window test split, onnxruntime")
    X, y, snr = load_rml2016(args.data)
    _, test_idx = load_split(args.split)
    Xte = normalize_window(torch.from_numpy(X[test_idx]).float()).numpy()
    yte, snr_te = y[test_idx], snr[test_idx]
    res = {}
    for label, path in (("shipped", Path(args.shipped_int8)), ("regenerated", regen / "dscamcnet_int8.onnx")):
        if not path.exists():
            continue
        r = eval_int8(path, Xte)
        acc = float((r["pred"] == yte).mean())
        hi = float((r["pred"][snr_te == 10] == yte[snr_te == 10]).mean())
        hi10 = float((r["pred"][snr_te >= 10] == yte[snr_te >= 10]).mean())
        res[label] = r
        say(f"  {label:<11s}: overall {acc * 100:.4f}%   +10 dB {hi * 100:.4f}%   "
            f">=10 dB {hi10 * 100:.4f}%   (n={len(yte):,})")
    if "shipped" in res and "regenerated" in res:
        agree = int((res["shipped"]["pred"] == res["regenerated"]["pred"]).sum())
        say(f"  argmax agreement: {agree:,}/{len(yte):,} = {agree / len(yte) * 100:.4f}%")
        say(f"  max |logit diff| : {np.abs(res['shipped']['logits'] - res['regenerated']['logits']).max():.3e}")
    say("")
    say("=" * 74)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwritten: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
