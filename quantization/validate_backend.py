"""
validate_backend.py
===================
Validates that the fbgemm backend (signed INT8) must be used for
X-CUBE-AI compatibility, and demonstrates the silent failure mode
caused by using the qnnpack backend (unsigned INT8).

Background
----------
X-CUBE-AI 10.2.0 expects signed INT8 weights and activations, consistent
with the ARM CMSIS-NN kernels used internally. PyTorch's 'fbgemm' backend
produces signed INT8 (range −128 to +127). PyTorch's 'qnnpack' backend
produces unsigned INT8 (range 0 to 255), which causes a silent scale/zero-
point mismatch on device — the model runs without an error, but outputs
garbage classifications.

This failure mode is documented in the paper (Section V-C) as a key
engineering lesson. It was discovered during hardware bring-up when the
device produced random classification outputs despite successful ONNX import.

This script
-----------
Quantizes DSC-AMCNet with both backends and compares:
1. Quantized weight dtype and zero-point range
2. On-host INT8 accuracy (both should be similar on host)
3. X-CUBE-AI ONNX compatibility (fbgemm passes, qnnpack fails on device)

Usage
-----
    python quantization/validate_backend.py \\
        --checkpoint models/checkpoints/dscamcnet_best.pt \\
        --data       datasets/RML2016.10a_dict.dat \\
        --split      datasets/split_indices.npz
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.quantization

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.dsc_amcnet import build_dsc_amcnet, normalize_window
from datasets.prepare_dataset import load_raw, load_splits

CALIB_N = 500


def quantize_with_backend(checkpoint_path, data_path, split_path, backend):
    """Quantize model with specified backend and return (model, accuracy)."""
    model = build_dsc_amcnet()
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()

    model.qconfig = torch.quantization.get_default_qconfig(backend)
    torch.quantization.prepare(model, inplace=True)

    # Calibrate
    train_X, _, _, _ = load_splits(data_path, split_path)
    rng = np.random.default_rng(42)
    idx = rng.choice(len(train_X), size=CALIB_N, replace=False)
    with torch.no_grad():
        for i in range(0, CALIB_N, 64):
            xb = torch.from_numpy(train_X[idx[i:i+64]]).float()
            model(xb)

    torch.quantization.convert(model, inplace=True)

    # Evaluate at +10 dB (paper Table II high-SNR bin, the same filter
    # models/train.py uses), on the preprocessed input the model expects.
    X_hi, y_hi, snr_hi = load_raw(data_path)
    test_idx = np.load(split_path)["test_idx"]
    mask = snr_hi[test_idx] == 10
    X_hi = normalize_window(
        torch.from_numpy(X_hi[test_idx][mask]).float()
    ).numpy()
    y_hi = y_hi[test_idx][mask]
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.from_numpy(X_hi).float(), torch.from_numpy(y_hi).long()
        ),
        batch_size=256,
    )
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            correct += (model(xb).argmax(1) == yb).sum().item()
            total += len(yb)
    return model, correct / total


def inspect_zero_point(model):
    """Check zero-point of first quantized conv layer to identify signed vs unsigned."""
    for name, module in model.named_modules():
        if hasattr(module, "weight") and hasattr(module.weight, "q_zero_point"):
            zp = module.weight.q_zero_point()
            dtype = module.weight.dtype
            return name, dtype, zp
    return None, None, None


def check_onnx_artifact(path: str, fp32_path: str, data_path: str,
                        split_path: str, snr_db: int = 10) -> int:
    """Check the exported INT8 ONNX, the artifact X-CUBE-AI actually consumes.

    Verifies (a) weights quantized as signed int8 with zero-point 0,
    (b) every Q/DQ zero-point tensor is int8 rather than uint8, and
    (c) INT8 accuracy still tracks FP32 at +10 dB.

    Used when PyTorch's eager quantized kernels are unavailable.
    """
    import collections

    import onnx
    from onnx import numpy_helper

    if not Path(path).exists():
        print(f"  No ONNX artifact at {path}; run ptq_quantize.py first.")
        return 1

    model = onnx.load(path)
    kinds = collections.Counter(n.op_type for n in model.graph.node)
    qdq = kinds["QuantizeLinear"] + kinds["DequantizeLinear"]

    inits = {i.name: i for i in model.graph.initializer}
    weight_zps = [int(v) for i in model.graph.initializer
                  if ".weight_zero_point" in i.name
                  for v in numpy_helper.to_array(i).ravel()]
    zp_dtypes = set()
    for node in model.graph.node:
        if node.op_type not in ("QuantizeLinear", "DequantizeLinear"):
            continue
        for name in node.input[1:3]:
            # bias zero-points (int32, ONNX spec) are not activation quantizers
            if not name.endswith("zero_point") or "_quantized_zero_point" in name:
                continue
            init = inits.get(name)
            if init is not None:
                zp_dtypes.add(numpy_helper.to_array(init).dtype.name)

    print(f"  Artifact   : {path}")
    print(f"  IR / opset : {model.ir_version} / {model.opset_import[0].version}")
    print(f"  QDQ nodes  : {qdq}")
    print(f"  Conv nodes : {kinds['Conv']}")

    signed_weights = bool(weight_zps) and all(zp == 0 for zp in weight_zps)
    all_int8 = zp_dtypes == {"int8"}

    print(f"  Weight zero-points : {sorted(set(weight_zps))} -> "
          f"{'signed int8 (0) ✓' if signed_weights else 'UNSIGNED ✗'}")
    print(f"  Q/DQ zero-point dtype(s): {sorted(zp_dtypes)} -> "
          f"{'all int8 ✓' if all_int8 else 'uint8 present ✗'}")
    print(f"  IR version <= 8 (X-CUBE-AI 10.2): "
          f"{'yes ✓' if model.ir_version <= 8 else 'NO ✗'}")

    acc_ok, acc_line = True, "not measured"
    if Path(fp32_path).exists():
        acc_ok, acc_line = _compare_accuracy(path, fp32_path, data_path,
                                             split_path, snr_db)
        print(f"  {acc_line}")

    ok = signed_weights and all_int8 and model.ir_version <= 8 and qdq > 0 and acc_ok
    print(f"\n  VERDICT: {'signed-int8 QDQ artifact -> X-CUBE-AI COMPATIBLE ✓' if ok else 'not deployment-ready ✗'}")
    return 0 if ok else 1


def _compare_accuracy(int8_path: str, fp32_path: str, data_path: str,
                      split_path: str, snr_db: int) -> tuple:
    """INT8 vs FP32 accuracy at one SNR bin, on preprocessed windows."""
    import onnxruntime as ort

    X, y, snr = load_raw(data_path)
    test_idx = np.load(split_path)["test_idx"]
    mask = snr[test_idx] == snr_db
    idx = test_idx[mask]
    xs = normalize_window(torch.from_numpy(X[idx]).float()).numpy()
    ys = y[idx]

    def acc(p):
        sess = ort.InferenceSession(p, providers=["CPUExecutionProvider"])
        name = sess.get_inputs()[0].name
        hit = 0
        for i in range(len(idx)):
            hit += (sess.run(None, {name: xs[i:i + 1]})[0].argmax(1) == ys[i]).sum()
        return hit / len(idx)

    a_i8, a_fp32 = acc(int8_path), acc(fp32_path)
    delta = a_i8 - a_fp32
    line = (f"  +{snr_db} dB accuracy (n={len(idx)}): "
            f"FP32 {a_fp32*100:.2f}%  INT8 {a_i8*100:.2f}%  Δ {delta*100:+.2f} pp")
    return abs(delta) < 0.02, line


def main(args):
    print("=" * 65)
    print("Backend Validation: fbgemm (signed INT8) vs qnnpack (unsigned INT8)")
    print("=" * 65)

    backends = [b for b in ("fbgemm", "qnnpack")
                if b in torch.backends.quantized.supported_engines]

    if not backends:
        print("\nEager-mode quantization skipped: this PyTorch build registers no "
              "fbgemm/qnnpack\nkernel (supported engines: "
              f"{torch.backends.quantized.supported_engines}).\n"
              "On such a host, validate the exported ONNX artifact instead:\n")
        return check_onnx_artifact(args.onnx, args.fp32, args.data, args.split)

    results = {}
    for backend in backends:
        print(f"\n[{backend}] Quantizing ...")
        model, acc = quantize_with_backend(args.checkpoint, args.data, args.split, backend)
        layer, dtype, zp = inspect_zero_point(model)
        results[backend] = {"acc": acc, "layer": layer, "dtype": dtype, "zp": zp}
        print(f"  +10 dB accuracy: {acc*100:.2f}%")
        print(f"  First layer: {layer}")
        print(f"  Weight dtype: {dtype}")
        print(f"  Zero-point: {zp}  ({'signed (0=correct for symmetric)' if zp == 0 else 'unsigned — incompatible with X-CUBE-AI'})")

    print("\n" + "=" * 65)
    print("VERDICT")
    print("=" * 65)
    for backend in backends:
        r = results[backend]
        verdict = ("Signed INT8 → X-CUBE-AI COMPATIBLE ✓" if r["zp"] == 0 else
                   "Unsigned INT8 → X-CUBE-AI INCOMPATIBLE ✗")
        print(f"\n  {backend:<8} → zero-point={r['zp']}, dtype={r['dtype']}")
        print(f"            {verdict}")
        if r["zp"] != 0:
            print(f"            On-host accuracy {r['acc']*100:.2f}% looks fine,")
            print("            but ON DEVICE this produces random outputs (silent failure).")
    if "fbgemm" not in backends:
        print("\n  NOTE: fbgemm kernels are not registered in this PyTorch build;")
        print("        run check_onnx_artifact() / --onnx to validate the exported")
        print("        artifact instead.")
    print("\n  ALWAYS use signed INT8 (zero-point 0) for X-CUBE-AI deployment.")
    print("  This failure mode is documented in paper Section V-C.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data",       required=True)
    parser.add_argument("--split",      required=True)
    parser.add_argument("--onnx",
                        default="quantization/outputs/dscamcnet_int8.onnx",
                        help="INT8 ONNX artifact to check when eager-mode "
                             "quantized kernels are unavailable")
    parser.add_argument("--fp32",
                        default="quantization/outputs/dscamcnet_fp32.onnx",
                        help="FP32 ONNX reference for the accuracy delta")
    main(parser.parse_args())
