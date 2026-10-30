# -*- coding: utf-8 -*-
"""实测脚本：把仓库里能离线复算的东西全部重算一遍，输出 measured_latest.json。

覆盖：
  1) 数据集完整性 / Q 通道 / split_indices 校验和 / 是否 stratified
  2) checkpoint 哈希与 ONNX 参数量
  3) FP32 ONNX 在 44k 测试集上的准确率（全 SNR / +10 dB / 逐 SNR）
  4) INT8 ONNX 在 44k 测试集上的准确率，并与 results/ref_accuracy.json 对比
  5) 用混淆矩阵重算 results/p5_accuracy_44k.json 的准确率（校验板端 JSON 自洽）
  6) 从部署工程 map 解析资源占用（ROM / RAM）
  7) 从 evidence 日志解析延迟 / golden / stream 关键数字

只读输入、只写一个新文件 results/measured_latest.json。
"""
import hashlib
import json
import pickle
import re
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

ROOT = Path(r"D:\dsc-amcnet-cortex-m7-amc")
DATA = ROOT / "datasets" / "RML2016.10a_dict.dat"
SPLIT = ROOT / "datasets" / "split_indices.npz"
CKPT = ROOT / "models" / "checkpoints" / "dscamcnet_best.pt"
ONNX_FP32 = ROOT / "quantization" / "outputs" / "dscamcnet_fp32.onnx"
ONNX_INT8 = ROOT / "quantization" / "outputs" / "dscamcnet_int8.onnx"
RES = ROOT / "stm32_deployment" / "results"
EV = RES / "evidence"
MAP = Path(r"D:\DSC_AMCNet_Deployment\MDK-ARM\DSC_AMCNet_Deployment.map")
OUT = RES / "measured_latest.json"

CLASSES = ["8PSK", "AM-DSB", "AM-SSB", "BPSK", "CPFSK", "GFSK",
           "PAM4", "QAM16", "QAM64", "QPSK", "WBFM"]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_rml():
    t0 = time.time()
    with open(DATA, "rb") as f:
        d = pickle.load(f, encoding="latin1")
    keys = sorted(d.keys(), key=lambda k: (k[0], k[1]))
    X, y, snr = [], [], []
    c2i = {c: i for i, c in enumerate(sorted({k[0] for k in keys}))}
    for (mod, s) in keys:
        arr = d[(mod, s)]
        X.append(arr)
        y.extend([c2i[mod]] * len(arr))
        snr.extend([s] * len(arr))
    X = np.concatenate(X).astype(np.float32)
    return X, np.asarray(y, np.int64), np.asarray(snr, np.int64), keys, d, time.time() - t0


def l2n(x):
    flat = x.reshape(len(x), -1)
    n = np.maximum(np.linalg.norm(flat, axis=1, keepdims=True), 1e-12)
    return (flat / n).reshape(x.shape)


def eval_model(model_path, X, y, snr):
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    preds = np.empty(len(X), np.int64)
    t0 = time.time()
    for i in range(len(X)):
        preds[i] = int(sess.run(None, {iname: X[i:i + 1]})[0].argmax())
    acc = float((preds == y).mean())
    per_snr = {int(s): float((preds[snr == s] == y[snr == s]).mean()) for s in np.unique(snr)}
    per_cls = {CLASSES[c]: float((preds[y == c] == c).mean()) for c in range(len(CLASSES))}
    cm = np.zeros((len(CLASSES), len(CLASSES)), np.int64)
    for t, p in zip(y, preds):
        cm[t, p] += 1
    return {
        "acc": acc,
        "acc_plus10": per_snr.get(10),
        "per_snr": per_snr,
        "per_class": per_cls,
        "confusion": cm.tolist(),
        "seconds": round(time.time() - t0, 1),
    }, preds


def main():
    out = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "env": {}}
    out["env"] = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "onnx": onnx.__version__,
        "onnxruntime": ort.__version__,
    }
    print("env:", out["env"], flush=True)

    # ---- 1. dataset ----
    X, y, snr, keys, d, load_s = load_rml()
    out["dataset"] = {
        "path": str(DATA),
        "bytes": DATA.stat().st_size,
        "sha256": sha256(DATA),
        "n_keys": len(keys),
        "n_windows": int(len(X)),
        "shape": list(X.shape),
        "dtype": str(X.dtype),
        "classes": sorted({k[0] for k in keys}),
        "snrs": sorted({int(k[1]) for k in keys}),
        "load_seconds": round(load_s, 1),
        # Q 通道是否真是独立数据（取若干 cell 的 std）
        "q_std_sample": {str(k): float(np.std(d[k][:, 1, :])) for k in keys[:5]},
        "q_all_zero": bool(np.all(np.std(X[:, 1, :], axis=1) == 0)),
        "per_cell_count": sorted({int(len(d[k])) for k in keys}),
    }
    print("dataset:", out["dataset"]["n_windows"], "windows", flush=True)

    sp = np.load(SPLIT)
    tr, te = sp["train_idx"], sp["test_idx"]
    union = np.concatenate([tr, te])
    per_cell_test = {}
    for ci, c in enumerate(CLASSES):
        for s in np.unique(snr):
            m = (y == ci) & (snr == s)
            per_cell_test[f"{c}_{int(s)}"] = int((te == np.where(m)[0][0]).sum() if m.any() else 0)
    cell_counts = {f"{CLASSES[ci]}_{int(s)}": int(np.isin(te, np.where((y == ci) & (snr == s))[0]).sum())
                   for ci in range(len(CLASSES)) for s in np.unique(snr)}
    out["split"] = {
        "path": str(SPLIT),
        "train": int(len(tr)), "test": int(len(te)),
        "md5": hashlib.md5(SPLIT.read_bytes()).hexdigest(),
        "no_overlap": bool(len(np.intersect1d(tr, te)) == 0),
        "full_cover": bool(len(np.unique(union)) == len(X)),
        "test_per_class": {CLASSES[c]: int((y[te] == c).sum()) for c in range(len(CLASSES))},
        "test_per_cell_min": int(min(cell_counts.values())),
        "test_per_cell_max": int(max(cell_counts.values())),
        "stratified": bool(all(v == 4000 for v in
                              [int((y[te] == c).sum()) for c in range(len(CLASSES))])),
    }
    print("split:", out["split"]["train"], out["split"]["test"], "stratified=", out["split"]["stratified"], flush=True)

    # ---- 2. checkpoint / param count ----
    out["checkpoint"] = {"path": str(CKPT), "bytes": CKPT.stat().st_size, "sha256": sha256(CKPT)}
    m = onnx.load(str(ONNX_INT8))
    n_params = int(sum(int(np.prod(t.dims)) for t in m.graph.initializer))
    out["onnx"] = {
        "fp32_bytes": ONNX_FP32.stat().st_size,
        "int8_bytes": ONNX_INT8.stat().st_size,
        "int8_params_incl_bias": n_params,
        "nodes": len(m.graph.node),
        "input": [str(i.name) for i in m.graph.input],
        "output": [str(o.name) for o in m.graph.output],
    }
    print("onnx params:", n_params, "nodes:", len(m.graph.node), flush=True)

    # ---- 3/4. accuracy on locked 44k test split ----
    Xt = l2n(X[te])
    yt, st = y[te], snr[te]
    fp32, _ = eval_model(ONNX_FP32, Xt, yt, st)
    print("fp32:", round(fp32["acc"], 4), "+10dB:", round(fp32["acc_plus10"], 4), flush=True)
    int8, _ = eval_model(ONNX_INT8, Xt, yt, st)
    print("int8:", round(int8["acc"], 4), "+10dB:", round(int8["acc_plus10"], 4), flush=True)
    out["accuracy"] = {"fp32": fp32, "int8": int8}

    # ---- 5. board JSON 自洽 ----
    ref = json.loads((RES / "ref_accuracy.json").read_text(encoding="utf-8"))
    b44 = json.loads((RES / "p5_accuracy_44k.json").read_text(encoding="utf-8"))
    cm = np.asarray(b44["confusion"], np.int64)
    out["json_recompute"] = {
        "ref_accuracy_json_acc": ref.get("acc"),
        "ref_vs_measured_int8_diff": None if "acc" not in ref else round(float(int8["acc"]) - float(ref["acc"]), 6),
        "board_json_board_acc": b44.get("board_acc"),
        "board_json_ref_acc": b44.get("ref_acc"),
        "board_json_from_confusion": float(np.trace(cm) / cm.sum()),
        "board_json_n_from_confusion": int(cm.sum()),
        "board_json_bit_exact": b44.get("bit_exact"),
        "board_json_argmax": b44.get("argmax"),
    }
    print("board json:", out["json_recompute"], flush=True)

    # ---- 6. map 资源 ----
    if MAP.exists():
        txt = MAP.read_text(encoding="utf-8", errors="replace")
        ro = re.search(r"Total RO\s+Size\s+(\d+)\s*\(([^)]+)\)", txt)
        rom = re.search(r"Total ROM\s+Size\s+(\d+)\s*\(([^)]+)\)", txt)
        rw = re.search(r"Total RW\s+Size\s+(\d+)\s*\(([^)]+)\)", txt)
        code = re.search(r"Code \(inc\. data\)\s+(\d+)", txt)
        out["resources_map"] = {
            "map_file": str(MAP),
            "map_mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(MAP.stat().st_mtime)),
            "total_ro": ro.groups() if ro else None,
            "total_rom": rom.groups() if rom else None,
            "total_rw": rw.groups() if rw else None,
            "code": code.group(1) if code else None,
        }
        print("map:", out["resources_map"]["total_rom"], out["resources_map"]["total_rw"], flush=True)

    # ---- 7. evidence 日志 ----
    boot = (EV / "serial_boot.log").read_text(encoding="utf-8", errors="replace")
    dwts = re.findall(r"\[DWT\] N=(\d+)\s+mean=([\d.]+) ms  min=([\d.]+) ms  max=([\d.]+) ms", boot)
    gold = (EV / "golden.log").read_text(encoding="utf-8", errors="replace")
    stream = (EV / "stream.log").read_text(encoding="utf-8", errors="replace")
    out["evidence"] = {
        "boot_dwt_lines": dwts,
        "golden_pass": len(re.findall(r"^PASS ", gold, re.M)),
        "golden_fail": len(re.findall(r"^FAIL ", gold, re.M)),
        "golden_tail": gold.strip().splitlines()[-1] if gold.strip() else "",
        "stream_tail": [ln for ln in stream.strip().splitlines()[-3:]],
    }
    print("evidence:", out["evidence"], flush=True)

    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("saved:", OUT, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
