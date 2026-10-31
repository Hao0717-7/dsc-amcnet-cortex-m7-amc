#!/usr/bin/env python3
"""check_docs_consistency.py — 关键数字一致性巡检。

把仓库文档里的口径与 `stm32_deployment/results/measured_latest.json` 的最新指纹对齐：

1. **自洽性**：JSON 内部（P2 / C1 / base 三段）延迟、资源、精度必须互相一致，且资源与 C1 后
   构建相同（ROM 80,104 B / RAM 48,424 B）；P2 延迟必须优于 C1（550 MHz 只提高主频）。
2. **必含项**：README / PROJECT_STATUS / REPORT.html 等当前口径文档必须包含最新数字。
3. **陈旧值扫描**：在「当前口径」文档里不应再出现旧数值，除非该行自带历史标记
   （历史/C1/544/v1.0.x/P0 阶段/P1 阶段/早期），或文件本身是历史文件（RELEASE_*、P1_BENCHMARK、P4）。

用法：
    python stm32_deployment/tools/check_docs_consistency.py            # 人读
    python stm32_deployment/tools/check_docs_consistency.py --json out.json   # 机器读
退出码：0 = 全部通过；1 = 有问题。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MEASURED = ROOT / "stm32_deployment/results/measured_latest.json"

# 只扫描这些文件里的"当前口径"（历史文件整份豁免）
CURRENT_DOCS = [
    "README.md",
    "PROJECT_STATUS.md",
    "docs/P0_ACCURACY.md",
    "docs/P5_ACCURACY.md",
    "docs/P5_REALTIME.md",
    "docs/P5_RESOURCES.md",
    "docs/P5_BLOCKED.md",
    "docs/COMPARISON_vs_OFFICIAL_README.md",
    "docs/REPRODUCE.md",
    "docs/BUILD.md",
    "stm32_deployment/README.md",
    "stm32_deployment/custom_engine/README.md",
]
# 这些文件保留历史读数（含旧版本口径），不做陈旧值扫描
# PROJECT_MEMORY 是"历史归档 + 当前状态"混合体：旧读数带 ~~删除线~~/日期，允许保留
HISTORICAL_DOCS = [
    "docs/PROJECT_MEMORY.md",
    "docs/P1_BENCHMARK.md",
    "docs/P4_OPTIMIZATION.md",
    "docs/REPORT.html",
]
HISTORICAL_MARKERS = ("历史", "C1", "544", "v1.0.0", "v1.0.1", "v1.0.2",
                      "v1.0.3", "v1.0.4", "v1.0.5", "P0 阶段", "P1 阶段", "早期")

problems: list[str] = []
notes: list[str] = []


def fail(msg: str) -> None:
    problems.append(msg)


def read(rel: str) -> str:
    p = ROOT / rel
    if not p.exists():
        fail(f"缺少文件: {rel}")
        return ""
    return p.read_text(encoding="utf-8", errors="replace")


def check_json_selfconsistent() -> dict:
    data = json.loads(MEASURED.read_text(encoding="utf-8"))
    p2 = data.get("p2_clock_550mhz_2026_10_31") or {}
    c1 = data.get("c1_kernel_optimization_2026_10_30") or {}
    base = data.get("consistency") or {}
    if not p2 or not c1:
        fail("measured_latest.json 缺少 p2_clock_550mhz 或 c1_kernel_optimization 段")
        return {}

    lat = p2.get("latency", {})
    res = p2.get("resources_unchanged", {})
    c1_res = c1.get("resources_after", {})
    canon = {
        "zero_ms": lat.get("zero_input_mean_ms"),
        "real_ms": lat.get("real_window_mean_ms"),
        "clock_mhz": lat.get("clock_mhz"),
        "rom_b": res.get("rom_b"),
        "ram_b": res.get("ram_b"),
        "classify_us": p2.get("classify_loop_us"),
        "stream_us": (p2.get("stream") or {}).get("compute_us_mean"),
        "int8_acc": data["accuracy_locked_44k"]["int8_onnx"]["acc"],
        "int8_acc_plus10": data["accuracy_locked_44k"]["int8_onnx"]["acc_plus10db"],
        "ref_acc": data["accuracy_locked_44k"]["int8_vs_results_ref_accuracy_json"]["ref_json_acc"],
    }
    if canon["zero_ms"] is None or canon["real_ms"] is None:
        fail("P2 段缺少 latency 数字")
    if canon["clock_mhz"] != 550:
        fail(f"P2 段 clock_mhz={canon['clock_mhz']}，应为 550")
    if res.get("rom_b") != c1_res.get("rom_b") or res.get("ram_b") != c1_res.get("ram_b"):
        fail(f"P2 资源 {res.get('rom_b')}/{res.get('ram_b')} 与 C1 后 {c1_res.get('rom_b')}/{c1_res.get('ram_b')} 不一致")
    if (res.get("rom_b") or 0) <= 0:
        fail("P2 段 resources_unchanged 缺少 rom_b")
    # P2 必须快于 C1（只提主频）
    c1_lat = (c1.get("latency_ms") or {})
    c1_zero = (c1_lat.get("zero_input") or {}).get("mean_ms")
    c1_real = (c1_lat.get("real_window") or {}).get("mean_ms")
    for name, new, old in (("全零", canon["zero_ms"], c1_zero), ("真实窗", canon["real_ms"], c1_real)):
        if new is not None and old is not None and not (new < old):
            fail(f"P2 {name} {new} ms 未优于 C1 {old} ms（550 MHz 应更快）")
    # 精度：INT8 与 ref 在一个窗口内（44k 允许 2 窗差）
    if abs(canon["int8_acc"] - canon["ref_acc"]) > 1e-4:
        fail(f"INT8 acc {canon['int8_acc']} 与 ref {canon['ref_acc']} 偏差过大")
    notes.append(f"JSON: {canon['zero_ms']}/{canon['real_ms']} ms @{canon['clock_mhz']} MHz, "
                 f"ROM {canon['rom_b']} B / RAM {canon['ram_b']} B, "
                 f"classify {canon['classify_us']} us, stream {canon['stream_us']} us")
    return canon


def check_must_contain(canon: dict) -> None:
    zero, real = canon["zero_ms"], canon["real_ms"]
    rom_kb = canon["rom_b"] / 1024
    ram_kb = canon["ram_b"] / 1024
    required = {
        "README.md": [f"{zero:.3f}", f"{real:.3f}", f"{rom_kb:.2f}", f"{ram_kb:.2f}", "550 MHz", "23.6%"],
        "PROJECT_STATUS.md": [f"{zero:.3f}", f"{real:.3f}", "550 MHz", f"{rom_kb:.2f}"],
        "docs/REPORT.html": [f"{zero:.3f}", f"{real:.3f}", "550 MHz"],
        "docs/P5_RESOURCES.md": [f"{rom_kb:.2f}", f"{ram_kb:.2f}"],
        "docs/COMPARISON_vs_OFFICIAL_README.md": [f"{zero:.3f}", f"{real:.3f}", "550 MHz"],
        "docs/PROJECT_MEMORY.md": [f"{zero:.3f}", f"{real:.3f}", "550 MHz"],
        "stm32_deployment/custom_engine/README.md": [f"{zero:.3f}", f"{real:.3f}"],
    }
    for rel, needles in required.items():
        text = read(rel)
        for n in needles:
            if n not in text:
                fail(f"{rel} 缺少当前口径字符串 {n!r}")


def check_stale_values(canon: dict) -> None:
    zero, real = canon["zero_ms"], canon["real_ms"]
    stale = ["0.951", "0.960", "0.905", "0.914", "0.968", "0.969",
             "87408", "48560", "85.36", "47.42", "22.7%", "2.00×"]
    # 允许出现的当前值
    for doc in CURRENT_DOCS:
        text = read(doc)
        for lineno, line in enumerate(text.splitlines(), 1):
            stripped = line.strip()
            if not stripped:
                continue
            if any(m in stripped for m in HISTORICAL_MARKERS):
                continue
            for token in stale:
                if token in stripped:
                    fail(f"{doc}:{lineno} 出现陈旧值 {token!r}: {stripped[:110]}")
    # 历史文件里必须带历史标记（P4 有「C1 结果」段，P1 有演进注）
    for doc in HISTORICAL_DOCS:
        if doc.endswith("REPORT.html"):
            continue
        text = read(doc)
        if not any(m in text for m in HISTORICAL_MARKERS) and str(zero) not in text:
            fail(f"{doc} 既没有历史标记，也没有当前值 {zero}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", default="", help="把结果写成 JSON")
    args = ap.parse_args()

    canon = check_json_selfconsistent()
    if canon:
        check_must_contain(canon)
        check_stale_values(canon)

    print("== docs consistency ==")
    for n in notes:
        print("  note:", n)
    if problems:
        print(f"\nFAIL: {len(problems)} problem(s)")
        for p in problems:
            print("  -", p)
    else:
        print("PASS: 文档口径与 measured_latest.json 一致")

    if args.json:
        Path(args.json).write_text(json.dumps(
            {"ok": not problems, "canonical": canon, "problems": problems, "notes": notes},
            indent=2, ensure_ascii=False), encoding="utf-8")
    return 0 if not problems else 1


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    sys.exit(main())
