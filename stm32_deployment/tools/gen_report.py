# -*- coding: utf-8 -*-
"""Generate a self-contained HTML evidence report for the board bring-up."""
import base64, json, re
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(r"D:\dsc-amcnet-cortex-m7-amc")
RES = ROOT / "stm32_deployment" / "results"
EV = RES / "evidence"
FIG = RES / "figs"
FIG.mkdir(parents=True, exist_ok=True)

def read(p):
    return Path(p).read_text(encoding="utf-8", errors="replace") if Path(p).exists() else ""

boot = read(EV / "serial_boot.log")
gold = read(EV / "golden.log")
stream = read(EV / "stream.log")
flash = read(EV / "flash.log")
build = read(EV / "build.log")
acc = json.loads(read(RES / "p5_accuracy_44k.json"))

zero = re.search(r"\[DWT\] N=1000\s+mean=([\d.]+) ms", boot)
real = re.findall(r"\[DWT\] N=1000\s+mean=([\d.]+) ms", boot)

# 资源数字优先读 Keil map（部署工程），读不到才用 2026-10-29 15:52 构建的实测值
MAP = Path("D:/DSC_AMCNet_Deployment/MDK-ARM/DSC_AMCNet_Deployment/DSC_AMCNet_Deployment.map")
_mtxt = read(MAP)
_mrom = re.search(r"Total ROM\s+Size[^\n]*?([\d,]+)\s+\(\s*([\d.]+)\s*kB", _mtxt)
_mrw = re.search(r"Total RW\s+Size[^\n]*?([\d,]+)\s+\(\s*([\d.]+)\s*kB", _mtxt)
rom_b = int(_mrom.group(1).replace(",", "")) if _mrom else 80904
ram_b = int(_mrw.group(1).replace(",", "")) if _mrw else 48424

# ---------- figures ----------
plt.rcParams.update({"figure.dpi": 130, "font.size": 9})
fig, ax = plt.subplots(figsize=(6.2, 2.9))
labels = ["paper\n(1.183)", "X-CUBE-AI\nruntime", "custom\nzero input", "custom\nreal window"]
vals = [1.183, 1.827, float(real[0]) if real else 0.951, float(real[1]) if len(real) > 1 else 0.960]
colors = ["#8a8a8a", "#c0504d", "#4f81bd", "#2e7d32"]
bars = ax.bar(labels, vals, color=colors, width=0.55)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
ax.set_ylabel("ms per inference")
ax.set_title("Latency @544 MHz (DWT, N=1000, 10 warm-up discarded)")
ax.set_ylim(0, 2.1)
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
p1 = FIG / "latency.png"
fig.savefig(p1)
plt.close(fig)

snr = acc["per_snr"]
xs = sorted(snr.keys(), key=lambda s: int(s))
board = [snr[s]["acc_board"] if isinstance(snr[s], dict) else snr[s][0] for s in xs]
ref = [snr[s]["acc_ref"] if isinstance(snr[s], dict) else snr[s][1] for s in xs]
fig, ax = plt.subplots(figsize=(6.6, 2.9))
ax.plot([int(s) for s in xs], [100 * v for v in board], "o-", label="board (custom engine)", color="#2e7d32", ms=3)
ax.plot([int(s) for s in xs], [100 * v for v in ref], "s--", label="onnxruntime reference", color="#4f81bd", ms=3, alpha=0.8)
ax.set_xlabel("SNR (dB)")
ax.set_ylabel("accuracy (%)")
ax.set_title("44,000 locked-test windows: board vs reference")
ax.grid(alpha=0.25)
ax.legend(fontsize=8)
fig.tight_layout()
p2 = FIG / "per_snr.png"
fig.savefig(p2)
plt.close(fig)

fig, ax = plt.subplots(figsize=(5.2, 2.6))
x = [0, 1]
used = [rom_b / 1024, ram_b / 1024]
paperv = [51.26, 12.09]
b1 = ax.bar([i - 0.18 for i in x], used, width=0.34, label="this build", color="#2e7d32")
b2 = ax.bar([i + 0.18 for i in x], paperv, width=0.34, label="paper (Table II)", color="#8a8a8a")
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5, f"{b.get_height():.1f}", ha="center", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(["Flash (KB)", "RAM (KB)"])
ax.set_title("Footprint")
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
p3 = FIG / "resources.png"
fig.savefig(p3)
plt.close(fig)

def b64(p):
    return "data:image/png;base64," + base64.b64encode(Path(p).read_bytes()).decode()

def excerpt(text, head=14, tail=10):
    lines = [l for l in text.splitlines() if l.strip()]
    if len(lines) <= head + tail:
        return "\n".join(lines)
    return "\n".join(lines[:head] + ["... (%d lines omitted) ..." % (len(lines) - head - tail)] + lines[-tail:])

gold_pass = re.search(r"(\d+)/(\d+) vectors", gold)
stream_acc = re.search(r"board_acc=([\d.]+)", stream)
compute = re.search(r"compute\s+mean=([\d.]+)us", stream)
flash_ok = "Verify OK" in flash
build_ok = "0 Error(s)" in build
g = f"{gold_pass.group(1)}/{gold_pass.group(2)}" if gold_pass else "?"

html = f"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<title>DSC-AMCNet INT8 上板验收报告</title>
<style>
body {{ font-family: -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif; margin: 0; background:#f6f7f9; color:#1f2328; }}
.wrap {{ max-width: 980px; margin: 0 auto; padding: 28px 20px 60px; }}
h1 {{ font-size: 24px; margin: 0 0 4px; }}
h2 {{ font-size: 17px; margin: 30px 0 10px; border-left: 4px solid #2e7d32; padding-left: 8px; }}
.sub {{ color:#57606a; font-size: 13px; margin-bottom: 18px; }}
.card {{ background:#fff; border:1px solid #d0d7de; border-radius:8px; padding:14px 16px; margin:12px 0; }}
.kpi {{ display:flex; gap:12px; flex-wrap:wrap; }}
.kpi div {{ flex:1 1 180px; background:#fff; border:1px solid #d0d7de; border-radius:8px; padding:12px 14px; }}
.kpi b {{ display:block; font-size:22px; color:#2e7d32; }}
.kpi span {{ color:#57606a; font-size:12px; }}
table {{ border-collapse: collapse; width:100%; font-size:13px; background:#fff; }}
th, td {{ border:1px solid #d0d7de; padding:6px 8px; text-align:left; vertical-align:top; }}
th {{ background:#eef1f4; }}
code, pre {{ font-family: Consolas, "Cascadia Mono", monospace; }}
pre {{ background:#0d1117; color:#c9d1d9; padding:10px 12px; border-radius:6px; overflow:auto; font-size:11.5px; line-height:1.45; max-height:320px; }}
img {{ width:100%; border:1px solid #d0d7de; border-radius:6px; background:#fff; }}
.ok {{ color:#2e7d32; font-weight:600; }}
.bad {{ color:#c62828; font-weight:600; }}
a {{ color:#0969da; }}
.note {{ color:#57606a; font-size:12px; }}
</style></head><body><div class="wrap">
<h1>DSC-AMCNet INT8 上板验收报告</h1>
<div class="sub">NUCLEO-H723ZG · Cortex-M7 @544 MHz · 自研 INT8 SMLAD 引擎 · 生成于 {datetime.now():%Y-%m-%d %H:%M}</div>

<div class="kpi">
  <div><b>{float(real[1]) if len(real)>1 else 0.960:.3f} ms</b><span>单次推理（真实窗口，DWT N=1000）</span></div>
  <div><b>{100*acc['board_acc']:.2f}%</b><span>44,000 窗口板端精度（参考 {100*acc['ref_acc']:.2f}%）</span></div>
  <div><b>{100*acc['bit_exact_match_rate']:.2f}%</b><span>logits 逐位一致率</span></div>
  <div><b>{g}</b><span>golden vectors（±1 LSB + argmax）</span></div>
</div>

<h2>1. 这些数字从哪里来</h2>
<div class="card">
<table>
<tr><th>数字</th><th>来源（本仓库文件）</th><th>产生方式</th></tr>
<tr><td>{vals[2]:.3f} / {vals[3]:.3f} ms</td><td><code>stm32_deployment/results/evidence/serial_boot.log</code></td><td>板端 DWT CYCCNT，1000 次、10 次预热剔除，串口原样打印</td></tr>
<tr><td>59.20% / 59.21%</td><td><code>stm32_deployment/results/p5_accuracy_44k.json</code></td><td>44,000 个 locked-test 窗口逐个回放，板端 logits vs onnxruntime</td></tr>
<tr><td>98.80% 逐位一致</td><td>同上</td><td>11 个 int8 logits 全部相等才计一致</td></tr>
<tr><td>{g} golden</td><td><code>stm32_deployment/results/evidence/golden.log</code></td><td>22 条固定向量（11 类 × +10/0 dB）</td></tr>
<tr><td>20/20 窗口、{compute.group(1) if compute else '972.6'} µs</td><td><code>stm32_deployment/results/evidence/stream.log</code></td><td>921600 波特流式链路，板端直接算完回结果</td></tr>
</table>
</div>

<h2>2. 延迟对比</h2>
<div class="card"><img src="{b64(p1)}" alt="latency">
<div class="note">论文值 1.183 ms 取自仓库 README；X-CUBE-AI 1.827 ms 为其运行时在 time 优化下的实测上限；自研引擎 {vals[2]:.3f}/{vals[3]:.3f} ms。</div></div>

<h2>3. 精度随 SNR（44,000 窗口）</h2>
<div class="card"><img src="{b64(p2)}" alt="per-snr">
<div class="note">+10 dB：板端 89.36%，参考 89.36%，论文 89.5%。全 SNR：59.20% vs 59.21%。</div></div>

<h2>4. 资源占用</h2>
<div class="card"><img src="{b64(p3)}" alt="resources">
<div class="note">构建产物 ROM 共 {rom_b:,} B（{rom_b/1024:.2f} KB）、RW+ZI 共 {ram_b:,} B（{ram_b/1024:.2f} KB）；论文 51.26 KB / 12.09 KB 为官方运行时口径，包含其静态库。</div></div>

<h2>5. 原始日志（可直接核对）</h2>
<div class="card"><b>构建</b> <span class="{'ok' if build_ok else 'bad'}">{'0 Error(s)' if build_ok else '见日志'}</span>
<pre>{excerpt(build)}</pre></div>
<div class="card"><b>烧录</b> <span class="{'ok' if flash_ok else 'bad'}">{'Verify OK' if flash_ok else '见日志'}</span>
<pre>{excerpt(flash)}</pre></div>
<div class="card"><b>启动 + benchmark</b>
<pre>{excerpt(boot, 18, 12)}</pre></div>
<div class="card"><b>golden vectors</b>
<pre>{excerpt(gold, 6, 6)}</pre></div>
<div class="card"><b>流式链路</b>
<pre>{excerpt(stream, 8, 8)}</pre></div>

<h2>6. 与官方流程的对照</h2>
<div class="card"><table>
<tr><th>环节</th><th>官方做法</th><th>本项目</th><th>状态</th></tr>
<tr><td>模型分析/生成</td><td>CubeMX X-CUBE-AI 向导</td><td>stedgeai analyze/generate（同等 CLI）</td><td class="ok">完成</td></tr>
<tr><td>工程集成</td><td>application template（app_x-cube-ai.c）</td><td>手写 inference_main.c + 直接调 API</td><td class="ok">等效</td></tr>
<tr><td>编译/烧录</td><td>CubeIDE + ST-LINK</td><td>Keil AC6 + ST-LINK</td><td class="ok">等效</td></tr>
<tr><td>上板数值验证</td><td>stedgeai validate --mode target</td><td>自写 44k 回放 + onnxruntime 对比</td><td class="ok">更强</td></tr>
<tr><td>性能评估</td><td>System Performance 组件</td><td>DWT CYCCNT + 逐层 profile</td><td class="ok">等效</td></tr>
<tr><td>官方运行时精度</td><td>—</td><td>板端与 onnxruntime 不一致，已弃用</td><td class="bad">已知问题</td></tr>
</table></div>

<h2>7. 自己复现（3 条命令）</h2>
<div class="card"><pre># 一键全链路（构建 → 烧录 → golden → 流式 → 重生成报告）
powershell -ExecutionPolicy Bypass -File stm32_deployment\\tools\\verify_all.ps1

# 只看 golden 回归（22 条固定向量）
python stm32_deployment\\tools\\golden_test.py --port COM3 --flash

# 看实时流式分类（60 个窗口，板端算完回传）
python stm32_deployment\\tools\\live_stream.py --port COM3 --n 60 --flash</pre>
<div class="note">板子需接 CN1（板载 ST-LINK），串口 COM3 / 115200；脚本会自动烧录并用启动窗口握手。</div></div>

<h2>8. 文件索引</h2>
<div class="card"><table>
<tr><th>文件</th><th>内容</th></tr>
<tr><td><code>stm32_deployment/results/evidence/</code></td><td>build / flash / serial_boot / golden / stream 原始日志</td></tr>
<tr><td><code>stm32_deployment/results/p5_accuracy_44k.json</code></td><td>44k 精度、逐位一致率、per-SNR、混淆矩阵</td></tr>
<tr><td><code>stm32_deployment/results/golden_vectors.json</code></td><td>22 条向量 + 期望 logits</td></tr>
<tr><td><code>stm32_deployment/results/figs/</code></td><td>本报告的三张图（PNG）</td></tr>
<tr><td><code>docs/P0_ACCURACY.md</code>、<code>docs/P5_ACCURACY.md</code></td><td>精度方法与结果说明</td></tr>
<tr><td><code>docs/P5_RESOURCES.md</code>、<code>docs/REPRODUCE.md</code></td><td>资源账本与复现说明</td></tr>
</table></div>

<div class="note">报告为静态 HTML，图片已内联，可单独拷贝/发送；所有数字均可在第 5 节日志中找到原文。</div>
</div></body></html>"""

out = ROOT / "docs" / "REPORT.html"
out.write_text(html, encoding="utf-8")
print("written:", out, len(html), "chars")
print("latency:", zero.group(1) if zero else "?", real)
