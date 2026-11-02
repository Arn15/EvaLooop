#!/usr/bin/env python3
"""Report figures. Needs analysis/metrics.json (analyze_extension.py --package_dir ...) and
analysis/failure_classes.json (classify_failures.py).
usage: python scripts/report_figures.py --analysis_dir analysis
"""
import argparse, collections, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ap = argparse.ArgumentParser(); ap.add_argument("--analysis_dir", default="analysis"); A = ap.parse_args()
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": "#9a9a96", "axes.linewidth": 0.6,
                     "xtick.color": "#5b5b57", "ytick.color": "#5b5b57", "axes.labelcolor": "#3b3b38", "text.color": "#1f1f1d"})
BLUE, ORANGE, AQUA, YELLOW, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a8a86"
M = json.load(open(f"{A.analysis_dir}/metrics.json"))["runs"]
models = [("seed-coder-8b-instruct", "Seed-Coder-8B-Instruct", "Seed"), ("qwen2.5-coder-7b-instruct", "Qwen2.5-Coder-7B-Instruct", "Qwen")]

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5), sharey=True); k = list(range(1, 11))
for ax, (m, name, _) in zip(axes, models):
    s, p = M[f"{m}__standard"], M[f"{m}__code_pseudocode"]; ref = s.get("package_standard_reference")
    if ref: ax.plot(k, ref["pass_at_k"], color=GRAY, lw=1.6, ls=(0, (3, 2)), marker="o", ms=3.2)
    ax.plot(k, s["pass_at_k"], color=BLUE, lw=2, marker="o", ms=3.6, markeredgecolor="white", markeredgewidth=0.8)
    ax.plot(k, p["pass_at_k"], color=ORANGE, lw=2, marker="s", ms=3.6, markeredgecolor="white", markeredgewidth=0.8)
    for v, l in [(s["pass_at_k"][-1], "Std, ours"), (p["pass_at_k"][-1], "Pseudocode")] + ([(ref["pass_at_k"][-1], "Std, package")] if ref else []):
        ax.text(10.3, v, f"{l} {v:.3f}", va="center", fontsize=7, color="#3b3b38")
    ax.set_title(name, fontsize=9, loc="left"); ax.set_xticks(k); ax.set_xlim(0.7, 10.2); ax.set_ylim(0.55, 0.76)
    ax.grid(axis="y", color="#e6e5e1", lw=0.6); ax.set_axisbelow(True); ax.set_xlabel("cycle k")
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
axes[0].set_ylabel("share of 378 tasks\npassing at cycle k")
fig.subplots_adjust(left=0.1, right=0.83, wspace=0.75, bottom=0.17, top=0.88)
fig.savefig(f"{A.analysis_dir}/fig_survival.png", dpi=300)

rows = json.load(open(f"{A.analysis_dir}/failure_classes.json"))
C = collections.defaultdict(collections.Counter)
for key, _, _, c in rows: C[key][c] += 1
order = [(f"{m}|{loop}", f"{short} · {lab}") for m, _, short in models
         for loop, lab in (("package-standard", "std (package)"), ("standard", "std (ours)"), ("code_pseudocode", "pseudocode"))]
cats = [("semantic (base fails)", "Semantic (base tests fail)", BLUE), ("edge-case (plus only)", "Edge case (plus tests only)", AQUA),
        ("scaffolding", "Test-scaffolding breakage", ORANGE), ("name-drift", "Function-name drift", YELLOW)]
fig, ax = plt.subplots(figsize=(7.0, 2.4)); ys = list(range(len(order)))[::-1]
for y, (key, label) in zip(ys, order):
    left = 0
    for c, _, col in cats:
        v = C[key][c]
        if v:
            ax.barh(y, v, left=left, color=col, height=0.62, edgecolor="white", linewidth=1.2)
            if v >= 3: ax.text(left + v / 2, y, str(v), ha="center", va="center", fontsize=7, color="white" if col in (BLUE, ORANGE) else "#1f1f1d")
            left += v
    ax.text(left + 0.8, y, str(left), va="center", fontsize=7.5, color="#3b3b38")
ax.set_yticks(ys); ax.set_yticklabels([l for _, l in order]); ax.set_xlabel("loops that failed after cycle 1 (of 378 tasks)")
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
ax.grid(axis="x", color="#e6e5e1", lw=0.6); ax.set_axisbelow(True); ax.set_xlim(0, 62)
ax.legend(handles=[Patch(color=col, label=cl) for _, cl, col in cats], ncol=4, fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.42, 1.2))
fig.subplots_adjust(left=0.2, right=0.98, bottom=0.2, top=0.86)
fig.savefig(f"{A.analysis_dir}/fig_failures.png", dpi=300)
print("wrote fig_survival.png, fig_failures.png")
