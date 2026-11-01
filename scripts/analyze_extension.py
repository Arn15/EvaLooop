#!/usr/bin/env python3
"""Analysis for the code <-> pseudocode extension (Assignment 4).

Reads every framework results JSON in --results_dir and (optionally) the
course data package, and writes into --out_dir:

  summary.csv          one row per run: model,configuration,n_tasks,success_rate,asl_score
  metrics.json         full metrics per run (+ data-package reference on the same tasks)
  paired.csv           per-task sustained loops, standard vs pseudocode, per model
  failures_<run>.md    every failed loop: carried prompt before failure, code, test outcome
  survival.png         pass@k curves (k = 1..M) for every run

asl_score is the framework's ASL_pow  = sum_i n_i * i^2 / (T * M), which is the
`aslScore` column of the data package's leaderboard.json (no LLM judge needed).
"""

import argparse
import ast
import csv
import difflib
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaloop.analysis.metrics import asl_bias, asl_pow, cycle_histogram, pass_at_k, pass_drop  # noqa: E402

CONFIG_LABEL = {"standard": "standard-mbpp-plus", "code_pseudocode": "code-pseudocode-mbpp-plus",
                "code_generation_summarization": "repo-default-mbpp-plus"}
PY_LINE = re.compile(r"^\s*(def \w+\(|return\b|for \w+.* in .*:\s*$|if .*:\s*$|elif .*:|else:\s*$|"
                     r"while .*:\s*$|import \w|from \w+ import|\w+\s*=\s*\[.*\]\s*$)")


def inner_text(wrapped: str) -> str:
    """Strip the docstring wrapper and the re-attached assert line."""
    lines = (wrapped or "").strip().splitlines()
    if lines and lines[0].strip() == '"""':
        lines = lines[1:]
    if lines and lines[-1].strip() == '"""':
        lines = lines[:-1]
    if lines and lines[-1].lstrip().startswith("assert"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def is_python(text: str) -> bool:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return False
    return any(isinstance(n, ast.FunctionDef) for n in ast.walk(tree))


def python_line_frac(text: str) -> float:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return sum(bool(PY_LINE.match(ln)) for ln in lines) / len(lines) if lines else 0.0


def auto_tag(task, fail_cycle):
    """Heuristic pre-label of a failed loop (manual inspection refines it)."""
    code = fail_cycle.get("task_1_output", "") or ""
    tr = fail_cycle.get("test_results", {}) or {}
    entry = re.search(r"assert\s+(?:not\s+)?(?:set\(|math\.isclose\()?\s*(\w+)\(", task["initial_prompt"])
    entry = entry.group(1) if entry else None
    if not code.strip():
        return "empty/no-output"
    try:
        ast.parse(code)
    except SyntaxError:
        return "extraction/syntax"
    if entry and not re.search(rf"def\s+{re.escape(entry)}\s*\(", code):
        return "signature-drift"
    if fail_cycle["cycle"] == 1:
        return "fails-original-task"
    if tr.get("base_success") and not tr.get("plus_success"):
        return "edge-case(plus-only)"
    return "semantic(base-fails)"


def load_runs(results_dir: Path):
    runs = []
    for path in sorted(results_dir.glob("*_results.json")):
        data = json.loads(path.read_text())
        for res in data:
            model = res.get("model_alias") or (res.get("config", {}).get("model") or res["model"])
            model = model.rstrip("/").split("/")[-1].lower()
            runs.append({"path": path, "model": model, "loop": res.get("loop_type", "unknown"), "res": res})
    return runs


def run_metrics(res):
    M = res.get("max_cycles", 10)
    tasks = res["prompt_results"]
    sc = [t["successful_cycles"] for t in tasks]
    T = len(sc)
    out = {
        "n_tasks": T, "max_cycles": M,
        "success_rate": pass_at_k(sc, 1) / T,
        "asl_score": asl_pow(sc, M), "asl_bias": asl_bias(sc, M),
        "pass_drop": pass_drop(sc, M), "mean_sustained_loops": sum(sc) / T,
        "pass_at_k": [pass_at_k(sc, k) / T for k in range(1, M + 1)],
        "histogram": cycle_histogram(sc, M),
    }
    tags, base_plus = Counter(), Counter()
    carried_len, leak_strict, leak_frac, code_sim, text_sim = [], 0, [], [], []
    n_carried = 0
    for t in tasks:
        cyc = t["cycles"]
        fails = [c for c in cyc if not c.get("success")]
        if fails:
            tags[auto_tag(t, fails[0])] += 1
            tr = fails[0].get("test_results", {}) or {}
            base_plus[f"base={tr.get('base_success')},plus={tr.get('plus_success')}"] += 1
        prev_code = prev_text = None
        for c in cyc:
            if not c.get("success"):
                break
            txt = inner_text(c.get("task_2_output", ""))
            n_carried += 1
            carried_len.append(len(txt))
            leak_strict += is_python(txt)
            leak_frac.append(python_line_frac(txt))
            code = c.get("task_1_output", "")
            if prev_code is not None:
                code_sim.append(difflib.SequenceMatcher(None, prev_code, code).ratio())
                text_sim.append(difflib.SequenceMatcher(None, prev_text, txt).ratio())
            prev_code, prev_text = code, txt
    out.update({
        "failure_auto_tags": dict(tags), "failure_base_plus": dict(base_plus),
        "carried_text_mean_chars": statistics.mean(carried_len) if carried_len else None,
        "carried_text_is_python_rate": leak_strict / n_carried if n_carried else None,
        "carried_text_python_line_frac": statistics.mean(leak_frac) if leak_frac else None,
        "consecutive_code_similarity": statistics.mean(code_sim) if code_sim else None,
        "consecutive_carried_text_similarity": statistics.mean(text_sim) if text_sim else None,
        "usage": res.get("usage"), "wall_clock_seconds": res.get("wall_clock_seconds"),
    })
    return out


def package_reference(package_dir: Path, model: str, task_ids):
    path = package_dir / "loops" / f"{model}.json"
    if not path.exists():
        return None
    pkg = json.loads(path.read_text())
    by_id = {t["task_id"]: t for t in pkg["tasks"]}
    sc = [by_id[t]["successful_cycles"] for t in task_ids if t in by_id]
    if not sc:
        return None
    return {"n_tasks": len(sc), "success_rate": pass_at_k(sc, 1) / len(sc), "asl_score": asl_pow(sc, 10),
            "pass_at_k": [pass_at_k(sc, k) / len(sc) for k in range(1, 11)],
            "per_task": {t: by_id[t]["successful_cycles"] for t in task_ids if t in by_id}}


def spearman(xs, ys):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return r
    rx, ry = ranks(xs), ranks(ys)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else float("nan")


def write_failures(run, out_dir: Path):
    res = run["res"]
    lines = [f"# Failed loops: {run['model']} / {run['loop']}\n"]
    for t in res["prompt_results"]:
        fails = [c for c in t["cycles"] if not c.get("success")]
        if not fails:
            continue
        f = fails[0]
        prev = [c for c in t["cycles"] if c.get("success")]
        prompt_in = prev[-1]["task_2_output"] if prev else t["initial_prompt"]
        tr = f.get("test_results", {}) or {}
        lines += [f"## {t['task_id']} - failed at cycle {f['cycle']} "
                  f"(sustained {t['successful_cycles']}) - auto: {auto_tag(t, f)}",
                  f"base={tr.get('base_success')} plus={tr.get('plus_success')} "
                  f"base_details={str(tr.get('base_details'))[:200]}",
                  "### Original prompt", "```", t["initial_prompt"].strip(), "```",
                  f"### Input to failing cycle {f['cycle']}", "```", prompt_in.strip(), "```"]
        if prev:
            lines += ["### Last passing code", "```python", prev[-1].get("task_1_output", "").strip(), "```"]
        lines += ["### Failing code", "```python", (f.get("task_1_output") or "").strip(), "```", ""]
    (out_dir / f"failures_{run['model']}__{run['loop']}.md").write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--package_dir", default=None, help="path to the unzipped course data package")
    ap.add_argument("--out_dir", default="analysis")
    a = ap.parse_args()
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = load_runs(Path(a.results_dir))
    if not runs:
        sys.exit(f"no *_results.json in {a.results_dir}")

    metrics, rows = {}, []
    for run in runs:
        m = run_metrics(run["res"])
        key = f"{run['model']}__{run['loop']}"
        if a.package_dir:
            ids = [t["task_id"] for t in run["res"]["prompt_results"]]
            ref = package_reference(Path(a.package_dir), run["model"], ids)
            if ref:
                m["package_standard_reference"] = {k: v for k, v in ref.items() if k != "per_task"}
        metrics[key] = m
        rows.append([run["model"], CONFIG_LABEL.get(run["loop"], run["loop"]), m["n_tasks"],
                     f"{m['success_rate']:.3f}", f"{m['asl_score']:.3f}"])
        write_failures(run, out_dir)

    with open(out_dir / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "configuration", "n_tasks", "success_rate", "asl_score"])
        w.writerows(rows)

    # Paired per-task comparison standard vs pseudocode
    by_model = {}
    for run in runs:
        by_model.setdefault(run["model"], {})[run["loop"]] = {
            t["task_id"]: t["successful_cycles"] for t in run["res"]["prompt_results"]}
    paired_rows, paired = [], {}
    for model, loops in by_model.items():
        if "standard" in loops and "code_pseudocode" in loops:
            s, p = loops["standard"], loops["code_pseudocode"]
            common = [t for t in s if t in p]
            both1 = [t for t in common if s[t] >= 1 and p[t] >= 1]
            paired[model] = {
                "n_common": len(common), "n_pass_cycle1_both": len(both1),
                "pseudo_longer": sum(p[t] > s[t] for t in both1),
                "standard_longer": sum(s[t] > p[t] for t in both1),
                "tie": sum(s[t] == p[t] for t in both1),
                "mean_diff_pseudo_minus_std": statistics.mean([p[t] - s[t] for t in both1]) if both1 else None,
                "pseudo_reached_10_std_not": sum(p[t] == 10 and s[t] < 10 for t in common),
                "std_reached_10_pseudo_not": sum(s[t] == 10 and p[t] < 10 for t in common),
            }
            paired_rows += [[model, t, s[t], p[t]] for t in common]
    with open(out_dir / "paired.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "task_id", "standard_sustained", "pseudocode_sustained"])
        w.writerows(paired_rows)

    # Ranking agreement across configurations (meaningful with >= 3 models)
    ranking = {}
    for loop in ("standard", "code_pseudocode"):
        ranking[loop] = sorted(((metrics[f"{m}__{loop}"]["asl_score"], m) for m in by_model
                                if f"{m}__{loop}" in metrics), reverse=True)
    models_both = [m for m in by_model if f"{m}__standard" in metrics and f"{m}__code_pseudocode" in metrics]
    if len(models_both) >= 3:
        ranking["spearman_std_vs_pseudo"] = spearman(
            [metrics[f"{m}__standard"]["asl_score"] for m in models_both],
            [metrics[f"{m}__code_pseudocode"]["asl_score"] for m in models_both])

    (out_dir / "metrics.json").write_text(json.dumps(
        {"runs": metrics, "paired": paired, "ranking": ranking}, indent=2))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 3.6))
        for key, m in metrics.items():
            ax.plot(range(1, m["max_cycles"] + 1), m["pass_at_k"], marker="o", ms=3, label=key)
        ax.set_xlabel("cycle k")
        ax.set_ylabel("fraction of tasks passing at cycle k")
        ax.grid(alpha=.3)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out_dir / "survival.png", dpi=200)
    except ImportError:
        pass

    print((out_dir / "summary.csv").read_text())
    print(json.dumps(paired, indent=2))


if __name__ == "__main__":
    main()
