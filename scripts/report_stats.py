#!/usr/bin/env python3
"""Paired statistics for the report: bootstrap 95% CIs of ASL differences and sign tests.

usage: python scripts/report_stats.py --results_dir results --out analysis/stats.json
"""
import argparse, glob, json, math, random

ap = argparse.ArgumentParser()
ap.add_argument("--results_dir", default="results"); ap.add_argument("--out", default="analysis/stats.json")
ap.add_argument("--n_boot", type=int, default=5000); ap.add_argument("--seed", type=int, default=0)
A = ap.parse_args()
R = {}
for f in glob.glob(f"{A.results_dir}/*_results.json"):
    d = json.load(open(f))[0]
    R[(d.get("model_alias", d["model"]), d["loop_type"])] = {t["task_id"]: t["successful_cycles"] for t in d["prompt_results"]}

def asl(v): return sum(x * x for x in v) / (len(v) * 10)
def sign_p(k, n): return min(1.0, 2 * sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n) if n else 1.0
def boot(f, ids, rng):
    xs = sorted(f([rng.choice(ids) for _ in ids]) for _ in range(A.n_boot))
    return [round(xs[int(.025 * A.n_boot)], 3), round(xs[int(.975 * A.n_boot)], 3)]

rng = random.Random(A.seed); out = {}
models = sorted({m for m, _ in R})
for m in models:
    if (m, "standard") not in R or (m, "code_pseudocode") not in R: continue
    s, p = R[(m, "standard")], R[(m, "code_pseudocode")]; ids = sorted(s)
    worse, better = sum(p[t] < s[t] for t in ids), sum(p[t] > s[t] for t in ids)
    out[m] = {"asl_standard": round(asl([s[t] for t in ids]), 3), "asl_pseudocode": round(asl([p[t] for t in ids]), 3),
              "delta_ci95": boot(lambda b: asl([p[t] for t in b]) - asl([s[t] for t in b]), ids, rng),
              "tasks_worse": worse, "tasks_better": better, "sign_test_p": sign_p(better, worse + better)}
if len(models) == 2:
    a, b = models
    for loop in ("standard", "code_pseudocode"):
        if (a, loop) in R and (b, loop) in R:
            ids = sorted(R[(a, loop)])
            out[f"{a}_minus_{b}_{loop}_ci95"] = boot(lambda x: asl([R[(a, loop)][t] for t in x]) - asl([R[(b, loop)][t] for t in x]), ids, rng)
json.dump(out, open(A.out, "w"), indent=1)
print(json.dumps(out, indent=1))
