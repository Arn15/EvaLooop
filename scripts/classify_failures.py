#!/usr/bin/env python3
"""Classify every loop that passes cycle 1 and fails later (report Fig. 2).

Categories: name-drift (entry point no longer defined); scaffolding (stripping top-level
non-definition statements makes the code pass the FULL EvalPlus suite, i.e. the function was
correct and test/harness code broke the module); edge-case (base tests pass, plus fail);
semantic (base tests fail). Re-tests run through EvalPlus untrusted_check (separate process,
time/memory limits) - run inside a sandbox.

usage: python scripts/classify_failures.py --results_dir results --package_dir ../assignment2-data --out analysis/failure_classes.json
"""
import argparse, ast, json, glob, re, sys, collections
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ap = argparse.ArgumentParser()
ap.add_argument("--results_dir", default="results"); ap.add_argument("--package_dir", default=None)
ap.add_argument("--out", default="analysis/failure_classes.json")
A = ap.parse_args()
from evaloop.evaluation.py_testing import load_mbpp_dataset, EvalPlusCodeTester
problems, gt = load_mbpp_dataset()
tester = EvalPlusCodeTester(problems, gt)

def strip_scaffold(code, entry):
    """Keep imports + top-level def/class blocks that parse (except ones named like assert); drop the rest."""
    lines = code.splitlines()
    blocks, cur = [], []
    for ln in lines:
        if ln and not ln[0].isspace() and not ln.startswith(('#', ')', ']', '}')) and cur:
            blocks.append(cur); cur = []
        cur.append(ln)
    if cur: blocks.append(cur)
    keep = []
    for b in blocks:
        src = "\n".join(b)
        head = b[0]
        if re.match(r'\s*(import |from \S+ import )', head) or re.match(r'(def|class|@)\s*', head):
            if re.match(r'def\s+(assert|ASSERT)\b', head):
                continue
            try:
                ast.parse(src)
                keep.append(src)
            except SyntaxError:
                pass
    return "\n\n".join(keep)

def classify(code, entry, base, plus, task_id):
    if not code or not code.strip():
        return "no-code"
    if not re.search(rf'def\s+{re.escape(entry)}\s*\(', code):
        return "name-drift"
    stripped = strip_scaffold(code, entry)
    if stripped.strip() != code.strip():
        r = tester.test_code(stripped, task_id, entry)
        if r.get("success"):
            return "scaffolding"           # function itself correct; top-level harness code broke it
    if base and not plus:
        return "edge-case (plus only)"
    return "semantic (base fails)"

out = {}
rows = []
# our runs
for f in sorted(glob.glob(f'{A.results_dir}/*_results.json')):
    d = json.load(open(f))[0]
    key = f"{d.get('model_alias', d['model'])}|{d['loop_type']}"
    for t in d['prompt_results']:
        fails = [c for c in t['cycles'] if not c.get('success')]
        if not fails or fails[0]['cycle'] == 1: continue
        fc = fails[0]; tr = fc.get('test_results', {})
        cat = classify(fc.get('task_1_output', ''), problems[t['task_id']]['entry_point'],
                       tr.get('base_success'), tr.get('plus_success'), t['task_id'])
        rows.append((key, t['task_id'], fc['cycle'], cat))
# data-package baseline (same two models, standard loop)
ours = sorted({k.split('|')[0] for k, *_ in rows})
for m in (ours if A.package_dir else []):
    pf = Path(A.package_dir) / 'loops' / f'{m}.json'
    if not pf.exists(): continue
    d = json.load(open(pf))
    for t in d['tasks']:
        fails = [c for c in t['cycles'] if not c.get('passed')]
        if not fails or fails[0]['cycle'] == 1: continue
        fc = fails[0]
        cat = classify(fc.get('code', ''), problems[t['task_id']]['entry_point'],
                       fc.get('base_passed'), fc.get('plus_passed'), t['task_id'])
        rows.append((f"{m}|package-standard", t['task_id'], fc['cycle'], cat))
Path(A.out).parent.mkdir(parents=True, exist_ok=True)
json.dump(rows, open(A.out, 'w'), indent=0)
tab = collections.defaultdict(collections.Counter)
for k, _, _, c in rows: tab[k][c] += 1
for k in sorted(tab): print(k, sum(tab[k].values()), dict(tab[k]))
