#!/usr/bin/env bash
# Runs the full Assignment 4 experiment: 2 models x {standard, code_pseudocode}
# on all 378 MBPP+ tasks (greedy, max 10 cycles), then the analysis.
#   MODELS="a b"   override the models      N_TASKS=100  use a seeded subset
set -euo pipefail
cd "$(dirname "$0")/.."
MODELS="${MODELS:-gemini-2.5-flash-lite gemini-2.5-flash}"
PACKAGE_DIR="${PACKAGE_DIR:-../assignment2-data}"
CONCURRENCY="${CONCURRENCY:-6}"
SUBSET=(); [ -n "${N_TASKS:-}" ] && SUBSET=(--n_tasks "$N_TASKS" --sample_seed 0)
mkdir -p results logs

python scripts/check_models.py $MODELS

pids=()
for m in $MODELS; do
  for loop in standard code_pseudocode; do
    echo "starting $m / $loop  (log: logs/${m}__${loop}.log)"
    python scripts/run_loop.py --model "$m" --loop "$loop" --concurrency "$CONCURRENCY" \
        ${SUBSET[@]+"${SUBSET[@]}"} --output_dir results/ > "logs/${m}__${loop}.log" 2>&1 &
    pids+=($!)
  done
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
grep -h "^\[done\]\|WARNING" logs/*.log || true
[ "$fail" = 0 ] || { echo "a run failed - see logs/"; exit 1; }

PKG=(); [ -d "$PACKAGE_DIR/loops" ] && PKG=(--package_dir "$PACKAGE_DIR")
python scripts/analyze_extension.py --results_dir results --out_dir analysis ${PKG[@]+"${PKG[@]}"}
echo "ALL DONE - results/ and analysis/ are ready."
