#!/usr/bin/env python3
"""Run one (model, loop) configuration of EvaLoop.

Examples
--------
    python scripts/run_loop.py --model gemini-2.5-flash-lite --loop standard
    python scripts/run_loop.py --model gemini-2.5-flash-lite --loop code_pseudocode

Writes  results/<model>__<loop>_<loop>_results.json  (framework format).
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaloop.core.config import EvaluationConfig  # noqa: E402
from evaloop.core.evaluator import EvaLoopEvaluator  # noqa: E402
from evaloop.utils.logging_utils import setup_logging  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--loop", default="code_pseudocode",
                    choices=["standard", "code_pseudocode", "code_generation_summarization"])
    ap.add_argument("--n_tasks", type=int, default=None, help="random subset size (default: all 378)")
    ap.add_argument("--sample_seed", type=int, default=0)
    ap.add_argument("--max_cycles", type=int, default=10)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--thinking_budget", type=int, default=0)
    ap.add_argument("--backend", default=None, help='e.g. "vllm_chat" for local GPU models')
    ap.add_argument("--dtype", default="auto", help='vllm_chat: use "half" on T4/P100')
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    ap.add_argument("--max_model_len", type=int, default=8192)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    ap.add_argument("--max_num_seqs", type=int, default=None)
    ap.add_argument("--enforce_eager", action="store_true")
    ap.add_argument("--resume", default=None, help="path to a <name>.checkpoint.json to continue")
    ap.add_argument("--output_dir", default="results/")
    ap.add_argument("--log_level", default="WARNING")
    args = ap.parse_args()

    setup_logging(level=args.log_level)
    # e.g. "Qwen/Qwen2.5-Coder-7B-Instruct" -> "qwen2.5-coder-7b-instruct",
    # which is also the model_id used in the course data package.
    alias = args.model.rstrip("/").split("/")[-1].lower()
    name = f"{alias}__{args.loop}"
    config = EvaluationConfig(
        model=args.model, max_cycles=args.max_cycles, temperature=0.0, top_p=1.0,
        max_new_tokens=1024, output_dir=args.output_dir, experiment_name=name,
        loop_type=args.loop, n_tasks=args.n_tasks, sample_seed=args.sample_seed,
        concurrency=args.concurrency, thinking_budget=args.thinking_budget,
        is_chat_model=True, system_prompt="",
        backend=args.backend, dtype=args.dtype, tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs, enforce_eager=args.enforce_eager,
        resume_checkpoint=args.resume,
    )
    # Preflight: fail fast on a bad key / model name instead of silently
    # recording 378 empty generations as test failures.
    if config.get_model_config()["type"] == "gemini":
        from evaloop.models.gemini import GeminiLLM
        probe = GeminiLLM({**config.get_model_config(), "max_retries": 2})
        if not probe.generate("Reply with the single word OK.").strip():
            sys.exit("Preflight call to the model returned nothing - check GEMINI_API_KEY / "
                     "Vertex settings and the model name, then retry.")

    t0 = time.time()
    result = EvaLoopEvaluator(config).run_loop()[0]
    result["wall_clock_seconds"] = round(time.time() - t0, 1)
    result["model_alias"] = alias
    result["attention_backend"] = os.environ.get("VLLM_ATTENTION_BACKEND", "default")
    out = Path(args.output_dir) / f"{name}_{args.loop}_results.json"
    usage = result.get("usage") or {}
    if usage.get("fatal_error") or (usage.get("calls", 1) == 0):
        # Model crashed (e.g. CUDA OOM): never keep a run whose "failures" are infra errors.
        failed = out.with_suffix(".FAILED.json")
        failed.write_text(json.dumps([result], indent=2))
        if out.exists():
            out.unlink()
        sys.exit(f"[failed] {name}: generation error {usage.get('fatal_error')!r} -> {failed}")
    out.write_text(json.dumps([result], indent=2))

    cycles = [r["successful_cycles"] for r in result["prompt_results"]]
    T, M = len(cycles), args.max_cycles
    asl = sum(c * c for c in cycles) / (T * M)
    if usage.get("errors"):
        print(f"WARNING: {usage['errors']} API calls failed after retries (recorded as empty output).")
    print(f"[done] {name}: n_tasks={T} success_rate={sum(c >= 1 for c in cycles) / T:.3f} "
          f"asl_score={asl:.3f} usage={result.get('usage')} -> {out}")


if __name__ == "__main__":
    main()
