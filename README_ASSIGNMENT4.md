# Assignment 4 extension: the code ↔ pseudocode loop (Option B)

**Research question.** The standard EvaLoop cycle carries a free-form natural-language summary
between cycles. When the model then fails, is it because the summary was too vague (it lost information), or because the
model cannot re-implement its own description? We replace the summary with **pseudocode**, a
more structured, lower-abstraction intermediate representation. Then we ask: (1) does robustness (ASL)
change, (2) does the model ranking change, and (3) do the failure modes change?

## Loop specification

| | Standard loop (baseline) | **Code ↔ pseudocode loop (extension)** |
|---|---|---|
| Cycle-1 input | original MBPP+ prompt | same |
| Step 1 (testable) | `Generate Python code for the following task:\n\n{description}\n\nReturn only the code, without explanations or comments.` | **identical** |
| Code extraction | first fenced block, else full response | **identical** |
| Step 2 (inverse task) | `Summarize what the following code does in a detailed description: ... Return only the summary, ...` | `Rewrite the following code as language-agnostic pseudocode that fully describes its algorithm: ... Use structured pseudocode keywords (FUNCTION, IF, ELSE, FOR EACH, WHILE, RETURN) and plain-English operations; do not use Python or any other real programming language syntax. Return only the pseudocode, without explanations or comments.` |
| Carried to next cycle | `"""\n{summary}\n{original assert line}\n"""` | `"""\n{pseudocode}\n{original assert line}\n"""` (single fenced reply unwrapped; `"""` replaced by `'''`) |
| Signature stability | assert line re-attached every cycle | **same mechanism**: the assert line fixes the function name and arity. The pseudocode also tends to name the function, but we do not rely on that |
| Failure condition | code fails MBPP base **or** EvalPlus plus tests | identical |
| Stop | first failure or 10 cycles | identical |

The model's own output is still the only source of perturbation. The two loops differ in exactly
one prompt, so any ASL difference comes from the intermediate representation.

Both loops use the prompts from the course data package (`prompts/task_prompts.md`). This is on purpose: the
repo's `code_generation_summarization` config uses a different summarization prompt
("Use one sentence to summarize ... start with 'write a python function to'"), and that prompt would not be
comparable with the 60-model baseline.

## What was added / modified

New files:
- `evaloop/tasks/code_pseudocode.py`: `CodeToPseudocodeTask`, registered as `code_pseudocode`. It
  subclasses `CodeSummarizationTask` so that it reuses the assert-line `postprocess` unchanged.
- `evaloop/models/vllm_chat.py`: local-GPU vLLM backend that applies each model's **chat template**
  (`LLM.chat`). Upstream's `VllmLLM` uses a generic "System:/User:/Assistant:" text format instead. The new backend also adds a `dtype` option ("half" for T4 GPUs), greedy decoding, and a
  context-length guard. It is used for the reported Kaggle runs.
- `kaggle/evaloop_assignment4_kaggle.ipynb` is the notebook that produced the results, on Kaggle 2×T4 with no API key.
- `evaloop/models/gemini.py`: Gemini backend (google-genai SDK; works with an API key or with Vertex AI).
  Greedy decoding, thinking budget 0 on 2.5 models, retries with backoff, token accounting, and a
  thread-pooled `batch_generate`, so the unchanged loop runner uses its batch path.
- `scripts/run_loop.py` runs one (model, loop) configuration. `scripts/run_all.sh` runs the whole experiment.
  `scripts/check_models.py` checks model access. `scripts/analyze_extension.py` produces `summary.csv`, metrics, the paired
  per-task comparison, failure dumps and survival curves.
- `tests/test_code_pseudocode.py` has 8 unit tests, including a mocked 3-cycle run through the real
  `BatchEvaluationCycle` that checks the assert line survives every cycle.
- `Dockerfile.assignment4` is a sandbox for running the untrusted generated code.

Modified (kept minimal):
- `evaloop/tasks/__init__.py`: one import that registers the new task.
- `evaloop/core/config.py`: the `standard` and `code_pseudocode` task configs (data-package prompts);
  `loop_type`, `n_tasks`, `sample_seed`, `concurrency`, `thinking_budget` fields; `gemini` model type.
  `get_task_configs()` is the documented single extension point (CONTRIBUTING.md).
- `evaloop/core/evaluator.py`: `run_loop()` picks the task sequence from `config.loop_type`, records the
  prompts, the subset and API usage in the results JSON, and checkpoints after every cycle. There is an optional seeded task
  subset in `_load_dataset`. The loop logic and testing are unchanged.
- `evaloop/models/factory.py`: backends are imported lazily, so API-only runs do not need torch or vLLM, and
  `gemini` is registered.

`evaloop/evaluation/cycle.py` (the loop runner) and the EvalPlus tester are **not** modified.

## How to run

### Reported setup: Kaggle (free GPU, no API key)
Import `kaggle/evaloop_assignment4_kaggle.ipynb` into Kaggle. Set Accelerator to GPU T4 x2 and Internet to On, set
`FORK_URL`, then Save & Run All. It runs Qwen/Qwen2.5-Coder-7B-Instruct and ByteDance-Seed/Seed-Coder-8B-Instruct
× {standard, code_pseudocode} × 378 tasks and writes `evaloop_results.zip`. One configuration by hand:
```bash
python scripts/run_loop.py --model Qwen/Qwen2.5-Coder-7B-Instruct --loop code_pseudocode \
    --backend vllm_chat --dtype half --tensor_parallel_size 2
```

### Alternative: Gemini API

Requirements: Docker (recommended, because generated code is untrusted) or Python ≥ 3.9 with
`pip install evalplus==0.3.1 google-genai fire tqdm matplotlib pyyaml`.

Credentials, either:
```bash
export GEMINI_API_KEY=...                       # Google AI Studio key
# or Vertex AI (course credits):
export GOOGLE_GENAI_USE_VERTEXAI=true GOOGLE_CLOUD_PROJECT=<project> GOOGLE_CLOUD_LOCATION=us-central1
gcloud auth application-default login
```

Full experiment (2 models × 2 loops × 378 tasks, greedy, max 10 cycles):
```bash
docker build -f Dockerfile.assignment4 -t evaloop-a4 .
docker run --rm -e GEMINI_API_KEY \
  -v "$PWD/results:/work/EvaLooop/results" -v "$PWD/logs:/work/EvaLooop/logs" \
  -v "$PWD/analysis:/work/EvaLooop/analysis" -v "$PWD/../assignment2-data:/work/assignment2-data:ro" \
  evaloop-a4
```
Without Docker: `bash scripts/run_all.sh`. Options: `MODELS="m1 m2"`, `N_TASKS=100` (seeded subset,
seed 0), `CONCURRENCY=6`. A single configuration:
`python scripts/run_loop.py --model gemini-2.5-flash-lite --loop code_pseudocode`.

Reproducing every number: each `results/<model>__<loop>_<loop>_results.json` holds all per-cycle
prompts, code and test outcomes. `python scripts/analyze_extension.py --results_dir results
--package_dir ../assignment2-data` recomputes `summary.csv` and `metrics.json` from them.
`asl_score` = ASL_pow = Σᵢ i²·nᵢ /(T·M) (`evaloop.analysis.metrics.asl_pow`), which is the
`aslScore` column of the data package. No LLM judge is used.

Unit tests: `pytest tests` (84 pass; `test_cli_lightweight` needs `fire`).
