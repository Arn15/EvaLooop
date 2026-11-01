"""vLLM backend that applies the model's own chat template (Assignment 4 addition).

The upstream ``VllmLLM`` wraps prompts in a generic "System: ... User: ...
Assistant:" string. That format is not the instruction format the instruct models were trained on.
This backend uses ``LLM.chat`` instead, which applies the tokenizer's chat template,
and sends each prompt as a single user turn with no system prompt.
Decoding is greedy (temperature 0) with at most 1024 new tokens.
It also exposes ``dtype`` (use "half" on T4/P100 GPUs, e.g. Kaggle) and
tensor parallelism. ``batch_generate`` sends the whole cycle in one vLLM call.
"""

import logging
from typing import Any, Dict, List

from .base import BaseLLM


class VllmChatLLM(BaseLLM):
    def __init__(self, model_config: Dict[str, Any]):
        super().__init__(model_config)
        from vllm import LLM, SamplingParams  # optional dependency

        self.logger = logging.getLogger(__name__)
        self.model_path = model_config["path"]
        llm_kwargs = {}
        if model_config.get("max_num_seqs"):
            llm_kwargs["max_num_seqs"] = int(model_config["max_num_seqs"])
        if model_config.get("enforce_eager"):
            llm_kwargs["enforce_eager"] = True
        self.llm = LLM(
            **llm_kwargs,
            model=self.model_path,
            dtype=model_config.get("dtype", "auto"),
            tensor_parallel_size=int(model_config.get("tensor_parallel_size", 1)),
            max_model_len=int(model_config.get("max_model_len", 8192)),
            gpu_memory_utilization=float(model_config.get("gpu_memory_utilization", 0.90)),
            trust_remote_code=True,
            seed=0,
        )
        self.sampling = SamplingParams(
            temperature=0.0, top_p=1.0, seed=0,
            max_tokens=int(model_config.get("max_new_tokens", 1024)),
        )
        self.tokenizer = self.llm.get_tokenizer()
        self.max_prompt_tokens = int(model_config.get("max_model_len", 8192)) - self.sampling.max_tokens
        self.usage = {"calls": 0, "prompt_tokens": 0, "output_tokens": 0, "truncated_prompts": 0,
                      "fatal_error": None}

    def _messages(self, prompt: str):
        # Guard against a runaway prompt (e.g. a degenerate 1024-token summary
        # loop) exceeding the context window, which would abort the whole batch.
        ids = self.tokenizer.encode(prompt)
        if len(ids) > self.max_prompt_tokens - 64:
            self.usage["truncated_prompts"] += 1
            prompt = self.tokenizer.decode(ids[: self.max_prompt_tokens - 64])
        return [{"role": "user", "content": prompt}]

    def batch_generate(self, prompts: List[str]) -> List[str]:
        if not prompts:
            return []
        try:
            outputs = self.llm.chat([self._messages(p) for p in prompts], self.sampling, use_tqdm=True)
        except Exception as exc:
            # The loop runner would otherwise record every task as a test failure.
            # Remember the error so run_loop.py can discard the run and exit non-zero.
            self.usage["fatal_error"] = f"{type(exc).__name__}: {exc}"[:500]
            raise
        texts = []
        for out in outputs:
            self.usage["calls"] += 1
            self.usage["prompt_tokens"] += len(out.prompt_token_ids or [])
            self.usage["output_tokens"] += len(out.outputs[0].token_ids)
            texts.append(out.outputs[0].text)
        return texts

    def generate(self, prompt: str, **kwargs) -> str:
        return self.batch_generate([prompt])[0]

    def __str__(self) -> str:
        return f"VllmChatLLM(model_name={self.model_path})"
