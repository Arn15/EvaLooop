"""Google Gemini backend (Assignment 4 addition).

Uses the official ``google-genai`` SDK, which works with either
  * a Gemini API key:   export GEMINI_API_KEY=...            (AI Studio), or
  * Vertex AI:          export GOOGLE_GENAI_USE_VERTEXAI=true
                        export GOOGLE_CLOUD_PROJECT=<project>
                        export GOOGLE_CLOUD_LOCATION=us-central1
                        gcloud auth application-default login

Greedy decoding (temperature 0, top_p 1). For Gemini 2.5 models the thinking
budget is set to 0 (configurable) so the loop measures the plain model, as
with the non-reasoning models in the original data package.

Implements ``batch_generate`` with a thread pool, so the loop runner
(BatchEvaluationCycle) automatically uses its batch path.
"""

import logging
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List

from .base import BaseLLM


class GeminiLLM(BaseLLM):
    def __init__(self, model_config: Dict[str, Any]):
        super().__init__(model_config)
        from google import genai  # imported lazily: optional dependency

        self.logger = logging.getLogger(__name__)
        self.model_id = model_config["model_id"]
        self.temperature = model_config.get("temperature", 0.0)
        self.top_p = model_config.get("top_p", 1.0)
        self.max_new_tokens = model_config.get("max_new_tokens", 1024)
        self.thinking_budget = model_config.get("thinking_budget", 0)
        self.concurrency = max(1, int(model_config.get("concurrency", 8)))
        self.max_retries = int(model_config.get("max_retries", 8))
        self.system_prompt = model_config.get("system_prompt")
        self.client = genai.Client()  # reads GEMINI_API_KEY / Vertex env vars
        self._lock = threading.Lock()
        self.usage = {"calls": 0, "prompt_tokens": 0, "output_tokens": 0,
                      "thinking_tokens": 0, "errors": 0}

    def _gen_config(self):
        from google.genai import types

        kwargs = dict(temperature=self.temperature, top_p=self.top_p,
                      max_output_tokens=self.max_new_tokens)
        if self.system_prompt:
            kwargs["system_instruction"] = self.system_prompt
        if "2.5" in self.model_id:  # 2.5 models: disable thinking (budget 0)
            kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=self.thinking_budget)
        return types.GenerateContentConfig(**kwargs)

    def generate(self, prompt: str, **kwargs) -> str:
        delay = 2.0
        for attempt in range(self.max_retries):
            try:
                resp = self.client.models.generate_content(
                    model=self.model_id, contents=prompt, config=self._gen_config())
                meta = getattr(resp, "usage_metadata", None)
                with self._lock:
                    self.usage["calls"] += 1
                    if meta is not None:
                        self.usage["prompt_tokens"] += meta.prompt_token_count or 0
                        self.usage["output_tokens"] += meta.candidates_token_count or 0
                        self.usage["thinking_tokens"] += getattr(meta, "thoughts_token_count", 0) or 0
                return resp.text or ""
            except Exception as exc:  # rate limits, transient 5xx
                self.logger.warning(f"Gemini call failed (attempt {attempt + 1}): {exc}")
                time.sleep(delay + random.random())
                delay = min(delay * 2, 60)
        with self._lock:
            self.usage["errors"] += 1
        return ""  # an empty response fails the test, like a refusal would

    def batch_generate(self, prompts: List[str]) -> List[str]:
        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            return list(pool.map(self.generate, prompts))

    def __str__(self) -> str:
        return f"GeminiLLM(model_name={self.model_id})"
