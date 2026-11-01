"""Code -> pseudocode inverse task (Assignment 4 extension, Option B).

The standard EvaLoop loop is   NL prompt -> code -> NL summary -> code -> ...
This loop replaces the summarization step with a *pseudocode* step:

    NL prompt -> code -> pseudocode -> code -> pseudocode -> ...

Everything else is held fixed so that the only manipulated variable is the
representation the model must carry between cycles:

* the generation step (prompt template, code extraction) is the unchanged
  ``CodeGenerationTask``;
* the carried text is wrapped exactly like the standard summary, i.e. as a
  docstring with the original prompt's final ``assert`` line re-attached
  (inherited ``CodeSummarizationTask.postprocess``). The assert line anchors the
  function name and arity, solving signature drift in the same way as the
  standard loop;
* the failure condition is unchanged: a cycle fails when the generated code
  does not pass both the MBPP base and EvalPlus plus tests.

The model's own output remains the only source of perturbation.
"""

import re
from typing import Any, Dict

from .code_summarization import CodeSummarizationTask
from .factory import register_task

DEFAULT_PSEUDOCODE_TEMPLATE = (
    "Rewrite the following code as language-agnostic pseudocode that fully "
    "describes its algorithm:\n\n"
    "```\n{code}\n```\n\n"
    "Use structured pseudocode keywords (FUNCTION, IF, ELSE, FOR EACH, WHILE, "
    "RETURN) and plain-English operations; do not use Python or any other real "
    "programming language syntax. Return only the pseudocode, without "
    "explanations or comments."
)

# A response that is nothing but one fenced block: unwrap it so the carried text
# is the pseudocode itself rather than markdown fences.
_SINGLE_FENCE = re.compile(r"^\s*```[\w+-]*[ \t]*\n([\s\S]*?)\n?```\s*$")


@register_task("code_pseudocode")
class CodeToPseudocodeTask(CodeSummarizationTask):
    """Translate generated code into pseudocode that seeds the next cycle."""

    produces_testable_code = False

    def create_prompt(self, input_text: str) -> str:
        template = self.task_config.get("prompt_template", DEFAULT_PSEUDOCODE_TEMPLATE)
        return template.format(code=input_text)

    def extract_output(self, raw_response: str) -> str:
        text = (raw_response or "").strip()
        match = _SINGLE_FENCE.match(text)
        if match:
            text = match.group(1).strip()
        # A literal triple quote would terminate the docstring wrapper early.
        return text.replace('"""', "'''")

    def postprocess(self, output: str, context: Dict[str, Any]) -> str:
        # Identical wrapping to the standard loop (docstring + original assert).
        return super().postprocess(output, context)
