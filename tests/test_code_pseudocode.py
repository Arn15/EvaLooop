"""Unit tests for the code <-> pseudocode loop (Assignment 4 extension)."""

from evaloop.evaluation.cycle import BatchEvaluationCycle
from evaloop.models.base import BaseLLM
from evaloop.tasks import CodeGenerationTask, CodeToPseudocodeTask, TaskFactory
from evaloop.tasks.code_pseudocode import DEFAULT_PSEUDOCODE_TEMPLATE

from conftest import ASSERT_LINE, MBPP_STYLE_PROMPT

CODE = "def similar_elements(a, b):\n    return tuple(set(a) & set(b))"
PSEUDO = "FUNCTION similar_elements(a, b)\n    RETURN elements present in both a and b\nEND FUNCTION"


def make_task():
    return CodeToPseudocodeTask({"type": "code_pseudocode"})


def test_registered_in_factory():
    assert "code_pseudocode" in TaskFactory().list_supported_types()
    assert isinstance(TaskFactory().create_task({"type": "code_pseudocode"}), CodeToPseudocodeTask)


def test_not_testable():
    assert make_task().produces_testable_code is False


def test_create_prompt_embeds_code_with_default_template():
    prompt = make_task().create_prompt(CODE)
    assert CODE in prompt
    assert prompt == DEFAULT_PSEUDOCODE_TEMPLATE.format(code=CODE)


def test_extract_output_unwraps_single_fence():
    assert make_task().extract_output(f"```text\n{PSEUDO}\n```") == PSEUDO
    assert make_task().extract_output(f"```\n{PSEUDO}\n```\n") == PSEUDO


def test_extract_output_keeps_unfenced_text():
    assert make_task().extract_output(f"  {PSEUDO}  ") == PSEUDO


def test_extract_output_neutralises_triple_quotes():
    out = make_task().extract_output('PRINT """hi"""')
    assert '"""' not in out


def test_postprocess_retains_assert_line_and_signature():
    wrapped = make_task().postprocess(PSEUDO, {"cycle_input": MBPP_STYLE_PROMPT})
    assert wrapped == f'"""\n{PSEUDO}\n{ASSERT_LINE}\n"""\n'
    # Same layout as the MBPP+ prompt, so the assert survives every later cycle.
    assert wrapped.splitlines()[-2] == ASSERT_LINE


class ScriptedLLM(BaseLLM):
    """Returns code for generation prompts and pseudocode for pseudocode prompts."""

    def __init__(self):
        super().__init__({"name": "scripted"})
        self.prompts = []

    def generate(self, prompt, **kwargs):
        self.prompts.append(prompt)
        if prompt.startswith("Rewrite the following code as language-agnostic pseudocode"):
            return f"```\n{PSEUDO}\n```"
        return f"```python\n{CODE}\n```"


def test_assert_line_survives_multiple_cycles():
    model = ScriptedLLM()
    tasks = [CodeGenerationTask({"type": "code_generation"}), make_task()]
    runner = BatchEvaluationCycle(model, tasks)
    prompt = MBPP_STYLE_PROMPT
    for cycle in range(1, 4):
        out = runner.run_single_cycle([{"task_id": "Mbpp/2", "prompt": prompt}], cycle)
        result = out["cycle_results"][0]
        assert result["task_1_output"] == CODE
        prompt = result["task_2_output"]
        assert prompt.splitlines()[-2] == ASSERT_LINE
        assert PSEUDO in prompt
    # generation in cycles 2-3 received the pseudocode, not the original prompt
    assert PSEUDO in model.prompts[2] and PSEUDO in model.prompts[4]
