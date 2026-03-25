from __future__ import annotations

from typing import Any

from math_reasoning_experiments.config.experiment_config import GenerationConfig
from math_reasoning_experiments.data.schema import ProblemInstance
from math_reasoning_experiments.models.backends import ModelBackend

from .base import ParsedAnswer, PromptMethod


class CoTPromptMethod(PromptMethod):
    def __init__(self) -> None:
        super().__init__(name="cot")

    def run(
        self,
        model: ModelBackend,
        problem: ProblemInstance,
        gen_config: GenerationConfig,
        **kwargs: Any,
    ) -> ParsedAnswer:
        instruction = (
            "You are an expert math problem solver. "
            "Think step by step and derive the final numeric or symbolic answer.\n\n"
        )
        user_part = (
            f"Problem: {problem.question}\n\n"
            'Please reason step by step, and put your final answer in the form "Final Answer: <answer>".'
        )
        prompt = instruction + user_part
        output = self._generate(model, prompt, gen_config)
        final = self._extract_final_answer(output)
        normalized = None
        if final is not None:
            normalized = self._normalize_for_dataset(output, problem.dataset)
        return ParsedAnswer(raw_output=output, final_answer=final, normalized=normalized)

