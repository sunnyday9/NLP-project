from __future__ import annotations

from typing import Any

from math_reasoning_experiments.config.experiment_config import GenerationConfig
from math_reasoning_experiments.data.schema import ProblemInstance
from math_reasoning_experiments.models.backends import ModelBackend

from .base import ParsedAnswer, PromptMethod


class LeastToMostPromptMethod(PromptMethod):
    def __init__(self) -> None:
        super().__init__(name="least_to_most")

    def run(
        self,
        model: ModelBackend,
        problem: ProblemInstance,
        gen_config: GenerationConfig,
        **kwargs: Any,
    ) -> ParsedAnswer:
        prompt = (
            "You will solve the following problem by first breaking it into smaller "
            "subproblems and then solving them one by one.\n\n"
            f"Problem: {problem.question}\n\n"
            "1. First, rewrite the problem as a sequence of simpler subproblems.\n"
            "2. Then, solve each subproblem in order.\n"
            "3. Finally, combine the results and give the final answer.\n\n"
            "Follow the format:\n"
            "Subproblems:\n"
            "- ...\n"
            "- ...\n\n"
            "Solutions:\n"
            "- Step 1: ...\n"
            "- Step 2: ...\n\n"
            "Final Answer: "
        )
        output = self._generate(model, prompt, gen_config)
        final = self._extract_final_answer(output)
        normalized = None
        if final is not None:
            normalized = self._normalize_for_dataset(output, problem.dataset)
        return ParsedAnswer(raw_output=output, final_answer=final, normalized=normalized)

