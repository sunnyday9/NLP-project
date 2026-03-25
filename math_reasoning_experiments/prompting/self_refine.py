from __future__ import annotations

from typing import Any

from math_reasoning_experiments.config.experiment_config import GenerationConfig
from math_reasoning_experiments.data.schema import ProblemInstance
from math_reasoning_experiments.models.backends import ModelBackend

from .base import ParsedAnswer, PromptMethod


class SelfRefinePromptMethod(PromptMethod):
    def __init__(self, num_refine_steps: int = 1) -> None:
        super().__init__(name="self_refine")
        self.num_refine_steps = max(1, num_refine_steps)

    def run(
        self,
        model: ModelBackend,
        problem: ProblemInstance,
        gen_config: GenerationConfig,
        **kwargs: Any,
    ) -> ParsedAnswer:
        # Step 1: draft solution (CoT style)
        base_instruction = (
            "You are an expert math problem solver. "
            "Think step by step and derive a draft solution.\n\n"
        )
        user_part = (
            f"Problem: {problem.question}\n\n"
            'Provide a detailed draft solution and end with "Final Answer: <answer>".'
        )
        draft_prompt = base_instruction + user_part
        draft_output = self._generate(model, draft_prompt, gen_config)

        current_solution = draft_output
        feedback_text = ""

        # Step 2+: self-feedback and refinement
        for _ in range(self.num_refine_steps):
            feedback_prompt = (
                "You are an expert math teacher.\n\n"
                f"Problem:\n{problem.question}\n\n"
                f"Draft solution:\n{current_solution}\n\n"
                "First, analyze whether the draft solution is correct. "
                "List possible mistakes or unclear steps. "
                "Then propose how to improve the solution.\n"
            )
            feedback_output = self._generate(model, feedback_prompt, gen_config)
            feedback_text = feedback_output

            refine_prompt = (
                "You are an expert math problem solver.\n\n"
                f"Problem:\n{problem.question}\n\n"
                f"Draft solution:\n{current_solution}\n\n"
                f"Feedback on the draft:\n{feedback_text}\n\n"
                "Now write an improved solution that fixes the issues. "
                'Think step by step and end with "Final Answer: <answer>".'
            )
            refined_output = self._generate(model, refine_prompt, gen_config)
            current_solution = refined_output

        final_output = current_solution
        final_answer = self._extract_final_answer(final_output)
        normalized = None
        if final_answer is not None:
            normalized = self._normalize_for_dataset(final_output, problem.dataset)
        return ParsedAnswer(raw_output=final_output, final_answer=final_answer, normalized=normalized)

