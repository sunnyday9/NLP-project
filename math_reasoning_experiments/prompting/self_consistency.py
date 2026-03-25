from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional

from math_reasoning_experiments.config.experiment_config import GenerationConfig
from math_reasoning_experiments.data.schema import ProblemInstance
from math_reasoning_experiments.models.backends import ModelBackend
from math_reasoning_experiments.models.generation import generate_with_config

from .base import ParsedAnswer, PromptMethod


class SelfConsistencyPromptMethod(PromptMethod):
    """
    Self-consistency: generate multiple candidate solutions, extract each "Final Answer",
    then pick the most frequent answer (majority vote).
    """

    def __init__(self, num_samples: int = 5) -> None:
        super().__init__(name="self_consistency")
        self.num_samples = max(1, num_samples)

    def _extract_vote(self, output: str) -> Optional[str]:
        return self._extract_final_answer(output)

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

        # Force sampling so multiple generations are diverse.
        # (HF generate defaults to greedy decoding unless do_sample=True.)
        outputs: List[str] = []
        for _ in range(self.num_samples):
            out = generate_with_config(
                model,
                prompt,
                gen_config,
                extra_kwargs={"do_sample": True},
            )
            outputs.append(out)

        extracted: List[Optional[str]] = [self._extract_vote(o) for o in outputs]
        candidates = [a for a in extracted if a is not None]

        chosen_answer: Optional[str] = None
        chosen_index: int = 0
        if candidates:
            counts = Counter(candidates)
            max_count = max(counts.values())
            # Tie-break: choose the earliest occurrence among the top-count answers.
            top_answers = {a for a, c in counts.items() if c == max_count}
            for idx, a in enumerate(extracted):
                if a in top_answers:
                    chosen_answer = a
                    chosen_index = idx
                    break

        # Choose the output corresponding to the selected vote (if any),
        # otherwise just use the first generation.
        best_output = outputs[chosen_index] if outputs else ""
        normalized = self._normalize_for_dataset(best_output, problem.dataset)

        return ParsedAnswer(
            raw_output=best_output,
            final_answer=chosen_answer,
            normalized=normalized,
        )

