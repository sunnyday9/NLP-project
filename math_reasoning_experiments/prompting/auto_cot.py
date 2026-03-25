from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Sequence

from math_reasoning_experiments.config.experiment_config import GenerationConfig
from math_reasoning_experiments.data.schema import ProblemInstance
from math_reasoning_experiments.models.backends import ModelBackend

from .base import ParsedAnswer, PromptMethod


@dataclass
class AutoCotExample:
    question: str
    cot_solution: str
    final_answer: str


def load_auto_cot_examples(path: str | Path, max_examples: int | None = None) -> List[AutoCotExample]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    examples: List[AutoCotExample] = []
    for rec in data:
        examples.append(
            AutoCotExample(
                question=str(rec["question"]),
                cot_solution=str(rec["cot_solution"]),
                final_answer=str(rec.get("final_answer", "")),
            )
        )
        if max_examples is not None and len(examples) >= max_examples:
            break
    return examples


def build_auto_cot_examples_from_problems(
    problems: Sequence[ProblemInstance],
    max_examples: int = 50,
) -> List[AutoCotExample]:
    """
    Build simple Auto-CoT examples directly from dataset problems.

    Each example uses:
    - question: the original problem.question
    - cot_solution: a minimal CoT-style wrapper ending with "Final Answer: <answer>"
    - final_answer: the (normalized) gold answer string
    """
    examples: List[AutoCotExample] = []
    for p in problems:
        if len(examples) >= max_examples:
            break
        ans = str(p.answer)
        cot = (
            "Let's carefully solve this math problem step by step.\n\n"
            f"Problem: {p.question}\n\n"
            "Reasoning: (omitted here; this is a reference example constructed from the gold answer.)\n\n"
            f"Final Answer: {ans}"
        )
        examples.append(
            AutoCotExample(
                question=str(p.question),
                cot_solution=cot,
                final_answer=ans,
            )
        )
    return examples


class AutoCoTPromptMethod(PromptMethod):
    def __init__(self, examples: Sequence[AutoCotExample], k: int = 3) -> None:
        super().__init__(name="auto_cot")
        self.examples: List[AutoCotExample] = list(examples)
        self.k = max(1, min(k, len(self.examples))) if self.examples else 0

    def _select_examples(self, problem: ProblemInstance) -> Iterable[AutoCotExample]:
        """
        Select few-shot examples for the given problem.

        We avoid using examples whose question text is identical to the
        current problem.question, so that the exact evaluation item is not
        reused as a few-shot example.
        """
        if not self.examples or self.k <= 0:
            return []

        filtered = [
            ex for ex in self.examples
            if ex.question.strip() != problem.question.strip()
        ]
        if not filtered:
            return []
        return filtered[: self.k]

    def run(
        self,
        model: ModelBackend,
        problem: ProblemInstance,
        gen_config: GenerationConfig,
        **kwargs: Any,
    ) -> ParsedAnswer:
        parts: List[str] = ["You are an expert math problem solver. Here are some examples.\n\n"]
        for i, ex in enumerate(self._select_examples(problem), start=1):
            parts.append(
                f"Example {i}:\n"
                f"Problem: {ex.question}\n"
                f"Solution:\n{ex.cot_solution}\n"
                f"Final Answer: {ex.final_answer}\n\n"
            )
        parts.append(
            "Now solve the following problem in a similar way.\n\n"
            f"Problem: {problem.question}\n"
            "Solution:"
        )
        prompt = "".join(parts)
        output = self._generate(model, prompt, gen_config)

        # Extract Final Answer from the part after the last occurrence of the current question
        search_region = output
        q_idx = output.rfind(problem.question[:40])
        if q_idx != -1:
            search_region = output[q_idx:]
        final_answer = self._extract_final_answer(search_region)

        normalized = None
        if final_answer is not None:
            normalized = self._normalize_for_dataset(output, problem.dataset)
        return ParsedAnswer(raw_output=output, final_answer=final_answer, normalized=normalized)

