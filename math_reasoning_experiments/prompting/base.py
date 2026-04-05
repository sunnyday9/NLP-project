from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

from math_reasoning_experiments.data.schema import NormalizedAnswer, ProblemInstance
from math_reasoning_experiments.data.normalization import normalize_prediction
from math_reasoning_experiments.models.backends import ModelBackend
from math_reasoning_experiments.config.experiment_config import GenerationConfig
from math_reasoning_experiments.models.generation import generate_with_config


# Matches the last \boxed{...} in a string — used as a fallback when the model
# writes its answer in boxed LaTeX rather than with the "Final Answer:" marker.
_BOXED_ANSWER_RE = re.compile(r"\\boxed\{([^}]*)\}")


@dataclass
class ParsedAnswer:
    raw_output: str
    final_answer: Optional[str]
    normalized: Optional[NormalizedAnswer]


class PromptMethod(ABC):
    name: str

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def run(
        self,
        model: ModelBackend,
        problem: ProblemInstance,
        gen_config: GenerationConfig,
        **kwargs: Any,
    ) -> ParsedAnswer:
        raise NotImplementedError

    @staticmethod
    def _extract_final_answer(output_text: str) -> Optional[str]:
        """Extract the model's final answer from its generated output.

        Strategy:
        1. Look for the last "Final Answer:" marker the model wrote.
           Skip the match if it still contains '<answer>' — that means the
           model was cut off before writing a real answer and the regex hit
           the prompt's own example phrase.
        2. Fall back to the last \\boxed{...} occurrence, which reasoning
           models (DeepSeek-R1, Qwen-Math) commonly use.
        """
        marker = "Final Answer:"
        idx = output_text.rfind(marker)
        if idx != -1:
            after = output_text[idx + len(marker):]
            line = after.strip().splitlines()[0].strip()
            # Guard: if '<answer>' is still present the model was cut off
            # and never replaced the placeholder — ignore this match.
            if line and "<answer>" not in line:
                return line

        # Fallback: extract the last \boxed{...} the model wrote.
        matches = _BOXED_ANSWER_RE.findall(output_text)
        if matches:
            return matches[-1].strip()

        return None

    @staticmethod
    def _normalize_for_dataset(raw_output: str, dataset: str) -> NormalizedAnswer:
        return normalize_prediction(raw_output, dataset=dataset)  # type: ignore[arg-type]

    def _generate(self, model: ModelBackend, prompt: str, gen_config: GenerationConfig) -> str:
        return generate_with_config(model, prompt, gen_config, extra_kwargs=None)
