from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

from math_reasoning_experiments.data.schema import NormalizedAnswer, ProblemInstance
from math_reasoning_experiments.data.normalization import normalize_prediction
from math_reasoning_experiments.models.backends import ModelBackend
from math_reasoning_experiments.config.experiment_config import GenerationConfig
from math_reasoning_experiments.models.generation import generate_with_config


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
        marker = "Final Answer:"
        idx = output_text.rfind(marker)
        if idx == -1:
            return None
        after = output_text[idx + len(marker) :]
        return after.strip().splitlines()[0].strip()

    @staticmethod
    def _normalize_for_dataset(raw_output: str, dataset: str) -> NormalizedAnswer:
        return normalize_prediction(raw_output, dataset=dataset)  # type: ignore[arg-type]

    def _generate(self, model: ModelBackend, prompt: str, gen_config: GenerationConfig) -> str:
        return generate_with_config(model, prompt, gen_config, extra_kwargs=None)

