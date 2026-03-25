from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict

from math_reasoning_experiments.config.experiment_config import GenerationConfig
from .backends import ModelBackend


def generate_with_config(
    model: ModelBackend,
    prompt: str,
    gen_config: GenerationConfig,
    extra_kwargs: Dict[str, Any] | None = None,
) -> str:
    params: Dict[str, Any] = asdict(gen_config)
    if extra_kwargs:
        params.update(extra_kwargs)
    return model.generate(prompt, **params)

