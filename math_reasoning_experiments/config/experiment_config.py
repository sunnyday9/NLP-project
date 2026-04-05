from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from math_reasoning_experiments.data.schema import DatasetName


@dataclass
class GenerationConfig:
    temperature: float = 0.2
    top_p: float = 1.0
    max_new_tokens: int = 8192


@dataclass
class ExperimentConfig:
    data_paths: Dict[DatasetName, Path]
    models: List[str]
    prompt_methods: List[str]
    output_dir: Path
    model_cache_dir: Path | None = None
    seed: int = 42
    generation: GenerationConfig = field(default_factory=GenerationConfig)

