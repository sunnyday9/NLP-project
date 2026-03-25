from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Literal, Optional


DatasetName = Literal["math500", "gsm8k", "aime2024"]


class DatasetEnum(str, Enum):
    MATH500 = "math500"
    GSM8K = "gsm8k"
    AIME2024 = "aime2024"


@dataclass
class ProblemInstance:
    problem_id: str
    dataset: DatasetName
    question: str
    answer: str
    meta: Optional[Dict[str, Any]] = None


@dataclass
class NormalizedAnswer:
    raw: str
    normalized: str
    type: Literal["integer", "float", "expression", "string"]

