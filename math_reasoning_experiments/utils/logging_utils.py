from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass
class ResultRecord:
    problem_id: str
    dataset: str
    model_name: str
    prompt_method: str
    question: str
    gold_answer: str
    pred_answer: str | None
    correct: bool
    raw_output: str
    metrics: Dict[str, Any]


def append_result_jsonl(path: Path, record: ResultRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

