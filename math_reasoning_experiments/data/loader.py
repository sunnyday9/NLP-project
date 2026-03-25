from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

from .normalization import normalize_gold_answer
from .schema import DatasetEnum, DatasetName, NormalizedAnswer, ProblemInstance


def _read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_math500(path: str | Path) -> List[ProblemInstance]:
    """
    Load MATH-500 dataset from JSON/JSONL/CSV-like JSON.
    Expected fields (flexible): problem, solution or answer, level, type, id.
    """
    p = Path(path)
    items: List[ProblemInstance] = []

    if p.suffix == ".jsonl":
        records = list(_read_jsonl(p))
    else:
        with p.open("r", encoding="utf-8") as f:
            records = json.load(f)

    for idx, rec in enumerate(records):
        question = rec.get("problem") or rec.get("question") or rec.get("prompt")
        if not question:
            continue
        raw_answer = rec.get("answer") or rec.get("solution") or ""
        pid = rec.get("id") or f"MATH500-{idx:03d}"
        norm = normalize_gold_answer(str(raw_answer), "math500")
        items.append(
            ProblemInstance(
                problem_id=str(pid),
                dataset=DatasetEnum.MATH500.value,
                question=str(question),
                answer=norm.raw,
                meta={k: v for k, v in rec.items() if k not in {"problem", "question", "prompt", "answer", "solution"}},
            )
        )
    return items


def load_gsm8k_test(path: str | Path) -> List[ProblemInstance]:
    """
    Load GSM8K test set from JSON/JSONL.
    Expected fields: question, answer, id (optional).
    """
    p = Path(path)
    items: List[ProblemInstance] = []

    if p.suffix == ".jsonl":
        records = list(_read_jsonl(p))
    else:
        with p.open("r", encoding="utf-8") as f:
            records = json.load(f)

    for idx, rec in enumerate(records):
        question = rec.get("question")
        if not question:
            continue
        raw_answer = rec.get("answer") or ""
        pid = rec.get("id") or f"GSM8K-test-{idx}"
        norm = normalize_gold_answer(str(raw_answer), "gsm8k")
        items.append(
            ProblemInstance(
                problem_id=str(pid),
                dataset=DatasetEnum.GSM8K.value,
                question=str(question),
                answer=norm.raw,
                meta={k: v for k, v in rec.items() if k not in {"question", "answer"}},
            )
        )
    return items


def load_aime2024(path: str | Path) -> List[ProblemInstance]:
    """
    Load AIME 2024 dataset from JSON/JSONL.
    Expected fields: problem_text/question, answer, id/index/year (optional).
    """
    p = Path(path)
    items: List[ProblemInstance] = []

    if p.suffix == ".jsonl":
        records = list(_read_jsonl(p))
    else:
        with p.open("r", encoding="utf-8") as f:
            records = json.load(f)

    for idx, rec in enumerate(records):
        question = rec.get("problem_text") or rec.get("question")
        if not question:
            continue
        raw_answer = rec.get("answer") or ""
        pid = rec.get("id") or rec.get("index") or f"AIME2024-{idx + 1:02d}"
        norm = normalize_gold_answer(str(raw_answer), "aime2024")
        items.append(
            ProblemInstance(
                problem_id=str(pid),
                dataset=DatasetEnum.AIME2024.value,
                question=str(question),
                answer=norm.raw,
                meta={k: v for k, v in rec.items() if k not in {"problem_text", "question", "answer"}},
            )
        )
    return items


def load_all_datasets(paths: Dict[DatasetName, str | Path]) -> Dict[DatasetName, List[ProblemInstance]]:
    dataset_map: Dict[DatasetName, List[ProblemInstance]] = {}
    if "math500" in paths:
        dataset_map["math500"] = load_math500(paths["math500"])
    if "gsm8k" in paths:
        dataset_map["gsm8k"] = load_gsm8k_test(paths["gsm8k"])
    if "aime2024" in paths:
        dataset_map["aime2024"] = load_aime2024(paths["aime2024"])
    return dataset_map

