from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from statistics import mean, median
from typing import Iterable, List, Mapping, Sequence

from transformers import PreTrainedTokenizerBase

from math_reasoning_experiments.data.schema import NormalizedAnswer


@dataclass
class ExampleResult:
    gold: NormalizedAnswer
    pred: NormalizedAnswer | None
    raw_output: str
    correct: bool


def _float_equal(a: float, b: float, eps: float = 1e-6) -> bool:
    return abs(a - b) <= eps


def _answers_equal(gold: NormalizedAnswer, pred: NormalizedAnswer | None) -> bool:
    if pred is None:
        return False
    if gold.type in {"integer", "float"} and pred.type in {"integer", "float"}:
        try:
            return _float_equal(float(gold.normalized), float(pred.normalized))
        except Exception:
            return False
    return gold.normalized == pred.normalized


def accuracy(results: Sequence[ExampleResult]) -> float:
    if not results:
        return 0.0
    num_correct = sum(1 for r in results if r.correct)
    return num_correct / len(results)


def response_length_stats(outputs: Iterable[str], tokenizer: PreTrainedTokenizerBase) -> Mapping[str, float]:
    lengths: List[int] = []
    for text in outputs:
        if not text:
            lengths.append(0)
            continue
        tokens = tokenizer.encode(text, add_special_tokens=False)
        lengths.append(len(tokens))
    if not lengths:
        return {"mean": 0.0, "median": 0.0, "std": 0.0}
    mu = mean(lengths)
    med = median(lengths)
    var = mean((l - mu) ** 2 for l in lengths)
    return {"mean": mu, "median": med, "std": var**0.5}


def reasoning_depth_score(output: str) -> int:
    lines = [ln.strip() for ln in output.splitlines() if ln.strip()]
    num_steps = 0
    num_subproblems = 0

    in_subproblems = False
    for line in lines:
        lower = line.lower()
        if lower.startswith("subproblems:"):
            in_subproblems = True
            continue
        if in_subproblems:
            if line.startswith("-"):
                num_subproblems += 1
            elif not line:
                in_subproblems = False
        if any(
            lower.startswith(prefix)
            for prefix in ("step ", "step1", "step 1", "1.", "2.", "3.", "- step", "* step")
        ):
            num_steps += 1

    depth = num_steps + num_subproblems
    return min(depth, 20)


def reasoning_depth_stats(outputs: Iterable[str]) -> Mapping[str, float]:
    scores = [reasoning_depth_score(o) for o in outputs]
    if not scores:
        return {"mean": 0.0, "median": 0.0, "histogram_mode": 0.0}
    mu = mean(scores)
    med = median(scores)
    hist = Counter(scores)
    mode_score, _ = max(hist.items(), key=lambda kv: kv[1])
    return {"mean": mu, "median": med, "histogram_mode": float(mode_score)}

