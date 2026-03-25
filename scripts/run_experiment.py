from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from math_reasoning_experiments.data.loader import load_all_datasets
from math_reasoning_experiments.data.schema import DatasetName, ProblemInstance
from math_reasoning_experiments.config.experiment_config import ExperimentConfig, GenerationConfig
from math_reasoning_experiments.evaluation.runner import run_experiment
from math_reasoning_experiments.prompting.auto_cot import AutoCotExample, build_auto_cot_examples_from_problems


def _sample_n(rng: random.Random, items: list[ProblemInstance], n: int) -> list[ProblemInstance]:
    if n >= len(items):
        return list(items)
    return rng.sample(items, n)


def _stratified_sample_by_topic(
    rng: random.Random, items: list[ProblemInstance], n: int, topic_key: str = "type"
) -> list[ProblemInstance]:
    """
    Equal distribution across topics for MATH-500:
    - topic is read from problem.meta[topic_key] (falls back to "unknown")
    - allocate counts as evenly as possible across topics
    - if topics > n, sample n topics and take 1 from each
    """
    if n >= len(items):
        return list(items)

    buckets: dict[str, list[ProblemInstance]] = {}
    for p in items:
        meta = p.meta or {}
        topic = meta.get(topic_key) or "unknown"
        buckets.setdefault(str(topic), []).append(p)

    topics = sorted(buckets.keys())
    if not topics:
        return _sample_n(rng, items, n)

    # Shuffle within each bucket for randomness + determinism via rng
    for t in topics:
        rng.shuffle(buckets[t])

    if len(topics) >= n:
        chosen_topics = rng.sample(topics, n)
        return [buckets[t][0] for t in chosen_topics if buckets[t]]

    base = n // len(topics)
    rem = n % len(topics)

    # Start with equal base allocation (capped by bucket size)
    selected: list[ProblemInstance] = []
    remaining: list[ProblemInstance] = []

    for t in topics:
        take = min(base, len(buckets[t]))
        selected.extend(buckets[t][:take])
        remaining.extend(buckets[t][take:])

    # Distribute remainder: first try one extra from as many topics as possible
    extra_topics = topics[:]
    rng.shuffle(extra_topics)
    for t in extra_topics:
        if len(selected) >= n:
            break
        if buckets[t][base:base + 1]:
            # take one more if available after base slice
            idx = base
            if idx < len(buckets[t]):
                selected.append(buckets[t][idx])

    # If still short (some topics too small), fill uniformly from remaining pool
    if len(selected) < n:
        rng.shuffle(remaining)
        need = n - len(selected)
        selected.extend(remaining[:need])

    return selected[:n]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to JSON experiment config.")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg_raw = json.load(f)

    gen_cfg = GenerationConfig(**cfg_raw.get("generation", {}))
    data_paths = {k: Path(v) for k, v in cfg_raw["data_paths"].items()}
    exp_cfg = ExperimentConfig(
        data_paths=data_paths,
        models=cfg_raw["models"],
        prompt_methods=cfg_raw["prompt_methods"],
        output_dir=Path(cfg_raw["output_dir"]),
        model_cache_dir=Path(cfg_raw["model_cache_dir"]) if "model_cache_dir" in cfg_raw else None,
        seed=cfg_raw.get("seed", 42),
        generation=gen_cfg,
    )

    # Load datasets once, then sample:
    # - gsm8k: 50 questions
    # - aime2024: all questions (typically 30)
    # - math500: 50 questions with equal distribution across topics (meta["type"])
    rng = random.Random(exp_cfg.seed)
    datasets = load_all_datasets(exp_cfg.data_paths)

    # If auto_cot is requested, first reserve a small disjoint subset of problems
    # as few-shot examples, then sample evaluation problems from the remaining pool.
    want_auto_cot = "auto_cot" in exp_cfg.prompt_methods
    example_ids: dict[DatasetName, set[str]] = {  # type: ignore[typeddict-item]
        "math500": set(),
        "gsm8k": set(),
        "aime2024": set(),
    }
    auto_cot_examples: list[AutoCotExample] = []

    if want_auto_cot:
        # Simple heuristic: take up to 10 examples per dataset (if available),
        # then build Auto-CoT examples from their gold answers.
        per_dataset_max = 10
        for name, problems in datasets.items():
            pool = list(problems)
            rng.shuffle(pool)
            take = min(per_dataset_max, len(pool))
            chosen = pool[:take]
            example_ids[name].update(p.problem_id for p in chosen)
            auto_cot_examples.extend(build_auto_cot_examples_from_problems(chosen))

    sampled: dict[DatasetName, list[ProblemInstance]] = {}

    if "gsm8k" in datasets:
        candidates = [
            p for p in datasets["gsm8k"]
            if not want_auto_cot or p.problem_id not in example_ids["gsm8k"]
        ]
        sampled["gsm8k"] = _sample_n(rng, candidates, 50)
    if "aime2024" in datasets:
        candidates = [
            p for p in datasets["aime2024"]
            if not want_auto_cot or p.problem_id not in example_ids["aime2024"]
        ]
        sampled["aime2024"] = list(candidates)
    if "math500" in datasets:
        candidates = [
            p for p in datasets["math500"]
            if not want_auto_cot or p.problem_id not in example_ids["math500"]
        ]
        sampled["math500"] = _stratified_sample_by_topic(rng, candidates, 50, topic_key="type")

    run_experiment(exp_cfg, auto_cot_examples if want_auto_cot else None, datasets_override=sampled)


if __name__ == "__main__":
    main()

