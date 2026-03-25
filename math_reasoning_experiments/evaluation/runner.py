from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

from transformers import AutoTokenizer

from math_reasoning_experiments.config.experiment_config import ExperimentConfig
from math_reasoning_experiments.data.loader import load_all_datasets
from math_reasoning_experiments.data.schema import DatasetName, ProblemInstance
from math_reasoning_experiments.data.normalization import normalize_gold_answer
from math_reasoning_experiments.evaluation.metrics import ExampleResult, accuracy, reasoning_depth_stats, response_length_stats
from math_reasoning_experiments.models.backends import DeepSeekR1Backend, ModelBackend, QwenMathBackend
from math_reasoning_experiments.prompting.auto_cot import AutoCoTPromptMethod, AutoCotExample
from math_reasoning_experiments.prompting.base import ParsedAnswer
from math_reasoning_experiments.prompting.cot import CoTPromptMethod
from math_reasoning_experiments.prompting.least_to_most import LeastToMostPromptMethod
from math_reasoning_experiments.prompting.self_consistency import SelfConsistencyPromptMethod
from math_reasoning_experiments.prompting.self_refine import SelfRefinePromptMethod
from math_reasoning_experiments.utils.logging_utils import ResultRecord, append_result_jsonl


def _init_model_backend(model_name: str, cache_dir: Path | None = None) -> ModelBackend:
    c_str = str(cache_dir) if cache_dir else None
    if "Qwen2.5-Math" in model_name or "Qwen2.5-Math-1.5B" in model_name:
        return QwenMathBackend(model_name=model_name, cache_dir=c_str)
    if "DeepSeek-R1" in model_name or "DeepSeek-R1-Distill-Qwen-1.5B" in model_name:
        return DeepSeekR1Backend(model_name=model_name, cache_dir=c_str)
    # fallback generic HF backend
    return QwenMathBackend(model_name=model_name, cache_dir=c_str)


def _init_prompt_methods(auto_cot_examples: Sequence[AutoCotExample] | None = None):
    methods = {
        "cot": CoTPromptMethod(),
        "self_refine": SelfRefinePromptMethod(num_refine_steps=1),
        "self_consistency": SelfConsistencyPromptMethod(num_samples=5),
        "least_to_most": LeastToMostPromptMethod(),
    }
    if auto_cot_examples is not None and len(auto_cot_examples) > 0:
        methods["auto_cot"] = AutoCoTPromptMethod(list(auto_cot_examples), k=3)
    return methods


def run_experiment(
    config: ExperimentConfig,
    auto_cot_examples: Optional[Sequence[AutoCotExample]] = None,
    datasets_override: Optional[Dict[DatasetName, List[ProblemInstance]]] = None,
) -> None:
    datasets = datasets_override if datasets_override is not None else load_all_datasets(config.data_paths)
    prompt_methods = _init_prompt_methods(auto_cot_examples)

    for model_name in config.models:
        backend = _init_model_backend(model_name, config.model_cache_dir)
        tokenizer = backend.tokenizer

        for method_name in config.prompt_methods:
            if method_name not in prompt_methods:
                continue
            method = prompt_methods[method_name]

            for dataset_name, problems in datasets.items():
                results: List[ExampleResult] = []
                outputs: List[str] = []

                out_path = (
                    config.output_dir
                    / f"{dataset_name}"
                    / f"{backend.name().replace('/', '_')}_{method.name}.jsonl"
                )

                for problem in problems:
                    gold_norm = normalize_gold_answer(problem.answer, dataset_name)
                    parsed: ParsedAnswer = method.run(backend, problem, config.generation)
                    pred_norm = None
                    if parsed.normalized is not None:
                        pred_norm = parsed.normalized
                    correct = (
                        parsed.normalized is not None
                        and gold_norm is not None
                        and gold_norm.normalized == parsed.normalized.normalized
                    )
                    outputs.append(parsed.raw_output)
                    results.append(
                        ExampleResult(
                            gold=gold_norm,
                            pred=pred_norm,
                            raw_output=parsed.raw_output,
                            correct=correct,
                        )
                    )

                    record = ResultRecord(
                        problem_id=problem.problem_id,
                        dataset=dataset_name,
                        model_name=backend.name(),
                        prompt_method=method.name,
                        question=problem.question,
                        gold_answer=gold_norm.raw,
                        pred_answer=parsed.final_answer,
                        correct=correct,
                        raw_output=parsed.raw_output,
                        metrics={},
                    )
                    append_result_jsonl(out_path, record)

                # Compute aggregate metrics for this combo
                acc = accuracy(results)
                length_stats = response_length_stats(outputs, tokenizer)
                depth_stats = reasoning_depth_stats(outputs)

                summary = {
                    "accuracy": acc,
                    "response_length": length_stats,
                    "reasoning_depth": depth_stats,
                }
                summary_path = out_path.with_suffix(".summary.json")
                summary_path.parent.mkdir(parents=True, exist_ok=True)
                summary_path.write_text(str(summary), encoding="utf-8")

