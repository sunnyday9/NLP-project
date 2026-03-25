from __future__ import annotations

import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from datasets import load_dataset


DATA_DIR = Path("data")


def ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def download_gsm8k_test() -> Path:
    """
    下载完整 GSM8K test 集，并保存为 JSONL：data/gsm8k_test.jsonl
    """
    print("Downloading GSM8K (main)...")
    ds = load_dataset("gsm8k", "main")
    test = ds["test"]

    out_path = DATA_DIR / "gsm8k_test.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for ex in test:
            rec = {
                "question": ex["question"],
                "answer": ex["answer"],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"GSM8K test saved to {out_path} (n={len(test)})")
    return out_path


def download_math_full() -> Path:
    """
    下载 EleutherAI/hendrycks_math 所有科目的 test 集，并合并保存为 JSON：data/math_all.json

    说明：
    - 你的 loader.load_math500 对样本数量没有限制，
      因此虽然文件名不叫 math500，你仍可以在 config 中映射为 \"math500\"。
    """
    print("Downloading EleutherAI/hendrycks_math (all configs, test split)...")
    subjects = [
        "algebra",
        "counting_and_probability",
        "geometry",
        "intermediate_algebra",
        "number_theory",
        "prealgebra",
        "precalculus",
    ]

    records = []
    for subject in subjects:
        print(f"  -> loading subject: {subject}")
        ds = load_dataset("EleutherAI/hendrycks_math", subject)
        test = ds["test"]
        for i, ex in enumerate(test):
            rec = {
                "id": ex.get("id", f"{subject}-{i+1:05d}"),
                "problem": ex.get("problem", ""),
                "solution": ex.get("solution", ""),
                "level": ex.get("level", None),
                "type": ex.get("type", None),
                "subject": subject,
            }
            records.append(rec)

    out_path = DATA_DIR / "math_all.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"MATH all-subject test saved to {out_path} (n={len(records)})")
    return out_path


def download_aime24_hf() -> Path:
    """
    从 HuggingFace 数据集 math-ai/aime24 下载 AIME 2024 题目，
    并转换为 data/aime2024.json（字段: id, problem_text, answer）。
    """
    print("Downloading math-ai/aime24 from HuggingFace...")
    ds = load_dataset("math-ai/aime24")

    # 数据集中可能只有一个 split（例如 'train' 或 'test'），优先使用 'test'
    if "test" in ds:
        split = ds["test"]
    else:
        # 回退到第一个可用 split
        split = next(iter(ds.values()))

    records = []
    for i, ex in enumerate(split):
        # 尝试多种常见字段名
        problem_text = (
            ex.get("problem_text")
            or ex.get("question")
            or ex.get("problem")
            or ""
        )
        answer = ex.get("answer") or ex.get("solution") or ex.get("label") or ""

        rec = {
            "id": f"AIME2024-{i+1:02d}",
            "problem_text": str(problem_text),
            "answer": str(answer).strip(),
        }
        records.append(rec)

    out_path = DATA_DIR / "aime2024.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"AIME 2024 problems saved to {out_path} (n={len(records)})")
    return out_path


def main() -> None:
    ensure_data_dir()
    download_gsm8k_test()
    download_math_full()
    download_aime24_hf()


if __name__ == "__main__":
    main()

