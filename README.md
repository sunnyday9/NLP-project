## Math Reasoning Experiments

This project compares different prompting strategies (CoT, Self-Refine, Auto-CoT, Least-to-Most)
on math reasoning benchmarks (MATH-500, GSM8K test, AIME 2024) using two HuggingFace models:
`Qwen/Qwen2.5-Math-1.5B` and `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`.

### Structure

- `math_reasoning_experiments/`
  - `data/`: dataset schema, normalization, and loaders.
  - `models/`: HuggingFace model backends and generation helpers.
  - `prompting/`: implementations of CoT, Self-Refine, Auto-CoT, and Least-to-Most prompting.
  - `evaluation/`: metrics and experiment runner.
  - `config/`: experiment configuration dataclasses.
- `scripts/`
  - `run_experiment.py`: CLI entrypoint to run the full grid of experiments.

### Quick usage

1. Prepare JSON configs and dataset files.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run experiments, for example:

```bash
python scripts/run_experiment.py --config configs/experiment_math.json --auto_cot_examples data/auto_cot_examples.json
```

