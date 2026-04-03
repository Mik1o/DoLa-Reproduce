"""Evaluate a small TruthfulQA-MC subset with vanilla candidate scoring."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.generation import score_candidate_answers
from src.metrics import aggregate_mc_metrics, compute_mc_metrics, format_metrics
from src.modeling import load_model_and_tokenizer
from src.truthfulqa_mc import (
    TruthfulQAEvalResult,
    build_mc_prompt,
    get_mc_candidate_sets,
    load_truthfulqa_samples,
)
from src.utils import ensure_output_dir, load_yaml_config



def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for subset evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate a small TruthfulQA-MC subset with a tiny HF causal LM."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "hf_tiny_eval_subset.yaml",
        help="Path to a YAML config file.",
    )
    return parser.parse_args()



def main() -> None:
    """Run vanilla MC scoring on the first N samples and save JSON outputs."""
    args = parse_args()
    config = load_yaml_config(args.config)
    output_dir = ensure_output_dir(
        config.get("output_dir", PROJECT_ROOT / "outputs" / "hf_tiny_eval_subset")
    )

    model_name = str(config["model_name"])
    device = config.get("device", "cpu")
    csv_path = Path(str(config["csv_path"]))
    max_samples = int(config.get("max_samples", 1))
    if max_samples <= 0:
        raise ValueError("max_samples must be a positive integer.")

    model_kwargs: dict[str, object] = {}
    if "use_safetensors" in config:
        model_kwargs["use_safetensors"] = bool(config["use_safetensors"])

    samples = load_truthfulqa_samples(csv_path)
    subset = samples[:max_samples]
    if not subset:
        raise ValueError("No samples were loaded for evaluation.")

    print(f"[hf_eval_mc_subset] Model: {model_name}")
    print(f"[hf_eval_mc_subset] Device: {device}")
    print(f"[hf_eval_mc_subset] Loaded samples: {len(samples)}")
    print(f"[hf_eval_mc_subset] Evaluating first {len(subset)} samples")
    print(f"[hf_eval_mc_subset] Output directory: {output_dir}")

    model, tokenizer = load_model_and_tokenizer(
        model_name=model_name,
        device=device,
        **model_kwargs,
    )

    sample_results: list[TruthfulQAEvalResult] = []
    metric_rows: list[dict[str, float]] = []

    for index, sample in enumerate(subset):
        prompt = build_mc_prompt(sample)
        true_candidates, false_candidates = get_mc_candidate_sets(sample)
        true_scored = score_candidate_answers(model, tokenizer, prompt, true_candidates)
        false_scored = score_candidate_answers(model, tokenizer, prompt, false_candidates)

        scores_true = [score for _, score in true_scored]
        scores_false = [score for _, score in false_scored]
        metrics = compute_mc_metrics(scores_true, scores_false)
        metric_rows.append(metrics)

        sample_results.append(
            TruthfulQAEvalResult(
                question=sample.question,
                prompt=prompt,
                true_candidates=true_candidates,
                false_candidates=false_candidates,
                scores_true=scores_true,
                scores_false=scores_false,
                mc1=metrics["MC1"],
                mc2=metrics["MC2"],
                mc3=metrics["MC3"],
            )
        )
        print(
            f"[hf_eval_mc_subset] Sample {index}: "
            f"MC1={metrics['MC1']:.4f}, MC2={metrics['MC2']:.4f}, MC3={metrics['MC3']:.4f}"
        )

    summary = aggregate_mc_metrics(metric_rows)
    summary.update(
        {
            "task_name": str(config.get("task_name", "hf_tiny_eval_subset")),
            "model_name": model_name,
            "csv_path": str(csv_path),
            "output_dir": str(output_dir),
        }
    )

    sample_results_path = output_dir / "sample_results.jsonl"
    with sample_results_path.open("w", encoding="utf-8") as handle:
        for result in sample_results:
            handle.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")

    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(f"[hf_eval_mc_subset] Summary: {format_metrics(summary)}")
    print(f"[hf_eval_mc_subset] Saved sample results to: {sample_results_path}")
    print(f"[hf_eval_mc_subset] Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()