"""Compare vanilla and DoLa-style MC scoring over a small TruthfulQA subset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dola_utils import describe_dola_pair, get_mature_layer_index, validate_premature_layer
from src.generation import score_candidate_answers_dola_with_details, score_candidate_answers_with_details
from src.metrics import (
    aggregate_mc_metrics,
    compare_aggregate_metrics,
    compute_mc_metrics,
    format_metrics,
)
from src.modeling import load_model_and_tokenizer
from src.truthfulqa_mc import TruthfulQASample, build_mc_prompt, get_mc_candidate_sets, load_truthfulqa_samples
from src.utils import ensure_output_dir, load_yaml_config


LOAD_CONFIG_KEYS = (
    "use_safetensors",
    "torch_dtype",
    "use_fast_tokenizer",
    "trust_remote_code",
    "attn_implementation",
)



def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for subset comparison evaluation."""
    parser = argparse.ArgumentParser(
        description="Compare vanilla and DoLa-style MC scoring over a small subset."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "hf_tiny_compare_subset.yaml",
        help="Path to a YAML config file.",
    )
    return parser.parse_args()



def _serialize_candidate_scores(items: list[object]) -> list[dict[str, object]]:
    """Convert candidate score objects into JSON-friendly dictionaries."""
    return [
        {
            "candidate": item.candidate,
            "score": item.score,
            "continuation_token_count": item.continuation_token_count,
        }
        for item in items
    ]



def evaluate_compare_subset(
    model: Any,
    tokenizer: Any,
    samples: list[TruthfulQASample],
    *,
    max_samples: int,
    premature_layer: int,
    prompt_style: str,
    score_mode: str,
    dola_score_mode: str = "legacy_contrastive",
    post_softmax: bool = False,
    relative_top: float = 0.0,
    relative_top_value: float = -1000.0,
) -> tuple[list[dict[str, object]], dict[str, float | int | str]]:
    """Evaluate vanilla vs DoLa-style scoring on the first N samples."""
    if max_samples <= 0:
        raise ValueError("max_samples must be a positive integer.")

    subset = samples[:max_samples]
    if not subset:
        raise ValueError("No samples were loaded for comparison evaluation.")

    num_hidden_layers = int(getattr(model.config, "num_hidden_layers", 0))
    validate_premature_layer(premature_layer, num_hidden_layers)
    mature_layer = get_mature_layer_index(num_hidden_layers)
    pair_description = describe_dola_pair(premature_layer, mature_layer)

    vanilla_metric_rows: list[dict[str, float]] = []
    dola_metric_rows: list[dict[str, float]] = []
    sample_results: list[dict[str, object]] = []

    for index, sample in enumerate(subset):
        prompt = build_mc_prompt(sample, prompt_style=prompt_style)
        true_candidates, false_candidates = get_mc_candidate_sets(sample, prompt_style=prompt_style)

        vanilla_true = score_candidate_answers_with_details(
            model,
            tokenizer,
            prompt,
            true_candidates,
            score_mode=score_mode,
        )
        vanilla_false = score_candidate_answers_with_details(
            model,
            tokenizer,
            prompt,
            false_candidates,
            score_mode=score_mode,
        )
        vanilla_metrics = compute_mc_metrics(
            [item.score for item in vanilla_true],
            [item.score for item in vanilla_false],
        )
        vanilla_metric_rows.append(vanilla_metrics)

        dola_true = score_candidate_answers_dola_with_details(
            model,
            tokenizer,
            prompt,
            true_candidates,
            premature_layer=premature_layer,
            score_mode=score_mode,
            dola_score_mode=dola_score_mode,
            post_softmax=post_softmax,
            relative_top=relative_top,
            relative_top_value=relative_top_value,
        )
        dola_false = score_candidate_answers_dola_with_details(
            model,
            tokenizer,
            prompt,
            false_candidates,
            premature_layer=premature_layer,
            score_mode=score_mode,
            dola_score_mode=dola_score_mode,
            post_softmax=post_softmax,
            relative_top=relative_top,
            relative_top_value=relative_top_value,
        )
        dola_metrics = compute_mc_metrics(
            [item.score for item in dola_true],
            [item.score for item in dola_false],
        )
        dola_metric_rows.append(dola_metrics)

        sample_results.append(
            {
                "sample_index": index,
                "question": sample.question,
                "prompt": prompt,
                "prompt_style": prompt_style,
                "score_mode": score_mode,
                "dola_score_mode": dola_score_mode,
                "post_softmax": post_softmax,
                "relative_top": relative_top,
                "relative_top_value": relative_top_value,
                "premature_layer": premature_layer,
                "mature_layer": mature_layer,
                "vanilla": {
                    "true_scores": _serialize_candidate_scores(vanilla_true),
                    "false_scores": _serialize_candidate_scores(vanilla_false),
                    "metrics": vanilla_metrics,
                },
                "dola": {
                    "true_scores": _serialize_candidate_scores(dola_true),
                    "false_scores": _serialize_candidate_scores(dola_false),
                    "metrics": dola_metrics,
                },
            }
        )

    vanilla_summary = aggregate_mc_metrics(vanilla_metric_rows)
    dola_summary = aggregate_mc_metrics(dola_metric_rows)
    comparison_summary = compare_aggregate_metrics(vanilla_summary, dola_summary)
    comparison_summary.update(
        {
            "prompt_style": prompt_style,
            "score_mode": score_mode,
            "dola_score_mode": dola_score_mode,
            "post_softmax": post_softmax,
            "relative_top": relative_top,
            "relative_top_value": relative_top_value,
            "premature_layer": premature_layer,
            "mature_layer": mature_layer,
            "dola_pair": pair_description,
        }
    )
    return sample_results, comparison_summary



def main() -> None:
    """Run vanilla and DoLa-style scoring on the first N samples and save results."""
    args = parse_args()
    config = load_yaml_config(args.config)
    output_dir = ensure_output_dir(
        config.get("output_dir", PROJECT_ROOT / "outputs" / "hf_tiny_compare_subset")
    )

    model_name = str(config["model_name"])
    device = config.get("device", "cpu")
    csv_path = Path(str(config["csv_path"]))
    max_samples = int(config.get("max_samples", 1))
    premature_layer = int(config["premature_layer"])
    prompt_style = str(config.get("prompt_style", "plain_mc"))
    score_mode = str(config.get("score_mode", "sum_logprob"))
    dola_score_mode = str(config.get("dola_score_mode", "legacy_contrastive"))
    post_softmax = bool(config.get("post_softmax", False))
    relative_top = float(config.get("relative_top", 0.0))
    relative_top_value = float(config.get("relative_top_value", -1000.0))

    model_kwargs = {key: config[key] for key in LOAD_CONFIG_KEYS if key in config}

    samples = load_truthfulqa_samples(csv_path)

    print(f"[hf_eval_compare_subset] Model: {model_name}")
    print(f"[hf_eval_compare_subset] Device: {device}")
    print(f"[hf_eval_compare_subset] Score mode: {score_mode}")
    print(f"[hf_eval_compare_subset] DoLa score mode: {dola_score_mode}")
    print(f"[hf_eval_compare_subset] Loaded samples: {len(samples)}")
    print(f"[hf_eval_compare_subset] Evaluating first {min(len(samples), max_samples)} samples")
    print(f"[hf_eval_compare_subset] Output directory: {output_dir}")

    model, tokenizer = load_model_and_tokenizer(
        model_name=model_name,
        device=device,
        **model_kwargs,
    )

    sample_results, comparison_summary = evaluate_compare_subset(
        model,
        tokenizer,
        samples,
        max_samples=max_samples,
        premature_layer=premature_layer,
        prompt_style=prompt_style,
        score_mode=score_mode,
        dola_score_mode=dola_score_mode,
        post_softmax=post_softmax,
        relative_top=relative_top,
        relative_top_value=relative_top_value,
    )

    for result in sample_results:
        vanilla_metrics = result["vanilla"]["metrics"]
        dola_metrics = result["dola"]["metrics"]
        print(
            f"[hf_eval_compare_subset] Sample {result['sample_index']}: "
            f"vanilla=({format_metrics(vanilla_metrics)}), "
            f"dola=({format_metrics(dola_metrics)})"
        )

    comparison_summary.update(
        {
            "task_name": str(config.get("task_name", "hf_tiny_compare_subset")),
            "model_name": model_name,
            "csv_path": str(csv_path),
            "output_dir": str(output_dir),
        }
    )

    sample_results_path = output_dir / "compare_sample_results.jsonl"
    with sample_results_path.open("w", encoding="utf-8") as handle:
        for result in sample_results:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")

    summary_path = output_dir / "compare_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(comparison_summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(f"[hf_eval_compare_subset] Summary: {format_metrics(comparison_summary)}")
    print(f"[hf_eval_compare_subset] Saved sample results to: {sample_results_path}")
    print(f"[hf_eval_compare_subset] Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
