"""Run a small official-parity diagnosis over a fixed TruthfulQA subset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.hf_eval_compare_subset import LOAD_CONFIG_KEYS, evaluate_compare_subset
from src.dola_utils import normalize_layer_bucket
from src.modeling import load_model_and_tokenizer
from src.truthfulqa_mc import load_truthfulqa_samples
from src.utils import ensure_output_dir, load_yaml_config, select_fixed_subset



def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the official parity diagnosis script."""
    parser = argparse.ArgumentParser(
        description="Diagnose official DoLa parity on a small TruthfulQA subset."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "mistral7b_truthfulqa_real_official_diagnose.yaml",
        help="Path to a YAML config file.",
    )
    return parser.parse_args()



def _extract_vanilla_summary(summary: dict[str, object]) -> dict[str, object]:
    """Extract only the vanilla average metrics from a compare summary."""
    return {
        "avg_mc1": float(summary["vanilla_avg_mc1"]),
        "avg_mc2": float(summary["vanilla_avg_mc2"]),
        "avg_mc3": float(summary["vanilla_avg_mc3"]),
        "num_samples": int(summary["num_samples"]),
    }



def _extract_dola_summary(summary: dict[str, object]) -> dict[str, object]:
    """Extract only the DoLa-side metrics and layer usage from a compare summary."""
    result = {
        "avg_mc1": float(summary["dola_avg_mc1"]),
        "avg_mc2": float(summary["dola_avg_mc2"]),
        "avg_mc3": float(summary["dola_avg_mc3"]),
        "num_samples": int(summary["num_samples"]),
    }
    if summary.get("premature_layer_dist"):
        result["premature_layer_dist"] = summary["premature_layer_dist"]
    if summary.get("candidate_premature_layers") is not None:
        result["candidate_premature_layers"] = summary["candidate_premature_layers"]
    if summary.get("premature_layer") is not None:
        result["premature_layer"] = summary["premature_layer"]
    if summary.get("mature_layer") is not None:
        result["mature_layer"] = summary["mature_layer"]
    return result



def _build_answer_format_audit(samples: list[object], limit: int = 3) -> list[dict[str, object]]:
    """Build a small preview of normalized answers after official-style formatting."""
    audit_rows: list[dict[str, object]] = []
    for sample in samples[:limit]:
        audit_rows.append(
            {
                "question": sample.question,
                "best_answer": sample.best_answer,
                "correct_answers": sample.correct_answers,
                "incorrect_answers_preview": sample.incorrect_answers[:3],
            }
        )
    return audit_rows



def _print_summary_row(name: str, summary: dict[str, object]) -> None:
    """Print a compact one-line metric summary for one diagnosis setting."""
    print(
        f"[hf_diagnose_official_parity] {name}: "
        f"MC1={float(summary['avg_mc1']):.4f}, "
        f"MC2={float(summary['avg_mc2']):.4f}, "
        f"MC3={float(summary['avg_mc3']):.4f}"
    )
    if summary.get("premature_layer_dist"):
        print(
            f"[hf_diagnose_official_parity] {name} premature_layer_dist="
            f"{summary['premature_layer_dist']}"
        )



def main() -> None:
    """Run the official parity diagnosis over a fixed small subset."""
    args = parse_args()
    config = load_yaml_config(args.config)
    output_dir = ensure_output_dir(config.get("output_dir", PROJECT_ROOT / "outputs" / "official_parity_diagnosis"))

    model_name = str(config["model_name"])
    device = config.get("device", "cpu")
    csv_path = Path(str(config["csv_path"]))
    max_samples = int(config.get("max_samples", 5))
    subset_seed = config.get("subset_seed")
    subset_seed = None if subset_seed is None else int(subset_seed)
    prompt_style = str(config.get("prompt_style", "official_tfqa_mc"))
    score_mode = str(config.get("score_mode", "sum_logprob"))
    post_softmax = bool(config.get("post_softmax", False))
    relative_top = float(config.get("relative_top", 0.0))
    relative_top_value = float(config.get("relative_top_value", -1000.0))
    mature_layer = int(config["mature_layer"])
    static_premature_layers = [int(layer) for layer in config.get("static_premature_layers", [])]
    current_bucket = [int(layer) for layer in config.get("candidate_premature_layers", [])]
    shifted_bucket = [int(layer) for layer in config.get("candidate_premature_layers_shifted", [])]

    model_kwargs = {key: config[key] for key in LOAD_CONFIG_KEYS if key in config}

    print(f"[hf_diagnose_official_parity] Model: {model_name}")
    print(f"[hf_diagnose_official_parity] CSV: {csv_path}")
    print(f"[hf_diagnose_official_parity] Output directory: {output_dir}")
    print(f"[hf_diagnose_official_parity] Subset size: {max_samples}, subset_seed={subset_seed}")

    samples = load_truthfulqa_samples(csv_path)
    if not samples:
        raise ValueError("No TruthfulQA samples were loaded for diagnosis.")
    subset_samples, subset_indices = select_fixed_subset(samples, max_samples, subset_seed)

    model, tokenizer = load_model_and_tokenizer(
        model_name=model_name,
        device=device,
        **model_kwargs,
    )

    num_hidden_layers = int(getattr(model.config, "num_hidden_layers", 0))
    normalized_static_layers = normalize_layer_bucket(
        static_premature_layers,
        mature_layer,
        num_hidden_layers,
        field_name="static_premature_layers",
    )
    normalized_current_bucket = normalize_layer_bucket(
        current_bucket,
        mature_layer,
        num_hidden_layers,
        field_name="candidate_premature_layers",
    )
    normalized_shifted_bucket = (
        normalize_layer_bucket(
            shifted_bucket,
            mature_layer,
            num_hidden_layers,
            field_name="candidate_premature_layers_shifted",
        )
        if shifted_bucket
        else []
    )

    run_summaries: dict[str, dict[str, object]] = {}
    run_sample_results: dict[str, list[dict[str, object]]] = {}
    vanilla_summary: dict[str, object] | None = None

    for static_layer in normalized_static_layers:
        compare_sample_results, compare_summary = evaluate_compare_subset(
            model,
            tokenizer,
            subset_samples,
            max_samples=len(subset_samples),
            premature_layer=static_layer,
            prompt_style=prompt_style,
            score_mode=score_mode,
            dola_score_mode="official_static_dola",
            post_softmax=post_softmax,
            relative_top=relative_top,
            relative_top_value=relative_top_value,
            mature_layer=mature_layer,
        )
        run_name = f"static_{static_layer}"
        run_summaries[run_name] = _extract_dola_summary(compare_summary)
        run_sample_results[run_name] = compare_sample_results
        if vanilla_summary is None:
            vanilla_summary = _extract_vanilla_summary(compare_summary)

    if vanilla_summary is None:
        raise ValueError("At least one static_premature_layer is required for official diagnosis.")

    dynamic_current_sample_results, dynamic_current_summary = evaluate_compare_subset(
        model,
        tokenizer,
        subset_samples,
        max_samples=len(subset_samples),
        premature_layer=normalized_current_bucket[0],
        prompt_style=prompt_style,
        score_mode=score_mode,
        dola_score_mode="official_dynamic_dola",
        post_softmax=post_softmax,
        relative_top=relative_top,
        relative_top_value=relative_top_value,
        candidate_premature_layers=normalized_current_bucket,
        mature_layer=mature_layer,
    )
    run_summaries["dynamic_current_bucket"] = _extract_dola_summary(dynamic_current_summary)
    run_sample_results["dynamic_current_bucket"] = dynamic_current_sample_results

    if normalized_shifted_bucket:
        dynamic_shifted_sample_results, dynamic_shifted_summary = evaluate_compare_subset(
            model,
            tokenizer,
            subset_samples,
            max_samples=len(subset_samples),
            premature_layer=normalized_shifted_bucket[0],
            prompt_style=prompt_style,
            score_mode=score_mode,
            dola_score_mode="official_dynamic_dola",
            post_softmax=post_softmax,
            relative_top=relative_top,
            relative_top_value=relative_top_value,
            candidate_premature_layers=normalized_shifted_bucket,
            mature_layer=mature_layer,
        )
        run_summaries["dynamic_shifted_bucket"] = _extract_dola_summary(dynamic_shifted_summary)
        run_sample_results["dynamic_shifted_bucket"] = dynamic_shifted_sample_results

    summary = {
        "task_name": str(config.get("task_name", "official_parity_diagnosis")),
        "model_name": model_name,
        "csv_path": str(csv_path),
        "max_samples": max_samples,
        "subset_seed": subset_seed,
        "subset_indices": subset_indices,
        "prompt_style": prompt_style,
        "score_mode": score_mode,
        "mature_layer": mature_layer,
        "post_softmax": post_softmax,
        "relative_top": relative_top,
        "relative_top_value": relative_top_value,
        "answer_format_audit": {
            "aligned_to": "tfqa_mc_eval.py split_multi_answer / format_best",
            "preview": _build_answer_format_audit(subset_samples),
        },
        "runs": {
            "vanilla_official": vanilla_summary,
            **run_summaries,
        },
        "diagnostics": {
            "dynamic_current_minus_static_first_mc2": (
                float(run_summaries["dynamic_current_bucket"]["avg_mc2"])
                - float(run_summaries[f"static_{normalized_static_layers[0]}"]["avg_mc2"])
            ),
            "dynamic_current_minus_static_first_mc1": (
                float(run_summaries["dynamic_current_bucket"]["avg_mc1"])
                - float(run_summaries[f"static_{normalized_static_layers[0]}"]["avg_mc1"])
            ),
        },
        "output_dir": str(output_dir),
    }

    if len(normalized_static_layers) > 1:
        second_static_name = f"static_{normalized_static_layers[1]}"
        summary["diagnostics"]["dynamic_current_minus_static_second_mc2"] = (
            float(run_summaries["dynamic_current_bucket"]["avg_mc2"])
            - float(run_summaries[second_static_name]["avg_mc2"])
        )
        summary["diagnostics"]["dynamic_current_minus_static_second_mc1"] = (
            float(run_summaries["dynamic_current_bucket"]["avg_mc1"])
            - float(run_summaries[second_static_name]["avg_mc1"])
        )

    if "dynamic_shifted_bucket" in run_summaries:
        summary["diagnostics"]["dynamic_shifted_minus_dynamic_current_mc2"] = (
            float(run_summaries["dynamic_shifted_bucket"]["avg_mc2"])
            - float(run_summaries["dynamic_current_bucket"]["avg_mc2"])
        )
        summary["diagnostics"]["dynamic_shifted_minus_dynamic_current_mc1"] = (
            float(run_summaries["dynamic_shifted_bucket"]["avg_mc1"])
            - float(run_summaries["dynamic_current_bucket"]["avg_mc1"])
        )

    print("[hf_diagnose_official_parity] Diagnosis summary:")
    _print_summary_row("vanilla_official", vanilla_summary)
    for run_name in [*run_summaries.keys()]:
        _print_summary_row(run_name, run_summaries[run_name])

    output_path = output_dir / "official_parity_summary.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    runs_path = output_dir / "official_parity_runs.json"
    with runs_path.open("w", encoding="utf-8") as handle:
        json.dump(run_sample_results, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(f"[hf_diagnose_official_parity] Saved summary to: {output_path}")
    print(f"[hf_diagnose_official_parity] Saved run details to: {runs_path}")


if __name__ == "__main__":
    main()
