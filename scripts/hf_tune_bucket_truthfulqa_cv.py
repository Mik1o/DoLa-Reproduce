"""Run a paper-faithful two-fold tuned-bucket TruthfulQA-MC baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from random import Random
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.hf_eval_compare_subset import LOAD_CONFIG_KEYS, evaluate_compare_subset
from src.dola_utils import normalize_layer_bucket
from src.metrics import aggregate_mc_metrics, compare_aggregate_metrics
from src.modeling import load_model_and_tokenizer
from src.truthfulqa_mc import TruthfulQASample, load_truthfulqa_samples
from src.utils import ensure_output_dir, load_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run two-fold TruthfulQA-MC bucket tuning using MC3 on validation folds."
    )
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def _two_fold_split(samples: list[TruthfulQASample], seed: int) -> tuple[list[int], list[int], dict[str, object]]:
    grouped: dict[str, list[int]] = {}
    for index, sample in enumerate(samples):
        key = sample.category or "__missing__"
        grouped.setdefault(key, []).append(index)

    rng = Random(seed)
    fold_a: list[int] = []
    fold_b: list[int] = []
    for indices in grouped.values():
        shuffled = indices[:]
        rng.shuffle(shuffled)
        for offset, item in enumerate(shuffled):
            if offset % 2 == 0:
                fold_a.append(item)
            else:
                fold_b.append(item)
    fold_a.sort()
    fold_b.sort()
    meta = {
        "split_strategy": "approximate_stratified_by_category",
        "is_stratified_exact": False,
        "fold_a_size": len(fold_a),
        "fold_b_size": len(fold_b),
        "seed": seed,
    }
    return fold_a, fold_b, meta


def _subset_from_indices(samples: list[TruthfulQASample], indices: list[int]) -> list[TruthfulQASample]:
    return [samples[index] for index in indices]


def _evaluate_dynamic_candidate(
    model: Any,
    tokenizer: Any,
    samples: list[TruthfulQASample],
    bucket_name: str,
    candidate_layers: list[int],
    mature_layer: int,
    prompt_style: str,
    score_mode: str,
    post_softmax: bool,
    relative_top: float,
    relative_top_value: float,
) -> dict[str, object]:
    _, summary = evaluate_compare_subset(
        model,
        tokenizer,
        samples,
        max_samples=len(samples),
        premature_layer=candidate_layers[0],
        prompt_style=prompt_style,
        score_mode=score_mode,
        dola_score_mode="official_dynamic_dola",
        post_softmax=post_softmax,
        relative_top=relative_top,
        relative_top_value=relative_top_value,
        candidate_premature_layers=candidate_layers,
        mature_layer=mature_layer,
    )
    return {"name": bucket_name, "candidate_premature_layers": candidate_layers, "summary": summary}


def _evaluate_static_candidate(
    model: Any,
    tokenizer: Any,
    samples: list[TruthfulQASample],
    premature_layer: int,
    mature_layer: int,
    prompt_style: str,
    score_mode: str,
    post_softmax: bool,
    relative_top: float,
    relative_top_value: float,
) -> dict[str, object]:
    _, summary = evaluate_compare_subset(
        model,
        tokenizer,
        samples,
        max_samples=len(samples),
        premature_layer=premature_layer,
        prompt_style=prompt_style,
        score_mode=score_mode,
        dola_score_mode="official_static_dola",
        post_softmax=post_softmax,
        relative_top=relative_top,
        relative_top_value=relative_top_value,
        mature_layer=mature_layer,
    )
    return {"premature_layer": premature_layer, "summary": summary}


def _select_best_dynamic(candidates: list[dict[str, object]]) -> dict[str, object]:
    return max(
        candidates,
        key=lambda item: (
            float(item["summary"]["dola_avg_mc3"]),
            float(item["summary"]["dola_avg_mc2"]),
            float(item["summary"]["dola_avg_mc1"]),
        ),
    )


def _select_best_static(candidates: list[dict[str, object]]) -> dict[str, object]:
    return max(
        candidates,
        key=lambda item: (
            float(item["summary"]["dola_avg_mc3"]),
            float(item["summary"]["dola_avg_mc2"]),
            float(item["summary"]["dola_avg_mc1"]),
        ),
    )


def _evaluate_fold_test(
    model: Any,
    tokenizer: Any,
    test_samples: list[TruthfulQASample],
    selected_dynamic: dict[str, object],
    selected_static: dict[str, object],
    mature_layer: int,
    prompt_style: str,
    score_mode: str,
    post_softmax: bool,
    relative_top: float,
    relative_top_value: float,
) -> dict[str, object]:
    _, dynamic_summary = evaluate_compare_subset(
        model,
        tokenizer,
        test_samples,
        max_samples=len(test_samples),
        premature_layer=selected_dynamic["candidate_premature_layers"][0],
        prompt_style=prompt_style,
        score_mode=score_mode,
        dola_score_mode="official_dynamic_dola",
        post_softmax=post_softmax,
        relative_top=relative_top,
        relative_top_value=relative_top_value,
        candidate_premature_layers=selected_dynamic["candidate_premature_layers"],
        mature_layer=mature_layer,
    )
    _, static_summary = evaluate_compare_subset(
        model,
        tokenizer,
        test_samples,
        max_samples=len(test_samples),
        premature_layer=int(selected_static["premature_layer"]),
        prompt_style=prompt_style,
        score_mode=score_mode,
        dola_score_mode="official_static_dola",
        post_softmax=post_softmax,
        relative_top=relative_top,
        relative_top_value=relative_top_value,
        mature_layer=mature_layer,
    )
    return {
        "vanilla": {
            "MC1": float(dynamic_summary["vanilla_avg_mc1"]),
            "MC2": float(dynamic_summary["vanilla_avg_mc2"]),
            "MC3": float(dynamic_summary["vanilla_avg_mc3"]),
        },
        "tuned_static": {
            "MC1": float(static_summary["dola_avg_mc1"]),
            "MC2": float(static_summary["dola_avg_mc2"]),
            "MC3": float(static_summary["dola_avg_mc3"]),
        },
        "tuned_dynamic": {
            "MC1": float(dynamic_summary["dola_avg_mc1"]),
            "MC2": float(dynamic_summary["dola_avg_mc2"]),
            "MC3": float(dynamic_summary["dola_avg_mc3"]),
        },
        "num_samples": int(dynamic_summary["num_samples"]),
    }


def _average_fold_results(folds: list[dict[str, object]]) -> dict[str, object]:
    total = sum(int(fold["num_samples"]) for fold in folds)
    if total <= 0:
        raise ValueError("No fold samples were evaluated.")

    def weighted(branch: str, metric: str) -> float:
        return sum(float(fold[branch][metric]) * int(fold["num_samples"]) for fold in folds) / total

    result = {
        "vanilla": {metric: weighted("vanilla", metric) for metric in ("MC1", "MC2", "MC3")},
        "tuned_static": {metric: weighted("tuned_static", metric) for metric in ("MC1", "MC2", "MC3")},
        "tuned_dynamic": {metric: weighted("tuned_dynamic", metric) for metric in ("MC1", "MC2", "MC3")},
        "num_samples": total,
    }
    result["comparisons"] = {
        "tuned_dynamic_minus_vanilla": {
            metric: result["tuned_dynamic"][metric] - result["vanilla"][metric]
            for metric in ("MC1", "MC2", "MC3")
        },
        "tuned_dynamic_minus_tuned_static": {
            metric: result["tuned_dynamic"][metric] - result["tuned_static"][metric]
            for metric in ("MC1", "MC2", "MC3")
        },
        "dynamic_still_matches_static": (
            abs(result["tuned_dynamic"]["MC2"] - result["tuned_static"]["MC2"]) <= 0.01
            and abs(result["tuned_dynamic"]["MC3"] - result["tuned_static"]["MC3"]) <= 0.01
        ),
    }
    return result


def _classify(cv_average: dict[str, object]) -> str:
    dynamic_minus_vanilla_mc3 = float(cv_average["comparisons"]["tuned_dynamic_minus_vanilla"]["MC3"])
    dynamic_minus_static_mc3 = float(cv_average["comparisons"]["tuned_dynamic_minus_tuned_static"]["MC3"])
    if dynamic_minus_vanilla_mc3 > 0.0 and cv_average["comparisons"]["dynamic_still_matches_static"]:
        return "LIKELY_BUCKET_TUNING_HELPS_BUT_DYNAMIC_STILL_COLLAPSES"
    if dynamic_minus_vanilla_mc3 > 0.0 and dynamic_minus_static_mc3 > 0.0:
        return "LIKELY_TUNED_DOLA_BEATS_VANILLA"
    if cv_average["comparisons"]["dynamic_still_matches_static"]:
        return "LIKELY_TUNED_DOLA_ONLY_MATCHES_STATIC"
    return "LIKELY_NEED_TO_REVISE_SEARCH_SPACE"


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    output_dir = ensure_output_dir(config.get("output_dir", PROJECT_ROOT / "outputs" / "truthfulqa_tuned_bucket_cv"))

    model_name = str(config["model_name"])
    device = config.get("device", "cpu")
    csv_path = Path(str(config["csv_path"]))
    split_seed = int(config.get("split_seed", 20260407))
    prompt_style = str(config.get("prompt_style", "official_tfqa_mc"))
    score_mode = str(config.get("score_mode", "sum_logprob"))
    post_softmax = bool(config.get("post_softmax", False))
    relative_top = float(config.get("relative_top", 0.0))
    relative_top_value = float(config.get("relative_top_value", -1000.0))
    mature_layer = int(config["mature_layer"])

    raw_dynamic_buckets = config["dynamic_bucket_candidates"]
    raw_static_layers = [int(layer) for layer in config["static_layer_candidates"]]

    model_kwargs = {key: config[key] for key in LOAD_CONFIG_KEYS if key in config}

    samples = load_truthfulqa_samples(csv_path)
    model, tokenizer = load_model_and_tokenizer(model_name=model_name, device=device, **model_kwargs)
    num_hidden_layers = int(getattr(model.config, "num_hidden_layers", 0))

    dynamic_buckets: list[dict[str, object]] = []
    for bucket in raw_dynamic_buckets:
        name = str(bucket["name"])
        internal_layers = normalize_layer_bucket(
            [int(layer) for layer in bucket["candidate_premature_layers"]],
            mature_layer,
            num_hidden_layers,
            field_name=name,
        )
        official_layers = [layer + 1 for layer in internal_layers]
        dynamic_buckets.append(
            {
                "name": name,
                "candidate_premature_layers": internal_layers,
                "official_layer_ids": official_layers,
            }
        )
    static_layers = normalize_layer_bucket(
        raw_static_layers,
        mature_layer,
        num_hidden_layers,
        field_name="static_layer_candidates",
    )

    fold_a, fold_b, split_meta = _two_fold_split(samples, split_seed)
    fold_specs = [
        {"validation_indices": fold_a, "test_indices": fold_b, "name": "fold1"},
        {"validation_indices": fold_b, "test_indices": fold_a, "name": "fold2"},
    ]

    fold_reports: list[dict[str, object]] = []
    for fold_spec in fold_specs:
        validation_samples = _subset_from_indices(samples, fold_spec["validation_indices"])
        test_samples = _subset_from_indices(samples, fold_spec["test_indices"])

        dynamic_candidates = [
            _evaluate_dynamic_candidate(
                model=model,
                tokenizer=tokenizer,
                samples=validation_samples,
                bucket_name=str(bucket["name"]),
                candidate_layers=list(bucket["candidate_premature_layers"]),
                mature_layer=mature_layer,
                prompt_style=prompt_style,
                score_mode=score_mode,
                post_softmax=post_softmax,
                relative_top=relative_top,
                relative_top_value=relative_top_value,
            )
            for bucket in dynamic_buckets
        ]
        static_candidates = [
            _evaluate_static_candidate(
                model=model,
                tokenizer=tokenizer,
                samples=validation_samples,
                premature_layer=layer,
                mature_layer=mature_layer,
                prompt_style=prompt_style,
                score_mode=score_mode,
                post_softmax=post_softmax,
                relative_top=relative_top,
                relative_top_value=relative_top_value,
            )
            for layer in static_layers
        ]

        selected_dynamic = _select_best_dynamic(dynamic_candidates)
        selected_static = _select_best_static(static_candidates)
        held_out_test = _evaluate_fold_test(
            model=model,
            tokenizer=tokenizer,
            test_samples=test_samples,
            selected_dynamic=selected_dynamic,
            selected_static=selected_static,
            mature_layer=mature_layer,
            prompt_style=prompt_style,
            score_mode=score_mode,
            post_softmax=post_softmax,
            relative_top=relative_top,
            relative_top_value=relative_top_value,
        )
        fold_reports.append(
            {
                "fold_name": fold_spec["name"],
                "validation_size": len(validation_samples),
                "test_size": len(test_samples),
                "selected_dynamic_bucket": selected_dynamic,
                "selected_static_layer": selected_static,
                "held_out_test": held_out_test,
            }
        )

    cv_average = _average_fold_results([fold["held_out_test"] for fold in fold_reports])
    final_label = _classify(cv_average)

    report = {
        "task_name": str(config.get("task_name", "truthfulqa_tuned_bucket_cv")),
        "model_name": model_name,
        "csv_path": str(csv_path),
        "split": split_meta,
        "prompt_style": prompt_style,
        "score_mode": score_mode,
        "mature_layer": mature_layer,
        "dynamic_bucket_candidates": dynamic_buckets,
        "static_layer_candidates": [{"internal": layer, "official": layer + 1} for layer in static_layers],
        "transfer_reference": config.get("transfer_reference_summary"),
        "folds": fold_reports,
        "cv_average": cv_average,
        "final_label": final_label,
    }

    output_path = output_dir / "truthfulqa_tuned_bucket_cv_summary.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(f"[hf_tune_bucket_truthfulqa_cv] final_label={final_label}")
    print(f"[hf_tune_bucket_truthfulqa_cv] Saved summary to: {output_path}")


if __name__ == "__main__":
    main()
