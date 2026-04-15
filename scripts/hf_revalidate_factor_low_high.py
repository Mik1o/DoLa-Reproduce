"""Run FACTOR paper-style low-vs-high DoLa revalidation with fixed Wiki/News folds."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.hf_eval_compare_subset import LOAD_CONFIG_KEYS
from src.dola_utils import (
    internal_layer_to_official_layer_id,
    normalize_layer_bucket,
    official_layer_id_to_internal,
)
from src.factor import (
    FactorSample,
    aggregate_factor_accuracy,
    build_factor_candidates,
    compute_factor_is_correct,
    load_factor_samples,
)
from src.generation import score_candidate_answers_multi_config_with_details
from src.modeling import load_model_and_tokenizer
from src.utils import ensure_output_dir, load_yaml_config


SUMMARY_FILE = "factor_low_high_revalidation_summary.json"
RUN_LOG_FILE = "run.log"
TIE_TOL = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run FACTOR paper-style low-vs-high DoLa bucket revalidation."
    )
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def _log(output_dir: Path, message: str) -> None:
    line = f"[hf_revalidate_factor_low_high] {message}"
    print(line, flush=True)
    log_path = output_dir / RUN_LOG_FILE
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
    tmp_path.replace(path)


def _serialize_layer_dist(layer_dist: dict[int, int] | None) -> dict[str, int] | None:
    if not layer_dist:
        return None
    return {str(layer): int(count) for layer, count in sorted(layer_dist.items())}


def _aggregate_layer_usage(items: list[object]) -> dict[int, int] | None:
    aggregate: dict[int, int] = {}
    for item in items:
        layer_dist = getattr(item, "premature_layer_dist", None)
        if not layer_dist:
            continue
        for layer, count in layer_dist.items():
            aggregate[int(layer)] = aggregate.get(int(layer), 0) + int(count)
    return aggregate or None


def _resolve_bucket(
    *,
    bucket_config: dict[str, Any],
    mature_layer: int,
    num_hidden_layers: int,
) -> dict[str, Any]:
    official_layers = [int(layer) for layer in bucket_config["official_candidate_premature_layers"]]
    internal_layers = normalize_layer_bucket(
        [official_layer_id_to_internal(layer, num_hidden_layers) for layer in official_layers],
        mature_layer,
        num_hidden_layers,
        field_name=str(bucket_config["name"]),
        allow_embedding_output=True,
    )
    return {
        "name": str(bucket_config["name"]),
        "official_candidate_premature_layers": official_layers,
        "official_mature_layer": int(bucket_config.get("official_mature_layer", mature_layer + 1)),
        "candidate_premature_layers_internal": internal_layers,
    }


def _static_result_key(layer: int, num_hidden_layers: int) -> str:
    return f"static_official_{internal_layer_to_official_layer_id(layer, num_hidden_layers)}"


def _select_dynamic_bucket_by_accuracy(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    if not candidates:
        raise ValueError("dynamic candidates must not be empty.")
    best_accuracy = max(float(item["summary"]["accuracy"]) for item in candidates)
    tied = [
        item
        for item in candidates
        if math.isclose(float(item["summary"]["accuracy"]), best_accuracy, rel_tol=0.0, abs_tol=TIE_TOL)
    ]
    tied_names = sorted(str(item["name"]) for item in tied)
    return {
        "best_accuracy": best_accuracy,
        "is_tie": len(tied) > 1,
        "selected_bucket": None if len(tied) > 1 else tied[0],
        "tied_bucket_names": tied_names,
    }


def _select_best_static_by_accuracy(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    if not candidates:
        raise ValueError("static candidates must not be empty.")
    return max(
        candidates,
        key=lambda item: (float(item["summary"]["accuracy"]), int(item["summary"]["correct_count"])),
    )


def _average_accuracy(rows: list[dict[str, float | int]]) -> dict[str, float | int]:
    return aggregate_factor_accuracy([bool(item["is_correct"]) for item in rows])


def _score_factor_split(
    *,
    output_dir: Path,
    stage_label: str,
    model: Any,
    tokenizer: Any,
    samples: list[FactorSample],
    score_mode: str,
    post_softmax: bool,
    relative_top: float,
    relative_top_value: float,
    mature_layer: int,
    candidate_batch_size: int,
    dynamic_buckets: list[dict[str, Any]],
    static_layers: list[int],
) -> dict[str, Any]:
    dynamic_bucket_map = {
        str(bucket["name"]): list(bucket["candidate_premature_layers_internal"])
        for bucket in dynamic_buckets
    }
    vanilla_rows: list[dict[str, float | bool]] = []
    static_rows = {layer: [] for layer in static_layers}
    dynamic_rows = {str(bucket["name"]): [] for bucket in dynamic_buckets}
    dynamic_layer_usage = {str(bucket["name"]): {} for bucket in dynamic_buckets}

    total_answers = 0
    total_batches = 0
    start_time = time.perf_counter()
    last_log_time = start_time

    for sample_index, sample in enumerate(samples):
        try:
            true_candidate, false_candidates = build_factor_candidates(sample)
            all_candidates = [true_candidate, *false_candidates]

            score_result = score_candidate_answers_multi_config_with_details(
                model=model,
                tokenizer=tokenizer,
                prompt=sample.prefix,
                candidate_answers=all_candidates,
                static_layers=static_layers,
                dynamic_buckets=dynamic_bucket_map,
                score_mode=score_mode,
                post_softmax=post_softmax,
                relative_top=relative_top,
                relative_top_value=relative_top_value,
                mature_layer=mature_layer,
                candidate_batch_size=candidate_batch_size,
            )
        except Exception as error:
            _log(
                output_dir,
                "FAILED "
                f"stage={stage_label} "
                f"sample_index={sample_index} "
                f"prefix_column={sample.prefix_column} "
                f"prefix_chars={len(sample.prefix)} "
                f"completion_chars={len(sample.completion)} "
                f"contradiction_chars={[len(sample.contradiction_0), len(sample.contradiction_1), len(sample.contradiction_2)]} "
                f"error={error}",
            )
            for line in traceback.format_exc().splitlines():
                _log(output_dir, line)
            raise

        vanilla_true = score_result.vanilla[0].score
        vanilla_false = [item.score for item in score_result.vanilla[1:]]
        vanilla_rows.append(
            {
                "is_correct": compute_factor_is_correct(vanilla_true, vanilla_false),
                "true_score": vanilla_true,
            }
        )

        for layer in static_layers:
            items = score_result.static[layer]
            static_rows[layer].append(
                {
                    "is_correct": compute_factor_is_correct(
                        items[0].score,
                        [item.score for item in items[1:]],
                    )
                }
            )

        for bucket in dynamic_buckets:
            bucket_name = str(bucket["name"])
            items = score_result.dynamic[bucket_name]
            dynamic_rows[bucket_name].append(
                {
                    "is_correct": compute_factor_is_correct(
                        items[0].score,
                        [item.score for item in items[1:]],
                    )
                }
            )
            sample_dynamic_usage = _aggregate_layer_usage(items)
            if sample_dynamic_usage:
                current_usage = dynamic_layer_usage[bucket_name]
                for layer, count in sample_dynamic_usage.items():
                    current_usage[int(layer)] = current_usage.get(int(layer), 0) + int(count)

        total_answers += len(all_candidates)
        total_batches += int(score_result.batch_count)
        now = time.perf_counter()
        completed = sample_index + 1
        if completed <= 5 or completed == len(samples) or completed % 10 == 0 or (now - last_log_time) >= 30.0:
            elapsed = now - start_time
            avg_sample = elapsed / completed
            eta = avg_sample * (len(samples) - completed)
            _log(
                output_dir,
                f"{stage_label} {completed}/{len(samples)} samples | elapsed={elapsed / 3600:.2f}h | "
                f"avg_sample_s={avg_sample:.2f} | eta={eta / 3600:.2f}h",
            )
            last_log_time = now

    duration = time.perf_counter() - start_time
    vanilla_summary = aggregate_factor_accuracy([bool(item["is_correct"]) for item in vanilla_rows])
    static_summaries = {
        layer: aggregate_factor_accuracy([bool(item["is_correct"]) for item in rows])
        for layer, rows in static_rows.items()
    }
    dynamic_summaries = {
        bucket_name: aggregate_factor_accuracy([bool(item["is_correct"]) for item in rows])
        for bucket_name, rows in dynamic_rows.items()
    }

    return {
        "vanilla": vanilla_summary,
        "static": {
            _static_result_key(layer, int(getattr(model.config, "num_hidden_layers", 0))): {
                "internal": int(layer),
                "official": internal_layer_to_official_layer_id(layer, int(getattr(model.config, "num_hidden_layers", 0))),
                **summary,
            }
            for layer, summary in static_summaries.items()
        },
        "dynamic": {
            bucket_name: {
                "name": bucket_name,
                **summary,
                "layer_usage": _serialize_layer_dist(dynamic_layer_usage[bucket_name] or None),
            }
            for bucket_name, summary in dynamic_summaries.items()
        },
        "profiling": {
            "num_samples": len(samples),
            "total_answers": total_answers,
            "total_forward_batches": total_batches,
            "total_wall_time_sec": duration,
            "avg_seconds_per_sample": (duration / len(samples)) if samples else 0.0,
            "avg_seconds_per_answer": (duration / total_answers) if total_answers else 0.0,
        },
    }


def _weighted_average_accuracy(items: list[tuple[float, int]]) -> float:
    total_weight = sum(weight for _, weight in items)
    if total_weight <= 0:
        raise ValueError("total_weight must be positive.")
    return sum(value * weight for value, weight in items) / total_weight


def _classify_revalidation(folds: list[dict[str, Any]]) -> str:
    selected = [fold["validation"]["selection"]["selected_bucket_name"] for fold in folds]
    if selected == ["paper_low_0_16", "paper_low_0_16"]:
        return "LIKELY_FACTOR_LOW_DIRECTION_HOLDS"
    if selected == ["paper_high_16_32", "paper_high_16_32"]:
        return "LIKELY_FACTOR_HIGH_DIRECTION_BETTER_FOR_LLAMA31"
    return "LIKELY_FACTOR_SPLIT_DEPENDENT_NO_SINGLE_DIRECTION"


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    output_dir = ensure_output_dir(
        config.get(
            "output_dir",
            PROJECT_ROOT / "outputs" / "factor_low_high_revalidation",
        )
    )

    model_name = str(config["model_name"])
    device = config.get("device", "cpu")
    score_mode = str(config.get("score_mode", "sum_logprob"))
    mature_layer = int(config["mature_layer"])
    post_softmax = bool(config.get("post_softmax", True))
    relative_top = float(config.get("relative_top", 0.1))
    relative_top_value = float(config.get("relative_top_value", -1000.0))
    candidate_batch_size = int(config.get("candidate_batch_size", 1))
    wiki_csv_path = Path(str(config["wiki_csv_path"]))
    news_csv_path = Path(str(config["news_csv_path"]))

    model_kwargs = {key: config[key] for key in LOAD_CONFIG_KEYS if key in config}
    model, tokenizer = load_model_and_tokenizer(
        model_name=model_name,
        device=device,
        **model_kwargs,
    )
    num_hidden_layers = int(getattr(model.config, "num_hidden_layers", 0))

    low_bucket = _resolve_bucket(
        bucket_config=dict(config["paper_low_bucket"]),
        mature_layer=mature_layer,
        num_hidden_layers=num_hidden_layers,
    )
    high_bucket = _resolve_bucket(
        bucket_config=dict(config["paper_high_bucket"]),
        mature_layer=mature_layer,
        num_hidden_layers=num_hidden_layers,
    )
    dynamic_buckets = [low_bucket, high_bucket]

    static_low_layers = normalize_layer_bucket(
        [official_layer_id_to_internal(layer, num_hidden_layers) for layer in config.get("static_low_candidates_official", [])],
        mature_layer,
        num_hidden_layers,
        field_name="static_low_candidates_official",
        allow_embedding_output=True,
    )
    static_high_layers = normalize_layer_bucket(
        [official_layer_id_to_internal(layer, num_hidden_layers) for layer in config.get("static_high_candidates_official", [])],
        mature_layer,
        num_hidden_layers,
        field_name="static_high_candidates_official",
        allow_embedding_output=True,
    )
    static_layers = list(dict.fromkeys([*static_low_layers, *static_high_layers]))

    wiki_samples = load_factor_samples(wiki_csv_path)
    news_samples = load_factor_samples(news_csv_path)

    _log(output_dir, f"Model: {model_name}")
    _log(output_dir, f"Wiki CSV: {wiki_csv_path}")
    _log(output_dir, f"News CSV: {news_csv_path}")
    _log(output_dir, f"paper_low official={low_bucket['official_candidate_premature_layers']} internal={low_bucket['candidate_premature_layers_internal']}")
    _log(output_dir, f"paper_high official={high_bucket['official_candidate_premature_layers']} internal={high_bucket['candidate_premature_layers_internal']}")

    fold_specs = [
        {
            "name": "fold1",
            "validation_name": "wiki",
            "validation_samples": wiki_samples,
            "held_out_name": "news",
            "held_out_samples": news_samples,
        },
        {
            "name": "fold2",
            "validation_name": "news",
            "validation_samples": news_samples,
            "held_out_name": "wiki",
            "held_out_samples": wiki_samples,
        },
    ]

    fold_reports: list[dict[str, Any]] = []

    for fold_spec in fold_specs:
        fold_name = str(fold_spec["name"])
        validation_result = _score_factor_split(
            output_dir=output_dir,
            stage_label=f"{fold_name} validation={fold_spec['validation_name']}",
            model=model,
            tokenizer=tokenizer,
            samples=list(fold_spec["validation_samples"]),
            score_mode=score_mode,
            post_softmax=post_softmax,
            relative_top=relative_top,
            relative_top_value=relative_top_value,
            mature_layer=mature_layer,
            candidate_batch_size=candidate_batch_size,
            dynamic_buckets=dynamic_buckets,
            static_layers=static_layers,
        )

        dynamic_candidates = [
            {
                "name": bucket["name"],
                "summary": validation_result["dynamic"][bucket["name"]],
            }
            for bucket in dynamic_buckets
        ]
        dynamic_choice = _select_dynamic_bucket_by_accuracy(dynamic_candidates)

        best_static_low = _select_best_static_by_accuracy(
            [
                {
                    "name": _static_result_key(layer, num_hidden_layers),
                    "internal": int(layer),
                    "official": internal_layer_to_official_layer_id(layer, num_hidden_layers),
                    "summary": validation_result["static"][_static_result_key(layer, num_hidden_layers)],
                }
                for layer in static_low_layers
            ]
        )
        best_static_high = _select_best_static_by_accuracy(
            [
                {
                    "name": _static_result_key(layer, num_hidden_layers),
                    "internal": int(layer),
                    "official": internal_layer_to_official_layer_id(layer, num_hidden_layers),
                    "summary": validation_result["static"][_static_result_key(layer, num_hidden_layers)],
                }
                for layer in static_high_layers
            ]
        )

        selected_bucket_name = (
            None if dynamic_choice["is_tie"] else str(dynamic_choice["selected_bucket"]["name"])
        )
        selected_bucket_internal = (
            None
            if dynamic_choice["is_tie"]
            else list(dynamic_choice["selected_bucket"]["summary"].get("layer_usage", {}).keys())
        )
        _log(
            output_dir,
            f"{fold_name} validation low_acc={float(validation_result['dynamic']['paper_low_0_16']['accuracy']):.4f} "
            f"high_acc={float(validation_result['dynamic']['paper_high_16_32']['accuracy']):.4f} "
            f"selected={'tie' if dynamic_choice['is_tie'] else selected_bucket_name}",
        )

        held_out_result = _score_factor_split(
            output_dir=output_dir,
            stage_label=f"{fold_name} held_out={fold_spec['held_out_name']}",
            model=model,
            tokenizer=tokenizer,
            samples=list(fold_spec["held_out_samples"]),
            score_mode=score_mode,
            post_softmax=post_softmax,
            relative_top=relative_top,
            relative_top_value=relative_top_value,
            mature_layer=mature_layer,
            candidate_batch_size=candidate_batch_size,
            dynamic_buckets=dynamic_buckets,
            static_layers=static_layers,
        )

        held_out_payload: dict[str, Any] = {
            "set_name": str(fold_spec["held_out_name"]),
            "num_samples": len(list(fold_spec["held_out_samples"])),
            "vanilla": held_out_result["vanilla"],
        }
        if dynamic_choice["is_tie"]:
            held_out_payload["selected_dynamic_bucket_name"] = None
            held_out_payload["selected_dynamic_bucket_tie"] = dynamic_choice["tied_bucket_names"]
            held_out_payload["tied_dynamic_candidates"] = {
                name: held_out_result["dynamic"][name]
                for name in dynamic_choice["tied_bucket_names"]
            }
        else:
            held_out_payload["selected_dynamic_bucket_name"] = selected_bucket_name
            held_out_payload["selected_dynamic"] = held_out_result["dynamic"][selected_bucket_name]

        fold_reports.append(
            {
                "fold_name": fold_name,
                "validation": {
                    "set_name": str(fold_spec["validation_name"]),
                    "num_samples": len(list(fold_spec["validation_samples"])),
                    "vanilla": validation_result["vanilla"],
                    "dynamic": {
                        bucket["name"]: validation_result["dynamic"][bucket["name"]]
                        for bucket in dynamic_buckets
                    },
                    "selection": {
                        "best_accuracy": float(dynamic_choice["best_accuracy"]),
                        "is_tie": bool(dynamic_choice["is_tie"]),
                        "selected_bucket_name": selected_bucket_name,
                        "tied_bucket_names": list(dynamic_choice["tied_bucket_names"]),
                    },
                    "best_static_low": best_static_low,
                    "best_static_high": best_static_high,
                },
                "held_out": held_out_payload,
            }
        )

    if any(fold["validation"]["selection"]["selected_bucket_name"] is None for fold in fold_reports):
        average_selected_dynamic = None
    else:
        average_selected_dynamic = {
            "accuracy": _weighted_average_accuracy(
                [
                    (
                        float(fold["held_out"]["selected_dynamic"]["accuracy"]),
                        int(fold["held_out"]["num_samples"]),
                    )
                    for fold in fold_reports
                ]
            )
        }

    average_vanilla = {
        "accuracy": _weighted_average_accuracy(
            [
                (float(fold["held_out"]["vanilla"]["accuracy"]), int(fold["held_out"]["num_samples"]))
                for fold in fold_reports
            ]
        )
    }

    summary = {
        "task_name": str(config.get("task_name", "llama31_8b_factor_low_high_revalidation")),
        "model_name": model_name,
        "score_mode": score_mode,
        "mature_layer": mature_layer,
        "post_softmax": post_softmax,
        "relative_top": relative_top,
        "relative_top_value": relative_top_value,
        "candidate_batch_size": candidate_batch_size,
        "factor_subsets": {
            "wiki_csv_path": str(wiki_csv_path),
            "news_csv_path": str(news_csv_path),
        },
        "dynamic_bucket_candidates": {
            "paper_low_0_16": low_bucket,
            "paper_high_16_32": high_bucket,
        },
        "static_sanity_candidates": {
            "low": [
                {
                    "internal": int(layer),
                    "official": internal_layer_to_official_layer_id(layer, num_hidden_layers),
                }
                for layer in static_low_layers
            ],
            "high": [
                {
                    "internal": int(layer),
                    "official": internal_layer_to_official_layer_id(layer, num_hidden_layers),
                }
                for layer in static_high_layers
            ],
        },
        "folds": fold_reports,
        "held_out_average": {
            "vanilla": average_vanilla,
            "selected_dynamic": average_selected_dynamic,
        },
        "final_label": _classify_revalidation(fold_reports),
    }

    summary_path = output_dir / SUMMARY_FILE
    _atomic_write_json(summary_path, summary)
    _log(output_dir, f"held_out_average vanilla={average_vanilla['accuracy']:.4f}")
    if average_selected_dynamic is None:
        _log(output_dir, "held_out_average selected_dynamic=tie_unresolved")
    else:
        _log(output_dir, f"held_out_average selected_dynamic={average_selected_dynamic['accuracy']:.4f}")
    _log(output_dir, f"final_label={summary['final_label']}")
    _log(output_dir, f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
