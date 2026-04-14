"""Run the FACTOR paper-transferred low-bucket baseline for a local causal LM."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.hf_eval_compare_subset import LOAD_CONFIG_KEYS
from src.dola_utils import normalize_layer_bucket
from src.factor import (
    aggregate_factor_accuracy,
    build_factor_candidates,
    compute_factor_is_correct,
    load_factor_samples,
)
from src.generation import score_candidate_answers_multi_config_with_details
from src.modeling import load_model_and_tokenizer
from src.utils import ensure_output_dir, load_yaml_config


SUMMARY_FILE = "factor_paper_low_baseline_summary.json"
RUN_LOG_FILE = "run.log"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the FACTOR paper-transferred low-bucket baseline on a local model."
    )
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def _log(output_dir: Path, message: str) -> None:
    line = f"[hf_run_factor_paper_low_baseline] {message}"
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


def _embedding_output_candidate_supported() -> bool:
    """Current local scorer only addresses transformer block outputs."""
    return False


def _classify_factor_baseline(
    *,
    embedding_inclusive: bool,
    vanilla_accuracy: float,
    dynamic_accuracy: float,
) -> str:
    if embedding_inclusive:
        if dynamic_accuracy > vanilla_accuracy:
            return "LIKELY_FACTOR_PAPER_LOW_BUCKET_HOLDS"
        return "LIKELY_FACTOR_LOW_BUCKET_NOT_EFFECTIVE"

    if dynamic_accuracy > vanilla_accuracy:
        return "LIKELY_FACTOR_LOW_BUCKET_ONLY_APPROXIMATE"
    return "LIKELY_NEED_EMBEDDING_SUPPORT_FIRST"


def _select_best_static_observed(static_results: dict[int, dict[str, float | int]]) -> tuple[int, dict[str, float | int]]:
    return max(
        static_results.items(),
        key=lambda item: (float(item[1]["accuracy"]), int(item[1]["correct_count"])),
    )


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    output_dir = ensure_output_dir(
        config.get(
            "output_dir",
            PROJECT_ROOT / "outputs" / "factor_paper_low_baseline",
        )
    )

    model_name = str(config["model_name"])
    device = config.get("device", "cpu")
    csv_path = Path(str(config["csv_path"]))
    score_mode = str(config.get("score_mode", "sum_logprob"))
    mature_layer = int(config["mature_layer"])
    post_softmax = bool(config.get("post_softmax", True))
    relative_top = float(config.get("relative_top", 0.1))
    relative_top_value = float(config.get("relative_top_value", -1000.0))
    candidate_batch_size = int(config.get("candidate_batch_size", 1))
    max_samples = config.get("max_samples")
    max_samples = None if max_samples is None else int(max_samples)

    dynamic_bucket_config = config.get("dynamic_bucket")
    if not isinstance(dynamic_bucket_config, dict):
        raise ValueError("dynamic_bucket must be a mapping with name and candidate_premature_layers.")
    dynamic_bucket_name = str(dynamic_bucket_config["name"])
    raw_dynamic_layers = [int(layer) for layer in dynamic_bucket_config["candidate_premature_layers"]]
    raw_static_layers = [int(layer) for layer in config.get("static_layer_candidates", [])]
    if not raw_static_layers:
        raise ValueError("static_layer_candidates must contain at least one layer.")

    model_kwargs = {key: config[key] for key in LOAD_CONFIG_KEYS if key in config}
    samples = load_factor_samples(csv_path)
    if max_samples is not None:
        samples = samples[:max_samples]
        if not samples:
            raise ValueError("max_samples truncated the FACTOR run to zero samples.")

    model, tokenizer = load_model_and_tokenizer(
        model_name=model_name,
        device=device,
        **model_kwargs,
    )
    num_hidden_layers = int(getattr(model.config, "num_hidden_layers", 0))

    dynamic_layers = normalize_layer_bucket(
        raw_dynamic_layers,
        mature_layer,
        num_hidden_layers,
        field_name=f"{dynamic_bucket_name}.candidate_premature_layers",
    )
    static_layers = normalize_layer_bucket(
        raw_static_layers,
        mature_layer,
        num_hidden_layers,
        field_name="static_layer_candidates",
    )
    dynamic_bucket_map = {dynamic_bucket_name: dynamic_layers}

    embedding_inclusive = _embedding_output_candidate_supported()
    official_dynamic_layers = [int(layer) for layer in dynamic_bucket_config.get("official_candidate_premature_layers", [])]
    official_mature_layer = int(dynamic_bucket_config.get("official_mature_layer", mature_layer + 1))
    omitted_official_layers = [layer for layer in official_dynamic_layers if layer == 0 and not embedding_inclusive]

    _log(output_dir, f"Baseline label: {config.get('baseline_name', 'approx_paper_low_bucket_factor')}")
    _log(output_dir, f"Model: {model_name}")
    _log(output_dir, f"CSV: {csv_path}")
    _log(output_dir, f"Samples: {len(samples)}")
    _log(output_dir, f"Dynamic bucket: {dynamic_bucket_name} -> {dynamic_layers}")
    _log(output_dir, f"Static layers: {static_layers}")
    _log(output_dir, f"Candidate batch size: {candidate_batch_size}")
    _log(output_dir, f"post_softmax={post_softmax} relative_top={relative_top}")

    vanilla_rows: list[bool] = []
    static_rows = {layer: [] for layer in static_layers}
    dynamic_rows: list[bool] = []
    dynamic_layer_usage: dict[int, int] = {}
    total_answers = 0
    total_batches = 0
    start_time = time.perf_counter()
    last_log_time = start_time

    for sample_index, sample in enumerate(samples):
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

        vanilla_true = score_result.vanilla[0].score
        vanilla_false = [item.score for item in score_result.vanilla[1:]]
        vanilla_rows.append(compute_factor_is_correct(vanilla_true, vanilla_false))

        for layer in static_layers:
            items = score_result.static[layer]
            static_rows[layer].append(
                compute_factor_is_correct(
                    items[0].score,
                    [item.score for item in items[1:]],
                )
            )

        dynamic_items = score_result.dynamic[dynamic_bucket_name]
        dynamic_rows.append(
            compute_factor_is_correct(
                dynamic_items[0].score,
                [item.score for item in dynamic_items[1:]],
            )
        )
        sample_dynamic_usage = _aggregate_layer_usage(dynamic_items)
        if sample_dynamic_usage:
            for layer, count in sample_dynamic_usage.items():
                dynamic_layer_usage[layer] = dynamic_layer_usage.get(layer, 0) + count

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
                f"{completed}/{len(samples)} samples | elapsed={elapsed / 3600:.2f}h | "
                f"avg_sample_s={avg_sample:.2f} | eta={eta / 3600:.2f}h",
            )
            last_log_time = now

    duration = time.perf_counter() - start_time
    vanilla_summary = aggregate_factor_accuracy(vanilla_rows)
    static_summaries = {
        layer: aggregate_factor_accuracy(static_rows[layer])
        for layer in static_layers
    }
    dynamic_summary = aggregate_factor_accuracy(dynamic_rows)
    best_static_layer, best_static_summary = _select_best_static_observed(static_summaries)

    summary = {
        "task_name": str(config.get("task_name", "llama31_8b_factor_paper_low_baseline")),
        "baseline_name": str(config.get("baseline_name", "approx_paper_low_bucket_factor")),
        "model_name": model_name,
        "csv_path": str(csv_path),
        "evaluator_alignment": (
            "Matches official factor_eval.py semantics: one true completion versus three contradictions, "
            "and the sample counts as correct when the true completion score is not smaller than any contradiction."
        ),
        "score_mode": score_mode,
        "mature_layer": mature_layer,
        "post_softmax": post_softmax,
        "relative_top": relative_top,
        "relative_top_value": relative_top_value,
        "candidate_batch_size": candidate_batch_size,
        "embedding_output_candidate_supported": embedding_inclusive,
        "dynamic_bucket": {
            "name": dynamic_bucket_name,
            "bucket_kind": str(config.get("baseline_name", "approx_paper_low_bucket_factor")),
            "embedding_inclusive": embedding_inclusive,
            "note": str(
                config.get(
                    "low_bucket_note",
                    "Current implementation selects transformer block outputs only; official layer 0 "
                    "(embedding output) is not selectable, so this FACTOR low bucket is approximate.",
                )
            ),
            "official_candidate_premature_layers": official_dynamic_layers,
            "official_mature_layer": official_mature_layer,
            "candidate_premature_layers": dynamic_layers,
            "omitted_official_candidate_layers": omitted_official_layers,
        },
        "static_layer_candidates": [
            {"internal": layer, "official": layer + 1}
            for layer in static_layers
        ],
        "results": {
            "vanilla": vanilla_summary,
            **{
                f"static_{layer}": summary_payload
                for layer, summary_payload in static_summaries.items()
            },
            str(config.get("baseline_name", "approx_paper_low_bucket_factor")): dynamic_summary,
        },
        "observed_best_static_low": {
            "internal": best_static_layer,
            "official": best_static_layer + 1,
            **best_static_summary,
        },
        "comparisons": {
            "dynamic_minus_vanilla_accuracy": float(dynamic_summary["accuracy"]) - float(vanilla_summary["accuracy"]),
            "dynamic_minus_best_static_low_accuracy": float(dynamic_summary["accuracy"]) - float(best_static_summary["accuracy"]),
            "dynamic_matches_best_static_low": abs(
                float(dynamic_summary["accuracy"]) - float(best_static_summary["accuracy"])
            ) <= 1e-12,
        },
        "dynamic_layer_usage": _serialize_layer_dist(dynamic_layer_usage or None),
        "profiling": {
            "num_samples": len(samples),
            "total_answers": total_answers,
            "total_forward_batches": total_batches,
            "total_wall_time_sec": duration,
            "avg_seconds_per_sample": (duration / len(samples)) if samples else 0.0,
            "avg_seconds_per_answer": (duration / total_answers) if total_answers else 0.0,
        },
    }
    summary["final_label"] = _classify_factor_baseline(
        embedding_inclusive=embedding_inclusive,
        vanilla_accuracy=float(vanilla_summary["accuracy"]),
        dynamic_accuracy=float(dynamic_summary["accuracy"]),
    )

    summary_path = output_dir / SUMMARY_FILE
    _atomic_write_json(summary_path, summary)
    _log(output_dir, f"vanilla accuracy={float(vanilla_summary['accuracy']):.4f}")
    for layer in static_layers:
        _log(output_dir, f"static_{layer} accuracy={float(static_summaries[layer]['accuracy']):.4f}")
    _log(
        output_dir,
        f"{config.get('baseline_name', 'approx_paper_low_bucket_factor')} accuracy="
        f"{float(dynamic_summary['accuracy']):.4f}",
    )
    _log(output_dir, f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
