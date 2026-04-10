"""Run the fixed DoLa paper-transferred TruthfulQA-MC baseline on a full dataset."""

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
from scripts.hf_tune_bucket_truthfulqa_cv import _score_sample_multi_config
from src.dola_utils import normalize_layer_bucket
from src.metrics import aggregate_mc_metrics
from src.modeling import load_model_and_tokenizer
from src.truthfulqa_mc import load_truthfulqa_samples
from src.utils import ensure_output_dir, load_yaml_config


SUMMARY_FILE = "paper_transferred_baseline_summary.json"
RUN_LOG_FILE = "run.log"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the fixed paper-transferred DoLa baseline on TruthfulQA-MC."
    )
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def _log(output_dir: Path, message: str) -> None:
    line = f"[hf_run_paper_transferred_truthfulqa_baseline] {message}"
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


def _aggregate_summary(metric_rows: list[dict[str, float]]) -> dict[str, float | int]:
    aggregate = aggregate_mc_metrics(metric_rows)
    return {
        "MC1": float(aggregate["avg_mc1"]),
        "MC2": float(aggregate["avg_mc2"]),
        "MC3": float(aggregate["avg_mc3"]),
        "num_samples": int(aggregate["num_samples"]),
    }


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    output_dir = ensure_output_dir(
        config.get(
            "output_dir",
            PROJECT_ROOT / "outputs" / "truthfulqa_paper_transferred_baseline",
        )
    )

    model_name = str(config["model_name"])
    device = config.get("device", "cpu")
    csv_path = Path(str(config["csv_path"]))
    prompt_style = str(config.get("prompt_style", "official_tfqa_mc"))
    score_mode = str(config.get("score_mode", "sum_logprob"))
    mature_layer = int(config["mature_layer"])
    post_softmax = bool(config.get("post_softmax", False))
    relative_top = float(config.get("relative_top", 0.0))
    relative_top_value = float(config.get("relative_top_value", -1000.0))
    candidate_batch_size = int(config.get("candidate_batch_size", 1))

    dynamic_bucket_config = config.get("dynamic_bucket")
    if not isinstance(dynamic_bucket_config, dict):
        raise ValueError("dynamic_bucket must be a mapping with name and candidate_premature_layers.")
    dynamic_bucket_name = str(dynamic_bucket_config["name"])
    raw_dynamic_layers = [int(layer) for layer in dynamic_bucket_config["candidate_premature_layers"]]
    raw_static_layers = [int(layer) for layer in config.get("static_layer_candidates", [15])]
    if not raw_static_layers:
        raise ValueError("static_layer_candidates must contain at least one layer.")

    model_kwargs = {key: config[key] for key in LOAD_CONFIG_KEYS if key in config}
    samples = load_truthfulqa_samples(csv_path)
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

    _log(output_dir, f"Baseline label: {config.get('baseline_name', 'DoLa paper-transferred baseline')}")
    _log(output_dir, f"Model: {model_name}")
    _log(output_dir, f"CSV: {csv_path}")
    _log(output_dir, f"Samples: {len(samples)}")
    _log(output_dir, f"Dynamic bucket: {dynamic_bucket_name} -> {dynamic_layers}")
    _log(output_dir, f"Static layers: {static_layers}")
    _log(output_dir, f"Candidate batch size: {candidate_batch_size}")

    vanilla_metric_rows: list[dict[str, float]] = []
    static_metric_rows = {str(layer): [] for layer in static_layers}
    dynamic_metric_rows = {dynamic_bucket_name: []}
    total_answers = 0
    total_batches = 0
    start_time = time.perf_counter()
    last_log_time = start_time

    for sample_index, sample in enumerate(samples):
        bundle = _score_sample_multi_config(
            model=model,
            tokenizer=tokenizer,
            sample=sample,
            prompt_style=prompt_style,
            score_mode=score_mode,
            post_softmax=post_softmax,
            relative_top=relative_top,
            relative_top_value=relative_top_value,
            mature_layer=mature_layer,
            static_layers=static_layers,
            dynamic_buckets=dynamic_bucket_map,
            candidate_batch_size=candidate_batch_size,
        )
        vanilla_metric_rows.append(bundle["vanilla_metrics"])
        for layer in static_layers:
            static_metric_rows[str(layer)].append(bundle["static_metrics"][str(layer)])
        dynamic_metric_rows[dynamic_bucket_name].append(bundle["dynamic_metrics"][dynamic_bucket_name])
        total_answers += int(bundle["profiling"]["answer_count"])
        total_batches += int(bundle["profiling"]["batch_count"])

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
    summary = {
        "task_name": str(config.get("task_name", "truthfulqa_paper_transferred_baseline")),
        "baseline_name": str(config.get("baseline_name", "DoLa paper-transferred baseline")),
        "model_name": model_name,
        "csv_path": str(csv_path),
        "prompt_style": prompt_style,
        "score_mode": score_mode,
        "mature_layer": mature_layer,
        "post_softmax": post_softmax,
        "relative_top": relative_top,
        "relative_top_value": relative_top_value,
        "candidate_batch_size": candidate_batch_size,
        "dynamic_bucket": {
            "name": dynamic_bucket_name,
            "candidate_premature_layers": dynamic_layers,
            "official_early_exit_layers": [layer + 1 for layer in dynamic_layers] + [mature_layer + 1],
        },
        "static_layer_candidates": static_layers,
        "results": {
            "vanilla": _aggregate_summary(vanilla_metric_rows),
            **{
                f"static_{layer}": _aggregate_summary(static_metric_rows[str(layer)])
                for layer in static_layers
            },
            "paper_transferred_dynamic": _aggregate_summary(dynamic_metric_rows[dynamic_bucket_name]),
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

    summary_path = output_dir / SUMMARY_FILE
    _atomic_write_json(summary_path, summary)
    _log(output_dir, f"vanilla MC3={summary['results']['vanilla']['MC3']:.4f}")
    for layer in static_layers:
        _log(output_dir, f"static_{layer} MC3={summary['results'][f'static_{layer}']['MC3']:.4f}")
    _log(output_dir, f"paper_transferred_dynamic MC3={summary['results']['paper_transferred_dynamic']['MC3']:.4f}")
    _log(output_dir, f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
