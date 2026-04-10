"""Run a paper-faithful two-fold tuned-bucket TruthfulQA-MC baseline with resume-safe checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from random import Random
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.hf_eval_compare_subset import LOAD_CONFIG_KEYS
from src.dola_utils import normalize_layer_bucket
from src.generation import score_candidate_answers_multi_config_with_details
from src.metrics import aggregate_mc_metrics, compare_aggregate_metrics, compute_mc_metrics
from src.modeling import load_model_and_tokenizer
from src.truthfulqa_mc import (
    TruthfulQASample,
    build_mc_prompt,
    get_mc_candidate_sets,
    load_truthfulqa_samples,
)
from src.utils import ensure_output_dir, load_yaml_config


STATE_FILE = "cv_state.json"
PROGRESS_FILE = "truthfulqa_tuned_bucket_cv_progress.json"
SUMMARY_FILE = "truthfulqa_tuned_bucket_cv_summary.json"
LEGACY_PARTIAL_FILE = "truthfulqa_tuned_bucket_cv_partial.json"
STAGE_PROFILE_FILE = "stage_profile.jsonl"
STAGE_PROFILE_SUMMARY_FILE = "stage_profile_summary.json"
ETA_WARMUP_SAMPLES = 8


class RunLogger:
    def __init__(self, output_dir: Path) -> None:
        self.path = output_dir / "run.log"

    def log(self, message: str) -> None:
        line = f"[hf_tune_bucket_truthfulqa_cv] {message}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()


class ProgressTracker:
    def __init__(
        self,
        *,
        stage_plan: list[dict[str, object]],
        output_dir: Path,
        logger: RunLogger,
        completed_stage_ids: set[str],
        stage_durations: dict[str, float],
    ) -> None:
        self.stage_plan = stage_plan
        self.output_dir = output_dir
        self.logger = logger
        self.stage_lookup = {str(stage["id"]): stage for stage in stage_plan}
        self.total_stages = len(stage_plan)
        self.total_weight = sum(float(stage["weight"]) for stage in stage_plan)
        self.completed_stage_ids = set(completed_stage_ids)
        self.completed_weight = sum(
            float(stage["weight"])
            for stage in stage_plan
            if str(stage["id"]) in self.completed_stage_ids
        )
        self.completed_stages = sum(
            1 for stage in stage_plan if str(stage["id"]) in self.completed_stage_ids
        )
        self.start_time = time.perf_counter()
        self.current_stage: dict[str, object] | None = None
        self.current_stage_start = self.start_time
        self.current_stage_total_samples = 0
        self.current_stage_layer_count = 1
        self.last_logged_sample = 0
        self.last_log_time = self.start_time
        self.dynamic_seconds_per_sample_layer: float | None = None
        self.static_seconds_per_sample: float | None = None
        self.stage_seconds_per_sample_by_family: dict[str, float] = {}
        self._bootstrap_rates(stage_durations)

    def _bootstrap_rates(self, stage_durations: dict[str, float]) -> None:
        dynamic_rates: list[float] = []
        static_rates: list[float] = []
        family_rates: dict[str, list[float]] = {}
        for stage_id, duration in stage_durations.items():
            stage = self.stage_lookup.get(stage_id)
            if stage is None:
                continue
            samples = int(stage["samples"])
            if samples <= 0:
                continue
            family = _infer_stage_rate_family(stage)
            family_rates.setdefault(family, []).append(float(duration) / samples)
            kind = str(stage["kind"])
            if kind in {"dynamic", "shared"}:
                layer_count = max(int(stage["layer_count"]), 1)
                dynamic_rates.append(float(duration) / (samples * layer_count))
            elif kind == "static":
                static_rates.append(float(duration) / samples)
        self.stage_seconds_per_sample_by_family = {
            family: sum(values) / len(values)
            for family, values in family_rates.items()
            if values
        }
        if dynamic_rates:
            self.dynamic_seconds_per_sample_layer = sum(dynamic_rates) / len(dynamic_rates)
        if static_rates:
            self.static_seconds_per_sample = sum(static_rates) / len(static_rates)

    def _reconcile_stage_runtime_shape(self, stage: dict[str, object]) -> dict[str, object]:
        stage_id = str(stage["id"])
        planned = self.stage_lookup.get(stage_id)
        if planned is None:
            return stage
        planned_weight = float(planned["weight"])
        runtime_weight = float(stage["weight"])
        if math.isclose(planned_weight, runtime_weight, rel_tol=1e-9, abs_tol=1e-9):
            return stage
        self.total_weight += runtime_weight - planned_weight
        planned.update(
            {
                "label": stage["label"],
                "samples": int(stage["samples"]),
                "layer_count": int(stage["layer_count"]),
                "weight": runtime_weight,
            }
        )
        for index, item in enumerate(self.stage_plan):
            if str(item["id"]) == stage_id:
                self.stage_plan[index] = planned
                break
        return planned

    def start_stage(self, stage: dict[str, object]) -> None:
        stage = self._reconcile_stage_runtime_shape(stage)
        self.current_stage = stage
        self.current_stage_start = time.perf_counter()
        self.current_stage_total_samples = int(stage["samples"])
        self.current_stage_layer_count = max(int(stage["layer_count"]), 1)
        self.last_logged_sample = 0
        self.last_log_time = time.perf_counter()
        stage_eta_seconds = self._estimate_current_stage_remaining_seconds(0)
        total_eta_seconds = self._estimate_eta(0, stage_eta_seconds=stage_eta_seconds)
        self._write_progress_snapshot(
            completed_samples=0,
            stage_eta_seconds=stage_eta_seconds,
            total_eta_seconds=total_eta_seconds,
        )
        self.logger.log(
            f"Stage {self.completed_stages + 1}/{self.total_stages} start: {stage['label']} "
            f"| samples={self.current_stage_total_samples} | layer_factor={self.current_stage_layer_count}"
        )

    def make_callback(self) -> Any:
        def _callback(event: dict[str, object]) -> None:
            completed_samples = int(event["completed_samples"])
            now = time.perf_counter()
            sample_stride = 1 if completed_samples <= 5 else 5
            should_log = (
                completed_samples <= 5
                or completed_samples == self.current_stage_total_samples
                or completed_samples - self.last_logged_sample >= sample_stride
                or now - self.last_log_time >= 30.0
            )
            if not should_log:
                return
            self.last_logged_sample = completed_samples
            self.last_log_time = now
            elapsed = time.perf_counter() - self.start_time
            percent = 100.0 * self._completed_weight_within_stage(completed_samples) / max(self.total_weight, 1.0)
            stage_eta_seconds = self._estimate_current_stage_remaining_seconds(completed_samples)
            total_eta_seconds = self._estimate_eta(completed_samples, stage_eta_seconds=stage_eta_seconds)
            stage_bar = _progress_bar(completed_samples, self.current_stage_total_samples)
            self.logger.log(
                f"{stage_bar} {percent:5.1f}% | {self.current_stage['label']} "
                f"| {completed_samples}/{self.current_stage_total_samples} samples "
                f"| elapsed={_format_seconds(elapsed)} "
                f"| stage_eta={_format_seconds(stage_eta_seconds)} "
                f"| total_eta={_format_seconds(total_eta_seconds)}"
            )
            self._write_progress_snapshot(
                completed_samples=completed_samples,
                stage_eta_seconds=stage_eta_seconds,
                total_eta_seconds=total_eta_seconds,
            )

        return _callback

    def finish_stage(self) -> float:
        if self.current_stage is None:
            raise RuntimeError("No active stage to finish.")
        stage_elapsed = time.perf_counter() - self.current_stage_start
        if self.current_stage_total_samples > 0:
            observed_seconds_per_sample = stage_elapsed / self.current_stage_total_samples
            family = _infer_stage_rate_family(self.current_stage)
            self.stage_seconds_per_sample_by_family[family] = _ema(
                self.stage_seconds_per_sample_by_family.get(family),
                observed_seconds_per_sample,
            )
            current_kind = str(self.current_stage["kind"])
            if current_kind in {"dynamic", "shared"}:
                observed = stage_elapsed / (self.current_stage_total_samples * self.current_stage_layer_count)
                self.dynamic_seconds_per_sample_layer = _ema(self.dynamic_seconds_per_sample_layer, observed)
            else:
                self.static_seconds_per_sample = _ema(self.static_seconds_per_sample, observed_seconds_per_sample)
        self.completed_stage_ids.add(str(self.current_stage["id"]))
        self.completed_weight += float(self.current_stage["weight"])
        self.completed_stages += 1
        self.logger.log(
            f"Stage {self.completed_stages}/{self.total_stages} done: {self.current_stage['label']} "
            f"| duration={_format_seconds(stage_elapsed)}"
        )
        self._write_progress_snapshot(
            completed_samples=self.current_stage_total_samples,
            stage_eta_seconds=0.0,
            total_eta_seconds=self._estimate_eta(self.current_stage_total_samples, stage_eta_seconds=0.0),
        )
        self.current_stage = None
        return stage_elapsed

    def _completed_weight_within_stage(self, completed_samples: int) -> float:
        if self.current_stage is None or self.current_stage_total_samples <= 0:
            return self.completed_weight
        fraction = min(max(completed_samples / self.current_stage_total_samples, 0.0), 1.0)
        return self.completed_weight + float(self.current_stage["weight"]) * fraction

    def _estimate_current_stage_remaining_seconds(self, completed_samples: int) -> float:
        if self.current_stage is None or self.current_stage_total_samples <= 0:
            return 0.0
        if completed_samples >= self.current_stage_total_samples:
            return 0.0
        remaining_samples = self.current_stage_total_samples - completed_samples
        prior_stage_seconds = self._estimate_stage_seconds(self.current_stage)
        prior_seconds_per_sample: float | None = None
        if prior_stage_seconds > 0.0:
            prior_seconds_per_sample = prior_stage_seconds / self.current_stage_total_samples
        if completed_samples <= 0:
            return remaining_samples * (prior_seconds_per_sample or 0.0)
        observed_seconds_per_sample = (time.perf_counter() - self.current_stage_start) / completed_samples
        blended_seconds_per_sample = _blend_eta_seconds_per_sample(
            prior_seconds_per_sample=prior_seconds_per_sample,
            observed_seconds_per_sample=observed_seconds_per_sample,
            completed_samples=completed_samples,
        )
        return remaining_samples * blended_seconds_per_sample

    def _estimate_eta(self, completed_samples: int, *, stage_eta_seconds: float | None = None) -> float:
        if stage_eta_seconds is None:
            stage_eta_seconds = self._estimate_current_stage_remaining_seconds(completed_samples)

        current_index = -1
        if self.current_stage is not None:
            current_index = next(
                index for index, stage in enumerate(self.stage_plan) if str(stage["id"]) == str(self.current_stage["id"])
            )
        remaining_future = 0.0
        for stage in self.stage_plan[current_index + 1 :]:
            if str(stage["id"]) in self.completed_stage_ids:
                continue
            remaining_future += self._estimate_stage_seconds(stage)
        return max(stage_eta_seconds + remaining_future, 0.0)

    def _estimate_stage_seconds(self, stage: dict[str, object]) -> float:
        samples = int(stage["samples"])
        if samples <= 0:
            return 0.0
        family = _infer_stage_rate_family(stage)
        family_rate = self.stage_seconds_per_sample_by_family.get(family)
        if family_rate is not None:
            return samples * family_rate
        kind = str(stage["kind"])
        layer_count = max(int(stage["layer_count"]), 1)
        if kind in {"dynamic", "shared"}:
            if self.dynamic_seconds_per_sample_layer is not None:
                return samples * layer_count * self.dynamic_seconds_per_sample_layer
            return samples * layer_count * 1.0
        if self.static_seconds_per_sample is not None:
            return samples * self.static_seconds_per_sample
        return samples * 1.0

    def _write_progress_snapshot(
        self,
        *,
        completed_samples: int,
        stage_eta_seconds: float,
        total_eta_seconds: float,
    ) -> None:
        snapshot = {
            "completed_stages": self.completed_stages,
            "total_stages": self.total_stages,
            "current_stage": None if self.current_stage is None else self.current_stage["label"],
            "current_stage_kind": None if self.current_stage is None else self.current_stage["kind"],
            "current_stage_rate_family": None if self.current_stage is None else _infer_stage_rate_family(self.current_stage),
            "current_stage_completed_samples": completed_samples,
            "current_stage_total_samples": self.current_stage_total_samples,
            "elapsed_seconds": time.perf_counter() - self.start_time,
            "stage_eta_seconds": stage_eta_seconds,
            "eta_seconds": total_eta_seconds,
            "total_eta_seconds": total_eta_seconds,
            "eta_method": "blended_stage_family_resume_safe",
            "completed_weight": self._completed_weight_within_stage(completed_samples),
            "total_weight": self.total_weight,
            "dynamic_seconds_per_sample_layer": self.dynamic_seconds_per_sample_layer,
            "static_seconds_per_sample": self.static_seconds_per_sample,
            "stage_seconds_per_sample_by_family": {
                key: float(value)
                for key, value in sorted(self.stage_seconds_per_sample_by_family.items())
            },
        }
        _atomic_write_json(self.output_dir / PROGRESS_FILE, snapshot)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run two-fold TruthfulQA-MC bucket tuning using MC3 on validation folds."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--rebuild-summary-only", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--profile-first-n", type=int, default=0)
    parser.add_argument("--profile-batch-sizes", type=str, default="1,2,4")
    return parser.parse_args()


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _ema(previous: float | None, current: float, momentum: float = 0.7) -> float:
    if previous is None:
        return current
    return previous * momentum + current * (1.0 - momentum)


def _format_seconds(seconds: float) -> str:
    total_seconds = max(int(seconds), 0)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _progress_bar(completed: int, total: int, width: int = 20) -> str:
    if total <= 0:
        return "[" + "-" * width + "]"
    filled = min(width, int(width * completed / total))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def _infer_stage_rate_family(stage: dict[str, object]) -> str:
    stage_id = str(stage.get("id", ""))
    label = str(stage.get("label", "")).lower()
    kind = str(stage.get("kind", ""))
    if ".validation.shared" in stage_id or "validation shared" in label:
        return "validation_shared"
    if ".test.shared" in stage_id or "held-out shared" in label:
        return "test_shared"
    if ".validation.dynamic." in stage_id:
        return "validation_dynamic"
    if ".validation.static." in stage_id:
        return "validation_static"
    if kind in {"shared", "dynamic", "static"}:
        return kind
    return "generic"


def _blend_eta_seconds_per_sample(
    *,
    prior_seconds_per_sample: float | None,
    observed_seconds_per_sample: float,
    completed_samples: int,
    warmup_samples: int = ETA_WARMUP_SAMPLES,
) -> float:
    if prior_seconds_per_sample is None or prior_seconds_per_sample <= 0.0:
        return observed_seconds_per_sample
    if completed_samples <= 0 or warmup_samples <= 0:
        return observed_seconds_per_sample
    observed_weight = min(max(completed_samples / warmup_samples, 0.0), 1.0)
    return ((1.0 - observed_weight) * prior_seconds_per_sample) + (observed_weight * observed_seconds_per_sample)

def _parse_profile_batch_sizes(raw: str) -> list[int]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("profile batch sizes must contain at least one integer.")
    batch_sizes = [int(value) for value in values]
    if any(value <= 0 for value in batch_sizes):
        raise ValueError("profile batch sizes must be positive integers.")
    return batch_sizes


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        handle.flush()


def _write_jsonl_records(path: Path, records: list[dict[str, Any]]) -> None:
    if path.exists():
        path.unlink()
    for record in records:
        _append_jsonl(path, record)



def _merge_layer_usage_counts(
    aggregate_usage: dict[int, int],
    sample_usage: dict[str, int] | dict[int, int] | None,
) -> None:
    if not sample_usage:
        return
    for layer, count in sample_usage.items():
        aggregate_usage[int(layer)] = aggregate_usage.get(int(layer), 0) + int(count)



def _build_inner_sample_profile_record(
    *,
    sample_index: int,
    sample: TruthfulQASample,
    profiling: dict[str, Any],
    union_layer_count: int,
) -> dict[str, Any]:
    return {
        "sample_index": sample_index,
        "question": sample.question,
        "num_candidates": int(profiling["answer_count"]),
        "candidate_batch_size": int(profiling["batch_size"]),
        "prompt_len": int(profiling["prompt_len"]),
        "max_total_len": int(profiling["max_total_len"]),
        "union_layer_count": int(profiling["union_layer_count"] or union_layer_count),
        "build_prompt_and_candidates_sec": float(profiling["build_prompt_and_candidates_sec"]),
        "tokenize_sec": float(profiling["tokenize_sec"]),
        "model_forward_sec": float(profiling["model_forward_sec"]),
        "dynamic_rescore_sec": float(profiling["dynamic_rescore_sec"]),
        "static_rescore_sec": float(profiling["static_rescore_sec"]),
        "aggregate_write_sec": float(profiling["aggregate_write_sec"]),
        "materialize_sec": float(profiling["materialize_sec"]),
        "total_sample_sec": float(profiling["total_sample_sec"]),
        "forward_batches": int(profiling["batch_count"]),
    }



def _build_stage_sample_profile_record(
    *,
    sample_index: int,
    sample: TruthfulQASample,
    bundle: dict[str, Any],
    inner_score_sample_sec: float,
    metrics_aggregation_sec: float,
    usage_merge_sec: float,
    progress_callback_sec: float,
) -> dict[str, Any]:
    profiling = bundle["profiling"]
    total_stage_sample_sec = (
        inner_score_sample_sec
        + metrics_aggregation_sec
        + usage_merge_sec
        + progress_callback_sec
    )
    return {
        "sample_index": sample_index,
        "question": sample.question,
        "answer_count": int(profiling["answer_count"]),
        "batch_count": int(profiling["batch_count"]),
        "prompt_len": int(profiling["prompt_len"]),
        "max_total_len": int(profiling["max_total_len"]),
        "inner_score_sample_sec": float(inner_score_sample_sec),
        "metrics_aggregation_sec": float(metrics_aggregation_sec),
        "usage_merge_sec": float(usage_merge_sec),
        "progress_callback_sec": float(progress_callback_sec),
        "state_persist_sec": 0.0,
        "total_stage_sample_sec": float(total_stage_sample_sec),
        "build_prompt_and_candidates_sec": float(profiling["build_prompt_and_candidates_sec"]),
        "tokenize_sec": float(profiling["tokenize_sec"]),
        "model_forward_sec": float(profiling["model_forward_sec"]),
        "dynamic_rescore_sec": float(profiling["dynamic_rescore_sec"]),
        "static_rescore_sec": float(profiling["static_rescore_sec"]),
        "materialize_sec": float(profiling["materialize_sec"]),
    }



def _summarize_inner_profile_records(
    records: list[dict[str, Any]],
    *,
    batch_size: int,
    sample_profile_path: Path,
) -> dict[str, Any]:
    return {
        "candidate_batch_size": batch_size,
        "num_samples_completed": len(records),
        "avg_total_sample_sec": _average([float(item["total_sample_sec"]) for item in records]),
        "avg_build_prompt_and_candidates_sec": _average([float(item["build_prompt_and_candidates_sec"]) for item in records]),
        "avg_tokenize_sec": _average([float(item["tokenize_sec"]) for item in records]),
        "avg_model_forward_sec": _average([float(item["model_forward_sec"]) for item in records]),
        "avg_dynamic_rescore_sec": _average([float(item["dynamic_rescore_sec"]) for item in records]),
        "avg_static_rescore_sec": _average([float(item["static_rescore_sec"]) for item in records]),
        "avg_aggregate_write_sec": _average([float(item["aggregate_write_sec"]) for item in records]),
        "avg_forward_batches": _average([float(item["forward_batches"]) for item in records]),
        "sample_profile_path": str(sample_profile_path),
    }



def _summarize_stage_profile_records(
    records: list[dict[str, Any]],
    *,
    batch_size: int,
    stage_profile_path: Path,
) -> dict[str, Any]:
    return {
        "candidate_batch_size": batch_size,
        "num_samples_completed": len(records),
        "avg_inner_score_sample_sec": _average([float(item["inner_score_sample_sec"]) for item in records]),
        "avg_metrics_aggregation_sec": _average([float(item["metrics_aggregation_sec"]) for item in records]),
        "avg_usage_merge_sec": _average([float(item["usage_merge_sec"]) for item in records]),
        "avg_progress_callback_sec": _average([float(item["progress_callback_sec"]) for item in records]),
        "avg_state_persist_sec": _average([float(item["state_persist_sec"]) for item in records]),
        "avg_total_stage_sample_sec": _average([float(item["total_stage_sample_sec"]) for item in records]),
        "avg_answer_count": _average([float(item["answer_count"]) for item in records]),
        "avg_batch_count": _average([float(item["batch_count"]) for item in records]),
        "avg_prompt_len": _average([float(item["prompt_len"]) for item in records]),
        "avg_max_total_len": _average([float(item["max_total_len"]) for item in records]),
        "stage_profile_path": str(stage_profile_path),
    }



def _consume_validation_shared_bundle(
    *,
    bundle: dict[str, Any],
    dynamic_buckets: list[dict[str, object]],
    static_layers: list[int],
    vanilla_metric_rows: list[dict[str, float]],
    dynamic_metric_rows: dict[str, list[dict[str, float]]],
    dynamic_layer_usage: dict[str, dict[int, int]],
    static_metric_rows: dict[str, list[dict[str, float]]],
    static_layer_usage: dict[str, dict[int, int]],
) -> dict[str, float | int]:
    metrics_start = time.perf_counter()
    vanilla_metric_rows.append(bundle["vanilla_metrics"])
    for bucket in dynamic_buckets:
        bucket_name = str(bucket["name"])
        dynamic_metric_rows[bucket_name].append(bundle["dynamic_metrics"][bucket_name])
    for layer in static_layers:
        layer_key = str(layer)
        static_metric_rows[layer_key].append(bundle["static_metrics"][layer_key])
    metrics_aggregation_sec = time.perf_counter() - metrics_start

    usage_start = time.perf_counter()
    for bucket in dynamic_buckets:
        bucket_name = str(bucket["name"])
        _merge_layer_usage_counts(dynamic_layer_usage[bucket_name], bundle["dynamic_layer_usage"].get(bucket_name) or {})
    for layer in static_layers:
        layer_key = str(layer)
        _merge_layer_usage_counts(static_layer_usage[layer_key], bundle["static_layer_usage"].get(layer_key) or {})
    usage_merge_sec = time.perf_counter() - usage_start
    profiling = bundle["profiling"]
    return {
        "metrics_aggregation_sec": metrics_aggregation_sec,
        "usage_merge_sec": usage_merge_sec,
        "answer_count": int(profiling["answer_count"]),
        "batch_count": int(profiling["batch_count"]),
    }



def _safe_empty_cuda_cache() -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _average(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _run_profile_mode(
    *,
    logger: RunLogger,
    output_dir: Path,
    model: Any,
    tokenizer: Any,
    samples: list[TruthfulQASample],
    prompt_style: str,
    score_mode: str,
    post_softmax: bool,
    relative_top: float,
    relative_top_value: float,
    mature_layer: int,
    dynamic_buckets: list[dict[str, object]],
    static_layers: list[int],
    profile_first_n: int,
    profile_batch_sizes: list[int],
) -> None:
    profile_samples = samples[:profile_first_n]
    union_layer_count = len({*static_layers, *(layer for item in dynamic_buckets for layer in item["candidate_premature_layers"])})
    summary: dict[str, Any] = {"profile_first_n": profile_first_n, "num_profile_samples": len(profile_samples), "batch_size_results": []}
    logger.log(f"Profiling mode: running first {len(profile_samples)} validation-stage samples on the real shared-scoring path with batch sizes {profile_batch_sizes}.")
    multi_batch = len(profile_batch_sizes) > 1
    for batch_size in profile_batch_sizes:
        sample_profile_path = output_dir / f"sample_profile.batch{batch_size}.jsonl"
        stage_profile_path = output_dir / (f"stage_profile.batch{batch_size}.jsonl" if multi_batch else STAGE_PROFILE_FILE)
        stage_profile_summary_path = output_dir / (f"stage_profile_summary.batch{batch_size}.json" if multi_batch else STAGE_PROFILE_SUMMARY_FILE)
        for artifact_path in (sample_profile_path, stage_profile_path, stage_profile_summary_path):
            if artifact_path.exists():
                artifact_path.unlink()

        scratch_output_dir = output_dir / "_stage_profile_scratch" / f"batch{batch_size}"
        stage = {
            "id": f"profile.validation.shared.batch{batch_size}",
            "label": f"profile validation shared batch {batch_size}",
            "kind": "shared",
            "samples": len(profile_samples),
            "layer_count": max(union_layer_count, 1),
            "weight": _stage_weight("shared", len(profile_samples), max(union_layer_count, 1)),
        }
        tracker = ProgressTracker(stage_plan=[stage], output_dir=scratch_output_dir, logger=logger, completed_stage_ids=set(), stage_durations={})
        scratch_state = {
            "task_name": "truthfulqa_tuned_bucket_cv_stage_profile",
            "model_name": str(getattr(model, "name_or_path", "profile_model")),
            "csv_path": "profile_first_n",
            "split": {"mode": "profile_validation_shared_first_n", "num_samples": len(profile_samples)},
            "prompt_style": prompt_style,
            "score_mode": score_mode,
            "mature_layer": mature_layer,
            "dynamic_bucket_candidates": dynamic_buckets,
            "static_layer_candidates": [{"internal": layer, "official": layer + 1} for layer in static_layers],
            "transfer_reference": None,
            "completed_stages": [],
            "stage_durations_seconds": {},
            "folds": {},
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
        }
        scratch_fold_state = _ensure_fold_state(scratch_state, {"name": f"profile_batch{batch_size}", "validation_size": len(profile_samples), "test_size": 0})

        logger.log(f"Profiling batch size {batch_size} -> stage={stage_profile_path}, inner={sample_profile_path}")
        status = "ok"
        error_message: str | None = None
        total_wall_start = time.perf_counter()
        run_artifacts: dict[str, Any] | None = None
        try:
            run_artifacts = _run_validation_shared_stage(
                stage=stage,
                tracker=tracker,
                logger=logger,
                state=scratch_state,
                fold_state=scratch_fold_state,
                output_dir=scratch_output_dir,
                model=model,
                tokenizer=tokenizer,
                validation_sample_ids=list(range(len(profile_samples))),
                validation_samples=profile_samples,
                dynamic_buckets=dynamic_buckets,
                static_layers=static_layers,
                mature_layer=mature_layer,
                prompt_style=prompt_style,
                score_mode=score_mode,
                post_softmax=post_softmax,
                relative_top=relative_top,
                relative_top_value=relative_top_value,
                candidate_batch_size=batch_size,
                stage_profile_limit=profile_first_n,
                stage_profile_path=stage_profile_path,
                stage_profile_summary_path=stage_profile_summary_path,
                sample_profile_path=sample_profile_path,
            )
        except RuntimeError as error:
            status = "runtime_error"
            error_message = str(error)
            if "out of memory" in error_message.lower():
                status = "oom"
            _safe_empty_cuda_cache()
            logger.log(f"Profiling batch size {batch_size} failed with {status}: {error_message}")
        total_wall_time = time.perf_counter() - total_wall_start

        sample_profile_summary = None if run_artifacts is None else run_artifacts.get("sample_profile_summary")
        stage_profile_summary = None if run_artifacts is None else run_artifacts.get("stage_profile_summary")
        selected_dynamic = _select_best_dynamic(list(scratch_fold_state["validation_dynamic_candidates"].values())) if scratch_fold_state["validation_dynamic_candidates"] else None
        selected_static = _select_best_static(list(scratch_fold_state["validation_static_candidates"].values())) if scratch_fold_state["validation_static_candidates"] else None
        result = {
            "candidate_batch_size": batch_size,
            "status": status,
            "error": error_message,
            "num_samples_completed": 0 if run_artifacts is None else int(run_artifacts["num_samples_completed"]),
            "total_wall_time": total_wall_time,
            "avg_total_sample_sec": 0.0 if sample_profile_summary is None else float(sample_profile_summary["avg_total_sample_sec"]),
            "avg_build_prompt_and_candidates_sec": 0.0 if sample_profile_summary is None else float(sample_profile_summary["avg_build_prompt_and_candidates_sec"]),
            "avg_tokenize_sec": 0.0 if sample_profile_summary is None else float(sample_profile_summary["avg_tokenize_sec"]),
            "avg_model_forward_sec": 0.0 if sample_profile_summary is None else float(sample_profile_summary["avg_model_forward_sec"]),
            "avg_dynamic_rescore_sec": 0.0 if sample_profile_summary is None else float(sample_profile_summary["avg_dynamic_rescore_sec"]),
            "avg_static_rescore_sec": 0.0 if sample_profile_summary is None else float(sample_profile_summary["avg_static_rescore_sec"]),
            "avg_aggregate_write_sec": 0.0 if sample_profile_summary is None else float(sample_profile_summary["avg_aggregate_write_sec"]),
            "avg_forward_batches": 0.0 if sample_profile_summary is None else float(sample_profile_summary["avg_forward_batches"]),
            "avg_inner_score_sample_sec": 0.0 if stage_profile_summary is None else float(stage_profile_summary["avg_inner_score_sample_sec"]),
            "avg_metrics_aggregation_sec": 0.0 if stage_profile_summary is None else float(stage_profile_summary["avg_metrics_aggregation_sec"]),
            "avg_usage_merge_sec": 0.0 if stage_profile_summary is None else float(stage_profile_summary["avg_usage_merge_sec"]),
            "avg_progress_callback_sec": 0.0 if stage_profile_summary is None else float(stage_profile_summary["avg_progress_callback_sec"]),
            "avg_state_persist_sec": 0.0 if stage_profile_summary is None else float(stage_profile_summary["avg_state_persist_sec"]),
            "avg_total_stage_sample_sec": 0.0 if stage_profile_summary is None else float(stage_profile_summary["avg_total_stage_sample_sec"]),
            "sample_profile_path": str(sample_profile_path),
            "stage_profile_path": str(stage_profile_path),
            "stage_profile_summary_path": str(stage_profile_summary_path),
            "selected_dynamic_bucket": None if selected_dynamic is None else str(selected_dynamic["name"]),
            "selected_static_layer": None if selected_static is None else int(selected_static["premature_layer"]),
        }
        summary["batch_size_results"].append(result)
        if stage_profile_summary is not None:
            logger.log(f"profile batch={batch_size} avg_stage_sample_s={result['avg_total_stage_sample_sec']:.2f} inner_s={result['avg_inner_score_sample_sec']:.2f} callback_s={result['avg_progress_callback_sec']:.4f} persist_s={result['avg_state_persist_sec']:.4f}")

    summary_path = output_dir / "profile_summary.json"
    _atomic_write_json(summary_path, summary)
    logger.log(f"Saved end-to-end stage profiling summary to: {summary_path}")



def _load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
    tmp_path.replace(path)


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


def _aggregate_layer_usage(items: list[object]) -> dict[int, int] | None:
    aggregate: dict[int, int] = {}
    for item in items:
        layer_dist = getattr(item, "premature_layer_dist", None)
        if not layer_dist:
            continue
        for layer, count in layer_dist.items():
            aggregate[int(layer)] = aggregate.get(int(layer), 0) + int(count)
    return aggregate or None



def _build_candidate_summary(
    *,
    vanilla_metric_rows: list[dict[str, float]],
    dola_metric_rows: list[dict[str, float]],
    prompt_style: str,
    score_mode: str,
    dola_score_mode: str,
    post_softmax: bool,
    relative_top: float,
    relative_top_value: float,
    premature_layer: int,
    candidate_premature_layers: list[int] | None,
    mature_layer: int,
    layer_usage: dict[int, int] | None,
) -> dict[str, Any]:
    vanilla_summary = aggregate_mc_metrics(vanilla_metric_rows)
    dola_summary = aggregate_mc_metrics(dola_metric_rows)
    summary = compare_aggregate_metrics(vanilla_summary, dola_summary)
    summary.update(
        {
            "prompt_style": prompt_style,
            "score_mode": score_mode,
            "dola_score_mode": dola_score_mode,
            "post_softmax": post_softmax,
            "relative_top": relative_top,
            "relative_top_value": relative_top_value,
            "premature_layer": premature_layer,
            "candidate_premature_layers": candidate_premature_layers,
            "mature_layer": mature_layer,
            "premature_layer_dist": None if not layer_usage else {str(layer): count for layer, count in sorted(layer_usage.items())},
        }
    )
    return summary



def _score_sample_multi_config(
    *,
    model: Any,
    tokenizer: Any,
    sample: TruthfulQASample,
    prompt_style: str,
    score_mode: str,
    post_softmax: bool,
    relative_top: float,
    relative_top_value: float,
    mature_layer: int,
    static_layers: list[int],
    dynamic_buckets: dict[str, list[int]],
    candidate_batch_size: int,
) -> dict[str, Any]:
    sample_start = time.perf_counter()
    build_start = sample_start
    prompt = build_mc_prompt(sample, prompt_style=prompt_style)
    true_candidates, false_candidates = get_mc_candidate_sets(sample, prompt_style=prompt_style)
    all_candidates = [*true_candidates, *false_candidates]
    num_true = len(true_candidates)
    build_prompt_and_candidates_sec = time.perf_counter() - build_start

    score_result = score_candidate_answers_multi_config_with_details(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        candidate_answers=all_candidates,
        static_layers=static_layers,
        dynamic_buckets=dynamic_buckets,
        score_mode=score_mode,
        post_softmax=post_softmax,
        relative_top=relative_top,
        relative_top_value=relative_top_value,
        mature_layer=mature_layer,
        candidate_batch_size=candidate_batch_size,
    )

    aggregate_start = time.perf_counter()
    vanilla_true = score_result.vanilla[:num_true]
    vanilla_false = score_result.vanilla[num_true:]
    static_metrics = {}
    static_usage = {}
    for layer, items in score_result.static.items():
        static_metrics[str(layer)] = compute_mc_metrics(
            [item.score for item in items[:num_true]],
            [item.score for item in items[num_true:]],
        )
        static_usage[str(layer)] = _aggregate_layer_usage(items)
    dynamic_metrics = {}
    dynamic_usage = {}
    for bucket_name, items in score_result.dynamic.items():
        dynamic_metrics[bucket_name] = compute_mc_metrics(
            [item.score for item in items[:num_true]],
            [item.score for item in items[num_true:]],
        )
        dynamic_usage[bucket_name] = _aggregate_layer_usage(items)

    vanilla_metrics = compute_mc_metrics(
        [item.score for item in vanilla_true],
        [item.score for item in vanilla_false],
    )
    aggregate_write_sec = time.perf_counter() - aggregate_start
    scorer_profile = score_result.profile or {}
    total_sample_sec = time.perf_counter() - sample_start

    return {
        "vanilla_metrics": vanilla_metrics,
        "static_metrics": static_metrics,
        "static_layer_usage": static_usage,
        "dynamic_metrics": dynamic_metrics,
        "dynamic_layer_usage": dynamic_usage,
        "profiling": {
            "answer_count": len(all_candidates),
            "batch_count": score_result.batch_count,
            "batch_size": score_result.batch_size,
            "build_prompt_and_candidates_sec": build_prompt_and_candidates_sec,
            "tokenize_sec": float(scorer_profile.get("tokenize_sec", 0.0)),
            "model_forward_sec": float(scorer_profile.get("model_forward_sec", 0.0)),
            "dynamic_rescore_sec": float(scorer_profile.get("dynamic_rescore_sec", 0.0)),
            "static_rescore_sec": float(scorer_profile.get("static_rescore_sec", 0.0)),
            "aggregate_write_sec": aggregate_write_sec,
            "materialize_sec": float(scorer_profile.get("materialize_sec", 0.0)),
            "total_sample_sec": total_sample_sec,
            "prompt_len": int(scorer_profile.get("prompt_len", 0)),
            "max_total_len": int(scorer_profile.get("max_total_len", 0)),
            "union_layer_count": int(scorer_profile.get("union_layer_count", 0)),
        },
    }



def _get_or_build_sample_bundle(
    *,
    sample_cache: dict[int, dict[str, Any]],
    sample_id: int,
    model: Any,
    tokenizer: Any,
    sample: TruthfulQASample,
    prompt_style: str,
    score_mode: str,
    post_softmax: bool,
    relative_top: float,
    relative_top_value: float,
    mature_layer: int,
    static_layers: list[int],
    dynamic_buckets: dict[str, list[int]],
    candidate_batch_size: int,
) -> tuple[dict[str, Any], bool]:
    cached = sample_cache.get(sample_id)
    if cached is not None:
        return cached, True
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
        dynamic_buckets=dynamic_buckets,
        candidate_batch_size=candidate_batch_size,
    )
    sample_cache[sample_id] = bundle
    return bundle, False


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
    progress_callback: Any,
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
        progress_callback=progress_callback,
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
    progress_callback: Any,
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
        progress_callback=progress_callback,
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


def _stage_weight(kind: str, samples: int, layer_count: int) -> float:
    return float(samples * (layer_count if kind in {"dynamic", "shared"} else 1))


def _estimate_expected_test_shared_layer_count(
    dynamic_buckets: list[dict[str, object]],
    static_layers: list[int],
) -> int:
    if not dynamic_buckets and not static_layers:
        return 1
    normalized_static_layers = [int(layer) for layer in static_layers]
    if not dynamic_buckets:
        return max(len(set(normalized_static_layers)), 1)
    if not normalized_static_layers:
        return max(
            int(round(sum(len(set(int(layer) for layer in bucket["candidate_premature_layers"])) for bucket in dynamic_buckets) / len(dynamic_buckets))),
            1,
        )
    union_counts: list[int] = []
    for bucket in dynamic_buckets:
        bucket_layers = {int(layer) for layer in bucket["candidate_premature_layers"]}
        for static_layer in normalized_static_layers:
            union_counts.append(len(bucket_layers | {int(static_layer)}))
    return max(int(round(sum(union_counts) / len(union_counts))), 1)


def _build_stage_plan(
    fold_specs: list[dict[str, object]],
    dynamic_buckets: list[dict[str, object]],
    static_layers: list[int],
) -> list[dict[str, object]]:
    plan: list[dict[str, object]] = []
    validation_union_layers = sorted(
        {
            *static_layers,
            *(layer for bucket in dynamic_buckets for layer in bucket["candidate_premature_layers"]),
        }
    )
    validation_union_count = max(len(validation_union_layers), 1)
    expected_test_shared_layer_count = _estimate_expected_test_shared_layer_count(dynamic_buckets, static_layers)
    for fold_spec in fold_specs:
        fold_name = str(fold_spec["name"])
        validation_size = int(fold_spec["validation_size"])
        test_size = int(fold_spec["test_size"])
        plan.append(
            {
                "id": f"{fold_name}.validation.shared",
                "label": f"{fold_name} validation shared scoring",
                "kind": "shared",
                "samples": validation_size,
                "layer_count": validation_union_count,
                "weight": _stage_weight("shared", validation_size, validation_union_count),
            }
        )
        plan.append(
            {
                "id": f"{fold_name}.test.shared",
                "label": f"{fold_name} held-out shared scoring",
                "kind": "shared",
                "samples": test_size,
                "layer_count": expected_test_shared_layer_count,
                "weight": _stage_weight("shared", test_size, expected_test_shared_layer_count),
            }
        )
    return plan


def _candidate_count_stats(samples: list[TruthfulQASample]) -> dict[str, float | int]:
    return {
        "num_samples": len(samples),
        "avg_true_answers": sum(len(sample.correct_answers) for sample in samples) / len(samples),
        "avg_false_answers": sum(len(sample.incorrect_answers) for sample in samples) / len(samples),
        "max_true_answers": max(len(sample.correct_answers) for sample in samples),
        "max_false_answers": max(len(sample.incorrect_answers) for sample in samples),
    }


def _build_run_signature(config: dict[str, Any]) -> str:
    payload = {
        "model_name": str(config["model_name"]),
        "csv_path": str(config["csv_path"]),
        "split_seed": int(config.get("split_seed", 20260407)),
        "prompt_style": str(config.get("prompt_style", "official_tfqa_mc")),
        "score_mode": str(config.get("score_mode", "sum_logprob")),
        "mature_layer": int(config["mature_layer"]),
        "dynamic_bucket_candidates": config["dynamic_bucket_candidates"],
        "static_layer_candidates": config["static_layer_candidates"],
        "post_softmax": bool(config.get("post_softmax", False)),
        "relative_top": float(config.get("relative_top", 0.0)),
        "relative_top_value": float(config.get("relative_top_value", -1000.0)),
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

def _state_skeleton(
    config: dict[str, Any],
    run_signature: str,
    split_meta: dict[str, object],
    dynamic_buckets: list[dict[str, object]],
    static_layers: list[int],
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "run_signature": run_signature,
        "task_name": str(config.get("task_name", "truthfulqa_tuned_bucket_cv")),
        "model_name": str(config["model_name"]),
        "csv_path": str(config["csv_path"]),
        "subset_mode": "full_dataset",
        "prompt_style": str(config.get("prompt_style", "official_tfqa_mc")),
        "score_mode": str(config.get("score_mode", "sum_logprob")),
        "mature_layer": int(config["mature_layer"]),
        "split_seed": int(config.get("split_seed", 20260407)),
        "split": split_meta,
        "dynamic_bucket_candidates": dynamic_buckets,
        "static_layer_candidates": [{"internal": layer, "official": layer + 1} for layer in static_layers],
        "transfer_reference": config.get("transfer_reference_summary"),
        "completed_stages": [],
        "stage_durations_seconds": {},
        "folds": {},
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
    }


def _validate_state(state: dict[str, Any], run_signature: str) -> None:
    if str(state.get("run_signature", "")) != run_signature:
        raise ValueError(
            "Existing cv_state.json does not match the current config/signature. "
            "Use a new output_dir or remove the stale state before rerunning."
        )


def _mark_stage_complete(
    state: dict[str, Any],
    *,
    stage_id: str,
    duration_seconds: float,
) -> None:
    completed = set(str(item) for item in state.get("completed_stages", []))
    if stage_id not in completed:
        completed.add(stage_id)
        state["completed_stages"] = sorted(completed)
    stage_durations = state.setdefault("stage_durations_seconds", {})
    stage_durations[stage_id] = float(duration_seconds)
    state["updated_at"] = _utc_now()


def _fold_partial_path(output_dir: Path, fold_name: str) -> Path:
    return output_dir / f"{fold_name}_partial.json"


def _ensure_fold_state(state: dict[str, Any], fold_spec: dict[str, object]) -> dict[str, Any]:
    fold_name = str(fold_spec["name"])
    folds = state.setdefault("folds", {})
    if fold_name not in folds:
        folds[fold_name] = {
            "fold_name": fold_name,
            "validation_size": int(fold_spec["validation_size"]),
            "test_size": int(fold_spec["test_size"]),
            "validation_dynamic_candidates": {},
            "validation_static_candidates": {},
            "selected_dynamic_bucket": None,
            "selected_static_layer": None,
            "held_out_test": None,
            "completed_stages": [],
            "updated_at": _utc_now(),
        }
    return folds[fold_name]


def _write_fold_partial(output_dir: Path, fold_state: dict[str, Any]) -> None:
    payload = dict(fold_state)
    payload["updated_at"] = _utc_now()
    _atomic_write_json(_fold_partial_path(output_dir, str(fold_state["fold_name"])), payload)


def _write_legacy_partial(output_dir: Path, state: dict[str, Any]) -> None:
    folds = [state["folds"][name] for name in sorted(state.get("folds", {})) if state["folds"][name].get("held_out_test")]
    payload = {
        "task_name": state["task_name"],
        "model_name": state["model_name"],
        "csv_path": state["csv_path"],
        "split": state["split"],
        "prompt_style": state["prompt_style"],
        "score_mode": state["score_mode"],
        "mature_layer": state["mature_layer"],
        "dynamic_bucket_candidates": state["dynamic_bucket_candidates"],
        "static_layer_candidates": state["static_layer_candidates"],
        "transfer_reference": state.get("transfer_reference"),
        "folds_completed": len(folds),
        "folds": folds,
    }
    _atomic_write_json(output_dir / LEGACY_PARTIAL_FILE, payload)


def _persist_state(output_dir: Path, state: dict[str, Any], *, fold_name: str | None = None) -> None:
    state["updated_at"] = _utc_now()
    _atomic_write_json(output_dir / STATE_FILE, state)
    if fold_name is not None:
        _write_fold_partial(output_dir, state["folds"][fold_name])
    _write_legacy_partial(output_dir, state)


def _migrate_legacy_partial(
    output_dir: Path,
    state: dict[str, Any],
    fold_specs: list[dict[str, object]],
    dynamic_buckets: list[dict[str, object]],
    static_layers: list[int],
    logger: RunLogger,
) -> dict[str, Any]:
    legacy = _load_json_if_exists(output_dir / LEGACY_PARTIAL_FILE)
    if legacy is None:
        return state
    if state.get("folds"):
        return state

    fold_lookup = {str(spec["name"]): spec for spec in fold_specs}
    completed = set(str(item) for item in state.get("completed_stages", []))
    for fold_report in legacy.get("folds", []):
        fold_name = str(fold_report["fold_name"])
        fold_spec = fold_lookup.get(fold_name)
        if fold_spec is None:
            continue
        fold_state = _ensure_fold_state(state, fold_spec)
        fold_state["selected_dynamic_bucket"] = fold_report.get("selected_dynamic_bucket")
        fold_state["selected_static_layer"] = fold_report.get("selected_static_layer")
        fold_state["held_out_test"] = fold_report.get("held_out_test")
        fold_state["updated_at"] = _utc_now()
        for bucket in dynamic_buckets:
            completed.add(f"{fold_name}.validation.dynamic.{bucket['name']}")
        for layer in static_layers:
            completed.add(f"{fold_name}.validation.static.{layer}")
        completed.add(f"{fold_name}.selection")
        completed.add(f"{fold_name}.test.dynamic")
        completed.add(f"{fold_name}.test.vanilla")
        completed.add(f"{fold_name}.test.static")
        completed.add(f"{fold_name}.partial")
        _write_fold_partial(output_dir, fold_state)

    if completed:
        state["completed_stages"] = sorted(completed)
        state["updated_at"] = _utc_now()
        _atomic_write_json(output_dir / STATE_FILE, state)
        logger.log(
            "Migrated legacy truthfulqa_tuned_bucket_cv_partial.json into resume-safe cv_state.json."
        )
    return state


def _collect_fold_reports(state: dict[str, Any]) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for fold_name in sorted(state.get("folds", {})):
        fold_state = state["folds"][fold_name]
        if not fold_state.get("held_out_test"):
            continue
        reports.append(
            {
                "fold_name": fold_name,
                "validation_size": int(fold_state["validation_size"]),
                "test_size": int(fold_state["test_size"]),
                "selected_dynamic_bucket": fold_state["selected_dynamic_bucket"],
                "selected_static_layer": fold_state["selected_static_layer"],
                "held_out_test": fold_state["held_out_test"],
            }
        )
    return reports


def _build_summary_payload(state: dict[str, Any]) -> dict[str, Any]:
    fold_reports = _collect_fold_reports(state)
    if len(fold_reports) != 2:
        raise ValueError("Cannot build final summary yet: both fold partials are not complete.")
    cv_average = _average_fold_results([fold["held_out_test"] for fold in fold_reports])
    final_label = _classify(cv_average)
    return {
        "task_name": state["task_name"],
        "model_name": state["model_name"],
        "csv_path": state["csv_path"],
        "split": state["split"],
        "prompt_style": state["prompt_style"],
        "score_mode": state["score_mode"],
        "mature_layer": state["mature_layer"],
        "dynamic_bucket_candidates": state["dynamic_bucket_candidates"],
        "static_layer_candidates": state["static_layer_candidates"],
        "transfer_reference": state.get("transfer_reference"),
        "folds": fold_reports,
        "cv_average": cv_average,
        "final_label": final_label,
        "run_signature": state["run_signature"],
    }


def _state_from_legacy_partial(legacy: dict[str, Any], run_signature: str) -> dict[str, Any]:
    folds = {str(fold["fold_name"]): fold for fold in legacy.get("folds", [])}
    return {
        "schema_version": 2,
        "run_signature": run_signature,
        "task_name": legacy.get("task_name", "truthfulqa_tuned_bucket_cv"),
        "model_name": legacy["model_name"],
        "csv_path": legacy["csv_path"],
        "subset_mode": "full_dataset",
        "prompt_style": legacy.get("prompt_style", "official_tfqa_mc"),
        "score_mode": legacy.get("score_mode", "sum_logprob"),
        "mature_layer": legacy["mature_layer"],
        "split_seed": legacy.get("split", {}).get("seed"),
        "split": legacy["split"],
        "dynamic_bucket_candidates": legacy.get("dynamic_bucket_candidates", []),
        "static_layer_candidates": legacy.get("static_layer_candidates", []),
        "transfer_reference": legacy.get("transfer_reference"),
        "completed_stages": ["final.aggregate_summary"],
        "stage_durations_seconds": {"final.aggregate_summary": 0.0},
        "folds": folds,
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
    }


def _write_final_summary(output_dir: Path, state: dict[str, Any], logger: RunLogger) -> dict[str, Any]:
    payload = _build_summary_payload(state)
    _atomic_write_json(output_dir / SUMMARY_FILE, payload)
    _mark_stage_complete(state, stage_id="final.aggregate_summary", duration_seconds=0.0)
    _persist_state(output_dir, state)
    logger.log(f"final_label={payload['final_label']}")
    logger.log(f"Saved summary to: {output_dir / SUMMARY_FILE}")
    return payload

def _current_best_mc3(candidates: dict[str, Any]) -> float | None:
    if not candidates:
        return None
    return max(float(item["summary"]["dola_avg_mc3"]) for item in candidates.values())


def _run_validation_dynamic_stage(
    *,
    stage: dict[str, object],
    tracker: ProgressTracker,
    logger: RunLogger,
    state: dict[str, Any],
    fold_state: dict[str, Any],
    output_dir: Path,
    model: Any,
    tokenizer: Any,
    validation_sample_ids: list[int],
    validation_samples: list[TruthfulQASample],
    validation_cache: dict[int, dict[str, Any]],
    bucket: dict[str, object],
    all_dynamic_buckets: list[dict[str, object]],
    static_layers: list[int],
    mature_layer: int,
    prompt_style: str,
    score_mode: str,
    post_softmax: bool,
    relative_top: float,
    relative_top_value: float,
    candidate_batch_size: int,
) -> dict[str, Any]:
    bucket_name = str(bucket["name"])
    existing = fold_state["validation_dynamic_candidates"].get(bucket_name)
    if str(stage["id"]) in state["completed_stages"]:
        if existing is None:
            raise ValueError(f"Completed stage {stage['id']} is missing its saved result.")
        logger.log(
            f"Skipping completed stage: {stage['label']} | saved MC3={existing['summary']['dola_avg_mc3']:.4f}"
        )
        return existing

    stage_for_run = dict(stage)
    if not validation_cache:
        total_dynamic_layers = sum(len(list(item["candidate_premature_layers"])) for item in all_dynamic_buckets)
        stage_for_run["label"] = (
            f"{fold_state['fold_name']} validation cache warmup via dynamic {bucket_name}"
        )
        stage_for_run["layer_count"] = max(total_dynamic_layers + len(static_layers), 1)
        stage_for_run["weight"] = _stage_weight("dynamic", len(validation_samples), int(stage_for_run["layer_count"]))
        logger.log(
            f"{fold_state['fold_name']} cache warmup note: this first validation dynamic stage computes all "
            f"{len(all_dynamic_buckets)} dynamic buckets and {len(static_layers)} static layers for the fold; "
            "early ETA is conservative until cache misses stabilize."
        )
    tracker.start_stage(stage_for_run)
    callback = tracker.make_callback()
    vanilla_metric_rows: list[dict[str, float]] = []
    dola_metric_rows: list[dict[str, float]] = []
    layer_usage: dict[int, int] = {}
    cache_hits = 0
    cache_misses = 0
    total_answers = 0
    total_forward_batches = 0
    dynamic_bucket_map = {
        str(item["name"]): list(item["candidate_premature_layers"])
        for item in all_dynamic_buckets
    }

    for sample_position, (sample_id, sample) in enumerate(zip(validation_sample_ids, validation_samples)):
        bundle, from_cache = _get_or_build_sample_bundle(
            sample_cache=validation_cache,
            sample_id=sample_id,
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
        if from_cache:
            cache_hits += 1
        else:
            cache_misses += 1
            total_answers += int(bundle["profiling"]["answer_count"])
            total_forward_batches += int(bundle["profiling"]["batch_count"])

        vanilla_metric_rows.append(bundle["vanilla_metrics"])
        dola_metric_rows.append(bundle["dynamic_metrics"][bucket_name])
        sample_usage = bundle["dynamic_layer_usage"].get(bucket_name) or {}
        for layer, count in sample_usage.items():
            layer_usage[int(layer)] = layer_usage.get(int(layer), 0) + int(count)
        callback(
            {
                "completed_samples": sample_position + 1,
                "total_samples": len(validation_samples),
                "sample_index": sample_position,
                "question": sample.question,
            }
        )

    duration = tracker.finish_stage()
    result = {
        "name": bucket_name,
        "candidate_premature_layers": list(bucket["candidate_premature_layers"]),
        "summary": _build_candidate_summary(
            vanilla_metric_rows=vanilla_metric_rows,
            dola_metric_rows=dola_metric_rows,
            prompt_style=prompt_style,
            score_mode=score_mode,
            dola_score_mode="official_dynamic_dola",
            post_softmax=post_softmax,
            relative_top=relative_top,
            relative_top_value=relative_top_value,
            premature_layer=int(bucket["candidate_premature_layers"][0]),
            candidate_premature_layers=list(bucket["candidate_premature_layers"]),
            mature_layer=mature_layer,
            layer_usage=layer_usage,
        ),
        "profiling": {
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "cache_size": len(validation_cache),
            "avg_answers_per_miss": (total_answers / cache_misses) if cache_misses else 0.0,
            "avg_seconds_per_sample": duration / len(validation_samples),
            "avg_seconds_per_answer": (duration / total_answers) if total_answers else 0.0,
            "candidate_batch_size": candidate_batch_size,
            "forward_batches": total_forward_batches,
        },
    }
    fold_state["validation_dynamic_candidates"][bucket_name] = result
    _mark_stage_complete(state, stage_id=str(stage["id"]), duration_seconds=duration)
    fold_state["updated_at"] = _utc_now()
    _persist_state(output_dir, state, fold_name=str(fold_state["fold_name"]))
    logger.log(
        f"{fold_state['fold_name']} validation dynamic candidate {bucket_name} -> MC3={result['summary']['dola_avg_mc3']:.4f}"
    )
    logger.log(
        f"{fold_state['fold_name']} validation dynamic {bucket_name} profiling: "
        f"cache_hits={cache_hits}, cache_misses={cache_misses}, forward_batches={total_forward_batches}, "
        f"candidate_batch_size={candidate_batch_size}, avg_sample_s={result['profiling']['avg_seconds_per_sample']:.2f}, "
        f"avg_answer_s={result['profiling']['avg_seconds_per_answer']:.4f}"
    )
    best_so_far = _current_best_mc3(fold_state["validation_dynamic_candidates"])
    if best_so_far is not None:
        logger.log(f"{fold_state['fold_name']} current best validation dynamic MC3={best_so_far:.4f}")
    return result


def _run_validation_static_stage(
    *,
    stage: dict[str, object],
    tracker: ProgressTracker,
    logger: RunLogger,
    state: dict[str, Any],
    fold_state: dict[str, Any],
    output_dir: Path,
    model: Any,
    tokenizer: Any,
    validation_sample_ids: list[int],
    validation_samples: list[TruthfulQASample],
    validation_cache: dict[int, dict[str, Any]],
    layer: int,
    all_dynamic_buckets: list[dict[str, object]],
    static_layers: list[int],
    mature_layer: int,
    prompt_style: str,
    score_mode: str,
    post_softmax: bool,
    relative_top: float,
    relative_top_value: float,
    candidate_batch_size: int,
) -> dict[str, Any]:
    layer_key = str(layer)
    existing = fold_state["validation_static_candidates"].get(layer_key)
    if str(stage["id"]) in state["completed_stages"]:
        if existing is None:
            raise ValueError(f"Completed stage {stage['id']} is missing its saved result.")
        logger.log(
            f"Skipping completed stage: {stage['label']} | saved MC3={existing['summary']['dola_avg_mc3']:.4f}"
        )
        return existing

    tracker.start_stage(stage)
    callback = tracker.make_callback()
    vanilla_metric_rows: list[dict[str, float]] = []
    dola_metric_rows: list[dict[str, float]] = []
    layer_usage: dict[int, int] = {}
    cache_hits = 0
    cache_misses = 0
    total_answers = 0
    total_forward_batches = 0
    dynamic_bucket_map = {
        str(item["name"]): list(item["candidate_premature_layers"])
        for item in all_dynamic_buckets
    }

    for sample_position, (sample_id, sample) in enumerate(zip(validation_sample_ids, validation_samples)):
        bundle, from_cache = _get_or_build_sample_bundle(
            sample_cache=validation_cache,
            sample_id=sample_id,
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
        if from_cache:
            cache_hits += 1
        else:
            cache_misses += 1
            total_answers += int(bundle["profiling"]["answer_count"])
            total_forward_batches += int(bundle["profiling"]["batch_count"])

        vanilla_metric_rows.append(bundle["vanilla_metrics"])
        dola_metric_rows.append(bundle["static_metrics"][layer_key])
        sample_usage = bundle["static_layer_usage"].get(layer_key) or {}
        for used_layer, count in sample_usage.items():
            layer_usage[int(used_layer)] = layer_usage.get(int(used_layer), 0) + int(count)
        callback(
            {
                "completed_samples": sample_position + 1,
                "total_samples": len(validation_samples),
                "sample_index": sample_position,
                "question": sample.question,
            }
        )

    duration = tracker.finish_stage()
    result = {
        "premature_layer": layer,
        "summary": _build_candidate_summary(
            vanilla_metric_rows=vanilla_metric_rows,
            dola_metric_rows=dola_metric_rows,
            prompt_style=prompt_style,
            score_mode=score_mode,
            dola_score_mode="official_static_dola",
            post_softmax=post_softmax,
            relative_top=relative_top,
            relative_top_value=relative_top_value,
            premature_layer=layer,
            candidate_premature_layers=None,
            mature_layer=mature_layer,
            layer_usage=layer_usage,
        ),
        "profiling": {
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "cache_size": len(validation_cache),
            "avg_answers_per_miss": (total_answers / cache_misses) if cache_misses else 0.0,
            "avg_seconds_per_sample": duration / len(validation_samples),
            "avg_seconds_per_answer": (duration / total_answers) if total_answers else 0.0,
            "candidate_batch_size": candidate_batch_size,
            "forward_batches": total_forward_batches,
        },
    }
    fold_state["validation_static_candidates"][layer_key] = result
    _mark_stage_complete(state, stage_id=str(stage["id"]), duration_seconds=duration)
    fold_state["updated_at"] = _utc_now()
    _persist_state(output_dir, state, fold_name=str(fold_state["fold_name"]))
    logger.log(
        f"{fold_state['fold_name']} validation static candidate {layer} -> MC3={result['summary']['dola_avg_mc3']:.4f}"
    )
    logger.log(
        f"{fold_state['fold_name']} validation static {layer} profiling: "
        f"cache_hits={cache_hits}, cache_misses={cache_misses}, forward_batches={total_forward_batches}, "
        f"candidate_batch_size={candidate_batch_size}, avg_sample_s={result['profiling']['avg_seconds_per_sample']:.2f}, "
        f"avg_answer_s={result['profiling']['avg_seconds_per_answer']:.4f}"
    )
    best_so_far = _current_best_mc3(fold_state["validation_static_candidates"])
    if best_so_far is not None:
        logger.log(f"{fold_state['fold_name']} current best validation static MC3={best_so_far:.4f}")
    return result


def _run_validation_shared_stage(
    *,
    stage: dict[str, object],
    tracker: ProgressTracker,
    logger: RunLogger,
    state: dict[str, Any],
    fold_state: dict[str, Any],
    output_dir: Path,
    model: Any,
    tokenizer: Any,
    validation_sample_ids: list[int],
    validation_samples: list[TruthfulQASample],
    dynamic_buckets: list[dict[str, object]],
    static_layers: list[int],
    mature_layer: int,
    prompt_style: str,
    score_mode: str,
    post_softmax: bool,
    relative_top: float,
    relative_top_value: float,
    candidate_batch_size: int,
    stage_profile_limit: int = 0,
    stage_profile_path: Path | None = None,
    stage_profile_summary_path: Path | None = None,
    sample_profile_path: Path | None = None,
) -> dict[str, Any]:
    if str(stage["id"]) in state["completed_stages"]:
        if fold_state["validation_dynamic_candidates"] and fold_state["validation_static_candidates"]:
            logger.log(f"Skipping completed stage: {stage['label']}")
            return {"num_samples_completed": len(validation_samples), "sample_profile_summary": None, "stage_profile_summary": None}
        raise ValueError(f"Completed stage {stage['id']} is missing saved validation results.")

    tracker.start_stage(stage)
    callback = tracker.make_callback()
    dynamic_metric_rows = {str(bucket["name"]): [] for bucket in dynamic_buckets}
    dynamic_layer_usage = {str(bucket["name"]): {} for bucket in dynamic_buckets}
    static_metric_rows = {str(layer): [] for layer in static_layers}
    static_layer_usage = {str(layer): {} for layer in static_layers}
    vanilla_metric_rows: list[dict[str, float]] = []
    total_answers = 0
    total_forward_batches = 0
    dynamic_bucket_map = {str(item["name"]): list(item["candidate_premature_layers"]) for item in dynamic_buckets}
    stage_profile_records: list[dict[str, Any]] = []
    sample_profile_records: list[dict[str, Any]] = []
    profile_limit = min(max(int(stage_profile_limit), 0), len(validation_samples))
    union_layer_count = len({*static_layers, *(layer for layers in dynamic_bucket_map.values() for layer in layers)})

    logger.log(f"{fold_state['fold_name']} online reuse note: each validation sample is scored once across the union of {len(dynamic_buckets)} dynamic buckets and {len(static_layers)} static layers; no fold-wide logits cache is kept.")

    for sample_position, sample in enumerate(validation_samples):
        inner_score_start = time.perf_counter()
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
        inner_score_sample_sec = time.perf_counter() - inner_score_start
        consume_stats = _consume_validation_shared_bundle(
            bundle=bundle,
            dynamic_buckets=dynamic_buckets,
            static_layers=static_layers,
            vanilla_metric_rows=vanilla_metric_rows,
            dynamic_metric_rows=dynamic_metric_rows,
            dynamic_layer_usage=dynamic_layer_usage,
            static_metric_rows=static_metric_rows,
            static_layer_usage=static_layer_usage,
        )
        total_answers += int(consume_stats["answer_count"])
        total_forward_batches += int(consume_stats["batch_count"])

        progress_callback_start = time.perf_counter()
        callback({"completed_samples": sample_position + 1, "total_samples": len(validation_samples), "sample_index": sample_position, "question": sample.question})
        progress_callback_sec = time.perf_counter() - progress_callback_start

        if sample_position < profile_limit:
            stage_profile_records.append(_build_stage_sample_profile_record(sample_index=sample_position, sample=sample, bundle=bundle, inner_score_sample_sec=inner_score_sample_sec, metrics_aggregation_sec=float(consume_stats["metrics_aggregation_sec"]), usage_merge_sec=float(consume_stats["usage_merge_sec"]), progress_callback_sec=progress_callback_sec))
            if sample_profile_path is not None:
                sample_profile_records.append(_build_inner_sample_profile_record(sample_index=sample_position, sample=sample, profiling=bundle["profiling"], union_layer_count=union_layer_count))

    duration = tracker.finish_stage()
    avg_sample = duration / len(validation_samples)
    avg_answer = (duration / total_answers) if total_answers else 0.0
    fold_state["validation_dynamic_candidates"] = {}
    for bucket in dynamic_buckets:
        bucket_name = str(bucket["name"])
        fold_state["validation_dynamic_candidates"][bucket_name] = {
            "name": bucket_name,
            "candidate_premature_layers": list(bucket["candidate_premature_layers"]),
            "summary": _build_candidate_summary(vanilla_metric_rows=vanilla_metric_rows, dola_metric_rows=dynamic_metric_rows[bucket_name], prompt_style=prompt_style, score_mode=score_mode, dola_score_mode="official_dynamic_dola", post_softmax=post_softmax, relative_top=relative_top, relative_top_value=relative_top_value, premature_layer=int(bucket["candidate_premature_layers"][0]), candidate_premature_layers=list(bucket["candidate_premature_layers"]), mature_layer=mature_layer, layer_usage=dynamic_layer_usage[bucket_name]),
            "profiling": {"online_reuse_scope": "sample_union_layers", "cache_hits": 0, "cache_misses": len(validation_samples), "cache_size": 0, "avg_answers_per_miss": (total_answers / len(validation_samples)) if validation_samples else 0.0, "avg_seconds_per_sample": avg_sample, "avg_seconds_per_answer": avg_answer, "candidate_batch_size": candidate_batch_size, "forward_batches": total_forward_batches},
        }

    fold_state["validation_static_candidates"] = {}
    for layer in static_layers:
        layer_key = str(layer)
        fold_state["validation_static_candidates"][layer_key] = {
            "premature_layer": layer,
            "summary": _build_candidate_summary(vanilla_metric_rows=vanilla_metric_rows, dola_metric_rows=static_metric_rows[layer_key], prompt_style=prompt_style, score_mode=score_mode, dola_score_mode="official_static_dola", post_softmax=post_softmax, relative_top=relative_top, relative_top_value=relative_top_value, premature_layer=layer, candidate_premature_layers=None, mature_layer=mature_layer, layer_usage=static_layer_usage[layer_key]),
            "profiling": {"online_reuse_scope": "sample_union_layers", "cache_hits": 0, "cache_misses": len(validation_samples), "cache_size": 0, "avg_answers_per_miss": (total_answers / len(validation_samples)) if validation_samples else 0.0, "avg_seconds_per_sample": avg_sample, "avg_seconds_per_answer": avg_answer, "candidate_batch_size": candidate_batch_size, "forward_batches": total_forward_batches},
        }

    _mark_stage_complete(state, stage_id=str(stage["id"]), duration_seconds=duration)
    fold_state["updated_at"] = _utc_now()
    persist_start = time.perf_counter()
    _persist_state(output_dir, state, fold_name=str(fold_state["fold_name"]))
    state_persist_sec = time.perf_counter() - persist_start
    if stage_profile_records:
        stage_profile_records[-1]["state_persist_sec"] = float(state_persist_sec)
        stage_profile_records[-1]["total_stage_sample_sec"] = float(stage_profile_records[-1]["total_stage_sample_sec"] + state_persist_sec)

    sample_profile_summary = None
    if sample_profile_path is not None and sample_profile_records:
        _write_jsonl_records(sample_profile_path, sample_profile_records)
        sample_profile_summary = _summarize_inner_profile_records(sample_profile_records, batch_size=candidate_batch_size, sample_profile_path=sample_profile_path)

    stage_profile_summary = None
    if stage_profile_path is not None and stage_profile_records:
        _write_jsonl_records(stage_profile_path, stage_profile_records)
        stage_profile_summary = _summarize_stage_profile_records(stage_profile_records, batch_size=candidate_batch_size, stage_profile_path=stage_profile_path)
        if stage_profile_summary_path is not None:
            _atomic_write_json(stage_profile_summary_path, stage_profile_summary)

    best_dynamic = _current_best_mc3(fold_state["validation_dynamic_candidates"])
    best_static = _current_best_mc3(fold_state["validation_static_candidates"])
    logger.log(f"{fold_state['fold_name']} validation shared profiling: forward_batches={total_forward_batches}, candidate_batch_size={candidate_batch_size}, avg_sample_s={avg_sample:.2f}, avg_answer_s={avg_answer:.4f}")
    if stage_profile_summary is not None:
        logger.log(f"{fold_state['fold_name']} stage-profile avg_inner_s={stage_profile_summary['avg_inner_score_sample_sec']:.2f}, metrics_s={stage_profile_summary['avg_metrics_aggregation_sec']:.4f}, usage_s={stage_profile_summary['avg_usage_merge_sec']:.4f}, callback_s={stage_profile_summary['avg_progress_callback_sec']:.4f}, persist_s={stage_profile_summary['avg_state_persist_sec']:.4f}")
    if best_dynamic is not None:
        logger.log(f"{fold_state['fold_name']} current best validation dynamic MC3={best_dynamic:.4f}")
    if best_static is not None:
        logger.log(f"{fold_state['fold_name']} current best validation static MC3={best_static:.4f}")
    return {"num_samples_completed": len(validation_samples), "sample_profile_summary": sample_profile_summary, "stage_profile_summary": stage_profile_summary}



def _run_test_shared_stage(
    *,
    stage: dict[str, object],
    tracker: ProgressTracker,
    logger: RunLogger,
    state: dict[str, Any],
    fold_state: dict[str, Any],
    output_dir: Path,
    model: Any,
    tokenizer: Any,
    test_samples: list[TruthfulQASample],
    prompt_style: str,
    score_mode: str,
    post_softmax: bool,
    relative_top: float,
    relative_top_value: float,
    mature_layer: int,
    candidate_batch_size: int,
) -> None:
    held_out = fold_state.get("held_out_test")
    if str(stage["id"]) in state["completed_stages"]:
        if held_out and {"vanilla", "tuned_dynamic", "tuned_static"}.issubset(held_out):
            logger.log(f"Skipping completed stage: {stage['label']}")
            return
        raise ValueError(f"Completed stage {stage['id']} is missing saved held-out results.")

    selected_dynamic = fold_state.get("selected_dynamic_bucket")
    selected_static = fold_state.get("selected_static_layer")
    if selected_dynamic is None or selected_static is None:
        raise ValueError(f"Cannot run shared held-out test for {fold_state['fold_name']}: selection is incomplete.")

    stage_for_run = dict(stage)
    selected_union_count = len(set(list(selected_dynamic["candidate_premature_layers"]) + [int(selected_static["premature_layer"])]))
    stage_for_run["label"] = f"{fold_state['fold_name']} held-out shared scoring"
    stage_for_run["layer_count"] = max(selected_union_count, 1)
    stage_for_run["weight"] = _stage_weight("shared", len(test_samples), int(stage_for_run["layer_count"]))
    tracker.start_stage(stage_for_run)
    callback = tracker.make_callback()

    selected_dynamic_name = str(selected_dynamic["name"])
    dynamic_bucket_map = {selected_dynamic_name: list(selected_dynamic["candidate_premature_layers"])}
    static_layers = [int(selected_static["premature_layer"])]
    vanilla_metric_rows: list[dict[str, float]] = []
    dynamic_metric_rows: list[dict[str, float]] = []
    static_metric_rows: list[dict[str, float]] = []
    total_answers = 0
    total_forward_batches = 0

    logger.log(
        f"{fold_state['fold_name']} online reuse note: each held-out sample is scored once for vanilla, tuned dynamic, and tuned static; no test-cache is kept."
    )

    for sample_position, sample in enumerate(test_samples):
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
        dynamic_metric_rows.append(bundle["dynamic_metrics"][selected_dynamic_name])
        static_metric_rows.append(bundle["static_metrics"][str(selected_static["premature_layer"])])
        total_answers += int(bundle["profiling"]["answer_count"])
        total_forward_batches += int(bundle["profiling"]["batch_count"])
        callback(
            {
                "completed_samples": sample_position + 1,
                "total_samples": len(test_samples),
                "sample_index": sample_position,
                "question": sample.question,
            }
        )

    duration = tracker.finish_stage()
    dynamic_summary = _build_candidate_summary(
        vanilla_metric_rows=vanilla_metric_rows,
        dola_metric_rows=dynamic_metric_rows,
        prompt_style=prompt_style,
        score_mode=score_mode,
        dola_score_mode="official_dynamic_dola",
        post_softmax=post_softmax,
        relative_top=relative_top,
        relative_top_value=relative_top_value,
        premature_layer=int(selected_dynamic["candidate_premature_layers"][0]),
        candidate_premature_layers=list(selected_dynamic["candidate_premature_layers"]),
        mature_layer=mature_layer,
        layer_usage=None,
    )
    static_summary = aggregate_mc_metrics(static_metric_rows)
    held_out = held_out or {}
    held_out["vanilla"] = {
        "MC1": float(dynamic_summary["vanilla_avg_mc1"]),
        "MC2": float(dynamic_summary["vanilla_avg_mc2"]),
        "MC3": float(dynamic_summary["vanilla_avg_mc3"]),
    }
    held_out["tuned_dynamic"] = {
        "MC1": float(dynamic_summary["dola_avg_mc1"]),
        "MC2": float(dynamic_summary["dola_avg_mc2"]),
        "MC3": float(dynamic_summary["dola_avg_mc3"]),
    }
    held_out["tuned_static"] = {
        "MC1": float(static_summary["avg_mc1"]),
        "MC2": float(static_summary["avg_mc2"]),
        "MC3": float(static_summary["avg_mc3"]),
    }
    held_out["num_samples"] = int(dynamic_summary["num_samples"])
    fold_state["held_out_test"] = held_out
    _mark_stage_complete(state, stage_id=str(stage["id"]), duration_seconds=duration)
    _mark_stage_complete(state, stage_id=f"{fold_state['fold_name']}.test.vanilla", duration_seconds=0.0)
    fold_state["updated_at"] = _utc_now()
    _persist_state(output_dir, state, fold_name=str(fold_state["fold_name"]))
    logger.log(
        f"{fold_state['fold_name']} held-out shared profiling: forward_batches={total_forward_batches}, candidate_batch_size={candidate_batch_size}, "
        f"avg_sample_s={duration / len(test_samples):.2f}, avg_answer_s={(duration / total_answers) if total_answers else 0.0:.4f}"
    )


def _complete_selection_stage(
    *,
    logger: RunLogger,
    state: dict[str, Any],
    fold_state: dict[str, Any],
    output_dir: Path,
) -> None:
    stage_id = f"{fold_state['fold_name']}.selection"
    if stage_id in state["completed_stages"]:
        logger.log(
            f"Skipping completed stage: {fold_state['fold_name']} selection | dynamic={fold_state['selected_dynamic_bucket']['name']} | static={fold_state['selected_static_layer']['premature_layer']}"
        )
        return

    dynamic_candidates = list(fold_state["validation_dynamic_candidates"].values())
    static_candidates = list(fold_state["validation_static_candidates"].values())
    if not dynamic_candidates or not static_candidates:
        raise ValueError(f"Cannot select best bucket for {fold_state['fold_name']}: validation results incomplete.")

    selected_dynamic = _select_best_dynamic(dynamic_candidates)
    selected_static = _select_best_static(static_candidates)
    fold_state["selected_dynamic_bucket"] = selected_dynamic
    fold_state["selected_static_layer"] = selected_static
    _mark_stage_complete(state, stage_id=stage_id, duration_seconds=0.0)
    fold_state["updated_at"] = _utc_now()
    _persist_state(output_dir, state, fold_name=str(fold_state["fold_name"]))
    logger.log(
        f"{fold_state['fold_name']} selected dynamic={selected_dynamic['name']} official_layers={selected_dynamic['candidate_premature_layers']} | selected static={selected_static['premature_layer']}"
    )


def _run_test_dynamic_stage(
    *,
    stage: dict[str, object],
    tracker: ProgressTracker,
    logger: RunLogger,
    state: dict[str, Any],
    fold_state: dict[str, Any],
    output_dir: Path,
    model: Any,
    tokenizer: Any,
    test_sample_ids: list[int],
    test_samples: list[TruthfulQASample],
    test_cache: dict[int, dict[str, Any]],
    prompt_style: str,
    score_mode: str,
    post_softmax: bool,
    relative_top: float,
    relative_top_value: float,
    mature_layer: int,
    candidate_batch_size: int,
) -> None:
    held_out = fold_state.get("held_out_test")
    if str(stage["id"]) in state["completed_stages"]:
        if not held_out or "tuned_dynamic" not in held_out or "vanilla" not in held_out:
            raise ValueError(f"Completed stage {stage['id']} is missing its saved held-out result.")
        logger.log(f"Skipping completed stage: {stage['label']}")
        return

    selected_dynamic = fold_state.get("selected_dynamic_bucket")
    selected_static = fold_state.get("selected_static_layer")
    if selected_dynamic is None:
        raise ValueError(f"Cannot run test dynamic for {fold_state['fold_name']}: no selected dynamic bucket.")
    if selected_static is None:
        raise ValueError(f"Cannot run test dynamic for {fold_state['fold_name']}: no selected static layer.")

    stage_for_run = dict(stage)
    stage_for_run["label"] = f"{fold_state['fold_name']} test cache warmup via tuned dynamic {selected_dynamic['name']}"
    effective_layer_factor = len(selected_dynamic["candidate_premature_layers"]) + 1
    stage_for_run["layer_count"] = effective_layer_factor
    stage_for_run["weight"] = _stage_weight("dynamic", len(test_samples), effective_layer_factor)
    if not test_cache:
        logger.log(
            f"{fold_state['fold_name']} cache warmup note: this test dynamic stage also prepares held-out vanilla and tuned static reuse; "
            "early ETA is conservative until cache misses stabilize."
        )
    tracker.start_stage(stage_for_run)
    callback = tracker.make_callback()
    vanilla_metric_rows: list[dict[str, float]] = []
    dynamic_metric_rows: list[dict[str, float]] = []
    cache_hits = 0
    cache_misses = 0
    total_answers = 0
    total_forward_batches = 0
    selected_dynamic_name = str(selected_dynamic["name"])
    dynamic_bucket_map = {selected_dynamic_name: list(selected_dynamic["candidate_premature_layers"])}
    static_layers = [int(selected_static["premature_layer"])]

    for sample_position, (sample_id, sample) in enumerate(zip(test_sample_ids, test_samples)):
        bundle, from_cache = _get_or_build_sample_bundle(
            sample_cache=test_cache,
            sample_id=sample_id,
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
        if from_cache:
            cache_hits += 1
        else:
            cache_misses += 1
            total_answers += int(bundle["profiling"]["answer_count"])
            total_forward_batches += int(bundle["profiling"]["batch_count"])

        vanilla_metric_rows.append(bundle["vanilla_metrics"])
        dynamic_metric_rows.append(bundle["dynamic_metrics"][selected_dynamic_name])
        callback(
            {
                "completed_samples": sample_position + 1,
                "total_samples": len(test_samples),
                "sample_index": sample_position,
                "question": sample.question,
            }
        )

    duration = tracker.finish_stage()
    dynamic_summary = _build_candidate_summary(
        vanilla_metric_rows=vanilla_metric_rows,
        dola_metric_rows=dynamic_metric_rows,
        prompt_style=prompt_style,
        score_mode=score_mode,
        dola_score_mode="official_dynamic_dola",
        post_softmax=post_softmax,
        relative_top=relative_top,
        relative_top_value=relative_top_value,
        premature_layer=int(selected_dynamic["candidate_premature_layers"][0]),
        candidate_premature_layers=list(selected_dynamic["candidate_premature_layers"]),
        mature_layer=mature_layer,
        layer_usage=None,
    )
    held_out = held_out or {}
    held_out["vanilla"] = {
        "MC1": float(dynamic_summary["vanilla_avg_mc1"]),
        "MC2": float(dynamic_summary["vanilla_avg_mc2"]),
        "MC3": float(dynamic_summary["vanilla_avg_mc3"]),
    }
    held_out["tuned_dynamic"] = {
        "MC1": float(dynamic_summary["dola_avg_mc1"]),
        "MC2": float(dynamic_summary["dola_avg_mc2"]),
        "MC3": float(dynamic_summary["dola_avg_mc3"]),
    }
    held_out["num_samples"] = int(dynamic_summary["num_samples"])
    fold_state["held_out_test"] = held_out
    _mark_stage_complete(state, stage_id=str(stage["id"]), duration_seconds=duration)
    _mark_stage_complete(state, stage_id=f"{fold_state['fold_name']}.test.vanilla", duration_seconds=0.0)
    fold_state["updated_at"] = _utc_now()
    _persist_state(output_dir, state, fold_name=str(fold_state["fold_name"]))
    logger.log(
        f"{fold_state['fold_name']} test tuned dynamic profiling: cache_hits={cache_hits}, cache_misses={cache_misses}, "
        f"forward_batches={total_forward_batches}, candidate_batch_size={candidate_batch_size}, "
        f"avg_sample_s={duration / len(test_samples):.2f}, avg_answer_s={(duration / total_answers) if total_answers else 0.0:.4f}"
    )


def _run_test_static_stage(
    *,
    stage: dict[str, object],
    tracker: ProgressTracker,
    logger: RunLogger,
    state: dict[str, Any],
    fold_state: dict[str, Any],
    output_dir: Path,
    test_sample_ids: list[int],
    test_samples: list[TruthfulQASample],
    test_cache: dict[int, dict[str, Any]],
) -> None:
    held_out = fold_state.get("held_out_test")
    if str(stage["id"]) in state["completed_stages"]:
        if not held_out or "tuned_static" not in held_out:
            raise ValueError(f"Completed stage {stage['id']} is missing its saved held-out result.")
        logger.log(f"Skipping completed stage: {stage['label']}")
        return

    selected_static = fold_state.get("selected_static_layer")
    if selected_static is None:
        raise ValueError(f"Cannot run test static for {fold_state['fold_name']}: no selected static layer.")

    stage_for_run = dict(stage)
    stage_for_run["label"] = f"{fold_state['fold_name']} test tuned static layer {selected_static['premature_layer']}"
    tracker.start_stage(stage_for_run)
    callback = tracker.make_callback()
    static_metric_rows: list[dict[str, float]] = []
    cache_hits = 0
    for sample_position, (sample_id, sample) in enumerate(zip(test_sample_ids, test_samples)):
        bundle = test_cache.get(sample_id)
        if bundle is None:
            raise ValueError(
                f"Missing cached test bundle for sample {sample_id} while scoring tuned static. "
                "The dynamic test stage should build test-cache entries first."
            )
        cache_hits += 1
        static_metric_rows.append(bundle["static_metrics"][str(selected_static["premature_layer"])])
        callback(
            {
                "completed_samples": sample_position + 1,
                "total_samples": len(test_samples),
                "sample_index": sample_position,
                "question": sample.question,
            }
        )

    duration = tracker.finish_stage()
    static_summary = aggregate_mc_metrics(static_metric_rows)
    held_out = held_out or {}
    held_out["tuned_static"] = {
        "MC1": float(static_summary["avg_mc1"]),
        "MC2": float(static_summary["avg_mc2"]),
        "MC3": float(static_summary["avg_mc3"]),
    }
    held_out["num_samples"] = int(static_summary["num_samples"])
    fold_state["held_out_test"] = held_out
    _mark_stage_complete(state, stage_id=str(stage["id"]), duration_seconds=duration)
    fold_state["updated_at"] = _utc_now()
    _persist_state(output_dir, state, fold_name=str(fold_state["fold_name"]))
    logger.log(
        f"{fold_state['fold_name']} test tuned static profiling: cache_hits={cache_hits}, cache_misses=0, "
        f"forward_batches=0, candidate_batch_size=reused, avg_sample_s={duration / len(test_samples):.2f}"
    )


def _finalize_fold(*, logger: RunLogger, state: dict[str, Any], fold_state: dict[str, Any], output_dir: Path) -> None:
    stage_id = f"{fold_state['fold_name']}.partial"
    if stage_id in state["completed_stages"]:
        return
    if not fold_state.get("held_out_test"):
        raise ValueError(f"Cannot finalize {fold_state['fold_name']}: held-out test is incomplete.")
    _mark_stage_complete(state, stage_id=stage_id, duration_seconds=0.0)
    fold_state["completed_stages"] = sorted(
        stage for stage in state["completed_stages"] if stage.startswith(f"{fold_state['fold_name']}.")
    )
    fold_state["updated_at"] = _utc_now()
    _persist_state(output_dir, state, fold_name=str(fold_state["fold_name"]))
    logger.log(f"Wrote fold partial to: {_fold_partial_path(output_dir, str(fold_state['fold_name']))}")

def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    output_dir = ensure_output_dir(config.get("output_dir", PROJECT_ROOT / "outputs" / "truthfulqa_tuned_bucket_cv"))
    logger = RunLogger(output_dir)
    resume_enabled = not args.no_resume
    run_signature = _build_run_signature(config)
    profile_first_n = int(args.profile_first_n)
    profile_batch_sizes = _parse_profile_batch_sizes(args.profile_batch_sizes)

    state = None if profile_first_n > 0 else _load_json_if_exists(output_dir / STATE_FILE)
    if state is not None:
        _validate_state(state, run_signature)

    if args.rebuild_summary_only:
        if state is None:
            legacy = _load_json_if_exists(output_dir / LEGACY_PARTIAL_FILE)
            if legacy is None or int(legacy.get("folds_completed", 0)) != 2:
                raise RuntimeError("No complete checkpoint state found. Need cv_state.json or a complete legacy partial to rebuild summary.")
            state = _state_from_legacy_partial(legacy, run_signature)
            _atomic_write_json(output_dir / STATE_FILE, state)
        _write_final_summary(output_dir, state, logger)
        return

    if not resume_enabled and state is not None:
        raise RuntimeError(
            "Existing cv_state.json found but --no-resume was requested. Use a fresh output_dir or remove the old state first."
        )

    if resume_enabled and state is not None:
        if "final.aggregate_summary" in state.get("completed_stages", []) and (output_dir / SUMMARY_FILE).exists():
            logger.log("Summary already exists and final aggregate stage is complete; nothing to do.")
            return
        if len(_collect_fold_reports(state)) == 2 and "final.aggregate_summary" not in state.get("completed_stages", []):
            logger.log("Both fold partials are already complete; rebuilding final summary without recomputation.")
            _write_final_summary(output_dir, state, logger)
            return

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
    candidate_batch_size = int(config.get("candidate_batch_size", 1))

    raw_dynamic_buckets = config["dynamic_bucket_candidates"]
    raw_static_layers = [int(layer) for layer in config["static_layer_candidates"]]
    model_kwargs = {key: config[key] for key in LOAD_CONFIG_KEYS if key in config}

    samples = load_truthfulqa_samples(csv_path)
    sample_stats = _candidate_count_stats(samples)
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
        dynamic_buckets.append(
            {
                "name": name,
                "candidate_premature_layers": internal_layers,
                "official_layer_ids": [layer + 1 for layer in internal_layers],
            }
        )
    static_layers = normalize_layer_bucket(
        raw_static_layers,
        mature_layer,
        num_hidden_layers,
        field_name="static_layer_candidates",
    )

    if profile_first_n > 0:
        _run_profile_mode(
            logger=logger,
            output_dir=output_dir,
            model=model,
            tokenizer=tokenizer,
            samples=samples,
            prompt_style=prompt_style,
            score_mode=score_mode,
            post_softmax=post_softmax,
            relative_top=relative_top,
            relative_top_value=relative_top_value,
            mature_layer=mature_layer,
            dynamic_buckets=dynamic_buckets,
            static_layers=static_layers,
            profile_first_n=profile_first_n,
            profile_batch_sizes=profile_batch_sizes,
        )
        return

    fold_a, fold_b, split_meta = _two_fold_split(samples, split_seed)
    fold_specs = [
        {"validation_indices": fold_a, "test_indices": fold_b, "name": "fold1", "validation_size": len(fold_a), "test_size": len(fold_b)},
        {"validation_indices": fold_b, "test_indices": fold_a, "name": "fold2", "validation_size": len(fold_b), "test_size": len(fold_a)},
    ]

    if state is None:
        state = _state_skeleton(config, run_signature, split_meta, dynamic_buckets, static_layers)
        _mark_stage_complete(state, stage_id="split.metadata", duration_seconds=0.0)
    state = _migrate_legacy_partial(output_dir, state, fold_specs, dynamic_buckets, static_layers, logger)
    if "split.metadata" not in state.get("completed_stages", []):
        _mark_stage_complete(state, stage_id="split.metadata", duration_seconds=0.0)
    _persist_state(output_dir, state)

    stage_plan = _build_stage_plan(fold_specs, dynamic_buckets, static_layers)
    tracker = ProgressTracker(
        stage_plan=stage_plan,
        output_dir=output_dir,
        logger=logger,
        completed_stage_ids=set(str(item) for item in state.get("completed_stages", [])),
        stage_durations={str(key): float(value) for key, value in state.get("stage_durations_seconds", {}).items()},
    )
    stage_lookup = {str(stage["id"]): stage for stage in stage_plan}

    logger.log(f"Model: {model_name}")
    logger.log(f"CSV: {csv_path}")
    logger.log(
        f"Samples: {sample_stats['num_samples']} | avg_true={sample_stats['avg_true_answers']:.2f} | avg_false={sample_stats['avg_false_answers']:.2f}"
    )
    logger.log(f"Split: {split_meta}")
    logger.log(f"Dynamic buckets: {[bucket['name'] for bucket in dynamic_buckets]}")
    logger.log(f"Static layers (internal): {static_layers}")
    logger.log(f"Estimated sample-stage count: {tracker.total_stages}")
    logger.log(f"Candidate batch size: {candidate_batch_size}")
    logger.log(
        "Why this can be slow: two full folds, official 6-shot prompt, multiple true/false continuations per sample, and dynamic DoLa recomputes JS ranking across candidate layers."
    )

    for fold_spec in fold_specs:
        fold_name = str(fold_spec["name"])
        fold_state = _ensure_fold_state(state, fold_spec)
        if f"{fold_name}.partial" in state.get("completed_stages", []) and fold_state.get("held_out_test"):
            logger.log(f"Skipping completed fold: {fold_name} | using saved partial results.")
            continue
        validation_sample_ids = list(fold_spec["validation_indices"])
        test_sample_ids = list(fold_spec["test_indices"])
        validation_samples = _subset_from_indices(samples, validation_sample_ids)
        test_samples = _subset_from_indices(samples, test_sample_ids)

        _run_validation_shared_stage(
            stage=stage_lookup[f"{fold_name}.validation.shared"],
            tracker=tracker,
            logger=logger,
            state=state,
            fold_state=fold_state,
            output_dir=output_dir,
            model=model,
            tokenizer=tokenizer,
            validation_sample_ids=validation_sample_ids,
            validation_samples=validation_samples,
            dynamic_buckets=dynamic_buckets,
            static_layers=static_layers,
            mature_layer=mature_layer,
            prompt_style=prompt_style,
            score_mode=score_mode,
            post_softmax=post_softmax,
            relative_top=relative_top,
            relative_top_value=relative_top_value,
            candidate_batch_size=candidate_batch_size,
        )

        _complete_selection_stage(logger=logger, state=state, fold_state=fold_state, output_dir=output_dir)
        _run_test_shared_stage(
            stage=stage_lookup[f"{fold_name}.test.shared"],
            tracker=tracker,
            logger=logger,
            state=state,
            fold_state=fold_state,
            output_dir=output_dir,
            model=model,
            tokenizer=tokenizer,
            test_samples=test_samples,
            prompt_style=prompt_style,
            score_mode=score_mode,
            post_softmax=post_softmax,
            relative_top=relative_top,
            relative_top_value=relative_top_value,
            mature_layer=mature_layer,
            candidate_batch_size=candidate_batch_size,
        )
        _finalize_fold(logger=logger, state=state, fold_state=fold_state, output_dir=output_dir)

    _write_final_summary(output_dir, state, logger)


if __name__ == "__main__":
    main()
