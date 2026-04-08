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

from scripts.hf_eval_compare_subset import LOAD_CONFIG_KEYS, evaluate_compare_subset
from src.dola_utils import normalize_layer_bucket
from src.modeling import load_model_and_tokenizer
from src.truthfulqa_mc import TruthfulQASample, load_truthfulqa_samples
from src.utils import ensure_output_dir, load_yaml_config


STATE_FILE = "cv_state.json"
PROGRESS_FILE = "truthfulqa_tuned_bucket_cv_progress.json"
SUMMARY_FILE = "truthfulqa_tuned_bucket_cv_summary.json"
LEGACY_PARTIAL_FILE = "truthfulqa_tuned_bucket_cv_partial.json"


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
        self.dynamic_seconds_per_sample_layer: float | None = None
        self.static_seconds_per_sample: float | None = None
        self._bootstrap_rates(stage_durations)

    def _bootstrap_rates(self, stage_durations: dict[str, float]) -> None:
        dynamic_rates: list[float] = []
        static_rates: list[float] = []
        for stage_id, duration in stage_durations.items():
            stage = self.stage_lookup.get(stage_id)
            if stage is None:
                continue
            samples = int(stage["samples"])
            if samples <= 0:
                continue
            kind = str(stage["kind"])
            if kind == "dynamic":
                layer_count = max(int(stage["layer_count"]), 1)
                dynamic_rates.append(float(duration) / (samples * layer_count))
            elif kind == "static":
                static_rates.append(float(duration) / samples)
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
        self._write_progress_snapshot(completed_samples=0, eta_seconds=self._estimate_eta(0))
        self.logger.log(
            f"Stage {self.completed_stages + 1}/{self.total_stages} start: {stage['label']} "
            f"| samples={self.current_stage_total_samples} | layer_factor={self.current_stage_layer_count}"
        )

    def make_callback(self) -> Any:
        def _callback(event: dict[str, object]) -> None:
            completed_samples = int(event["completed_samples"])
            should_log = (
                completed_samples == 1
                or completed_samples == self.current_stage_total_samples
                or completed_samples - self.last_logged_sample >= 25
            )
            if not should_log:
                return
            self.last_logged_sample = completed_samples
            elapsed = time.perf_counter() - self.start_time
            percent = 100.0 * self._completed_weight_within_stage(completed_samples) / max(self.total_weight, 1.0)
            eta_seconds = self._estimate_eta(completed_samples)
            stage_bar = _progress_bar(completed_samples, self.current_stage_total_samples)
            self.logger.log(
                f"{stage_bar} {percent:5.1f}% | {self.current_stage['label']} "
                f"| {completed_samples}/{self.current_stage_total_samples} samples "
                f"| elapsed={_format_seconds(elapsed)} | eta={_format_seconds(eta_seconds)}"
            )
            self._write_progress_snapshot(completed_samples=completed_samples, eta_seconds=eta_seconds)

        return _callback

    def finish_stage(self) -> float:
        if self.current_stage is None:
            raise RuntimeError("No active stage to finish.")
        stage_elapsed = time.perf_counter() - self.current_stage_start
        if self.current_stage_total_samples > 0:
            if str(self.current_stage["kind"]) == "dynamic":
                observed = stage_elapsed / (self.current_stage_total_samples * self.current_stage_layer_count)
                self.dynamic_seconds_per_sample_layer = _ema(self.dynamic_seconds_per_sample_layer, observed)
            else:
                observed = stage_elapsed / self.current_stage_total_samples
                self.static_seconds_per_sample = _ema(self.static_seconds_per_sample, observed)
        self.completed_stage_ids.add(str(self.current_stage["id"]))
        self.completed_weight += float(self.current_stage["weight"])
        self.completed_stages += 1
        self.logger.log(
            f"Stage {self.completed_stages}/{self.total_stages} done: {self.current_stage['label']} "
            f"| duration={_format_seconds(stage_elapsed)}"
        )
        self._write_progress_snapshot(
            completed_samples=self.current_stage_total_samples,
            eta_seconds=self._estimate_eta(self.current_stage_total_samples),
        )
        self.current_stage = None
        return stage_elapsed

    def _completed_weight_within_stage(self, completed_samples: int) -> float:
        if self.current_stage is None or self.current_stage_total_samples <= 0:
            return self.completed_weight
        fraction = min(max(completed_samples / self.current_stage_total_samples, 0.0), 1.0)
        return self.completed_weight + float(self.current_stage["weight"]) * fraction

    def _estimate_eta(self, completed_samples: int) -> float:
        remaining_current = 0.0
        if self.current_stage is not None and 0 < completed_samples < self.current_stage_total_samples:
            current_elapsed = time.perf_counter() - self.current_stage_start
            seconds_per_sample = current_elapsed / completed_samples
            remaining_current = seconds_per_sample * (self.current_stage_total_samples - completed_samples)

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
        return max(remaining_current + remaining_future, 0.0)

    def _estimate_stage_seconds(self, stage: dict[str, object]) -> float:
        samples = int(stage["samples"])
        if samples <= 0:
            return 0.0
        kind = str(stage["kind"])
        layer_count = max(int(stage["layer_count"]), 1)
        if kind == "dynamic":
            if self.dynamic_seconds_per_sample_layer is not None:
                return samples * layer_count * self.dynamic_seconds_per_sample_layer
            return samples * layer_count * 1.0
        if self.static_seconds_per_sample is not None:
            return samples * self.static_seconds_per_sample
        return samples * 1.0

    def _write_progress_snapshot(self, *, completed_samples: int, eta_seconds: float) -> None:
        snapshot = {
            "completed_stages": self.completed_stages,
            "total_stages": self.total_stages,
            "current_stage": None if self.current_stage is None else self.current_stage["label"],
            "current_stage_kind": None if self.current_stage is None else self.current_stage["kind"],
            "current_stage_completed_samples": completed_samples,
            "current_stage_total_samples": self.current_stage_total_samples,
            "elapsed_seconds": time.perf_counter() - self.start_time,
            "eta_seconds": eta_seconds,
            "eta_method": "empirical_stage_rate_resume_safe",
            "completed_weight": self._completed_weight_within_stage(completed_samples),
            "total_weight": self.total_weight,
            "dynamic_seconds_per_sample_layer": self.dynamic_seconds_per_sample_layer,
            "static_seconds_per_sample": self.static_seconds_per_sample,
        }
        _atomic_write_json(self.output_dir / PROGRESS_FILE, snapshot)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run two-fold TruthfulQA-MC bucket tuning using MC3 on validation folds."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--rebuild-summary-only", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
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
    return float(samples * (layer_count if kind == "dynamic" else 1))


def _build_stage_plan(
    fold_specs: list[dict[str, object]],
    dynamic_buckets: list[dict[str, object]],
    static_layers: list[int],
) -> list[dict[str, object]]:
    plan: list[dict[str, object]] = []
    for fold_spec in fold_specs:
        fold_name = str(fold_spec["name"])
        validation_size = int(fold_spec["validation_size"])
        test_size = int(fold_spec["test_size"])
        for bucket in dynamic_buckets:
            layer_count = len(bucket["candidate_premature_layers"])
            plan.append(
                {
                    "id": f"{fold_name}.validation.dynamic.{bucket['name']}",
                    "label": f"{fold_name} validation dynamic {bucket['name']}",
                    "kind": "dynamic",
                    "samples": validation_size,
                    "layer_count": layer_count,
                    "weight": _stage_weight("dynamic", validation_size, layer_count),
                }
            )
        for layer in static_layers:
            plan.append(
                {
                    "id": f"{fold_name}.validation.static.{layer}",
                    "label": f"{fold_name} validation static layer {layer}",
                    "kind": "static",
                    "samples": validation_size,
                    "layer_count": 1,
                    "weight": _stage_weight("static", validation_size, 1),
                }
            )
        plan.append(
            {
                "id": f"{fold_name}.test.dynamic",
                "label": f"{fold_name} test tuned dynamic",
                "kind": "dynamic",
                "samples": test_size,
                "layer_count": 1,
                "weight": _stage_weight("dynamic", test_size, 1),
            }
        )
        plan.append(
            {
                "id": f"{fold_name}.test.static",
                "label": f"{fold_name} test tuned static",
                "kind": "static",
                "samples": test_size,
                "layer_count": 1,
                "weight": _stage_weight("static", test_size, 1),
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
    validation_samples: list[TruthfulQASample],
    bucket: dict[str, object],
    mature_layer: int,
    prompt_style: str,
    score_mode: str,
    post_softmax: bool,
    relative_top: float,
    relative_top_value: float,
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

    tracker.start_stage(stage)
    result = _evaluate_dynamic_candidate(
        model=model,
        tokenizer=tokenizer,
        samples=validation_samples,
        bucket_name=bucket_name,
        candidate_layers=list(bucket["candidate_premature_layers"]),
        mature_layer=mature_layer,
        prompt_style=prompt_style,
        score_mode=score_mode,
        post_softmax=post_softmax,
        relative_top=relative_top,
        relative_top_value=relative_top_value,
        progress_callback=tracker.make_callback(),
    )
    duration = tracker.finish_stage()
    fold_state["validation_dynamic_candidates"][bucket_name] = result
    _mark_stage_complete(state, stage_id=str(stage["id"]), duration_seconds=duration)
    fold_state["updated_at"] = _utc_now()
    _persist_state(output_dir, state, fold_name=str(fold_state["fold_name"]))
    logger.log(
        f"{fold_state['fold_name']} validation dynamic candidate {bucket_name} -> MC3={result['summary']['dola_avg_mc3']:.4f}"
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
    validation_samples: list[TruthfulQASample],
    layer: int,
    mature_layer: int,
    prompt_style: str,
    score_mode: str,
    post_softmax: bool,
    relative_top: float,
    relative_top_value: float,
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
    result = _evaluate_static_candidate(
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
        progress_callback=tracker.make_callback(),
    )
    duration = tracker.finish_stage()
    fold_state["validation_static_candidates"][layer_key] = result
    _mark_stage_complete(state, stage_id=str(stage["id"]), duration_seconds=duration)
    fold_state["updated_at"] = _utc_now()
    _persist_state(output_dir, state, fold_name=str(fold_state["fold_name"]))
    logger.log(
        f"{fold_state['fold_name']} validation static candidate {layer} -> MC3={result['summary']['dola_avg_mc3']:.4f}"
    )
    best_so_far = _current_best_mc3(fold_state["validation_static_candidates"])
    if best_so_far is not None:
        logger.log(f"{fold_state['fold_name']} current best validation static MC3={best_so_far:.4f}")
    return result


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
    test_samples: list[TruthfulQASample],
    prompt_style: str,
    score_mode: str,
    post_softmax: bool,
    relative_top: float,
    relative_top_value: float,
    mature_layer: int,
) -> None:
    held_out = fold_state.get("held_out_test")
    if str(stage["id"]) in state["completed_stages"]:
        if not held_out or "tuned_dynamic" not in held_out or "vanilla" not in held_out:
            raise ValueError(f"Completed stage {stage['id']} is missing its saved held-out result.")
        logger.log(f"Skipping completed stage: {stage['label']}")
        return

    selected_dynamic = fold_state.get("selected_dynamic_bucket")
    if selected_dynamic is None:
        raise ValueError(f"Cannot run test dynamic for {fold_state['fold_name']}: no selected dynamic bucket.")

    stage_for_run = dict(stage)
    stage_for_run["label"] = f"{fold_state['fold_name']} test tuned dynamic {selected_dynamic['name']}"
    stage_for_run["layer_count"] = len(selected_dynamic["candidate_premature_layers"])
    stage_for_run["weight"] = _stage_weight("dynamic", len(test_samples), stage_for_run["layer_count"])
    tracker.start_stage(stage_for_run)
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
        progress_callback=tracker.make_callback(),
    )
    duration = tracker.finish_stage()
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


def _run_test_static_stage(
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
        progress_callback=tracker.make_callback(),
    )
    duration = tracker.finish_stage()
    held_out = held_out or {}
    held_out["tuned_static"] = {
        "MC1": float(static_summary["dola_avg_mc1"]),
        "MC2": float(static_summary["dola_avg_mc2"]),
        "MC3": float(static_summary["dola_avg_mc3"]),
    }
    held_out["num_samples"] = int(static_summary["num_samples"])
    fold_state["held_out_test"] = held_out
    _mark_stage_complete(state, stage_id=str(stage["id"]), duration_seconds=duration)
    fold_state["updated_at"] = _utc_now()
    _persist_state(output_dir, state, fold_name=str(fold_state["fold_name"]))


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

    state = _load_json_if_exists(output_dir / STATE_FILE)
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
    logger.log(
        "Why this can be slow: two full folds, official 6-shot prompt, multiple true/false continuations per sample, and dynamic DoLa recomputes JS ranking across candidate layers."
    )

    for fold_spec in fold_specs:
        fold_name = str(fold_spec["name"])
        fold_state = _ensure_fold_state(state, fold_spec)
        if f"{fold_name}.partial" in state.get("completed_stages", []) and fold_state.get("held_out_test"):
            logger.log(f"Skipping completed fold: {fold_name} | using saved partial results.")
            continue
        validation_samples = _subset_from_indices(samples, list(fold_spec["validation_indices"]))
        test_samples = _subset_from_indices(samples, list(fold_spec["test_indices"]))

        for bucket in dynamic_buckets:
            stage_id = f"{fold_name}.validation.dynamic.{bucket['name']}"
            _run_validation_dynamic_stage(
                stage=stage_lookup[stage_id],
                tracker=tracker,
                logger=logger,
                state=state,
                fold_state=fold_state,
                output_dir=output_dir,
                model=model,
                tokenizer=tokenizer,
                validation_samples=validation_samples,
                bucket=bucket,
                mature_layer=mature_layer,
                prompt_style=prompt_style,
                score_mode=score_mode,
                post_softmax=post_softmax,
                relative_top=relative_top,
                relative_top_value=relative_top_value,
            )

        for layer in static_layers:
            stage_id = f"{fold_name}.validation.static.{layer}"
            _run_validation_static_stage(
                stage=stage_lookup[stage_id],
                tracker=tracker,
                logger=logger,
                state=state,
                fold_state=fold_state,
                output_dir=output_dir,
                model=model,
                tokenizer=tokenizer,
                validation_samples=validation_samples,
                layer=layer,
                mature_layer=mature_layer,
                prompt_style=prompt_style,
                score_mode=score_mode,
                post_softmax=post_softmax,
                relative_top=relative_top,
                relative_top_value=relative_top_value,
            )

        _complete_selection_stage(logger=logger, state=state, fold_state=fold_state, output_dir=output_dir)
        _run_test_dynamic_stage(
            stage=stage_lookup[f"{fold_name}.test.dynamic"],
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
        )
        _run_test_static_stage(
            stage=stage_lookup[f"{fold_name}.test.static"],
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
        )
        _finalize_fold(logger=logger, state=state, fold_state=fold_state, output_dir=output_dir)

    _write_final_summary(output_dir, state, logger)


if __name__ == "__main__":
    main()
