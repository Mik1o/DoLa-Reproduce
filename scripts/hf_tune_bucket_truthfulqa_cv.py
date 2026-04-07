"""Run a paper-faithful two-fold tuned-bucket TruthfulQA-MC baseline."""

from __future__ import annotations

import argparse
import json
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


class ProgressTracker:
    def __init__(self, *, total_work_units: float, total_stages: int, output_dir: Path) -> None:
        self.total_work_units = max(float(total_work_units), 1.0)
        self.total_stages = total_stages
        self.output_dir = output_dir
        self.start_time = time.perf_counter()
        self.completed_work_units = 0.0
        self.completed_stages = 0
        self.current_stage_name = ""
        self.current_stage_total_samples = 0
        self.current_stage_work_per_sample = 0.0
        self.current_stage_start = self.start_time
        self.last_logged_sample = 0

    def start_stage(self, name: str, total_samples: int, work_per_sample: float) -> None:
        self.current_stage_name = name
        self.current_stage_total_samples = total_samples
        self.current_stage_work_per_sample = work_per_sample
        self.current_stage_start = time.perf_counter()
        self.last_logged_sample = 0
        self._write_progress_snapshot(completed_samples=0)
        print(
            f"[hf_tune_bucket_truthfulqa_cv] Stage {self.completed_stages + 1}/{self.total_stages} start: {name} "
            f"| samples={total_samples} | work_per_sample={work_per_sample:.2f}",
            flush=True,
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
            total_completed = self.completed_work_units + completed_samples * self.current_stage_work_per_sample
            remaining_work = max(self.total_work_units - total_completed, 0.0)
            eta_seconds = (elapsed / total_completed) * remaining_work if total_completed > 0 else 0.0
            percent = 100.0 * total_completed / self.total_work_units
            stage_bar = _progress_bar(completed_samples, self.current_stage_total_samples)
            print(
                f"[hf_tune_bucket_truthfulqa_cv] {stage_bar} {percent:5.1f}% | {self.current_stage_name} "
                f"| {completed_samples}/{self.current_stage_total_samples} samples "
                f"| elapsed={_format_seconds(elapsed)} | eta={_format_seconds(eta_seconds)}",
                flush=True,
            )
            self._write_progress_snapshot(completed_samples=completed_samples, eta_seconds=eta_seconds)

        return _callback

    def finish_stage(self) -> None:
        self.completed_work_units += self.current_stage_total_samples * self.current_stage_work_per_sample
        self.completed_stages += 1
        stage_elapsed = time.perf_counter() - self.current_stage_start
        print(
            f"[hf_tune_bucket_truthfulqa_cv] Stage {self.completed_stages}/{self.total_stages} done: {self.current_stage_name} "
            f"| duration={_format_seconds(stage_elapsed)}",
            flush=True,
        )
        self._write_progress_snapshot(completed_samples=self.current_stage_total_samples, eta_seconds=0.0)

    def _write_progress_snapshot(self, *, completed_samples: int, eta_seconds: float = 0.0) -> None:
        snapshot = {
            "completed_stages": self.completed_stages,
            "total_stages": self.total_stages,
            "current_stage": self.current_stage_name,
            "current_stage_completed_samples": completed_samples,
            "current_stage_total_samples": self.current_stage_total_samples,
            "elapsed_seconds": time.perf_counter() - self.start_time,
            "eta_seconds": eta_seconds,
            "completed_work_units": self.completed_work_units,
            "total_work_units": self.total_work_units,
        }
        progress_path = self.output_dir / "truthfulqa_tuned_bucket_cv_progress.json"
        with progress_path.open("w", encoding="utf-8") as handle:
            json.dump(snapshot, handle, indent=2, ensure_ascii=False)
            handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run two-fold TruthfulQA-MC bucket tuning using MC3 on validation folds."
    )
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


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


def _estimate_total_work(
    fold_specs: list[dict[str, object]],
    dynamic_buckets: list[dict[str, object]],
    static_layers: list[int],
) -> float:
    max_dynamic_weight = max((1 + len(bucket["candidate_premature_layers"]) for bucket in dynamic_buckets), default=2)
    total = 0.0
    for fold_spec in fold_specs:
        validation_size = int(fold_spec["validation_size"])
        test_size = int(fold_spec["test_size"])
        for bucket in dynamic_buckets:
            total += validation_size * (1 + len(bucket["candidate_premature_layers"]))
        for _layer in static_layers:
            total += validation_size * 2
        total += test_size * max_dynamic_weight
        total += test_size * 2
    return total


def _candidate_count_stats(samples: list[TruthfulQASample]) -> dict[str, float | int]:
    return {
        "num_samples": len(samples),
        "avg_true_answers": sum(len(sample.correct_answers) for sample in samples) / len(samples),
        "avg_false_answers": sum(len(sample.incorrect_answers) for sample in samples) / len(samples),
        "max_true_answers": max(len(sample.correct_answers) for sample in samples),
        "max_false_answers": max(len(sample.incorrect_answers) for sample in samples),
    }


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
        {"validation_indices": fold_a, "test_indices": fold_b, "name": "fold1", "validation_size": len(fold_a), "test_size": len(fold_b)},
        {"validation_indices": fold_b, "test_indices": fold_a, "name": "fold2", "validation_size": len(fold_b), "test_size": len(fold_a)},
    ]
    total_stages = len(fold_specs) * (len(dynamic_buckets) + len(static_layers) + 2)
    total_work_units = _estimate_total_work(fold_specs, dynamic_buckets, static_layers)
    tracker = ProgressTracker(total_work_units=total_work_units, total_stages=total_stages, output_dir=output_dir)

    print(f"[hf_tune_bucket_truthfulqa_cv] Model: {model_name}", flush=True)
    print(f"[hf_tune_bucket_truthfulqa_cv] CSV: {csv_path}", flush=True)
    print(f"[hf_tune_bucket_truthfulqa_cv] Samples: {sample_stats['num_samples']} | avg_true={sample_stats['avg_true_answers']:.2f} | avg_false={sample_stats['avg_false_answers']:.2f}", flush=True)
    print(f"[hf_tune_bucket_truthfulqa_cv] Split: {split_meta}", flush=True)
    print(f"[hf_tune_bucket_truthfulqa_cv] Dynamic buckets: {[bucket['name'] for bucket in dynamic_buckets]}", flush=True)
    print(f"[hf_tune_bucket_truthfulqa_cv] Static layers (internal): {static_layers}", flush=True)
    print(f"[hf_tune_bucket_truthfulqa_cv] Estimated weighted work units: {total_work_units:.0f}", flush=True)
    print(
        "[hf_tune_bucket_truthfulqa_cv] Why this can be slow: two full folds, official 6-shot prompt, "
        "multiple true/false continuations per sample, and dynamic DoLa recomputes JS ranking across candidate layers.",
        flush=True,
    )

    fold_reports: list[dict[str, object]] = []
    for fold_spec in fold_specs:
        validation_samples = _subset_from_indices(samples, fold_spec["validation_indices"])
        test_samples = _subset_from_indices(samples, fold_spec["test_indices"])

        dynamic_candidates: list[dict[str, object]] = []
        for bucket in dynamic_buckets:
            stage_name = f"{fold_spec['name']} validation dynamic {bucket['name']}"
            tracker.start_stage(stage_name, len(validation_samples), 1 + len(bucket["candidate_premature_layers"]))
            candidate_result = _evaluate_dynamic_candidate(
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
                progress_callback=tracker.make_callback(),
            )
            dynamic_candidates.append(candidate_result)
            tracker.finish_stage()
            print(
                f"[hf_tune_bucket_truthfulqa_cv] {fold_spec['name']} validation dynamic candidate {bucket['name']} "
                f"-> MC3={candidate_result['summary']['dola_avg_mc3']:.4f}",
                flush=True,
            )

        static_candidates: list[dict[str, object]] = []
        for layer in static_layers:
            stage_name = f"{fold_spec['name']} validation static layer {layer}"
            tracker.start_stage(stage_name, len(validation_samples), 2.0)
            candidate_result = _evaluate_static_candidate(
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
            static_candidates.append(candidate_result)
            tracker.finish_stage()
            print(
                f"[hf_tune_bucket_truthfulqa_cv] {fold_spec['name']} validation static candidate {layer} "
                f"-> MC3={candidate_result['summary']['dola_avg_mc3']:.4f}",
                flush=True,
            )

        selected_dynamic = _select_best_dynamic(dynamic_candidates)
        selected_static = _select_best_static(static_candidates)
        print(
            f"[hf_tune_bucket_truthfulqa_cv] {fold_spec['name']} selected dynamic={selected_dynamic['name']} "
            f"official_layers={selected_dynamic['candidate_premature_layers']} | selected static={selected_static['premature_layer']}",
            flush=True,
        )

        # Run dynamic and static held-out tests as two explicit stages so ETA stays readable.
        tracker.start_stage(
            f"{fold_spec['name']} test tuned dynamic {selected_dynamic['name']}",
            len(test_samples),
            1 + len(selected_dynamic["candidate_premature_layers"]),
        )
        dynamic_cb = tracker.make_callback()
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
            progress_callback=dynamic_cb,
        )
        tracker.finish_stage()

        tracker.start_stage(
            f"{fold_spec['name']} test tuned static layer {selected_static['premature_layer']}",
            len(test_samples),
            2.0,
        )
        static_cb = tracker.make_callback()
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
            progress_callback=static_cb,
        )
        tracker.finish_stage()

        held_out_test = {
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
        partial_report = {
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
            "folds_completed": len(fold_reports),
            "folds": fold_reports,
        }
        partial_path = output_dir / "truthfulqa_tuned_bucket_cv_partial.json"
        with partial_path.open("w", encoding="utf-8") as handle:
            json.dump(partial_report, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        print(f"[hf_tune_bucket_truthfulqa_cv] Wrote partial results to: {partial_path}", flush=True)

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

    print(f"[hf_tune_bucket_truthfulqa_cv] final_label={final_label}", flush=True)
    print(f"[hf_tune_bucket_truthfulqa_cv] Saved summary to: {output_path}", flush=True)


if __name__ == "__main__":
    main()
