"""Revalidate the DoLa paper low-vs-high bucket story on TruthfulQA-MC with two-fold CV."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.hf_eval_compare_subset import LOAD_CONFIG_KEYS
from scripts.hf_tune_bucket_truthfulqa_cv import (
    ProgressTracker,
    RunLogger,
    _average_fold_results,
    _build_run_signature,
    _build_stage_plan,
    _mark_stage_complete,
    _run_validation_shared_stage,
    _score_sample_multi_config,
    _subset_from_indices,
    _two_fold_split,
    _validate_state,
)
from src.dola_utils import normalize_layer_bucket
from src.metrics import aggregate_mc_metrics
from src.modeling import load_model_and_tokenizer
from src.truthfulqa_mc import TruthfulQASample, load_truthfulqa_samples
from src.utils import ensure_output_dir, load_yaml_config


STATE_FILE = "cv_state.json"
SUMMARY_FILE = "paper_revalidation_summary.json"
TIE_TOL = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a two-fold low-vs-high paper-style DoLa bucket revalidation."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--rebuild-summary-only", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


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


def _fold_partial_path(output_dir: Path, fold_name: str) -> Path:
    return output_dir / f"{fold_name}_partial.json"


def _state_skeleton(
    config: dict[str, Any],
    run_signature: str,
    split_meta: dict[str, object],
    dynamic_buckets: list[dict[str, object]],
    static_layers: list[int],
    low_bucket_info: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run_signature": run_signature,
        "task_name": str(config.get("task_name", "truthfulqa_paper_revalidation_cv")),
        "model_name": str(config["model_name"]),
        "csv_path": str(config["csv_path"]),
        "subset_mode": "full_dataset",
        "prompt_style": str(config.get("prompt_style", "official_tfqa_mc")),
        "score_mode": str(config.get("score_mode", "sum_logprob")),
        "validation_metric": "MC3",
        "mature_layer": int(config["mature_layer"]),
        "split_seed": int(config.get("split_seed", 20260407)),
        "split": split_meta,
        "dynamic_bucket_candidates": dynamic_buckets,
        "static_layer_candidates": [{"internal": layer, "official": layer + 1} for layer in static_layers],
        "low_bucket_info": low_bucket_info,
        "completed_stages": [],
        "stage_durations_seconds": {},
        "folds": {},
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
    }


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
            "selected_dynamic_bucket_tie": [],
            "selected_static_layer": None,
            "held_out_test": None,
            "updated_at": _utc_now(),
        }
    return folds[fold_name]


def _write_fold_partial(output_dir: Path, fold_state: dict[str, Any]) -> None:
    payload = dict(fold_state)
    payload["updated_at"] = _utc_now()
    _atomic_write_json(_fold_partial_path(output_dir, str(fold_state["fold_name"])), payload)


def _persist_state(output_dir: Path, state: dict[str, Any], *, fold_name: str | None = None) -> None:
    state["updated_at"] = _utc_now()
    _atomic_write_json(output_dir / STATE_FILE, state)
    if fold_name is not None:
        _write_fold_partial(output_dir, state["folds"][fold_name])


def _select_dynamic_by_mc3_only(candidates: list[dict[str, object]]) -> dict[str, Any]:
    if not candidates:
        raise ValueError("dynamic candidates must not be empty.")
    best_mc3 = max(float(item["summary"]["dola_avg_mc3"]) for item in candidates)
    tied = [
        item
        for item in candidates
        if math.isclose(float(item["summary"]["dola_avg_mc3"]), best_mc3, rel_tol=0.0, abs_tol=TIE_TOL)
    ]
    tied_names = sorted(str(item["name"]) for item in tied)
    return {
        "best_mc3": best_mc3,
        "is_tie": len(tied) > 1,
        "selected_bucket": None if len(tied) > 1 else tied[0],
        "tied_bucket_names": tied_names,
    }


def _select_best_static(candidates: list[dict[str, object]]) -> dict[str, object]:
    return max(
        candidates,
        key=lambda item: (
            float(item["summary"]["dola_avg_mc3"]),
            float(item["summary"]["dola_avg_mc2"]),
            float(item["summary"]["dola_avg_mc1"]),
        ),
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
        logger.log(f"Skipping completed stage: {fold_state['fold_name']} selection")
        return

    dynamic_candidates = list(fold_state["validation_dynamic_candidates"].values())
    static_candidates = list(fold_state["validation_static_candidates"].values())
    if not dynamic_candidates or not static_candidates:
        raise ValueError(f"Cannot select for {fold_state['fold_name']}: validation results incomplete.")

    dynamic_choice = _select_dynamic_by_mc3_only(dynamic_candidates)
    selected_static = _select_best_static(static_candidates)
    fold_state["selected_dynamic_bucket"] = dynamic_choice["selected_bucket"]
    fold_state["selected_dynamic_bucket_tie"] = dynamic_choice["tied_bucket_names"]
    fold_state["selected_static_layer"] = selected_static
    _mark_stage_complete(state, stage_id=stage_id, duration_seconds=0.0)
    fold_state["updated_at"] = _utc_now()
    _persist_state(output_dir, state, fold_name=str(fold_state["fold_name"]))

    if dynamic_choice["is_tie"]:
        logger.log(
            f"{fold_state['fold_name']} dynamic selection=tie on MC3={dynamic_choice['best_mc3']:.4f} "
            f"| tied_buckets={dynamic_choice['tied_bucket_names']} | selected static={selected_static['premature_layer']}"
        )
    else:
        logger.log(
            f"{fold_state['fold_name']} selected dynamic={dynamic_choice['selected_bucket']['name']} "
            f"| selected static={selected_static['premature_layer']} "
            f"| validation MC3={dynamic_choice['best_mc3']:.4f}"
        )


def _aggregate_mc_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    aggregate = aggregate_mc_metrics(rows)
    return {
        "MC1": float(aggregate["avg_mc1"]),
        "MC2": float(aggregate["avg_mc2"]),
        "MC3": float(aggregate["avg_mc3"]),
    }


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
    stage_id = str(stage["id"])
    held_out = fold_state.get("held_out_test")
    if stage_id in state["completed_stages"]:
        if held_out is not None:
            logger.log(f"Skipping completed stage: {stage['label']}")
            return
        raise ValueError(f"Completed stage {stage_id} is missing held-out results.")

    selected_static = fold_state.get("selected_static_layer")
    if selected_static is None:
        raise ValueError(f"Cannot run held-out stage for {fold_state['fold_name']}: no selected static layer.")

    selected_dynamic = fold_state.get("selected_dynamic_bucket")
    selected_dynamic_tie = list(fold_state.get("selected_dynamic_bucket_tie") or [])
    if selected_dynamic is None and not selected_dynamic_tie:
        raise ValueError(f"Cannot run held-out stage for {fold_state['fold_name']}: no dynamic selection state.")

    dynamic_bucket_entries = []
    if selected_dynamic is not None:
        dynamic_bucket_entries = [selected_dynamic]
    else:
        lookup = {
            str(item["name"]): item
            for item in fold_state["validation_dynamic_candidates"].values()
        }
        dynamic_bucket_entries = [lookup[name] for name in selected_dynamic_tie]

    dynamic_bucket_map = {
        str(item["name"]): list(item["candidate_premature_layers"])
        for item in dynamic_bucket_entries
    }
    static_layers = [int(selected_static["premature_layer"])]
    selected_union_layers = set(static_layers)
    for item in dynamic_bucket_entries:
        selected_union_layers.update(int(layer) for layer in item["candidate_premature_layers"])

    stage_for_run = dict(stage)
    stage_for_run["label"] = f"{fold_state['fold_name']} held-out shared scoring"
    stage_for_run["layer_count"] = max(len(selected_union_layers), 1)
    stage_for_run["weight"] = float(len(test_samples) * stage_for_run["layer_count"])
    tracker.start_stage(stage_for_run)
    callback = tracker.make_callback()

    vanilla_metric_rows: list[dict[str, float]] = []
    static_metric_rows: list[dict[str, float]] = []
    dynamic_metric_rows = {name: [] for name in dynamic_bucket_map}
    total_answers = 0
    total_batches = 0

    logger.log(
        f"{fold_state['fold_name']} held-out note: scoring vanilla, selected static, "
        f"and dynamic buckets {sorted(dynamic_bucket_map)} without using test metrics for selection."
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
        static_metric_rows.append(bundle["static_metrics"][str(selected_static["premature_layer"])])
        for bucket_name in dynamic_bucket_map:
            dynamic_metric_rows[bucket_name].append(bundle["dynamic_metrics"][bucket_name])
        total_answers += int(bundle["profiling"]["answer_count"])
        total_batches += int(bundle["profiling"]["batch_count"])
        callback(
            {
                "completed_samples": sample_position + 1,
                "total_samples": len(test_samples),
                "sample_index": sample_position,
                "question": sample.question,
            }
        )

    duration = tracker.finish_stage()
    held_out = {
        "vanilla": _aggregate_mc_rows(vanilla_metric_rows),
        "tuned_static": _aggregate_mc_rows(static_metric_rows),
        "num_samples": len(test_samples),
        "dynamic_selection_status": "tie" if selected_dynamic is None else "unique",
        "selected_static_layer": int(selected_static["premature_layer"]),
    }
    if selected_dynamic is not None:
        selected_name = str(selected_dynamic["name"])
        held_out["tuned_dynamic"] = _aggregate_mc_rows(dynamic_metric_rows[selected_name])
        held_out["selected_dynamic_bucket_name"] = selected_name
    else:
        held_out["selected_dynamic_bucket_name"] = None
        held_out["tied_dynamic_candidates"] = {
            name: _aggregate_mc_rows(rows)
            for name, rows in dynamic_metric_rows.items()
        }
        held_out["selected_dynamic_bucket_tie"] = selected_dynamic_tie

    fold_state["held_out_test"] = held_out
    _mark_stage_complete(state, stage_id=stage_id, duration_seconds=duration)
    fold_state["updated_at"] = _utc_now()
    _persist_state(output_dir, state, fold_name=str(fold_state["fold_name"]))
    logger.log(
        f"{fold_state['fold_name']} held-out profiling: forward_batches={total_batches}, "
        f"candidate_batch_size={candidate_batch_size}, avg_sample_s={duration / len(test_samples):.2f}, "
        f"avg_answer_s={(duration / total_answers) if total_answers else 0.0:.4f}"
    )


def _finalize_fold(
    *,
    logger: RunLogger,
    state: dict[str, Any],
    fold_state: dict[str, Any],
    output_dir: Path,
) -> None:
    stage_id = f"{fold_state['fold_name']}.partial"
    if stage_id in state["completed_stages"]:
        return
    if not fold_state.get("held_out_test"):
        raise ValueError(f"Cannot finalize {fold_state['fold_name']}: held-out results missing.")
    _mark_stage_complete(state, stage_id=stage_id, duration_seconds=0.0)
    fold_state["updated_at"] = _utc_now()
    _persist_state(output_dir, state, fold_name=str(fold_state["fold_name"]))
    logger.log(f"Wrote fold partial to: {_fold_partial_path(output_dir, str(fold_state['fold_name']))}")


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
                "selected_dynamic_bucket_tie": fold_state.get("selected_dynamic_bucket_tie") or [],
                "selected_static_layer": fold_state["selected_static_layer"],
                "held_out_test": fold_state["held_out_test"],
            }
        )
    return reports


def _average_selected_results(fold_reports: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not fold_reports:
        return None
    if any("tuned_dynamic" not in fold["held_out_test"] for fold in fold_reports):
        return None
    return _average_fold_results([fold["held_out_test"] for fold in fold_reports])


def _classify_revalidation(
    *,
    fold_reports: list[dict[str, Any]],
    low_bucket_embedding_inclusive: bool,
) -> str:
    high_selected_everywhere = True
    saw_unique_dynamic = False
    for fold in fold_reports:
        selected_dynamic = fold.get("selected_dynamic_bucket")
        if selected_dynamic is None:
            high_selected_everywhere = False
            continue
        saw_unique_dynamic = True
        if str(selected_dynamic["name"]) != "paper_high_16_32":
            high_selected_everywhere = False
    if low_bucket_embedding_inclusive:
        if saw_unique_dynamic and high_selected_everywhere:
            return "LIKELY_PAPER_HIGH_BUCKET_STILL_HOLDS"
        return "LIKELY_NO_CLEAR_LOW_VS_HIGH_WINNER"
    if saw_unique_dynamic and high_selected_everywhere:
        return "LIKELY_PAPER_HIGH_BUCKET_HOLDS_BUT_LOW_TEST_IS_APPROXIMATE"
    return "LIKELY_NO_CLEAR_LOW_VS_HIGH_WINNER"


def _build_summary_payload(state: dict[str, Any]) -> dict[str, Any]:
    fold_reports = _collect_fold_reports(state)
    if len(fold_reports) != 2:
        raise ValueError("Cannot build final summary yet: both folds are not complete.")
    selected_average = _average_selected_results(fold_reports)
    low_bucket_info = dict(state["low_bucket_info"])
    final_label = _classify_revalidation(
        fold_reports=fold_reports,
        low_bucket_embedding_inclusive=bool(low_bucket_info["embedding_inclusive"]),
    )
    return {
        "task_name": state["task_name"],
        "model_name": state["model_name"],
        "csv_path": state["csv_path"],
        "split": state["split"],
        "prompt_style": state["prompt_style"],
        "score_mode": state["score_mode"],
        "validation_metric": state["validation_metric"],
        "mature_layer": state["mature_layer"],
        "dynamic_bucket_candidates": state["dynamic_bucket_candidates"],
        "static_layer_candidates": state["static_layer_candidates"],
        "low_bucket_info": low_bucket_info,
        "folds": fold_reports,
        "selected_cv_average": selected_average,
        "final_label": final_label,
        "run_signature": state["run_signature"],
    }


def _write_final_summary(output_dir: Path, state: dict[str, Any], logger: RunLogger) -> dict[str, Any]:
    payload = _build_summary_payload(state)
    _atomic_write_json(output_dir / SUMMARY_FILE, payload)
    _mark_stage_complete(state, stage_id="final.aggregate_summary", duration_seconds=0.0)
    _persist_state(output_dir, state)
    logger.log(f"final_label={payload['final_label']}")
    logger.log(f"Saved summary to: {output_dir / SUMMARY_FILE}")
    return payload


def _current_low_bucket_info(config: dict[str, Any], dynamic_buckets: list[dict[str, object]]) -> dict[str, Any]:
    low_bucket_name = str(config.get("low_bucket_name", "paper_low_0_16"))
    low_bucket = next((bucket for bucket in dynamic_buckets if str(bucket["name"]) == low_bucket_name), None)
    if low_bucket is None:
        raise ValueError(f"Configured low_bucket_name '{low_bucket_name}' is missing from dynamic_bucket_candidates.")
    return {
        "bucket_name": low_bucket_name,
        "bucket_kind": str(config.get("low_bucket_kind", "approx_low_bucket_test")),
        "embedding_inclusive": False,
        "note": str(
            config.get(
                "low_bucket_note",
                "Current implementation only addresses transformer block outputs; embedding output is not selectable, so this is an approx_low_bucket_test.",
            )
        ),
        "candidate_premature_layers": list(low_bucket["candidate_premature_layers"]),
        "official_layer_ids": list(low_bucket["official_layer_ids"]),
    }


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    output_dir = ensure_output_dir(
        config.get(
            "output_dir",
            PROJECT_ROOT / "outputs" / "truthfulqa_paper_revalidation_cv",
        )
    )
    logger = RunLogger(output_dir)
    run_signature = _build_run_signature(config)
    state = _load_json_if_exists(output_dir / STATE_FILE)
    resume_enabled = not args.no_resume

    if state is not None:
        _validate_state(state, run_signature)
    if args.rebuild_summary_only:
        if state is None:
            raise RuntimeError("No existing cv_state.json found for summary rebuild.")
        _write_final_summary(output_dir, state, logger)
        return
    if not resume_enabled and state is not None:
        raise RuntimeError("Existing cv_state.json found but --no-resume was requested.")
    if resume_enabled and state is not None:
        if "final.aggregate_summary" in state.get("completed_stages", []) and (output_dir / SUMMARY_FILE).exists():
            logger.log("Summary already exists and final aggregate stage is complete; nothing to do.")
            return

    validation_metric = str(config.get("validation_metric", "MC3"))
    if validation_metric != "MC3":
        raise ValueError("paper-style revalidation only supports validation_metric=MC3.")

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
    model, tokenizer = load_model_and_tokenizer(
        model_name=model_name,
        device=device,
        **model_kwargs,
    )
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
    low_bucket_info = _current_low_bucket_info(config, dynamic_buckets)

    fold_a, fold_b, split_meta = _two_fold_split(samples, split_seed)
    fold_specs = [
        {"validation_indices": fold_a, "test_indices": fold_b, "name": "fold1", "validation_size": len(fold_a), "test_size": len(fold_b)},
        {"validation_indices": fold_b, "test_indices": fold_a, "name": "fold2", "validation_size": len(fold_b), "test_size": len(fold_a)},
    ]

    if state is None:
        state = _state_skeleton(config, run_signature, split_meta, dynamic_buckets, static_layers, low_bucket_info)
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
    logger.log(f"Split: {split_meta}")
    logger.log(f"Dynamic buckets: {[bucket['name'] for bucket in dynamic_buckets]}")
    logger.log(f"Static layers (internal): {static_layers}")
    logger.log(f"Low bucket note: {low_bucket_info['note']}")

    for fold_spec in fold_specs:
        fold_name = str(fold_spec["name"])
        fold_state = _ensure_fold_state(state, fold_spec)
        if f"{fold_name}.partial" in state.get("completed_stages", []) and fold_state.get("held_out_test"):
            logger.log(f"Skipping completed fold: {fold_name} | using saved partial results.")
            continue

        validation_sample_ids = list(fold_spec["validation_indices"])
        test_samples = _subset_from_indices(samples, list(fold_spec["test_indices"]))
        validation_samples = _subset_from_indices(samples, validation_sample_ids)

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
