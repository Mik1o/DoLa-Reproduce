from __future__ import annotations

from pathlib import Path
import shutil

import pytest

from scripts.hf_tune_bucket_truthfulqa_cv import (
    ProgressTracker,
    RunLogger,
    _blend_eta_seconds_per_sample,
    _build_stage_plan,
)


def test_build_stage_plan_uses_expected_held_out_layer_count() -> None:
    stage_plan = _build_stage_plan(
        fold_specs=[{"name": "fold1", "validation_size": 418, "test_size": 399}],
        dynamic_buckets=[
            {"name": "official_16_32", "candidate_premature_layers": [15, 17, 19, 21, 23, 25, 27, 29]},
            {"name": "official_14_32", "candidate_premature_layers": [13, 15, 17, 19, 21, 23, 25, 27, 29]},
            {"name": "official_18_32", "candidate_premature_layers": [17, 19, 21, 23, 25, 27, 29]},
            {"name": "official_12_32", "candidate_premature_layers": [11, 13, 15, 17, 19, 21, 23, 25, 27, 29]},
        ],
        static_layers=[15, 17],
    )

    held_out_stage = next(stage for stage in stage_plan if stage["id"] == "fold1.test.shared")
    assert held_out_stage["layer_count"] == 9


def test_progress_tracker_uses_stage_family_specific_sample_rates() -> None:
    stage_plan = [
        {"id": "fold1.validation.shared", "label": "fold1 validation shared scoring", "kind": "shared", "samples": 418, "layer_count": 10, "weight": 4180.0},
        {"id": "fold1.test.shared", "label": "fold1 held-out shared scoring", "kind": "shared", "samples": 399, "layer_count": 9, "weight": 3591.0},
        {"id": "fold2.validation.shared", "label": "fold2 validation shared scoring", "kind": "shared", "samples": 399, "layer_count": 10, "weight": 3990.0},
        {"id": "fold2.test.shared", "label": "fold2 held-out shared scoring", "kind": "shared", "samples": 418, "layer_count": 9, "weight": 3762.0},
    ]
    output_dir = Path('tests/.tmp_progress_tracker_eta')
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tracker = ProgressTracker(
        stage_plan=stage_plan,
        output_dir=output_dir,
        logger=RunLogger(output_dir),
        completed_stage_ids={"fold1.validation.shared", "fold1.test.shared"},
        stage_durations={
            "fold1.validation.shared": 418.0 * 70.0,
            "fold1.test.shared": 399.0 * 33.0,
        },
    )

    try:
        assert tracker._estimate_stage_seconds(stage_plan[2]) == pytest.approx(399.0 * 70.0)
        assert tracker._estimate_stage_seconds(stage_plan[3]) == pytest.approx(418.0 * 33.0)
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)


def test_blend_eta_seconds_per_sample_damps_first_sample_spike() -> None:
    blended = _blend_eta_seconds_per_sample(
        prior_seconds_per_sample=70.0,
        observed_seconds_per_sample=125.0,
        completed_samples=1,
    )

    assert blended == pytest.approx(76.875)
    assert 70.0 < blended < 125.0
