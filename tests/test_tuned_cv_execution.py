from scripts.hf_tune_bucket_truthfulqa_cv import _build_stage_plan


def test_stage_plan_uses_union_layer_factor_for_shared_validation() -> None:
    fold_specs = [
        {"name": "fold1", "validation_size": 418, "test_size": 399},
        {"name": "fold2", "validation_size": 399, "test_size": 418},
    ]
    dynamic_buckets = [
        {"name": "official_16_32", "candidate_premature_layers": [15, 17, 19, 21, 23, 25, 27, 29]},
        {"name": "official_14_32", "candidate_premature_layers": [13, 15, 17, 19, 21, 23, 25, 27, 29]},
        {"name": "official_18_32", "candidate_premature_layers": [17, 19, 21, 23, 25, 27, 29]},
        {"name": "official_12_32", "candidate_premature_layers": [11, 13, 15, 17, 19, 21, 23, 25, 27, 29]},
    ]
    static_layers = [15, 17]

    plan = _build_stage_plan(fold_specs, dynamic_buckets, static_layers)
    validation_stage = next(stage for stage in plan if stage["id"] == "fold1.validation.shared")

    assert validation_stage["layer_count"] == 10
    assert validation_stage["weight"] == 4180.0
