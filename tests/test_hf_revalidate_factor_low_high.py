from __future__ import annotations

from scripts.hf_revalidate_factor_low_high import (
    _classify_revalidation,
    _select_dynamic_bucket_by_accuracy,
)


def test_select_dynamic_bucket_by_accuracy_marks_exact_tie_without_tiebreak() -> None:
    choice = _select_dynamic_bucket_by_accuracy(
        [
            {"name": "paper_low_0_16", "summary": {"accuracy": 0.75, "correct_count": 100}},
            {"name": "paper_high_16_32", "summary": {"accuracy": 0.75, "correct_count": 1}},
        ]
    )

    assert choice["is_tie"] is True
    assert choice["selected_bucket"] is None
    assert choice["tied_bucket_names"] == ["paper_high_16_32", "paper_low_0_16"]


def test_classify_revalidation_reports_low_direction_hold() -> None:
    label = _classify_revalidation(
        [
            {"validation": {"selection": {"selected_bucket_name": "paper_low_0_16"}}},
            {"validation": {"selection": {"selected_bucket_name": "paper_low_0_16"}}},
        ]
    )

    assert label == "LIKELY_FACTOR_LOW_DIRECTION_HOLDS"


def test_classify_revalidation_reports_high_direction_hold() -> None:
    label = _classify_revalidation(
        [
            {"validation": {"selection": {"selected_bucket_name": "paper_high_16_32"}}},
            {"validation": {"selection": {"selected_bucket_name": "paper_high_16_32"}}},
        ]
    )

    assert label == "LIKELY_FACTOR_HIGH_DIRECTION_BETTER_FOR_LLAMA31"


def test_classify_revalidation_reports_split_when_folds_disagree() -> None:
    label = _classify_revalidation(
        [
            {"validation": {"selection": {"selected_bucket_name": "paper_low_0_16"}}},
            {"validation": {"selection": {"selected_bucket_name": "paper_high_16_32"}}},
        ]
    )

    assert label == "LIKELY_FACTOR_SPLIT_DEPENDENT_NO_SINGLE_DIRECTION"
