from __future__ import annotations

from scripts.hf_revalidate_paper_bucket_truthfulqa_cv import (
    _classify_revalidation,
    _select_dynamic_by_mc3_only,
)


def test_select_dynamic_by_mc3_only_marks_exact_tie_without_secondary_metrics() -> None:
    choice = _select_dynamic_by_mc3_only(
        [
            {"name": "paper_high_16_32", "summary": {"dola_avg_mc3": 0.4, "dola_avg_mc2": 0.9, "dola_avg_mc1": 0.9}},
            {"name": "paper_low_0_16", "summary": {"dola_avg_mc3": 0.4, "dola_avg_mc2": 0.1, "dola_avg_mc1": 0.1}},
        ]
    )

    assert choice["is_tie"] is True
    assert choice["selected_bucket"] is None
    assert choice["tied_bucket_names"] == ["paper_high_16_32", "paper_low_0_16"]


def test_classify_revalidation_reports_approximate_high_bucket_hold() -> None:
    label = _classify_revalidation(
        fold_reports=[
            {"selected_dynamic_bucket": {"name": "paper_high_16_32"}},
            {"selected_dynamic_bucket": {"name": "paper_high_16_32"}},
        ],
        low_bucket_embedding_inclusive=False,
    )

    assert label == "LIKELY_PAPER_HIGH_BUCKET_HOLDS_BUT_LOW_TEST_IS_APPROXIMATE"
