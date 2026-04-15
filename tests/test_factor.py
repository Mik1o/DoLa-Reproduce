from __future__ import annotations

from pathlib import Path

from scripts.hf_run_factor_paper_low_baseline import _classify_factor_baseline
from src.factor import (
    aggregate_factor_accuracy,
    build_factor_candidates,
    compute_factor_is_correct,
    load_factor_samples,
    resolve_factor_prefix_column,
)


def test_resolve_factor_prefix_column_matches_official_policy() -> None:
    assert resolve_factor_prefix_column("data/factor/wiki_factor.csv") == "turncated_prefixes"
    assert resolve_factor_prefix_column("data/factor/news_factor.csv") == "full_prefix"


def test_load_factor_samples_uses_turncated_prefixes_for_wiki() -> None:
    csv_path = Path(__file__).resolve().parent / "fixtures" / "wiki_factor_sample.csv"
    samples = load_factor_samples(csv_path)

    assert len(samples) == 1
    assert samples[0].prefix == "truncated context"
    assert samples[0].prefix_column == "turncated_prefixes"


def test_build_factor_candidates_adds_official_leading_space() -> None:
    from src.factor import FactorSample

    item = FactorSample(
        prefix="ctx",
        completion="true ending",
        contradiction_0="false a",
        contradiction_1="false b",
        contradiction_2="false c",
        prefix_column="turncated_prefixes",
    )

    true_candidate, false_candidates = build_factor_candidates(item)

    assert true_candidate == " true ending"
    assert false_candidates == [" false a", " false b", " false c"]


def test_compute_factor_is_correct_treats_ties_as_correct_like_official_eval() -> None:
    assert compute_factor_is_correct(1.0, [0.5, 1.0, -0.2]) is True
    assert compute_factor_is_correct(0.9, [1.0, 0.1, 0.2]) is False


def test_aggregate_factor_accuracy_reports_accuracy_and_counts() -> None:
    summary = aggregate_factor_accuracy([True, False, True])

    assert summary == {"accuracy": 2 / 3, "correct_count": 2, "num_samples": 3}


def test_classify_factor_baseline_marks_positive_approximate_transfer() -> None:
    label = _classify_factor_baseline(
        embedding_inclusive=False,
        vanilla_accuracy=0.40,
        dynamic_accuracy=0.52,
    )

    assert label == "LIKELY_FACTOR_LOW_BUCKET_ONLY_APPROXIMATE"


def test_classify_factor_baseline_marks_positive_paper_faithful_transfer() -> None:
    label = _classify_factor_baseline(
        embedding_inclusive=True,
        vanilla_accuracy=0.40,
        dynamic_accuracy=0.52,
    )

    assert label == "LIKELY_FACTOR_PAPER_LOW_BUCKET_HOLDS"


def test_classify_factor_baseline_requires_embedding_when_approximate_transfer_fails() -> None:
    label = _classify_factor_baseline(
        embedding_inclusive=False,
        vanilla_accuracy=0.52,
        dynamic_accuracy=0.40,
    )

    assert label == "LIKELY_NEED_EMBEDDING_SUPPORT_FIRST"
