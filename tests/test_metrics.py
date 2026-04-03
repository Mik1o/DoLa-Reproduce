"""Tests for TruthfulQA multiple-choice metrics."""

from __future__ import annotations

import math

import pytest

from src.metrics import (
    aggregate_mc_metrics,
    compare_aggregate_metrics,
    compute_mc1,
    compute_mc2,
    compute_mc3,
    compute_mc_metrics,
    select_best_layer_summary,
)



def test_compute_mc1_uses_best_true_against_all_false() -> None:
    """MC1 should check whether the best true answer beats every false answer."""
    assert compute_mc1([1.5, 0.2], [1.4, 0.1]) == 1.0
    assert compute_mc1([1.5, 2.0], [1.6, 0.1]) == 0.0



def test_compute_mc2_returns_true_probability_mass() -> None:
    """MC2 should sum softmax mass over all true answers."""
    result = compute_mc2([0.0, math.log(2.0)], [math.log(3.0)])
    assert result == pytest.approx(0.5)



def test_compute_mc3_returns_fraction_of_true_answers_above_false_max() -> None:
    """MC3 should count how many true answers beat every false answer."""
    assert compute_mc3([2.0, 0.5, -1.0], [1.0, 0.0]) == pytest.approx(1 / 3)



def test_compute_mc_metrics_collects_all_metric_values() -> None:
    """Combined metric computation should expose MC1, MC2, and MC3."""
    metrics = compute_mc_metrics([2.0, 0.0], [1.0, -1.0])

    assert metrics["MC1"] == 1.0
    assert metrics["MC2"] == pytest.approx(0.73105858)
    assert metrics["MC3"] == 0.5



def test_aggregate_mc_metrics_returns_simple_averages() -> None:
    """Aggregation should average each MC metric across samples."""
    summary = aggregate_mc_metrics(
        [
            {"MC1": 1.0, "MC2": 0.8, "MC3": 0.5},
            {"MC1": 0.0, "MC2": 0.4, "MC3": 1.0},
        ]
    )

    assert summary["avg_mc1"] == 0.5
    assert summary["avg_mc2"] == pytest.approx(0.6)
    assert summary["avg_mc3"] == pytest.approx(0.75)
    assert summary["num_samples"] == 2



def test_compare_aggregate_metrics_returns_vanilla_dola_and_deltas() -> None:
    """Comparison aggregation should expose both summaries and their deltas."""
    summary = compare_aggregate_metrics(
        {"avg_mc1": 0.4, "avg_mc2": 0.5, "avg_mc3": 0.6, "num_samples": 3},
        {"avg_mc1": 0.5, "avg_mc2": 0.7, "avg_mc3": 0.4, "num_samples": 3},
    )

    assert summary["vanilla_avg_mc1"] == pytest.approx(0.4)
    assert summary["vanilla_avg_mc2"] == pytest.approx(0.5)
    assert summary["vanilla_avg_mc3"] == pytest.approx(0.6)
    assert summary["dola_avg_mc1"] == pytest.approx(0.5)
    assert summary["dola_avg_mc2"] == pytest.approx(0.7)
    assert summary["dola_avg_mc3"] == pytest.approx(0.4)
    assert summary["delta_mc1"] == pytest.approx(0.1)
    assert summary["delta_mc2"] == pytest.approx(0.2)
    assert summary["delta_mc3"] == pytest.approx(-0.2)
    assert summary["num_samples"] == 3



def test_select_best_layer_summary_prefers_higher_dola_mc2_then_mc1() -> None:
    """Layer selection should rank by DoLa MC2, then DoLa MC1."""
    best = select_best_layer_summary(
        [
            {"premature_layer": 2, "dola_avg_mc2": 0.75, "dola_avg_mc1": 0.4},
            {"premature_layer": 4, "dola_avg_mc2": 0.75, "dola_avg_mc1": 0.6},
            {"premature_layer": 8, "dola_avg_mc2": 0.70, "dola_avg_mc1": 1.0},
        ]
    )

    assert best["premature_layer"] == 4



def test_metric_functions_reject_empty_inputs() -> None:
    """Metric functions should raise clear errors on empty score lists."""
    with pytest.raises(ValueError, match="scores_true"):
        compute_mc1([], [0.0])
    with pytest.raises(ValueError, match="scores_false"):
        compute_mc_metrics([0.0], [])
    with pytest.raises(ValueError, match="sample_metrics"):
        aggregate_mc_metrics([])
    with pytest.raises(ValueError, match="same num_samples"):
        compare_aggregate_metrics(
            {"avg_mc1": 0.4, "avg_mc2": 0.5, "avg_mc3": 0.6, "num_samples": 2},
            {"avg_mc1": 0.5, "avg_mc2": 0.7, "avg_mc3": 0.4, "num_samples": 3},
        )
    with pytest.raises(ValueError, match="layer_summaries"):
        select_best_layer_summary([])