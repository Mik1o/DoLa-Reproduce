"""Metric helpers for future TruthfulQA multiple-choice evaluation."""

from __future__ import annotations

from typing import Any


def compute_mc_metrics(predictions: list[Any], references: list[Any]) -> dict[str, float]:
    """Compute placeholder multiple-choice metrics.

    The scaffold currently returns an empty metric dictionary so scripts can
    keep a stable interface while the real evaluator is still pending.
    """
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length.")
    return {}


def format_metrics(metrics: dict[str, float]) -> str:
    """Format metric values for lightweight logging."""
    if not metrics:
        return "No metrics computed yet."
    return ", ".join(f"{key}={value:.4f}" for key, value in metrics.items())
