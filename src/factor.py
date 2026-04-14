"""Helpers for the FACTOR multiple-choice benchmark."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class FactorSample:
    prefix: str
    completion: str
    contradiction_0: str
    contradiction_1: str
    contradiction_2: str
    prefix_column: str


def resolve_factor_prefix_column(csv_path: str | Path) -> str:
    """Match the official FACTOR evaluator's prefix-column choice."""
    normalized = str(csv_path).lower()
    return "full_prefix" if "news" in normalized else "turncated_prefixes"


def load_factor_samples(csv_path: str | Path) -> list[FactorSample]:
    """Load FACTOR samples from the official CSV schema."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"FACTOR CSV not found: {path}")

    prefix_column = resolve_factor_prefix_column(path)
    required_columns = {
        prefix_column,
        "completion",
        "contradiction_0",
        "contradiction_1",
        "contradiction_2",
    }

    samples: list[FactorSample] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing_columns = sorted(required_columns - fieldnames)
        if missing_columns:
            raise ValueError(
                f"FACTOR CSV is missing required columns: {', '.join(missing_columns)}"
            )

        for row in reader:
            samples.append(
                FactorSample(
                    prefix=str(row[prefix_column]),
                    completion=str(row["completion"]),
                    contradiction_0=str(row["contradiction_0"]),
                    contradiction_1=str(row["contradiction_1"]),
                    contradiction_2=str(row["contradiction_2"]),
                    prefix_column=prefix_column,
                )
            )
    if not samples:
        raise ValueError(f"No FACTOR samples were loaded from {path}.")
    return samples


def build_factor_candidates(sample: FactorSample) -> tuple[str, list[str]]:
    """Return the official true completion and three contradictions."""
    return (
        " " + sample.completion,
        [
            " " + sample.contradiction_0,
            " " + sample.contradiction_1,
            " " + sample.contradiction_2,
        ],
    )


def compute_factor_is_correct(true_score: float, false_scores: list[float]) -> bool:
    """Mirror the official FACTOR correctness rule from factor_eval.py."""
    if not false_scores:
        raise ValueError("false_scores must contain at least one contradiction score.")
    return bool(all(true_score >= score for score in false_scores))


def aggregate_factor_accuracy(is_correct_rows: list[bool]) -> dict[str, float | int]:
    """Aggregate FACTOR correctness booleans into an accuracy summary."""
    if not is_correct_rows:
        raise ValueError("is_correct_rows must contain at least one item.")
    correct_count = sum(bool(item) for item in is_correct_rows)
    num_samples = len(is_correct_rows)
    return {
        "accuracy": correct_count / num_samples,
        "correct_count": correct_count,
        "num_samples": num_samples,
    }
