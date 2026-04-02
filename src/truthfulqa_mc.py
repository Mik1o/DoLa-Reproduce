"""TruthfulQA multiple-choice data loading and prompt construction helpers."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


QUESTION_FIELDS = ("question", "Question")
BEST_ANSWER_FIELDS = ("best_answer", "Best Answer")
CORRECT_ANSWER_FIELDS = ("correct_answers", "Correct Answers")
INCORRECT_ANSWER_FIELDS = ("incorrect_answers", "Incorrect Answers")
CATEGORY_FIELDS = ("category", "Category")
OPTION_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


@dataclass(slots=True)
class TruthfulQASample:
    """Normalized TruthfulQA multiple-choice sample."""

    question: str
    best_answer: str
    correct_answers: list[str]
    incorrect_answers: list[str]
    category: str | None


def load_truthfulqa_csv(csv_path: str | Path) -> pd.DataFrame:
    """Load a TruthfulQA-style CSV file into a DataFrame."""
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"TruthfulQA CSV file not found: {path}")
    return pd.read_csv(path)


def parse_list_field(raw: str | None) -> list[str]:
    """Parse a TruthfulQA list-like field into a clean list of strings.

    The parser accepts common list-literal strings such as ``['a', 'b']`` as
    well as simple delimiter-based forms like ``a; b``.
    """
    if _is_missing(raw):
        return []
    if not isinstance(raw, str):
        raise TypeError(f"Expected a string-like list field, but received {type(raw)!r}.")

    text = raw.strip()
    if not text:
        return []

    parsed_literal = _parse_literal_list(text)
    if parsed_literal is not None:
        return parsed_literal

    for delimiter in (";", "|", "\n"):
        if delimiter in text:
            return _split_and_clean(text.split(delimiter))

    return [text]


def normalize_truthfulqa_row(row: pd.Series) -> TruthfulQASample:
    """Normalize a CSV row into a lightweight internal TruthfulQA sample."""
    question = _get_required_text(row, QUESTION_FIELDS)
    best_answer = _get_required_text(row, BEST_ANSWER_FIELDS)
    correct_answers = parse_list_field(_get_optional_value(row, CORRECT_ANSWER_FIELDS))
    incorrect_answers = parse_list_field(_get_optional_value(row, INCORRECT_ANSWER_FIELDS))
    category = _get_optional_text(row, CATEGORY_FIELDS)

    if not correct_answers:
        correct_answers = [best_answer]
    elif best_answer not in correct_answers:
        correct_answers = [best_answer, *correct_answers]

    if not incorrect_answers:
        raise ValueError(
            f"Row {row.name!r} is missing incorrect answers required for MC prompts."
        )

    return TruthfulQASample(
        question=question,
        best_answer=best_answer,
        correct_answers=_dedupe_preserve_order(correct_answers),
        incorrect_answers=_dedupe_preserve_order(incorrect_answers),
        category=category,
    )


def load_truthfulqa_samples(csv_path: str | Path) -> list[TruthfulQASample]:
    """Load and normalize all samples from a TruthfulQA-style CSV file."""
    dataframe = load_truthfulqa_csv(csv_path)
    return [normalize_truthfulqa_row(row) for _, row in dataframe.iterrows()]


def build_mc_prompt(sample: TruthfulQASample) -> str:
    """Build a simple multiple-choice prompt for manual inspection or future runs."""
    options = _dedupe_preserve_order([sample.best_answer, *sample.incorrect_answers])
    if len(options) > len(OPTION_LABELS):
        raise ValueError("Too many answer options for the current prompt label set.")

    lines = [
        "Answer the following multiple-choice question by selecting the best option.",
        f"Question: {sample.question}",
        "Options:",
    ]
    for label, option in zip(OPTION_LABELS, options, strict=False):
        lines.append(f"{label}. {option}")
    lines.append("Answer:")
    return "\n".join(lines)


def _get_required_text(row: pd.Series, field_names: tuple[str, ...]) -> str:
    """Return a required text field from a row or raise a clear error."""
    value = _get_optional_value(row, field_names)
    if _is_missing(value):
        field_list = ", ".join(field_names)
        raise ValueError(f"Row {row.name!r} is missing required field(s): {field_list}")
    if not isinstance(value, str):
        value = str(value)
    text = value.strip()
    if not text:
        field_list = ", ".join(field_names)
        raise ValueError(f"Row {row.name!r} has an empty required field: {field_list}")
    return text


def _get_optional_text(row: pd.Series, field_names: tuple[str, ...]) -> str | None:
    """Return an optional text field from a row."""
    value = _get_optional_value(row, field_names)
    if _is_missing(value):
        return None
    if not isinstance(value, str):
        value = str(value)
    text = value.strip()
    return text or None


def _get_optional_value(row: pd.Series, field_names: tuple[str, ...]) -> Any:
    """Return the first matching field value from a row."""
    for field_name in field_names:
        if field_name in row.index:
            return row[field_name]
    return None


def _parse_literal_list(text: str) -> list[str] | None:
    """Parse a Python-style list literal if present."""
    is_literal = (
        (text.startswith("[") and text.endswith("]"))
        or (text.startswith("(") and text.endswith(")"))
    )
    if not is_literal:
        return None

    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return None

    if isinstance(parsed, str):
        return [parsed.strip()] if parsed.strip() else []
    if isinstance(parsed, (list, tuple)):
        return _split_and_clean(str(item) for item in parsed)
    return None


def _split_and_clean(items: Any) -> list[str]:
    """Strip whitespace and drop empty values from a sequence of strings."""
    cleaned: list[str] = []
    for item in items:
        text = str(item).strip()
        if text:
            cleaned.append(text)
    return cleaned


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    """Deduplicate a list of strings while preserving order."""
    seen: set[str] = set()
    deduped: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped


def _is_missing(value: object) -> bool:
    """Return True when a scalar field should be treated as missing."""
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    return bool(pd.isna(value))