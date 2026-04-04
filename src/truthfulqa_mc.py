"""TruthfulQA multiple-choice data loading and prompt construction helpers."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


QUESTION_FIELDS = ("question", "Question", "Question Text")
BEST_ANSWER_FIELDS = ("best_answer", "Best Answer", "BestAnswer")
CORRECT_ANSWER_FIELDS = ("correct_answers", "Correct Answers", "CorrectAnswer")
INCORRECT_ANSWER_FIELDS = (
    "incorrect_answers",
    "Incorrect Answers",
    "IncorrectAnswer",
)
CATEGORY_FIELDS = ("category", "Category")
OPTION_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_REQUIRED_COLUMN_GROUPS = {
    "question": QUESTION_FIELDS,
    "best_answer": BEST_ANSWER_FIELDS,
    "incorrect_answers": INCORRECT_ANSWER_FIELDS,
}


@dataclass(slots=True)
class TruthfulQASample:
    question: str
    best_answer: str
    correct_answers: list[str]
    incorrect_answers: list[str]
    category: str | None


@dataclass(slots=True)
class TruthfulQAEvalResult:
    question: str
    prompt: str
    true_candidates: list[str]
    false_candidates: list[str]
    scores_true: list[float]
    scores_false: list[float]
    mc1: float
    mc2: float
    mc3: float



def load_truthfulqa_csv(csv_path: str | Path) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"TruthfulQA CSV file not found: {path}")

    dataframe = pd.read_csv(path)
    _validate_required_columns(dataframe.columns)
    return dataframe



def parse_list_field(raw: str | None, *, field_name: str = "list field") -> list[str]:
    if _is_missing(raw):
        return []
    if isinstance(raw, (list, tuple)):
        return _split_and_clean(raw)
    if not isinstance(raw, str):
        raise TypeError(
            f"Expected {field_name} to be string-like, but received {type(raw)!r}."
        )

    text = raw.strip()
    if not text:
        return []

    parsed_literal = _parse_literal_list(text)
    if parsed_literal is not None:
        return parsed_literal

    if _looks_like_literal_list(text):
        raise ValueError(f"Could not parse {field_name} as a list literal: {raw!r}")

    for delimiter in (";", "|", "\n"):
        if delimiter in text:
            return _split_and_clean(text.split(delimiter))

    return [text]



def normalize_truthfulqa_row(row: pd.Series) -> TruthfulQASample:
    question = _get_required_text(row, QUESTION_FIELDS, field_label="question")
    best_answer = _get_required_text(row, BEST_ANSWER_FIELDS, field_label="best_answer")

    try:
        correct_answers = parse_list_field(
            _get_optional_value(row, CORRECT_ANSWER_FIELDS),
            field_name="correct_answers",
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"Row {row.name!r} has invalid correct_answers: {error}") from error

    try:
        incorrect_answers = parse_list_field(
            _get_optional_value(row, INCORRECT_ANSWER_FIELDS),
            field_name="incorrect_answers",
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"Row {row.name!r} has invalid incorrect_answers: {error}") from error

    category = _get_optional_text(row, CATEGORY_FIELDS)

    if not correct_answers:
        correct_answers = [best_answer]
    elif best_answer not in correct_answers:
        correct_answers = [best_answer, *correct_answers]

    if not incorrect_answers:
        raise ValueError(
            f"Row {row.name!r} has empty incorrect_answers after parsing; "
            "multiple-choice prompts require at least one false answer."
        )

    return TruthfulQASample(
        question=question,
        best_answer=best_answer,
        correct_answers=_dedupe_preserve_order(correct_answers),
        incorrect_answers=_dedupe_preserve_order(incorrect_answers),
        category=category,
    )



def load_truthfulqa_samples(csv_path: str | Path) -> list[TruthfulQASample]:
    dataframe = load_truthfulqa_csv(csv_path)
    return [normalize_truthfulqa_row(row) for _, row in dataframe.iterrows()]



def get_mc_candidate_sets(sample: TruthfulQASample) -> tuple[list[str], list[str]]:
    true_candidates = _dedupe_preserve_order([sample.best_answer, *sample.correct_answers])
    false_candidates = _dedupe_preserve_order(sample.incorrect_answers)
    return true_candidates, false_candidates



def build_mc_prompt(sample: TruthfulQASample, prompt_style: str = "plain_mc") -> str:
    normalized_style = prompt_style.strip().lower()
    if normalized_style in {"plain_mc", "options_mc"}:
        return _build_options_mc_prompt(sample)
    if normalized_style == "direct_answer_mc":
        return _build_direct_answer_mc_prompt(sample)
    if normalized_style == "tinyllama_chat_mc":
        options_prompt = _build_options_mc_prompt(sample)
        return (
            "<|system|>\nYou are a helpful assistant.</s>\n"
            f"<|user|>\n{options_prompt}</s>\n"
            "<|assistant|>\n"
        )
    raise ValueError(
        "Unsupported prompt_style "
        f"'{prompt_style}'. Use 'options_mc', 'direct_answer_mc', 'plain_mc', or 'tinyllama_chat_mc'."
    )



def _build_options_mc_prompt(sample: TruthfulQASample) -> str:
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



def _build_direct_answer_mc_prompt(sample: TruthfulQASample) -> str:
    return "\n".join(
        [
            "Answer the following question truthfully and concisely.",
            f"Question: {sample.question}",
            "Answer:",
        ]
    )



def _validate_required_columns(columns: Iterable[object]) -> None:
    normalized_columns = {_normalize_field_name(str(column)) for column in columns}
    missing_labels = [
        field_label
        for field_label, aliases in _REQUIRED_COLUMN_GROUPS.items()
        if not any(_normalize_field_name(alias) in normalized_columns for alias in aliases)
    ]
    if missing_labels:
        raise ValueError(
            "TruthfulQA CSV is missing required column groups: "
            f"{', '.join(missing_labels)}. "
            f"Available columns: {', '.join(str(column) for column in columns)}"
        )



def _get_required_text(
    row: pd.Series,
    field_names: tuple[str, ...],
    *,
    field_label: str,
) -> str:
    value = _get_optional_value(row, field_names)
    if _is_missing(value):
        alias_list = ", ".join(field_names)
        raise ValueError(
            f"Row {row.name!r} is missing required field '{field_label}'. "
            f"Accepted column names: {alias_list}"
        )
    if not isinstance(value, str):
        value = str(value)
    text = value.strip()
    if not text:
        raise ValueError(f"Row {row.name!r} has an empty required field '{field_label}'.")
    return text



def _get_optional_text(row: pd.Series, field_names: tuple[str, ...]) -> str | None:
    value = _get_optional_value(row, field_names)
    if _is_missing(value):
        return None
    if not isinstance(value, str):
        value = str(value)
    text = value.strip()
    return text or None



def _get_optional_value(row: pd.Series, field_names: tuple[str, ...]) -> Any:
    matching_column = _find_matching_column(row.index, field_names)
    if matching_column is None:
        return None
    return row[matching_column]



def _find_matching_column(columns: Iterable[object], field_names: tuple[str, ...]) -> str | None:
    normalized_aliases = {_normalize_field_name(field_name) for field_name in field_names}
    for column in columns:
        column_name = str(column)
        if _normalize_field_name(column_name) in normalized_aliases:
            return column_name
    return None



def _normalize_field_name(field_name: str) -> str:
    return "".join(character.lower() for character in field_name if character.isalnum())



def _parse_literal_list(text: str) -> list[str] | None:
    if not _looks_like_literal_list(text):
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



def _looks_like_literal_list(text: str) -> bool:
    return (
        (text.startswith("[") and text.endswith("]"))
        or (text.startswith("(") and text.endswith(")"))
        or text.startswith("[")
        or text.startswith("(")
    )



def _split_and_clean(items: Any) -> list[str]:
    cleaned: list[str] = []
    for item in items:
        text = str(item).strip()
        if text:
            cleaned.append(text)
    return cleaned



def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped



def _is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    return bool(pd.isna(value))
