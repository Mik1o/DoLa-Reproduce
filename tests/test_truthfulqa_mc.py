"""Tests for TruthfulQA multiple-choice data normalization helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.truthfulqa_mc import (
    build_mc_prompt,
    load_truthfulqa_samples,
    parse_list_field,
)


FIXTURE_CSV = Path(__file__).resolve().parent / "fixtures" / "truthfulqa_sample.csv"
HEADER_VARIANTS_CSV = Path(__file__).resolve().parent / "fixtures" / "truthfulqa_header_variants.csv"



def test_parse_list_field_accepts_common_formats() -> None:
    """The parser should handle literal and delimiter-based list strings."""
    assert parse_list_field("['A', 'B']") == ["A", "B"]
    assert parse_list_field("Alpha; Beta; Gamma") == ["Alpha", "Beta", "Gamma"]
    assert parse_list_field("") == []



def test_parse_list_field_rejects_malformed_literal_lists() -> None:
    """Broken list-literal strings should fail with a clearer message."""
    with pytest.raises(ValueError, match="Could not parse incorrect_answers"):
        parse_list_field("['A', 'B'", field_name="incorrect_answers")



def test_load_truthfulqa_samples_normalizes_rows() -> None:
    """Fixture rows should load into the expected internal sample structure."""
    samples = load_truthfulqa_samples(FIXTURE_CSV)

    assert len(samples) == 3
    assert samples[0].question == "What is the capital of France?"
    assert samples[0].best_answer == "Paris"
    assert samples[0].correct_answers == [
        "Paris",
        "The capital of France is Paris",
    ]
    assert samples[1].category is None
    assert samples[2].correct_answers == ["366"]
    assert samples[2].incorrect_answers == ["365", "364", "360"]



def test_load_truthfulqa_samples_accepts_realistic_column_variants() -> None:
    """Lowercase and underscore-style column names should also load cleanly."""
    samples = load_truthfulqa_samples(HEADER_VARIANTS_CSV)

    assert len(samples) == 1
    assert samples[0].best_answer == "4"
    assert samples[0].correct_answers == ["4", "four"]
    assert samples[0].incorrect_answers == ["3", "5"]



def test_build_mc_prompt_contains_question_and_options() -> None:
    """Prompt construction should include the question and labeled options."""
    sample = load_truthfulqa_samples(FIXTURE_CSV)[0]
    prompt = build_mc_prompt(sample)

    assert "Question: What is the capital of France?" in prompt
    assert "A. Paris" in prompt
    assert "B. London" in prompt
    assert prompt.rstrip().endswith("Answer:")



def test_build_mc_prompt_supports_tinyllama_chat_style() -> None:
    """A small chat-style prompt variant should be available by config."""
    sample = load_truthfulqa_samples(FIXTURE_CSV)[0]
    prompt = build_mc_prompt(sample, prompt_style="tinyllama_chat_mc")

    assert "<|system|>" in prompt
    assert "<|user|>" in prompt
    assert "<|assistant|>" in prompt
    assert "Question: What is the capital of France?" in prompt
