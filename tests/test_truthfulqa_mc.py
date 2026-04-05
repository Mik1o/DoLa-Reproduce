"""Tests for TruthfulQA multiple-choice data normalization helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.truthfulqa_mc import (
    build_answer_continuation,
    build_mc_prompt,
    format_best_answer,
    load_truthfulqa_samples,
    parse_list_field,
    split_multi_answer,
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



def test_official_answer_formatting_matches_split_multi_answer_and_format_best() -> None:
    """Official-aligned answer normalization should split on semicolons and close with periods."""
    assert split_multi_answer("Paris; London") == ["Paris.", "London."]
    assert format_best_answer("Paris") == "Paris."
    assert format_best_answer("Already closed.") == "Already closed."



def test_load_truthfulqa_samples_normalizes_rows() -> None:
    """Fixture rows should load into the expected internal sample structure."""
    samples = load_truthfulqa_samples(FIXTURE_CSV)

    assert len(samples) == 3
    assert samples[0].question == "What is the capital of France?"
    assert samples[0].best_answer == "Paris."
    assert samples[0].correct_answers == [
        "Paris.",
        "The capital of France is Paris.",
    ]
    assert samples[1].category is None
    assert samples[2].correct_answers == ["366."]
    assert samples[2].incorrect_answers == ["365.", "364.", "360."]



def test_load_truthfulqa_samples_accepts_realistic_column_variants() -> None:
    """Lowercase and underscore-style column names should also load cleanly."""
    samples = load_truthfulqa_samples(HEADER_VARIANTS_CSV)

    assert len(samples) == 1
    assert samples[0].best_answer == "4."
    assert samples[0].correct_answers == ["4.", "four."]
    assert samples[0].incorrect_answers == ["3.", "5."]



def test_build_mc_prompt_options_mode_keeps_existing_behavior() -> None:
    """The options-style prompt should still expose the full answer list."""
    sample = load_truthfulqa_samples(FIXTURE_CSV)[0]
    prompt = build_mc_prompt(sample, prompt_style="options_mc")

    assert "Question: What is the capital of France?" in prompt
    assert "A. Paris." in prompt
    assert "B. London." in prompt
    assert prompt.rstrip().endswith("Answer:")



def test_build_mc_prompt_direct_mode_hides_candidates() -> None:
    """The direct-answer prompt should not expose answer choices in advance."""
    sample = load_truthfulqa_samples(FIXTURE_CSV)[0]
    prompt = build_mc_prompt(sample, prompt_style="direct_answer_mc")

    assert "Question: What is the capital of France?" in prompt
    assert "Answer:" in prompt
    assert "Paris." not in prompt
    assert "London." not in prompt
    assert "Options:" not in prompt



def test_build_mc_prompt_official_mode_matches_demo_structure() -> None:
    """The official-aligned prompt should contain instruction, demos, and Q/A cue."""
    sample = load_truthfulqa_samples(FIXTURE_CSV)[0]
    prompt = build_mc_prompt(sample, prompt_style="official_tfqa_mc")

    assert 'Interpret each question literally' in prompt
    assert prompt.count('Q: ') == 7
    assert 'Q: What is human life expectancy in the United States?' in prompt
    assert prompt.rstrip().endswith('A:')



def test_build_answer_continuation_uses_leading_space_for_official_mode() -> None:
    """The official-aligned scoring continuation should start with one space."""
    assert build_answer_continuation('Paris.', prompt_style='official_tfqa_mc') == ' Paris.'
    assert build_answer_continuation('Paris.', prompt_style='direct_answer_mc') == 'Paris.'



def test_build_mc_prompt_supports_tinyllama_chat_style() -> None:
    """A small chat-style prompt variant should be available by config."""
    sample = load_truthfulqa_samples(FIXTURE_CSV)[0]
    prompt = build_mc_prompt(sample, prompt_style="tinyllama_chat_mc")

    assert "<|system|>" in prompt
    assert "<|user|>" in prompt
    assert "<|assistant|>" in prompt
    assert "Question: What is the capital of France?" in prompt
