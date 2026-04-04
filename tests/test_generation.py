"""Tests for generation-side prompt/candidate boundary handling."""

from __future__ import annotations

import pytest

from src.generation import (
    _build_scoring_text_attempts,
    _find_continuation_token_index,
    _prepare_scoring_inputs,
)


class _FakeTensor:
    def __init__(self, token_count: int) -> None:
        self.shape = (1, token_count)

    def to(self, device: str) -> "_FakeTensor":
        return self


class _FallbackTokenizer:
    def __call__(
        self,
        text: str,
        return_tensors: str,
        add_special_tokens: bool,
        return_offsets_mapping: bool = False,
    ) -> dict[str, _FakeTensor]:
        del return_tensors, add_special_tokens
        if return_offsets_mapping:
            raise TypeError("offset mapping not supported")
        token_count = len([piece for piece in text.split(" ") if piece])
        return {
            "input_ids": _FakeTensor(token_count),
            "attention_mask": _FakeTensor(token_count),
        }


class _OffsetTokenizer:
    def __call__(
        self,
        text: str,
        return_tensors: str,
        add_special_tokens: bool,
        return_offsets_mapping: bool = False,
    ) -> dict[str, object]:
        del return_tensors, add_special_tokens
        if return_offsets_mapping:
            if text == "Answer: Paris":
                return {
                    "input_ids": _FakeTensor(2),
                    "attention_mask": _FakeTensor(2),
                    "offset_mapping": [[(0, 9), (9, 13)]],
                }
            return {
                "input_ids": _FakeTensor(1),
                "attention_mask": _FakeTensor(1),
                "offset_mapping": [[(0, len(text))]],
            }
        token_count = len([piece for piece in text.split(" ") if piece])
        return {
            "input_ids": _FakeTensor(token_count),
            "attention_mask": _FakeTensor(token_count),
        }


class _AlwaysMergedTokenizer:
    def __call__(
        self,
        text: str,
        return_tensors: str,
        add_special_tokens: bool,
        return_offsets_mapping: bool = False,
    ) -> dict[str, _FakeTensor]:
        del text, return_tensors, add_special_tokens, return_offsets_mapping
        return {
            "input_ids": _FakeTensor(1),
            "attention_mask": _FakeTensor(1),
        }


class _FakeModel:
    device = "cpu"



def test_build_scoring_text_attempts_adds_boundary_and_strips_leading_space() -> None:
    """Scoring should use an explicit prompt/candidate boundary when needed."""
    attempts = _build_scoring_text_attempts("Answer:", "  Paris", " ")

    assert attempts[0] == ("Answer: ", "Answer: Paris")



def test_find_continuation_token_index_uses_end_offset_for_cross_boundary_token() -> None:
    """A token that overlaps the prompt/candidate boundary still counts as continuation."""
    token_index = _find_continuation_token_index(
        offset_mapping=[(0, 9), (9, 13)],
        prompt_char_len=8,
    )

    assert token_index == 0



def test_prepare_scoring_inputs_prefers_offset_mapping_when_available() -> None:
    """Fast-tokenizer offsets should determine the continuation token start."""
    prompt_text, _, _, prompt_len = _prepare_scoring_inputs(
        model=_FakeModel(),
        tokenizer=_OffsetTokenizer(),
        prompt="Answer:",
        candidate_answer="Paris",
        separator=" ",
    )

    assert prompt_text == "Answer: "
    assert prompt_len == 0



def test_prepare_scoring_inputs_uses_fallback_boundary_when_offsets_are_unavailable() -> None:
    """A fallback prompt boundary should rescue tokenization-sensitive cases."""
    prompt_text, _, _, prompt_len = _prepare_scoring_inputs(
        model=_FakeModel(),
        tokenizer=_FallbackTokenizer(),
        prompt="Answer:\n",
        candidate_answer="Paris",
        separator=" ",
    )

    assert prompt_text == "Answer: "
    assert prompt_len == 1



def test_prepare_scoring_inputs_raises_after_fallback_failure() -> None:
    """The helper should still fail clearly if no boundary adds continuation tokens."""
    with pytest.raises(ValueError, match="scoring-boundary fallback"):
        _prepare_scoring_inputs(
            model=_FakeModel(),
            tokenizer=_AlwaysMergedTokenizer(),
            prompt="Answer:",
            candidate_answer="Paris",
            separator=" ",
        )
