"""Tests for generation-side prompt/candidate boundary handling."""

from __future__ import annotations

import pytest

from src.generation import (
    _aggregate_continuation_log_probs,
    _build_scoring_text_attempts,
    _count_premature_layer_usage,
    _find_continuation_token_index,
    _normalize_dola_score_mode,
    _prepare_scoring_inputs,
    score_candidate_answers_dola_with_details,
    score_candidate_answers_multi_config_with_details,
    score_candidate_answers_with_details,
)
from src.metrics import compute_mc_metrics


class _FakeTensor:
    def __init__(self, token_count: int) -> None:
        self.shape = (1, token_count)

    def to(self, device: str) -> "_FakeTensor":
        return self


class _FakeLogProbTensor:
    def __init__(self, values: list[float]) -> None:
        self._values = values
        self.shape = (1, len(values))

    def sum(self) -> "_FakeScalar":
        return _FakeScalar(sum(self._values))


class _FakeScalar:
    def __init__(self, value: float) -> None:
        self._value = value

    def item(self) -> float:
        return self._value


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



def test_build_scoring_text_attempts_preserves_explicit_leading_space() -> None:
    """Official-aligned continuations should keep the intentional leading space."""
    attempts = _build_scoring_text_attempts("A:", " Paris", " ")

    assert attempts[0] == ("A:", "A: Paris")



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



def test_aggregate_continuation_log_probs_sum_mode_preserves_old_behavior() -> None:
    """The default score mode should still use raw summed log-probabilities."""
    score, token_count = _aggregate_continuation_log_probs(
        _FakeLogProbTensor([-1.0, -2.0, -3.0]),
        "sum_logprob",
    )

    assert score == -6.0
    assert token_count == 3



def test_aggregate_continuation_log_probs_mean_mode_normalizes_by_token_count() -> None:
    """The mean mode should divide by the scored continuation token count."""
    score, token_count = _aggregate_continuation_log_probs(
        _FakeLogProbTensor([-1.0, -2.0, -3.0]),
        "mean_logprob",
    )

    assert score == -2.0
    assert token_count == 3



def test_count_premature_layer_usage_returns_selected_layer_counts() -> None:
    """Dynamic DoLa should expose how often each candidate premature layer was chosen."""
    usage = _count_premature_layer_usage([15, 17, 15, 29], [15, 17, 29])

    assert usage == {15: 2, 17: 1, 29: 1}



def test_normalize_dola_score_mode_accepts_official_modes() -> None:
    """Official-aligned DoLa score modes should be recognized."""
    assert _normalize_dola_score_mode("official_static_dola") == "official_static_dola"
    assert _normalize_dola_score_mode("official_dynamic_dola") == "official_dynamic_dola"


class _ToyTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def __init__(self) -> None:
        self._vocab = {
            "<pad>": 0,
            "Answer:": 1,
            "Paris": 2,
            "London": 3,
            "Rome": 4,
            "Berlin": 5,
        }

    def __call__(
        self,
        text: str,
        return_tensors: str,
        add_special_tokens: bool,
        return_offsets_mapping: bool = False,
    ):
        import torch

        del return_tensors, add_special_tokens
        if return_offsets_mapping:
            raise TypeError("offset mapping not supported")
        token_ids = [self._vocab[token] for token in text.split(" ") if token]
        input_ids = torch.tensor([token_ids], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class _ToyOutput:
    def __init__(self, logits, hidden_states):
        self.logits = logits
        self.hidden_states = hidden_states


class _ToyModel:
    def __init__(self) -> None:
        import torch
        from types import SimpleNamespace

        self.device = torch.device("cpu")
        self.config = SimpleNamespace(num_hidden_layers=3)
        self.embedding = torch.nn.Embedding(6, 6)
        self.lm_head = torch.nn.Linear(6, 6, bias=False)
        with torch.no_grad():
            self.embedding.weight.copy_(torch.eye(6, dtype=torch.float32))
            self.lm_head.weight.copy_(torch.tensor([
                [0.9, 0.1, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.8, 0.1, 0.1, 0.0, 0.0],
                [0.2, 0.1, 0.9, 0.1, 0.0, 0.0],
                [0.1, 0.2, 0.1, 0.9, 0.0, 0.0],
                [0.0, 0.0, 0.1, 0.0, 0.9, 0.1],
                [0.0, 0.0, 0.0, 0.1, 0.1, 0.9],
            ], dtype=torch.float32))

    def parameters(self):
        return iter([self.embedding.weight])

    def get_output_embeddings(self):
        return self.lm_head

    def __call__(self, input_ids, attention_mask=None, output_hidden_states=False):
        del attention_mask
        hidden = self.embedding(input_ids)
        hidden_states = [hidden]
        layer_scales = (0.90, 1.05, 1.20)
        for scale in layer_scales:
            hidden = hidden * scale + scale * 0.01
            hidden_states.append(hidden)
        logits = self.lm_head(hidden_states[-1])
        return _ToyOutput(logits=logits, hidden_states=tuple(hidden_states) if output_hidden_states else None)


def test_multi_config_scoring_matches_single_candidate_paths() -> None:
    torch = pytest.importorskip("torch")
    del torch
    model = _ToyModel()
    tokenizer = _ToyTokenizer()
    prompt = "Answer:"
    true_candidates = [" Paris", " London"]
    false_candidates = [" Rome", " Berlin"]
    all_candidates = true_candidates + false_candidates

    vanilla = score_candidate_answers_with_details(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        candidate_answers=all_candidates,
        score_mode="sum_logprob",
    )
    static = score_candidate_answers_dola_with_details(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        candidate_answers=all_candidates,
        premature_layer=0,
        score_mode="sum_logprob",
        dola_score_mode="official_static_dola",
        mature_layer=2,
    )
    dynamic = score_candidate_answers_dola_with_details(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        candidate_answers=all_candidates,
        premature_layer=0,
        score_mode="sum_logprob",
        dola_score_mode="official_dynamic_dola",
        candidate_premature_layers=[0, 1],
        mature_layer=2,
    )
    multi = score_candidate_answers_multi_config_with_details(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        candidate_answers=all_candidates,
        static_layers=[0],
        dynamic_buckets={"bucket": [0, 1]},
        score_mode="sum_logprob",
        mature_layer=2,
        candidate_batch_size=2,
    )

    assert multi.batch_count == 2

    for expected, actual in zip(vanilla, multi.vanilla):
        assert actual.candidate == expected.candidate
        assert actual.continuation_token_count == expected.continuation_token_count
        assert actual.score == pytest.approx(expected.score)

    for expected, actual in zip(static, multi.static[0]):
        assert actual.candidate == expected.candidate
        assert actual.continuation_token_count == expected.continuation_token_count
        assert actual.score == pytest.approx(expected.score)
        assert actual.premature_layer_dist == expected.premature_layer_dist

    for expected, actual in zip(dynamic, multi.dynamic["bucket"]):
        assert actual.candidate == expected.candidate
        assert actual.continuation_token_count == expected.continuation_token_count
        assert actual.score == pytest.approx(expected.score)
        assert actual.premature_layer_dist == expected.premature_layer_dist

    expected_static_metrics = compute_mc_metrics(
        [item.score for item in static[: len(true_candidates)]],
        [item.score for item in static[len(true_candidates) :]],
    )
    actual_static_metrics = compute_mc_metrics(
        [item.score for item in multi.static[0][: len(true_candidates)]],
        [item.score for item in multi.static[0][len(true_candidates) :]],
    )
    expected_dynamic_metrics = compute_mc_metrics(
        [item.score for item in dynamic[: len(true_candidates)]],
        [item.score for item in dynamic[len(true_candidates) :]],
    )
    actual_dynamic_metrics = compute_mc_metrics(
        [item.score for item in multi.dynamic["bucket"][: len(true_candidates)]],
        [item.score for item in multi.dynamic["bucket"][len(true_candidates) :]],
    )

    assert actual_static_metrics == expected_static_metrics
    assert actual_dynamic_metrics == expected_dynamic_metrics
