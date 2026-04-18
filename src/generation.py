"""Generation entry points for vanilla decoding and candidate scoring."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

from src.dola_utils import (
    get_mature_layer_index,
    internal_layer_to_hidden_state_index,
    validate_candidate_premature_layers,
    validate_mature_layer,
)


def _get_model_runtime_device(model: Any) -> Any:
    """Resolve the device that tokenized scoring inputs should be moved onto."""
    device = getattr(model, "device", None)
    if device is not None:
        return device
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration):
        return "cpu"


def _forward_hidden_states_only(
    model: Any,
    *,
    input_ids: Any,
    attention_mask: Any,
) -> Any:
    """Run a hidden-state-only forward to avoid allocating unused LM logits/cache."""
    base_model = getattr(model, "model", None)
    if base_model is not None:
        try:
            outputs = base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        except TypeError:
            outputs = base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
    else:
        try:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        except TypeError:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states is None:
        raise ValueError("The model did not return hidden_states for DoLa-style scoring.")
    return hidden_states


@dataclass(slots=True)
class CandidateScoreTrace:
    token_ids: list[int]
    token_texts: list[str]
    scoring_start_token_index: int
    token_scores: list[float]
    total_score: float
    avg_score: float
    final_token_scores: list[float] | None = None
    premature_token_scores: list[float] | None = None
    contrast_token_scores: list[float] | None = None
    selected_premature_layers: list[int] | None = None
    token_selected_mask: list[bool] | None = None
    token_selected_reason: list[str] | None = None
    token_effective_score_source: list[str] | None = None
    token_contrast_weight: list[float] | None = None
    token_selection_tier: list[str] | None = None


@dataclass(slots=True)
class CandidateScore:
    candidate: str
    score: float
    continuation_token_count: int
    premature_layer_dist: dict[int, int] | None = None
    trace: CandidateScoreTrace | None = None


@dataclass(slots=True)
class MultiConfigScoreResult:
    vanilla: list[CandidateScore]
    static: dict[int, list[CandidateScore]]
    dynamic: dict[str, list[CandidateScore]]
    batch_size: int
    batch_count: int
    profile: dict[str, Any] | None = None


TOKEN_SELECTIVE_DOLA_MODE_HEURISTIC_FACT_CRITICAL_V1 = "heuristic_fact_critical_v1"
TOKEN_SELECTIVE_DOLA_MODE_HEURISTIC_FACT_CRITICAL_V1_HARD = (
    "heuristic_fact_critical_v1_hard"
)
TOKEN_SELECTIVE_DOLA_MODE_HEURISTIC_FACT_CRITICAL_V2_SOFT = (
    "heuristic_fact_critical_v2_soft"
)
TOKEN_SELECTIVE_V2_STRONG_WEIGHT = 1.0
TOKEN_SELECTIVE_V2_MEDIUM_WEIGHT = 0.5
TOKEN_SELECTIVE_V2_UNSELECTED_WEIGHT = 0.2

_TOKEN_SELECTIVE_DATE_WORDS = {
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
}

_TOKEN_SELECTIVE_RELATION_WORDS = {
    "author",
    "born",
    "called",
    "capital",
    "city",
    "country",
    "died",
    "discovered",
    "invented",
    "king",
    "known",
    "located",
    "president",
    "queen",
    "state",
    "wrote",
}

_TOKEN_SELECTIVE_CAPITALIZED_EXCLUSIONS = {
    "A",
    "An",
    "And",
    "As",
    "At",
    "But",
    "For",
    "He",
    "Her",
    "His",
    "I",
    "In",
    "It",
    "It's",
    "No",
    "Of",
    "On",
    "Or",
    "She",
    "The",
    "They",
    "This",
    "To",
    "We",
    "What",
    "When",
    "Where",
    "Which",
    "Who",
    "Why",
    "Yes",
    "You",
}

_TOKEN_SELECTIVE_MEDIUM_STOPWORDS = {
    "about",
    "after",
    "also",
    "and",
    "because",
    "been",
    "before",
    "being",
    "between",
    "both",
    "but",
    "could",
    "does",
    "down",
    "each",
    "from",
    "have",
    "here",
    "into",
    "itself",
    "just",
    "more",
    "most",
    "only",
    "over",
    "same",
    "should",
    "some",
    "such",
    "than",
    "that",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "under",
    "very",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "with",
    "would",
    "your",
}


@dataclass(frozen=True, slots=True)
class TokenSelectiveDolaConfig:
    """Configuration for opt-in token-selective DoLa ablations."""

    strong_weight: float = TOKEN_SELECTIVE_V2_STRONG_WEIGHT
    medium_weight: float = TOKEN_SELECTIVE_V2_MEDIUM_WEIGHT
    unselected_weight: float = TOKEN_SELECTIVE_V2_UNSELECTED_WEIGHT
    selector_enable_number: bool = True
    selector_enable_date_word: bool = True
    selector_enable_relation_word: bool = True
    selector_enable_capitalized_lexical: bool = True
    selector_enable_strong_continuation_inheritance: bool = True
    selector_enable_lowercase_medium: bool = True
    selector_enable_adjacent_support_medium: bool = True

    def __post_init__(self) -> None:
        _validate_token_selective_weight(self.strong_weight, "token_selective_strong_weight")
        _validate_token_selective_weight(self.medium_weight, "token_selective_medium_weight")
        _validate_token_selective_weight(self.unselected_weight, "token_selective_unselected_weight")

    @classmethod
    def from_mapping(cls, config: dict[str, Any]) -> "TokenSelectiveDolaConfig":
        """Build from YAML-style config keys while preserving v2 defaults."""
        return cls(
            strong_weight=float(config.get("token_selective_strong_weight", TOKEN_SELECTIVE_V2_STRONG_WEIGHT)),
            medium_weight=float(config.get("token_selective_medium_weight", TOKEN_SELECTIVE_V2_MEDIUM_WEIGHT)),
            unselected_weight=float(
                config.get("token_selective_unselected_weight", TOKEN_SELECTIVE_V2_UNSELECTED_WEIGHT)
            ),
            selector_enable_number=bool(config.get("selector_enable_number", True)),
            selector_enable_date_word=bool(config.get("selector_enable_date_word", True)),
            selector_enable_relation_word=bool(config.get("selector_enable_relation_word", True)),
            selector_enable_capitalized_lexical=bool(config.get("selector_enable_capitalized_lexical", True)),
            selector_enable_strong_continuation_inheritance=bool(
                config.get("selector_enable_strong_continuation_inheritance", True)
            ),
            selector_enable_lowercase_medium=bool(config.get("selector_enable_lowercase_medium", True)),
            selector_enable_adjacent_support_medium=bool(
                config.get("selector_enable_adjacent_support_medium", True)
            ),
        )

    def to_config_dict(self) -> dict[str, bool | float]:
        """Serialize using the YAML key names used by experiment configs."""
        return {
            "token_selective_strong_weight": float(self.strong_weight),
            "token_selective_medium_weight": float(self.medium_weight),
            "token_selective_unselected_weight": float(self.unselected_weight),
            "selector_enable_number": bool(self.selector_enable_number),
            "selector_enable_date_word": bool(self.selector_enable_date_word),
            "selector_enable_relation_word": bool(self.selector_enable_relation_word),
            "selector_enable_capitalized_lexical": bool(self.selector_enable_capitalized_lexical),
            "selector_enable_strong_continuation_inheritance": bool(
                self.selector_enable_strong_continuation_inheritance
            ),
            "selector_enable_lowercase_medium": bool(self.selector_enable_lowercase_medium),
            "selector_enable_adjacent_support_medium": bool(self.selector_enable_adjacent_support_medium),
        }


def _validate_token_selective_weight(value: float, name: str) -> None:
    if not math.isfinite(float(value)) or float(value) < 0.0:
        raise ValueError(f"{name} must be a finite non-negative float.")


def _resolve_token_selective_config(
    token_selective_config: TokenSelectiveDolaConfig | None,
) -> TokenSelectiveDolaConfig:
    return token_selective_config or TokenSelectiveDolaConfig()



def generate_vanilla(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = 32,
    do_sample: bool = False,
    **kwargs: Any,
) -> str:
    """Generate text with a minimal vanilla decoding path."""
    if not prompt.strip():
        raise ValueError("prompt must be a non-empty string.")

    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = encoded.to(_get_model_runtime_device(model))
    output_ids = model.generate(
        **encoded,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        pad_token_id=tokenizer.pad_token_id,
        **kwargs,
    )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)



def score_continuation_logprob(
    model: Any,
    tokenizer: Any,
    prompt: str,
    candidate_answer: str,
    separator: str = " ",
    score_mode: str = "sum_logprob",
) -> float:
    """Score a candidate continuation with configurable log-prob aggregation."""
    return score_continuation_details(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        candidate_answer=candidate_answer,
        separator=separator,
        score_mode=score_mode,
    ).score



def score_candidate_answers(
    model: Any,
    tokenizer: Any,
    prompt: str,
    candidate_answers: list[str],
    separator: str = " ",
    score_mode: str = "sum_logprob",
) -> list[tuple[str, float]]:
    """Score each candidate answer for a shared prompt."""
    return [
        (item.candidate, item.score)
        for item in score_candidate_answers_with_details(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            candidate_answers=candidate_answers,
            separator=separator,
            score_mode=score_mode,
        )
    ]



def score_candidate_answers_with_details(
    model: Any,
    tokenizer: Any,
    prompt: str,
    candidate_answers: list[str],
    separator: str = " ",
    score_mode: str = "sum_logprob",
    return_trace: bool = False,
) -> list[CandidateScore]:
    """Score each candidate answer and keep continuation token counts."""
    if not candidate_answers:
        raise ValueError("candidate_answers must contain at least one answer.")

    return [
        score_continuation_details(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            candidate_answer=candidate_answer,
            separator=separator,
            score_mode=score_mode,
            return_trace=return_trace,
        )
        for candidate_answer in candidate_answers
    ]



def score_continuation_details(
    model: Any,
    tokenizer: Any,
    prompt: str,
    candidate_answer: str,
    separator: str = " ",
    score_mode: str = "sum_logprob",
    return_trace: bool = False,
) -> CandidateScore:
    """Score one candidate continuation and return score plus token length."""
    try:
        import torch
    except ImportError as error:
        raise ImportError(
            "torch is required for candidate scoring. "
            "Install the extra model dependencies from requirements-model.txt."
        ) from error

    _, input_ids, attention_mask, prompt_len = _prepare_scoring_inputs(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        candidate_answer=candidate_answer,
        separator=separator,
    )

    with torch.inference_mode():
        try:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        except TypeError:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1, :]
        token_log_probs = _gather_token_log_probs(logits, input_ids)

    continuation_start = _get_continuation_start_index(prompt_len)
    continuation_log_probs = token_log_probs[:, continuation_start:]
    score, continuation_token_count = _aggregate_continuation_log_probs(
        continuation_log_probs=continuation_log_probs,
        score_mode=score_mode,
    )
    trace = None
    if return_trace:
        trace = _build_candidate_score_trace(
            tokenizer=tokenizer,
            input_ids=input_ids,
            scoring_score_start_index=continuation_start,
            token_scores=continuation_log_probs,
        )
    return CandidateScore(
        candidate=candidate_answer,
        score=score,
        continuation_token_count=continuation_token_count,
        trace=trace,
    )



def score_continuation_dola_logprob(
    model: Any,
    tokenizer: Any,
    prompt: str,
    candidate_answer: str,
    premature_layer: int,
    separator: str = " ",
    score_mode: str = "sum_logprob",
    dola_score_mode: str = "legacy_contrastive",
    post_softmax: bool = False,
    relative_top: float = 0.0,
    relative_top_value: float = -1000.0,
    candidate_premature_layers: list[int] | None = None,
    mature_layer: int | None = None,
    enable_token_selective_dola: bool = False,
    token_selective_mode: str = TOKEN_SELECTIVE_DOLA_MODE_HEURISTIC_FACT_CRITICAL_V1,
    token_selective_config: TokenSelectiveDolaConfig | None = None,
) -> float:
    """Score a continuation with configurable DoLa-style scoring."""
    return score_continuation_dola_details(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        candidate_answer=candidate_answer,
        premature_layer=premature_layer,
        separator=separator,
        score_mode=score_mode,
        dola_score_mode=dola_score_mode,
        post_softmax=post_softmax,
        relative_top=relative_top,
        relative_top_value=relative_top_value,
        candidate_premature_layers=candidate_premature_layers,
        mature_layer=mature_layer,
        enable_token_selective_dola=enable_token_selective_dola,
        token_selective_mode=token_selective_mode,
        token_selective_config=token_selective_config,
    ).score



def score_candidate_answers_dola(
    model: Any,
    tokenizer: Any,
    prompt: str,
    candidate_answers: list[str],
    premature_layer: int,
    separator: str = " ",
    score_mode: str = "sum_logprob",
    dola_score_mode: str = "legacy_contrastive",
    post_softmax: bool = False,
    relative_top: float = 0.0,
    relative_top_value: float = -1000.0,
    candidate_premature_layers: list[int] | None = None,
    mature_layer: int | None = None,
    enable_token_selective_dola: bool = False,
    token_selective_mode: str = TOKEN_SELECTIVE_DOLA_MODE_HEURISTIC_FACT_CRITICAL_V1,
    token_selective_config: TokenSelectiveDolaConfig | None = None,
) -> list[tuple[str, float]]:
    """Score each candidate answer with configurable DoLa-style scoring."""
    return [
        (item.candidate, item.score)
        for item in score_candidate_answers_dola_with_details(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            candidate_answers=candidate_answers,
            premature_layer=premature_layer,
            separator=separator,
            score_mode=score_mode,
            dola_score_mode=dola_score_mode,
            post_softmax=post_softmax,
            relative_top=relative_top,
            relative_top_value=relative_top_value,
            candidate_premature_layers=candidate_premature_layers,
            mature_layer=mature_layer,
            enable_token_selective_dola=enable_token_selective_dola,
            token_selective_mode=token_selective_mode,
            token_selective_config=token_selective_config,
        )
    ]



def score_candidate_answers_dola_with_details(
    model: Any,
    tokenizer: Any,
    prompt: str,
    candidate_answers: list[str],
    premature_layer: int,
    separator: str = " ",
    score_mode: str = "sum_logprob",
    dola_score_mode: str = "legacy_contrastive",
    post_softmax: bool = False,
    relative_top: float = 0.0,
    relative_top_value: float = -1000.0,
    candidate_premature_layers: list[int] | None = None,
    mature_layer: int | None = None,
    return_trace: bool = False,
    enable_token_selective_dola: bool = False,
    token_selective_mode: str = TOKEN_SELECTIVE_DOLA_MODE_HEURISTIC_FACT_CRITICAL_V1,
    token_selective_config: TokenSelectiveDolaConfig | None = None,
) -> list[CandidateScore]:
    """Score each candidate answer with DoLa-style scoring and token counts."""
    if not candidate_answers:
        raise ValueError("candidate_answers must contain at least one answer.")

    return [
        score_continuation_dola_details(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            candidate_answer=candidate_answer,
            premature_layer=premature_layer,
            separator=separator,
            score_mode=score_mode,
            dola_score_mode=dola_score_mode,
            post_softmax=post_softmax,
            relative_top=relative_top,
            relative_top_value=relative_top_value,
            candidate_premature_layers=candidate_premature_layers,
            mature_layer=mature_layer,
            return_trace=return_trace,
            enable_token_selective_dola=enable_token_selective_dola,
            token_selective_mode=token_selective_mode,
            token_selective_config=token_selective_config,
        )
        for candidate_answer in candidate_answers
    ]



def score_candidate_answers_multi_config_with_details(
    model: Any,
    tokenizer: Any,
    prompt: str,
    candidate_answers: list[str],
    *,
    static_layers: list[int],
    dynamic_buckets: dict[str, list[int]],
    separator: str = " ",
    score_mode: str = "sum_logprob",
    post_softmax: bool = False,
    relative_top: float = 0.0,
    relative_top_value: float = -1000.0,
    mature_layer: int | None = None,
    candidate_batch_size: int = 1,
) -> MultiConfigScoreResult:
    """Score one prompt against many answers while reusing the same forwards.

    This helper keeps scoring semantics identical to the per-answer paths above,
    but lets higher-level evaluation code batch answers and rescore multiple
    static/dynamic configurations from one hidden-state pass.
    """
    try:
        import torch
    except ImportError as error:
        raise ImportError(
            "torch is required for DoLa-style candidate scoring. "
            "Install the extra model dependencies from requirements-model.txt."
        ) from error

    if not candidate_answers:
        raise ValueError("candidate_answers must contain at least one answer.")
    if candidate_batch_size <= 0:
        raise ValueError("candidate_batch_size must be a positive integer.")

    num_hidden_layers = int(getattr(model.config, "num_hidden_layers", 0))
    resolved_mature_layer = _resolve_mature_layer(mature_layer, num_hidden_layers)
    resolved_static_layers = validate_candidate_premature_layers(
        static_layers,
        resolved_mature_layer,
        num_hidden_layers,
        allow_embedding_output=any(int(layer) < 0 for layer in static_layers),
    )
    resolved_dynamic_buckets = {
        str(name): validate_candidate_premature_layers(
            layers,
            resolved_mature_layer,
            num_hidden_layers,
            allow_embedding_output=any(int(layer) < 0 for layer in layers),
        )
        for name, layers in dynamic_buckets.items()
    }

    lm_head = model.get_output_embeddings()
    if lm_head is None:
        raise ValueError("The model does not expose output embeddings for DoLa-style scoring.")

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        pad_token_id = 0

    results = MultiConfigScoreResult(
        vanilla=[],
        static={layer: [] for layer in resolved_static_layers},
        dynamic={name: [] for name in resolved_dynamic_buckets},
        batch_size=candidate_batch_size,
        batch_count=0,
        profile=None,
    )
    tokenize_sec = 0.0
    model_forward_sec = 0.0
    static_rescore_sec = 0.0
    dynamic_rescore_sec = 0.0
    materialize_sec = 0.0
    max_prompt_len = 0
    max_total_len = 0

    dynamic_union_layers = sorted(
        {
            layer
            for layers in resolved_dynamic_buckets.values()
            for layer in layers
        }
    )
    all_needed_layers = sorted({*resolved_static_layers, *dynamic_union_layers})

    for batch_start in range(0, len(candidate_answers), candidate_batch_size):
        batch_candidates = candidate_answers[batch_start : batch_start + candidate_batch_size]
        tokenize_start = time.perf_counter()
        prepared = [
            _prepare_scoring_inputs(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                candidate_answer=candidate_answer,
                separator=separator,
            )
            for candidate_answer in batch_candidates
        ]
        batch_input_ids, batch_attention_mask, prompt_lens, valid_lengths = _pad_prepared_batch(
            prepared,
            pad_token_id=pad_token_id,
        )
        tokenize_sec += time.perf_counter() - tokenize_start
        if prompt_lens:
            max_prompt_len = max(max_prompt_len, max(int(prompt_len) for prompt_len in prompt_lens))
        if valid_lengths:
            max_total_len = max(max_total_len, max(int(length) for length in valid_lengths))

        with torch.inference_mode():
            forward_start = time.perf_counter()
            hidden_states = _forward_hidden_states_only(
                model,
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask,
            )
            model_forward_sec += time.perf_counter() - forward_start

            mature_hidden = hidden_states[
                internal_layer_to_hidden_state_index(resolved_mature_layer, num_hidden_layers)
            ]
            mature_logits = lm_head(mature_hidden[:, :-1, :])
            mature_log_probs = torch.log_softmax(mature_logits, dim=-1)
            mature_probs = mature_log_probs.exp()
            vanilla_scores = _gather_scores_at_target_ids(mature_log_probs, batch_input_ids)

            dynamic_layer_memberships: dict[int, list[tuple[str, int]]] = {}
            for bucket_name, candidate_layers in resolved_dynamic_buckets.items():
                for layer_index, layer in enumerate(candidate_layers):
                    dynamic_layer_memberships.setdefault(layer, []).append((bucket_name, layer_index))

            dynamic_best_js: dict[str, Any] = {}
            dynamic_selected_indices: dict[str, Any] = {
                bucket_name: torch.zeros(
                    mature_log_probs.shape[:2],
                    device=mature_log_probs.device,
                    dtype=torch.long,
                )
                for bucket_name in resolved_dynamic_buckets
            }

            static_scores: dict[int, Any] = {}
            for layer in all_needed_layers:
                layer_log_probs = torch.log_softmax(
                    lm_head(
                        hidden_states[
                            internal_layer_to_hidden_state_index(layer, num_hidden_layers)
                        ][:, :-1, :]
                    ),
                    dim=-1,
                )
                if layer in resolved_static_layers:
                    static_start = time.perf_counter()
                    static_scores[layer] = _compute_official_dola_token_scores_from_log_probs(
                        final_log_probs=mature_log_probs,
                        base_log_probs=layer_log_probs,
                        input_ids=batch_input_ids,
                        post_softmax=post_softmax,
                        relative_top=relative_top,
                        relative_top_value=relative_top_value,
                    )
                    static_rescore_sec += time.perf_counter() - static_start
                if layer in dynamic_layer_memberships:
                    dynamic_first_pass_start = time.perf_counter()
                    layer_js_divergence = _compute_dynamic_js_divergence_from_distributions(
                        mature_probs=mature_probs,
                        mature_log_probs=mature_log_probs,
                        candidate_probs=layer_log_probs.exp().unsqueeze(0),
                        candidate_log_probs=layer_log_probs.unsqueeze(0),
                    ).squeeze(0)
                    for bucket_name, layer_index in dynamic_layer_memberships[layer]:
                        if bucket_name not in dynamic_best_js:
                            dynamic_best_js[bucket_name] = layer_js_divergence
                            dynamic_selected_indices[bucket_name] = torch.full_like(
                                dynamic_selected_indices[bucket_name],
                                layer_index,
                            )
                            continue
                        update_mask = layer_js_divergence > dynamic_best_js[bucket_name]
                        dynamic_best_js[bucket_name] = torch.where(
                            update_mask,
                            layer_js_divergence,
                            dynamic_best_js[bucket_name],
                        )
                        dynamic_selected_indices[bucket_name] = torch.where(
                            update_mask,
                            torch.full_like(dynamic_selected_indices[bucket_name], layer_index),
                            dynamic_selected_indices[bucket_name],
                        )
                    dynamic_rescore_sec += time.perf_counter() - dynamic_first_pass_start
            dynamic_scores: dict[str, Any] = {}
            if resolved_dynamic_buckets:
                for bucket_name, candidate_layers in resolved_dynamic_buckets.items():
                    dynamic_second_pass_start = time.perf_counter()
                    selected_bucket_indices = dynamic_selected_indices[bucket_name]
                    base_log_probs = None
                    for layer_index, layer in enumerate(candidate_layers):
                        layer_mask = selected_bucket_indices == layer_index
                        if not bool(layer_mask.any().item()):
                            continue
                        layer_log_probs = torch.log_softmax(
                            lm_head(
                                hidden_states[
                                    internal_layer_to_hidden_state_index(layer, num_hidden_layers)
                                ][:, :-1, :]
                            ),
                            dim=-1,
                        )
                        if base_log_probs is None:
                            base_log_probs = layer_log_probs.clone()
                            continue
                        base_log_probs = torch.where(
                            layer_mask.unsqueeze(-1),
                            layer_log_probs,
                            base_log_probs,
                        )
                    if base_log_probs is None:
                        raise ValueError(
                            f"Dynamic bucket '{bucket_name}' did not select any candidate layers."
                    )
                    dynamic_scores[bucket_name] = _compute_official_dola_token_scores_from_log_probs(
                        final_log_probs=mature_log_probs,
                        base_log_probs=base_log_probs,
                        input_ids=batch_input_ids,
                        post_softmax=post_softmax,
                        relative_top=relative_top,
                        relative_top_value=relative_top_value,
                    )
                    dynamic_rescore_sec += time.perf_counter() - dynamic_second_pass_start

        materialize_start = time.perf_counter()
        for batch_index, candidate_answer in enumerate(batch_candidates):
            prompt_len = prompt_lens[batch_index]
            valid_target_length = max(valid_lengths[batch_index] - 1, 0)
            continuation_slice = _slice_continuation_scores(
                token_scores=vanilla_scores[batch_index : batch_index + 1],
                prompt_len=prompt_len,
                valid_target_length=valid_target_length,
            )
            vanilla_score, continuation_token_count = _aggregate_continuation_log_probs(
                continuation_log_probs=continuation_slice,
                score_mode=score_mode,
            )
            results.vanilla.append(
                CandidateScore(
                    candidate=candidate_answer,
                    score=vanilla_score,
                    continuation_token_count=continuation_token_count,
                )
            )

            for layer in resolved_static_layers:
                static_slice = _slice_continuation_scores(
                    token_scores=static_scores[layer][batch_index : batch_index + 1],
                    prompt_len=prompt_len,
                    valid_target_length=valid_target_length,
                )
                score, token_count = _aggregate_continuation_log_probs(
                    continuation_log_probs=static_slice,
                    score_mode=score_mode,
                )
                results.static[layer].append(
                    CandidateScore(
                        candidate=candidate_answer,
                        score=score,
                        continuation_token_count=token_count,
                        premature_layer_dist={layer: token_count},
                    )
                )

            for bucket_name, candidate_layers in resolved_dynamic_buckets.items():
                dynamic_slice = _slice_continuation_scores(
                    token_scores=dynamic_scores[bucket_name][batch_index : batch_index + 1],
                    prompt_len=prompt_len,
                    valid_target_length=valid_target_length,
                )
                score, token_count = _aggregate_continuation_log_probs(
                    continuation_log_probs=dynamic_slice,
                    score_mode=score_mode,
                )
                selected_bucket_indices = dynamic_selected_indices[bucket_name][
                    batch_index,
                    _get_continuation_start_index(prompt_len) : valid_target_length,
                ]
                results.dynamic[bucket_name].append(
                    CandidateScore(
                        candidate=candidate_answer,
                        score=score,
                        continuation_token_count=token_count,
                        premature_layer_dist=_count_premature_layer_usage_from_selected_indices(
                            selected_bucket_indices,
                            candidate_layers,
                        )
                        or None,
                    )
                )

        materialize_sec += time.perf_counter() - materialize_start
        results.batch_count += 1

    results.profile = {
        "tokenize_sec": tokenize_sec,
        "model_forward_sec": model_forward_sec,
        "dynamic_rescore_sec": dynamic_rescore_sec,
        "static_rescore_sec": static_rescore_sec,
        "materialize_sec": materialize_sec,
        "num_candidates": len(candidate_answers),
        "candidate_batch_size": candidate_batch_size,
        "prompt_len": max_prompt_len,
        "max_total_len": max_total_len,
        "union_layer_count": len(all_needed_layers),
    }
    return results


def score_continuation_dola_details(
    model: Any,
    tokenizer: Any,
    prompt: str,
    candidate_answer: str,
    premature_layer: int,
    separator: str = " ",
    score_mode: str = "sum_logprob",
    dola_score_mode: str = "legacy_contrastive",
    post_softmax: bool = False,
    relative_top: float = 0.0,
    relative_top_value: float = -1000.0,
    candidate_premature_layers: list[int] | None = None,
    mature_layer: int | None = None,
    return_trace: bool = False,
    enable_token_selective_dola: bool = False,
    token_selective_mode: str = TOKEN_SELECTIVE_DOLA_MODE_HEURISTIC_FACT_CRITICAL_V1,
    token_selective_config: TokenSelectiveDolaConfig | None = None,
) -> CandidateScore:
    """Score one candidate with configurable DoLa-style scoring and token counts.

    Notes
    -----
    For decoder-only models, ``hidden_states[0]`` is the embedding output and
    ``hidden_states[k + 1]`` corresponds to the output of decoder block ``k``.
    This scorer keeps the existing local block ids (``0`` means decoder block 0)
    and additionally allows ``-1`` as a special embedding-output candidate id.
    """
    try:
        import torch
    except ImportError as error:
        raise ImportError(
            "torch is required for DoLa-style candidate scoring. "
            "Install the extra model dependencies from requirements-model.txt."
        ) from error

    _, input_ids, attention_mask, prompt_len = _prepare_scoring_inputs(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        candidate_answer=candidate_answer,
        separator=separator,
    )

    num_hidden_layers = int(getattr(model.config, "num_hidden_layers", 0))
    resolved_mature_layer = _resolve_mature_layer(mature_layer, num_hidden_layers)
    normalized_dola_mode = _normalize_dola_score_mode(dola_score_mode)
    normalized_token_selective_mode = (
        _normalize_token_selective_mode(token_selective_mode)
        if enable_token_selective_dola
        else TOKEN_SELECTIVE_DOLA_MODE_HEURISTIC_FACT_CRITICAL_V1
    )
    resolved_premature_layer: int | None = None
    resolved_candidate_layers: list[int] = []

    if normalized_dola_mode in {"legacy_contrastive", "official_static_dola"}:
        resolved_premature_layer = validate_candidate_premature_layers(
            [premature_layer],
            resolved_mature_layer,
            num_hidden_layers,
            allow_embedding_output=int(premature_layer) < 0,
        )[0]
    else:
        resolved_candidate_layers = validate_candidate_premature_layers(
            candidate_premature_layers,
            resolved_mature_layer,
            num_hidden_layers,
            allow_embedding_output=any(int(layer) < 0 for layer in candidate_premature_layers or []),
        )

    lm_head = model.get_output_embeddings()
    if lm_head is None:
        raise ValueError("The model does not expose output embeddings for DoLa-style scoring.")

    with torch.inference_mode():
        hidden_states = _forward_hidden_states_only(
            model,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        mature_hidden = hidden_states[
            internal_layer_to_hidden_state_index(resolved_mature_layer, num_hidden_layers)
        ]
        mature_logits = lm_head(mature_hidden[:, :-1, :])

        final_target_scores = None
        premature_target_scores = None
        contrast_token_scores = None
        token_selected_mask: list[bool] | None = None
        token_selected_reason: list[str] | None = None
        token_effective_score_source: list[str] | None = None
        token_contrast_weight: list[float] | None = None
        token_selection_tier: list[str] | None = None

        if normalized_dola_mode == "legacy_contrastive":
            premature_hidden = hidden_states[
                internal_layer_to_hidden_state_index(resolved_premature_layer, num_hidden_layers)
            ]
            premature_logits = lm_head(premature_hidden[:, :-1, :])
            token_scores = _gather_token_log_probs(mature_logits - premature_logits, input_ids)
            selected_layers: list[int] = [resolved_premature_layer] * int(mature_logits.shape[1])
        elif normalized_dola_mode == "official_static_dola":
            premature_hidden = hidden_states[
                internal_layer_to_hidden_state_index(resolved_premature_layer, num_hidden_layers)
            ]
            premature_logits = lm_head(premature_hidden[:, :-1, :])
            token_scores = _compute_official_dola_token_scores(
                mature_logits=mature_logits,
                base_logits=premature_logits,
                input_ids=input_ids,
                post_softmax=post_softmax,
                relative_top=relative_top,
                relative_top_value=relative_top_value,
            )
            selected_layers = [resolved_premature_layer] * int(mature_logits.shape[1])
        else:
            candidate_logits = torch.stack(
                [
                    lm_head(
                        hidden_states[
                            internal_layer_to_hidden_state_index(layer, num_hidden_layers)
                        ][:, :-1, :]
                    )
                    for layer in resolved_candidate_layers
                ],
                dim=0,
            )
            base_logits, selected_layers = _select_dynamic_base_logits(
                mature_logits=mature_logits,
                candidate_logits=candidate_logits,
                candidate_layers=resolved_candidate_layers,
            )
            token_scores = _compute_official_dola_token_scores(
                mature_logits=mature_logits,
                base_logits=base_logits,
                input_ids=input_ids,
                post_softmax=post_softmax,
                relative_top=relative_top,
                relative_top_value=relative_top_value,
            )

        if enable_token_selective_dola:
            mature_log_probs = torch.log_softmax(mature_logits, dim=-1)
            final_target_scores = _gather_scores_at_target_ids(mature_log_probs, input_ids)
            contrast_token_scores = token_scores
            (
                token_scores,
                token_selected_mask,
                token_selected_reason,
                token_effective_score_source,
                token_contrast_weight,
                token_selection_tier,
            ) = _apply_token_selective_dola_scores(
                tokenizer=tokenizer,
                input_ids=input_ids,
                contrast_token_scores=contrast_token_scores,
                final_token_scores=final_target_scores,
                token_selective_mode=normalized_token_selective_mode,
                token_selective_config=token_selective_config,
            )

        if return_trace:
            if final_target_scores is None:
                mature_log_probs = torch.log_softmax(mature_logits, dim=-1)
                final_target_scores = _gather_scores_at_target_ids(mature_log_probs, input_ids)
            if normalized_dola_mode in {"legacy_contrastive", "official_static_dola"}:
                base_log_probs = torch.log_softmax(premature_logits, dim=-1)
            else:
                base_log_probs = torch.log_softmax(base_logits, dim=-1)
            premature_target_scores = _gather_scores_at_target_ids(base_log_probs, input_ids)
            if contrast_token_scores is None:
                contrast_token_scores = token_scores

    continuation_start = _get_continuation_start_index(prompt_len)
    continuation_scores = token_scores[:, continuation_start:]
    score, continuation_token_count = _aggregate_continuation_log_probs(
        continuation_log_probs=continuation_scores,
        score_mode=score_mode,
    )
    premature_layer_dist = _count_premature_layer_usage(
        selected_layers[continuation_start:],
        resolved_candidate_layers if resolved_candidate_layers else [resolved_premature_layer],
    )
    trace = None
    if return_trace:
        continuation_token_count_for_trace = int(continuation_scores.shape[-1])
        trace = _build_candidate_score_trace(
            tokenizer=tokenizer,
            input_ids=input_ids,
            scoring_score_start_index=continuation_start,
            token_scores=continuation_scores,
            final_token_scores=None
            if final_target_scores is None
            else final_target_scores[:, continuation_start:],
            premature_token_scores=None
            if premature_target_scores is None
            else premature_target_scores[:, continuation_start:],
            contrast_token_scores=None
            if contrast_token_scores is None
            else contrast_token_scores[:, continuation_start:],
            selected_premature_layers=selected_layers[
                continuation_start : continuation_start + continuation_token_count_for_trace
            ],
            token_selected_mask=None
            if token_selected_mask is None
            else token_selected_mask[
                continuation_start : continuation_start + continuation_token_count_for_trace
            ],
            token_selected_reason=None
            if token_selected_reason is None
            else token_selected_reason[
                continuation_start : continuation_start + continuation_token_count_for_trace
            ],
            token_effective_score_source=None
            if token_effective_score_source is None
            else token_effective_score_source[
                continuation_start : continuation_start + continuation_token_count_for_trace
            ],
            token_contrast_weight=None
            if token_contrast_weight is None
            else token_contrast_weight[
                continuation_start : continuation_start + continuation_token_count_for_trace
            ],
            token_selection_tier=None
            if token_selection_tier is None
            else token_selection_tier[
                continuation_start : continuation_start + continuation_token_count_for_trace
            ],
        )
    return CandidateScore(
        candidate=candidate_answer,
        score=score,
        continuation_token_count=continuation_token_count,
        premature_layer_dist=premature_layer_dist or None,
        trace=trace,
    )



def generate_dola(
    model: Any,
    tokenizer: Any,
    prompt: str,
    dola_layers: list[int] | None = None,
    max_new_tokens: int = 32,
    **kwargs: Any,
) -> str:
    """Generate text with the future DoLa decoding path.

    This function is intentionally left as a stub during the scaffold phase.
    """
    raise NotImplementedError(
        "TODO: implement DoLa generation after baseline plumbing is ready."
    )



def _prepare_scoring_inputs(
    model: Any,
    tokenizer: Any,
    prompt: str,
    candidate_answer: str,
    separator: str,
) -> tuple[str, Any, Any, int]:
    """Build shared tokenized inputs for continuation scoring."""
    if not prompt.strip():
        raise ValueError("prompt must be a non-empty string.")
    if not candidate_answer.strip():
        raise ValueError("candidate_answer must be a non-empty string.")

    attempts = _build_scoring_text_attempts(prompt, candidate_answer, separator)

    for prompt_text, full_text in attempts:
        offset_result = _prepare_scoring_inputs_with_offsets(
            model=model,
            tokenizer=tokenizer,
            prompt_text=prompt_text,
            full_text=full_text,
        )
        if offset_result is not None:
            return offset_result

    for prompt_text, full_text in attempts:
        prompt_batch = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
        full_batch = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)

        prompt_len = int(prompt_batch["input_ids"].shape[1])
        full_len = int(full_batch["input_ids"].shape[1])
        if full_len > prompt_len:
            model_device = _get_model_runtime_device(model)
            input_ids = full_batch["input_ids"].to(model_device)
            attention_mask = full_batch["attention_mask"].to(model_device)
            return prompt_text, input_ids, attention_mask, prompt_len

    raise ValueError(
        "candidate_answer did not add any continuation tokens after scoring-boundary fallback. "
        f"prompt={prompt!r}, candidate_answer={candidate_answer!r}"
    )



def _build_scoring_text_attempts(
    prompt: str,
    candidate_answer: str,
    separator: str,
) -> list[tuple[str, str]]:
    """Build prompt/full-text attempts with slightly different token boundaries."""
    normalized_separator = separator or " "
    preserve_leading_separator = bool(
        normalized_separator
        and candidate_answer.startswith(normalized_separator)
        and not candidate_answer.startswith(normalized_separator * 2)
    )
    normalized_candidate = (
        candidate_answer if preserve_leading_separator else candidate_answer.lstrip()
    )

    prompt_needs_separator = not prompt[-1].isspace() and not preserve_leading_separator
    primary_prompt = f"{prompt}{normalized_separator}" if prompt_needs_separator else prompt
    fallback_prompt = (
        f"{prompt.rstrip()}{normalized_separator}"
        if not preserve_leading_separator
        else prompt.rstrip()
    )

    attempts: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for prompt_text in (primary_prompt, fallback_prompt):
        full_text = f"{prompt_text}{normalized_candidate}"
        pair = (prompt_text, full_text)
        if pair not in seen:
            seen.add(pair)
            attempts.append(pair)
    return attempts



def _prepare_scoring_inputs_with_offsets(
    model: Any,
    tokenizer: Any,
    prompt_text: str,
    full_text: str,
) -> tuple[str, Any, Any, int] | None:
    """Use offset mapping to locate the continuation start when available."""
    try:
        full_batch = tokenizer(
            full_text,
            return_tensors="pt",
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
    except (NotImplementedError, TypeError, ValueError):
        return None

    offset_mapping = full_batch.get("offset_mapping")
    if offset_mapping is None:
        return None

    continuation_token_index = _find_continuation_token_index(
        offset_mapping=offset_mapping[0],
        prompt_char_len=len(prompt_text),
    )
    if continuation_token_index is None:
        return None

    model_device = _get_model_runtime_device(model)
    input_ids = full_batch["input_ids"].to(model_device)
    attention_mask = full_batch["attention_mask"].to(model_device)
    return prompt_text, input_ids, attention_mask, continuation_token_index



def _find_continuation_token_index(
    offset_mapping: Any,
    prompt_char_len: int,
) -> int | None:
    """Find the first token whose character span reaches into the candidate text."""
    for token_index, offset_pair in enumerate(offset_mapping):
        start_offset, end_offset = _coerce_offset_pair(offset_pair)
        if start_offset == end_offset == 0:
            continue
        if end_offset > prompt_char_len:
            return token_index
    return None



def _coerce_offset_pair(offset_pair: Any) -> tuple[int, int]:
    """Normalize one tokenizer offset pair to plain Python integers."""
    if len(offset_pair) != 2:
        raise ValueError(f"Invalid offset pair: {offset_pair!r}")
    start_offset, end_offset = offset_pair
    return int(start_offset), int(end_offset)



def _gather_token_log_probs(logits: Any, input_ids: Any) -> Any:
    """Convert logits into log-probs gathered at the target token ids."""
    try:
        import torch
    except ImportError as error:
        raise ImportError(
            "torch is required for candidate scoring. "
            "Install the extra model dependencies from requirements-model.txt."
        ) from error

    target_ids = input_ids[:, 1:]
    log_probs = torch.log_softmax(logits, dim=-1)
    return log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)



def _gather_scores_at_target_ids(scores: Any, input_ids: Any) -> Any:
    """Gather already-normalized token scores at the target token ids."""
    target_ids = input_ids[:, 1:]
    return scores.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)



def _compute_official_dola_token_scores_from_log_probs(
    final_log_probs: Any,
    base_log_probs: Any,
    input_ids: Any,
    post_softmax: bool,
    relative_top: float,
    relative_top_value: float,
) -> Any:
    """Apply official-style contrastive scoring from precomputed log-probs."""
    try:
        import torch
    except ImportError as error:
        raise ImportError(
            "torch is required for DoLa-style candidate scoring. "
            "Install the extra model dependencies from requirements-model.txt."
        ) from error

    if not post_softmax and relative_top == 0.0:
        # Fast path for tuned TruthfulQA-MC CV: only target-token scores are needed.
        final_target_log_probs = _gather_scores_at_target_ids(final_log_probs, input_ids)
        base_target_log_probs = _gather_scores_at_target_ids(base_log_probs, input_ids)
        return final_target_log_probs - base_target_log_probs

    diff_logits = final_log_probs - base_log_probs
    if post_softmax:
        diff_logits = torch.log_softmax(diff_logits, dim=-1)
    if relative_top > 0.0:
        diff_logits = _apply_relative_top_mask(
            diff_logits=diff_logits,
            final_log_probs=final_log_probs,
            relative_top=relative_top,
            relative_top_value=relative_top_value,
        )
    return _gather_scores_at_target_ids(diff_logits, input_ids)



def _compute_official_dola_token_scores(
    mature_logits: Any,
    base_logits: Any,
    input_ids: Any,
    post_softmax: bool,
    relative_top: float,
    relative_top_value: float,
) -> Any:
    """Apply official-style static contrastive scoring once base logits are chosen."""
    try:
        import torch
    except ImportError as error:
        raise ImportError(
            "torch is required for DoLa-style candidate scoring. "
            "Install the extra model dependencies from requirements-model.txt."
        ) from error

    final_log_probs = torch.log_softmax(mature_logits, dim=-1)
    base_log_probs = torch.log_softmax(base_logits, dim=-1)
    return _compute_official_dola_token_scores_from_log_probs(
        final_log_probs=final_log_probs,
        base_log_probs=base_log_probs,
        input_ids=input_ids,
        post_softmax=post_softmax,
        relative_top=relative_top,
        relative_top_value=relative_top_value,
    )



def _compute_dynamic_js_divergence_from_distributions(
    mature_probs: Any,
    mature_log_probs: Any,
    candidate_probs: Any,
    candidate_log_probs: Any,
) -> Any:
    """Compute per-layer, per-token JS divergence from precomputed distributions."""
    try:
        import torch.nn.functional as F
    except ImportError as error:
        raise ImportError(
            "torch is required for DoLa-style candidate scoring. "
            "Install the extra model dependencies from requirements-model.txt."
        ) from error

    vocab_size = int(mature_probs.shape[-1])
    if vocab_size <= 0:
        raise ValueError("candidate distributions must have a positive vocabulary dimension.")

    kl_mature_sum = None
    kl_candidate_sum = None
    chunk_size = 4096

    for start in range(0, vocab_size, chunk_size):
        end = min(start + chunk_size, vocab_size)
        mature_probs_chunk = mature_probs[..., start:end]
        mature_log_probs_chunk = mature_log_probs[..., start:end]
        candidate_probs_chunk = candidate_probs[..., start:end]
        candidate_log_probs_chunk = candidate_log_probs[..., start:end]
        mixture = 0.5 * (mature_probs_chunk.unsqueeze(0) + candidate_probs_chunk)

        current_kl_mature = F.kl_div(
            mature_log_probs_chunk.unsqueeze(0).expand_as(candidate_log_probs_chunk),
            mixture,
            reduction="none",
        ).sum(dim=-1)
        current_kl_candidate = F.kl_div(
            candidate_log_probs_chunk,
            mixture,
            reduction="none",
        ).sum(dim=-1)

        if kl_mature_sum is None:
            kl_mature_sum = current_kl_mature
            kl_candidate_sum = current_kl_candidate
        else:
            kl_mature_sum = kl_mature_sum + current_kl_mature
            kl_candidate_sum = kl_candidate_sum + current_kl_candidate

    if kl_mature_sum is None or kl_candidate_sum is None:
        raise ValueError("Failed to accumulate dynamic JS divergence chunks.")
    return 0.5 * ((kl_mature_sum / vocab_size) + (kl_candidate_sum / vocab_size))



def _precompute_union_layer_dynamic_js(
    mature_probs: Any,
    mature_log_probs: Any,
    layer_probs: dict[int, Any],
    layer_log_probs: dict[int, Any],
    union_layers: list[int],
) -> tuple[Any, Any]:
    """Compute union-layer log-probs and JS divergence once for overlapping buckets."""
    try:
        import torch
    except ImportError as error:
        raise ImportError(
            "torch is required for DoLa-style candidate scoring. "
            "Install the extra model dependencies from requirements-model.txt."
        ) from error

    union_layer_log_probs = torch.stack(
        [layer_log_probs[layer] for layer in union_layers],
        dim=0,
    )
    union_layer_probs = torch.stack(
        [layer_probs[layer] for layer in union_layers],
        dim=0,
    )
    union_js_divergence = _compute_dynamic_js_divergence_from_distributions(
        mature_probs=mature_probs,
        mature_log_probs=mature_log_probs,
        candidate_probs=union_layer_probs,
        candidate_log_probs=union_layer_log_probs,
    )
    return union_layer_log_probs, union_js_divergence



def _select_dynamic_base_log_probs_from_union(
    union_layer_log_probs_by_batch: Any,
    union_js_divergence: Any,
    union_layer_to_index: dict[int, int],
    candidate_layers: list[int],
    *,
    return_selected_layers_trace: bool = True,
) -> tuple[Any, Any, list[list[int]] | None]:
    """Select bucket-local dynamic base log-probs by slicing precomputed union JS."""
    try:
        import torch
    except ImportError as error:
        raise ImportError(
            "torch is required for DoLa-style candidate scoring. "
            "Install the extra model dependencies from requirements-model.txt."
        ) from error

    bucket_union_indices = torch.tensor(
        [union_layer_to_index[layer] for layer in candidate_layers],
        device=union_js_divergence.device,
        dtype=torch.long,
    )
    bucket_js_divergence = union_js_divergence.index_select(0, bucket_union_indices)
    selected_bucket_indices = bucket_js_divergence.argmax(dim=0)
    selected_union_indices = bucket_union_indices[selected_bucket_indices]
    gather_index = selected_union_indices.unsqueeze(-1).unsqueeze(-1).expand(
        -1,
        -1,
        1,
        union_layer_log_probs_by_batch.shape[-1],
    )
    base_log_probs = union_layer_log_probs_by_batch.gather(2, gather_index).squeeze(2)
    selected_layers_trace = _build_selected_layers_trace(
        selected_indices=selected_bucket_indices,
        candidate_layers=candidate_layers,
        return_selected_layers_trace=return_selected_layers_trace,
    )
    return base_log_probs, selected_bucket_indices, selected_layers_trace



def _compute_dynamic_js_divergence(
    mature_logits: Any,
    candidate_logits: Any,
) -> Any:
    """Compute per-layer, per-token JS divergence for official dynamic DoLa."""
    try:
        import torch
    except ImportError as error:
        raise ImportError(
            "torch is required for DoLa-style candidate scoring. "
            "Install the extra model dependencies from requirements-model.txt."
        ) from error

    mature_probs = torch.softmax(mature_logits, dim=-1)
    mature_log_probs = torch.log_softmax(mature_logits, dim=-1)
    candidate_probs = torch.softmax(candidate_logits, dim=-1)
    candidate_log_probs = torch.log_softmax(candidate_logits, dim=-1)
    return _compute_dynamic_js_divergence_from_distributions(
        mature_probs=mature_probs,
        mature_log_probs=mature_log_probs,
        candidate_probs=candidate_probs,
        candidate_log_probs=candidate_log_probs,
    )



def _select_dynamic_base_logits_batched(
    mature_logits: Any,
    candidate_logits: Any,
    candidate_layers: list[int],
    *,
    return_selected_layers_trace: bool = True,
) -> tuple[Any, Any, list[list[int]] | None]:
    """Pick one premature layer per token position via JS divergence for each batch item."""
    try:
        import torch
    except ImportError as error:
        raise ImportError(
            "torch is required for DoLa-style candidate scoring. "
            "Install the extra model dependencies from requirements-model.txt."
        ) from error

    js_divergence = _compute_dynamic_js_divergence(
        mature_logits=mature_logits,
        candidate_logits=candidate_logits,
    )
    selected_indices = js_divergence.argmax(dim=0)
    candidate_logits_by_batch = candidate_logits.permute(1, 2, 0, 3)
    gather_index = selected_indices.unsqueeze(-1).unsqueeze(-1).expand(
        -1,
        -1,
        1,
        candidate_logits_by_batch.shape[-1],
    )
    base_logits = candidate_logits_by_batch.gather(2, gather_index).squeeze(2)
    selected_layers_trace = _build_selected_layers_trace(
        selected_indices=selected_indices,
        candidate_layers=candidate_layers,
        return_selected_layers_trace=return_selected_layers_trace,
    )
    return base_logits, selected_indices, selected_layers_trace



def _select_dynamic_base_logits(
    mature_logits: Any,
    candidate_logits: Any,
    candidate_layers: list[int],
) -> tuple[Any, list[int]]:
    """Pick one premature layer per token position via JS divergence."""
    base_logits, _, selected_layers = _select_dynamic_base_logits_batched(
        mature_logits=mature_logits,
        candidate_logits=candidate_logits,
        candidate_layers=candidate_layers,
        return_selected_layers_trace=True,
    )
    if selected_layers is None or len(selected_layers) != 1:
        raise ValueError("official_dynamic_dola single-candidate path expects batch size 1 scoring.")
    return base_logits, selected_layers[0]



def _build_selected_layers_trace(
    *,
    selected_indices: Any,
    candidate_layers: list[int],
    return_selected_layers_trace: bool,
) -> list[list[int]] | None:
    """Materialize per-token selected-layer traces only when a caller explicitly needs them."""
    if not return_selected_layers_trace:
        return None
    return [
        [candidate_layers[int(index)] for index in batch_indices]
        for batch_indices in selected_indices.tolist()
    ]



def _count_premature_layer_usage_from_selected_indices(
    selected_layer_indices: Any,
    candidate_layers: list[int],
) -> dict[int, int]:
    """Count selected premature layers directly from tensor indices without token-level Python traces."""
    try:
        import torch
    except ImportError as error:
        raise ImportError(
            "torch is required for DoLa-style candidate scoring. "
            "Install the extra model dependencies from requirements-model.txt."
        ) from error

    if selected_layer_indices.numel() == 0:
        return {}
    counts = torch.bincount(
        selected_layer_indices.reshape(-1),
        minlength=len(candidate_layers),
    )
    return {
        int(candidate_layers[index]): int(count)
        for index, count in enumerate(counts.tolist())
        if int(count) > 0
    }



def _count_premature_layer_usage(
    selected_layers: list[int],
    candidate_layers: list[int],
) -> dict[int, int]:
    """Count how many scored continuation tokens chose each candidate layer."""
    usage = {int(layer): 0 for layer in candidate_layers if layer is not None}
    for layer in selected_layers:
        usage[int(layer)] = usage.get(int(layer), 0) + 1
    return {layer: count for layer, count in usage.items() if count > 0}



def _pad_prepared_batch(
    prepared: list[tuple[str, Any, Any, int]],
    *,
    pad_token_id: int,
) -> tuple[Any, Any, list[int], list[int]]:
    """Pad individually prepared scoring inputs into one batch tensor."""
    try:
        import torch
    except ImportError as error:
        raise ImportError(
            "torch is required for candidate scoring. "
            "Install the extra model dependencies from requirements-model.txt."
        ) from error

    if not prepared:
        raise ValueError("prepared must contain at least one scoring input.")

    max_length = max(int(input_ids.shape[1]) for _, input_ids, _, _ in prepared)
    batch_size = len(prepared)
    device = prepared[0][1].device
    input_dtype = prepared[0][1].dtype
    mask_dtype = prepared[0][2].dtype
    batch_input_ids = torch.full(
        (batch_size, max_length),
        int(pad_token_id),
        dtype=input_dtype,
        device=device,
    )
    batch_attention_mask = torch.zeros(
        (batch_size, max_length),
        dtype=mask_dtype,
        device=device,
    )
    prompt_lens: list[int] = []
    valid_lengths: list[int] = []
    for batch_index, (_, input_ids, attention_mask, prompt_len) in enumerate(prepared):
        seq_length = int(input_ids.shape[1])
        batch_input_ids[batch_index, :seq_length] = input_ids[0]
        batch_attention_mask[batch_index, :seq_length] = attention_mask[0]
        prompt_lens.append(int(prompt_len))
        valid_lengths.append(seq_length)
    return batch_input_ids, batch_attention_mask, prompt_lens, valid_lengths



def _slice_continuation_scores(
    token_scores: Any,
    *,
    prompt_len: int,
    valid_target_length: int,
) -> Any:
    """Slice padded token scores down to the scored continuation span."""
    continuation_start = _get_continuation_start_index(prompt_len)
    return token_scores[:, continuation_start:valid_target_length]



def _build_candidate_score_trace(
    *,
    tokenizer: Any,
    input_ids: Any,
    scoring_score_start_index: int,
    token_scores: Any,
    final_token_scores: Any | None = None,
    premature_token_scores: Any | None = None,
    contrast_token_scores: Any | None = None,
    selected_premature_layers: list[int] | None = None,
    token_selected_mask: list[bool] | None = None,
    token_selected_reason: list[str] | None = None,
    token_effective_score_source: list[str] | None = None,
    token_contrast_weight: list[float] | None = None,
    token_selection_tier: list[str] | None = None,
) -> CandidateScoreTrace:
    """Materialize JSON-friendly token-level scoring details on demand."""
    token_score_values = _tensor_first_row_to_float_list(token_scores)
    token_count = len(token_score_values)
    target_token_ids = input_ids[:, 1:]
    scored_target_ids = target_token_ids[
        :, scoring_score_start_index : scoring_score_start_index + token_count
    ]
    token_ids = _tensor_first_row_to_int_list(scored_target_ids)
    total_score = float(sum(token_score_values))
    avg_score = float(total_score / token_count) if token_count else 0.0

    return CandidateScoreTrace(
        token_ids=token_ids,
        token_texts=_token_ids_to_texts(tokenizer, token_ids),
        scoring_start_token_index=int(scoring_score_start_index + 1),
        token_scores=token_score_values,
        total_score=total_score,
        avg_score=avg_score,
        final_token_scores=None
        if final_token_scores is None
        else _tensor_first_row_to_float_list(final_token_scores),
        premature_token_scores=None
        if premature_token_scores is None
        else _tensor_first_row_to_float_list(premature_token_scores),
        contrast_token_scores=None
        if contrast_token_scores is None
        else _tensor_first_row_to_float_list(contrast_token_scores),
        selected_premature_layers=None
        if selected_premature_layers is None
        else [int(layer) for layer in selected_premature_layers],
        token_selected_mask=None
        if token_selected_mask is None
        else [bool(selected) for selected in token_selected_mask],
        token_selected_reason=None
        if token_selected_reason is None
        else [str(reason) for reason in token_selected_reason],
        token_effective_score_source=None
        if token_effective_score_source is None
        else [str(source) for source in token_effective_score_source],
        token_contrast_weight=None
        if token_contrast_weight is None
        else [float(weight) for weight in token_contrast_weight],
        token_selection_tier=None
        if token_selection_tier is None
        else [str(tier) for tier in token_selection_tier],
    )


def _tensor_first_row_to_float_list(values: Any) -> list[float]:
    raw_values = _tensor_first_row_to_plain_list(values)
    return [float(value) for value in raw_values]


def _tensor_first_row_to_int_list(values: Any) -> list[int]:
    raw_values = _tensor_first_row_to_plain_list(values)
    return [int(value) for value in raw_values]


def _tensor_first_row_to_plain_list(values: Any) -> list[Any]:
    if hasattr(values, "detach"):
        raw_values = values.detach().cpu().tolist()
    elif hasattr(values, "tolist"):
        raw_values = values.tolist()
    else:
        raw_values = list(values)
    if raw_values and isinstance(raw_values[0], list):
        raw_values = raw_values[0]
    return list(raw_values)


def _token_ids_to_texts(tokenizer: Any, token_ids: list[int]) -> list[str]:
    if not token_ids:
        return []

    convert_ids_to_tokens = getattr(tokenizer, "convert_ids_to_tokens", None)
    if callable(convert_ids_to_tokens):
        try:
            tokens = convert_ids_to_tokens(token_ids)
            if isinstance(tokens, str):
                return [tokens]
            return [str(token) for token in tokens]
        except Exception:
            pass

    decode = getattr(tokenizer, "decode", None)
    if callable(decode):
        texts: list[str] = []
        for token_id in token_ids:
            try:
                texts.append(str(decode([token_id], skip_special_tokens=False)))
            except TypeError:
                texts.append(str(decode([token_id])))
            except Exception:
                texts.append(str(token_id))
        return texts

    return [str(token_id) for token_id in token_ids]


def _apply_token_selective_dola_scores(
    *,
    tokenizer: Any,
    input_ids: Any,
    contrast_token_scores: Any,
    final_token_scores: Any,
    token_selective_mode: str,
    token_selective_config: TokenSelectiveDolaConfig | None = None,
) -> tuple[Any, list[bool], list[str], list[str], list[float], list[str]]:
    """Mix DoLa contrast scores through the requested token-selective policy."""
    try:
        import torch
    except ImportError as error:
        raise ImportError(
            "torch is required for token-selective DoLa scoring. "
            "Install the extra model dependencies from requirements-model.txt."
        ) from error

    target_token_ids = _tensor_first_row_to_int_list(input_ids[:, 1:])
    target_token_texts = _token_ids_to_texts(tokenizer, target_token_ids)
    normalized_mode = _normalize_token_selective_mode(token_selective_mode)

    if normalized_mode == TOKEN_SELECTIVE_DOLA_MODE_HEURISTIC_FACT_CRITICAL_V2_SOFT:
        resolved_config = _resolve_token_selective_config(token_selective_config)
        selected_mask, selected_reasons, contrast_weights, selection_tiers = (
            _select_token_selective_dola_tokens_v2(
                target_token_texts,
                token_selective_config=resolved_config,
            )
        )
        weight_tensor = torch.tensor(
            [contrast_weights],
            dtype=contrast_token_scores.dtype,
            device=contrast_token_scores.device,
        )
        effective_scores = final_token_scores + (weight_tensor * contrast_token_scores)
        score_sources = ["soft_mixed"] * len(selected_mask)
        return (
            effective_scores,
            selected_mask,
            selected_reasons,
            score_sources,
            contrast_weights,
            selection_tiers,
        )

    selected_mask, selected_reasons = _select_token_selective_dola_tokens(target_token_texts)
    mask_tensor = torch.tensor(
        [selected_mask],
        dtype=torch.bool,
        device=contrast_token_scores.device,
    )
    effective_scores = torch.where(mask_tensor, contrast_token_scores, final_token_scores)
    score_sources = [
        "contrast" if selected else "vanilla_final"
        for selected in selected_mask
    ]
    contrast_weights = [1.0 if selected else 0.0 for selected in selected_mask]
    selection_tiers = ["strong" if selected else "unselected" for selected in selected_mask]
    return (
        effective_scores,
        selected_mask,
        selected_reasons,
        score_sources,
        contrast_weights,
        selection_tiers,
    )


def _select_token_selective_dola_tokens(
    token_texts: list[str],
    *,
    token_selective_config: TokenSelectiveDolaConfig | None = None,
) -> tuple[list[bool], list[str]]:
    """Return conservative token-selection decisions for token-selective DoLa v1."""
    resolved_config = _resolve_token_selective_config(token_selective_config)
    selected_mask: list[bool] = []
    selected_reasons: list[str] = []
    for token_index, token_text in enumerate(token_texts):
        selected, reason = _select_token_for_token_selective_dola(
            token_text,
            token_index=token_index,
            token_selective_config=resolved_config,
        )
        selected_mask.append(selected)
        selected_reasons.append(reason)
    return selected_mask, selected_reasons


def _select_token_selective_dola_tokens_v2(
    token_texts: list[str],
    *,
    token_selective_config: TokenSelectiveDolaConfig | None = None,
) -> tuple[list[bool], list[str], list[float], list[str]]:
    """Return three-tier token-selection decisions for soft token-selective DoLa v2."""
    resolved_config = _resolve_token_selective_config(token_selective_config)
    strong_mask, strong_reasons = _select_token_selective_dola_tokens(
        token_texts,
        token_selective_config=resolved_config,
    )
    selected_mask: list[bool] = []
    selected_reasons: list[str] = []
    contrast_weights: list[float] = []
    selection_tiers: list[str] = []
    previous_strong_can_extend = False
    uses_word_boundaries = any(
        _has_tokenizer_word_boundary(token_text)
        for token_text in token_texts
    )

    for token_index, token_text in enumerate(token_texts):
        clean_token = _clean_token_for_token_selective_dola(token_text)
        stripped = _strip_token_selective_punctuation(clean_token)
        is_continuation = _is_tokenizer_continuation_piece(
            token_text,
            token_index=token_index,
            uses_word_boundaries=uses_word_boundaries,
        )
        is_alphaish = _is_alphaish_lexical_token(stripped)

        if strong_mask[token_index]:
            tier = "strong"
            reason = strong_reasons[token_index]
            weight = float(resolved_config.strong_weight)
            previous_strong_can_extend = is_alphaish and _has_tokenizer_word_boundary(token_text)
        elif (
            resolved_config.selector_enable_strong_continuation_inheritance
            and previous_strong_can_extend
            and is_continuation
            and is_alphaish
        ):
            tier = "strong"
            reason = "strong_continuation"
            weight = float(resolved_config.strong_weight)
            previous_strong_can_extend = True
        else:
            medium_reason = _medium_token_selective_reason_v2(
                stripped,
                token_index=token_index,
                strong_mask=strong_mask,
                token_selective_config=resolved_config,
            )
            if medium_reason:
                tier = "medium"
                reason = medium_reason
                weight = float(resolved_config.medium_weight)
            else:
                tier = "unselected"
                reason = ""
                weight = float(resolved_config.unselected_weight)
            previous_strong_can_extend = False

        selected_mask.append(tier != "unselected")
        selected_reasons.append(reason)
        contrast_weights.append(weight)
        selection_tiers.append(tier)

    return selected_mask, selected_reasons, contrast_weights, selection_tiers


def _select_token_for_token_selective_dola(
    token_text: str,
    *,
    token_index: int = 0,
    token_selective_config: TokenSelectiveDolaConfig | None = None,
) -> tuple[bool, str]:
    """Conservative lexical selector for fact-critical-ish answer tokens."""
    resolved_config = _resolve_token_selective_config(token_selective_config)
    clean_token = _clean_token_for_token_selective_dola(token_text)
    stripped = _strip_token_selective_punctuation(clean_token)
    if not stripped:
        return False, ""

    lower = stripped.lower()
    if resolved_config.selector_enable_number and any(char.isdigit() for char in stripped):
        return True, "number"
    if resolved_config.selector_enable_date_word and lower in _TOKEN_SELECTIVE_DATE_WORDS:
        return True, "date_word"
    if resolved_config.selector_enable_relation_word and lower in _TOKEN_SELECTIVE_RELATION_WORDS:
        return True, "relation_word"
    if (
        resolved_config.selector_enable_capitalized_lexical
        and _is_conservative_capitalized_token(token_text, stripped, token_index=token_index)
    ):
        return True, "capitalized_lexical"
    return False, ""


def _medium_token_selective_reason_v2(
    stripped_token: str,
    *,
    token_index: int,
    strong_mask: list[bool],
    token_selective_config: TokenSelectiveDolaConfig | None = None,
) -> str:
    """Allow obvious content words into v2 at a reduced contrast weight."""
    resolved_config = _resolve_token_selective_config(token_selective_config)
    if not _is_medium_lexical_token_v2(stripped_token):
        return ""
    adjacent_to_strong = (
        (token_index > 0 and strong_mask[token_index - 1])
        or (token_index + 1 < len(strong_mask) and strong_mask[token_index + 1])
    )
    if adjacent_to_strong and resolved_config.selector_enable_adjacent_support_medium:
        return "medium_adjacent_support"
    if resolved_config.selector_enable_lowercase_medium:
        return "medium_lexical"
    return ""


def _is_medium_lexical_token_v2(stripped_token: str) -> bool:
    if not _is_alphaish_lexical_token(stripped_token):
        return False
    if len(stripped_token) < 4:
        return False
    if stripped_token.lower() in _TOKEN_SELECTIVE_MEDIUM_STOPWORDS:
        return False
    return True


def _is_alphaish_lexical_token(stripped_token: str) -> bool:
    alpha_count = sum(1 for char in stripped_token if char.isalpha())
    return alpha_count >= 2 and all(char.isalpha() or char in {"-", "'"} for char in stripped_token)


def _strip_token_selective_punctuation(clean_token: str) -> str:
    return clean_token.strip(" \t\r\n\"'`.,;:!?()[]{}")


def _has_tokenizer_word_boundary(raw_token: str) -> bool:
    return str(raw_token).startswith(("Ġ", "▁"))


def _is_tokenizer_continuation_piece(
    raw_token: str,
    *,
    token_index: int,
    uses_word_boundaries: bool,
) -> bool:
    return token_index > 0 and uses_word_boundaries and not _has_tokenizer_word_boundary(raw_token)


def _clean_token_for_token_selective_dola(token_text: str) -> str:
    """Remove common tokenizer word-boundary markers without joining subwords."""
    return (
        str(token_text)
        .replace("Ġ", "")
        .replace("▁", "")
        .replace("Ċ", "")
        .strip()
    )


def _is_conservative_capitalized_token(
    raw_token: str,
    stripped_token: str,
    *,
    token_index: int,
) -> bool:
    """Select only lexical uppercase word starts, not arbitrary subword fragments."""
    if stripped_token in _TOKEN_SELECTIVE_CAPITALIZED_EXCLUSIONS:
        return False
    if not stripped_token[:1].isupper():
        return False
    if sum(1 for char in stripped_token if char.isalpha()) < 2:
        return False
    if token_index > 0 and not str(raw_token).startswith(("Ġ", "▁")):
        return False
    return True


def _aggregate_continuation_log_probs(
    continuation_log_probs: Any,
    score_mode: str,
) -> tuple[float, int]:
    """Aggregate continuation token log-probs with the requested normalization."""
    normalized_mode = _normalize_score_mode(score_mode)
    continuation_token_count = int(continuation_log_probs.shape[-1])
    if continuation_token_count <= 0:
        raise ValueError("candidate_answer did not add any continuation tokens.")

    total_logprob = continuation_log_probs.sum().item()
    if normalized_mode == "sum_logprob":
        return float(total_logprob), continuation_token_count
    return float(total_logprob / continuation_token_count), continuation_token_count



def _normalize_score_mode(score_mode: str) -> str:
    """Normalize the configurable candidate-scoring aggregation mode."""
    normalized_mode = score_mode.strip().lower()
    if normalized_mode not in {"sum_logprob", "mean_logprob"}:
        raise ValueError(
            f"Unsupported score_mode '{score_mode}'. Use 'sum_logprob' or 'mean_logprob'."
        )
    return normalized_mode



def _normalize_dola_score_mode(dola_score_mode: str) -> str:
    """Normalize the DoLa scoring rule used for candidate comparison."""
    normalized_mode = dola_score_mode.strip().lower()
    if normalized_mode not in {
        "legacy_contrastive",
        "official_static_dola",
        "official_dynamic_dola",
    }:
        raise ValueError(
            "Unsupported dola_score_mode "
            f"'{dola_score_mode}'. Use 'legacy_contrastive', 'official_static_dola', or 'official_dynamic_dola'."
        )
    return normalized_mode


def _normalize_token_selective_mode(token_selective_mode: str) -> str:
    """Normalize the opt-in token-selective DoLa selector mode."""
    normalized_mode = token_selective_mode.strip().lower()
    if normalized_mode == TOKEN_SELECTIVE_DOLA_MODE_HEURISTIC_FACT_CRITICAL_V1:
        return TOKEN_SELECTIVE_DOLA_MODE_HEURISTIC_FACT_CRITICAL_V1_HARD
    if normalized_mode not in {
        TOKEN_SELECTIVE_DOLA_MODE_HEURISTIC_FACT_CRITICAL_V1_HARD,
        TOKEN_SELECTIVE_DOLA_MODE_HEURISTIC_FACT_CRITICAL_V2_SOFT,
    }:
        raise ValueError(
            "Unsupported token_selective_mode "
            f"'{token_selective_mode}'. Use "
            f"'{TOKEN_SELECTIVE_DOLA_MODE_HEURISTIC_FACT_CRITICAL_V1_HARD}' or "
            f"'{TOKEN_SELECTIVE_DOLA_MODE_HEURISTIC_FACT_CRITICAL_V2_SOFT}'."
        )
    return normalized_mode



def _apply_relative_top_mask(
    diff_logits: Any,
    final_log_probs: Any,
    relative_top: float,
    relative_top_value: float,
) -> Any:
    """Mask low-probability tokens relative to the final-layer distribution."""
    try:
        import torch
    except ImportError as error:
        raise ImportError(
            "torch is required for DoLa-style candidate scoring. "
            "Install the extra model dependencies from requirements-model.txt."
        ) from error

    if relative_top <= 0.0:
        return diff_logits

    max_log_probs = final_log_probs.max(dim=-1, keepdim=True).values
    threshold = max_log_probs + math.log(relative_top)
    mask = final_log_probs < threshold
    return torch.where(mask, torch.full_like(diff_logits, relative_top_value), diff_logits)



def _resolve_mature_layer(mature_layer: int | None, num_hidden_layers: int) -> int:
    """Resolve the mature layer index from config or default final layer."""
    if mature_layer is None:
        return get_mature_layer_index(num_hidden_layers)
    validate_mature_layer(mature_layer, num_hidden_layers)
    return mature_layer



def _get_continuation_start_index(prompt_len: int) -> int:
    """Return the gathered-log-prob start index for continuation tokens."""
    return max(prompt_len - 1, 0)
