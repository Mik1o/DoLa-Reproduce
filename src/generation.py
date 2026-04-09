"""Generation entry points for vanilla decoding and candidate scoring."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

from src.dola_utils import (
    get_mature_layer_index,
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


@dataclass(slots=True)
class CandidateScore:
    candidate: str
    score: float
    continuation_token_count: int
    premature_layer_dist: dict[int, int] | None = None


@dataclass(slots=True)
class MultiConfigScoreResult:
    vanilla: list[CandidateScore]
    static: dict[int, list[CandidateScore]]
    dynamic: dict[str, list[CandidateScore]]
    batch_size: int
    batch_count: int
    profile: dict[str, Any] | None = None



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

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1, :]
        token_log_probs = _gather_token_log_probs(logits, input_ids)

    continuation_log_probs = token_log_probs[:, _get_continuation_start_index(prompt_len) :]
    score, continuation_token_count = _aggregate_continuation_log_probs(
        continuation_log_probs=continuation_log_probs,
        score_mode=score_mode,
    )
    return CandidateScore(candidate=candidate_answer, score=score, continuation_token_count=continuation_token_count)



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
    )
    resolved_dynamic_buckets = {
        str(name): validate_candidate_premature_layers(
            layers,
            resolved_mature_layer,
            num_hidden_layers,
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

    all_needed_layers = sorted(
        {
            *resolved_static_layers,
            *(layer for layers in resolved_dynamic_buckets.values() for layer in layers),
        }
    )

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

        with torch.no_grad():
            forward_start = time.perf_counter()
            outputs = model(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask,
                output_hidden_states=True,
            )
            model_forward_sec += time.perf_counter() - forward_start
            hidden_states = outputs.hidden_states
            if hidden_states is None:
                raise ValueError("The model did not return hidden_states for multi-config scoring.")

            mature_hidden = hidden_states[resolved_mature_layer + 1]
            mature_logits = lm_head(mature_hidden[:, :-1, :])
            mature_log_probs = torch.log_softmax(mature_logits, dim=-1)
            mature_probs = mature_log_probs.exp()
            vanilla_scores = _gather_scores_at_target_ids(mature_log_probs, batch_input_ids)

            layer_logits = {
                layer: lm_head(hidden_states[layer + 1][:, :-1, :])
                for layer in all_needed_layers
            }
            layer_log_probs = {
                layer: torch.log_softmax(logits, dim=-1)
                for layer, logits in layer_logits.items()
            }
            layer_probs = {
                layer: log_probs.exp()
                for layer, log_probs in layer_log_probs.items()
            }
            static_start = time.perf_counter()
            static_scores = {
                layer: _compute_official_dola_token_scores_from_log_probs(
                    final_log_probs=mature_log_probs,
                    base_log_probs=layer_log_probs[layer],
                    input_ids=batch_input_ids,
                    post_softmax=post_softmax,
                    relative_top=relative_top,
                    relative_top_value=relative_top_value,
                )
                for layer in resolved_static_layers
            }
            static_rescore_sec += time.perf_counter() - static_start
            dynamic_scores: dict[str, Any] = {}
            dynamic_selected_layers: dict[str, list[list[int]]] = {}
            dynamic_start = time.perf_counter()
            for bucket_name, candidate_layers in resolved_dynamic_buckets.items():
                candidate_log_probs = torch.stack(
                    [layer_log_probs[layer] for layer in candidate_layers],
                    dim=0,
                )
                candidate_probs = torch.stack(
                    [layer_probs[layer] for layer in candidate_layers],
                    dim=0,
                )
                js_divergence = _compute_dynamic_js_divergence_from_distributions(
                    mature_probs=mature_probs,
                    mature_log_probs=mature_log_probs,
                    candidate_probs=candidate_probs,
                    candidate_log_probs=candidate_log_probs,
                )
                selected_indices = js_divergence.argmax(dim=0)
                candidate_log_probs_by_batch = candidate_log_probs.permute(1, 2, 0, 3)
                gather_index = selected_indices.unsqueeze(-1).unsqueeze(-1).expand(
                    -1,
                    -1,
                    1,
                    candidate_log_probs_by_batch.shape[-1],
                )
                base_log_probs = candidate_log_probs_by_batch.gather(2, gather_index).squeeze(2)
                dynamic_scores[bucket_name] = _compute_official_dola_token_scores_from_log_probs(
                    final_log_probs=mature_log_probs,
                    base_log_probs=base_log_probs,
                    input_ids=batch_input_ids,
                    post_softmax=post_softmax,
                    relative_top=relative_top,
                    relative_top_value=relative_top_value,
                )
                dynamic_selected_layers[bucket_name] = [
                    [candidate_layers[int(index)] for index in batch_indices]
                    for batch_indices in selected_indices.tolist()
                ]
            dynamic_rescore_sec += time.perf_counter() - dynamic_start

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
                continuation_layers = dynamic_selected_layers[bucket_name][batch_index][
                    _get_continuation_start_index(prompt_len) : valid_target_length
                ]
                results.dynamic[bucket_name].append(
                    CandidateScore(
                        candidate=candidate_answer,
                        score=score,
                        continuation_token_count=token_count,
                        premature_layer_dist=_count_premature_layer_usage(
                            continuation_layers,
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
) -> CandidateScore:
    """Score one candidate with configurable DoLa-style scoring and token counts.

    Notes
    -----
    For decoder-only models, ``hidden_states[0]`` is the embedding output and
    ``hidden_states[k + 1]`` corresponds to the output of decoder block ``k``.
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
    resolved_premature_layer: int | None = None
    resolved_candidate_layers: list[int] = []

    if normalized_dola_mode in {"legacy_contrastive", "official_static_dola"}:
        resolved_premature_layer = validate_candidate_premature_layers(
            [premature_layer],
            resolved_mature_layer,
            num_hidden_layers,
        )[0]
    else:
        resolved_candidate_layers = validate_candidate_premature_layers(
            candidate_premature_layers,
            resolved_mature_layer,
            num_hidden_layers,
        )

    lm_head = model.get_output_embeddings()
    if lm_head is None:
        raise ValueError("The model does not expose output embeddings for DoLa-style scoring.")

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise ValueError("The model did not return hidden_states for DoLa-style scoring.")

        mature_hidden = hidden_states[resolved_mature_layer + 1]
        mature_logits = lm_head(mature_hidden[:, :-1, :])

        if normalized_dola_mode == "legacy_contrastive":
            premature_hidden = hidden_states[resolved_premature_layer + 1]
            premature_logits = lm_head(premature_hidden[:, :-1, :])
            token_scores = _gather_token_log_probs(mature_logits - premature_logits, input_ids)
            selected_layers: list[int] = [resolved_premature_layer] * int(mature_logits.shape[1])
        elif normalized_dola_mode == "official_static_dola":
            premature_hidden = hidden_states[resolved_premature_layer + 1]
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
                [lm_head(hidden_states[layer + 1][:, :-1, :]) for layer in resolved_candidate_layers],
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
    return CandidateScore(
        candidate=candidate_answer,
        score=score,
        continuation_token_count=continuation_token_count,
        premature_layer_dist=premature_layer_dist or None,
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

    mixture = 0.5 * (mature_probs.unsqueeze(0) + candidate_probs)
    kl_mature = F.kl_div(
        mature_log_probs.unsqueeze(0).expand_as(candidate_log_probs),
        mixture,
        reduction="none",
    ).mean(dim=-1)
    kl_candidate = F.kl_div(
        candidate_log_probs,
        mixture,
        reduction="none",
    ).mean(dim=-1)
    return 0.5 * (kl_mature + kl_candidate)



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
) -> tuple[Any, list[list[int]]]:
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
    selected_layers = [
        [candidate_layers[int(index)] for index in batch_indices]
        for batch_indices in selected_indices.tolist()
    ]
    return base_logits, selected_layers



def _select_dynamic_base_logits(
    mature_logits: Any,
    candidate_logits: Any,
    candidate_layers: list[int],
) -> tuple[Any, list[int]]:
    """Pick one premature layer per token position via JS divergence."""
    base_logits, selected_layers = _select_dynamic_base_logits_batched(
        mature_logits=mature_logits,
        candidate_logits=candidate_logits,
        candidate_layers=candidate_layers,
    )
    if len(selected_layers) != 1:
        raise ValueError("official_dynamic_dola single-candidate path expects batch size 1 scoring.")
    return base_logits, selected_layers[0]



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
