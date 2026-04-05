"""Generation entry points for vanilla decoding and candidate scoring."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from src.dola_utils import get_mature_layer_index, validate_premature_layer


@dataclass(slots=True)
class CandidateScore:
    candidate: str
    score: float
    continuation_token_count: int



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
    encoded = encoded.to(model.device)
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
        )
        for candidate_answer in candidate_answers
    ]



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
) -> CandidateScore:
    """Score one candidate with configurable DoLa-style scoring and token counts.

    Notes
    -----
    For decoder-only models, ``hidden_states[0]`` is the embedding output and
    ``hidden_states[k + 1]`` corresponds to the output of decoder block ``k``.
    The mature representation is taken from ``hidden_states[-1]``.
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
    validate_premature_layer(premature_layer, num_hidden_layers)

    lm_head = model.get_output_embeddings()
    if lm_head is None:
        raise ValueError("The model does not expose output embeddings for DoLa-style scoring.")

    normalized_dola_mode = _normalize_dola_score_mode(dola_score_mode)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise ValueError("The model did not return hidden_states for DoLa-style scoring.")

        mature_hidden = hidden_states[-1]
        premature_hidden = hidden_states[premature_layer + 1]
        mature_logits = lm_head(mature_hidden[:, :-1, :])
        premature_logits = lm_head(premature_hidden[:, :-1, :])

        if normalized_dola_mode == "legacy_contrastive":
            token_scores = _gather_token_log_probs(mature_logits - premature_logits, input_ids)
        else:
            final_log_probs = torch.log_softmax(mature_logits, dim=-1)
            base_log_probs = torch.log_softmax(premature_logits, dim=-1)
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
            token_scores = _gather_scores_at_target_ids(diff_logits, input_ids)

    continuation_scores = token_scores[:, _get_continuation_start_index(prompt_len) :]
    score, continuation_token_count = _aggregate_continuation_log_probs(
        continuation_log_probs=continuation_scores,
        score_mode=score_mode,
    )
    return CandidateScore(candidate=candidate_answer, score=score, continuation_token_count=continuation_token_count)



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
            input_ids = full_batch["input_ids"].to(model.device)
            attention_mask = full_batch["attention_mask"].to(model.device)
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

    input_ids = full_batch["input_ids"].to(model.device)
    attention_mask = full_batch["attention_mask"].to(model.device)
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
    if normalized_mode not in {"legacy_contrastive", "official_static_dola"}:
        raise ValueError(
            "Unsupported dola_score_mode "
            f"'{dola_score_mode}'. Use 'legacy_contrastive' or 'official_static_dola'."
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



def _get_continuation_start_index(prompt_len: int) -> int:
    """Return the gathered-log-prob start index for continuation tokens."""
    return max(prompt_len - 1, 0)
