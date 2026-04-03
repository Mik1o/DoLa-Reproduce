"""Generation entry points for vanilla decoding and candidate scoring."""

from __future__ import annotations

from typing import Any

from src.dola_utils import get_mature_layer_index, validate_premature_layer



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
) -> float:
    """Score a candidate continuation by summing token log-probabilities."""
    try:
        import torch
    except ImportError as error:
        raise ImportError(
            "torch is required for candidate scoring. "
            "Install the extra model dependencies from requirements-model.txt."
        ) from error

    prompt_text, input_ids, attention_mask, prompt_len = _prepare_scoring_inputs(
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

    continuation_score = token_log_probs[:, _get_continuation_start_index(prompt_len) :].sum().item()
    return float(continuation_score)



def score_candidate_answers(
    model: Any,
    tokenizer: Any,
    prompt: str,
    candidate_answers: list[str],
    separator: str = " ",
) -> list[tuple[str, float]]:
    """Score each candidate answer for a shared prompt."""
    if not candidate_answers:
        raise ValueError("candidate_answers must contain at least one answer.")

    scored_candidates: list[tuple[str, float]] = []
    for candidate_answer in candidate_answers:
        score = score_continuation_logprob(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            candidate_answer=candidate_answer,
            separator=separator,
        )
        scored_candidates.append((candidate_answer, score))
    return scored_candidates



def score_continuation_dola_logprob(
    model: Any,
    tokenizer: Any,
    prompt: str,
    candidate_answer: str,
    premature_layer: int,
    separator: str = " ",
) -> float:
    """Score a continuation with a minimal DoLa-style contrastive log-prob sum.

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
    mature_layer = get_mature_layer_index(num_hidden_layers)

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

        mature_hidden = hidden_states[-1]
        premature_hidden = hidden_states[premature_layer + 1]

        mature_logits = lm_head(mature_hidden[:, :-1, :])
        premature_logits = lm_head(premature_hidden[:, :-1, :])
        contrastive_logits = mature_logits - premature_logits
        token_log_probs = _gather_token_log_probs(contrastive_logits, input_ids)

    continuation_score = token_log_probs[:, _get_continuation_start_index(prompt_len) :].sum().item()
    return float(continuation_score)



def score_candidate_answers_dola(
    model: Any,
    tokenizer: Any,
    prompt: str,
    candidate_answers: list[str],
    premature_layer: int,
    separator: str = " ",
) -> list[tuple[str, float]]:
    """Score each candidate answer with minimal DoLa-style contrastive scoring."""
    if not candidate_answers:
        raise ValueError("candidate_answers must contain at least one answer.")

    scored_candidates: list[tuple[str, float]] = []
    for candidate_answer in candidate_answers:
        score = score_continuation_dola_logprob(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            candidate_answer=candidate_answer,
            premature_layer=premature_layer,
            separator=separator,
        )
        scored_candidates.append((candidate_answer, score))
    return scored_candidates



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

    prompt_text = prompt if prompt.endswith((" ", "\n", "\t")) else f"{prompt}{separator}"
    full_text = f"{prompt_text}{candidate_answer}"

    prompt_batch = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
    full_batch = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)

    prompt_len = int(prompt_batch["input_ids"].shape[1])
    full_len = int(full_batch["input_ids"].shape[1])
    if full_len <= prompt_len:
        raise ValueError("candidate_answer did not add any continuation tokens.")

    input_ids = full_batch["input_ids"].to(model.device)
    attention_mask = full_batch["attention_mask"].to(model.device)
    return prompt_text, input_ids, attention_mask, prompt_len



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



def _get_continuation_start_index(prompt_len: int) -> int:
    """Return the gathered-log-prob start index for continuation tokens."""
    return max(prompt_len - 1, 0)