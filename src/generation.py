"""Generation entry points for vanilla and future DoLa decoding."""

from __future__ import annotations

from typing import Any



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