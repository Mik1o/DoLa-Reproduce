"""Minimal Hugging Face model-loading helpers for causal LM smoke tests."""

from __future__ import annotations

from typing import Any


def resolve_device(device: str | None) -> str:
    """Resolve a requested runtime device string."""
    if device is None:
        try:
            import torch
        except ImportError:
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"

    normalized = device.strip().lower()
    if not normalized:
        return resolve_device(None)
    if normalized == "cpu":
        return "cpu"
    if normalized.startswith("cuda"):
        try:
            import torch
        except ImportError as error:
            raise ImportError(
                "CUDA was requested, but PyTorch is not installed. "
                "Install the extra model dependencies from requirements-model.txt."
            ) from error
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but no CUDA device is available.")
        return normalized
    raise ValueError(f"Unsupported device '{device}'. Use 'cpu' or 'cuda'.")


def load_model_and_tokenizer(
    model_name: str,
    device: str | None = "cpu",
    **kwargs: Any,
) -> tuple[Any, Any]:
    """Load a tokenizer and causal language model for a minimal smoke test."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise ImportError(
            "transformers is required for model loading. "
            "Install the extra model dependencies from requirements-model.txt."
        ) from error

    resolved_device = resolve_device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.to(resolved_device)
    model.eval()
    return model, tokenizer