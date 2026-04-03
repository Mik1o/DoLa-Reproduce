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
            "transformers and torch are required for model loading. "
            "Install the extra model dependencies from requirements-model.txt."
        ) from error

    resolved_device = resolve_device(device)
    tokenizer_load_kwargs = dict(kwargs)
    model_load_kwargs = dict(kwargs)

    resolved_dtype = _resolve_torch_dtype(model_load_kwargs.pop("torch_dtype", None))
    tokenizer_load_kwargs.pop("torch_dtype", None)
    if resolved_dtype is not None:
        model_load_kwargs["dtype"] = resolved_dtype

    use_fast_tokenizer = tokenizer_load_kwargs.pop("use_fast_tokenizer", None)
    if use_fast_tokenizer is not None:
        tokenizer_load_kwargs["use_fast"] = bool(use_fast_tokenizer)
    model_load_kwargs.pop("use_fast_tokenizer", None)

    trust_remote_code = bool(model_load_kwargs.pop("trust_remote_code", False))
    tokenizer_load_kwargs["trust_remote_code"] = trust_remote_code
    model_load_kwargs["trust_remote_code"] = trust_remote_code

    attn_implementation = model_load_kwargs.pop("attn_implementation", None)
    tokenizer_load_kwargs.pop("attn_implementation", None)
    if attn_implementation is not None:
        model_load_kwargs["attn_implementation"] = str(attn_implementation)

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_load_kwargs)
    except Exception as error:
        raise RuntimeError(f"Failed to load tokenizer for '{model_name}': {error}") from error

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            raise RuntimeError(
                f"Tokenizer for '{model_name}' has neither pad_token nor eos_token/unk_token."
            )
    tokenizer.padding_side = "left"

    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_load_kwargs)
    except Exception as error:
        raise RuntimeError(
            f"Failed to load model for '{model_name}'. "
            f"Check torch_dtype/attn_implementation/trust_remote_code settings. Original error: {error}"
        ) from error

    model.to(resolved_device)
    model.eval()
    if getattr(model, "generation_config", None) is not None and tokenizer.pad_token_id is not None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer



def _resolve_torch_dtype(torch_dtype: object) -> Any:
    """Resolve an optional config torch_dtype field into a torch dtype."""
    if torch_dtype is None:
        return None

    import torch

    if not isinstance(torch_dtype, str):
        raise ValueError("torch_dtype must be a string such as 'float16', 'float32', or 'auto'.")

    normalized = torch_dtype.strip().lower()
    if normalized == "auto":
        return "auto"
    if normalized == "float16":
        return torch.float16
    if normalized == "float32":
        return torch.float32
    if normalized == "bfloat16":
        return torch.bfloat16
    raise ValueError(
        f"Unsupported torch_dtype '{torch_dtype}'. "
        "Use 'float16', 'float32', 'bfloat16', or 'auto'."
    )
