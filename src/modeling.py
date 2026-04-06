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
    """Load a tokenizer and causal language model with optional 4-bit support."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer
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

    tokenizer_class = tokenizer_load_kwargs.pop("tokenizer_class", None)
    model_load_kwargs.pop("tokenizer_class", None)

    trust_remote_code = bool(model_load_kwargs.pop("trust_remote_code", False))
    tokenizer_load_kwargs["trust_remote_code"] = trust_remote_code
    model_load_kwargs["trust_remote_code"] = trust_remote_code

    attn_implementation = model_load_kwargs.pop("attn_implementation", None)
    tokenizer_load_kwargs.pop("attn_implementation", None)
    if attn_implementation is not None:
        model_load_kwargs["attn_implementation"] = str(attn_implementation)

    local_files_only = tokenizer_load_kwargs.pop("local_files_only", None)
    if local_files_only is not None:
        tokenizer_load_kwargs["local_files_only"] = bool(local_files_only)
        model_load_kwargs["local_files_only"] = bool(local_files_only)
    model_load_kwargs.pop("local_files_only", None)

    use_4bit = bool(model_load_kwargs.pop("use_4bit", False))
    tokenizer_load_kwargs.pop("use_4bit", None)
    device_map = model_load_kwargs.pop("device_map", None)
    tokenizer_load_kwargs.pop("device_map", None)

    if use_4bit:
        if resolved_device != "cuda" and not resolved_device.startswith("cuda"):
            raise ValueError("4-bit loading currently expects a CUDA device.")
        quantization_config = _build_4bit_quantization_config(
            model_load_kwargs=model_load_kwargs,
            resolved_dtype=resolved_dtype,
        )
        model_load_kwargs["quantization_config"] = quantization_config
        model_load_kwargs["device_map"] = device_map or "auto"
    elif device_map is not None:
        model_load_kwargs["device_map"] = device_map

    try:
        tokenizer = _load_tokenizer(
            model_name=model_name,
            auto_tokenizer_cls=AutoTokenizer,
            llama_tokenizer_cls=LlamaTokenizer,
            tokenizer_class=tokenizer_class,
            tokenizer_load_kwargs=tokenizer_load_kwargs,
        )
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
            f"Check torch_dtype/attn_implementation/trust_remote_code/4-bit settings. Original error: {error}"
        ) from error

    if not use_4bit and device_map is None:
        model.to(resolved_device)
    model.eval()
    if getattr(model, "generation_config", None) is not None and tokenizer.pad_token_id is not None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer




def _load_tokenizer(
    *,
    model_name: str,
    auto_tokenizer_cls: Any,
    llama_tokenizer_cls: Any,
    tokenizer_class: str | None,
    tokenizer_load_kwargs: dict[str, Any],
) -> Any:
    """Load a tokenizer with an explicit slow-LLaMA fallback for OpenLLaMA-style models."""
    normalized_tokenizer_class = None if tokenizer_class is None else str(tokenizer_class).strip()
    use_fast = bool(tokenizer_load_kwargs.get("use_fast", True))

    if normalized_tokenizer_class == "LlamaTokenizer":
        return llama_tokenizer_cls.from_pretrained(model_name, **_without_use_fast(tokenizer_load_kwargs))

    try:
        return auto_tokenizer_cls.from_pretrained(model_name, **tokenizer_load_kwargs)
    except Exception as error:
        error_text = str(error)
        should_try_llama_slow = (
            not use_fast
            and (
                "open_llama" in model_name.lower()
                or "tiktoken" in error_text.lower()
                or normalized_tokenizer_class == "AutoTokenizer"
            )
        )
        if not should_try_llama_slow:
            raise
        try:
            return llama_tokenizer_cls.from_pretrained(
                model_name,
                **_without_use_fast(tokenizer_load_kwargs),
            )
        except Exception:
            raise error



def _without_use_fast(tokenizer_load_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Drop use_fast for slow tokenizer classes that do not accept it."""
    result = dict(tokenizer_load_kwargs)
    result.pop("use_fast", None)
    return result

def _build_4bit_quantization_config(
    *,
    model_load_kwargs: dict[str, Any],
    resolved_dtype: Any,
) -> Any:
    """Build a BitsAndBytes 4-bit quantization config from YAML fields."""
    try:
        import accelerate  # noqa: F401
        import bitsandbytes  # noqa: F401
        import torch
        from transformers import BitsAndBytesConfig
    except ImportError as error:
        raise ImportError(
            "4-bit loading requires bitsandbytes, accelerate, and transformers support. "
            "Install the extra model dependencies from requirements-model.txt."
        ) from error

    compute_dtype = _resolve_torch_dtype(model_load_kwargs.pop("bnb_4bit_compute_dtype", None))
    quant_type = str(model_load_kwargs.pop("bnb_4bit_quant_type", "nf4"))
    use_double_quant = bool(model_load_kwargs.pop("bnb_4bit_use_double_quant", True))

    if compute_dtype is None:
        if resolved_dtype in {torch.float16, torch.bfloat16, torch.float32}:
            compute_dtype = resolved_dtype
        else:
            compute_dtype = torch.float16
    if compute_dtype == "auto":
        compute_dtype = torch.float16

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type=quant_type,
        bnb_4bit_use_double_quant=use_double_quant,
    )



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
    if normalized == "float64":
        return torch.float64
    raise ValueError(
        f"Unsupported torch_dtype '{torch_dtype}'. "
        "Use 'float16', 'float32', 'bfloat16', 'float64', or 'auto'."
    )
