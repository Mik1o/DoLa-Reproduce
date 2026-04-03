"""Run a minimal Hugging Face causal LM smoke test."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.generation import generate_vanilla
from src.modeling import load_model_and_tokenizer
from src.utils import ensure_output_dir, load_yaml_config


LOAD_CONFIG_KEYS = (
    "use_safetensors",
    "torch_dtype",
    "use_fast_tokenizer",
    "trust_remote_code",
    "attn_implementation",
)



def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the HF smoke test."""
    parser = argparse.ArgumentParser(
        description="Load a tiny Hugging Face causal LM and run a vanilla generate."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "hf_tiny_smoke.yaml",
        help="Path to a YAML config file.",
    )
    return parser.parse_args()



def main() -> None:
    """Load config, run a tiny HF generation, and print the result."""
    args = parse_args()
    config = load_yaml_config(args.config)
    output_dir = ensure_output_dir(
        config.get("output_dir", PROJECT_ROOT / "outputs" / "hf_tiny_smoke")
    )

    model_name = str(config["model_name"])
    device = config.get("device", "cpu")
    prompt = str(config["prompt"])
    max_new_tokens = int(config.get("max_new_tokens", 20))
    do_sample = bool(config.get("do_sample", False))

    model_kwargs = {key: config[key] for key in LOAD_CONFIG_KEYS if key in config}

    print(f"[hf_smoke_test] Model: {model_name}")
    print(f"[hf_smoke_test] Device: {device}")
    print(f"[hf_smoke_test] Output directory: {output_dir}")

    model, tokenizer = load_model_and_tokenizer(
        model_name=model_name,
        device=device,
        **model_kwargs,
    )
    generated_text = generate_vanilla(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
    )

    print("[hf_smoke_test] Prompt:")
    print(prompt)
    print("[hf_smoke_test] Generated text:")
    print(generated_text)


if __name__ == "__main__":
    main()
