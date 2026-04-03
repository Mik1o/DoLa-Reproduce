"""Compare vanilla and DoLa-style scoring on one TruthfulQA multiple-choice sample."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dola_utils import describe_dola_pair, get_mature_layer_index, validate_premature_layer
from src.generation import score_candidate_answers, score_candidate_answers_dola
from src.metrics import compute_mc_metrics, format_metrics
from src.modeling import load_model_and_tokenizer
from src.truthfulqa_mc import build_mc_prompt, get_mc_candidate_sets, load_truthfulqa_samples
from src.utils import ensure_output_dir, load_yaml_config


LOAD_CONFIG_KEYS = (
    "use_safetensors",
    "torch_dtype",
    "use_fast_tokenizer",
    "trust_remote_code",
    "attn_implementation",
)



def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the single-sample compare script."""
    parser = argparse.ArgumentParser(
        description="Compare vanilla and DoLa-style scoring on one TruthfulQA-MC sample."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "hf_tiny_compare_single.yaml",
        help="Path to a YAML config file.",
    )
    return parser.parse_args()



def main() -> None:
    """Load one sample, score it twice, and save a compact comparison JSON."""
    args = parse_args()
    config = load_yaml_config(args.config)
    output_dir = ensure_output_dir(
        config.get("output_dir", PROJECT_ROOT / "outputs" / "hf_tiny_compare_single")
    )

    model_name = str(config["model_name"])
    device = config.get("device", "cpu")
    csv_path = Path(str(config["csv_path"]))
    sample_index = int(config.get("sample_index", 0))
    premature_layer = int(config["premature_layer"])
    prompt_style = str(config.get("prompt_style", "plain_mc"))

    model_kwargs = {key: config[key] for key in LOAD_CONFIG_KEYS if key in config}

    samples = load_truthfulqa_samples(csv_path)
    if sample_index < 0 or sample_index >= len(samples):
        raise IndexError(
            f"sample_index {sample_index} is out of range for {len(samples)} loaded samples."
        )

    sample = samples[sample_index]
    prompt = build_mc_prompt(sample, prompt_style=prompt_style)
    true_candidates, false_candidates = get_mc_candidate_sets(sample)

    print(f"[hf_compare_single_mc] Model: {model_name}")
    print(f"[hf_compare_single_mc] Device: {device}")
    print(f"[hf_compare_single_mc] Output directory: {output_dir}")

    model, tokenizer = load_model_and_tokenizer(
        model_name=model_name,
        device=device,
        **model_kwargs,
    )

    num_hidden_layers = int(getattr(model.config, "num_hidden_layers", 0))
    validate_premature_layer(premature_layer, num_hidden_layers)
    mature_layer = get_mature_layer_index(num_hidden_layers)
    pair_description = describe_dola_pair(premature_layer, mature_layer)

    vanilla_true = score_candidate_answers(model, tokenizer, prompt, true_candidates)
    vanilla_false = score_candidate_answers(model, tokenizer, prompt, false_candidates)
    vanilla_metrics = compute_mc_metrics(
        [score for _, score in vanilla_true],
        [score for _, score in vanilla_false],
    )

    dola_true = score_candidate_answers_dola(
        model,
        tokenizer,
        prompt,
        true_candidates,
        premature_layer=premature_layer,
    )
    dola_false = score_candidate_answers_dola(
        model,
        tokenizer,
        prompt,
        false_candidates,
        premature_layer=premature_layer,
    )
    dola_metrics = compute_mc_metrics(
        [score for _, score in dola_true],
        [score for _, score in dola_false],
    )

    print("[hf_compare_single_mc] Question:")
    print(sample.question)
    print("[hf_compare_single_mc] Prompt:")
    print(prompt)
    print(f"[hf_compare_single_mc] DoLa pair: {pair_description}")
    print(f"[hf_compare_single_mc] Vanilla metrics: {format_metrics(vanilla_metrics)}")
    print(f"[hf_compare_single_mc] DoLa metrics: {format_metrics(dola_metrics)}")

    result = {
        "question": sample.question,
        "prompt": prompt,
        "prompt_style": prompt_style,
        "premature_layer": premature_layer,
        "mature_layer": mature_layer,
        "vanilla": {
            "true_scores": [{"candidate": c, "score": s} for c, s in vanilla_true],
            "false_scores": [{"candidate": c, "score": s} for c, s in vanilla_false],
            "metrics": vanilla_metrics,
        },
        "dola": {
            "true_scores": [{"candidate": c, "score": s} for c, s in dola_true],
            "false_scores": [{"candidate": c, "score": s} for c, s in dola_false],
            "metrics": dola_metrics,
        },
    }

    output_path = output_dir / "compare_single_result.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(f"[hf_compare_single_mc] Saved comparison to: {output_path}")


if __name__ == "__main__":
    main()
