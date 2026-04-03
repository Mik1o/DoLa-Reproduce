"""Score one TruthfulQA multiple-choice sample with a tiny HF causal LM."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.generation import score_candidate_answers
from src.metrics import compute_mc_metrics, format_metrics
from src.modeling import load_model_and_tokenizer
from src.truthfulqa_mc import build_mc_prompt, get_mc_candidate_sets, load_truthfulqa_samples
from src.utils import ensure_output_dir, load_yaml_config



def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the single-sample scoring script."""
    parser = argparse.ArgumentParser(
        description="Score one TruthfulQA-MC sample with a tiny Hugging Face causal LM."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "hf_tiny_score_single.yaml",
        help="Path to a YAML config file.",
    )
    return parser.parse_args()



def main() -> None:
    """Load one sample, score candidates, and print MC metrics."""
    args = parse_args()
    config = load_yaml_config(args.config)
    output_dir = ensure_output_dir(
        config.get("output_dir", PROJECT_ROOT / "outputs" / "hf_tiny_score_single")
    )

    model_name = str(config["model_name"])
    device = config.get("device", "cpu")
    csv_path = Path(str(config["csv_path"]))
    sample_index = int(config.get("sample_index", 0))

    model_kwargs: dict[str, object] = {}
    if "use_safetensors" in config:
        model_kwargs["use_safetensors"] = bool(config["use_safetensors"])

    samples = load_truthfulqa_samples(csv_path)
    if sample_index < 0 or sample_index >= len(samples):
        raise IndexError(
            f"sample_index {sample_index} is out of range for {len(samples)} loaded samples."
        )

    sample = samples[sample_index]
    prompt = build_mc_prompt(sample)
    true_candidates, false_candidates = get_mc_candidate_sets(sample)

    print(f"[hf_score_single_mc] Model: {model_name}")
    print(f"[hf_score_single_mc] Device: {device}")
    print(f"[hf_score_single_mc] Output directory: {output_dir}")

    model, tokenizer = load_model_and_tokenizer(
        model_name=model_name,
        device=device,
        **model_kwargs,
    )

    true_scored = score_candidate_answers(model, tokenizer, prompt, true_candidates)
    false_scored = score_candidate_answers(model, tokenizer, prompt, false_candidates)

    scores_true = [score for _, score in true_scored]
    scores_false = [score for _, score in false_scored]
    metrics = compute_mc_metrics(scores_true, scores_false)

    print("[hf_score_single_mc] Question:")
    print(sample.question)
    print("[hf_score_single_mc] Prompt:")
    print(prompt)
    print("[hf_score_single_mc] True candidates:")
    for candidate, score in true_scored:
        print(f"  {candidate}: {score:.4f}")
    print("[hf_score_single_mc] False candidates:")
    for candidate, score in false_scored:
        print(f"  {candidate}: {score:.4f}")
    print(f"[hf_score_single_mc] Metrics: {format_metrics(metrics)}")


if __name__ == "__main__":
    main()