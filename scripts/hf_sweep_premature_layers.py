"""Sweep DoLa premature layers over a small TruthfulQA subset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.hf_eval_compare_subset import evaluate_compare_subset
from src.metrics import format_metrics, select_best_layer_summary
from src.modeling import load_model_and_tokenizer
from src.truthfulqa_mc import load_truthfulqa_samples
from src.utils import ensure_output_dir, load_yaml_config



def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the premature-layer sweep."""
    parser = argparse.ArgumentParser(
        description="Sweep DoLa premature layers over a small TruthfulQA subset."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "tinyllama_sweep_subset.yaml",
        help="Path to a YAML config file.",
    )
    return parser.parse_args()



def main() -> None:
    """Load model and samples once, then evaluate multiple premature layers."""
    args = parse_args()
    config = load_yaml_config(args.config)
    output_dir = ensure_output_dir(
        config.get("output_dir", PROJECT_ROOT / "outputs" / "tinyllama_sweep_subset")
    )

    model_name = str(config["model_name"])
    device = config.get("device", "cpu")
    csv_path = Path(str(config["csv_path"]))
    max_samples = int(config.get("max_samples", 1))
    prompt_style = str(config.get("prompt_style", "plain_mc"))
    premature_layers = config.get("premature_layers")
    if not isinstance(premature_layers, list) or not premature_layers:
        raise ValueError("premature_layers must be a non-empty list of layer indices.")

    model_kwargs: dict[str, object] = {}
    if "use_safetensors" in config:
        model_kwargs["use_safetensors"] = bool(config["use_safetensors"])
    if "torch_dtype" in config:
        model_kwargs["torch_dtype"] = str(config["torch_dtype"])

    samples = load_truthfulqa_samples(csv_path)
    model, tokenizer = load_model_and_tokenizer(
        model_name=model_name,
        device=device,
        **model_kwargs,
    )

    print(f"[hf_sweep_premature_layers] Model: {model_name}")
    print(f"[hf_sweep_premature_layers] Device: {device}")
    print(f"[hf_sweep_premature_layers] Loaded samples: {len(samples)}")
    print(f"[hf_sweep_premature_layers] Evaluating first {min(len(samples), max_samples)} samples")
    print(f"[hf_sweep_premature_layers] Output directory: {output_dir}")
    print("[hf_sweep_premature_layers] Layer sweep:")
    print("layer | vanilla_mc2 | dola_mc2 | delta_mc2 | vanilla_mc1 | dola_mc1")

    layer_summaries: list[dict[str, object]] = []
    for raw_layer in premature_layers:
        premature_layer = int(raw_layer)
        _, comparison_summary = evaluate_compare_subset(
            model,
            tokenizer,
            samples,
            max_samples=max_samples,
            premature_layer=premature_layer,
            prompt_style=prompt_style,
        )
        comparison_summary.update(
            {
                "task_name": str(config.get("task_name", "tinyllama_sweep_subset")),
                "model_name": model_name,
                "csv_path": str(csv_path),
                "output_dir": str(output_dir),
            }
        )
        layer_summaries.append(comparison_summary)
        print(
            f"{premature_layer:>5} | "
            f"{float(comparison_summary['vanilla_avg_mc2']):>11.4f} | "
            f"{float(comparison_summary['dola_avg_mc2']):>8.4f} | "
            f"{float(comparison_summary['delta_mc2']):>9.4f} | "
            f"{float(comparison_summary['vanilla_avg_mc1']):>11.4f} | "
            f"{float(comparison_summary['dola_avg_mc1']):>8.4f}"
        )

    best_summary = select_best_layer_summary(layer_summaries)
    best_summary = dict(best_summary)
    best_summary["selection_rule"] = "max dola_avg_mc2, then max dola_avg_mc1"

    results_path = output_dir / "layer_sweep_results.jsonl"
    with results_path.open("w", encoding="utf-8") as handle:
        for summary in layer_summaries:
            handle.write(json.dumps(summary, ensure_ascii=False) + "\n")

    best_path = output_dir / "best_layer_summary.json"
    with best_path.open("w", encoding="utf-8") as handle:
        json.dump(best_summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(f"[hf_sweep_premature_layers] Best layer: {best_summary['premature_layer']}")
    print(f"[hf_sweep_premature_layers] Best summary: {format_metrics(best_summary)}")
    print(f"[hf_sweep_premature_layers] Saved sweep results to: {results_path}")
    print(f"[hf_sweep_premature_layers] Saved best summary to: {best_path}")


if __name__ == "__main__":
    main()