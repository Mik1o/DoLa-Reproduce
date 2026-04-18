"""Compare vanilla and DoLa-style MC scoring over a small TruthfulQA subset."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis_logging import (
    TruthfulQAMCAnalysisLogger,
    maybe_create_truthfulqa_mc_analysis_logger,
)
from src.dola_utils import (
    describe_dola_pair,
    get_mature_layer_index,
    validate_candidate_premature_layers,
    validate_mature_layer,
)
from src.generation import (
    TokenSelectiveDolaConfig,
    score_candidate_answers_dola_with_details,
    score_candidate_answers_with_details,
)
from src.metrics import (
    aggregate_mc_metrics,
    compare_aggregate_metrics,
    compute_mc_metrics,
    format_metrics,
)
from src.modeling import load_model_and_tokenizer
from src.truthfulqa_mc import TruthfulQASample, build_mc_prompt, get_mc_candidate_sets, load_truthfulqa_samples
from src.utils import ensure_output_dir, load_yaml_config


LOAD_CONFIG_KEYS = (
    "use_safetensors",
    "torch_dtype",
    "use_fast_tokenizer",
    "trust_remote_code",
    "attn_implementation",
    "local_files_only",
    "device_map",
    "use_4bit",
    "bnb_4bit_compute_dtype",
    "bnb_4bit_quant_type",
    "bnb_4bit_use_double_quant",
    "tokenizer_class",
)

PROGRESS_LOG_SECONDS = 30.0
PROGRESS_SAMPLE_STRIDE = 5
PROGRESS_EARLY_SAMPLE_LIMIT = 5



def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for subset comparison evaluation."""
    parser = argparse.ArgumentParser(
        description="Compare vanilla and DoLa-style MC scoring over a small subset."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "hf_tiny_compare_subset.yaml",
        help="Path to a YAML config file.",
    )
    return parser.parse_args()



def _serialize_layer_dist(layer_dist: dict[int, int] | None) -> dict[str, int] | None:
    """Convert an integer-keyed layer-usage mapping into JSON-friendly keys."""
    if not layer_dist:
        return None
    return {str(layer): count for layer, count in sorted(layer_dist.items())}



def _serialize_candidate_scores(items: list[object]) -> list[dict[str, object]]:
    """Convert candidate score objects into JSON-friendly dictionaries."""
    return [
        {
            "candidate": item.candidate,
            "score": item.score,
            "continuation_token_count": item.continuation_token_count,
            "premature_layer_dist": _serialize_layer_dist(item.premature_layer_dist),
        }
        for item in items
    ]



def _aggregate_layer_usage(items: list[object]) -> dict[int, int] | None:
    """Aggregate per-candidate dynamic layer usage counts into one summary."""
    aggregate: dict[int, int] = {}
    for item in items:
        layer_dist = getattr(item, "premature_layer_dist", None)
        if not layer_dist:
            continue
        for layer, count in layer_dist.items():
            aggregate[int(layer)] = aggregate.get(int(layer), 0) + int(count)
    return aggregate or None


class ProgressReporter:
    """Lightweight stdout progress reporter with elapsed time and ETA."""

    def __init__(self, *, label: str) -> None:
        self.label = label
        self.start_time = time.perf_counter()
        self.last_log_time = self.start_time
        self.last_logged_sample = 0

    def start(self, *, total_samples: int) -> None:
        self._log(completed_samples=0, total_samples=total_samples)

    def __call__(self, event: dict[str, object]) -> None:
        completed_samples = int(event["completed_samples"])
        total_samples = int(event["total_samples"])
        now = time.perf_counter()
        should_log = (
            completed_samples <= PROGRESS_EARLY_SAMPLE_LIMIT
            or completed_samples == total_samples
            or completed_samples - self.last_logged_sample >= PROGRESS_SAMPLE_STRIDE
            or now - self.last_log_time >= PROGRESS_LOG_SECONDS
        )
        if not should_log:
            return
        self.last_logged_sample = completed_samples
        self.last_log_time = now
        self._log(completed_samples=completed_samples, total_samples=total_samples)

    def _log(self, *, completed_samples: int, total_samples: int) -> None:
        elapsed = time.perf_counter() - self.start_time
        avg_sample_seconds = elapsed / completed_samples if completed_samples > 0 else 0.0
        remaining_samples = max(total_samples - completed_samples, 0)
        eta_seconds = avg_sample_seconds * remaining_samples if completed_samples > 0 else 0.0
        eta_at = datetime.now() + timedelta(seconds=eta_seconds)
        percent = 100.0 * completed_samples / total_samples if total_samples > 0 else 0.0
        print(
            f"[hf_eval_compare_subset] {_progress_bar(completed_samples, total_samples)} "
            f"{percent:5.1f}% | {self.label} | {completed_samples}/{total_samples} samples "
            f"| elapsed={_format_seconds(elapsed)} | avg_sample_s={avg_sample_seconds:.2f} "
            f"| eta={_format_seconds(eta_seconds)} | eta_at={eta_at.strftime('%Y-%m-%d %H:%M:%S')}",
            flush=True,
        )


def _progress_bar(completed: int, total: int, width: int = 24) -> str:
    if total <= 0:
        return "[" + "-" * width + "]"
    filled = int(width * min(max(completed / total, 0.0), 1.0))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def _format_seconds(seconds: float) -> str:
    seconds = max(float(seconds), 0.0)
    if seconds >= 3600:
        return f"{seconds / 3600:.2f}h"
    if seconds >= 60:
        return f"{seconds / 60:.1f}m"
    return f"{seconds:.1f}s"



def evaluate_compare_subset(
    model: Any,
    tokenizer: Any,
    samples: list[TruthfulQASample],
    *,
    max_samples: int,
    premature_layer: int,
    prompt_style: str,
    score_mode: str,
    dola_score_mode: str = "legacy_contrastive",
    post_softmax: bool = False,
    relative_top: float = 0.0,
    relative_top_value: float = -1000.0,
    candidate_premature_layers: list[int] | None = None,
    mature_layer: int | None = None,
    enable_token_selective_dola: bool = False,
    token_selective_mode: str = "heuristic_fact_critical_v1",
    token_selective_config: TokenSelectiveDolaConfig | None = None,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
    analysis_logger: TruthfulQAMCAnalysisLogger | None = None,
) -> tuple[list[dict[str, object]], dict[str, float | int | str | dict[str, int] | list[int] | None]]:
    """Evaluate vanilla vs DoLa-style scoring on the first N samples."""
    if max_samples <= 0:
        raise ValueError("max_samples must be a positive integer.")

    subset = samples[:max_samples]
    if not subset:
        raise ValueError("No samples were loaded for comparison evaluation.")

    num_hidden_layers = int(getattr(model.config, "num_hidden_layers", 0))
    resolved_mature_layer = get_mature_layer_index(num_hidden_layers)
    if mature_layer is not None:
        validate_mature_layer(mature_layer, num_hidden_layers)
        resolved_mature_layer = mature_layer

    if dola_score_mode == "official_dynamic_dola":
        resolved_candidate_layers = validate_candidate_premature_layers(
            candidate_premature_layers,
            resolved_mature_layer,
            num_hidden_layers,
        )
        pair_description = (
            f"candidate_premature_layers={resolved_candidate_layers}, "
            f"mature_layer={resolved_mature_layer}"
        )
    else:
        resolved_candidate_layers = []
        validate_candidate_premature_layers([premature_layer], resolved_mature_layer, num_hidden_layers)
        pair_description = describe_dola_pair(premature_layer, resolved_mature_layer)

    vanilla_metric_rows: list[dict[str, float]] = []
    dola_metric_rows: list[dict[str, float]] = []
    sample_results: list[dict[str, object]] = []
    overall_layer_usage: dict[int, int] = {}

    for index, sample in enumerate(subset):
        prompt = build_mc_prompt(sample, prompt_style=prompt_style)
        true_candidates, false_candidates = get_mc_candidate_sets(sample, prompt_style=prompt_style)
        log_analysis = analysis_logger is not None and analysis_logger.should_log(index)

        vanilla_true = score_candidate_answers_with_details(
            model,
            tokenizer,
            prompt,
            true_candidates,
            score_mode=score_mode,
            return_trace=log_analysis,
        )
        vanilla_false = score_candidate_answers_with_details(
            model,
            tokenizer,
            prompt,
            false_candidates,
            score_mode=score_mode,
            return_trace=log_analysis,
        )
        vanilla_metrics = compute_mc_metrics(
            [item.score for item in vanilla_true],
            [item.score for item in vanilla_false],
        )
        vanilla_metric_rows.append(vanilla_metrics)

        dola_true = score_candidate_answers_dola_with_details(
            model,
            tokenizer,
            prompt,
            true_candidates,
            premature_layer=premature_layer,
            score_mode=score_mode,
            dola_score_mode=dola_score_mode,
            post_softmax=post_softmax,
            relative_top=relative_top,
            relative_top_value=relative_top_value,
            candidate_premature_layers=resolved_candidate_layers,
            mature_layer=resolved_mature_layer,
            return_trace=log_analysis,
            enable_token_selective_dola=enable_token_selective_dola,
            token_selective_mode=token_selective_mode,
            token_selective_config=token_selective_config,
        )
        dola_false = score_candidate_answers_dola_with_details(
            model,
            tokenizer,
            prompt,
            false_candidates,
            premature_layer=premature_layer,
            score_mode=score_mode,
            dola_score_mode=dola_score_mode,
            post_softmax=post_softmax,
            relative_top=relative_top,
            relative_top_value=relative_top_value,
            candidate_premature_layers=resolved_candidate_layers,
            mature_layer=resolved_mature_layer,
            return_trace=log_analysis,
            enable_token_selective_dola=enable_token_selective_dola,
            token_selective_mode=token_selective_mode,
            token_selective_config=token_selective_config,
        )
        dola_metrics = compute_mc_metrics(
            [item.score for item in dola_true],
            [item.score for item in dola_false],
        )
        dola_metric_rows.append(dola_metrics)

        sample_layer_usage = _aggregate_layer_usage(dola_true + dola_false)
        if sample_layer_usage:
            for layer, count in sample_layer_usage.items():
                overall_layer_usage[layer] = overall_layer_usage.get(layer, 0) + count

        sample_results.append(
            {
                "sample_index": index,
                "question": sample.question,
                "prompt": prompt,
                "prompt_style": prompt_style,
                "score_mode": score_mode,
                "dola_score_mode": dola_score_mode,
                "post_softmax": post_softmax,
                "relative_top": relative_top,
                "relative_top_value": relative_top_value,
                "premature_layer": premature_layer,
                "candidate_premature_layers": resolved_candidate_layers or None,
                "mature_layer": resolved_mature_layer,
                "enable_token_selective_dola": enable_token_selective_dola,
                "token_selective_mode": token_selective_mode if enable_token_selective_dola else None,
                "token_selective_config": (
                    token_selective_config.to_config_dict()
                    if enable_token_selective_dola and token_selective_config is not None
                    else None
                ),
                "vanilla": {
                    "true_scores": _serialize_candidate_scores(vanilla_true),
                    "false_scores": _serialize_candidate_scores(vanilla_false),
                    "metrics": vanilla_metrics,
                },
                "dola": {
                    "true_scores": _serialize_candidate_scores(dola_true),
                    "false_scores": _serialize_candidate_scores(dola_false),
                    "metrics": dola_metrics,
                    "premature_layer_dist": _serialize_layer_dist(sample_layer_usage),
                },
            }
        )
        if log_analysis and analysis_logger is not None:
            analysis_logger.log_sample(
                sample_idx=index,
                sample=sample,
                true_candidates=true_candidates,
                false_candidates=false_candidates,
                vanilla_true=vanilla_true,
                vanilla_false=vanilla_false,
                dola_true=dola_true,
                dola_false=dola_false,
                mature_layer=resolved_mature_layer,
                num_hidden_layers=num_hidden_layers,
                premature_layer=premature_layer,
                candidate_premature_layers=resolved_candidate_layers or None,
                dola_score_mode=dola_score_mode,
                score_mode=score_mode,
            )

        if progress_callback is not None:
            progress_callback(
                {
                    "completed_samples": index + 1,
                    "total_samples": len(subset),
                    "sample_index": index,
                    "question": sample.question,
                }
            )

    vanilla_summary = aggregate_mc_metrics(vanilla_metric_rows)
    dola_summary = aggregate_mc_metrics(dola_metric_rows)
    comparison_summary = compare_aggregate_metrics(vanilla_summary, dola_summary)
    comparison_summary.update(
        {
            "prompt_style": prompt_style,
            "score_mode": score_mode,
            "dola_score_mode": dola_score_mode,
            "post_softmax": post_softmax,
            "relative_top": relative_top,
            "relative_top_value": relative_top_value,
            "premature_layer": premature_layer,
            "candidate_premature_layers": resolved_candidate_layers or None,
            "mature_layer": resolved_mature_layer,
            "dola_pair": pair_description,
            "premature_layer_dist": _serialize_layer_dist(overall_layer_usage or None),
            "enable_token_selective_dola": enable_token_selective_dola,
            "token_selective_mode": token_selective_mode if enable_token_selective_dola else None,
            "token_selective_config": (
                token_selective_config.to_config_dict()
                if enable_token_selective_dola and token_selective_config is not None
                else None
            ),
        }
    )
    return sample_results, comparison_summary



def main() -> None:
    """Run vanilla and DoLa-style scoring on the first N samples and save results."""
    args = parse_args()
    config = load_yaml_config(args.config)
    output_dir = ensure_output_dir(
        config.get("output_dir", PROJECT_ROOT / "outputs" / "hf_tiny_compare_subset")
    )

    model_name = str(config["model_name"])
    device = config.get("device", "cpu")
    csv_path = Path(str(config["csv_path"]))
    max_samples = int(config.get("max_samples", 1))
    premature_layer = int(config.get("premature_layer", 0))
    prompt_style = str(config.get("prompt_style", "plain_mc"))
    score_mode = str(config.get("score_mode", "sum_logprob"))
    dola_score_mode = str(config.get("dola_score_mode", "legacy_contrastive"))
    post_softmax = bool(config.get("post_softmax", False))
    relative_top = float(config.get("relative_top", 0.0))
    relative_top_value = float(config.get("relative_top_value", -1000.0))
    candidate_premature_layers = [int(layer) for layer in config.get("candidate_premature_layers", [])]
    mature_layer = config.get("mature_layer")
    mature_layer = None if mature_layer is None else int(mature_layer)
    enable_token_selective_dola = bool(config.get("enable_token_selective_dola", False))
    token_selective_mode = str(config.get("token_selective_mode", "heuristic_fact_critical_v1"))
    token_selective_config = TokenSelectiveDolaConfig.from_mapping(config)

    model_kwargs = {key: config[key] for key in LOAD_CONFIG_KEYS if key in config}
    analysis_logger = maybe_create_truthfulqa_mc_analysis_logger(config, output_dir=output_dir)

    samples = load_truthfulqa_samples(csv_path)

    print(f"[hf_eval_compare_subset] Model: {model_name}")
    print(f"[hf_eval_compare_subset] Device: {device}")
    print(f"[hf_eval_compare_subset] Score mode: {score_mode}")
    print(f"[hf_eval_compare_subset] DoLa score mode: {dola_score_mode}")
    print(f"[hf_eval_compare_subset] Token-selective DoLa: {enable_token_selective_dola}")
    if enable_token_selective_dola:
        print(f"[hf_eval_compare_subset] Token-selective mode: {token_selective_mode}")
        print(
            "[hf_eval_compare_subset] Token-selective config: "
            f"{json.dumps(token_selective_config.to_config_dict(), sort_keys=True)}"
        )
    print(f"[hf_eval_compare_subset] Loaded samples: {len(samples)}")
    print(f"[hf_eval_compare_subset] Evaluating first {min(len(samples), max_samples)} samples")
    print(f"[hf_eval_compare_subset] Output directory: {output_dir}")
    if analysis_logger is not None:
        print(f"[hf_eval_compare_subset] Analysis log directory: {analysis_logger.log_dir}")

    model, tokenizer = load_model_and_tokenizer(
        model_name=model_name,
        device=device,
        **model_kwargs,
    )

    progress_reporter = ProgressReporter(
        label=(
            "token-selective compare"
            if enable_token_selective_dola
            else "compare"
        )
    )
    progress_reporter.start(total_samples=min(len(samples), max_samples))

    sample_results, comparison_summary = evaluate_compare_subset(
        model,
        tokenizer,
        samples,
        max_samples=max_samples,
        premature_layer=premature_layer,
        prompt_style=prompt_style,
        score_mode=score_mode,
        dola_score_mode=dola_score_mode,
        post_softmax=post_softmax,
        relative_top=relative_top,
        relative_top_value=relative_top_value,
        candidate_premature_layers=candidate_premature_layers,
        mature_layer=mature_layer,
        enable_token_selective_dola=enable_token_selective_dola,
        token_selective_mode=token_selective_mode,
        token_selective_config=token_selective_config,
        progress_callback=progress_reporter,
        analysis_logger=analysis_logger,
    )

    for result in sample_results:
        vanilla_metrics = result["vanilla"]["metrics"]
        dola_metrics = result["dola"]["metrics"]
        print(
            f"[hf_eval_compare_subset] Sample {result['sample_index']}: "
            f"vanilla=({format_metrics(vanilla_metrics)}), "
            f"dola=({format_metrics(dola_metrics)})"
        )
        if result["dola"].get("premature_layer_dist"):
            print(
                f"[hf_eval_compare_subset] Sample {result['sample_index']} "
                f"premature_layer_dist={result['dola']['premature_layer_dist']}"
            )

    comparison_summary.update(
        {
            "task_name": str(config.get("task_name", "hf_tiny_compare_subset")),
            "model_name": model_name,
            "csv_path": str(csv_path),
            "output_dir": str(output_dir),
        }
    )

    sample_results_path = output_dir / "compare_sample_results.jsonl"
    with sample_results_path.open("w", encoding="utf-8") as handle:
        for result in sample_results:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")

    summary_path = output_dir / "compare_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(comparison_summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


    print(f"[hf_eval_compare_subset] Summary: {format_metrics(comparison_summary)}")
    if comparison_summary.get("premature_layer_dist"):
        print(
            "[hf_eval_compare_subset] Overall premature_layer_dist: "
            f"{comparison_summary['premature_layer_dist']}"
        )
    print(f"[hf_eval_compare_subset] Saved sample results to: {sample_results_path}")
    print(f"[hf_eval_compare_subset] Saved summary to: {summary_path}")
    if analysis_logger is not None:
        print(
            "[hf_eval_compare_subset] Saved analysis logs to: "
            f"{analysis_logger.sample_level_path} and {analysis_logger.candidate_level_path}"
        )


if __name__ == "__main__":
    main()
