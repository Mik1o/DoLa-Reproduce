"""Compare vanilla and DoLa-style scoring on one TruthfulQA multiple-choice sample."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis_logging import maybe_create_truthfulqa_mc_analysis_logger
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
    "local_files_only",
    "device_map",
    "use_4bit",
    "bnb_4bit_compute_dtype",
    "bnb_4bit_quant_type",
    "bnb_4bit_use_double_quant",
    "tokenizer_class",
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
    if sample_index < 0 or sample_index >= len(samples):
        raise IndexError(
            f"sample_index {sample_index} is out of range for {len(samples)} loaded samples."
        )

    sample = samples[sample_index]
    prompt = build_mc_prompt(sample, prompt_style=prompt_style)
    true_candidates, false_candidates = get_mc_candidate_sets(sample, prompt_style=prompt_style)

    print(f"[hf_compare_single_mc] Model: {model_name}")
    print(f"[hf_compare_single_mc] Device: {device}")
    print(f"[hf_compare_single_mc] Score mode: {score_mode}")
    print(f"[hf_compare_single_mc] DoLa score mode: {dola_score_mode}")
    print(f"[hf_compare_single_mc] Token-selective DoLa: {enable_token_selective_dola}")
    if enable_token_selective_dola:
        print(f"[hf_compare_single_mc] Token-selective mode: {token_selective_mode}")
        print(
            "[hf_compare_single_mc] Token-selective config: "
            f"{json.dumps(token_selective_config.to_config_dict(), sort_keys=True)}"
        )
    print(f"[hf_compare_single_mc] Output directory: {output_dir}")
    if analysis_logger is not None:
        print(f"[hf_compare_single_mc] Analysis log directory: {analysis_logger.log_dir}")

    model, tokenizer = load_model_and_tokenizer(
        model_name=model_name,
        device=device,
        **model_kwargs,
    )

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

    log_analysis = analysis_logger is not None and analysis_logger.should_log(sample_index)

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
    aggregate_premature_layer_dist = _aggregate_layer_usage(dola_true + dola_false)

    print("[hf_compare_single_mc] Question:")
    print(sample.question)
    print("[hf_compare_single_mc] Prompt:")
    print(prompt)
    print(f"[hf_compare_single_mc] DoLa pair: {pair_description}")
    print(f"[hf_compare_single_mc] Vanilla metrics: {format_metrics(vanilla_metrics)}")
    print(f"[hf_compare_single_mc] DoLa metrics: {format_metrics(dola_metrics)}")
    if aggregate_premature_layer_dist:
        print(
            "[hf_compare_single_mc] premature_layer_dist: "
            f"{_serialize_layer_dist(aggregate_premature_layer_dist)}"
        )

    result = {
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
            if enable_token_selective_dola
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
            "premature_layer_dist": _serialize_layer_dist(aggregate_premature_layer_dist),
        },
    }

    output_path = output_dir / "compare_single_result.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    if log_analysis and analysis_logger is not None:
        analysis_logger.log_sample(
            sample_idx=sample_index,
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

    print(f"[hf_compare_single_mc] Saved comparison to: {output_path}")
    if analysis_logger is not None:
        print(
            "[hf_compare_single_mc] Saved analysis logs to: "
            f"{analysis_logger.sample_level_path} and {analysis_logger.candidate_level_path}"
        )


if __name__ == "__main__":
    main()
