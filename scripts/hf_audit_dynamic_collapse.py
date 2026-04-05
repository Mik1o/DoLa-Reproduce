"""Audit whether official dynamic DoLa collapses toward one static layer on a tiny subset."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.generation import (
    _compute_dynamic_js_divergence,
    _compute_official_dola_token_scores,
    _count_premature_layer_usage,
    _get_continuation_start_index,
    _prepare_scoring_inputs,
    _select_dynamic_base_logits,
)
from src.modeling import load_model_and_tokenizer
from src.truthfulqa_mc import build_answer_continuation, build_mc_prompt, load_truthfulqa_samples
from src.utils import ensure_output_dir, load_yaml_config


LOAD_CONFIG_KEYS = (
    "use_safetensors",
    "torch_dtype",
    "use_fast_tokenizer",
    "trust_remote_code",
    "attn_implementation",
)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit official dynamic DoLa collapse on a tiny fixed TruthfulQA subset."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "mistral7b_truthfulqa_real_dynamic_collapse_audit.yaml",
        help="Path to a YAML config file.",
    )
    return parser.parse_args()



def _serialize_layer_dist(layer_dist: dict[int, int]) -> dict[str, int]:
    return {str(layer): int(count) for layer, count in sorted(layer_dist.items())}



def _candidate_entries(sample: Any, prompt_style: str) -> list[tuple[str, str]]:
    return [
        ("best_answer", build_answer_continuation(sample.best_answer, prompt_style=prompt_style)),
        ("false_answer", build_answer_continuation(sample.incorrect_answers[0], prompt_style=prompt_style)),
    ]



def _top2_from_values(values: list[float], layers: list[int]) -> dict[str, float | int | None]:
    ranked = sorted(zip(layers, values, strict=False), key=lambda item: item[1], reverse=True)
    top1_layer, top1_value = ranked[0]
    if len(ranked) > 1:
        top2_layer, top2_value = ranked[1]
    else:
        top2_layer, top2_value = None, float("nan")
    margin = top1_value - top2_value if top2_layer is not None else float("nan")
    return {
        "top1_layer": int(top1_layer),
        "top1_js": float(top1_value),
        "top2_layer": None if top2_layer is None else int(top2_layer),
        "top2_js": float(top2_value),
        "margin": float(margin),
    }



def _build_dynamic_trace(js_divergence: Any, candidate_layers: list[int], continuation_start: int) -> tuple[list[dict[str, object]], list[int], dict[int, int], float]:
    per_token_js = js_divergence[:, 0, continuation_start:].detach().cpu().tolist()
    selected_layers: list[int] = []
    token_reports: list[dict[str, object]] = []
    margins: list[float] = []

    for token_index, token_values in enumerate(zip(*per_token_js, strict=False)):
        js_values = [float(value) for value in token_values]
        top2 = _top2_from_values(js_values, candidate_layers)
        selected_layers.append(int(top2["top1_layer"]))
        if not math.isnan(float(top2["margin"])):
            margins.append(float(top2["margin"]))
        token_reports.append(
            {
                "continuation_token_index": token_index,
                "js_divergence_by_layer": {
                    str(layer): float(value)
                    for layer, value in zip(candidate_layers, js_values, strict=False)
                },
                **top2,
            }
        )

    layer_dist = _count_premature_layer_usage(selected_layers, candidate_layers)
    avg_margin = sum(margins) / len(margins) if margins else 0.0
    return token_reports, selected_layers, layer_dist, avg_margin



def _compute_bucket_report(
    model: Any,
    tokenizer: Any,
    prompt: str,
    candidate_answer: str,
    candidate_layers: list[int],
    mature_layer: int,
    post_softmax: bool,
    relative_top: float,
    relative_top_value: float,
) -> dict[str, object]:
    try:
        import torch
    except ImportError as error:
        raise ImportError("torch is required for dynamic collapse auditing.") from error

    _, input_ids, attention_mask, prompt_len = _prepare_scoring_inputs(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        candidate_answer=candidate_answer,
        separator=" ",
    )
    continuation_start = _get_continuation_start_index(prompt_len)

    lm_head = model.get_output_embeddings()
    if lm_head is None:
        raise ValueError("The model does not expose output embeddings for collapse auditing.")

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise ValueError("The model did not return hidden_states for collapse auditing.")

        mature_logits = lm_head(hidden_states[mature_layer + 1][:, :-1, :])
        candidate_logits = torch.stack(
            [lm_head(hidden_states[layer + 1][:, :-1, :]) for layer in candidate_layers],
            dim=0,
        )
        js_divergence = _compute_dynamic_js_divergence(
            mature_logits=mature_logits,
            candidate_logits=candidate_logits,
        )
        base_logits, selected_layers_full = _select_dynamic_base_logits(
            mature_logits=mature_logits,
            candidate_logits=candidate_logits,
            candidate_layers=candidate_layers,
        )
        dynamic_token_scores = _compute_official_dola_token_scores(
            mature_logits=mature_logits,
            base_logits=base_logits,
            input_ids=input_ids,
            post_softmax=post_softmax,
            relative_top=relative_top,
            relative_top_value=relative_top_value,
        )

        static_scores: dict[int, float] = {}
        for layer, layer_logits in zip(candidate_layers, candidate_logits, strict=False):
            static_token_scores = _compute_official_dola_token_scores(
                mature_logits=mature_logits,
                base_logits=layer_logits,
                input_ids=input_ids,
                post_softmax=post_softmax,
                relative_top=relative_top,
                relative_top_value=relative_top_value,
            )
            static_scores[int(layer)] = float(static_token_scores[:, continuation_start:].sum().item())

    token_reports, selected_layers, layer_dist, avg_margin = _build_dynamic_trace(
        js_divergence=js_divergence,
        candidate_layers=candidate_layers,
        continuation_start=continuation_start,
    )
    dynamic_score = float(dynamic_token_scores[:, continuation_start:].sum().item())
    dominant_layer, dominant_count = max(layer_dist.items(), key=lambda item: item[1])
    dominant_ratio = dominant_count / len(selected_layers) if selected_layers else 0.0
    best_static_layer, best_static_score = max(static_scores.items(), key=lambda item: item[1])

    return {
        "candidate_layers": candidate_layers,
        "token_reports": token_reports,
        "selected_layers": selected_layers,
        "premature_layer_dist": _serialize_layer_dist(layer_dist),
        "dominant_layer": int(dominant_layer),
        "dominant_layer_ratio": float(dominant_ratio),
        "average_top1_top2_margin": float(avg_margin),
        "dynamic_score": dynamic_score,
        "static_scores": {str(layer): float(score) for layer, score in static_scores.items()},
        "best_static_layer": int(best_static_layer),
        "best_static_score": float(best_static_score),
        "dynamic_minus_best_static": float(dynamic_score - best_static_score),
    }



def _compare_buckets(current_report: dict[str, object], shifted_report: dict[str, object]) -> dict[str, object]:
    current_selected = current_report["selected_layers"]
    shifted_selected = shifted_report["selected_layers"]
    same_pattern = current_selected == shifted_selected
    overlap = sum(a == b for a, b in zip(current_selected, shifted_selected, strict=False))
    overlap_ratio = overlap / len(current_selected) if current_selected else 0.0
    return {
        "same_selected_pattern": same_pattern,
        "selected_pattern_overlap_ratio": float(overlap_ratio),
        "dynamic_score_delta": float(current_report["dynamic_score"] - shifted_report["dynamic_score"]),
        "best_static_layer_pair": [current_report["best_static_layer"], shifted_report["best_static_layer"]],
    }



def _classify_audit(case_reports: list[dict[str, object]]) -> str:
    if not case_reports:
        raise ValueError("case_reports must contain at least one audited case.")

    avg_dominant_ratio = sum(float(case["current_bucket"]["dominant_layer_ratio"]) for case in case_reports) / len(case_reports)
    avg_margin = sum(float(case["current_bucket"]["average_top1_top2_margin"]) for case in case_reports) / len(case_reports)
    avg_dynamic_best_static_delta = sum(float(case["current_bucket"]["dynamic_minus_best_static"]) for case in case_reports) / len(case_reports)
    avg_shifted_delta = sum(abs(float(case["bucket_comparison"]["dynamic_score_delta"])) for case in case_reports) / len(case_reports)
    avg_overlap = sum(float(case["bucket_comparison"]["selected_pattern_overlap_ratio"]) for case in case_reports) / len(case_reports)

    if avg_shifted_delta > 0.5 and avg_overlap < 0.5:
        return "LIKELY_BUCKET_CHOICE_TOO_NARROW"
    if avg_dominant_ratio >= 0.8 and abs(avg_dynamic_best_static_delta) <= 1e-3:
        return "LIKELY_TRUE_DYNAMIC_COLLAPSE_ON_MISTRAL"
    if avg_overlap >= 0.8 and avg_margin <= 1e-3:
        return "LIKELY_NEIGHBOR_LAYERS_ARE_INTRINSICALLY_SIMILAR"
    return "LIKELY_STILL_HAS_IMPLEMENTATION_ISSUE"



def _print_case_summary(case: dict[str, object]) -> None:
    current_report = case["current_bucket"]
    shifted_report = case["shifted_bucket"]
    comparison = case["bucket_comparison"]
    print(
        f"[hf_audit_dynamic_collapse] sample={case['sample_index']} answer={case['answer_name']}: "
        f"current_dominant={current_report['dominant_layer']} ({current_report['dominant_layer_ratio']:.2f}), "
        f"current_avg_margin={current_report['average_top1_top2_margin']:.6f}, "
        f"current_dyn-best_static={current_report['dynamic_minus_best_static']:.6f}, "
        f"shifted_dominant={shifted_report['dominant_layer']} ({shifted_report['dominant_layer_ratio']:.2f}), "
        f"shifted_avg_margin={shifted_report['average_top1_top2_margin']:.6f}, "
        f"current-vs-shifted-delta={comparison['dynamic_score_delta']:.6f}"
    )



def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    output_dir = ensure_output_dir(config.get("output_dir", PROJECT_ROOT / "outputs" / "dynamic_collapse_audit"))

    model_name = str(config["model_name"])
    device = config.get("device", "cpu")
    csv_path = Path(str(config["csv_path"]))
    max_samples = int(config.get("max_samples", 5))
    prompt_style = str(config.get("prompt_style", "official_tfqa_mc"))
    mature_layer = int(config["mature_layer"])
    current_bucket = [int(layer) for layer in config["candidate_premature_layers"]]
    shifted_bucket = [int(layer) for layer in config["candidate_premature_layers_shifted"]]
    post_softmax = bool(config.get("post_softmax", False))
    relative_top = float(config.get("relative_top", 0.0))
    relative_top_value = float(config.get("relative_top_value", -1000.0))

    model_kwargs = {key: config[key] for key in LOAD_CONFIG_KEYS if key in config}

    print(f"[hf_audit_dynamic_collapse] Model: {model_name}")
    print(f"[hf_audit_dynamic_collapse] CSV: {csv_path}")
    print(f"[hf_audit_dynamic_collapse] Output directory: {output_dir}")

    samples = load_truthfulqa_samples(csv_path)[:max_samples]
    if not samples:
        raise ValueError("No TruthfulQA samples were loaded for dynamic collapse auditing.")

    model, tokenizer = load_model_and_tokenizer(
        model_name=model_name,
        device=device,
        **model_kwargs,
    )

    case_reports: list[dict[str, object]] = []
    for sample_index, sample in enumerate(samples):
        prompt = build_mc_prompt(sample, prompt_style=prompt_style)
        for answer_name, candidate_answer in _candidate_entries(sample, prompt_style):
            current_report = _compute_bucket_report(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                candidate_answer=candidate_answer,
                candidate_layers=current_bucket,
                mature_layer=mature_layer,
                post_softmax=post_softmax,
                relative_top=relative_top,
                relative_top_value=relative_top_value,
            )
            shifted_report = _compute_bucket_report(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                candidate_answer=candidate_answer,
                candidate_layers=shifted_bucket,
                mature_layer=mature_layer,
                post_softmax=post_softmax,
                relative_top=relative_top,
                relative_top_value=relative_top_value,
            )
            case_report = {
                "sample_index": sample_index,
                "question": sample.question,
                "answer_name": answer_name,
                "candidate_answer": candidate_answer,
                "current_bucket": current_report,
                "shifted_bucket": shifted_report,
                "bucket_comparison": _compare_buckets(current_report, shifted_report),
            }
            case_reports.append(case_report)
            _print_case_summary(case_report)

    final_judgement = _classify_audit(case_reports)
    report = {
        "task_name": str(config.get("task_name", "hf_audit_dynamic_collapse")),
        "model_name": model_name,
        "csv_path": str(csv_path),
        "max_samples": max_samples,
        "prompt_style": prompt_style,
        "mature_layer": mature_layer,
        "current_bucket": current_bucket,
        "shifted_bucket": shifted_bucket,
        "cases": case_reports,
        "final_judgement": final_judgement,
    }

    output_path = output_dir / "dynamic_collapse_audit.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(f"[hf_audit_dynamic_collapse] final_judgement={final_judgement}")
    print(f"[hf_audit_dynamic_collapse] Saved audit to: {output_path}")


if __name__ == "__main__":
    main()
