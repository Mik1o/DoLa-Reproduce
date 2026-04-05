"""Audit whether dynamic DoLa collapse on Mistral is sensitive to recomputation dtype."""

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
        description="Audit precision sensitivity of official dynamic DoLa on a tiny TruthfulQA subset."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "mistral7b_truthfulqa_real_precision_sensitivity_audit.yaml",
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



def _dtype_name(dtype: Any) -> str:
    return str(dtype).replace("torch.", "")



def _resolve_recompute_dtypes(torch: Any, runtime_dtype: Any, configured_names: list[str]) -> list[tuple[str, Any]]:
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "float64": torch.float64,
        "fp64": torch.float64,
    }
    resolved: list[tuple[str, Any]] = []
    seen: set[str] = set()
    for raw_name in configured_names:
        name = str(raw_name).strip().lower()
        if name == "runtime":
            key = f"runtime:{_dtype_name(runtime_dtype)}"
            if key not in seen:
                seen.add(key)
                resolved.append(("runtime", runtime_dtype))
            continue
        if name not in mapping:
            raise ValueError(
                f"Unsupported recompute dtype '{raw_name}'. Use runtime/float32/float64/bfloat16/float16."
            )
        dtype = mapping[name]
        key = _dtype_name(dtype)
        if key not in seen:
            seen.add(key)
            resolved.append((key, dtype))
    return resolved



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



def _build_dynamic_trace(
    js_divergence: Any,
    candidate_layers: list[int],
    continuation_start: int,
) -> tuple[list[dict[str, object]], list[int], dict[int, int], float]:
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



def _recompute_bucket_for_dtype(
    *,
    mature_logits: Any,
    candidate_logits_map: dict[int, Any],
    candidate_layers: list[int],
    input_ids: Any,
    continuation_start: int,
    recompute_dtype: Any,
    post_softmax: bool,
    relative_top: float,
    relative_top_value: float,
) -> dict[str, object]:
    try:
        import torch
    except ImportError as error:
        raise ImportError("torch is required for precision sensitivity auditing.") from error

    candidate_stack = torch.stack(
        [candidate_logits_map[layer].to(dtype=recompute_dtype) for layer in candidate_layers],
        dim=0,
    )
    mature_logits_cast = mature_logits.to(dtype=recompute_dtype)
    js_divergence = _compute_dynamic_js_divergence(
        mature_logits=mature_logits_cast,
        candidate_logits=candidate_stack,
    )
    base_logits, _ = _select_dynamic_base_logits(
        mature_logits=mature_logits_cast,
        candidate_logits=candidate_stack,
        candidate_layers=candidate_layers,
    )
    dynamic_token_scores = _compute_official_dola_token_scores(
        mature_logits=mature_logits_cast,
        base_logits=base_logits,
        input_ids=input_ids,
        post_softmax=post_softmax,
        relative_top=relative_top,
        relative_top_value=relative_top_value,
    )

    static_scores: dict[int, float] = {}
    for layer, layer_logits in zip(candidate_layers, candidate_stack, strict=False):
        static_token_scores = _compute_official_dola_token_scores(
            mature_logits=mature_logits_cast,
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
    best_static_layer, best_static_score = max(static_scores.items(), key=lambda item: item[1])
    dominant_layer, dominant_count = max(layer_dist.items(), key=lambda item: item[1])
    dominant_ratio = dominant_count / len(selected_layers) if selected_layers else 0.0

    return {
        "dtype": _dtype_name(recompute_dtype),
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



def _selected_layer_change_rate(reference: list[int], target: list[int]) -> float:
    if len(reference) != len(target):
        raise ValueError("Selected-layer sequences must have the same length.")
    if not reference:
        return 0.0
    changed = sum(left != right for left, right in zip(reference, target, strict=False))
    return changed / len(reference)



def _compare_dtype_reports(reference: dict[str, object], target: dict[str, object]) -> dict[str, object]:
    return {
        "reference_dtype": str(reference["dtype"]),
        "target_dtype": str(target["dtype"]),
        "selected_layer_change_rate": float(
            _selected_layer_change_rate(reference["selected_layers"], target["selected_layers"])
        ),
        "dominant_layer_changed": int(reference["dominant_layer"]) != int(target["dominant_layer"]),
        "average_margin_delta": float(target["average_top1_top2_margin"] - reference["average_top1_top2_margin"]),
        "dynamic_score_delta": float(target["dynamic_score"] - reference["dynamic_score"]),
        "best_static_layer_changed": int(reference["best_static_layer"]) != int(target["best_static_layer"]),
        "dynamic_minus_best_static_delta": float(
            target["dynamic_minus_best_static"] - reference["dynamic_minus_best_static"]
        ),
    }



def _classify_precision_audit(case_reports: list[dict[str, object]]) -> str:
    if not case_reports:
        raise ValueError("case_reports must contain at least one audited case.")

    change_rates: list[float] = []
    margin_deltas: list[float] = []
    score_deltas: list[float] = []
    dominant_changes = 0
    total_comparisons = 0

    for case in case_reports:
        for diff in case["dtype_differences"]:
            total_comparisons += 1
            change_rates.append(float(diff["selected_layer_change_rate"]))
            margin_deltas.append(float(diff["average_margin_delta"]))
            score_deltas.append(abs(float(diff["dynamic_score_delta"])))
            dominant_changes += int(bool(diff["dominant_layer_changed"]))

    avg_change_rate = sum(change_rates) / len(change_rates) if change_rates else 0.0
    avg_margin_delta = sum(margin_deltas) / len(margin_deltas) if margin_deltas else 0.0
    avg_score_delta = sum(score_deltas) / len(score_deltas) if score_deltas else 0.0
    dominant_change_ratio = dominant_changes / total_comparisons if total_comparisons else 0.0

    if avg_change_rate >= 0.2 or dominant_change_ratio >= 0.3:
        if avg_margin_delta > 1e-5 or avg_score_delta > 1e-4:
            return "LIKELY_NUMERICAL_TIE_COLLAPSE"
        return "LIKELY_MIXED_NUMERICAL_AND_STRUCTURAL"

    if avg_change_rate <= 0.05 and dominant_change_ratio <= 0.1 and avg_score_delta <= 1e-4:
        return "LIKELY_TRUE_LAYER_SIMILARITY_COLLAPSE"

    if avg_margin_delta > 1e-5 or avg_score_delta > 1e-4:
        return "LIKELY_MIXED_NUMERICAL_AND_STRUCTURAL"

    return "LIKELY_STILL_UNKNOWN"



def _print_case_summary(case: dict[str, object]) -> None:
    runtime_report = case["dtype_reports"][0]
    fp32_report = next((item for item in case["dtype_reports"] if item["dtype"] == "float32"), None)
    fp64_report = next((item for item in case["dtype_reports"] if item["dtype"] == "float64"), None)
    fp32_diff = next((item for item in case["dtype_differences"] if item["target_dtype"] == "float32"), None)
    fp64_diff = next((item for item in case["dtype_differences"] if item["target_dtype"] == "float64"), None)

    print(
        f"[hf_audit_precision_sensitivity] sample={case['sample_index']} answer={case['answer_name']} bucket={case['bucket_name']}: "
        f"runtime_dtype={runtime_report['dtype']}, "
        f"fp32_dominant={None if fp32_report is None else fp32_report['dominant_layer']}, "
        f"fp64_dominant={None if fp64_report is None else fp64_report['dominant_layer']}, "
        f"change_rate(fp32)={0.0 if fp32_diff is None else fp32_diff['selected_layer_change_rate']:.4f}, "
        f"change_rate(fp64)={0.0 if fp64_diff is None else fp64_diff['selected_layer_change_rate']:.4f}, "
        f"margin(runtime/fp32/fp64)=({runtime_report['average_top1_top2_margin']:.6f}/"
        f"{0.0 if fp32_report is None else fp32_report['average_top1_top2_margin']:.6f}/"
        f"{0.0 if fp64_report is None else fp64_report['average_top1_top2_margin']:.6f}), "
        f"score_delta(fp32)={0.0 if fp32_diff is None else fp32_diff['dynamic_score_delta']:.6f}, "
        f"score_delta(fp64)={0.0 if fp64_diff is None else fp64_diff['dynamic_score_delta']:.6f}"
    )



def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    output_dir = ensure_output_dir(
        config.get("output_dir", PROJECT_ROOT / "outputs" / "precision_sensitivity_audit")
    )

    model_name = str(config["model_name"])
    device = config.get("device", "cpu")
    csv_path = Path(str(config["csv_path"]))
    max_samples = int(config.get("max_samples", 5))
    prompt_style = str(config.get("prompt_style", "official_tfqa_mc"))
    mature_layer = int(config["mature_layer"])
    current_bucket = [int(layer) for layer in config["candidate_premature_layers"]]
    shifted_bucket = [int(layer) for layer in config["candidate_premature_layers_shifted"]]
    recompute_dtype_names = [str(item) for item in config.get("recompute_dtypes", ["runtime", "float32", "float64"])]
    post_softmax = bool(config.get("post_softmax", False))
    relative_top = float(config.get("relative_top", 0.0))
    relative_top_value = float(config.get("relative_top_value", -1000.0))

    model_kwargs = {key: config[key] for key in LOAD_CONFIG_KEYS if key in config}

    print(f"[hf_audit_precision_sensitivity] Model: {model_name}")
    print(f"[hf_audit_precision_sensitivity] CSV: {csv_path}")
    print(f"[hf_audit_precision_sensitivity] Output directory: {output_dir}")

    samples = load_truthfulqa_samples(csv_path)[:max_samples]
    if not samples:
        raise ValueError("No TruthfulQA samples were loaded for precision sensitivity auditing.")

    model, tokenizer = load_model_and_tokenizer(
        model_name=model_name,
        device=device,
        **model_kwargs,
    )

    try:
        import torch
    except ImportError as error:
        raise ImportError("torch is required for precision sensitivity auditing.") from error

    case_reports: list[dict[str, object]] = []
    for sample_index, sample in enumerate(samples):
        prompt = build_mc_prompt(sample, prompt_style=prompt_style)
        for answer_name, candidate_answer in _candidate_entries(sample, prompt_style):
            _, input_ids, attention_mask, prompt_len = _prepare_scoring_inputs(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                candidate_answer=candidate_answer,
                separator=" ",
            )
            continuation_start = _get_continuation_start_index(prompt_len)

            all_layers = sorted(set(current_bucket + shifted_bucket + [mature_layer]))
            lm_head = model.get_output_embeddings()
            if lm_head is None:
                raise ValueError("The model does not expose output embeddings for precision auditing.")

            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
            hidden_states = outputs.hidden_states
            if hidden_states is None:
                raise ValueError("The model did not return hidden_states for precision auditing.")

            layer_logits = {
                layer: lm_head(hidden_states[layer + 1][:, :-1, :])
                for layer in all_layers
            }
            runtime_dtype = layer_logits[mature_layer].dtype
            resolved_dtypes = _resolve_recompute_dtypes(torch, runtime_dtype, recompute_dtype_names)

            for bucket_name, candidate_layers in (
                ("current_bucket", current_bucket),
                ("shifted_bucket", shifted_bucket),
            ):
                dtype_reports: list[dict[str, object]] = []
                for dtype_name, recompute_dtype in resolved_dtypes:
                    report = _recompute_bucket_for_dtype(
                        mature_logits=layer_logits[mature_layer],
                        candidate_logits_map=layer_logits,
                        candidate_layers=candidate_layers,
                        input_ids=input_ids,
                        continuation_start=continuation_start,
                        recompute_dtype=recompute_dtype,
                        post_softmax=post_softmax,
                        relative_top=relative_top,
                        relative_top_value=relative_top_value,
                    )
                    if dtype_name == "runtime":
                        report["dtype"] = f"runtime:{_dtype_name(runtime_dtype)}"
                    dtype_reports.append(report)

                reference_report = dtype_reports[0]
                dtype_differences = [
                    _compare_dtype_reports(reference_report, target_report)
                    for target_report in dtype_reports[1:]
                ]
                case_report = {
                    "sample_index": sample_index,
                    "question": sample.question,
                    "prompt": prompt,
                    "answer_name": answer_name,
                    "candidate_answer": candidate_answer,
                    "bucket_name": bucket_name,
                    "bucket": candidate_layers,
                    "runtime_dtype": _dtype_name(runtime_dtype),
                    "dtype_reports": dtype_reports,
                    "dtype_differences": dtype_differences,
                }
                case_reports.append(case_report)
                _print_case_summary(case_report)

    final_conclusion = _classify_precision_audit(case_reports)
    report = {
        "task_name": str(config.get("task_name", "hf_audit_precision_sensitivity")),
        "model_name": model_name,
        "csv_path": str(csv_path),
        "max_samples": max_samples,
        "prompt_style": prompt_style,
        "mature_layer": mature_layer,
        "current_bucket": current_bucket,
        "shifted_bucket": shifted_bucket,
        "cases": case_reports,
        "final_conclusion": final_conclusion,
    }

    output_path = output_dir / "precision_sensitivity_audit.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(f"[hf_audit_precision_sensitivity] final_conclusion={final_conclusion}")
    print(f"[hf_audit_precision_sensitivity] Saved audit to: {output_path}")


if __name__ == "__main__":
    main()
