"""Trace whether compare/eval, oracle probe, and audit use the same dynamic scoring path."""

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

from scripts.hf_audit_dynamic_collapse import _build_dynamic_trace, _compute_bucket_report
from scripts.hf_probe_early_exit_oracle_parity import _get_final_norm_module, _oracle_hidden_and_logits
from src.generation import (
    _compute_dynamic_js_divergence,
    _compute_official_dola_token_scores,
    _get_continuation_start_index,
    _prepare_scoring_inputs,
    _select_dynamic_base_logits,
    score_continuation_dola_details,
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
    "local_files_only",
    "device_map",
    "use_4bit",
    "bnb_4bit_compute_dtype",
    "bnb_4bit_quant_type",
    "bnb_4bit_use_double_quant",
    "tokenizer_class",
)
SCORE_TOLERANCE = 1e-6
JS_TOLERANCE = 1e-6



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trace scoring-path consistency across compare/eval, oracle probe, and audit routes."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "mistral7b_truthfulqa_real_trace_scoring_path_consistency.yaml",
        help="Path to a YAML config file.",
    )
    return parser.parse_args()



def _serialize_layer_dist(layer_dist: dict[int, int] | None) -> dict[str, int] | None:
    if not layer_dist:
        return None
    return {str(layer): int(count) for layer, count in sorted(layer_dist.items())}



def _build_answer_cases(sample: Any, prompt_style: str) -> list[tuple[str, str]]:
    return [
        ("best_answer", build_answer_continuation(sample.best_answer, prompt_style=prompt_style)),
        ("false_answer", build_answer_continuation(sample.incorrect_answers[0], prompt_style=prompt_style)),
    ]



def _local_to_official_layer(local_layer: int, num_hidden_layers: int) -> int:
    if local_layer == num_hidden_layers - 1:
        return num_hidden_layers
    return local_layer + 1



def _build_js_report_lookup(token_reports: list[dict[str, object]]) -> list[dict[str, object]]:
    normalized: list[dict[str, object]] = []
    for report in token_reports:
        js_by_layer = {
            str(layer): float(value)
            for layer, value in sorted(report["js_divergence_by_layer"].items(), key=lambda item: int(item[0]))
        }
        normalized.append(
            {
                "continuation_token_index": int(report["continuation_token_index"]),
                "js_divergence_by_layer": js_by_layer,
                "top1_layer": int(report["top1_layer"]),
                "top2_layer": None if report["top2_layer"] is None else int(report["top2_layer"]),
                "margin": float(report["margin"]),
            }
        )
    return normalized



def _float_close(left: float, right: float, tolerance: float = SCORE_TOLERANCE) -> bool:
    return abs(float(left) - float(right)) <= tolerance



def _dict_float_close(
    left: dict[str, float],
    right: dict[str, float],
    tolerance: float = SCORE_TOLERANCE,
) -> bool:
    if set(left.keys()) != set(right.keys()):
        return False
    return all(_float_close(float(left[key]), float(right[key]), tolerance) for key in left)



def _token_reports_match(
    left: list[dict[str, object]],
    right: list[dict[str, object]],
    tolerance: float = JS_TOLERANCE,
) -> bool:
    if len(left) != len(right):
        return False
    for left_item, right_item in zip(left, right, strict=False):
        if int(left_item["continuation_token_index"]) != int(right_item["continuation_token_index"]):
            return False
        if int(left_item["top1_layer"]) != int(right_item["top1_layer"]):
            return False
        if left_item["top2_layer"] != right_item["top2_layer"]:
            return False
        if not _float_close(float(left_item["margin"]), float(right_item["margin"]), tolerance):
            return False
        left_js = left_item["js_divergence_by_layer"]
        right_js = right_item["js_divergence_by_layer"]
        if set(left_js.keys()) != set(right_js.keys()):
            return False
        for key in left_js:
            if not _float_close(float(left_js[key]), float(right_js[key]), tolerance):
                return False
    return True



def _run_main_route_trace(
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
        raise ImportError("torch is required for scoring-path tracing.") from error

    _, input_ids, attention_mask, prompt_len = _prepare_scoring_inputs(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        candidate_answer=candidate_answer,
        separator=" ",
    )
    continuation_start = _get_continuation_start_index(prompt_len)

    public_dynamic = score_continuation_dola_details(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        candidate_answer=candidate_answer,
        premature_layer=candidate_layers[0],
        score_mode="sum_logprob",
        dola_score_mode="official_dynamic_dola",
        post_softmax=post_softmax,
        relative_top=relative_top,
        relative_top_value=relative_top_value,
        candidate_premature_layers=candidate_layers,
        mature_layer=mature_layer,
    )

    public_static_scores: dict[str, float] = {}
    for layer in candidate_layers:
        public_static_scores[str(layer)] = float(
            score_continuation_dola_details(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                candidate_answer=candidate_answer,
                premature_layer=layer,
                score_mode="sum_logprob",
                dola_score_mode="official_static_dola",
                post_softmax=post_softmax,
                relative_top=relative_top,
                relative_top_value=relative_top_value,
                mature_layer=mature_layer,
            ).score
        )

    lm_head = model.get_output_embeddings()
    if lm_head is None:
        raise ValueError("The model does not expose output embeddings for consistency tracing.")

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise ValueError("The model did not return hidden_states for consistency tracing.")

        mature_logits = lm_head(hidden_states[mature_layer + 1][:, :-1, :])
        candidate_logits = torch.stack(
            [lm_head(hidden_states[layer + 1][:, :-1, :]) for layer in candidate_layers],
            dim=0,
        )
        js_divergence = _compute_dynamic_js_divergence(
            mature_logits=mature_logits,
            candidate_logits=candidate_logits,
        )
        base_logits, _ = _select_dynamic_base_logits(
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

        static_scores: dict[str, float] = {}
        for layer, layer_logits in zip(candidate_layers, candidate_logits, strict=False):
            static_token_scores = _compute_official_dola_token_scores(
                mature_logits=mature_logits,
                base_logits=layer_logits,
                input_ids=input_ids,
                post_softmax=post_softmax,
                relative_top=relative_top,
                relative_top_value=relative_top_value,
            )
            static_scores[str(layer)] = float(static_token_scores[:, continuation_start:].sum().item())

    token_reports, selected_layers, layer_dist, avg_margin = _build_dynamic_trace(
        js_divergence=js_divergence,
        candidate_layers=candidate_layers,
        continuation_start=continuation_start,
    )
    low_level_dynamic_score = float(dynamic_token_scores[:, continuation_start:].sum().item())
    return {
        "path_name": "main_compare_eval_path",
        "function_chain": [
            "src.generation.score_continuation_dola_details",
            "src.generation._prepare_scoring_inputs",
            "src.generation._compute_dynamic_js_divergence",
            "src.generation._select_dynamic_base_logits",
            "src.generation._compute_official_dola_token_scores",
        ],
        "public_dynamic_score": float(public_dynamic.score),
        "low_level_dynamic_score": low_level_dynamic_score,
        "public_static_scores": public_static_scores,
        "low_level_static_scores": static_scores,
        "selected_layers": selected_layers,
        "premature_layer_dist": _serialize_layer_dist(layer_dist),
        "token_reports": _build_js_report_lookup(token_reports),
        "average_top1_top2_margin": float(avg_margin),
        "public_dynamic_matches_low_level": _float_close(public_dynamic.score, low_level_dynamic_score),
        "public_static_matches_low_level": _dict_float_close(public_static_scores, static_scores),
    }



def _run_oracle_probe_route(
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
        raise ImportError("torch is required for scoring-path tracing.") from error

    _, input_ids, attention_mask, prompt_len = _prepare_scoring_inputs(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        candidate_answer=candidate_answer,
        separator=" ",
    )
    continuation_start = _get_continuation_start_index(prompt_len)
    num_hidden_layers = int(getattr(model.config, "num_hidden_layers", 0))

    lm_head = model.get_output_embeddings()
    if lm_head is None:
        raise ValueError("The model does not expose output embeddings for oracle-path tracing.")
    final_norm = _get_final_norm_module(model)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
    hidden_states = outputs.hidden_states
    if hidden_states is None:
        raise ValueError("The model did not return hidden_states for oracle-path tracing.")

    official_mature_layer = _local_to_official_layer(mature_layer, num_hidden_layers)
    official_candidate_layers = [
        _local_to_official_layer(layer, num_hidden_layers) for layer in candidate_layers
    ]

    mature_logits = _oracle_hidden_and_logits(
        hidden_states,
        lm_head,
        final_norm,
        official_mature_layer,
        num_hidden_layers,
    )["logits"]
    candidate_logits = torch.stack(
        [
            _oracle_hidden_and_logits(
                hidden_states,
                lm_head,
                final_norm,
                official_layer,
                num_hidden_layers,
            )["logits"]
            for official_layer in official_candidate_layers
        ],
        dim=0,
    )

    js_divergence = _compute_dynamic_js_divergence(
        mature_logits=mature_logits,
        candidate_logits=candidate_logits,
    )
    base_logits, _ = _select_dynamic_base_logits(
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

    static_scores: dict[str, float] = {}
    for layer, layer_logits in zip(candidate_layers, candidate_logits, strict=False):
        static_token_scores = _compute_official_dola_token_scores(
            mature_logits=mature_logits,
            base_logits=layer_logits,
            input_ids=input_ids,
            post_softmax=post_softmax,
            relative_top=relative_top,
            relative_top_value=relative_top_value,
        )
        static_scores[str(layer)] = float(static_token_scores[:, continuation_start:].sum().item())

    token_reports, selected_layers, layer_dist, avg_margin = _build_dynamic_trace(
        js_divergence=js_divergence,
        candidate_layers=candidate_layers,
        continuation_start=continuation_start,
    )
    return {
        "path_name": "oracle_probe_path",
        "function_chain": [
            "scripts.hf_probe_early_exit_oracle_parity._oracle_hidden_and_logits",
            "src.generation._prepare_scoring_inputs",
            "src.generation._compute_dynamic_js_divergence",
            "src.generation._select_dynamic_base_logits",
            "src.generation._compute_official_dola_token_scores",
        ],
        "dynamic_score": float(dynamic_token_scores[:, continuation_start:].sum().item()),
        "static_scores": static_scores,
        "selected_layers": selected_layers,
        "premature_layer_dist": _serialize_layer_dist(layer_dist),
        "token_reports": _build_js_report_lookup(token_reports),
        "average_top1_top2_margin": float(avg_margin),
        "official_candidate_layers": official_candidate_layers,
        "official_mature_layer": official_mature_layer,
    }



def _run_audit_route(
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
    report = _compute_bucket_report(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        candidate_answer=candidate_answer,
        candidate_layers=candidate_layers,
        mature_layer=mature_layer,
        post_softmax=post_softmax,
        relative_top=relative_top,
        relative_top_value=relative_top_value,
    )
    return {
        "path_name": "dynamic_collapse_audit_path",
        "function_chain": [
            "scripts.hf_audit_dynamic_collapse._compute_bucket_report",
            "src.generation._prepare_scoring_inputs",
            "src.generation._compute_dynamic_js_divergence",
            "src.generation._select_dynamic_base_logits",
            "src.generation._compute_official_dola_token_scores",
        ],
        "dynamic_score": float(report["dynamic_score"]),
        "static_scores": {str(layer): float(score) for layer, score in report["static_scores"].items()},
        "selected_layers": [int(layer) for layer in report["selected_layers"]],
        "premature_layer_dist": report["premature_layer_dist"],
        "token_reports": _build_js_report_lookup(report["token_reports"]),
        "average_top1_top2_margin": float(report["average_top1_top2_margin"]),
    }



def _build_pair_match(
    left_name: str,
    left: dict[str, object],
    right_name: str,
    right: dict[str, object],
) -> dict[str, object]:
    left_dynamic = left.get("dynamic_score", left.get("low_level_dynamic_score"))
    right_dynamic = right.get("dynamic_score", right.get("low_level_dynamic_score"))
    left_static = left.get("static_scores", left.get("low_level_static_scores"))
    right_static = right.get("static_scores", right.get("low_level_static_scores"))

    dynamic_score_match = _float_close(float(left_dynamic), float(right_dynamic))
    static_scores_match = _dict_float_close(left_static, right_static)
    selected_layers_match = left["selected_layers"] == right["selected_layers"]
    token_reports_match = _token_reports_match(left["token_reports"], right["token_reports"])
    return {
        "left": left_name,
        "right": right_name,
        "dynamic_score_match": dynamic_score_match,
        "dynamic_score_delta": float(left_dynamic) - float(right_dynamic),
        "static_scores_match": static_scores_match,
        "selected_layers_match": selected_layers_match,
        "token_reports_match": token_reports_match,
        "all_match": dynamic_score_match and static_scores_match and selected_layers_match and token_reports_match,
    }



def _classify_consistency(case_reports: list[dict[str, object]]) -> str:
    if not case_reports:
        raise ValueError("case_reports must contain at least one case.")

    main_oracle_all = all(case["matches"]["main_vs_oracle"]["all_match"] for case in case_reports)
    main_audit_all = all(case["matches"]["main_vs_audit"]["all_match"] for case in case_reports)
    oracle_audit_all = all(case["matches"]["oracle_vs_audit"]["all_match"] for case in case_reports)

    if not main_oracle_all:
        return "LIKELY_SHARED_GENERATION_SIDE_EFFECT"
    if not main_audit_all or not oracle_audit_all:
        return "LIKELY_AUDIT_SCRIPT_PATH_MISMATCH"

    avg_margin = sum(float(case["main_compare_eval_path"]["average_top1_top2_margin"]) for case in case_reports) / len(case_reports)
    avg_dynamic_minus_best_static = 0.0
    for case in case_reports:
        static_scores = [float(score) for score in case["main_compare_eval_path"]["low_level_static_scores"].values()]
        avg_dynamic_minus_best_static += float(case["main_compare_eval_path"]["low_level_dynamic_score"]) - max(static_scores)
    avg_dynamic_minus_best_static /= len(case_reports)

    bucket_pairs: dict[tuple[str, str], float] = {}
    for case in case_reports:
        key = (case["answer_name"], case["bucket_name"])
        bucket_pairs[key] = float(case["main_compare_eval_path"]["low_level_dynamic_score"])
    shifted_deltas: list[float] = []
    for answer_name in {case["answer_name"] for case in case_reports}:
        current_key = (answer_name, "current_bucket")
        shifted_key = (answer_name, "shifted_bucket")
        if current_key in bucket_pairs and shifted_key in bucket_pairs:
            shifted_deltas.append(abs(bucket_pairs[current_key] - bucket_pairs[shifted_key]))
    avg_shifted_delta = sum(shifted_deltas) / len(shifted_deltas) if shifted_deltas else 0.0

    if avg_margin <= 1e-5 and abs(avg_dynamic_minus_best_static) <= 1e-5 and avg_shifted_delta <= 1e-5:
        return "LIKELY_AUDIT_JUDGEMENT_RULE_IS_WRONG"
    return "LIKELY_TRUE_MISTRAL_DYNAMIC_COLLAPSE"



def _print_case_summary(case: dict[str, object]) -> None:
    main_path = case["main_compare_eval_path"]
    print(
        f"[hf_trace_scoring_path_consistency] answer={case['answer_name']} bucket={case['bucket_name']}: "
        f"dynamic={main_path['low_level_dynamic_score']:.6f}, "
        f"dominant={max(case['main_compare_eval_path']['premature_layer_dist'].items(), key=lambda item: item[1])[0]}, "
        f"avg_margin={main_path['average_top1_top2_margin']:.6f}, "
        f"main==oracle={case['matches']['main_vs_oracle']['all_match']}, "
        f"main==audit={case['matches']['main_vs_audit']['all_match']}"
    )



def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    output_dir = ensure_output_dir(
        config.get("output_dir", PROJECT_ROOT / "outputs" / "trace_scoring_path_consistency")
    )

    model_name = str(config["model_name"])
    device = config.get("device", "cpu")
    csv_path = Path(str(config["csv_path"]))
    sample_index = int(config.get("sample_index", 0))
    prompt_style = str(config.get("prompt_style", "official_tfqa_mc"))
    mature_layer = int(config["mature_layer"])
    current_bucket = [int(layer) for layer in config["candidate_premature_layers"]]
    shifted_bucket = [int(layer) for layer in config["candidate_premature_layers_shifted"]]
    post_softmax = bool(config.get("post_softmax", False))
    relative_top = float(config.get("relative_top", 0.0))
    relative_top_value = float(config.get("relative_top_value", -1000.0))

    model_kwargs = {key: config[key] for key in LOAD_CONFIG_KEYS if key in config}

    samples = load_truthfulqa_samples(csv_path)
    if sample_index < 0 or sample_index >= len(samples):
        raise IndexError(f"sample_index {sample_index} is out of range for {len(samples)} loaded samples.")
    sample = samples[sample_index]
    prompt = build_mc_prompt(sample, prompt_style=prompt_style)

    print(f"[hf_trace_scoring_path_consistency] Model: {model_name}")
    print(f"[hf_trace_scoring_path_consistency] Sample index: {sample_index}")
    print(f"[hf_trace_scoring_path_consistency] Output directory: {output_dir}")

    model, tokenizer = load_model_and_tokenizer(
        model_name=model_name,
        device=device,
        **model_kwargs,
    )

    case_reports: list[dict[str, object]] = []
    buckets = [
        ("current_bucket", current_bucket),
        ("shifted_bucket", shifted_bucket),
    ]
    for answer_name, candidate_answer in _build_answer_cases(sample, prompt_style):
        for bucket_name, candidate_layers in buckets:
            main_report = _run_main_route_trace(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                candidate_answer=candidate_answer,
                candidate_layers=candidate_layers,
                mature_layer=mature_layer,
                post_softmax=post_softmax,
                relative_top=relative_top,
                relative_top_value=relative_top_value,
            )
            oracle_report = _run_oracle_probe_route(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                candidate_answer=candidate_answer,
                candidate_layers=candidate_layers,
                mature_layer=mature_layer,
                post_softmax=post_softmax,
                relative_top=relative_top,
                relative_top_value=relative_top_value,
            )
            audit_report = _run_audit_route(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                candidate_answer=candidate_answer,
                candidate_layers=candidate_layers,
                mature_layer=mature_layer,
                post_softmax=post_softmax,
                relative_top=relative_top,
                relative_top_value=relative_top_value,
            )
            case = {
                "sample_index": sample_index,
                "question": sample.question,
                "prompt": prompt,
                "answer_name": answer_name,
                "answer": candidate_answer,
                "bucket_name": bucket_name,
                "bucket": candidate_layers,
                "main_compare_eval_path": main_report,
                "oracle_probe_path": oracle_report,
                "dynamic_collapse_audit_path": audit_report,
                "matches": {
                    "main_vs_oracle": _build_pair_match("main_compare_eval_path", main_report, "oracle_probe_path", oracle_report),
                    "main_vs_audit": _build_pair_match("main_compare_eval_path", main_report, "dynamic_collapse_audit_path", audit_report),
                    "oracle_vs_audit": _build_pair_match("oracle_probe_path", oracle_report, "dynamic_collapse_audit_path", audit_report),
                },
            }
            case_reports.append(case)
            _print_case_summary(case)

    final_conclusion = _classify_consistency(case_reports)
    report = {
        "task_name": str(config.get("task_name", "hf_trace_scoring_path_consistency")),
        "model_name": model_name,
        "csv_path": str(csv_path),
        "sample_index": sample_index,
        "prompt_style": prompt_style,
        "mature_layer": mature_layer,
        "cases": case_reports,
        "final_conclusion": final_conclusion,
    }

    output_path = output_dir / "trace_scoring_path_consistency.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(f"[hf_trace_scoring_path_consistency] final_conclusion={final_conclusion}")
    print(f"[hf_trace_scoring_path_consistency] Saved trace to: {output_path}")


if __name__ == "__main__":
    main()
