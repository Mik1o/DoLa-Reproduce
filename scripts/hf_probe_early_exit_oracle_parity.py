"""Probe current early-exit extraction against an official DoLa-style oracle path."""

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
    _compute_official_dola_token_scores,
    _count_premature_layer_usage,
    _get_continuation_start_index,
    _prepare_scoring_inputs,
    _select_dynamic_base_logits,
)
from src.modeling import load_model_and_tokenizer
from src.truthfulqa_mc import (
    BEST_ANSWER_FIELDS,
    CORRECT_ANSWER_FIELDS,
    INCORRECT_ANSWER_FIELDS,
    build_answer_continuation,
    build_mc_prompt,
    load_truthfulqa_csv,
    normalize_truthfulqa_row,
    parse_list_field,
)
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
        description="Probe current early-exit logits against an official-style oracle path."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "mistral7b_truthfulqa_real_probe_oracle_parity.yaml",
        help="Path to a YAML config file.",
    )
    return parser.parse_args()



def _official_format_best(text: str) -> str:
    cleaned = text.strip()
    if cleaned and cleaned[-1] != ".":
        cleaned += "."
    return cleaned



def _official_split_multi_answer(text: str, sep: str = ";") -> list[str]:
    answers: list[str] = []
    for item in text.strip().split(sep):
        cleaned = item.strip()
        if cleaned:
            answers.append(_official_format_best(cleaned))
    return answers



def _official_normalize_answer_field(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str) and not raw.strip():
        return []
    if isinstance(raw, str) and ";" in raw:
        return _official_split_multi_answer(raw)
    parsed = parse_list_field(raw, field_name="oracle_answer_field")
    return [_official_format_best(item) for item in parsed]



def _get_optional_row_value(row: Any, field_names: tuple[str, ...]) -> Any:
    for field_name in field_names:
        if field_name in row.index:
            return row[field_name]
    return None



def _build_answer_format_audit(row: Any, sample: Any) -> dict[str, object]:
    raw_best = _get_optional_row_value(row, BEST_ANSWER_FIELDS)
    raw_correct = _get_optional_row_value(row, CORRECT_ANSWER_FIELDS)
    raw_incorrect = _get_optional_row_value(row, INCORRECT_ANSWER_FIELDS)

    oracle_best = _official_format_best(str(raw_best)) if raw_best is not None else None
    oracle_correct = _official_normalize_answer_field(raw_correct)
    oracle_incorrect = _official_normalize_answer_field(raw_incorrect)
    if not oracle_correct and oracle_best is not None:
        oracle_correct = [oracle_best]
    if oracle_best is not None and oracle_best not in oracle_correct:
        oracle_correct = [oracle_best, *oracle_correct]

    return {
        "raw_best_answer": raw_best,
        "raw_correct_answers": raw_correct,
        "raw_incorrect_answers": raw_incorrect,
        "current_best_answer": sample.best_answer,
        "oracle_best_answer": oracle_best,
        "best_answer_match": sample.best_answer == oracle_best,
        "current_correct_answers": sample.correct_answers,
        "oracle_correct_answers": oracle_correct,
        "correct_answers_match": sample.correct_answers == oracle_correct,
        "current_incorrect_answers": sample.incorrect_answers,
        "oracle_incorrect_answers": oracle_incorrect,
        "incorrect_answers_match": sample.incorrect_answers == oracle_incorrect,
    }



def _resolve_candidate_text(sample: Any, config: dict[str, Any]) -> str:
    explicit_candidate = config.get("candidate_answer")
    if explicit_candidate:
        return str(explicit_candidate)

    candidate_source = str(config.get("candidate_source", "best_answer")).strip().lower()
    if candidate_source == "best_answer":
        return sample.best_answer
    if candidate_source == "first_true":
        return sample.correct_answers[0]
    if candidate_source == "first_false":
        return sample.incorrect_answers[0]
    raise ValueError(
        "Unsupported candidate_source "
        f"'{candidate_source}'. Use 'best_answer', 'first_true', 'first_false', or set candidate_answer."
    )



def _get_final_norm_module(model: Any) -> Any | None:
    base_model = getattr(model, "model", None)
    if base_model is None:
        return None
    return getattr(base_model, "norm", None)



def _current_hidden_and_logits(hidden_states: Any, lm_head: Any, local_layer_index: int, num_hidden_layers: int) -> dict[str, Any]:
    hidden_state = hidden_states[local_layer_index + 1]
    logits = lm_head(hidden_state[:, :-1, :])
    return {
        "hidden_state": hidden_state,
        "logits": logits,
        "used_final_norm": local_layer_index == num_hidden_layers - 1,
        "tuple_index": local_layer_index + 1,
    }



def _oracle_hidden_and_logits(
    hidden_states: Any,
    lm_head: Any,
    final_norm: Any | None,
    official_layer_id: int,
    num_hidden_layers: int,
) -> dict[str, Any]:
    if official_layer_id <= 0 or official_layer_id > num_hidden_layers:
        raise ValueError(
            f"official layer id {official_layer_id} must be within [1, {num_hidden_layers}]"
        )

    if official_layer_id == num_hidden_layers:
        hidden_state = hidden_states[-1]
        used_final_norm = True
        tuple_index = len(hidden_states) - 1
        forced_norm_logits = None
    else:
        hidden_state = hidden_states[official_layer_id]
        used_final_norm = False
        tuple_index = official_layer_id
        forced_norm_logits = None
        if final_norm is not None:
            forced_norm_logits = lm_head(final_norm(hidden_state)[:, :-1, :])

    logits = lm_head(hidden_state[:, :-1, :])
    return {
        "hidden_state": hidden_state,
        "logits": logits,
        "used_final_norm": used_final_norm,
        "tuple_index": tuple_index,
        "forced_norm_logits": forced_norm_logits,
    }



def _token_to_text(tokenizer: Any, token_id: int) -> str:
    if hasattr(tokenizer, "convert_ids_to_tokens"):
        token = tokenizer.convert_ids_to_tokens([token_id])[0]
        return str(token)
    return tokenizer.decode([token_id])



def _topk_report(tokenizer: Any, logits_row: Any, k: int = 10) -> list[dict[str, object]]:
    try:
        import torch
    except ImportError as error:
        raise ImportError("torch is required for oracle parity probing.") from error

    log_probs = torch.log_softmax(logits_row, dim=-1)
    top_values, top_indices = torch.topk(logits_row, k=min(k, logits_row.shape[-1]))
    result: list[dict[str, object]] = []
    for logit_value, token_id in zip(top_values.tolist(), top_indices.tolist(), strict=False):
        result.append(
            {
                "token_id": int(token_id),
                "token": _token_to_text(tokenizer, int(token_id)),
                "logit": float(logit_value),
                "logprob": float(log_probs[int(token_id)].item()),
            }
        )
    return result



def _tensor_diff_stats(current_tensor: Any, oracle_tensor: Any) -> dict[str, float]:
    diff = (current_tensor - oracle_tensor).abs()
    return {
        "max_abs_diff": float(diff.max().item()),
        "mean_abs_diff": float(diff.mean().item()),
    }



def _best_shift_alignment_mean_abs(
    current_tensor: Any,
    hidden_states: Any,
    lm_head: Any,
    official_layer_id: int,
    continuation_start: int,
) -> dict[str, float | int | None]:
    candidate_diffs: dict[int, float] = {}
    for neighbor in (official_layer_id - 1, official_layer_id, official_layer_id + 1):
        if neighbor <= 0 or neighbor >= len(hidden_states):
            continue
        oracle_logits = lm_head(hidden_states[neighbor][:, :-1, :])[:, continuation_start:, :]
        diff = (current_tensor - oracle_logits).abs().mean().item()
        candidate_diffs[neighbor] = float(diff)
    if not candidate_diffs:
        return {"best_matching_official_layer": None, "best_mean_abs_diff": math.inf}
    best_layer = min(candidate_diffs, key=candidate_diffs.get)
    return {
        "best_matching_official_layer": int(best_layer),
        "best_mean_abs_diff": float(candidate_diffs[best_layer]),
    }



def _convert_local_to_official_layers(local_layers: list[int], num_hidden_layers: int) -> list[int]:
    result: list[int] = []
    for local_layer in local_layers:
        if local_layer == num_hidden_layers - 1:
            result.append(num_hidden_layers)
        else:
            result.append(local_layer + 1)
    return result



def _classify_probe(
    layer_reports: list[dict[str, object]],
    static_reports: list[dict[str, object]],
    dynamic_report: dict[str, object],
) -> str:
    norm_suspicions = 0
    layer_index_suspicions = 0
    for report in layer_reports:
        diff_stats = report["diff_stats"]
        if report.get("forced_norm_diff_stats") and not bool(report["oracle_used_final_norm"]):
            forced_stats = report["forced_norm_diff_stats"]
            if float(forced_stats["mean_abs_diff"]) + 1e-6 < float(diff_stats["mean_abs_diff"]):
                norm_suspicions += 1
        shifted = report.get("best_shift_alignment")
        if shifted and shifted["best_matching_official_layer"] not in (None, report["official_layer_id"]):
            if float(shifted["best_mean_abs_diff"]) + 1e-6 < float(diff_stats["mean_abs_diff"]):
                layer_index_suspicions += 1

    static_close = all(abs(float(item["score_delta"])) < 1e-5 for item in static_reports)
    dynamic_score_close = abs(float(dynamic_report["score_delta"])) < 1e-5
    dynamic_selection_match = bool(dynamic_report["selection_match"])

    if norm_suspicions:
        return "LIKELY_NORM_MISMATCH"
    if layer_index_suspicions:
        return "LIKELY_LAYER_INDEX_SEMANTICS_MISMATCH"
    if static_close and (not dynamic_score_close or not dynamic_selection_match):
        return "LIKELY_DYNAMIC_SELECTION_SCORING_MISMATCH"
    return "LIKELY_MATCH"



def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    output_dir = ensure_output_dir(config.get("output_dir", PROJECT_ROOT / "outputs" / "oracle_parity_probe"))

    model_name = str(config["model_name"])
    device = config.get("device", "cpu")
    csv_path = Path(str(config["csv_path"]))
    sample_index = int(config.get("sample_index", 0))
    prompt_style = str(config.get("prompt_style", "official_tfqa_mc"))
    score_mode = str(config.get("score_mode", "sum_logprob"))
    post_softmax = bool(config.get("post_softmax", False))
    relative_top = float(config.get("relative_top", 0.0))
    relative_top_value = float(config.get("relative_top_value", -1000.0))
    oracle_premature_layers = [int(layer) for layer in config.get("oracle_premature_layers", [])]
    oracle_mature_layer = int(config["oracle_mature_layer"])

    if len(oracle_premature_layers) < 2:
        raise ValueError("oracle_premature_layers must contain at least two official layer ids.")

    model_kwargs = {key: config[key] for key in LOAD_CONFIG_KEYS if key in config}

    dataframe = load_truthfulqa_csv(csv_path)
    if sample_index < 0 or sample_index >= len(dataframe):
        raise IndexError(f"sample_index {sample_index} is out of range for {len(dataframe)} rows.")

    row = dataframe.iloc[sample_index]
    sample = normalize_truthfulqa_row(row)
    prompt = build_mc_prompt(sample, prompt_style=prompt_style)
    candidate_text = _resolve_candidate_text(sample, config)
    candidate_continuation = build_answer_continuation(candidate_text, prompt_style=prompt_style)

    print(f"[hf_probe_early_exit_oracle_parity] Model: {model_name}")
    print(f"[hf_probe_early_exit_oracle_parity] Sample index: {sample_index}")
    print(f"[hf_probe_early_exit_oracle_parity] Prompt style: {prompt_style}")
    print(f"[hf_probe_early_exit_oracle_parity] Candidate answer: {candidate_continuation!r}")
    print(f"[hf_probe_early_exit_oracle_parity] Output directory: {output_dir}")

    model, tokenizer = load_model_and_tokenizer(
        model_name=model_name,
        device=device,
        **model_kwargs,
    )

    try:
        import torch
    except ImportError as error:
        raise ImportError("torch is required for oracle parity probing.") from error

    num_hidden_layers = int(getattr(model.config, "num_hidden_layers", 0))
    if oracle_mature_layer != num_hidden_layers:
        raise ValueError(
            f"oracle_mature_layer should usually be {num_hidden_layers} for the final normalized layer, "
            f"but received {oracle_mature_layer}."
        )

    current_local_premature_layers = [layer - 1 for layer in oracle_premature_layers]
    current_local_mature_layer = oracle_mature_layer - 1

    prompt_text, input_ids, attention_mask, prompt_len = _prepare_scoring_inputs(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        candidate_answer=candidate_continuation,
        separator=" ",
    )
    continuation_start = _get_continuation_start_index(prompt_len)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
    hidden_states = outputs.hidden_states
    if hidden_states is None:
        raise ValueError("Model did not return hidden_states for oracle parity probing.")

    lm_head = model.get_output_embeddings()
    if lm_head is None:
        raise ValueError("Model does not expose output embeddings for oracle parity probing.")
    final_norm = _get_final_norm_module(model)

    layer_reports: list[dict[str, object]] = []
    current_candidate_logits: list[Any] = []
    oracle_candidate_logits: list[Any] = []

    for official_layer_id, current_local_layer in zip(
        oracle_premature_layers + [oracle_mature_layer],
        current_local_premature_layers + [current_local_mature_layer],
        strict=False,
    ):
        current_info = _current_hidden_and_logits(hidden_states, lm_head, current_local_layer, num_hidden_layers)
        oracle_info = _oracle_hidden_and_logits(hidden_states, lm_head, final_norm, official_layer_id, num_hidden_layers)

        current_cont_logits = current_info["logits"][:, continuation_start:, :]
        oracle_cont_logits = oracle_info["logits"][:, continuation_start:, :]
        report = {
            "official_layer_id": official_layer_id,
            "current_local_layer": current_local_layer,
            "current_tuple_index": current_info["tuple_index"],
            "oracle_tuple_index": oracle_info["tuple_index"],
            "current_used_final_norm": current_info["used_final_norm"],
            "oracle_used_final_norm": oracle_info["used_final_norm"],
            "logits_shape": list(current_cont_logits.shape),
            "diff_stats": _tensor_diff_stats(current_cont_logits, oracle_cont_logits),
            "current_top10": _topk_report(tokenizer, current_info["logits"][0, continuation_start, :]),
            "oracle_top10": _topk_report(tokenizer, oracle_info["logits"][0, continuation_start, :]),
        }
        if oracle_info["forced_norm_logits"] is not None:
            forced_cont_logits = oracle_info["forced_norm_logits"][:, continuation_start:, :]
            report["forced_norm_diff_stats"] = _tensor_diff_stats(current_cont_logits, forced_cont_logits)
        if official_layer_id != oracle_mature_layer:
            report["best_shift_alignment"] = _best_shift_alignment_mean_abs(
                current_cont_logits,
                hidden_states,
                lm_head,
                official_layer_id,
                continuation_start,
            )
            current_candidate_logits.append(current_info["logits"])
            oracle_candidate_logits.append(oracle_info["logits"])
        layer_reports.append(report)

    current_mature_logits = _current_hidden_and_logits(
        hidden_states,
        lm_head,
        current_local_mature_layer,
        num_hidden_layers,
    )["logits"]
    oracle_mature_logits = _oracle_hidden_and_logits(
        hidden_states,
        lm_head,
        final_norm,
        oracle_mature_layer,
        num_hidden_layers,
    )["logits"]

    static_reports: list[dict[str, object]] = []
    for official_layer_id, current_local_layer, current_base_logits, oracle_base_logits in zip(
        oracle_premature_layers,
        current_local_premature_layers,
        current_candidate_logits,
        oracle_candidate_logits,
        strict=False,
    ):
        current_token_scores = _compute_official_dola_token_scores(
            mature_logits=current_mature_logits,
            base_logits=current_base_logits,
            input_ids=input_ids,
            post_softmax=post_softmax,
            relative_top=relative_top,
            relative_top_value=relative_top_value,
        )
        oracle_token_scores = _compute_official_dola_token_scores(
            mature_logits=oracle_mature_logits,
            base_logits=oracle_base_logits,
            input_ids=input_ids,
            post_softmax=post_softmax,
            relative_top=relative_top,
            relative_top_value=relative_top_value,
        )
        current_score = float(current_token_scores[:, continuation_start:].sum().item())
        oracle_score = float(oracle_token_scores[:, continuation_start:].sum().item())
        static_reports.append(
            {
                "official_premature_layer": official_layer_id,
                "current_local_premature_layer": current_local_layer,
                "current_score": current_score,
                "oracle_score": oracle_score,
                "score_delta": current_score - oracle_score,
            }
        )

    current_candidate_stack = torch.stack(current_candidate_logits, dim=0)
    oracle_candidate_stack = torch.stack(oracle_candidate_logits, dim=0)
    current_base_logits, current_selected_local_layers = _select_dynamic_base_logits(
        mature_logits=current_mature_logits,
        candidate_logits=current_candidate_stack,
        candidate_layers=current_local_premature_layers,
    )
    oracle_base_logits, oracle_selected_official_layers = _select_dynamic_base_logits(
        mature_logits=oracle_mature_logits,
        candidate_logits=oracle_candidate_stack,
        candidate_layers=oracle_premature_layers,
    )
    current_dynamic_scores = _compute_official_dola_token_scores(
        mature_logits=current_mature_logits,
        base_logits=current_base_logits,
        input_ids=input_ids,
        post_softmax=post_softmax,
        relative_top=relative_top,
        relative_top_value=relative_top_value,
    )
    oracle_dynamic_scores = _compute_official_dola_token_scores(
        mature_logits=oracle_mature_logits,
        base_logits=oracle_base_logits,
        input_ids=input_ids,
        post_softmax=post_softmax,
        relative_top=relative_top,
        relative_top_value=relative_top_value,
    )
    current_selected_official_layers = _convert_local_to_official_layers(current_selected_local_layers, num_hidden_layers)
    current_dynamic_score = float(current_dynamic_scores[:, continuation_start:].sum().item())
    oracle_dynamic_score = float(oracle_dynamic_scores[:, continuation_start:].sum().item())
    dynamic_report = {
        "current_selected_layers_per_token": current_selected_official_layers[continuation_start:],
        "oracle_selected_layers_per_token": oracle_selected_official_layers[continuation_start:],
        "current_premature_layer_dist": _count_premature_layer_usage(
            current_selected_official_layers[continuation_start:],
            oracle_premature_layers,
        ),
        "oracle_premature_layer_dist": _count_premature_layer_usage(
            oracle_selected_official_layers[continuation_start:],
            oracle_premature_layers,
        ),
        "selection_match": current_selected_official_layers[continuation_start:] == oracle_selected_official_layers[continuation_start:],
        "current_score": current_dynamic_score,
        "oracle_score": oracle_dynamic_score,
        "score_delta": current_dynamic_score - oracle_dynamic_score,
    }

    answer_format_audit = _build_answer_format_audit(row, sample)
    final_conclusion = _classify_probe(layer_reports, static_reports, dynamic_report)

    result = {
        "task_name": str(config.get("task_name", "hf_probe_early_exit_oracle_parity")),
        "model_name": model_name,
        "csv_path": str(csv_path),
        "sample_index": sample_index,
        "question": sample.question,
        "prompt": prompt,
        "prompt_text_used_for_scoring": prompt_text,
        "candidate_answer": candidate_continuation,
        "oracle_premature_layers": oracle_premature_layers,
        "oracle_mature_layer": oracle_mature_layer,
        "current_local_premature_layers": current_local_premature_layers,
        "current_local_mature_layer": current_local_mature_layer,
        "answer_format_audit": answer_format_audit,
        "layer_reports": layer_reports,
        "static_reports": static_reports,
        "dynamic_report": dynamic_report,
        "final_conclusion": final_conclusion,
    }

    print("[hf_probe_early_exit_oracle_parity] Answer formatting audit:")
    print(json.dumps(answer_format_audit, ensure_ascii=False, indent=2))
    print("[hf_probe_early_exit_oracle_parity] Layer-by-layer diff summary:")
    for report in layer_reports:
        print(
            f"  layer {report['official_layer_id']}: "
            f"current_tuple_index={report['current_tuple_index']}, "
            f"oracle_tuple_index={report['oracle_tuple_index']}, "
            f"current_norm={report['current_used_final_norm']}, "
            f"oracle_norm={report['oracle_used_final_norm']}, "
            f"max_abs_diff={report['diff_stats']['max_abs_diff']:.6f}, "
            f"mean_abs_diff={report['diff_stats']['mean_abs_diff']:.6f}"
        )
    print("[hf_probe_early_exit_oracle_parity] Static score comparison:")
    for report in static_reports:
        print(
            f"  static_{report['official_premature_layer']}: "
            f"current={report['current_score']:.6f}, "
            f"oracle={report['oracle_score']:.6f}, "
            f"delta={report['score_delta']:.6f}"
        )
    print("[hf_probe_early_exit_oracle_parity] Dynamic score comparison:")
    print(
        f"  current_score={dynamic_report['current_score']:.6f}, "
        f"oracle_score={dynamic_report['oracle_score']:.6f}, "
        f"delta={dynamic_report['score_delta']:.6f}"
    )
    print(f"  current_selected_layers_per_token={dynamic_report['current_selected_layers_per_token']}")
    print(f"  oracle_selected_layers_per_token={dynamic_report['oracle_selected_layers_per_token']}")
    print(f"  current_premature_layer_dist={dynamic_report['current_premature_layer_dist']}")
    print(f"  oracle_premature_layer_dist={dynamic_report['oracle_premature_layer_dist']}")
    print(f"[hf_probe_early_exit_oracle_parity] final_conclusion={final_conclusion}")

    output_path = output_dir / "oracle_early_exit_parity_probe.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(f"[hf_probe_early_exit_oracle_parity] Saved report to: {output_path}")


if __name__ == "__main__":
    main()
