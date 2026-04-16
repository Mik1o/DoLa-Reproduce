"""Opt-in JSONL analysis logging for TruthfulQA-MC scoring."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.generation import CandidateScore
from src.truthfulqa_mc import TruthfulQASample


class TruthfulQAMCAnalysisLogger:
    """Write sample- and candidate-level TruthfulQA-MC analysis JSONL records."""

    def __init__(self, log_dir: str | Path, max_examples: int | None = None) -> None:
        if max_examples is not None and max_examples < 0:
            raise ValueError("analysis_log_max_examples must be non-negative when provided.")
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.sample_level_path = self.log_dir / "sample_level.jsonl"
        self.candidate_level_path = self.log_dir / "candidate_level.jsonl"
        self.max_examples = max_examples
        self.logged_examples = 0

        self.sample_level_path.write_text("", encoding="utf-8")
        self.candidate_level_path.write_text("", encoding="utf-8")

    def should_log(self, sample_idx: int) -> bool:
        del sample_idx
        return self.max_examples is None or self.logged_examples < self.max_examples

    def log_sample(
        self,
        *,
        sample_idx: int,
        sample: TruthfulQASample,
        true_candidates: list[str],
        false_candidates: list[str],
        vanilla_true: list[CandidateScore],
        vanilla_false: list[CandidateScore],
        dola_true: list[CandidateScore],
        dola_false: list[CandidateScore],
        mature_layer: int,
        num_hidden_layers: int,
        premature_layer: int,
        candidate_premature_layers: list[int] | None,
        dola_score_mode: str,
        score_mode: str,
    ) -> bool:
        if not self.should_log(sample_idx):
            return False

        sample_record, candidate_records = build_truthfulqa_mc_analysis_records(
            sample_idx=sample_idx,
            sample=sample,
            true_candidates=true_candidates,
            false_candidates=false_candidates,
            vanilla_true=vanilla_true,
            vanilla_false=vanilla_false,
            dola_true=dola_true,
            dola_false=dola_false,
            mature_layer=mature_layer,
            num_hidden_layers=num_hidden_layers,
            premature_layer=premature_layer,
            candidate_premature_layers=candidate_premature_layers,
            dola_score_mode=dola_score_mode,
            score_mode=score_mode,
        )

        with self.sample_level_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(sample_record, ensure_ascii=False) + "\n")
        with self.candidate_level_path.open("a", encoding="utf-8") as handle:
            for record in candidate_records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        self.logged_examples += 1
        return True


def maybe_create_truthfulqa_mc_analysis_logger(
    config: dict[str, Any],
    *,
    output_dir: str | Path,
) -> TruthfulQAMCAnalysisLogger | None:
    """Create the opt-in logger from existing YAML config keys."""
    if not bool(config.get("enable_analysis_logging", False)):
        return None

    raw_log_dir = config.get("analysis_log_dir")
    log_dir = Path(str(raw_log_dir)) if raw_log_dir else Path(output_dir) / "analysis_logs"
    raw_max_examples = config.get("analysis_log_max_examples")
    max_examples = None if raw_max_examples is None else int(raw_max_examples)
    return TruthfulQAMCAnalysisLogger(log_dir, max_examples=max_examples)


def build_truthfulqa_mc_analysis_records(
    *,
    sample_idx: int,
    sample: TruthfulQASample,
    true_candidates: list[str],
    false_candidates: list[str],
    vanilla_true: list[CandidateScore],
    vanilla_false: list[CandidateScore],
    dola_true: list[CandidateScore],
    dola_false: list[CandidateScore],
    mature_layer: int,
    num_hidden_layers: int,
    premature_layer: int,
    candidate_premature_layers: list[int] | None,
    dola_score_mode: str,
    score_mode: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    vanilla_candidates = [*vanilla_true, *vanilla_false]
    dola_candidates = [*dola_true, *dola_false]
    candidate_texts = [*true_candidates, *false_candidates]
    is_true_flags = [True] * len(true_candidates) + [False] * len(false_candidates)

    if len(vanilla_candidates) != len(dola_candidates):
        raise ValueError("vanilla and DoLa candidate counts must match for analysis logging.")
    if len(candidate_texts) != len(vanilla_candidates):
        raise ValueError("candidate text count must match scored candidate count for analysis logging.")
    if not vanilla_candidates:
        raise ValueError("analysis logging requires at least one scored candidate.")

    vanilla_ranks = _rank_candidate_scores(vanilla_candidates)
    dola_ranks = _rank_candidate_scores(dola_candidates)
    vanilla_top_idx = _top_candidate_index(vanilla_candidates)
    dola_top_idx = _top_candidate_index(dola_candidates)
    vanilla_correct = bool(is_true_flags[vanilla_top_idx])
    dola_correct = bool(is_true_flags[dola_top_idx])
    is_dynamic = str(dola_score_mode).strip().lower() == "official_dynamic_dola"
    premature_bucket = None
    if is_dynamic and candidate_premature_layers is not None:
        premature_bucket = [int(layer) for layer in candidate_premature_layers]

    sample_record: dict[str, Any] = {
        "sample_idx": int(sample_idx),
        "question": sample.question,
        "true_answers": list(true_candidates),
        "false_answers": list(false_candidates),
        "vanilla_pred": candidate_texts[vanilla_top_idx],
        "dola_pred": candidate_texts[dola_top_idx],
        "vanilla_correct": vanilla_correct,
        "dola_correct": dola_correct,
        "vanilla_top_score": float(vanilla_candidates[vanilla_top_idx].score),
        "dola_top_score": float(dola_candidates[dola_top_idx].score),
        "mature_layer": int(mature_layer),
        "mature_side": _mature_side(mature_layer, num_hidden_layers),
        "premature_layer": None if is_dynamic else int(premature_layer),
        "premature_bucket": premature_bucket,
        "dola_score_mode": str(dola_score_mode),
        "changed_by_dola": bool(vanilla_top_idx != dola_top_idx),
        "improved_by_dola": bool((not vanilla_correct) and dola_correct),
        "worsened_by_dola": bool(vanilla_correct and not dola_correct),
    }

    candidate_records: list[dict[str, Any]] = []
    for candidate_idx, (vanilla_item, dola_item, candidate_text, is_true) in enumerate(
        zip(vanilla_candidates, dola_candidates, candidate_texts, is_true_flags, strict=True)
    ):
        vanilla_trace = vanilla_item.trace
        dola_trace = dola_item.trace
        token_ids = _first_non_empty_list(
            _trace_value(vanilla_trace, "token_ids"),
            _trace_value(dola_trace, "token_ids"),
        )
        token_texts = _first_non_empty_list(
            _trace_value(vanilla_trace, "token_texts"),
            _trace_value(dola_trace, "token_texts"),
        )
        scoring_start = _first_non_none(
            _trace_value(vanilla_trace, "scoring_start_token_index"),
            _trace_value(dola_trace, "scoring_start_token_index"),
        )
        vanilla_total, vanilla_avg = _total_and_avg(vanilla_item, score_mode)
        dola_total, dola_avg = _total_and_avg(dola_item, score_mode)

        candidate_records.append(
            {
                "sample_idx": int(sample_idx),
                "candidate_idx": int(candidate_idx),
                "candidate_text": candidate_text,
                "is_true_candidate": bool(is_true),
                "token_ids": token_ids,
                "token_texts": token_texts,
                "scoring_start_token_index": scoring_start,
                "vanilla_token_logprobs": _trace_list(vanilla_trace, "token_scores"),
                "dola_final_token_scores": _trace_list(dola_trace, "final_token_scores"),
                "dola_premature_token_scores": _trace_list(dola_trace, "premature_token_scores"),
                "dola_token_contrast_scores": _trace_list(dola_trace, "token_scores"),
                "vanilla_total_score": vanilla_total,
                "vanilla_avg_score": vanilla_avg,
                "dola_total_score": dola_total,
                "dola_avg_score": dola_avg,
                "vanilla_rank": int(vanilla_ranks[candidate_idx]),
                "dola_rank": int(dola_ranks[candidate_idx]),
                "selected_premature_layers": _trace_list(dola_trace, "selected_premature_layers"),
                "premature_layer_dist": _serialize_layer_dist(dola_item.premature_layer_dist),
            }
        )

    return sample_record, candidate_records


def _rank_candidate_scores(items: list[CandidateScore]) -> list[int]:
    ranked_indices = sorted(range(len(items)), key=lambda index: (-float(items[index].score), index))
    ranks = [0] * len(items)
    for rank, index in enumerate(ranked_indices, start=1):
        ranks[index] = rank
    return ranks


def _top_candidate_index(items: list[CandidateScore]) -> int:
    return max(range(len(items)), key=lambda index: (float(items[index].score), -index))


def _mature_side(mature_layer: int, num_hidden_layers: int) -> str:
    if int(num_hidden_layers) > 0 and int(mature_layer) == int(num_hidden_layers) - 1:
        return "final"
    return "custom"


def _total_and_avg(item: CandidateScore, score_mode: str) -> tuple[float, float]:
    if item.trace is not None:
        return float(item.trace.total_score), float(item.trace.avg_score)

    token_count = int(item.continuation_token_count)
    score = float(item.score)
    if token_count <= 0:
        return score, 0.0
    if str(score_mode).strip().lower() == "mean_logprob":
        return float(score * token_count), score
    return score, float(score / token_count)


def _trace_value(trace: Any | None, field_name: str) -> Any:
    if trace is None:
        return None
    return getattr(trace, field_name)


def _trace_list(trace: Any | None, field_name: str) -> list[Any]:
    value = _trace_value(trace, field_name)
    if value is None:
        return []
    return list(value)


def _first_non_empty_list(*values: Any) -> list[Any]:
    for value in values:
        if value:
            return list(value)
    return []


def _first_non_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _serialize_layer_dist(layer_dist: dict[int, int] | None) -> dict[str, int] | None:
    if not layer_dist:
        return None
    return {str(layer): int(count) for layer, count in sorted(layer_dist.items())}
