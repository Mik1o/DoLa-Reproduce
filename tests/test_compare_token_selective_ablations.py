"""Tests for token-selective ablation comparison reporting."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.compare_token_selective_ablations import main as ablation_main


def test_compare_token_selective_ablations_writes_report_with_fallbacks(tmp_path) -> None:
    good_run = tmp_path / "good_run"
    sparse_run = tmp_path / "sparse_run"
    good_run.mkdir()
    sparse_run.mkdir()

    _write_json(
        good_run / "compare_summary.json",
        {"dola_avg_mc1": 0.3, "dola_avg_mc2": 0.6, "dola_avg_mc3": 0.4},
    )
    _write_json(
        sparse_run / "compare_summary.json",
        {"dola_avg_mc1": 0.2, "dola_avg_mc2": 0.5, "dola_avg_mc3": 0.3},
    )
    _write_jsonl(
        good_run / "sample_level.jsonl",
        [
            {"sample_idx": 0, "changed_by_dola": True, "improved_by_dola": True, "worsened_by_dola": False},
            {"sample_idx": 1, "changed_by_dola": True, "improved_by_dola": False, "worsened_by_dola": True},
        ],
    )
    _write_jsonl(
        sparse_run / "sample_level.jsonl",
        [
            {"sample_idx": 0, "changed_by_dola": True, "improved_by_dola": True, "worsened_by_dola": False},
        ],
    )
    _write_jsonl(
        good_run / "candidate_level.jsonl",
        [
            {
                "sample_idx": 0,
                "candidate_idx": 0,
                "dola_rank": 1,
                "dola_total_score": 3.0,
                "token_contrast_score": [2.0, 1.0],
                "token_contrast_weight": [1.0, 0.5],
                "token_selection_tier": ["strong", "medium"],
                "token_selected_reason": ["number", "medium_adjacent_support"],
            },
            {"sample_idx": 0, "candidate_idx": 1, "dola_rank": 2, "dola_total_score": 1.0},
            {
                "sample_idx": 1,
                "candidate_idx": 0,
                "dola_rank": 1,
                "dola_total_score": 4.0,
                "dola_token_contrast_scores": [3.0],
                "token_contrast_weight": [0.2],
                "token_selection_tier": ["unselected"],
                "token_selected_reason": [""],
            },
            {"sample_idx": 1, "candidate_idx": 1, "dola_rank": 2, "dola_total_score": 2.0},
        ],
    )
    _write_jsonl(
        sparse_run / "candidate_level.jsonl",
        [
            {"sample_idx": 0, "candidate_idx": 0, "dola_rank": 1, "dola_total_score": 1.0},
            {"sample_idx": 0, "candidate_idx": 1, "dola_rank": 2, "dola_total_score": 0.5},
        ],
    )
    _write_jsonl(
        good_run / "per_sample_five_question_review_200.jsonl",
        [
            {
                "sample_idx": 0,
                "q2_label": "fact-critical",
                "q3_label": "mixed",
                "q4_label": "not_applicable",
                "q5_label": "not_applicable_for_failure_type",
            },
            {
                "sample_idx": 1,
                "q2_label": "partial",
                "q3_label": "strengthens_smoother_or_more_common_expression",
                "q4_label": "yes",
                "q5_label": "token_frequency_bias",
            },
        ],
    )

    output_md = tmp_path / "summary.md"
    ablation_main(["--runs", str(good_run), str(sparse_run), "--output-md", str(output_md)])

    report = output_md.read_text(encoding="utf-8")
    assert "Decision guide for v3" in report
    assert "good_run" in report
    assert "sparse_run" in report
    assert "adjacent_support" in report
    assert "evidence_insufficient" in report


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
