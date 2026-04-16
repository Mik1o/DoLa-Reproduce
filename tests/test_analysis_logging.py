from __future__ import annotations

import json
from pathlib import Path

from scripts.inspect_mc_analysis_logs import inspect_analysis_logs


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_inspect_mc_analysis_logs_generates_markdown_with_fallback_margins(tmp_path) -> None:
    sample_log = tmp_path / "sample_level.jsonl"
    candidate_log = tmp_path / "candidate_level.jsonl"
    output_md = tmp_path / "manual_review_cases.md"

    _write_jsonl(
        sample_log,
        [
            {
                "sample_idx": 0,
                "question": "Which city is the capital of France?",
                "true_answers": ["Paris."],
                "false_answers": ["Rome.", "Berlin."],
                "vanilla_pred": "Rome.",
                "dola_pred": "Paris.",
                "vanilla_correct": False,
                "dola_correct": True,
                "changed_by_dola": True,
                "improved_by_dola": True,
                "worsened_by_dola": False,
            },
            {
                "sample_idx": 1,
                "question": "Which planet is known as the red planet?",
                "true_answers": ["Mars."],
                "false_answers": ["Venus.", "Jupiter."],
                "vanilla_pred": "Mars.",
                "dola_pred": "Venus.",
                "vanilla_correct": True,
                "dola_correct": False,
                "changed_by_dola": True,
                "improved_by_dola": False,
                "worsened_by_dola": True,
            },
            {
                "sample_idx": 2,
                "question": "What is the safe answer to an unknown trick question?",
                "true_answers": ["I have no comment."],
                "false_answers": ["A myth.", "A rumor."],
                "vanilla_pred": "A myth.",
                "dola_pred": "A rumor.",
                "vanilla_correct": False,
                "dola_correct": False,
                "changed_by_dola": True,
                "improved_by_dola": False,
                "worsened_by_dola": False,
            },
        ],
    )
    _write_jsonl(
        candidate_log,
        [
            _candidate(0, 0, "Paris.", True, vanilla=1.0, dola=4.0),
            _candidate(0, 1, "Rome.", False, vanilla=3.0, dola=2.0),
            _candidate(0, 2, "Berlin.", False, vanilla=2.0, dola=1.0, include_tokens=False),
            _candidate(1, 0, "Mars.", True, vanilla=5.0, dola=2.0),
            _candidate(1, 1, "Venus.", False, vanilla=2.0, dola=4.0),
            _candidate(1, 2, "Jupiter.", False, vanilla=1.0, dola=1.0),
            _candidate(2, 0, "I have no comment.", True, vanilla=0.0, dola=0.0),
            _candidate(2, 1, "A myth.", False, vanilla=5.0, dola=6.0),
            _candidate(2, 2, "A rumor.", False, vanilla=4.0, dola=9.0),
        ],
    )

    result = inspect_analysis_logs(
        sample_log=sample_log,
        candidate_log=candidate_log,
        output_md=output_md,
        max_cases_per_group=2,
        max_token_rows=2,
    )

    markdown = output_md.read_text(encoding="utf-8")
    assert output_md.is_file()
    assert "## Group A: vanilla ? / DoLa ?" in markdown
    assert "## Group B: vanilla ? / DoLa ?" in markdown
    assert "## Group C: ????? DoLa ???" in markdown
    assert "both wrong but DoLa more confident: 1" in markdown
    assert "Average vanilla top1-top2 margin" in markdown
    assert "Token-level comparison for candidate" in markdown
    assert result["summary"]["both_wrong_and_dola_more_confident"] == 1
    assert result["summary"]["avg_vanilla_margin"] > 0.0
    assert result["summary"]["avg_dola_margin"] > result["summary"]["avg_vanilla_margin"]


def _candidate(
    sample_idx: int,
    candidate_idx: int,
    text: str,
    is_true: bool,
    *,
    vanilla: float,
    dola: float,
    include_tokens: bool = True,
) -> dict[str, object]:
    row: dict[str, object] = {
        "sample_idx": sample_idx,
        "candidate_idx": candidate_idx,
        "candidate_text": text,
        "is_true_candidate": is_true,
        "vanilla_total_score": vanilla,
        "vanilla_avg_score": vanilla,
        "dola_total_score": dola,
        "dola_avg_score": dola,
        "vanilla_rank": 1,
        "dola_rank": 1,
        "vanilla_token_logprobs": [vanilla],
        "dola_final_token_scores": [dola + 0.25],
        "dola_premature_token_scores": [0.25],
        "dola_token_contrast_scores": [dola],
    }
    if include_tokens:
        row["token_ids"] = [candidate_idx + 10]
        row["token_texts"] = [text.split()[0] if text.split() else text]
    return row
