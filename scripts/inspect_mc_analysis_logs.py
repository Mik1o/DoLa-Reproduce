"""Build a manual-review markdown report from TruthfulQA-MC analysis logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_MAX_CASES_PER_GROUP = 8
DEFAULT_MAX_TOKEN_ROWS = 12
_MARGIN_KEYS = {
    "vanilla": (
        "vanilla_top1_top2_margin",
        "vanilla_top2_margin",
        "vanilla_margin",
        "vanilla_top_margin",
    ),
    "dola": (
        "dola_top1_top2_margin",
        "dola_top2_margin",
        "dola_margin",
        "dola_top_margin",
    ),
}
_SCORE_FIELDS = {
    "vanilla": ("vanilla_total_score", "vanilla_avg_score"),
    "dola": ("dola_total_score", "dola_avg_score"),
}
_GROUPS = (
    (
        "A",
        "vanilla ? / DoLa ?",
        lambda sample: (not _as_bool(sample.get("vanilla_correct")))
        and _as_bool(sample.get("dola_correct")),
    ),
    (
        "B",
        "vanilla ? / DoLa ?",
        lambda sample: _as_bool(sample.get("vanilla_correct"))
        and (not _as_bool(sample.get("dola_correct"))),
    ),
    (
        "C",
        "????? DoLa ???",
        lambda sample: (not _as_bool(sample.get("vanilla_correct")))
        and (not _as_bool(sample.get("dola_correct")))
        and _finite_number(sample.get("dola_margin"), default=0.0)
        > _finite_number(sample.get("vanilla_margin"), default=0.0),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect TruthfulQA-MC analysis JSONL logs and write a manual-review markdown report."
    )
    parser.add_argument("--sample-log", type=Path, required=True, help="Path to sample_level.jsonl.")
    parser.add_argument("--candidate-log", type=Path, required=True, help="Path to candidate_level.jsonl.")
    parser.add_argument("--output-md", type=Path, required=True, help="Markdown report path to write.")
    parser.add_argument(
        "--max-cases-per-group",
        type=int,
        default=DEFAULT_MAX_CASES_PER_GROUP,
        help=f"Maximum cases to show for each group. Default: {DEFAULT_MAX_CASES_PER_GROUP}.",
    )
    parser.add_argument(
        "--max-token-rows",
        type=int,
        default=DEFAULT_MAX_TOKEN_ROWS,
        help=f"Maximum token rows to show per selected candidate. Default: {DEFAULT_MAX_TOKEN_ROWS}.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inspect_analysis_logs(
        sample_log=args.sample_log,
        candidate_log=args.candidate_log,
        output_md=args.output_md,
        max_cases_per_group=args.max_cases_per_group,
        max_token_rows=args.max_token_rows,
    )


def inspect_analysis_logs(
    *,
    sample_log: str | Path,
    candidate_log: str | Path,
    output_md: str | Path,
    max_cases_per_group: int = DEFAULT_MAX_CASES_PER_GROUP,
    max_token_rows: int = DEFAULT_MAX_TOKEN_ROWS,
) -> dict[str, Any]:
    """Read analysis logs, choose useful review cases, and write markdown."""
    if max_cases_per_group < 0:
        raise ValueError("max_cases_per_group must be non-negative.")
    if max_token_rows < 0:
        raise ValueError("max_token_rows must be non-negative.")

    samples = _read_jsonl(Path(sample_log))
    candidates_by_sample = _group_candidates_by_sample(_read_jsonl(Path(candidate_log)))
    enriched_samples = [_enrich_sample(sample, candidates_by_sample) for sample in samples]
    summary = _build_summary(enriched_samples)
    groups = _select_groups(enriched_samples, max_cases_per_group=max_cases_per_group)

    markdown = _render_markdown(
        samples=enriched_samples,
        candidates_by_sample=candidates_by_sample,
        summary=summary,
        groups=groups,
        max_token_rows=max_token_rows,
    )

    output_path = Path(output_md)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    return {"summary": summary, "groups": groups, "output_md": str(output_path)}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"JSONL log not found: {path}")

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                item = json.loads(stripped)
            except json.JSONDecodeError:
                rows.append({"_parse_error": f"invalid JSON on line {line_number}"})
                continue
            if isinstance(item, dict):
                rows.append(item)
    return rows


def _group_candidates_by_sample(candidates: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for candidate in candidates:
        sample_idx = _sample_idx(candidate)
        if sample_idx is None:
            continue
        grouped.setdefault(sample_idx, []).append(candidate)
    for items in grouped.values():
        items.sort(key=lambda item: _finite_number(item.get("candidate_idx"), default=10**9))
    return grouped


def _enrich_sample(
    sample: dict[str, Any],
    candidates_by_sample: dict[int, list[dict[str, Any]]],
) -> dict[str, Any]:
    enriched = dict(sample)
    sample_idx = _sample_idx(sample)
    candidates = candidates_by_sample.get(sample_idx, []) if sample_idx is not None else []

    vanilla_top2, vanilla_margin = _top_candidates_and_margin(
        sample=enriched,
        candidates=candidates,
        side="vanilla",
    )
    dola_top2, dola_margin = _top_candidates_and_margin(
        sample=enriched,
        candidates=candidates,
        side="dola",
    )

    enriched["vanilla_margin"] = vanilla_margin
    enriched["dola_margin"] = dola_margin
    enriched["vanilla_top2_candidates"] = vanilla_top2
    enriched["dola_top2_candidates"] = dola_top2
    enriched["both_wrong_and_dola_more_confident"] = (
        (not _as_bool(enriched.get("vanilla_correct")))
        and (not _as_bool(enriched.get("dola_correct")))
        and dola_margin > vanilla_margin
    )
    return enriched


def _top_candidates_and_margin(
    *,
    sample: dict[str, Any],
    candidates: list[dict[str, Any]],
    side: str,
) -> tuple[list[dict[str, Any]], float]:
    ranked = _rank_candidates(candidates, side=side)
    sample_margin = _sample_margin(sample, side=side)
    if sample_margin is not None:
        margin = sample_margin
    elif len(ranked) >= 2:
        margin = ranked[0]["score"] - ranked[1]["score"]
    else:
        margin = 0.0
    return ranked[:2], float(margin)


def _sample_margin(sample: dict[str, Any], *, side: str) -> float | None:
    for key in _MARGIN_KEYS[side]:
        value = sample.get(key)
        if _is_number(value):
            return float(value)
    return None


def _rank_candidates(candidates: list[dict[str, Any]], *, side: str) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for candidate in candidates:
        score = _candidate_score(candidate, side=side)
        if score is None:
            continue
        ranked.append(
            {
                "candidate_idx": candidate.get("candidate_idx"),
                "candidate_text": str(candidate.get("candidate_text", "[missing candidate]")),
                "is_true_candidate": _as_bool(candidate.get("is_true_candidate")),
                "score": float(score),
                "record": candidate,
            }
        )
    ranked.sort(
        key=lambda item: (
            -float(item["score"]),
            _finite_number(item.get("candidate_idx"), default=10**9),
        )
    )
    return ranked


def _candidate_score(candidate: dict[str, Any], *, side: str) -> float | None:
    for field in _SCORE_FIELDS[side]:
        value = candidate.get(field)
        if _is_number(value):
            return float(value)
    rank_field = f"{side}_rank"
    if _is_number(candidate.get(rank_field)):
        return -float(candidate[rank_field])
    return None


def _build_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    vanilla_margins = [float(sample["vanilla_margin"]) for sample in samples if _is_number(sample.get("vanilla_margin"))]
    dola_margins = [float(sample["dola_margin"]) for sample in samples if _is_number(sample.get("dola_margin"))]
    return {
        "total_samples": len(samples),
        "changed_by_dola": sum(1 for sample in samples if _as_bool(sample.get("changed_by_dola"))),
        "improved_by_dola": sum(1 for sample in samples if _as_bool(sample.get("improved_by_dola"))),
        "worsened_by_dola": sum(1 for sample in samples if _as_bool(sample.get("worsened_by_dola"))),
        "both_wrong_and_dola_more_confident": sum(
            1 for sample in samples if _as_bool(sample.get("both_wrong_and_dola_more_confident"))
        ),
        "avg_vanilla_margin": _mean(vanilla_margins),
        "avg_dola_margin": _mean(dola_margins),
    }


def _select_groups(
    samples: list[dict[str, Any]],
    *,
    max_cases_per_group: int,
) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for group_id, _title, predicate in _GROUPS:
        matching = [sample for sample in samples if predicate(sample)]
        matching.sort(key=_case_sort_key(group_id))
        groups[group_id] = matching[:max_cases_per_group]
    return groups


def _case_sort_key(group_id: str) -> Any:
    if group_id == "A":
        return lambda sample: (
            -(float(sample.get("dola_margin", 0.0)) - float(sample.get("vanilla_margin", 0.0))),
            _finite_number(sample.get("sample_idx"), default=10**9),
        )
    if group_id == "B":
        return lambda sample: (
            -(float(sample.get("vanilla_margin", 0.0)) - float(sample.get("dola_margin", 0.0))),
            _finite_number(sample.get("sample_idx"), default=10**9),
        )
    return lambda sample: (
        -(float(sample.get("dola_margin", 0.0)) - float(sample.get("vanilla_margin", 0.0))),
        _finite_number(sample.get("sample_idx"), default=10**9),
    )


def _render_markdown(
    *,
    samples: list[dict[str, Any]],
    candidates_by_sample: dict[int, list[dict[str, Any]]],
    summary: dict[str, Any],
    groups: dict[str, list[dict[str, Any]]],
    max_token_rows: int,
) -> str:
    del samples
    lines = [
        "# TruthfulQA MC Manual Review Cases",
        "",
        "## Summary",
        "",
        f"- Total samples: {summary['total_samples']}",
        f"- changed_by_dola: {summary['changed_by_dola']}",
        f"- improved_by_dola: {summary['improved_by_dola']}",
        f"- worsened_by_dola: {summary['worsened_by_dola']}",
        f"- both wrong but DoLa more confident: {summary['both_wrong_and_dola_more_confident']}",
        f"- Average vanilla top1-top2 margin: {_format_float(summary['avg_vanilla_margin'])}",
        f"- Average DoLa top1-top2 margin: {_format_float(summary['avg_dola_margin'])}",
        "",
    ]

    for group_id, title, _predicate in _GROUPS:
        group_samples = groups.get(group_id, [])
        lines.extend([f"## Group {group_id}: {title}", "", f"Selected cases: {len(group_samples)}", ""])
        if not group_samples:
            lines.extend(["No cases selected.", ""])
            continue
        for sample in group_samples:
            sample_idx = _sample_idx(sample)
            candidates = candidates_by_sample.get(sample_idx, []) if sample_idx is not None else []
            lines.extend(_render_sample_block(sample, candidates, max_token_rows=max_token_rows))
    return "\n".join(lines).rstrip() + "\n"


def _render_sample_block(
    sample: dict[str, Any],
    candidates: list[dict[str, Any]],
    *,
    max_token_rows: int,
) -> list[str]:
    sample_idx = _sample_idx(sample)
    lines = [
        f"### Sample {sample_idx if sample_idx is not None else '[missing]'}",
        "",
        f"- Question: {_inline(sample.get('question', '[missing question]'))}",
        f"- true_answers: {_format_list(sample.get('true_answers'))}",
        f"- false_answers: {_format_list(sample.get('false_answers'))}",
        f"- vanilla_pred: {_inline(sample.get('vanilla_pred', '[missing]'))}",
        f"- dola_pred: {_inline(sample.get('dola_pred', '[missing]'))}",
        f"- vanilla_correct: {_as_bool(sample.get('vanilla_correct'))}",
        f"- dola_correct: {_as_bool(sample.get('dola_correct'))}",
        f"- changed_by_dola / improved_by_dola / worsened_by_dola: "
        f"{_as_bool(sample.get('changed_by_dola'))} / "
        f"{_as_bool(sample.get('improved_by_dola'))} / "
        f"{_as_bool(sample.get('worsened_by_dola'))}",
        f"- vanilla margin: {_format_float(sample.get('vanilla_margin'))}",
        f"- DoLa margin: {_format_float(sample.get('dola_margin'))}",
        "",
        "Vanilla top-2 candidates:",
        *_format_top_candidates(sample.get("vanilla_top2_candidates")),
        "",
        "DoLa top-2 candidates:",
        *_format_top_candidates(sample.get("dola_top2_candidates")),
        "",
    ]

    selected_candidates = _selected_candidate_records(sample, candidates)
    if not selected_candidates:
        lines.extend(["Token-level comparison: no candidate records available.", ""])
        return lines

    for candidate in selected_candidates:
        lines.extend(
            _render_token_table(
                candidate,
                max_token_rows=max_token_rows,
            )
        )
    return lines


def _format_top_candidates(items: Any) -> list[str]:
    if not items:
        return ["- [missing]"]
    lines: list[str] = []
    for rank, item in enumerate(list(items)[:2], start=1):
        marker = "true" if _as_bool(item.get("is_true_candidate")) else "false"
        lines.append(
            f"- {rank}. idx={item.get('candidate_idx', '[missing]')} "
            f"score={_format_float(item.get('score'))} ({marker}) "
            f"{_inline(item.get('candidate_text', '[missing]'))}"
        )
    return lines


def _selected_candidate_records(sample: dict[str, Any], candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    wanted_texts = [sample.get("vanilla_pred"), sample.get("dola_pred")]
    selected: list[dict[str, Any]] = []
    seen_indices: set[Any] = set()
    for wanted_text in wanted_texts:
        if wanted_text is None:
            continue
        match = _find_candidate_by_text(candidates, str(wanted_text))
        if match is None:
            continue
        key = match.get("candidate_idx", id(match))
        if key in seen_indices:
            continue
        seen_indices.add(key)
        selected.append(match)
    if selected:
        return selected
    return candidates[:1]


def _find_candidate_by_text(candidates: list[dict[str, Any]], text: str) -> dict[str, Any] | None:
    for candidate in candidates:
        if str(candidate.get("candidate_text", "")) == text:
            return candidate
    stripped = text.strip()
    for candidate in candidates:
        if str(candidate.get("candidate_text", "")).strip() == stripped:
            return candidate
    return None


def _render_token_table(candidate: dict[str, Any], *, max_token_rows: int) -> list[str]:
    candidate_idx = candidate.get("candidate_idx", "[missing]")
    candidate_text = _inline(candidate.get("candidate_text", "[missing candidate]"))
    lines = [
        f"Token-level comparison for candidate {candidate_idx}: {candidate_text}",
        "",
        "| token index | token text | vanilla token score | DoLa mature score | DoLa premature score | DoLa contrast score |",
        "|---:|---|---:|---:|---:|---:|",
    ]

    token_rows = _token_rows(candidate)
    visible_rows = token_rows[:max_token_rows]
    for row in visible_rows:
        lines.append(
            "| "
            f"{row['index']} | {_table_cell(row['text'])} | {_format_float(row['vanilla'])} | "
            f"{_format_float(row['dola_final'])} | {_format_float(row['dola_premature'])} | "
            f"{_format_float(row['dola_contrast'])} |"
        )
    if len(token_rows) > len(visible_rows):
        remaining = len(token_rows) - len(visible_rows)
        lines.append(f"| ... | ... {remaining} more token rows omitted ... |  |  |  |  |")
    if not token_rows:
        lines.append("| [missing] | [missing token data] |  |  |  |  |")
    lines.extend([""])
    return lines


def _token_rows(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    token_texts = _as_list(candidate.get("token_texts"))
    token_ids = _as_list(candidate.get("token_ids"))
    vanilla_scores = _as_list(candidate.get("vanilla_token_logprobs"))
    final_scores = _as_list(candidate.get("dola_final_token_scores"))
    premature_scores = _as_list(candidate.get("dola_premature_token_scores"))
    contrast_scores = _as_list(candidate.get("dola_token_contrast_scores"))
    row_count = max(
        len(token_texts),
        len(token_ids),
        len(vanilla_scores),
        len(final_scores),
        len(premature_scores),
        len(contrast_scores),
    )
    rows: list[dict[str, Any]] = []
    for index in range(row_count):
        token_text = _list_get(token_texts, index)
        if token_text is None:
            token_id = _list_get(token_ids, index)
            token_text = f"[id:{token_id}]" if token_id is not None else "[missing]"
        rows.append(
            {
                "index": index,
                "text": token_text,
                "vanilla": _list_get(vanilla_scores, index),
                "dola_final": _list_get(final_scores, index),
                "dola_premature": _list_get(premature_scores, index),
                "dola_contrast": _list_get(contrast_scores, index),
            }
        )
    return rows


def _sample_idx(item: dict[str, Any]) -> int | None:
    value = item.get("sample_idx", item.get("sample_index"))
    if _is_number(value):
        return int(value)
    return None


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _is_number(value: Any) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _finite_number(value: Any, *, default: float) -> float:
    if not _is_number(value):
        return default
    return float(value)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _list_get(values: list[Any], index: int) -> Any:
    return values[index] if index < len(values) else None


def _format_list(value: Any, *, max_items: int = 8) -> str:
    values = _as_list(value)
    if not values:
        return "[]"
    rendered = [_inline(item) for item in values[:max_items]]
    if len(values) > max_items:
        rendered.append(f"... {len(values) - max_items} more")
    return "; ".join(rendered)


def _inline(value: Any, *, max_chars: int = 260) -> str:
    text = str(value).replace("\n", " ").strip()
    if len(text) > max_chars:
        return text[: max_chars - 3].rstrip() + "..."
    return text


def _table_cell(value: Any) -> str:
    return _inline(value, max_chars=80).replace("|", "\\|")


def _format_float(value: Any) -> str:
    if not _is_number(value):
        return "n/a"
    return f"{float(value):.4f}"


if __name__ == "__main__":
    main()
