"""Compare completed token-selective subset ablation runs."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_MD = PROJECT_ROOT / "outputs" / "analysis_logs" / "ablation_decision_summary_subset200.md"
TOP_CONTRIBUTION_TOKENS = 3

Q2_LABELS = ("fact-critical", "partial", "no", "uncertain")
Q3_LABELS = (
    "strengthens_correct_fact",
    "strengthens_smoother_or_more_common_expression",
    "mixed",
    "uncertain",
)
Q4_LABELS = ("yes", "no", "not_applicable", "uncertain")
Q5_LABELS = (
    "confidence_sharpening",
    "token_frequency_bias",
    "middle_layer_shortcut",
    "other",
    "evidence_insufficient",
    "not_applicable_for_failure_type",
)
TIER_LABELS = ("strong", "medium", "unselected", "evidence_insufficient")
REASON_LABELS = (
    "number",
    "date_word",
    "relation_word",
    "capitalized_lexical",
    "lowercase_medium",
    "adjacent_support",
    "continuation_inheritance",
    "other",
    "evidence_insufficient",
)


@dataclass(slots=True)
class RunPaths:
    label: str
    main_dir: Path | None
    log_dir: Path | None


@dataclass(slots=True)
class RunSummary:
    label: str
    main_dir: Path | None
    log_dir: Path | None
    compare_summary: dict[str, Any]
    sample_records: list[dict[str, Any]]
    candidate_records: list[dict[str, Any]]
    review_records: list[dict[str, Any]]
    sample_counts: dict[str, int | None]
    avg_dola_margin: float | None
    q2_counts: Counter[str]
    q3_counts: Counter[str]
    q4_counts: Counter[str]
    q5_counts: Counter[str]
    tier_contributions: dict[str, Counter[str]]
    reason_contributions: dict[str, Counter[str]]
    evidence_notes: list[str]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare completed token-selective subset ablation runs."
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="Run names or directories. Each run may point to outputs/<run> or outputs/analysis_logs/<run>.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=DEFAULT_OUTPUT_MD,
        help="Markdown report path.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    summaries = [_load_run_summary(_resolve_run_paths(run_spec)) for run_spec in args.runs]
    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(_render_report(summaries), encoding="utf-8")
    print(json.dumps({"report": str(output_md), "num_runs": len(summaries)}, indent=2))


def _resolve_run_paths(run_spec: str) -> RunPaths:
    raw_path = Path(run_spec)
    path = raw_path if raw_path.is_absolute() else PROJECT_ROOT / raw_path
    label = raw_path.name

    main_dir: Path | None = None
    log_dir: Path | None = None

    if path.exists():
        label = path.name
        if (path / "sample_level.jsonl").exists() or (path / "candidate_level.jsonl").exists():
            log_dir = path
        if (path / "compare_summary.json").exists():
            main_dir = path

    if log_dir is None:
        candidate_log_dir = PROJECT_ROOT / "outputs" / "analysis_logs" / label
        if candidate_log_dir.exists():
            log_dir = candidate_log_dir
    if main_dir is None:
        candidate_main_dir = PROJECT_ROOT / "outputs" / label
        if candidate_main_dir.exists():
            main_dir = candidate_main_dir
    if main_dir is None and log_dir is not None:
        candidate_main_dir = PROJECT_ROOT / "outputs" / log_dir.name
        if candidate_main_dir.exists():
            main_dir = candidate_main_dir

    return RunPaths(label=label, main_dir=main_dir, log_dir=log_dir)


def _load_run_summary(paths: RunPaths) -> RunSummary:
    compare_summary = _read_json(paths.main_dir / "compare_summary.json") if paths.main_dir else {}
    sample_records = _read_jsonl(paths.log_dir / "sample_level.jsonl") if paths.log_dir else []
    candidate_records = _read_jsonl(paths.log_dir / "candidate_level.jsonl") if paths.log_dir else []
    review_records = (
        _read_jsonl(paths.log_dir / "per_sample_five_question_review_200.jsonl")
        if paths.log_dir
        else []
    )
    evidence_notes: list[str] = []
    if not compare_summary:
        evidence_notes.append("compare_summary.json missing")
    if not sample_records:
        evidence_notes.append("sample_level.jsonl missing")
    if not candidate_records:
        evidence_notes.append("candidate_level.jsonl missing")
    if not review_records:
        evidence_notes.append("per_sample_five_question_review_200.jsonl missing")

    sample_counts = _summarize_sample_flags(sample_records)
    avg_dola_margin = _average_dola_margin(candidate_records)
    q2_counts = Counter(str(row.get("q2_label", "evidence_insufficient")) for row in review_records)
    q3_counts = Counter(str(row.get("q3_label", "evidence_insufficient")) for row in review_records)
    q4_counts = Counter(str(row.get("q4_label", "evidence_insufficient")) for row in review_records)
    q5_counts = Counter(str(row.get("q5_label", "evidence_insufficient")) for row in review_records)
    tier_contributions, reason_contributions = _summarize_contributions(
        sample_records=sample_records,
        candidate_records=candidate_records,
    )

    return RunSummary(
        label=paths.label,
        main_dir=paths.main_dir,
        log_dir=paths.log_dir,
        compare_summary=compare_summary,
        sample_records=sample_records,
        candidate_records=candidate_records,
        review_records=review_records,
        sample_counts=sample_counts,
        avg_dola_margin=avg_dola_margin,
        q2_counts=q2_counts,
        q3_counts=q3_counts,
        q4_counts=q4_counts,
        q5_counts=q5_counts,
        tier_contributions=tier_contributions,
        reason_contributions=reason_contributions,
        evidence_notes=evidence_notes,
    )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def _summarize_sample_flags(sample_records: list[dict[str, Any]]) -> dict[str, int | None]:
    if not sample_records:
        return {"total": None, "changed": None, "improved": None, "worsened": None}
    return {
        "total": len(sample_records),
        "changed": sum(1 for row in sample_records if bool(row.get("changed_by_dola", False))),
        "improved": sum(1 for row in sample_records if bool(row.get("improved_by_dola", False))),
        "worsened": sum(1 for row in sample_records if bool(row.get("worsened_by_dola", False))),
    }


def _average_dola_margin(candidate_records: list[dict[str, Any]]) -> float | None:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_records:
        sample_idx = _safe_int(row.get("sample_idx"))
        if sample_idx is not None:
            grouped[sample_idx].append(row)

    margins: list[float] = []
    for rows in grouped.values():
        scores = [_score_value(row) for row in rows]
        scores = [score for score in scores if score is not None]
        if len(scores) < 2:
            continue
        ranked = sorted(scores, reverse=True)
        margins.append(ranked[0] - ranked[1])
    if not margins:
        return None
    return sum(margins) / len(margins)


def _score_value(row: dict[str, Any]) -> float | None:
    for key in ("dola_total_score", "dola_avg_score"):
        if key in row:
            return _safe_float(row.get(key))
    return None


def _summarize_contributions(
    *,
    sample_records: list[dict[str, Any]],
    candidate_records: list[dict[str, Any]],
) -> tuple[dict[str, Counter[str]], dict[str, Counter[str]]]:
    tier_counts = {"improved": Counter(), "worsened": Counter()}
    reason_counts = {"improved": Counter(), "worsened": Counter()}
    samples_by_idx = {
        int(row["sample_idx"]): row
        for row in sample_records
        if row.get("sample_idx") is not None
    }
    candidates_by_sample: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_records:
        sample_idx = _safe_int(row.get("sample_idx"))
        if sample_idx is not None:
            candidates_by_sample[sample_idx].append(row)

    for sample_idx, sample in samples_by_idx.items():
        bucket = None
        if bool(sample.get("improved_by_dola", False)):
            bucket = "improved"
        elif bool(sample.get("worsened_by_dola", False)):
            bucket = "worsened"
        if bucket is None:
            continue

        top_candidate = _top_dola_candidate(candidates_by_sample.get(sample_idx, []))
        if top_candidate is None:
            tier_counts[bucket]["evidence_insufficient"] += 1
            reason_counts[bucket]["evidence_insufficient"] += 1
            continue

        top_contributions = _positive_token_contributions(top_candidate)
        if not top_contributions:
            tier_counts[bucket]["evidence_insufficient"] += 1
            reason_counts[bucket]["evidence_insufficient"] += 1
            continue
        for contribution in top_contributions[:TOP_CONTRIBUTION_TOKENS]:
            tier_counts[bucket][contribution["tier"]] += 1
            reason_counts[bucket][_normalize_reason(str(contribution["reason"]))] += 1

    return tier_counts, reason_counts


def _top_dola_candidate(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    for row in rows:
        if _safe_int(row.get("dola_rank")) == 1:
            return row
    ranked = [(row, _score_value(row)) for row in rows]
    ranked = [(row, score) for row, score in ranked if score is not None]
    if not ranked:
        return None
    return max(ranked, key=lambda item: float(item[1]))[0]


def _positive_token_contributions(row: dict[str, Any]) -> list[dict[str, Any]]:
    contrast_scores = _list_value(row, ("token_contrast_score", "dola_token_contrast_scores"))
    if not contrast_scores:
        return []

    weights = _list_value(row, ("token_contrast_weight",))
    sources = _list_value(row, ("token_effective_score_source",))
    tiers = _list_value(row, ("token_selection_tier",))
    reasons = _list_value(row, ("token_selected_reason",))

    if not weights:
        weights = _infer_weights_from_sources(sources, len(contrast_scores))
    if not tiers:
        tiers = _infer_tiers_from_sources(sources, len(contrast_scores))
    if not reasons:
        reasons = [""] * len(contrast_scores)
    if not weights or not tiers:
        return []

    contributions: list[dict[str, Any]] = []
    length = min(len(contrast_scores), len(weights), len(tiers), len(reasons))
    for token_index in range(length):
        contrast = _safe_float(contrast_scores[token_index])
        weight = _safe_float(weights[token_index])
        if contrast is None or weight is None:
            continue
        contribution = contrast * weight
        if contribution <= 0:
            continue
        contributions.append(
            {
                "token_index": token_index,
                "contribution": contribution,
                "tier": _normalize_tier(str(tiers[token_index])),
                "reason": str(reasons[token_index]),
            }
        )
    return sorted(contributions, key=lambda item: float(item["contribution"]), reverse=True)


def _list_value(row: dict[str, Any], keys: tuple[str, ...]) -> list[Any]:
    for key in keys:
        value = row.get(key)
        if isinstance(value, list):
            return value
    return []


def _infer_weights_from_sources(sources: list[Any], length: int) -> list[float]:
    if not sources:
        return []
    weights: list[float] = []
    for source in sources[:length]:
        source_text = str(source)
        if source_text == "contrast":
            weights.append(1.0)
        elif source_text == "vanilla_final":
            weights.append(0.0)
        else:
            return []
    return weights


def _infer_tiers_from_sources(sources: list[Any], length: int) -> list[str]:
    if not sources:
        return []
    tiers: list[str] = []
    for source in sources[:length]:
        source_text = str(source)
        if source_text == "contrast":
            tiers.append("strong")
        elif source_text == "vanilla_final":
            tiers.append("unselected")
        else:
            return []
    return tiers


def _normalize_tier(tier: str) -> str:
    tier = tier.strip().lower()
    if tier in {"strong", "medium", "unselected"}:
        return tier
    return "evidence_insufficient"


def _normalize_reason(reason: str) -> str:
    normalized = reason.strip().lower()
    if normalized in {"number", "date_word", "relation_word", "capitalized_lexical"}:
        return normalized
    if normalized == "medium_lexical":
        return "lowercase_medium"
    if normalized == "medium_adjacent_support":
        return "adjacent_support"
    if normalized == "strong_continuation":
        return "continuation_inheritance"
    return "other"


def _render_report(summaries: list[RunSummary]) -> str:
    lines: list[str] = [
        "# Token-Selective Ablation Decision Summary Subset200",
        "",
        "## Runs",
        "",
        "| run | main_dir | log_dir | evidence notes |",
        "|---|---|---|---|",
    ]
    for summary in summaries:
        lines.append(
            "| {run} | {main_dir} | {log_dir} | {notes} |".format(
                run=summary.label,
                main_dir=_display_path(summary.main_dir),
                log_dir=_display_path(summary.log_dir),
                notes=", ".join(summary.evidence_notes) if summary.evidence_notes else "ok",
            )
        )

    lines.extend(["", "## Quantitative Comparison", ""])
    lines.extend(_main_metrics_table(summaries))
    lines.extend(["", "## Q2 Label Distribution", ""])
    lines.extend(_counter_table(summaries, "q2_counts", Q2_LABELS))
    lines.extend(["", "## Q3 Label Distribution", ""])
    lines.extend(_counter_table(summaries, "q3_counts", Q3_LABELS))
    lines.extend(["", "## Q4 Label Distribution", ""])
    lines.extend(_counter_table(summaries, "q4_counts", Q4_LABELS))
    lines.extend(["", "## Q5 Failure Type Distribution", ""])
    lines.extend(_counter_table(summaries, "q5_counts", Q5_LABELS))
    lines.extend(["", "## Tier Contribution Summary", ""])
    lines.extend(_contribution_table(summaries, "tier_contributions", TIER_LABELS))
    lines.extend(["", "## Reason Contribution Summary", ""])
    lines.extend(_contribution_table(summaries, "reason_contributions", REASON_LABELS))
    lines.extend(["", *decision_guide_lines(), ""])
    return "\n".join(lines)


def _main_metrics_table(summaries: list[RunSummary]) -> list[str]:
    lines = [
        "| run | MC1 | MC2 | MC3 | changed | improved | worsened | Q4 yes | avg DoLa top1-top2 margin |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        lines.append(
            "| {run} | {mc1} | {mc2} | {mc3} | {changed} | {improved} | {worsened} | {q4_yes} | {margin} |".format(
                run=summary.label,
                mc1=_format_float(summary.compare_summary.get("dola_avg_mc1")),
                mc2=_format_float(summary.compare_summary.get("dola_avg_mc2")),
                mc3=_format_float(summary.compare_summary.get("dola_avg_mc3")),
                changed=_format_int(summary.sample_counts.get("changed")),
                improved=_format_int(summary.sample_counts.get("improved")),
                worsened=_format_int(summary.sample_counts.get("worsened")),
                q4_yes=_format_int(summary.q4_counts.get("yes")),
                margin=_format_float(summary.avg_dola_margin),
            )
        )
    return lines


def _counter_table(
    summaries: list[RunSummary],
    attr_name: str,
    labels: tuple[str, ...],
) -> list[str]:
    lines = [
        "| run | " + " | ".join(labels) + " |",
        "|---" + "|---:" * len(labels) + "|",
    ]
    for summary in summaries:
        counter: Counter[str] = getattr(summary, attr_name)
        values = " | ".join(str(counter.get(label, 0)) for label in labels)
        lines.append(f"| {summary.label} | {values} |")
    return lines


def _contribution_table(
    summaries: list[RunSummary],
    attr_name: str,
    labels: tuple[str, ...],
) -> list[str]:
    lines = [
        "| run | sample bucket | " + " | ".join(labels) + " |",
        "|---|---" + "|---:" * len(labels) + "|",
    ]
    for summary in summaries:
        bucket_counters: dict[str, Counter[str]] = getattr(summary, attr_name)
        for bucket in ("improved", "worsened"):
            counter = bucket_counters.get(bucket, Counter())
            values = " | ".join(str(counter.get(label, 0)) for label in labels)
            lines.append(f"| {summary.label} | {bucket} | {values} |")
    return lines


def decision_guide_lines() -> list[str]:
    return [
        "## Decision guide for v3",
        "",
        "- If low-unselected / no-unselected clearly lowers Q4 yes while improved_by_dola drops only modestly, v3 should first reduce unselected contrast.",
        "- If strong-only-soft or low-medium clearly improves the tradeoff, v3 should first tighten the medium tier rather than changing strong rules.",
        "- If no-capitalized clearly improves the tradeoff, v3 should focus on repairing the capitalized lexical heuristic.",
        "- If factual-core-only is the most stable but loses too much improvement, v3 should add a small number of high-value support tokens on top of factual core rather than broadly reopening medium.",
        "- If no ablation improves on v2, selector-only patching is probably insufficient and the next direction should be bias-aware.",
    ]


def _display_path(path: Path | None) -> str:
    if path is None:
        return "missing"
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _format_float(value: Any) -> str:
    numeric = _safe_float(value)
    if numeric is None:
        return "n/a"
    return f"{numeric:.4f}"


def _format_int(value: Any) -> str:
    numeric = _safe_int(value)
    if numeric is None:
        return "n/a"
    return str(numeric)


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    main()
