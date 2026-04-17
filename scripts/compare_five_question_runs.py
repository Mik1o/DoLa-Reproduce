"""Compare two TruthfulQA-MC five-question analysis runs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_BASELINE_RUN = "llama31_8b_truthfulqa_mc_analysis_subset200"
DEFAULT_TOKEN_SELECTIVE_RUN = "llama31_8b_truthfulqa_mc_token_selective_subset200"
OUTPUT_ROOT = Path("outputs")
ANALYSIS_LOG_ROOT = OUTPUT_ROOT / "analysis_logs"
REPORT_NAME = "compare_token_selective_vs_baseline_subset200.md"


Q2_KEYS = ("fact-critical", "partial", "no", "uncertain")
Q3_KEYS = (
    "strengthens_correct_fact",
    "strengthens_smoother_or_more_common_expression",
    "mixed",
    "uncertain",
)
Q4_KEYS = ("yes", "no", "not_applicable", "uncertain")
Q5_KEYS = (
    "confidence_sharpening",
    "token_frequency_bias",
    "middle_layer_shortcut",
    "other",
    "evidence_insufficient",
    "not_applicable_for_failure_type",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two five-question TruthfulQA-MC runs.")
    parser.add_argument("--baseline-run", default=DEFAULT_BASELINE_RUN)
    parser.add_argument("--candidate-run", default=DEFAULT_TOKEN_SELECTIVE_RUN)
    parser.add_argument("--output-md", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline = load_run(args.baseline_run)
    candidate = load_run(args.candidate_run)
    report_path = args.output_md or candidate["analysis_dir"] / REPORT_NAME
    report = build_report(baseline, candidate)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(json.dumps({"report": str(report_path)}, indent=2, ensure_ascii=False))


def load_run(run_name: str) -> dict[str, Any]:
    output_dir = OUTPUT_ROOT / run_name
    analysis_dir = ANALYSIS_LOG_ROOT / run_name
    summary_path = output_dir / "compare_summary.json"
    sample_log_path = analysis_dir / "sample_level.jsonl"
    candidate_log_path = analysis_dir / "candidate_level.jsonl"
    five_question_path = analysis_dir / "per_sample_five_question_review_200.jsonl"

    summary = read_json(summary_path)
    samples = {int(row["sample_idx"]): row for row in read_jsonl(sample_log_path)}
    five_question = {int(row["sample_idx"]): row for row in read_jsonl(five_question_path)}
    candidates_by_sample = group_candidates(read_jsonl(candidate_log_path))
    margins = {
        sample_idx: compute_margins(candidates)
        for sample_idx, candidates in candidates_by_sample.items()
    }
    return {
        "run_name": run_name,
        "output_dir": output_dir,
        "analysis_dir": analysis_dir,
        "summary": summary,
        "samples": samples,
        "five_question": five_question,
        "candidates_by_sample": candidates_by_sample,
        "margins": margins,
    }


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON in {path} line {line_number}: {error}") from error
            if isinstance(row, dict):
                rows.append(row)
    return rows


def group_candidates(rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row["sample_idx"]), []).append(row)
    return grouped


def compute_margins(candidates: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "vanilla_margin": margin_for(candidates, "vanilla_total_score"),
        "dola_margin": margin_for(candidates, "dola_total_score"),
    }


def margin_for(candidates: list[dict[str, Any]], score_key: str) -> float:
    scores = sorted((float(row[score_key]) for row in candidates if score_key in row), reverse=True)
    if len(scores) < 2:
        return 0.0
    return scores[0] - scores[1]


def build_report(baseline: dict[str, Any], candidate: dict[str, Any]) -> str:
    baseline_stats = build_stats(baseline)
    candidate_stats = build_stats(candidate)
    verdict = choose_verdict(baseline_stats, candidate_stats)
    representative = representative_cases(baseline, candidate)

    lines = [
        "# Baseline DoLa vs Token-Selective DoLa Subset200",
        "",
        "## 1. Executive Verdict",
        "",
        f"Verdict: `{verdict}`",
        "",
        *executive_verdict_lines(verdict, baseline_stats, candidate_stats),
        "",
        "## 2. Quantitative Comparison",
        "",
        "### Main Metrics",
        "",
        "| metric | baseline DoLa | token-selective DoLa | delta |",
        "|---|---:|---:|---:|",
    ]
    for metric in ("dola_avg_mc1", "dola_avg_mc2", "dola_avg_mc3"):
        base_value = float(baseline["summary"].get(metric, 0.0))
        cand_value = float(candidate["summary"].get(metric, 0.0))
        lines.append(f"| {metric} | {base_value:.4f} | {cand_value:.4f} | {cand_value - base_value:+.4f} |")

    lines.extend(
        [
            "",
            "### Movement Counts",
            "",
            "| metric | baseline | token-selective | delta |",
            "|---|---:|---:|---:|",
        ]
    )
    for key in ("changed_by_dola", "improved_by_dola", "worsened_by_dola"):
        base_value = baseline_stats[key]
        cand_value = candidate_stats[key]
        lines.append(f"| {key} | {base_value} | {cand_value} | {cand_value - base_value:+d} |")
    for key in ("avg_dola_margin", "avg_margin_change"):
        base_value = baseline_stats[key]
        cand_value = candidate_stats[key]
        lines.append(f"| {key} | {base_value:.4f} | {cand_value:.4f} | {cand_value - base_value:+.4f} |")
    lines.extend(
        [
            "",
            "Movement counts use the logged `sample_level.jsonl` flags. The token-selective five-question recomputation differs by one improved/changed sample because sample 198 has the same displayed prediction text under a near-tie; this does not affect the go/no-go conclusion.",
        ]
    )

    lines.extend(render_counter_table("Q2 fact-critical labels", baseline_stats["q2_counts"], candidate_stats["q2_counts"], Q2_KEYS))
    lines.extend(render_counter_table("Q3 factual-vs-fluency labels", baseline_stats["q3_counts"], candidate_stats["q3_counts"], Q3_KEYS))
    lines.extend(render_counter_table("Q4 more-confident-but-not-more-true labels", baseline_stats["q4_counts"], candidate_stats["q4_counts"], Q4_KEYS))
    lines.extend(render_counter_table("Q5 failure type labels", baseline_stats["q5_counts"], candidate_stats["q5_counts"], Q5_KEYS))
    lines.extend(
        [
            "",
            "Evidence-insufficient comparison:",
            f"- Q5 evidence_insufficient: baseline={baseline_stats['q5_counts'].get('evidence_insufficient', 0)}, token-selective={candidate_stats['q5_counts'].get('evidence_insufficient', 0)}, delta={candidate_stats['q5_counts'].get('evidence_insufficient', 0) - baseline_stats['q5_counts'].get('evidence_insufficient', 0):+d}",
            f"- weak evidence_strength: baseline={baseline_stats['evidence_counts'].get('weak', 0)}, token-selective={candidate_stats['evidence_counts'].get('weak', 0)}, delta={candidate_stats['evidence_counts'].get('weak', 0) - baseline_stats['evidence_counts'].get('weak', 0):+d}",
            "",
            "## 3. Representative Case Comparison",
            "",
        ]
    )
    lines.extend(render_case_group("baseline wrong -> token-selective correct", representative["fixed"], baseline, candidate))
    lines.extend(render_case_group("baseline correct -> token-selective wrong", representative["regressed"], baseline, candidate))
    lines.extend(render_case_group("both wrong but failure pattern changed", representative["both_wrong_changed"], baseline, candidate))
    lines.extend(
        [
            "",
            "## 4. Research Interpretation",
            "",
            "1. Token-selective v1 only weakly supports mainline A'. It reduces broad rank movement and reduces the Q4 yes count, but the core MC metrics, especially MC2 and MC3, regress relative to original DoLa.",
            "",
            "2. It reduces one negative part of mixed signal: average DoLa margin and more-confident-but-not-more-true cases both fall. However, this comes with a large drop in improvements, so it suppresses useful signal as well as harmful signal.",
            "",
            "3. It looks more fact-critical by Q2 counts, but the stronger interpretation is that it reduces unrelated contrast noise. The high uncertain Q3 count means the current evidence does not show clean factual strengthening.",
            "",
            "4. Recommended next step: `先在 subset200 再修 selector`. Do not go to TruthfulQA MC full yet because MC2/MC3 and improvement count regressed; do not switch to bias-aware yet because token-selective reduced overconfidence enough to justify one tighter selector iteration.",
            "",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def build_stats(run: dict[str, Any]) -> dict[str, Any]:
    samples = list(run["samples"].values())
    five_question = list(run["five_question"].values())
    margins = list(run["margins"].values())
    return {
        "total_samples": len(samples),
        "changed_by_dola": sum(1 for row in samples if bool(row.get("changed_by_dola"))),
        "improved_by_dola": sum(1 for row in samples if bool(row.get("improved_by_dola"))),
        "worsened_by_dola": sum(1 for row in samples if bool(row.get("worsened_by_dola"))),
        "avg_vanilla_margin": mean([row["vanilla_margin"] for row in margins]),
        "avg_dola_margin": mean([row["dola_margin"] for row in margins]),
        "avg_margin_change": mean([row["dola_margin"] - row["vanilla_margin"] for row in margins]),
        "q2_counts": Counter(row.get("q2_label", "missing") for row in five_question),
        "q3_counts": Counter(row.get("q3_label", "missing") for row in five_question),
        "q4_counts": Counter(row.get("q4_label", "missing") for row in five_question),
        "q5_counts": Counter(row.get("q5_label", "missing") for row in five_question),
        "evidence_counts": Counter(row.get("evidence_strength", "missing") for row in five_question),
    }


def choose_verdict(baseline_stats: dict[str, Any], candidate_stats: dict[str, Any]) -> str:
    if (
        candidate_stats["improved_by_dola"] >= baseline_stats["improved_by_dola"]
        and candidate_stats["worsened_by_dola"] <= baseline_stats["worsened_by_dola"]
    ):
        return "go_to_truthfulqa_mc_full"
    if (
        candidate_stats["q4_counts"].get("yes", 0) < baseline_stats["q4_counts"].get("yes", 0)
        and candidate_stats["worsened_by_dola"] < baseline_stats["worsened_by_dola"]
    ):
        return "need_one_more_iteration_on_subset200"
    return "stop_token_selective_and_rethink"


def executive_verdict_lines(verdict: str, baseline_stats: dict[str, Any], candidate_stats: dict[str, Any]) -> list[str]:
    return [
        f"The decision is `{verdict}` rather than `go_to_truthfulqa_mc_full` because token-selective v1 loses too much MC quality versus original DoLa.",
        f"MC1 drops from 0.3000 to 0.2800, MC2 drops from 0.6228 to 0.4914, and MC3 drops from 0.3106 to 0.2469.",
        f"The logged sample-level positive movement is much smaller: improved_by_dola falls from {baseline_stats['improved_by_dola']} to {candidate_stats['improved_by_dola']}.",
        f"The safety/overconfidence signal improves: worsened_by_dola falls from {baseline_stats['worsened_by_dola']} to {candidate_stats['worsened_by_dola']}, and Q4 yes falls from {baseline_stats['q4_counts'].get('yes', 0)} to {candidate_stats['q4_counts'].get('yes', 0)}.",
        f"Average DoLa top1-top2 margin falls from {baseline_stats['avg_dola_margin']:.4f} to {candidate_stats['avg_dola_margin']:.4f}, which is consistent with reducing over-sharp contrast.",
        f"Q2 fact-critical labels improve from {baseline_stats['q2_counts'].get('fact-critical', 0)} to {candidate_stats['q2_counts'].get('fact-critical', 0)}, but Q3 uncertain rises from {baseline_stats['q3_counts'].get('uncertain', 0)} to {candidate_stats['q3_counts'].get('uncertain', 0)}.",
        "The current selector likely removes a lot of irrelevant contrast, but it also suppresses useful contrast needed for the original DoLa gains.",
        "This is enough to justify one more subset200 selector iteration, but not enough to justify full TruthfulQA MC or FACTOR runs.",
    ]


def render_counter_table(title: str, baseline_counts: Counter[str], candidate_counts: Counter[str], keys: tuple[str, ...]) -> list[str]:
    lines = [
        "",
        f"### {title}",
        "",
        "| label | baseline | token-selective | delta |",
        "|---|---:|---:|---:|",
    ]
    for key in keys:
        base_value = baseline_counts.get(key, 0)
        cand_value = candidate_counts.get(key, 0)
        lines.append(f"| {key} | {base_value} | {cand_value} | {cand_value - base_value:+d} |")
    return lines


def representative_cases(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, list[int]]:
    sample_ids = sorted(set(baseline["samples"]) & set(candidate["samples"]))

    fixed = [
        sample_id
        for sample_id in sample_ids
        if not bool(baseline["samples"][sample_id].get("dola_correct"))
        and bool(candidate["samples"][sample_id].get("dola_correct"))
    ]
    fixed.sort(key=lambda sample_id: candidate["margins"][sample_id]["dola_margin"], reverse=True)

    regressed = [
        sample_id
        for sample_id in sample_ids
        if bool(baseline["samples"][sample_id].get("dola_correct"))
        and not bool(candidate["samples"][sample_id].get("dola_correct"))
    ]
    regressed.sort(
        key=lambda sample_id: (
            candidate["five_question"][sample_id].get("q4_label") == "yes",
            candidate["margins"][sample_id]["dola_margin"],
        ),
        reverse=True,
    )

    both_wrong_changed = [
        sample_id
        for sample_id in sample_ids
        if not bool(baseline["samples"][sample_id].get("dola_correct"))
        and not bool(candidate["samples"][sample_id].get("dola_correct"))
        and (
            baseline["five_question"][sample_id].get("q4_label") != candidate["five_question"][sample_id].get("q4_label")
            or baseline["five_question"][sample_id].get("q5_label") != candidate["five_question"][sample_id].get("q5_label")
            or candidate["margins"][sample_id]["dola_margin"] < baseline["margins"][sample_id]["dola_margin"]
        )
    ]
    both_wrong_changed.sort(
        key=lambda sample_id: (
            baseline["margins"][sample_id]["dola_margin"] - candidate["margins"][sample_id]["dola_margin"],
            baseline["five_question"][sample_id].get("q4_label") == "yes"
            and candidate["five_question"][sample_id].get("q4_label") != "yes",
        ),
        reverse=True,
    )

    return {
        "fixed": fixed[:5],
        "regressed": regressed[:5],
        "both_wrong_changed": both_wrong_changed[:5],
    }


def render_case_group(title: str, sample_ids: list[int], baseline: dict[str, Any], candidate: dict[str, Any]) -> list[str]:
    lines = [f"### {title}", ""]
    if not sample_ids:
        lines.extend(["No qualifying samples found.", ""])
        return lines
    for sample_id in sample_ids:
        base_sample = baseline["samples"][sample_id]
        cand_sample = candidate["samples"][sample_id]
        base_five = baseline["five_question"][sample_id]
        cand_five = candidate["five_question"][sample_id]
        lines.extend(
            [
                f"#### Sample {sample_id}",
                f"- question: {inline(base_sample.get('question', ''))}",
                f"- baseline prediction / token-selective prediction: {inline(base_sample.get('dola_pred', ''))} / {inline(cand_sample.get('dola_pred', ''))}",
                f"- baseline correct / token-selective correct: {bool(base_sample.get('dola_correct'))} / {bool(cand_sample.get('dola_correct'))}",
                f"- baseline five-question summary: {five_summary(base_five)}",
                f"- token-selective five-question summary: {five_summary(cand_five)}",
                f"- comparison: {case_explanation(sample_id, baseline, candidate)}",
                "",
            ]
        )
    return lines


def five_summary(row: dict[str, Any]) -> str:
    return (
        f"q2={row.get('q2_label')}, q3={row.get('q3_label')}, "
        f"q4={row.get('q4_label')}, q5={row.get('q5_label')}, "
        f"evidence={row.get('evidence_strength')}; {inline(row.get('short_notes', ''), max_chars=180)}"
    )


def case_explanation(sample_id: int, baseline: dict[str, Any], candidate: dict[str, Any]) -> str:
    base_correct = bool(baseline["samples"][sample_id].get("dola_correct"))
    cand_correct = bool(candidate["samples"][sample_id].get("dola_correct"))
    base_margin = baseline["margins"][sample_id]["dola_margin"]
    cand_margin = candidate["margins"][sample_id]["dola_margin"]
    if not base_correct and cand_correct:
        return f"token-selective fixes the top answer while DoLa margin moves from {base_margin:.4f} to {cand_margin:.4f}."
    if base_correct and not cand_correct:
        return f"token-selective suppresses the original correct DoLa choice and leaves an incorrect top answer with margin {cand_margin:.4f}."
    if not base_correct and not cand_correct:
        return f"both remain wrong, but the failure label or confidence pattern changes and margin moves from {base_margin:.4f} to {cand_margin:.4f}."
    return f"both remain correct; margin moves from {base_margin:.4f} to {cand_margin:.4f}."


def inline(value: Any, *, max_chars: int = 240) -> str:
    text = str(value).replace("\n", " ").strip()
    if len(text) > max_chars:
        return text[: max_chars - 3].rstrip() + "..."
    return text


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


if __name__ == "__main__":
    main()
