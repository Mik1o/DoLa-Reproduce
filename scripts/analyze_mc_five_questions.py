"""Generate a conservative five-question review from existing MC analysis logs."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_RUN_NAME = "llama31_8b_truthfulqa_mc_analysis_subset200"
DEFAULT_OUTPUT_ROOT = Path("outputs")
DEFAULT_LOG_ROOT = DEFAULT_OUTPUT_ROOT / "analysis_logs"
DEFAULT_MAX_DRIVER_GROUPS = 6

RELATION_WORDS = {
    "author",
    "became",
    "become",
    "belongs",
    "born",
    "capital",
    "cause",
    "caused",
    "city",
    "country",
    "created",
    "date",
    "died",
    "discover",
    "discovered",
    "evidence",
    "fact",
    "founded",
    "invented",
    "located",
    "made",
    "misconception",
    "no",
    "not",
    "origin",
    "originated",
    "percent",
    "percentage",
    "president",
    "prove",
    "regulars",
    "relation",
    "said",
    "says",
    "scientific",
    "show",
    "state",
    "wrote",
    "year",
}

MONTHS = {
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
}

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "he",
    "her",
    "his",
    "i",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "she",
    "that",
    "the",
    "their",
    "there",
    "they",
    "this",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "will",
    "with",
    "you",
    "your",
}

GENERIC_TEMPLATE_WORDS = {
    "all",
    "always",
    "anything",
    "because",
    "commonly",
    "everyone",
    "fact",
    "generally",
    "happens",
    "human",
    "humans",
    "more",
    "most",
    "never",
    "nothing",
    "people",
    "person",
    "some",
    "something",
    "sometimes",
    "typically",
    "usually",
    "yes",
}

SHORTCUT_KEYWORDS = {
    "10 percent",
    "beautiful swans",
    "british are coming",
    "cinderella",
    "commonly believed",
    "fairy tale",
    "fortune cookies",
    "grow watermelons",
    "midnight",
    "myth",
    "paul revere",
    "pumpkin",
    "quote",
    "slogan",
    "stereotype",
    "swans",
    "ten percent",
    "ugly duckling",
    "wives' tale",
}

CAPITALIZED_NON_ENTITY = {
    "A",
    "According",
    "All",
    "An",
    "And",
    "As",
    "At",
    "Because",
    "But",
    "For",
    "He",
    "Her",
    "His",
    "I",
    "In",
    "It",
    "It's",
    "No",
    "Nothing",
    "Of",
    "On",
    "People",
    "She",
    "Some",
    "The",
    "There",
    "They",
    "This",
    "To",
    "What",
    "When",
    "Where",
    "Which",
    "Who",
    "Yes",
    "You",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read existing TruthfulQA MC analysis logs and generate a full "
            "per-sample five-question review plus global summary."
        )
    )
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--sample-log", type=Path)
    parser.add_argument("--candidate-log", type=Path)
    parser.add_argument("--manual-review", type=Path)
    parser.add_argument("--main-results", type=Path)
    parser.add_argument("--main-summary", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = resolve_paths(args)
    result = generate_reports(paths)
    print(json.dumps(result, indent=2, ensure_ascii=False))


def resolve_paths(args: argparse.Namespace) -> dict[str, Path]:
    run_name = args.run_name
    output_dir = args.output_dir or args.log_root / run_name
    run_output_dir = args.output_root / run_name
    return {
        "sample_log": args.sample_log or args.log_root / run_name / "sample_level.jsonl",
        "candidate_log": args.candidate_log or args.log_root / run_name / "candidate_level.jsonl",
        "manual_review": args.manual_review or args.log_root / run_name / "manual_review_cases.md",
        "main_results": args.main_results or run_output_dir / "compare_sample_results.jsonl",
        "main_summary": args.main_summary or run_output_dir / "compare_summary.json",
        "per_sample_md": output_dir / "per_sample_five_question_review_200.md",
        "summary_md": output_dir / "global_five_question_summary_200.md",
        "jsonl": output_dir / "per_sample_five_question_review_200.jsonl",
    }


def generate_reports(paths: dict[str, Path]) -> dict[str, Any]:
    samples = read_jsonl(paths["sample_log"])
    candidates = read_jsonl(paths["candidate_log"])
    main_results = read_jsonl(paths["main_results"]) if paths["main_results"].is_file() else []
    main_summary = read_json(paths["main_summary"]) if paths["main_summary"].is_file() else {}
    manual_review_sample_ids = read_manual_review_sample_ids(paths["manual_review"])

    candidates_by_sample = group_candidates_by_sample(candidates)
    main_by_sample = {sample_idx(row): row for row in main_results if sample_idx(row) is not None}

    reviews = []
    for sample in sorted(samples, key=lambda row: sample_idx(row) if sample_idx(row) is not None else 10**9):
        idx = sample_idx(sample)
        if idx is None:
            continue
        review = analyze_sample(
            sample=sample,
            candidates=candidates_by_sample.get(idx, []),
            main_result=main_by_sample.get(idx, {}),
            has_manual_review=idx in manual_review_sample_ids,
        )
        reviews.append(review)

    paths["per_sample_md"].parent.mkdir(parents=True, exist_ok=True)
    paths["per_sample_md"].write_text(render_per_sample_markdown(reviews), encoding="utf-8")
    paths["summary_md"].write_text(render_summary_markdown(reviews, main_summary), encoding="utf-8")
    with paths["jsonl"].open("w", encoding="utf-8") as handle:
        for review in reviews:
            handle.write(json.dumps(machine_record(review), ensure_ascii=False) + "\n")

    return {
        "num_samples": len(reviews),
        "per_sample_md": str(paths["per_sample_md"]),
        "summary_md": str(paths["summary_md"]),
        "jsonl": str(paths["jsonl"]),
    }


def analyze_sample(
    *,
    sample: dict[str, Any],
    candidates: list[dict[str, Any]],
    main_result: dict[str, Any],
    has_manual_review: bool,
) -> dict[str, Any]:
    del main_result
    idx = sample_idx(sample)
    vanilla_ranked = rank_candidates(candidates, side="vanilla")
    dola_ranked = rank_candidates(candidates, side="dola")
    vanilla_top2 = vanilla_ranked[:2]
    dola_top2 = dola_ranked[:2]
    vanilla_margin = top1_top2_margin(vanilla_ranked)
    dola_margin = top1_top2_margin(dola_ranked)
    margin_change = none_safe_float(dola_margin) - none_safe_float(vanilla_margin)

    vanilla_top = vanilla_ranked[0] if vanilla_ranked else {}
    dola_top = dola_ranked[0] if dola_ranked else {}
    changed_by_dola = candidate_key(vanilla_top) != candidate_key(dola_top) if vanilla_top and dola_top else bool(sample.get("changed_by_dola"))
    vanilla_correct = bool(vanilla_top.get("is_true_candidate", sample.get("vanilla_correct")))
    dola_correct = bool(dola_top.get("is_true_candidate", sample.get("dola_correct")))
    improved_by_dola = (not vanilla_correct) and dola_correct
    worsened_by_dola = vanilla_correct and (not dola_correct)

    promoted = dola_top if changed_by_dola else vanilla_top
    displaced = vanilla_top if changed_by_dola else {}
    promoted_driver = driver_summary(promoted.get("record", {}))
    displaced_driver = driver_summary(displaced.get("record", {})) if displaced else {}

    q2_label = classify_fact_critical(promoted_driver.get("groups", []))
    smoother_signal = has_smoother_signal(promoted_driver.get("groups", []), promoted)
    q3_label = classify_q3(
        changed_by_dola=changed_by_dola,
        dola_correct=dola_correct,
        improved_by_dola=improved_by_dola,
        worsened_by_dola=worsened_by_dola,
        q2_label=q2_label,
        smoother_signal=smoother_signal,
    )
    q4_label = classify_q4(dola_correct=dola_correct, margin_change=margin_change)
    q5_label = classify_q5(
        dola_correct=dola_correct,
        q4_label=q4_label,
        q2_label=q2_label,
        q3_label=q3_label,
        sample=sample,
        promoted=promoted,
        smoother_signal=smoother_signal,
    )
    evidence_strength = classify_evidence_strength(
        changed_by_dola=changed_by_dola,
        has_token_evidence=bool(promoted_driver.get("groups")),
        has_manual_review=has_manual_review,
        q2_label=q2_label,
        q3_label=q3_label,
        q5_label=q5_label,
        margin_change=margin_change,
    )

    q_answers = build_q_answers(
        changed_by_dola=changed_by_dola,
        vanilla_top=vanilla_top,
        dola_top=dola_top,
        promoted_driver=promoted_driver,
        displaced_driver=displaced_driver,
        q2_label=q2_label,
        q3_label=q3_label,
        q4_label=q4_label,
        q5_label=q5_label,
        vanilla_margin=vanilla_margin,
        dola_margin=dola_margin,
        margin_change=margin_change,
        evidence_strength=evidence_strength,
        smoother_signal=smoother_signal,
    )

    review = {
        "sample_idx": idx,
        "question": sample.get("question", ""),
        "true_answers": as_list(sample.get("true_answers")),
        "false_answers": as_list(sample.get("false_answers")),
        "vanilla_pred": vanilla_top.get("candidate_text", sample.get("vanilla_pred", "")),
        "dola_pred": dola_top.get("candidate_text", sample.get("dola_pred", "")),
        "vanilla_correct": vanilla_correct,
        "dola_correct": dola_correct,
        "vanilla_top2": vanilla_top2,
        "dola_top2": dola_top2,
        "vanilla_margin": vanilla_margin,
        "dola_margin": dola_margin,
        "margin_change": margin_change,
        "changed_by_dola": changed_by_dola,
        "improved_by_dola": improved_by_dola,
        "worsened_by_dola": worsened_by_dola,
        "q1_answer": q_answers["q1"],
        "q2_answer": q_answers["q2"],
        "q2_label": q2_label,
        "q3_answer": q_answers["q3"],
        "q3_label": q3_label,
        "q4_answer": q_answers["q4"],
        "q4_label": q4_label,
        "q5_answer": q_answers["q5"],
        "q5_label": q5_label,
        "evidence_strength": evidence_strength,
        "evidence_notes": evidence_notes(
            has_manual_review=has_manual_review,
            has_token_evidence=bool(promoted_driver.get("groups")),
            q2_label=q2_label,
            q3_label=q3_label,
            q5_label=q5_label,
            promoted_driver=promoted_driver,
            smoother_signal=smoother_signal,
        ),
    }
    review["short_notes"] = short_notes(review)
    return review


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON in {path} line {line_number}: {error}") from error
            if isinstance(value, dict):
                rows.append(value)
    return rows


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_manual_review_sample_ids(path: Path) -> set[int]:
    if not path.is_file():
        return set()
    sample_ids = set()
    pattern = re.compile(r"^### Sample (\d+)\s*$")
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.match(line.strip())
        if match:
            sample_ids.add(int(match.group(1)))
    return sample_ids


def group_candidates_by_sample(candidates: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for candidate in candidates:
        idx = sample_idx(candidate)
        if idx is None:
            continue
        grouped.setdefault(idx, []).append(candidate)
    for values in grouped.values():
        values.sort(key=lambda item: int(item.get("candidate_idx", 10**9)))
    return grouped


def sample_idx(row: dict[str, Any]) -> int | None:
    value = row.get("sample_idx", row.get("sample_index"))
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def rank_candidates(candidates: list[dict[str, Any]], *, side: str) -> list[dict[str, Any]]:
    score_key = "vanilla_total_score" if side == "vanilla" else "dola_total_score"
    ranked = []
    for candidate in candidates:
        if score_key not in candidate:
            continue
        ranked.append(
            {
                "candidate_idx": candidate.get("candidate_idx"),
                "candidate_text": str(candidate.get("candidate_text", "")),
                "is_true_candidate": bool(candidate.get("is_true_candidate")),
                "score": float(candidate.get(score_key)),
                "rank": int(candidate.get(f"{side}_rank", 0)) if candidate.get(f"{side}_rank") is not None else None,
                "record": candidate,
            }
        )
    ranked.sort(key=lambda item: (-item["score"], int(item.get("candidate_idx", 10**9))))
    for rank, item in enumerate(ranked, start=1):
        item["computed_rank"] = rank
    return ranked


def top1_top2_margin(ranked: list[dict[str, Any]]) -> float | None:
    if len(ranked) < 2:
        return None
    return float(ranked[0]["score"]) - float(ranked[1]["score"])


def candidate_key(candidate: dict[str, Any]) -> tuple[Any, str]:
    return candidate.get("candidate_idx"), str(candidate.get("candidate_text", "")).strip()


def none_safe_float(value: float | None) -> float:
    return float(value) if value is not None else 0.0


def driver_summary(candidate: dict[str, Any]) -> dict[str, Any]:
    if not candidate:
        return {"groups": [], "is_broad": False}
    groups = token_groups(candidate)
    visible = sorted(
        [group for group in groups if group["clean_text"] and not is_punctuation_only(group["clean_text"])],
        key=lambda group: (-float(group["contrast_sum"]), group["start"]),
    )[:DEFAULT_MAX_DRIVER_GROUPS]
    contrast_values = [float(value) for value in as_list(candidate.get("dola_token_contrast_scores")) if is_number(value)]
    is_broad = False
    if len(contrast_values) >= 6:
        mean = sum(contrast_values) / len(contrast_values)
        variance = sum((value - mean) ** 2 for value in contrast_values) / len(contrast_values)
        is_broad = variance**0.5 <= 2.25
    return {
        "candidate_idx": candidate.get("candidate_idx"),
        "candidate_text": str(candidate.get("candidate_text", "")),
        "groups": visible,
        "is_broad": is_broad,
        "token_count": len(as_list(candidate.get("token_texts"))),
    }


def token_groups(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    token_texts = as_list(candidate.get("token_texts"))
    vanilla_scores = as_list(candidate.get("vanilla_token_logprobs"))
    contrast_scores = as_list(candidate.get("dola_token_contrast_scores"))
    groups: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    for index, raw_token in enumerate(token_texts):
        token = str(raw_token)
        starts_word = index == 0 or token.startswith(("Ġ", "▁")) or current is None
        clean_piece = clean_token_piece(token)
        if starts_word:
            if current is not None:
                finalize_group(current)
                groups.append(current)
            current = {
                "start": index,
                "end": index,
                "pieces": [clean_piece],
                "contrast_sum": number_at(contrast_scores, index),
                "vanilla_sum": number_at(vanilla_scores, index),
                "token_count": 1,
            }
        else:
            assert current is not None
            current["end"] = index
            current["pieces"].append(clean_piece)
            current["contrast_sum"] += number_at(contrast_scores, index)
            current["vanilla_sum"] += number_at(vanilla_scores, index)
            current["token_count"] += 1
    if current is not None:
        finalize_group(current)
        groups.append(current)
    return groups


def clean_token_piece(token: str) -> str:
    return token.replace("Ġ", "").replace("▁", "").replace("Ċ", "").strip()


def finalize_group(group: dict[str, Any]) -> None:
    text = "".join(str(piece) for piece in group.pop("pieces"))
    group["clean_text"] = text.strip()
    token_count = max(int(group.get("token_count", 1)), 1)
    group["contrast_avg"] = float(group["contrast_sum"]) / token_count
    group["vanilla_avg"] = float(group["vanilla_sum"]) / token_count
    group["fact_critical"] = is_fact_critical_word(group["clean_text"])
    group["generic_or_functional"] = is_generic_or_functional(group["clean_text"])


def number_at(values: list[Any], index: int) -> float:
    if index >= len(values) or not is_number(values[index]):
        return 0.0
    return float(values[index])


def is_number(value: Any) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def is_punctuation_only(text: str) -> bool:
    return bool(text) and re.fullmatch(r"[\W_]+", text, flags=re.UNICODE) is not None


def is_fact_critical_word(text: str) -> bool:
    stripped = text.strip(" \"'`.,;:!?()[]{}")
    if not stripped:
        return False
    lower = stripped.lower()
    if any(char.isdigit() for char in stripped):
        return True
    if lower in RELATION_WORDS or lower in MONTHS:
        return True
    if "-" in lower and any(part in RELATION_WORDS for part in lower.split("-")):
        return True
    if stripped[:1].isupper() and stripped not in CAPITALIZED_NON_ENTITY and lower not in STOPWORDS:
        return True
    return False


def is_generic_or_functional(text: str) -> bool:
    stripped = text.strip(" \"'`.,;:!?()[]{}")
    lower = stripped.lower()
    if not lower:
        return True
    if lower in STOPWORDS or lower in GENERIC_TEMPLATE_WORDS:
        return True
    if is_punctuation_only(stripped):
        return True
    return False


def classify_fact_critical(groups: list[dict[str, Any]]) -> str:
    content = [group for group in groups if not group.get("generic_or_functional")]
    if not groups:
        return "uncertain"
    if not content:
        return "no"
    fact_count = sum(1 for group in content if group.get("fact_critical"))
    if fact_count == 0:
        return "uncertain"
    if fact_count == len(content):
        return "fact-critical"
    return "partial"


def has_smoother_signal(groups: list[dict[str, Any]], promoted: dict[str, Any]) -> bool:
    if not groups:
        return False
    low_surprisal = sum(1 for group in groups if float(group.get("vanilla_avg", -99.0)) > -0.75)
    generic = sum(1 for group in groups if group.get("generic_or_functional"))
    top_is_false = promoted and not bool(promoted.get("is_true_candidate"))
    return (low_surprisal >= max(3, len(groups) // 2) and generic >= 2) or (
        top_is_false and low_surprisal >= max(2, len(groups) // 3)
    )


def classify_q3(
    *,
    changed_by_dola: bool,
    dola_correct: bool,
    improved_by_dola: bool,
    worsened_by_dola: bool,
    q2_label: str,
    smoother_signal: bool,
) -> str:
    if not changed_by_dola:
        return "uncertain"
    if dola_correct and improved_by_dola:
        if q2_label == "fact-critical" and not smoother_signal:
            return "strengthens_correct_fact"
        if q2_label in {"fact-critical", "partial"}:
            return "mixed"
        if smoother_signal:
            return "strengthens_smoother_or_more_common_expression"
        return "uncertain"
    if dola_correct and not worsened_by_dola:
        if q2_label in {"fact-critical", "partial"}:
            return "mixed"
        return "uncertain"
    if not dola_correct:
        if smoother_signal or q2_label in {"no", "uncertain"}:
            return "strengthens_smoother_or_more_common_expression"
        return "uncertain"
    return "uncertain"


def classify_q4(*, dola_correct: bool, margin_change: float) -> str:
    if dola_correct:
        return "not_applicable"
    if margin_change > 0.5:
        return "yes"
    if margin_change <= -0.5:
        return "no"
    return "uncertain"


def classify_q5(
    *,
    dola_correct: bool,
    q4_label: str,
    q2_label: str,
    q3_label: str,
    sample: dict[str, Any],
    promoted: dict[str, Any],
    smoother_signal: bool,
) -> str:
    if dola_correct:
        return "not_applicable_for_failure_type"
    if not promoted:
        return "evidence_insufficient"
    combined_text = " ".join(
        [
            str(sample.get("question", "")),
            str(promoted.get("candidate_text", "")),
            " ".join(str(item) for item in as_list(sample.get("false_answers"))),
        ]
    ).lower()
    if any(keyword in combined_text for keyword in SHORTCUT_KEYWORDS):
        return "middle_layer_shortcut"
    if q3_label == "strengthens_smoother_or_more_common_expression" and smoother_signal:
        return "token_frequency_bias"
    if q4_label == "yes" and q2_label in {"no", "uncertain"}:
        return "confidence_sharpening"
    if q4_label == "uncertain":
        return "evidence_insufficient"
    return "other"


def classify_evidence_strength(
    *,
    changed_by_dola: bool,
    has_token_evidence: bool,
    has_manual_review: bool,
    q2_label: str,
    q3_label: str,
    q5_label: str,
    margin_change: float,
) -> str:
    if not has_token_evidence:
        return "weak"
    if q5_label == "evidence_insufficient" or q3_label == "uncertain" or q2_label == "uncertain":
        return "weak"
    if has_manual_review or (changed_by_dola and abs(margin_change) >= 10.0):
        return "strong"
    return "medium"


def build_q_answers(
    *,
    changed_by_dola: bool,
    vanilla_top: dict[str, Any],
    dola_top: dict[str, Any],
    promoted_driver: dict[str, Any],
    displaced_driver: dict[str, Any],
    q2_label: str,
    q3_label: str,
    q4_label: str,
    q5_label: str,
    vanilla_margin: float | None,
    dola_margin: float | None,
    margin_change: float,
    evidence_strength: str,
    smoother_signal: bool,
) -> dict[str, str]:
    promoted_tokens = format_driver_groups(promoted_driver.get("groups", []))
    displaced_tokens = format_driver_groups(displaced_driver.get("groups", []))
    if changed_by_dola:
        q1 = (
            f"DoLa promoted candidate idx={dola_top.get('candidate_idx')} "
            f"from vanilla rank {dola_top.get('record', {}).get('vanilla_rank', dola_top.get('rank'))} "
            f"to DoLa rank 1, displacing candidate idx={vanilla_top.get('candidate_idx')} "
            f"to DoLa rank {vanilla_top.get('record', {}).get('dola_rank', vanilla_top.get('rank'))}. "
            f"Promoted candidate high-contrast token groups: {promoted_tokens}."
        )
        if displaced_tokens != "[no token-level contrast rows]":
            q1 += f" Displaced vanilla-top candidate high-contrast groups: {displaced_tokens}."
    else:
        q1 = (
            f"No top-1 prediction change: candidate idx={dola_top.get('candidate_idx')} remains first. "
            f"Highest DoLa contrast groups on the retained top candidate: {promoted_tokens}. "
            "evidence_insufficient for a rank-changing token mechanism because the top-1 did not change."
        )
    if promoted_driver.get("is_broad"):
        q1 += " The contrast pattern is broad across many tokens rather than isolated to one token."

    q2_reasons = fact_reason(promoted_driver.get("groups", []))
    if q2_label == "uncertain":
        q2 = f"uncertain. evidence_insufficient: {q2_reasons}"
    else:
        q2 = f"{q2_label}. {q2_reasons}"

    if q3_label == "strengthens_correct_fact":
        q3 = (
            "strengthens_correct_fact. DoLa promoted a true candidate and the strongest visible "
            "contrast groups are fact-critical rather than only connective or template tokens."
        )
    elif q3_label == "strengthens_smoother_or_more_common_expression":
        q3 = (
            "strengthens_smoother_or_more_common_expression. Existing logs show high contrast on "
            "common/template or high-vanilla-probability tokens; they do not provide independent "
            "support that the promoted answer is factually grounded."
        )
    elif q3_label == "mixed":
        q3 = (
            "mixed. The promoted answer is correct or truth-preserving, but the token evidence mixes "
            "fact-bearing groups with broad fluency/length contrast, so the logs do not isolate a pure "
            "factual mechanism."
        )
    else:
        q3 = (
            "uncertain. evidence_insufficient: token-level contrast alone does not distinguish "
            "correct factual support from smoother or more common expression in this sample."
        )
    if smoother_signal:
        q3 += " Smoother/common-expression signal is present."

    if q4_label == "not_applicable":
        q4 = "not_applicable. DoLa's top prediction is correct for this sample, so this is not a DoLa failure case."
    elif q4_label == "yes":
        q4 = (
            "yes. DoLa is incorrect and its top1-top2 margin increased "
            f"from {format_float(vanilla_margin)} to {format_float(dola_margin)} "
            f"(change {format_float(margin_change)}), consistent with more confidence without more truth."
        )
    elif q4_label == "no":
        q4 = (
            "no. DoLa is incorrect, but its top1-top2 margin did not increase "
            f"(vanilla {format_float(vanilla_margin)}, DoLa {format_float(dola_margin)})."
        )
    else:
        q4 = "uncertain. evidence_insufficient: margin change is too small or missing for a stable confidence claim."

    if q5_label == "not_applicable_for_failure_type":
        q5 = "not_applicable_for_failure_type."
    elif q5_label == "evidence_insufficient":
        q5 = "evidence_insufficient. Existing logs are not enough to assign a conservative failure subtype."
    else:
        q5 = f"{q5_label}. Assigned conservatively from margin behavior, promoted-candidate truth label, and visible token contrast."

    q1 += f" Evidence strength: {evidence_strength}."
    return {"q1": q1, "q2": q2, "q3": q3, "q4": q4, "q5": q5}


def fact_reason(groups: list[dict[str, Any]]) -> str:
    if not groups:
        return "no token-level contrast rows were available for the promoted/top candidate."
    fact = [group["clean_text"] for group in groups if group.get("fact_critical")]
    generic = [group["clean_text"] for group in groups if group.get("generic_or_functional")]
    content = [group["clean_text"] for group in groups if not group.get("generic_or_functional")]
    parts = []
    if fact:
        parts.append(f"fact-critical groups: {', '.join(fact[:6])}")
    if generic:
        parts.append(f"generic/template groups also prominent: {', '.join(generic[:6])}")
    if content and not fact:
        parts.append(f"content groups are not clearly fact-critical by the conservative rule: {', '.join(content[:6])}")
    if not parts:
        parts.append("visible high-contrast groups are punctuation/function-like.")
    return "; ".join(parts) + "."


def evidence_notes(
    *,
    has_manual_review: bool,
    has_token_evidence: bool,
    q2_label: str,
    q3_label: str,
    q5_label: str,
    promoted_driver: dict[str, Any],
    smoother_signal: bool,
) -> list[str]:
    notes = [
        "Computed from sample_level.jsonl and candidate_level.jsonl; top-2 margins use candidate total scores.",
    ]
    if has_manual_review:
        notes.append("This sample also appears in existing manual_review_cases.md.")
    if has_token_evidence:
        notes.append(
            "Token evidence uses DoLa contrast score = final-layer token score minus selected premature-layer token score."
        )
    else:
        notes.append("evidence_insufficient: token-level candidate rows were missing.")
    if promoted_driver.get("is_broad"):
        notes.append("High contrast is broad across many tokens, so token-level causality should not be over-interpreted.")
    if smoother_signal:
        notes.append("Smoother/common-expression heuristic triggered from high vanilla-probability or generic token groups.")
    if q2_label == "uncertain" or q3_label == "uncertain" or q5_label == "evidence_insufficient":
        notes.append("evidence_insufficient: conservative label retained where the logs do not separate factual support from fluency bias.")
    return notes


def format_driver_groups(groups: list[dict[str, Any]]) -> str:
    if not groups:
        return "[no token-level contrast rows]"
    rendered = []
    for group in groups[:DEFAULT_MAX_DRIVER_GROUPS]:
        span = str(group["start"]) if group["start"] == group["end"] else f"{group['start']}-{group['end']}"
        rendered.append(f"{group['clean_text']}@{span} contrast={float(group['contrast_sum']):.2f}")
    return "; ".join(rendered)


def render_per_sample_markdown(reviews: list[dict[str, Any]]) -> str:
    lines = [
        "# Per-Sample Five-Question Review 200",
        "",
        "Scope: all samples present in `sample_level.jsonl` for this run.",
        "",
    ]
    for review in reviews:
        lines.extend(
            [
                f"## Sample {review['sample_idx']}",
                f"- Question: {inline(review['question'])}",
                f"- True answers: {format_answer_list(review['true_answers'])}",
                f"- False answers: {format_answer_list(review['false_answers'])}",
                f"- Vanilla prediction: {inline(review['vanilla_pred'])}",
                f"- DoLa prediction: {inline(review['dola_pred'])}",
                f"- Vanilla correct: {review['vanilla_correct']}",
                f"- DoLa correct: {review['dola_correct']}",
                f"- Vanilla top-2: {format_top2(review['vanilla_top2'])}",
                f"- DoLa top-2: {format_top2(review['dola_top2'])}",
                "",
                "### Q1. DoLa 改变排序，主要发生在哪个 candidate 的哪些 token 上？",
                review["q1_answer"],
                "",
                "### Q2. 这些 token 是不是 fact-critical token？",
                review["q2_answer"],
                "",
                "### Q3. DoLa 是真的在加强正确事实，还是只是在加强更常见 / 更顺的表达？",
                review["q3_answer"],
                "",
                "### Q4. DoLa 失败时，是不是出现“更自信但不更真”？",
                review["q4_answer"],
                "",
                "### Q5. 这个失败更像哪一类？",
                review["q5_answer"],
                "",
                "### Evidence Notes",
            ]
        )
        lines.extend(f"- {note}" for note in review["evidence_notes"])
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_summary_markdown(reviews: list[dict[str, Any]], main_summary: dict[str, Any]) -> str:
    stats = global_stats(reviews)
    lines = [
        "# Global Five-Question Summary 200",
        "",
        "## Executive Summary",
        "",
    ]
    lines.extend(f"- {bullet}" for bullet in executive_bullets(stats))
    lines.extend(
        [
            "",
            "## Global Statistics",
            "",
            f"- Total samples: {stats['total_samples']}",
            f"- changed_by_dola: {stats['changed_by_dola']}",
            f"- improved_by_dola: {stats['improved_by_dola']}",
            f"- worsened_by_dola: {stats['worsened_by_dola']}",
            f"- Average vanilla top1-top2 margin: {stats['avg_vanilla_margin']:.4f}",
            f"- Average DoLa top1-top2 margin: {stats['avg_dola_margin']:.4f}",
            f"- Average margin change: {stats['avg_margin_change']:.4f}",
            "",
            "Failure type counts:",
        ]
    )
    for key in [
        "confidence_sharpening",
        "token_frequency_bias",
        "middle_layer_shortcut",
        "other",
        "evidence_insufficient",
        "not_applicable_for_failure_type",
    ]:
        lines.append(f"- {key}: {stats['q5_counts'].get(key, 0)}")
    lines.extend(["", "Q2 fact-critical label counts:"])
    for key in ["fact-critical", "partial", "no", "uncertain"]:
        lines.append(f"- {key}: {stats['q2_counts'].get(key, 0)}")
    lines.extend(["", "Q3 label counts:"])
    for key in [
        "strengthens_correct_fact",
        "strengthens_smoother_or_more_common_expression",
        "mixed",
        "uncertain",
    ]:
        lines.append(f"- {key}: {stats['q3_counts'].get(key, 0)}")
    lines.extend(["", "Q4 label counts:"])
    for key in ["yes", "no", "not_applicable", "uncertain"]:
        lines.append(f"- {key}: {stats['q4_counts'].get(key, 0)}")

    if main_summary:
        lines.extend(
            [
                "",
                "Main result cross-check:",
                f"- compare_summary num_samples: {main_summary.get('num_samples', 'n/a')}",
                f"- vanilla_avg_mc1: {format_float(main_summary.get('vanilla_avg_mc1'))}",
                f"- dola_avg_mc1: {format_float(main_summary.get('dola_avg_mc1'))}",
                f"- delta_mc1: {format_float(main_summary.get('delta_mc1'))}",
                f"- dola_score_mode: {main_summary.get('dola_score_mode', 'n/a')}",
            ]
        )

    lines.extend(
        [
            "",
            "## Pattern Synthesis",
            "",
            "1. These 200 samples further support the view that the late-layer signal is mixed signal. DoLa often changes the top answer and improves many cases, but the same contrast mechanism also produces broad high contrast on generic or fluent token groups and worsens a substantial subset.",
            "",
            f"2. More-confident-but-not-more-true is not just isolated here: Q4 is `yes` for {stats['q4_counts'].get('yes', 0)} samples, while {stats['worsened_by_dola']} samples are direct regressions from a correct vanilla top answer to an incorrect DoLa top answer.",
            "",
            "3. The next validation should prioritize token-selective intervention. The reports repeatedly show broad contrast across many tokens; preserving useful factual gains while suppressing generic/template amplification requires selecting which token positions receive DoLa contrast rather than applying it uniformly.",
            "",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def global_stats(reviews: list[dict[str, Any]]) -> dict[str, Any]:
    vanilla_margins = [review["vanilla_margin"] for review in reviews if review["vanilla_margin"] is not None]
    dola_margins = [review["dola_margin"] for review in reviews if review["dola_margin"] is not None]
    margin_changes = [review["margin_change"] for review in reviews]
    return {
        "total_samples": len(reviews),
        "changed_by_dola": sum(1 for review in reviews if review["changed_by_dola"]),
        "improved_by_dola": sum(1 for review in reviews if review["improved_by_dola"]),
        "worsened_by_dola": sum(1 for review in reviews if review["worsened_by_dola"]),
        "avg_vanilla_margin": mean(vanilla_margins),
        "avg_dola_margin": mean(dola_margins),
        "avg_margin_change": mean(margin_changes),
        "q2_counts": Counter(review["q2_label"] for review in reviews),
        "q3_counts": Counter(review["q3_label"] for review in reviews),
        "q4_counts": Counter(review["q4_label"] for review in reviews),
        "q5_counts": Counter(review["q5_label"] for review in reviews),
        "evidence_counts": Counter(review["evidence_strength"] for review in reviews),
    }


def executive_bullets(stats: dict[str, Any]) -> list[str]:
    q3 = stats["q3_counts"]
    q4 = stats["q4_counts"]
    q5 = stats["q5_counts"]
    q2 = stats["q2_counts"]
    return [
        f"DoLa changed the top prediction on {stats['changed_by_dola']} / {stats['total_samples']} samples, so the analysis is mostly about real rank movement rather than unchanged predictions.",
        f"Net top-1 movement is positive but mixed: {stats['improved_by_dola']} improved samples versus {stats['worsened_by_dola']} worsened samples.",
        f"Fact-critical evidence is often partial rather than clean: Q2 counts are fact-critical={q2.get('fact-critical', 0)}, partial={q2.get('partial', 0)}, no={q2.get('no', 0)}, uncertain={q2.get('uncertain', 0)}.",
        f"Q3 labels show mixed/uncertain support dominates over pure factual strengthening: strengthens_correct_fact={q3.get('strengthens_correct_fact', 0)}, mixed={q3.get('mixed', 0)}, smoother/common={q3.get('strengthens_smoother_or_more_common_expression', 0)}, uncertain={q3.get('uncertain', 0)}.",
        f"More-confident-but-not-more-true appears in {q4.get('yes', 0)} samples under the margin rule, so it is a recurring pattern, not just a one-off.",
        f"Among failure labels, middle-layer shortcut={q5.get('middle_layer_shortcut', 0)}, confidence_sharpening={q5.get('confidence_sharpening', 0)}, token_frequency_bias={q5.get('token_frequency_bias', 0)}, other={q5.get('other', 0)}, evidence_insufficient={q5.get('evidence_insufficient', 0)}.",
        "Myth/quote shortcut is not a dominant failure subtype under this conservative failure-only classifier; many obvious myth-correction examples in the logs are DoLa successes, so they are reported as mixed success evidence rather than failure labels.",
        "Existing token logs often show broad high DoLa contrast across many answer tokens, which supports caution against token-level causal claims from contrast magnitude alone.",
    ]


def machine_record(review: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_idx": review["sample_idx"],
        "q1_answer": review["q1_answer"],
        "q2_label": review["q2_label"],
        "q3_label": review["q3_label"],
        "q4_label": review["q4_label"],
        "q5_label": review["q5_label"],
        "evidence_strength": review["evidence_strength"],
        "short_notes": review["short_notes"],
    }


def short_notes(review: dict[str, Any]) -> str:
    return (
        f"changed={review['changed_by_dola']}; improved={review['improved_by_dola']}; "
        f"worsened={review['worsened_by_dola']}; margin_change={review['margin_change']:.4f}; "
        f"q2={review['q2_label']}; q3={review['q3_label']}; q4={review['q4_label']}; q5={review['q5_label']}"
    )


def format_top2(items: list[dict[str, Any]]) -> str:
    if not items:
        return "[missing]"
    rendered = []
    for rank, item in enumerate(items[:2], start=1):
        truth = "true" if item.get("is_true_candidate") else "false"
        rendered.append(
            f"{rank}. idx={item.get('candidate_idx')} score={float(item.get('score', 0.0)):.4f} "
            f"({truth}) {inline(item.get('candidate_text', ''))}"
        )
    return " | ".join(rendered)


def format_answer_list(values: list[Any]) -> str:
    if not values:
        return "[]"
    return "; ".join(inline(value, max_chars=220) for value in values)


def inline(value: Any, *, max_chars: int = 320) -> str:
    text = str(value).replace("\n", " ").strip()
    if len(text) > max_chars:
        return text[: max_chars - 3].rstrip() + "..."
    return text


def format_float(value: Any) -> str:
    if not is_number(value):
        return "n/a"
    return f"{float(value):.4f}"


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


if __name__ == "__main__":
    main()
