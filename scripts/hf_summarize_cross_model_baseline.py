"""Summarize fixed-subset DoLa baseline characterization across two models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import ensure_output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize fixed-subset baseline characterization across two models."
    )
    parser.add_argument("--mistral-compare", type=Path, required=True)
    parser.add_argument("--mistral-audit", type=Path, required=True)
    parser.add_argument("--openllama-compare", type=Path, required=True)
    parser.add_argument("--openllama-audit", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "cross_model_baseline_characterization",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _find_best_static_run(runs: dict[str, dict[str, object]]) -> tuple[str, dict[str, object]]:
    static_runs = {
        name: summary
        for name, summary in runs.items()
        if name.startswith("static_")
    }
    if not static_runs:
        raise ValueError("Expected at least one static run in compare summary.")
    return max(static_runs.items(), key=lambda item: float(item[1]["avg_mc2"]))


def _average(values: list[float]) -> float:
    if not values:
        raise ValueError("Cannot average an empty list.")
    return sum(values) / len(values)


def _layer_dist_top(layer_dist: dict[str, int] | None) -> tuple[str | None, float]:
    if not layer_dist:
        return None, 0.0
    total = sum(int(count) for count in layer_dist.values())
    top_layer, top_count = max(layer_dist.items(), key=lambda item: int(item[1]))
    ratio = float(top_count) / float(total) if total else 0.0
    return top_layer, ratio


def _summarize_model(compare_summary: dict[str, object], audit_summary: dict[str, object]) -> dict[str, object]:
    runs = compare_summary["runs"]
    vanilla = runs["vanilla_official"]
    dynamic_current = runs["dynamic_current_bucket"]
    dynamic_shifted = runs.get("dynamic_shifted_bucket")
    best_static_name, best_static = _find_best_static_run(runs)

    cases = audit_summary["cases"]
    current_dominant_ratios = [float(case["current_bucket"]["dominant_layer_ratio"]) for case in cases]
    shifted_dominant_ratios = [float(case["shifted_bucket"]["dominant_layer_ratio"]) for case in cases]
    current_margins = [float(case["current_bucket"]["average_top1_top2_margin"]) for case in cases]
    shifted_margins = [float(case["shifted_bucket"]["average_top1_top2_margin"]) for case in cases]
    dynamic_minus_best_static = [float(case["current_bucket"]["dynamic_minus_best_static"]) for case in cases]
    shifted_dynamic_deltas = [abs(float(case["bucket_comparison"]["dynamic_score_delta"])) for case in cases]

    current_top_layer, current_top_ratio = _layer_dist_top(dynamic_current.get("premature_layer_dist"))
    shifted_top_layer, shifted_top_ratio = _layer_dist_top(
        dynamic_shifted.get("premature_layer_dist") if isinstance(dynamic_shifted, dict) else None
    )

    return {
        "model_name": compare_summary["model_name"],
        "subset_seed": compare_summary.get("subset_seed", audit_summary.get("subset_seed")),
        "subset_indices": compare_summary.get("subset_indices", audit_summary.get("subset_indices")),
        "num_samples": int(vanilla["num_samples"]),
        "vanilla_mc1": float(vanilla["avg_mc1"]),
        "vanilla_mc2": float(vanilla["avg_mc2"]),
        "vanilla_mc3": float(vanilla["avg_mc3"]),
        "best_static_name": best_static_name,
        "best_static_mc1": float(best_static["avg_mc1"]),
        "best_static_mc2": float(best_static["avg_mc2"]),
        "best_static_mc3": float(best_static["avg_mc3"]),
        "dynamic_current_mc1": float(dynamic_current["avg_mc1"]),
        "dynamic_current_mc2": float(dynamic_current["avg_mc2"]),
        "dynamic_current_mc3": float(dynamic_current["avg_mc3"]),
        "dynamic_shifted_mc1": None if not isinstance(dynamic_shifted, dict) else float(dynamic_shifted["avg_mc1"]),
        "dynamic_shifted_mc2": None if not isinstance(dynamic_shifted, dict) else float(dynamic_shifted["avg_mc2"]),
        "dynamic_shifted_mc3": None if not isinstance(dynamic_shifted, dict) else float(dynamic_shifted["avg_mc3"]),
        "dynamic_current_minus_best_static_mc2": float(dynamic_current["avg_mc2"]) - float(best_static["avg_mc2"]),
        "dynamic_shifted_minus_current_mc2": None if not isinstance(dynamic_shifted, dict) else float(dynamic_shifted["avg_mc2"]) - float(dynamic_current["avg_mc2"]),
        "audit_final_judgement": audit_summary.get("final_judgement"),
        "current_dominant_layer": current_top_layer,
        "current_dominant_layer_ratio": current_top_ratio,
        "shifted_dominant_layer": shifted_top_layer,
        "shifted_dominant_layer_ratio": shifted_top_ratio,
        "avg_current_dominant_ratio": _average(current_dominant_ratios),
        "avg_shifted_dominant_ratio": _average(shifted_dominant_ratios),
        "avg_current_margin": _average(current_margins),
        "avg_shifted_margin": _average(shifted_margins),
        "avg_dynamic_minus_best_static": _average(dynamic_minus_best_static),
        "avg_shifted_dynamic_score_delta": _average(shifted_dynamic_deltas),
    }


def _classify_baseline(mistral: dict[str, object], openllama: dict[str, object]) -> tuple[str, str]:
    both_collapse = (
        float(abs(mistral["dynamic_current_minus_best_static_mc2"])) <= 0.01
        and float(abs(openllama["dynamic_current_minus_best_static_mc2"])) <= 0.01
        and float(mistral["avg_current_dominant_ratio"]) >= 0.8
        and float(openllama["avg_current_dominant_ratio"]) >= 0.8
    )
    mistral_neighbor_similarity = float(mistral["avg_shifted_dynamic_score_delta"] or 0.0) < 0.01
    openllama_bucket_sensitive = float(openllama["avg_shifted_dynamic_score_delta"] or 0.0) >= 0.01

    if both_collapse and mistral_neighbor_similarity and openllama_bucket_sensitive:
        return (
            "LIKELY_BASELINE_READY_FOR_ALGO_IMPROVEMENT",
            "Both models show stable dynamic collapse on the same fixed subset, but they differ in how bucket shifts matter: Mistral looks more like neighbor-layer similarity collapse, while OpenLLaMA is more bucket-sensitive.",
        )
    if both_collapse:
        return (
            "LIKELY_NEED_ONE_MORE_LARGER_SUBSET_CHECK",
            "The fixed subset already reproduces collapse on both models, but the cross-model pattern is not yet clean enough to call fully stable.",
        )
    if float(abs(mistral["dynamic_current_minus_best_static_mc2"])) > 0.02 or float(abs(openllama["dynamic_current_minus_best_static_mc2"])) > 0.02:
        return (
            "LIKELY_MODEL_SPECIFIC_BEHAVIOR_NOT_YET_STABLE",
            "At least one model still shows non-trivial dynamic-vs-best-static gaps on the fixed subset, so the characterization is not yet stable enough.",
        )
    return (
        "LIKELY_STILL_TOO_EARLY_TO_CALL_BASELINE_TRUSTWORTHY",
        "The current cross-model evidence is still too weak or mixed to call the baseline characterization trustworthy.",
    )


def main() -> None:
    args = parse_args()
    output_dir = ensure_output_dir(args.output_dir)

    mistral_compare = _load_json(args.mistral_compare)
    mistral_audit = _load_json(args.mistral_audit)
    openllama_compare = _load_json(args.openllama_compare)
    openllama_audit = _load_json(args.openllama_audit)

    mistral_summary = _summarize_model(mistral_compare, mistral_audit)
    openllama_summary = _summarize_model(openllama_compare, openllama_audit)
    final_label, final_reason = _classify_baseline(mistral_summary, openllama_summary)

    report = {
        "task_name": "cross_model_baseline_characterization",
        "models": {
            "mistral": mistral_summary,
            "openllama": openllama_summary,
        },
        "cross_model_findings": {
            "both_models_collapse": (
                float(mistral_summary["avg_current_dominant_ratio"]) >= 0.8
                and float(openllama_summary["avg_current_dominant_ratio"]) >= 0.8
            ),
            "mistral_neighbor_layer_similarity_signal": float(mistral_summary["avg_shifted_dynamic_score_delta"]) < 0.01,
            "openllama_bucket_sensitivity_signal": float(openllama_summary["avg_shifted_dynamic_score_delta"]) >= 0.01,
            "dynamic_gain_over_best_static_mc2": {
                "mistral": float(mistral_summary["dynamic_current_minus_best_static_mc2"]),
                "openllama": float(openllama_summary["dynamic_current_minus_best_static_mc2"]),
            },
        },
        "final_label": final_label,
        "final_reason": final_reason,
    }

    output_path = output_dir / "cross_model_baseline_characterization.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print("[hf_summarize_cross_model_baseline] Mistral:")
    print(
        f"  vanilla_mc2={mistral_summary['vanilla_mc2']:.4f}, best_static={mistral_summary['best_static_name']} ({mistral_summary['best_static_mc2']:.4f}), dynamic_current={mistral_summary['dynamic_current_mc2']:.4f}, dynamic_shifted={mistral_summary['dynamic_shifted_mc2']:.4f}"
    )
    print(
        f"  dominant={mistral_summary['current_dominant_layer']} ({mistral_summary['avg_current_dominant_ratio']:.2f}), avg_margin={mistral_summary['avg_current_margin']:.6f}, shifted_delta={mistral_summary['avg_shifted_dynamic_score_delta']:.6f}"
    )
    print("[hf_summarize_cross_model_baseline] OpenLLaMA:")
    print(
        f"  vanilla_mc2={openllama_summary['vanilla_mc2']:.4f}, best_static={openllama_summary['best_static_name']} ({openllama_summary['best_static_mc2']:.4f}), dynamic_current={openllama_summary['dynamic_current_mc2']:.4f}, dynamic_shifted={openllama_summary['dynamic_shifted_mc2']:.4f}"
    )
    print(
        f"  dominant={openllama_summary['current_dominant_layer']} ({openllama_summary['avg_current_dominant_ratio']:.2f}), avg_margin={openllama_summary['avg_current_margin']:.6f}, shifted_delta={openllama_summary['avg_shifted_dynamic_score_delta']:.6f}"
    )
    print(f"[hf_summarize_cross_model_baseline] final_label={final_label}")
    print(f"[hf_summarize_cross_model_baseline] Saved summary to: {output_path}")


if __name__ == "__main__":
    main()
