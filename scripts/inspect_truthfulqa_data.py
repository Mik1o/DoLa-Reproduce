"""Inspect normalized TruthfulQA multiple-choice samples from a CSV file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.truthfulqa_mc import build_mc_prompt, load_truthfulqa_samples


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for dataset inspection."""
    parser = argparse.ArgumentParser(
        description="Inspect normalized TruthfulQA-MC samples and prompts."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="Path to a TruthfulQA-style CSV file.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Number of samples to print.",
    )
    return parser.parse_args()


def main() -> None:
    """Load a CSV file and print a small prompt inspection preview."""
    args = parse_args()
    if args.limit <= 0:
        raise ValueError("--limit must be a positive integer.")

    samples = load_truthfulqa_samples(args.csv)
    print(f"Loaded {len(samples)} normalized samples from {args.csv}")

    for index, sample in enumerate(samples[: args.limit], start=1):
        print("")
        print(f"Sample {index}")
        print(f"Question: {sample.question}")
        print(f"Category: {sample.category or 'N/A'}")
        print(f"Best answer: {sample.best_answer}")
        print(f"Correct answers: {len(sample.correct_answers)}")
        print(f"Incorrect answers: {len(sample.incorrect_answers)}")
        print("Prompt:")
        print(build_mc_prompt(sample))


if __name__ == "__main__":
    main()