"""Preview a real TruthfulQA CSV before running model-based subset evaluation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.truthfulqa_mc import build_mc_prompt, load_truthfulqa_csv, normalize_truthfulqa_row



def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for real TruthfulQA CSV inspection."""
    parser = argparse.ArgumentParser(
        description="Inspect a real TruthfulQA.csv file before running subset evaluation."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="Path to a real TruthfulQA CSV file.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Number of normalized samples to preview.",
    )
    return parser.parse_args()



def main() -> None:
    """Load a real TruthfulQA CSV and print a compact preview of the first rows."""
    args = parse_args()
    if args.limit <= 0:
        raise ValueError("--limit must be a positive integer.")

    dataframe = load_truthfulqa_csv(args.csv)
    preview_rows = min(len(dataframe), args.limit)

    print(f"[inspect_truthfulqa_real_csv] CSV: {args.csv}")
    print(f"[inspect_truthfulqa_real_csv] Total rows: {len(dataframe)}")
    print(f"[inspect_truthfulqa_real_csv] Previewing first {preview_rows} rows")

    for row_index, (_, row) in enumerate(dataframe.head(args.limit).iterrows(), start=1):
        sample = normalize_truthfulqa_row(row)
        prompt = build_mc_prompt(sample)
        print(f"\n[{row_index}] question: {sample.question}")
        print(f"best_answer: {sample.best_answer}")
        print(f"correct_answers: {len(sample.correct_answers)}")
        print(f"incorrect_answers: {len(sample.incorrect_answers)}")
        print("prompt preview:")
        print(prompt)


if __name__ == "__main__":
    main()
