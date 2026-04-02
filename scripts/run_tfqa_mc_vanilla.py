"""CLI scaffold for future TruthfulQA-MC vanilla baseline runs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import ensure_output_dir, load_yaml_config


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the vanilla baseline script."""
    parser = argparse.ArgumentParser(
        description="Prepare a future TruthfulQA-MC vanilla baseline run."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "mistral7b_tfqa_vanilla.yaml",
        help="Path to a YAML config file.",
    )
    return parser.parse_args()


def main() -> None:
    """Load config and describe the future vanilla evaluation run."""
    args = parse_args()
    config = load_yaml_config(args.config)
    output_dir = ensure_output_dir(config.get("output_dir", PROJECT_ROOT / "outputs" / "vanilla"))

    print("[run_tfqa_mc_vanilla] This is a scaffold-only entry point.")
    print(f"[run_tfqa_mc_vanilla] Model name: {config.get('model_name', 'N/A')}")
    print(f"[run_tfqa_mc_vanilla] Device: {config.get('device', 'N/A')}")
    print(f"[run_tfqa_mc_vanilla] Data path: {config.get('data_path', 'N/A')}")
    print(f"[run_tfqa_mc_vanilla] Output directory: {output_dir}")
    print("[run_tfqa_mc_vanilla] Future step: run vanilla decoding on TruthfulQA-MC.")


if __name__ == "__main__":
    main()
