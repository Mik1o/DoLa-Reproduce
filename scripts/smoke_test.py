"""Minimal smoke-test entry point for the scaffolded project."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import ensure_output_dir, load_yaml_config


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the smoke test."""
    parser = argparse.ArgumentParser(description="Run a scaffold smoke test.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "debug_small.yaml",
        help="Path to a YAML config file.",
    )
    return parser.parse_args()


def main() -> None:
    """Load config and print the actions future experiments will perform."""
    args = parse_args()
    config = load_yaml_config(args.config)
    output_dir = ensure_output_dir(config.get("output_dir", PROJECT_ROOT / "outputs" / "smoke"))

    print("[smoke_test] Scaffold is wired correctly.")
    print(f"[smoke_test] Config file: {args.config}")
    print(f"[smoke_test] Output directory: {output_dir}")
    print("[smoke_test] Future step: load model, dataset, and evaluation pipeline.")


if __name__ == "__main__":
    main()
