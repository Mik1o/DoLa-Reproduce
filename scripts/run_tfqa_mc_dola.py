"""CLI scaffold for future TruthfulQA-MC DoLa baseline runs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dola_utils import validate_dola_layers
from src.utils import ensure_output_dir, load_yaml_config


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the DoLa baseline script."""
    parser = argparse.ArgumentParser(
        description="Prepare a future TruthfulQA-MC DoLa baseline run."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "mistral7b_tfqa_dola.yaml",
        help="Path to a YAML config file.",
    )
    return parser.parse_args()


def main() -> None:
    """Load config and describe the future DoLa evaluation run."""
    args = parse_args()
    config = load_yaml_config(args.config)
    output_dir = ensure_output_dir(config.get("output_dir", PROJECT_ROOT / "outputs" / "dola"))
    dola_layers = validate_dola_layers(config.get("dola_layers"))

    print("[run_tfqa_mc_dola] This is a scaffold-only entry point.")
    print(f"[run_tfqa_mc_dola] Model name: {config.get('model_name', 'N/A')}")
    print(f"[run_tfqa_mc_dola] Device: {config.get('device', 'N/A')}")
    print(f"[run_tfqa_mc_dola] Data path: {config.get('data_path', 'N/A')}")
    print(f"[run_tfqa_mc_dola] DoLa layers: {dola_layers}")
    print(f"[run_tfqa_mc_dola] Output directory: {output_dir}")
    print("[run_tfqa_mc_dola] Future step: run DoLa decoding on TruthfulQA-MC.")


if __name__ == "__main__":
    main()
