"""Minimal local environment checks for the scaffolded project."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import ensure_output_dir, load_yaml_config


def check_python_version() -> None:
    """Ensure the local interpreter is Python 3.12 or newer."""
    version = sys.version_info
    if version < (3, 12):
        raise RuntimeError(
            f"Python 3.12+ is required, but found {version.major}.{version.minor}.{version.micro}."
        )


def check_import(module_name: str) -> None:
    """Import a module and raise a clear error if it fails."""
    try:
        importlib.import_module(module_name)
    except Exception as error:  # pragma: no cover - explicit failure path
        raise RuntimeError(f"Failed to import '{module_name}': {error}") from error


def main() -> None:
    """Run the minimal local environment checks for this project."""
    check_python_version()

    modules_to_check = [
        "yaml",
        "pandas",
        "src.modeling",
        "src.generation",
        "src.dola_utils",
        "src.truthfulqa_mc",
        "src.metrics",
        "src.utils",
    ]
    for module_name in modules_to_check:
        check_import(module_name)

    config_path = PROJECT_ROOT / "configs" / "debug_small.yaml"
    config = load_yaml_config(config_path)
    output_dir = ensure_output_dir(
        config.get("output_dir", PROJECT_ROOT / "outputs" / "debug_small")
    )

    print("Environment check passed.")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Config: {config_path}")
    print(f"Output directory ready: {output_dir}")


if __name__ == "__main__":
    main()
