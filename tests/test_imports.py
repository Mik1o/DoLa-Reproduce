"""Import smoke tests for the scaffolded project modules."""

from __future__ import annotations

import importlib


CORE_MODULES = [
    "src.dola_utils",
    "src.generation",
    "src.metrics",
    "src.modeling",
    "src.truthfulqa_mc",
    "src.utils",
]

SCRIPT_MODULES = [
    "scripts.check_env",
    "scripts.hf_smoke_test",
    "scripts.inspect_truthfulqa_data",
    "scripts.run_tfqa_mc_dola",
    "scripts.run_tfqa_mc_vanilla",
    "scripts.smoke_test",
]


def test_import_core_modules() -> None:
    """Ensure the main source modules are importable."""
    for module_name in CORE_MODULES:
        assert importlib.import_module(module_name) is not None


def test_import_script_modules() -> None:
    """Ensure the script entry points are importable without side effects."""
    for module_name in SCRIPT_MODULES:
        assert importlib.import_module(module_name) is not None