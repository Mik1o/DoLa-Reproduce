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
    "scripts.hf_audit_dynamic_collapse",
    "scripts.hf_audit_precision_sensitivity",
    "scripts.hf_tune_bucket_truthfulqa_cv",
    "scripts.hf_audit_truthfulqa_mc_kernel",
    "scripts.hf_diagnose_official_parity",
    "scripts.hf_probe_early_exit_oracle_parity",
    "scripts.hf_trace_scoring_path_consistency",
    "scripts.hf_compare_single_mc",
    "scripts.hf_eval_compare_subset",
    "scripts.hf_eval_mc_subset",
    "scripts.hf_score_single_mc",
    "scripts.hf_smoke_test",
    "scripts.hf_summarize_cross_model_baseline",
    "scripts.hf_sweep_premature_layers",
    "scripts.inspect_truthfulqa_data",
    "scripts.inspect_truthfulqa_real_csv",
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
