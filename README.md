# DoLa Baseline Research Scaffold

This repository is a lightweight Python project scaffold for studying a DoLa
baseline and related TruthfulQA multiple-choice evaluation workflows.

## Goal

The project is being organized so future code can move smoothly across:

- a Windows laptop for development,
- a remote WSL2 machine with a GPU for formal experiments,
- and a future Linux server environment.

## Current Stage

The project now includes lightweight scaffolding, local environment checks,
TruthfulQA-MC data normalization, a tiny Hugging Face vanilla generation
smoke test, single-sample candidate scoring, a small vanilla subset
evaluation loop with averaged MC metrics, a single-sample vanilla vs
DoLa-style comparison prototype, and a small-subset vanilla vs DoLa-style
comparison workflow. It also supports switching from the tiny-random model to
TinyLlama for a more realistic small-model comparison setup, and now includes
7B-ready config templates plus WSL run guidance for `mistralai/Mistral-7B-v0.1`.
It still does **not** implement 7B full experiments, dynamic layer selection,
or the full DoLa paper setup.

## Directory Layout

```text
.
|-- configs/
|-- data/
|-- docs/
|-- outputs/
|-- scripts/
|-- src/
`-- tests/
```

- `src/`: future research code for model loading, generation, metrics, and dataset helpers.
- `scripts/`: minimal command-line entry points for smoke tests and future experiment runs.
- `configs/`: YAML configuration examples with a consistent field layout.
- `tests/`: lightweight tests, currently focused on import-level checks.
- `data/`: local datasets and intermediate data files.
- `docs/`: short runbooks for moving experiments onto other machines.
- `outputs/`: run outputs, logs, and artifacts from future experiments.

## What Exists Right Now

- module boundaries and function signatures,
- minimal YAML config loading,
- output directory creation,
- import-safe script entry points,
- TruthfulQA-MC data loading and prompt construction,
- one tiny Hugging Face causal LM smoke path,
- single-sample candidate scoring plus MC1/MC2/MC3,
- small-subset vanilla evaluation with JSON result saving,
- single-sample vanilla vs DoLa-style comparison,
- small-subset vanilla vs DoLa-style comparison with saved summaries,
- configuration-driven switching between tiny-random and TinyLlama,
- 7B-ready config templates for WSL smoke and compare runs.

## What Comes Next

Later iterations will add:

- broader DoLa evaluation flows,
- larger TruthfulQA experiment loops,
- larger baseline experiment plumbing.

## Minimal Setup

Use Python 3.12 and install the current basic dependencies:

```bash
pip install -r requirements.txt
```

Run the current smoke test:

```bash
python scripts/smoke_test.py --config configs/debug_small.yaml
```

## Local Development Setup

Create the conda environment:

```bash
conda env create -f environment.yml
```

Activate the environment:

```bash
conda activate dola-dev
```

Run the minimal environment check:

```bash
python scripts/check_env.py
```

Run the current smoke test:

```bash
python scripts/smoke_test.py --config configs/debug_small.yaml
```

Run the tests:

```bash
pytest
```

For local tests, the project currently keeps imports stable by using the
repository root together with `tests/conftest.py` to add that root to
`sys.path` during pytest startup. The project is intentionally **not** packaged
or installed yet, because this stage is still a lightweight development
scaffold.

## TruthfulQA-MC Data Preview

The project includes the data-input side of TruthfulQA-MC: CSV loading,
row normalization into a lightweight internal sample structure, and a simple
multiple-choice prompt builder for inspection.

Run the data inspection script with:

```bash
python scripts/inspect_truthfulqa_data.py --csv tests/fixtures/truthfulqa_sample.csv --limit 2
```

## Minimal HF Smoke Test

Install the extra model-stage dependencies with:

```bash
pip install -r requirements-model.txt
```

Then run:

```bash
python scripts/hf_smoke_test.py --config configs/hf_tiny_smoke.yaml
```

## Single-Sample MC Scoring

The project supports scoring one TruthfulQA-style sample with a tiny causal
LM by summing continuation log-probabilities for true and false candidates, then
computing MC1, MC2, and MC3.

Run the single-sample scoring script with:

```bash
python scripts/hf_score_single_mc.py --config configs/hf_tiny_score_single.yaml
```

## Small-Subset MC Evaluation

The project supports running vanilla MC evaluation over the first N samples
from a TruthfulQA-style CSV, averaging MC1/MC2/MC3, and saving results to:

- `sample_results.jsonl`
- `summary.json`

Run the subset evaluation script with:

```bash
python scripts/hf_eval_mc_subset.py --config configs/hf_tiny_eval_subset.yaml
```

## Single-Sample Vanilla vs DoLa-Style Compare

The project supports a minimal DoLa-style prototype for one sample: one
premature layer, one mature final layer, and contrastive candidate scoring via
`mature logits - premature logits`.

Run the compare script with:

```bash
python scripts/hf_compare_single_mc.py --config configs/hf_tiny_compare_single.yaml
```

## Small-Subset Vanilla vs DoLa-Style Compare

The project supports comparing vanilla and DoLa-style MC scoring over the
first N samples of a small subset. The script writes:

- `compare_sample_results.jsonl`
- `compare_summary.json`

Run it with:

```bash
python scripts/hf_eval_compare_subset.py --config configs/hf_tiny_compare_subset.yaml
```

## TinyLlama Configs

In addition to the tiny-random fallback configs, the project now includes a
more realistic small public model setup based on `TinyLlama/TinyLlama-1.1B-Chat-v1.0`.
Use these configs to reuse the same scripts without changing the core logic:

```bash
python scripts/hf_smoke_test.py --config configs/tinyllama_smoke.yaml
python scripts/hf_compare_single_mc.py --config configs/tinyllama_compare_single.yaml
python scripts/hf_eval_compare_subset.py --config configs/tinyllama_compare_subset.yaml
```

## WSL 7B Baseline Prep

The project now also includes 7B-ready config templates for
`mistralai/Mistral-7B-v0.1`. These are meant for the WSL2 + RTX 5080 machine,
not for full experiments yet:

```bash
python scripts/hf_smoke_test.py --config configs/mistral7b_smoke.yaml
python scripts/hf_compare_single_mc.py --config configs/mistral7b_compare_single.yaml
python scripts/hf_eval_compare_subset.py --config configs/mistral7b_compare_subset.yaml
```

For the step-by-step WSL setup order, see `docs/wsl_7b_runbook.md`.
This is still baseline-prep only: no full TruthfulQA run, no 7B layer sweep,
and no final formal experiment pipeline yet.

## Premature-Layer Sweep

The project supports scanning several fixed premature layers over a small
subset and saving the layer-by-layer summaries to:

- `layer_sweep_results.jsonl`
- `best_layer_summary.json`

Run it with:

```bash
python scripts/hf_sweep_premature_layers.py --config configs/tinyllama_sweep_subset.yaml
```

This is an analysis-only sweep. It does **not** implement dynamic layer
selection, quantization, or the full DoLa paper variant.
