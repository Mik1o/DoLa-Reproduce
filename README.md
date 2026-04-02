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
TruthfulQA-MC data normalization, and one tiny Hugging Face vanilla generation
smoke test. It still does **not** implement DoLa, TruthfulQA scoring, or any
7B experiment workflow.

## Directory Layout

```text
.
|-- configs/
|-- data/
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
- `outputs/`: run outputs, logs, and artifacts from future experiments.

## What Exists Right Now

- module boundaries and function signatures,
- minimal YAML config loading,
- output directory creation,
- import-safe script entry points,
- TruthfulQA-MC data loading and prompt construction,
- one tiny Hugging Face causal LM smoke path.

## What Comes Next

Later iterations will add:

- DoLa-specific generation logic,
- TruthfulQA candidate scoring and metrics,
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
multiple-choice prompt builder for inspection. This still does **not** include
model-based scoring.

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

This stage only covers a tiny Hugging Face causal LM load plus one vanilla
`generate` call. It still does **not** implement DoLa, TruthfulQA scoring, or
any 7B baseline experiment flow.