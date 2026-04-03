# WSL 7B Runbook

This runbook is the minimum path for trying the 7B baseline on a WSL2 machine with an RTX 5080.

## Recommended order

1. Install the base Python dependencies.
2. Install a CUDA-enabled PyTorch build.
3. Install the model-stage dependencies.
4. Run the 7B smoke test.
5. Run the single-sample compare.
6. Run the tiny subset compare.

## Suggested commands

```bash
conda env create -f environment.yml
conda activate dola-dev
pip install -r requirements.txt
```

Install CUDA PyTorch first, using the official command that matches the WSL CUDA toolkit on the machine. For example:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

Then install the remaining model dependencies:

```bash
pip install -r requirements-model.txt
```

Run a small smoke test before any evaluation:

```bash
python scripts/hf_smoke_test.py --config configs/mistral7b_smoke.yaml
```

Then run one compare sample:

```bash
python scripts/hf_compare_single_mc.py --config configs/mistral7b_compare_single.yaml
```

Then run a tiny subset compare:

```bash
python scripts/hf_eval_compare_subset.py --config configs/mistral7b_compare_subset.yaml
```

## Notes

- Start with `max_samples: 1` or `2` until loading and scoring are stable.
- Do not start with full TruthfulQA runs yet.
- Do not start with premature-layer sweep yet.
- This stage is only for confirming 7B loading and small compare runs on WSL.
