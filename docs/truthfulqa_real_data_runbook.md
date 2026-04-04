# TruthfulQA Real Data Runbook

1. Put the real `TruthfulQA.csv` file at `data/truthfulqa/TruthfulQA.csv`.
2. Preview and validate the file first:

```bash
python scripts/inspect_truthfulqa_real_csv.py --csv data/truthfulqa/TruthfulQA.csv --limit 3
```

3. Then run the small real-data subset compare:

```bash
python scripts/hf_eval_compare_subset.py --config configs/mistral7b_truthfulqa_real_subset.yaml
```

4. Do not start with a full TruthfulQA run yet.
