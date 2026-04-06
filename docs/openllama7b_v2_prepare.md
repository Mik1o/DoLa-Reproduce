# OpenLLaMA 7B v2 Prepare Notes

## Model Choice

`openlm-research/open_llama_7b_v2` is an open Apache-2.0 LLaMA-family model.
For the current baseline stage, it is the practical replacement for the blocked
`meta-llama/Llama-2-7b-hf` route.

## Tokenizer Note

For this model, prefer:

- `use_fast_tokenizer: false`

This keeps tokenization closer to the model card recommendation and avoids fast-tokenizer edge cases.
The project configs also pin `tokenizer_class: "LlamaTokenizer"` for this route.

## Direct Repo-ID Loading

You can load directly from the Hub with:

```bash
python scripts/hf_smoke_test.py --config configs/openllama7b_v2_truthfulqa_real_smoke_4bit.yaml
```

The config already uses `model_name: openlm-research/open_llama_7b_v2`.

## Download to a Local Directory

Download to a local path with either command style:

```bash
hf download openlm-research/open_llama_7b_v2 --local-dir /home/yifei/DoLa-Reproduce/models/open_llama_7b_v2
```

```bash
huggingface-cli download openlm-research/open_llama_7b_v2 --local-dir /home/yifei/DoLa-Reproduce/models/open_llama_7b_v2
```

Then point config `model_name` at that local directory.

## Laptop 4-bit Smoke Test

Install model dependencies first:

```bash
pip install -r requirements-model.txt
```

Then run:

```bash
python scripts/hf_smoke_test.py --config configs/openllama7b_v2_truthfulqa_real_smoke_4bit.yaml
```

## WSL 16GB Follow-up

After the model is available locally, run the oracle-style parity probe:

```bash
python scripts/hf_probe_early_exit_oracle_parity.py --config configs/openllama7b_v2_truthfulqa_real_probe_oracle_parity.yaml
```

Only after the load / hidden-state / logits path is stable should you reuse the current compare or subset scripts.

## If Repo-ID Loading Still Fails on Tokenizer Parsing

If you still see errors mentioning `tiktoken` or `Error parsing line ... tokenizer.model`,
that usually means the cached tokenizer files are bad for the current environment.
In that case, skip repo-id loading and explicitly download a fresh local copy:

```bash
hf download openlm-research/open_llama_7b_v2 --local-dir /home/yifei/DoLa-Reproduce/models/open_llama_7b_v2
```

Then keep the config on:

- `use_fast_tokenizer: false`
- `tokenizer_class: "LlamaTokenizer"`
- `model_name: "/home/yifei/DoLa-Reproduce/models/open_llama_7b_v2"`
