---
base_model: Qwen/Qwen2.5-7B-Instruct
library_name: peft
pipeline_tag: text-generation
license: apache-2.0
tags:
- lora
- peft
- qwen2.5
- finance
- chinese
- stock-prediction
---

# MarketMind Qwen2.5 SFT CoT LoRA

This repository contains a LoRA adapter trained on the `Qwen/Qwen2.5-7B-Instruct` base model for the `MarketMind` project.

## What is included

- Final LoRA adapter weights
- Tokenizer files needed for inference
- A lightweight publishable package without intermediate training checkpoints

This is an adapter repository, not a merged full model.

## Base model

- Base model: `Qwen/Qwen2.5-7B-Instruct`
- Adapter type: `LoRA`
- Task type: `CAUSAL_LM`

## LoRA configuration

- Rank `r`: `64`
- `lora_alpha`: `128`
- `lora_dropout`: `0.05`
- Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`

## Training snapshot

From the local training artifacts:

- Train examples: `25838`
- Eval examples: `261`
- Train loss: `0.4362`
- Eval loss: `0.3558`
- Epochs: `2.0`

## Intended use

This adapter is intended for research and experimentation on Chinese financial reasoning, stock-related text understanding, and downstream prediction prompts built on top of market data plus summarized news and reports.

## Example usage with PEFT

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_id = "Qwen/Qwen2.5-7B-Instruct"
adapter_id = "FAKEPHCOOL/marketmind-qwen2_5_sft_cot_lora"

tokenizer = AutoTokenizer.from_pretrained(adapter_id, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(base_model_id, trust_remote_code=True)
model = PeftModel.from_pretrained(base_model, adapter_id)
```

## Notes

- The repository intentionally excludes intermediate training checkpoints and local cache directories.
- Please review outputs carefully before using them in any high-stakes financial setting.
