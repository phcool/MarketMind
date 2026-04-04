"""
Compare base Qwen2.5-7B-Instruct against the trained LoRA adapter with vLLM.

The script randomly samples prompts from the SFT dataset, applies the same chat template
used during training, then generates outputs from:
1. The original base model
2. The base model + trained LoRA adapter

Usage (from repo root):
  uv run python train/scripts/sft/eval_sft_vllm_compare.py

Examples:
  uv run python train/scripts/sft/eval_sft_vllm_compare.py --num_samples 3
  uv run python train/scripts/sft/eval_sft_vllm_compare.py --adapter_path /path/to/checkpoint-404
  uv run python train/scripts/sft/eval_sft_vllm_compare.py --tensor_parallel_size 2
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_DIR = SCRIPT_DIR.parents[1]
DEFAULT_DATASET = TRAIN_DIR / "dataset" / "quotes_7d_cot_from_batch.csv"
DEFAULT_ADAPTER_ROOT = Path("/nfs/hanpeng/huggingface/models/qwen2_5_sft_cot_lora")
DEFAULT_OUTPUT_JSON = TRAIN_DIR / "outputs" / "sft_vllm_compare.json"

FINAL_LABEL_RE = re.compile(r"^\s*([涨跌])\s*$")


def _find_latest_adapter_path(adapter_root: Path) -> Path:
    checkpoints = []
    for child in adapter_root.iterdir():
        if not child.is_dir() or not child.name.startswith("checkpoint-"):
            continue
        try:
            step = int(child.name.split("-", 1)[1])
        except ValueError:
            continue
        if (child / "adapter_model.safetensors").is_file():
            checkpoints.append((step, child))
    if checkpoints:
        checkpoints.sort()
        return checkpoints[-1][1]
    if (adapter_root / "adapter_model.safetensors").is_file():
        return adapter_root
    raise SystemExit(f"No LoRA adapter found under {adapter_root}")


def _load_rows(dataset_path: Path) -> list[dict[str, str]]:
    with dataset_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        expected = {"prompt", "completion"}
        actual = set(reader.fieldnames or [])
        missing = expected - actual
        if missing:
            raise SystemExit(f"Dataset missing required columns {sorted(missing)}; found {reader.fieldnames}")
        return list(reader)


def _extract_reference_label(completion: str) -> str | None:
    lines = [line.strip() for line in str(completion).splitlines() if line.strip()]
    for line in reversed(lines):
        match = FINAL_LABEL_RE.match(line)
        if match:
            return match.group(1)
    return None


def _build_templated_prompts(tokenizer, rows: list[dict[str, str]]) -> list[str]:
    prompts = []
    for row in rows:
        messages = [{"role": "user", "content": row["prompt"]}]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(prompt)
    return prompts


def _generate(
    llm: LLM,
    prompts: list[str],
    sampling_params: SamplingParams,
    *,
    lora_request: LoRARequest | None = None,
) -> list[str]:
    outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
    return [item.outputs[0].text for item in outputs]


def _print_console_summary(results: list[dict[str, str]], output_json: Path) -> None:
    print(f"Wrote comparison JSON to {output_json}")
    print()
    for i, row in enumerate(results, start=1):
        print("=" * 100)
        print(f"Sample {i}")
        print(f"Reference label: {row['reference_label']}")
        print("-" * 100)
        print("Prompt:")
        print(row["prompt"])
        print("-" * 100)
        print("Base model output:")
        print(row["base_output"])
        print("-" * 100)
        print("Fine-tuned LoRA output:")
        print(row["finetuned_output"])
        print()


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare base Qwen and SFT LoRA adapter with vLLM.")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--adapter_root", type=str, default=str(DEFAULT_ADAPTER_ROOT))
    parser.add_argument("--adapter_path", type=str, default="")
    parser.add_argument("--dataset", type=str, default=str(DEFAULT_DATASET))
    parser.add_argument("--output_json", type=str, default=str(DEFAULT_OUTPUT_JSON))
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--max_lora_rank", type=int, default=64)
    return parser


def main() -> None:
    args = build_argparser().parse_args()

    dataset_path = Path(args.dataset).expanduser().resolve()
    adapter_root = Path(args.adapter_root).expanduser().resolve()
    adapter_path = (
        Path(args.adapter_path).expanduser().resolve()
        if args.adapter_path.strip()
        else _find_latest_adapter_path(adapter_root)
    )
    output_json = Path(args.output_json).expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)

    rows = _load_rows(dataset_path)
    if not rows:
        raise SystemExit(f"No rows found in dataset: {dataset_path}")
    num_samples = min(args.num_samples, len(rows))
    rng = random.Random(args.seed)
    sampled_rows = rng.sample(rows, num_samples)

    tokenizer = AutoTokenizer.from_pretrained(str(adapter_path), use_fast=True, trust_remote_code=True)
    templated_prompts = _build_templated_prompts(tokenizer, sampled_rows)

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
    )

    llm = LLM(
        model=args.base_model,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enable_lora=True,
        max_lora_rank=args.max_lora_rank,
    )

    base_outputs = _generate(llm, templated_prompts, sampling_params)
    finetuned_outputs = _generate(
        llm,
        templated_prompts,
        sampling_params,
        lora_request=LoRARequest("sft_adapter", 1, str(adapter_path)),
    )

    results: list[dict[str, str | int | None]] = []
    for idx, (row, base_output, finetuned_output) in enumerate(
        zip(sampled_rows, base_outputs, finetuned_outputs),
        start=1,
    ):
        results.append(
            {
                "sample_id": idx,
                "reference_label": _extract_reference_label(row["completion"]),
                "prompt": row["prompt"],
                "reference_completion": row["completion"],
                "base_output": base_output,
                "finetuned_output": finetuned_output,
            }
        )

    payload = {
        "meta": {
            "base_model": args.base_model,
            "adapter_path": str(adapter_path),
            "dataset": str(dataset_path),
            "num_samples": num_samples,
            "seed": args.seed,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
            "tensor_parallel_size": args.tensor_parallel_size,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_model_len": args.max_model_len,
            "max_lora_rank": args.max_lora_rank,
        },
        "results": results,
    }
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _print_console_summary(results, output_json)


if __name__ == "__main__":
    main()
