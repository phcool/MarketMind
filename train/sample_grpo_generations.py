"""
Sample N prompts from the GRPO CSV with the same chat template + truncation as train_grpo_qwen.py,
run HF generate(), and write JSON for inspecting completion format vs reward parsing.

Usage (from repo root):
  python train/sample_grpo_generations.py --output_json train/outputs/grpo_sample_generations.json

Requires: transformers, torch, datasets (GPU recommended for 7B).
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_TRAIN_CSV = REPO_ROOT / "train" / "dataset" / "quotes_7d_pre2026_dataset.csv"

_FLOAT_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def extract_last_line_float(text: str) -> float | None:
    if not text or not str(text).strip():
        return None
    lines = [ln.strip() for ln in str(text).strip().splitlines() if ln.strip()]
    if not lines:
        return None
    last = lines[-1].replace("%", "").strip()
    m = _FLOAT_RE.search(last)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def build_templated_prompt(
    tokenizer,
    user_text: str,
    max_prompt_length: int,
) -> tuple[str, int]:
    messages = [{"role": "user", "content": user_text}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    n_before = len(prompt_ids)
    if len(prompt_ids) > max_prompt_length:
        prompt_ids = prompt_ids[-max_prompt_length:]
        prompt = tokenizer.decode(prompt_ids, skip_special_tokens=False)
    return prompt, n_before


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--train_file", type=str, default=str(DEFAULT_TRAIN_CSV))
    p.add_argument("--output_json", type=str, default=str(SCRIPT_DIR / "outputs" / "grpo_sample_generations.json"))
    p.add_argument("--num_prompts", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_prompt_length", type=int, default=6144)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=1.0)
    args = p.parse_args()

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    raw = load_dataset("csv", data_files=args.train_file, split="train")
    n = min(args.num_prompts, len(raw))
    # Match training: shuffle then take first n (same seed as default training)
    raw = raw.shuffle(seed=args.seed).select(range(n))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    torch.manual_seed(args.seed)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if not torch.cuda.is_available():
        model = model.to("cpu")

    samples: list[dict] = []
    for i in range(n):
        row = raw[i]
        user_text = row["prompt"]
        label = float(row["pct_change"])
        templated, n_tok_before_trunc = build_templated_prompt(
            tokenizer, user_text, args.max_prompt_length
        )
        n_tok_after = len(tokenizer.encode(templated, add_special_tokens=False))

        inputs = tokenizer(templated, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                pad_token_id=tokenizer.pad_token_id,
            )
        gen_ids = out[0, inputs["input_ids"].shape[1] :]
        completion = tokenizer.decode(gen_ids, skip_special_tokens=True)

        pred = extract_last_line_float(completion)
        if pred is None:
            reward = None
        else:
            reward = math.exp(-abs(pred - label) / 100.0)

        samples.append(
            {
                "index": i,
                "pct_change_label": label,
                "raw_user_prompt_char_len": len(user_text),
                "raw_user_prompt_head_400_chars": user_text[:400],
                "raw_user_prompt_tail_400_chars": user_text[-400:] if len(user_text) > 400 else "",
                "templated_prompt_token_count_before_trunc": n_tok_before_trunc,
                "templated_prompt_token_count_after_trunc": n_tok_after,
                "templated_prompt_head_500_chars": templated[:500],
                "templated_prompt_tail_500_chars": templated[-500:] if len(templated) > 500 else "",
                "completion": completion,
                "parsed_last_line_float": pred,
                "reward_if_parsed_exp_neg_abs_over_100": reward,
            }
        )

    payload = {
        "meta": {
            "model_name_or_path": args.model_name_or_path,
            "train_file": args.train_file,
            "num_prompts": n,
            "seed": args.seed,
            "max_prompt_length": args.max_prompt_length,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "note": "Same chat template + left token truncation as train_grpo_qwen.py; reward matches pct_change_exp_reward.",
        },
        "samples": samples,
    }

    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {n} samples to {out_path}")


if __name__ == "__main__":
    main()
