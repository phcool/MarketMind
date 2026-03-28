"""
GRPO training for Qwen2.5-7B-Instruct on quotes_7d CSV (prompt + pct_change).

Rollout uses vLLM by default (colocate with training on the same GPUs; TRL + ZeRO-3).
Reward: last line of completion -> float (strip %); reward = exp(-abs(pred-label)/100); else 0.

Launch (8 GPUs, DeepSpeed ZeRO-3 + vLLM colocate):
  cd <repo_root> && bash train/run_grpo_8gpu.sh

Requires: pip install -r train/requirements.txt  (includes trl[vllm])
"""

from __future__ import annotations

import argparse
import math
import os
import re
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_TRAIN_CSV = REPO_ROOT / "train" / "dataset" / "quotes_7d_pre2026_dataset.csv"

_FLOAT_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def extract_last_line_float(text: str) -> float | None:
    """Take the last non-empty line, strip %, extract first float token."""
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


def _completion_to_text(comp) -> str:
    if isinstance(comp, str):
        return comp
    if isinstance(comp, list) and comp:
        first = comp[0]
        if isinstance(first, dict) and "content" in first:
            return str(first.get("content", ""))
    return str(comp)


def pct_change_exp_reward(
    completions: list,
    pct_change: list[float],
    log_metric=None,
    **kwargs,
):
    """
    TRL GRPO reward: exp(-abs(pred - gt) / 100). pred from last line of completion.
    Dataset column must be named `pct_change` (passed through by GRPOTrainer).
    """
    rewards: list[float] = []
    ok = 0
    for comp, gt in zip(completions, pct_change):
        pred = extract_last_line_float(_completion_to_text(comp))
        if pred is None:
            rewards.append(0.0)
            continue
        ok += 1
        diff = abs(pred - float(gt))
        rewards.append(math.exp(-(diff / 100.0)))
    if log_metric and rewards:
        log_metric("pct_parse_success_rate", ok / len(rewards))
    return rewards


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="GRPO train Qwen2.5-7B on quotes 7d prompts.")
    p.add_argument(
        "--model_name_or_path",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="HF model id or local path.",
    )
    p.add_argument(
        "--train_file",
        type=str,
        default=str(DEFAULT_TRAIN_CSV),
        help="CSV with columns: prompt, pct_change",
    )
    p.add_argument("--output_dir", type=str, default=str(REPO_ROOT / "train" / "outputs" / "grpo_qwen25_7b"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_samples", type=int, default=-1, help="If >0, truncate dataset for debugging.")
    p.add_argument("--num_train_epochs", type=float, default=1.0)
    p.add_argument("--max_steps", type=int, default=-1, help="If >=0, overrides epochs.")
    p.add_argument("--learning_rate", type=float, default=1e-6)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--num_generations", type=int, default=8, help="GRPO group size G.")
    p.add_argument(
        "--max_prompt_length",
        type=int,
        default=6144,
        help="Left-truncate templated prompts to this many tokens; also sets vLLM context to max_prompt+completion.",
    )
    p.add_argument("--max_completion_length", type=int, default=128)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--beta", type=float, default=0.0, help="KL coefficient; 0 disables KL in GRPO.")
    p.add_argument(
        "--report_to",
        type=str,
        default="none",
        help="e.g. none, tensorboard, wandb (comma-separated for multiple).",
    )
    p.add_argument(
        "--no_vllm",
        action="store_true",
        help="Disable vLLM and use HF generate() for rollouts (slower).",
    )
    p.add_argument(
        "--vllm_mode",
        type=str,
        default="colocate",
        choices=("colocate", "server"),
        help="colocate: vLLM in trainer process (default for 8-GPU single node). "
        "server: connect to trl vllm-serve (set host/port or base URL).",
    )
    p.add_argument(
        "--vllm_gpu_memory_utilization",
        type=float,
        default=0.35,
        help="vLLM GPU memory fraction when vllm_mode=colocate (tune if OOM).",
    )
    p.add_argument(
        "--vllm_tensor_parallel_size",
        type=int,
        default=1,
        help="vLLM tensor parallel size when vllm_mode=colocate.",
    )
    p.add_argument(
        "--vllm_server_base_url",
        type=str,
        default="",
        help="If set (e.g. http://127.0.0.1:8000), used when vllm_mode=server.",
    )
    p.add_argument("--vllm_server_host", type=str, default="127.0.0.1")
    p.add_argument("--vllm_server_port", type=int, default=8000)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    raw = load_dataset("csv", data_files=args.train_file, split="train")
    if args.max_samples and args.max_samples > 0:
        raw = raw.select(range(min(args.max_samples, len(raw))))

    def apply_chat_template_row(example: dict) -> dict:
        user_text = example["prompt"]
        messages = [{"role": "user", "content": user_text}]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        # TRL >= 0.29 removed GRPOConfig.max_prompt_length; truncate here for both vLLM and HF generate.
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        max_pl = args.max_prompt_length
        if len(prompt_ids) > max_pl:
            prompt_ids = prompt_ids[-max_pl:]
            prompt = tokenizer.decode(prompt_ids, skip_special_tokens=False)
        return {
            "prompt": prompt,
            "pct_change": float(example["pct_change"]),
        }

    train_ds = raw.map(apply_chat_template_row, remove_columns=list(raw.column_names))
    train_ds = train_ds.shuffle(seed=args.seed)

    parts = [x.strip() for x in args.report_to.split(",") if x.strip()]
    if not parts or parts == ["none"]:
        report_to_val: str | list[str] = "none"
    elif len(parts) == 1:
        report_to_val = parts[0]
    else:
        report_to_val = parts

    training_kwargs: dict = dict(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        bf16=True,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        beta=args.beta,
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,
        report_to=report_to_val,
        seed=args.seed,
    )
    if args.max_steps > 0:
        training_kwargs["max_steps"] = args.max_steps
    else:
        training_kwargs["num_train_epochs"] = args.num_train_epochs

    use_vllm = not args.no_vllm
    training_kwargs["use_vllm"] = use_vllm
    if use_vllm:
        training_kwargs["vllm_mode"] = args.vllm_mode
        training_kwargs["vllm_gpu_memory_utilization"] = args.vllm_gpu_memory_utilization
        training_kwargs["vllm_tensor_parallel_size"] = args.vllm_tensor_parallel_size
        training_kwargs["vllm_max_model_length"] = args.max_prompt_length + args.max_completion_length
        if args.vllm_mode == "server":
            if args.vllm_server_base_url.strip():
                training_kwargs["vllm_server_base_url"] = args.vllm_server_base_url.strip()
            else:
                training_kwargs["vllm_server_host"] = args.vllm_server_host
                training_kwargs["vllm_server_port"] = args.vllm_server_port

    training_args = GRPOConfig(**training_kwargs)

    trainer = GRPOTrainer(
        model=args.model_name_or_path,
        args=training_args,
        reward_funcs=pct_change_exp_reward,
        train_dataset=train_ds,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
