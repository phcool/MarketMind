"""
GRPO training for Qwen2.5-7B-Instruct on quotes_7d CSV (prompt + pct_change).

Rollout uses vLLM by default (colocate with training on the same GPUs; TRL + ZeRO-3).
Reward: parse last line as `pct_change_prediction: <float>%` (or fallback: bare number); reward = exp(-abs(pred-label)/100); else 0.

Launch (8 GPUs, DeepSpeed ZeRO-3 + vLLM colocate):
  cd <repo_root> && bash train/run_grpo_8gpu.sh

Checkpointing: --resume_from_checkpoint auto (default) continues from the latest checkpoint-* under
output_dir when present. With validation CSV, eval runs every --eval_steps; load_best_model_at_end uses
eval_loss; --save_total_limit rotates checkpoints while keeping the best-by-eval-loss save.

Requires: pip install -r train/requirements.txt  (includes trl[vllm])
"""

from __future__ import annotations

import argparse
import logging
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
DEFAULT_VAL_CSV = REPO_ROOT / "train" / "dataset" / "quotes_7d_val_20260101_20260328_dataset.csv"

_LOG = logging.getLogger(__name__)

_FLOAT_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
# Last line: "pct_change_prediction: -1.25%" (dataset prompt format)
_PRED_LINE_RE = re.compile(
    r"^\s*pct_change_prediction\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*%?\s*$",
    re.IGNORECASE,
)


def extract_pct_change_prediction(text: str) -> float | None:
    """
    Match dataset output contract: last non-empty line is
    `pct_change_prediction: <float>%` (optional %). Fallback: last line is a lone number.
    """
    if not text or not str(text).strip():
        return None
    lines = [ln.strip() for ln in str(text).strip().splitlines() if ln.strip()]
    if not lines:
        return None
    last = lines[-1]
    m = _PRED_LINE_RE.match(last)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    last_no_pct = last.replace("%", "").strip()
    m2 = _FLOAT_RE.search(last_no_pct)
    if not m2:
        return None
    try:
        return float(m2.group(0))
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


def _resolve_resume_from_checkpoint(resume: str, output_dir: str) -> bool | str | None:
    """Return value for Trainer.train(resume_from_checkpoint=...)."""
    r = resume.strip()
    if r.lower() in ("none", "false", "no", "0"):
        return None
    if r.lower() == "auto":
        out = Path(output_dir)
        if out.is_dir() and any(out.glob("checkpoint-*")):
            return True
        return None
    p = Path(r)
    if not p.is_absolute():
        cand = Path(output_dir) / r
        if cand.is_dir():
            return str(cand.resolve())
    elif p.is_dir():
        return str(p.resolve())
    raise SystemExit(
        f"--resume_from_checkpoint: not a directory: {r!r} "
        f"(relative paths are resolved under output_dir={output_dir!r})"
    )


def _map_quotes_dataset(
    csv_path: str,
    tokenizer,
    max_prompt_length: int,
    max_samples: int,
    *,
    shuffle: bool,
    seed: int,
):
    """Load CSV (prompt, pct_change), apply chat template + left token truncation."""
    ds = load_dataset("csv", data_files=csv_path, split="train")
    if max_samples and max_samples > 0:
        ds = ds.select(range(min(max_samples, len(ds))))

    def apply_chat_template_row(example: dict) -> dict:
        user_text = example["prompt"]
        messages = [{"role": "user", "content": user_text}]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        max_pl = max_prompt_length
        if len(prompt_ids) > max_pl:
            prompt_ids = prompt_ids[-max_pl:]
            prompt = tokenizer.decode(prompt_ids, skip_special_tokens=False)
        return {
            "prompt": prompt,
            "pct_change": float(example["pct_change"]),
        }

    mapped = ds.map(apply_chat_template_row, remove_columns=list(ds.column_names))
    if shuffle:
        mapped = mapped.shuffle(seed=seed)
    return mapped


def _fit_per_device_eval_batch_size(world_size: int, num_generations: int, requested: int) -> int:
    """
    GRPO requires (per_device_eval_batch_size * world_size) % num_generations == 0.
    Raises if world_size or num_generations is invalid.
    """
    if world_size < 1 or num_generations < 2:
        return max(1, requested)
    g = math.gcd(world_size, num_generations)
    minimum = num_generations // g
    b = max(requested, minimum)
    while (b * world_size) % num_generations != 0:
        b += 1
    return b


def pct_change_exp_reward(
    completions: list,
    pct_change: list[float],
    log_metric=None,
    **kwargs,
):
    """
    TRL GRPO reward: exp(-abs(pred - gt) / 100). pred from last line
    (`pct_change_prediction: ...` or bare number).
    Dataset column must be named `pct_change` (passed through by GRPOTrainer).
    """
    rewards: list[float] = []
    ok = 0
    for comp, gt in zip(completions, pct_change):
        pred = extract_pct_change_prediction(_completion_to_text(comp))
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
        help="Training CSV with columns: prompt, pct_change",
    )
    p.add_argument(
        "--eval_file",
        type=str,
        default=str(DEFAULT_VAL_CSV),
        help="Validation CSV (same schema as train). Set to 'none' to disable eval.",
    )
    p.add_argument(
        "--max_eval_samples",
        type=int,
        default=-1,
        help="If >0, cap validation set size (debug).",
    )
    p.add_argument(
        "--eval_steps",
        type=int,
        default=-1,
        help="Run validation every N steps; default -1 means same as --save_steps.",
    )
    p.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=1,
        help="Must satisfy (per_device_eval_batch_size * num_GPUs) %% num_generations == 0 for GRPO eval.",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="/nfs/hanpeng/huggingface/model/Qwen2.5",
        help="Directory for checkpoints, trainer state, and saved tokenizer.",
    )
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
    p.add_argument("--max_completion_length", type=int, default=512)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument(
        "--save_total_limit",
        type=int,
        default=3,
        help="With eval: keeps best eval_loss checkpoint plus rotating step saves (HF semantics).",
    )
    p.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default="auto",
        help="auto: if output_dir has checkpoint-*, resume from the latest; none: start from scratch; "
        "or absolute/relative path to a checkpoint dir (e.g. .../checkpoint-600).",
    )
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

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    train_ds = _map_quotes_dataset(
        args.train_file,
        tokenizer,
        args.max_prompt_length,
        args.max_samples,
        shuffle=True,
        seed=args.seed,
    )

    eval_ds = None
    ef = args.eval_file.strip()
    if ef.lower() not in ("", "none", "false", "no"):
        ep = Path(ef)
        if not ep.is_file():
            _LOG.warning("Eval file not found (%s); continuing without validation.", ep)
        else:
            eval_ds = _map_quotes_dataset(
                str(ep.resolve()),
                tokenizer,
                args.max_prompt_length,
                args.max_eval_samples,
                shuffle=False,
                seed=args.seed,
            )
            if len(eval_ds) == 0:
                _LOG.warning("Eval dataset is empty; disabling validation.")
                eval_ds = None

    eval_steps = args.eval_steps if args.eval_steps > 0 else args.save_steps

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
        save_total_limit=args.save_total_limit,
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

    if eval_ds is not None:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        num_gen_eval = args.num_generations
        per_dev_eval = _fit_per_device_eval_batch_size(
            world_size,
            num_gen_eval,
            args.per_device_eval_batch_size,
        )
        if per_dev_eval != args.per_device_eval_batch_size:
            _LOG.info(
                "Adjusted per_device_eval_batch_size %s -> %s "
                "(WORLD_SIZE=%s, num_generations=%s for eval)",
                args.per_device_eval_batch_size,
                per_dev_eval,
                world_size,
                num_gen_eval,
            )
        training_kwargs["eval_strategy"] = "steps"
        training_kwargs["eval_steps"] = eval_steps
        training_kwargs["per_device_eval_batch_size"] = per_dev_eval
        training_kwargs["load_best_model_at_end"] = True
        training_kwargs["metric_for_best_model"] = "eval_loss"
        training_kwargs["greater_is_better"] = False

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
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )
    resume_ckpt = _resolve_resume_from_checkpoint(args.resume_from_checkpoint, args.output_dir)
    trainer.train(resume_from_checkpoint=resume_ckpt)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
