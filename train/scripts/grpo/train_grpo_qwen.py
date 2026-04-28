"""
GRPO training for Qwen2.5-7B-Instruct on the 7-day K-line multi-horizon dataset.

Default datasets:
  - train: dataset/quotes_7d_multi_pre2026_dataset.csv
  - eval:  dataset/quotes_7d_multi_eval_20260101_20260228.csv

Expected schema:
  - prompt: user prompt text
  - future_1_3_7_trade_day_labels: labels such as 涨，跌，涨

Compatibility:
  - A CSV with columns (prompt, completion) is also accepted if completion
    contains the final 1/3/7-day answers.

The model is asked to think first, then output final answers in the form:
  未来1个交易日：涨/跌
  未来3个交易日：涨/跌
  未来7个交易日：涨/跌

Reward is the average directional accuracy across the three horizons.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import random
import re
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_DIR = SCRIPT_DIR.parents[1]
REPO_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_TRAIN_CSV = REPO_ROOT / "dataset" / "quotes_7d_multi_pre2026_dataset.csv"
DEFAULT_VAL_CSV = REPO_ROOT / "dataset" / "quotes_7d_multi_eval_20260101_20260228.csv"

_LOG = logging.getLogger(__name__)

_FUTURE_RE = {
    1: re.compile(r"未来\s*1\s*个?交易日\s*[:：]\s*(涨|跌|上涨|下跌)"),
    3: re.compile(r"未来\s*3\s*个?交易日\s*[:：]\s*(涨|跌|上涨|下跌)"),
    7: re.compile(r"未来\s*7\s*个?交易日\s*[:：]\s*(涨|跌|上涨|下跌)"),
}
_FINAL_LINE_RE = re.compile(r"^(涨|跌|上涨|下跌)\s*$")


def _normalize_direction(text: str) -> str | None:
    value = (text or "").strip()
    if value in {"涨", "上涨"}:
        return "涨"
    if value in {"跌", "下跌"}:
        return "跌"
    return None


def parse_label_triplet(raw: str) -> tuple[str, str, str]:
    parts = [p.strip() for p in re.split(r"[，,]", str(raw).strip()) if p.strip()]
    if len(parts) != 3:
        raise ValueError(f"invalid future_1_3_7_trade_day_labels: {raw!r}")
    normalized = tuple(_normalize_direction(p) for p in parts)
    if any(x is None for x in normalized):
        raise ValueError(f"invalid direction in labels: {raw!r}")
    return normalized  # type: ignore[return-value]


def extract_final_triplet(text: str) -> tuple[str, str, str] | None:
    if not text or not str(text).strip():
        return None
    text = str(text)
    captures: dict[int, str] = {}
    for horizon, pat in _FUTURE_RE.items():
        m = pat.search(text)
        if m:
            norm = _normalize_direction(m.group(1))
            if norm is not None:
                captures[horizon] = norm
    if len(captures) == 3:
        return captures[1], captures[3], captures[7]

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) >= 3:
        tail = lines[-3:]
        values: list[str] = []
        for ln in tail:
            m = _FINAL_LINE_RE.search(ln)
            if m:
                norm = _normalize_direction(m.group(1))
            else:
                norm = _normalize_direction(ln.split("：")[-1].split(":")[-1].strip())
            if norm is None:
                return None
            values.append(norm)
        if len(values) == 3:
            return values[0], values[1], values[2]
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


def _extract_reference_triplet(example: dict) -> tuple[str, str, str]:
    if "future_1_3_7_trade_day_labels" in example and str(example["future_1_3_7_trade_day_labels"]).strip():
        return parse_label_triplet(example["future_1_3_7_trade_day_labels"])
    if "completion" in example and str(example["completion"]).strip():
        label = extract_final_triplet(str(example["completion"]))
        if label is None:
            raise ValueError("Could not parse final 1/3/7 directions from completion.")
        return label
    raise ValueError("CSV must include either future_1_3_7_trade_day_labels or completion column.")


def _map_quotes_dataset(
    csv_path: str,
    tokenizer,
    max_prompt_length: int,
    max_samples: int,
    *,
    shuffle: bool,
    seed: int,
):
    """Load CSV, apply chat template + left token truncation."""
    ds = load_dataset("csv", data_files=csv_path, split="train")
    if max_samples and max_samples > 0:
        ds = ds.select(range(min(max_samples, len(ds))))

    def apply_chat_template_row(example: dict) -> dict:
        user_text = str(example["prompt"]).strip()
        messages = [{"role": "user", "content": user_text}]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        if len(prompt_ids) > max_prompt_length:
            prompt_ids = prompt_ids[-max_prompt_length:]
            prompt = tokenizer.decode(prompt_ids, skip_special_tokens=False)
        return {
            "prompt": prompt,
            "future_1_3_7_direction": _extract_reference_triplet(example),
        }

    mapped = ds.map(apply_chat_template_row, remove_columns=list(ds.column_names))
    if shuffle:
        mapped = mapped.shuffle(seed=seed)
    return mapped


class GRPOTrainerEvalSubset(GRPOTrainer):
    """Each evaluate() runs on a random subset of eval_dataset for speed."""

    def __init__(self, *args, eval_subset_ratio: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self._eval_subset_ratio = float(eval_subset_ratio)

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval_"):
        if eval_dataset is None:
            eval_dataset = self.eval_dataset
        if eval_dataset is None:
            return super().evaluate(
                eval_dataset=None,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )
        r = self._eval_subset_ratio
        if r >= 1.0:
            return super().evaluate(
                eval_dataset=eval_dataset,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )
        n_total = len(eval_dataset)
        k = max(1, min(n_total, int(math.ceil(n_total * r))))
        step = int(self.state.global_step) if self.state is not None else 0
        seed = (int(self.args.seed) if self.args.seed is not None else 0) * 1_000_003 + step
        rng = random.Random(seed)
        indices = sorted(rng.sample(range(n_total), k))
        sub = eval_dataset.select(indices)
        if self.is_world_process_zero():
            _LOG.info(
                "Eval subset %s / %s examples (ratio=%.5f, global_step=%s)",
                k,
                n_total,
                r,
                step,
            )
        return super().evaluate(
            eval_dataset=sub,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )


def _fit_per_device_eval_batch_size(world_size: int, num_generations: int, requested: int) -> int:
    if world_size < 1 or num_generations < 2:
        return max(1, requested)
    g = math.gcd(world_size, num_generations)
    minimum = num_generations // g
    b = max(requested, minimum)
    while (b * world_size) % num_generations != 0:
        b += 1
    return b


def make_direction_accuracy_reward():
    def direction_accuracy_reward(
        completions: list,
        future_1_3_7_direction: list[tuple[str, str, str]],
        log_metric=None,
        **kwargs,
    ):
        rewards: list[float] = []
        parsed = 0
        full_match = 0
        hit_1 = 0
        hit_3 = 0
        hit_7 = 0
        for comp, gt in zip(completions, future_1_3_7_direction):
            pred = extract_final_triplet(_completion_to_text(comp))
            if pred is None:
                rewards.append(0.0)
                continue
            parsed += 1
            score = 0.0
            if pred[0] == gt[0]:
                hit_1 += 1
                score += 1.0 / 3.0
            if pred[1] == gt[1]:
                hit_3 += 1
                score += 1.0 / 3.0
            if pred[2] == gt[2]:
                hit_7 += 1
                score += 1.0 / 3.0
            if pred == gt:
                full_match += 1
            rewards.append(score)
        if log_metric and rewards:
            log_metric("direction_parse_success_rate", parsed / len(rewards))
            log_metric("direction_full_match_rate", full_match / len(rewards))
            log_metric("direction_1d_match_rate", hit_1 / len(rewards))
            log_metric("direction_3d_match_rate", hit_3 / len(rewards))
            log_metric("direction_7d_match_rate", hit_7 / len(rewards))
        return rewards

    direction_accuracy_reward.__name__ = "direction_accuracy_reward"
    return direction_accuracy_reward


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="GRPO train Qwen2.5-7B on 7-day K-line multi-horizon prompts.")
    p.add_argument(
        "--model_name_or_path",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="HF model id or local path.",
    )
    p.add_argument(
        "--train_file",
        type=str,
        default=str(DEFAULT_TRAIN_CSV),
        help="Training CSV with columns: prompt + future_1_3_7_trade_day_labels (preferred) or prompt + completion.",
    )
    p.add_argument(
        "--eval_file",
        type=str,
        default=str(DEFAULT_VAL_CSV),
        help="Validation CSV with the same schema as train. Set to 'none' to disable eval.",
    )
    p.add_argument("--max_eval_samples", type=int, default=-1, help="If >0, cap validation set size.")
    p.add_argument("--eval_steps", type=int, default=-1, help="Run validation every N steps; default uses save_steps.")
    p.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=1,
        help="Eval batch per GPU; auto-increased if needed so (batch * WORLD_SIZE) %% num_generations == 0.",
    )
    p.add_argument(
        "--eval_subset_ratio",
        type=float,
        default=-1.0,
        help="Fraction of validation rows per eval (random subset). Default -1 uses an automatic ratio.",
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
        default=4096,
        help="Left-truncate templated prompts to this many tokens; also sets vLLM context to max_prompt+completion.",
    )
    p.add_argument("--max_completion_length", type=int, default=768)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument(
        "--save_total_limit",
        type=int,
        default=3,
        help="With eval: keeps best eval__loss checkpoint plus rotating step saves.",
    )
    p.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default="auto",
        help="auto: resume from latest checkpoint under output_dir; none: start from scratch; or a checkpoint path.",
    )
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--beta", type=float, default=0.0, help="KL coefficient; 0 disables KL in GRPO.")
    p.add_argument(
        "--report_to",
        type=str,
        default="none",
        help="e.g. none, tensorboard, wandb (comma-separated for multiple).",
    )
    p.add_argument("--no_vllm", action="store_true", help="Disable vLLM and use HF generate() for rollouts.")
    p.add_argument(
        "--vllm_mode",
        type=str,
        default="colocate",
        choices=("colocate", "server"),
        help="colocate: vLLM in trainer process. server: connect to trl vllm-serve.",
    )
    p.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.35)
    p.add_argument("--vllm_tensor_parallel_size", type=int, default=1)
    p.add_argument("--vllm_server_base_url", type=str, default="")
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

    if args.eval_subset_ratio < 0:
        eval_subset_ratio = min(1.0, 2.0 * float(eval_steps) / max(1, len(train_ds)))
    else:
        eval_subset_ratio = min(1.0, max(0.0, float(args.eval_subset_ratio)))
    if eval_subset_ratio <= 0.0:
        eval_subset_ratio = min(1.0, 2.0 * float(eval_steps) / max(1, len(train_ds)))
        _LOG.warning("eval_subset_ratio was <= 0; using auto ratio %.5f", eval_subset_ratio)

    if eval_ds is not None and int(os.environ.get("RANK", "0")) == 0:
        _LOG.info(
            "Each validation run uses %.5f of val rows (train_rows=%s, eval_steps=%s)",
            eval_subset_ratio,
            len(train_ds),
            eval_steps,
        )

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
        per_dev_eval = _fit_per_device_eval_batch_size(
            world_size,
            args.num_generations,
            args.per_device_eval_batch_size,
        )
        if per_dev_eval != args.per_device_eval_batch_size:
            _LOG.info(
                "Adjusted per_device_eval_batch_size %s -> %s "
                "(WORLD_SIZE=%s, num_generations=%s for eval)",
                args.per_device_eval_batch_size,
                per_dev_eval,
                world_size,
                args.num_generations,
            )
        training_kwargs["eval_strategy"] = "steps"
        training_kwargs["eval_steps"] = eval_steps
        training_kwargs["per_device_eval_batch_size"] = per_dev_eval
        training_kwargs["load_best_model_at_end"] = True
        training_kwargs["metric_for_best_model"] = "eval__loss"
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

    trainer_cls = GRPOTrainer
    trainer_kwargs: dict = dict(
        model=args.model_name_or_path,
        args=training_args,
        reward_funcs=make_direction_accuracy_reward(),
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )
    if eval_ds is not None and eval_subset_ratio < 1.0:
        trainer_cls = GRPOTrainerEvalSubset
        trainer_kwargs["eval_subset_ratio"] = eval_subset_ratio

    trainer = trainer_cls(**trainer_kwargs)
    resume_ckpt = _resolve_resume_from_checkpoint(args.resume_from_checkpoint, args.output_dir)
    trainer.train(resume_from_checkpoint=resume_ckpt)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
