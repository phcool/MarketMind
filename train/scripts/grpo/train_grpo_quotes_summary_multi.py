"""
GRPO training for Qwen2.5-7B-Instruct on the multi-source 5-day summary dataset.

Dataset:
  dataset/quotes_summary_5d_2026-01-01_to_2026-03-01.csv

Each training row is converted into a user prompt that contains:
  - stock code
  - analysis date
  - 5-day normalized K-line
  - news summaries
  - report summaries

The model is asked to think first, then output final answers for future 1/3/7
trading-day directions. Reward is the average match rate against the dataset
labels for the three horizons.
"""

from __future__ import annotations

import argparse
import json
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

DEFAULT_TRAIN_CSV = REPO_ROOT / "dataset" / "quotes_summary_5d_2026-01-01_to_2026-03-01.csv"
DEFAULT_VAL_CSV = REPO_ROOT / "dataset" / "quotes_summary_5d_2026-03-01_to_2026-04-01.csv"

_LOG = logging.getLogger(__name__)

_FINAL_PATTERNS = {
    "future_1d": re.compile(r"未来\s*1\s*个?交易日\s*[:：]\s*(涨|跌|上涨|下跌)"),
    "future_3d": re.compile(r"未来\s*3\s*个?交易日\s*[:：]\s*(涨|跌|上涨|下跌)"),
    "future_7d": re.compile(r"未来\s*7\s*个?交易日\s*[:：]\s*(涨|跌|上涨|下跌)"),
}


def _normalize_direction(text: str) -> str | None:
    value = (text or "").strip()
    if value in {"涨", "上涨"}:
        return "涨"
    if value in {"跌", "下跌"}:
        return "跌"
    return None


def parse_label_triplet(raw: str) -> tuple[str, str, str]:
    text = (raw or "").strip().replace(" ", "")
    parts = re.split(r"[，,、/|]+", text)
    parts = [p for p in parts if p]
    if len(parts) != 3:
        raise ValueError(f"invalid future_1_3_7_trade_day_labels: {raw!r}")
    normalized = [_normalize_direction(p) for p in parts]
    if any(v is None for v in normalized):
        raise ValueError(f"invalid direction labels: {raw!r}")
    return normalized[0], normalized[1], normalized[2]


def parse_prediction_triplet(text: str) -> tuple[str | None, str | None, str | None]:
    if not text or not str(text).strip():
        return None, None, None
    body = str(text)
    results: list[str | None] = []
    for key in ("future_1d", "future_3d", "future_7d"):
        m = _FINAL_PATTERNS[key].search(body)
        results.append(_normalize_direction(m.group(1)) if m else None)
    return results[0], results[1], results[2]


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


def format_kline_rows(rows: list[dict]) -> str:
    lines = []
    for idx, row in enumerate(rows, start=1):
        lines.append(
            "Day{idx}: trade_date={trade_date}, open={open}, high={high}, low={low}, "
            "close={close}, volume={volume}, amplitude={amplitude}, pct_change={pct_change}, "
            "turnover={turnover}".format(
                idx=idx,
                trade_date=row.get("trade_date", ""),
                open=row.get("open", ""),
                high=row.get("high", ""),
                low=row.get("low", ""),
                close=row.get("close", ""),
                volume=row.get("volume", ""),
                amplitude=row.get("amplitude", ""),
                pct_change=row.get("pct_change", ""),
                turnover=row.get("turnover", ""),
            )
        )
    return "\n".join(lines)


def format_summary_items(items: list[dict], *, field: str) -> str:
    if not items:
        return "无"
    blocks = []
    for idx, item in enumerate(items, start=1):
        summary = str(item.get(field, "")).strip()
        if not summary:
            continue
        title = str(item.get("title", "")).strip()
        item_date = str(item.get("date", "")).strip()
        blocks.append(f"{idx}. 日期={item_date}\n标题={title}\n{summary}")
    return "\n\n".join(blocks) if blocks else "无"


def build_prompt(example: dict) -> str:
    stock = str(example["stock"]).strip()
    analysis_date = str(example["date"]).strip()
    kline_rows = json.loads(example["kline_5d"])
    news_items = json.loads(example["news"] or "[]")
    report_items = json.loads(example["reports"] or "[]")
    kline_text = format_kline_rows(kline_rows)
    news_text = format_summary_items(news_items, field="summary")
    reports_text = format_summary_items(report_items, field="summary")
    return (
        f"分析日期：{analysis_date}\n"
        f"股票代码：{stock}\n\n"
        "过去5个交易日的归一化K线数据如下：\n"
        "其中，open 表示开盘价，high 表示最高价，low 表示最低价，close 表示收盘价，"
        "volume 表示成交量，amplitude 表示振幅，pct_change 表示涨跌幅，turnover 表示换手率。\n"
        f"{kline_text}\n\n"
        "近期新闻摘要如下：\n"
        f"{news_text}\n\n"
        "近期研报摘要如下：\n"
        f"{reports_text}\n\n"
        "请结合上面的 K 线、新闻摘要和研报摘要，预测未来 1 个交易日、3 个交易日和 7 个交易日的涨跌方向。\n\n"
        "【输出要求（必须严格遵守）】\n"
        "请先分析 K 线趋势、量能、波动与文本信息；将推理过程写在下面一对标记之间（只写推理过程）：\n"
        "【思维链开始】\n"
        "（在此撰写推理过程）\n"
        "【思维链结束】\n\n"
        "思维链结束后，请严格按下面三行输出最终答案：\n"
        "未来1个交易日：涨或跌\n"
        "未来3个交易日：涨或跌\n"
        "未来7个交易日：涨或跌\n\n"
        "注意：每一行最终答案只能写单个字“涨”或“跌”，不要输出其他词。"
    )


def _map_dataset(
    csv_path: str,
    tokenizer,
    max_prompt_length: int,
    max_samples: int,
    *,
    shuffle: bool,
    seed: int,
):
    ds = load_dataset("csv", data_files=csv_path, split="train")
    if max_samples and max_samples > 0:
        ds = ds.select(range(min(max_samples, len(ds))))

    def apply_chat_template_row(example: dict) -> dict:
        prompt_text = build_prompt(example)
        messages = [{"role": "user", "content": prompt_text}]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        if len(prompt_ids) > max_prompt_length:
            prompt_ids = prompt_ids[-max_prompt_length:]
            prompt = tokenizer.decode(prompt_ids, skip_special_tokens=False)
        future_1d, future_3d, future_7d = parse_label_triplet(example["future_1_3_7_trade_day_labels"])
        return {
            "prompt": prompt,
            "future_1d": future_1d,
            "future_3d": future_3d,
            "future_7d": future_7d,
        }

    mapped = ds.map(apply_chat_template_row, remove_columns=list(ds.column_names))
    if shuffle:
        mapped = mapped.shuffle(seed=seed)
    return mapped


class GRPOTrainerEvalSubset(GRPOTrainer):
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


def make_direction_triplet_reward():
    def direction_triplet_reward(
        completions: list,
        future_1d: list[str],
        future_3d: list[str],
        future_7d: list[str],
        log_metric=None,
        **kwargs,
    ):
        rewards: list[float] = []
        parse_ok = 0
        full_match = 0
        for comp, gt1, gt3, gt7 in zip(completions, future_1d, future_3d, future_7d):
            pred1, pred3, pred7 = parse_prediction_triplet(_completion_to_text(comp))
            preds = [pred1, pred3, pred7]
            gts = [gt1, gt3, gt7]
            matched = 0
            valid = 0
            for pred, gt in zip(preds, gts):
                if pred is None:
                    continue
                valid += 1
                if pred == gt:
                    matched += 1
            if valid == 3:
                parse_ok += 1
            if matched == 3 and valid == 3:
                full_match += 1
            rewards.append(matched / 3.0)
        if log_metric and rewards:
            log_metric("direction_parse_success_rate", parse_ok / len(rewards))
            log_metric("direction_full_match_rate", full_match / len(rewards))
        return rewards

    direction_triplet_reward.__name__ = "direction_triplet_reward"
    return direction_triplet_reward


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="GRPO train Qwen2.5-7B on multi-source 5d summary prompts.")
    p.add_argument("--model_name_or_path", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--train_file", type=str, default=str(DEFAULT_TRAIN_CSV))
    p.add_argument("--eval_file", type=str, default=str(DEFAULT_VAL_CSV))
    p.add_argument("--max_eval_samples", type=int, default=-1)
    p.add_argument("--eval_steps", type=int, default=-1)
    p.add_argument("--per_device_eval_batch_size", type=int, default=1)
    p.add_argument("--eval_subset_ratio", type=float, default=-1.0)
    p.add_argument(
        "--output_dir",
        type=str,
        default="/nfs/hanpeng/huggingface/models/qwen2_5_grpo_quotes_summary_multi",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_samples", type=int, default=-1)
    p.add_argument("--num_train_epochs", type=float, default=1.0)
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--learning_rate", type=float, default=1e-6)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--num_generations", type=int, default=8)
    p.add_argument("--max_prompt_length", type=int, default=6144)
    p.add_argument("--max_completion_length", type=int, default=768)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--save_total_limit", type=int, default=3)
    p.add_argument("--resume_from_checkpoint", type=str, default="auto")
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--beta", type=float, default=0.0)
    p.add_argument("--report_to", type=str, default="none")
    p.add_argument("--no_vllm", action="store_true")
    p.add_argument("--vllm_mode", type=str, default="colocate", choices=("colocate", "server"))
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

    train_ds = _map_dataset(
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
            eval_ds = _map_dataset(
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
                "Adjusted per_device_eval_batch_size %s -> %s (WORLD_SIZE=%s, num_generations=%s for eval)",
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
        reward_funcs=make_direction_triplet_reward(),
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
