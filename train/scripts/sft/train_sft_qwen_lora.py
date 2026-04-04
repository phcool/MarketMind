"""
LoRA supervised fine-tuning for Qwen2.5-Instruct on the quotes_7d CoT CSV dataset.

Dataset schema:
  - prompt: user message text
  - completion: assistant target text

Launch on 8 GPUs with torchrun:
  bash train/scripts/launch/run_sft_qwen_lora_8gpu.sh

Notes:
  - This is LoRA SFT, not full-parameter training.
  - Loss is masked on the prompt tokens and only applied to the assistant reply.
  - If a validation file is not supplied, the script will split a small eval set from train.
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
    set_seed,
)

SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_DIR = SCRIPT_DIR.parents[2]
REPO_ROOT = SCRIPT_DIR.parents[3]
DEFAULT_TRAIN_FILE = TRAIN_DIR / "dataset" / "quotes_7d_cot_from_batch.csv"


def _default_hf_output_dir() -> Path:
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home).expanduser() / "models" / "qwen2_5_sft_cot_lora"
    return Path("/nfs/hanpeng/huggingface/models/qwen2_5_sft_cot_lora")


DEFAULT_OUTPUT_DIR = _default_hf_output_dir()

LOG = logging.getLogger(__name__)


def _resolve_resume_from_checkpoint(resume: str, output_dir: str) -> bool | str | None:
    value = resume.strip()
    if value.lower() in {"none", "false", "no", "0"}:
        return None
    if value.lower() == "auto":
        out = Path(output_dir)
        if out.is_dir() and any(out.glob("checkpoint-*")):
            return True
        return None
    path = Path(value)
    if not path.is_absolute():
        candidate = Path(output_dir) / value
        if candidate.is_dir():
            return str(candidate.resolve())
    elif path.is_dir():
        return str(path.resolve())
    raise SystemExit(
        f"--resume_from_checkpoint expects 'auto', 'none', or a checkpoint directory; got {resume!r}"
    )


def _load_csv_dataset(path: str) -> Dataset:
    ds = load_dataset("csv", data_files=path, split="train")
    expected = {"prompt", "completion"}
    actual = set(ds.column_names)
    missing = expected - actual
    if missing:
        raise SystemExit(f"CSV missing required columns {sorted(missing)}; found {ds.column_names}")
    return ds


def _prepare_datasets(
    train_file: str,
    eval_file: str,
    eval_ratio: float,
    seed: int,
) -> tuple[Dataset, Dataset | None]:
    train_ds = _load_csv_dataset(train_file)
    if eval_file.strip().lower() not in {"", "none", "null"}:
        return train_ds, _load_csv_dataset(eval_file)
    if eval_ratio <= 0:
        return train_ds, None
    split = train_ds.train_test_split(test_size=eval_ratio, seed=seed, shuffle=True)
    return split["train"], split["test"]


def _tokenize_example(
    example: dict[str, str],
    tokenizer: PreTrainedTokenizerBase,
    max_seq_length: int,
) -> dict[str, list[int]]:
    prompt = str(example["prompt"]).strip()
    completion = str(example["completion"]).strip()
    if not prompt or not completion:
        raise ValueError("Empty prompt or completion encountered.")

    prompt_messages = [{"role": "user", "content": prompt}]
    full_messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": completion},
    ]

    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = tokenizer.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    if len(full_ids) < len(prompt_ids):
        raise ValueError("Tokenized full conversation is shorter than tokenized prompt.")

    response_ids = full_ids[len(prompt_ids) :]
    if not response_ids:
        raise ValueError("Assistant response became empty after chat templating.")

    max_seq_length = max(int(max_seq_length), 8)
    if len(response_ids) >= max_seq_length:
        response_ids = response_ids[: max_seq_length - 1]

    max_prompt_tokens = max_seq_length - len(response_ids)
    if max_prompt_tokens <= 0:
        max_prompt_tokens = 1
        response_ids = response_ids[: max_seq_length - 1]
    prompt_ids = prompt_ids[-max_prompt_tokens:]

    input_ids = prompt_ids + response_ids
    labels = ([-100] * len(prompt_ids)) + response_ids
    attention_mask = [1] * len(input_ids)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def _parse_target_modules(raw: str) -> list[str]:
    modules = [item.strip() for item in raw.split(",") if item.strip()]
    if not modules:
        raise SystemExit("--lora_target_modules must contain at least one module name.")
    return modules


def _format_trainable_params(model) -> str:
    trainable = 0
    total = 0
    for param in model.parameters():
        count = param.numel()
        total += count
        if param.requires_grad:
            trainable += count
    ratio = (100.0 * trainable / total) if total else 0.0
    return f"trainable={trainable:,} total={total:,} ratio={ratio:.4f}%"


@dataclass
class SupervisedDataCollator:
    tokenizer: PreTrainedTokenizerBase
    label_pad_token_id: int = -100

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        max_length = max(len(feature["input_ids"]) for feature in features)
        pad_id = int(self.tokenizer.pad_token_id)

        input_ids = []
        attention_mask = []
        labels = []
        for feature in features:
            pad_len = max_length - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [pad_id] * pad_len)
            attention_mask.append(feature["attention_mask"] + [0] * pad_len)
            labels.append(feature["labels"] + [self.label_pad_token_id] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LoRA SFT for Qwen2.5-Instruct on CoT CSV.")
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--train_file", type=str, default=str(DEFAULT_TRAIN_FILE))
    parser.add_argument("--eval_file", type=str, default="none")
    parser.add_argument("--eval_ratio", type=float, default=0.01)
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--num_train_epochs", type=float, default=2.0)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--no_gradient_checkpointing", action="store_false", dest="gradient_checkpointing")
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--resume_from_checkpoint", type=str, default="auto")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16", choices=["auto", "bfloat16", "float16"])
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        level=logging.INFO,
    )
    set_seed(args.seed)

    train_dataset, eval_dataset = _prepare_datasets(
        train_file=args.train_file,
        eval_file=args.eval_file,
        eval_ratio=args.eval_ratio,
        seed=args.seed,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    preprocess = lambda row: _tokenize_example(row, tokenizer, args.max_seq_length)
    train_dataset = train_dataset.map(
        preprocess,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing train dataset",
    )
    if eval_dataset is not None:
        eval_dataset = eval_dataset.map(
            preprocess,
            remove_columns=eval_dataset.column_names,
            desc="Tokenizing eval dataset",
        )

    torch_dtype = {
        "auto": "auto",
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[args.torch_dtype]
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    model.config.use_cache = False

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=_parse_target_modules(args.lora_target_modules),
    )
    model = get_peft_model(model, peft_config)

    eval_strategy = "steps" if eval_dataset is not None else "no"
    load_best_model_at_end = eval_dataset is not None
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=False,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        save_total_limit=args.save_total_limit,
        dataloader_num_workers=args.dataloader_num_workers,
        bf16=args.torch_dtype == "bfloat16",
        fp16=args.torch_dtype == "float16",
        report_to="none",
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        eval_strategy=eval_strategy,
        save_strategy="steps",
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model="eval_loss" if load_best_model_at_end else None,
        greater_is_better=False if load_best_model_at_end else None,
        optim="adamw_torch",
        save_safetensors=True,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=SupervisedDataCollator(tokenizer=tokenizer),
    )

    if trainer.is_world_process_zero():
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        effective_batch = args.per_device_train_batch_size * args.gradient_accumulation_steps * max(world_size, 1)
        LOG.info("Train examples: %s", len(train_dataset))
        LOG.info("Eval examples: %s", len(eval_dataset) if eval_dataset is not None else 0)
        LOG.info("World size: %s", world_size)
        LOG.info("Effective train batch size: %s", effective_batch)
        LOG.info("LoRA config: r=%s alpha=%s dropout=%s", args.lora_r, args.lora_alpha, args.lora_dropout)
        LOG.info("Trainable params: %s", _format_trainable_params(model))

    resume = _resolve_resume_from_checkpoint(args.resume_from_checkpoint, args.output_dir)
    train_result = trainer.train(resume_from_checkpoint=resume)
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    trainer.save_state()

    metrics = train_result.metrics
    metrics["train_examples"] = len(train_dataset)
    if eval_dataset is not None:
        metrics["eval_examples"] = len(eval_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    if eval_dataset is not None:
        eval_metrics = trainer.evaluate()
        eval_metrics["eval_examples"] = len(eval_dataset)
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)


if __name__ == "__main__":
    main()
