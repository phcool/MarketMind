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
import csv
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import Dataset, load_dataset, load_from_disk
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
TRAIN_DIR = SCRIPT_DIR.parents[1]
REPO_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_TRAIN_FILE = REPO_ROOT / "dataset" / "quotes_7d_cot_from_batch.csv"


def _default_hf_output_dir() -> Path:
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home).expanduser() / "models" / "qwen2_5_sft_cot_lora"
    return Path("/nfs/hanpeng/huggingface/models/qwen2_5_sft_cot_lora")


DEFAULT_OUTPUT_DIR = _default_hf_output_dir()

LOG = logging.getLogger(__name__)


def _rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def _is_rank_zero() -> bool:
    return _rank() == 0


def _log_rank_zero(message: str, *args) -> None:
    if _is_rank_zero():
        LOG.info(message, *args)


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


def _inspect_csv_records(path: str) -> dict[str, int]:
    records = 0
    prompt_min = None
    prompt_max = 0
    completion_min = None
    completion_max = 0
    csv_path = Path(path).expanduser()
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        expected = {"prompt", "completion"}
        actual = set(reader.fieldnames or [])
        missing = expected - actual
        if missing:
            raise SystemExit(f"CSV missing required columns {sorted(missing)}; found {reader.fieldnames}")
        for row in reader:
            records += 1
            prompt_len = len(str(row["prompt"]))
            completion_len = len(str(row["completion"]))
            prompt_min = prompt_len if prompt_min is None else min(prompt_min, prompt_len)
            prompt_max = max(prompt_max, prompt_len)
            completion_min = completion_len if completion_min is None else min(completion_min, completion_len)
            completion_max = max(completion_max, completion_len)
    return {
        "records": records,
        "prompt_min_chars": int(prompt_min or 0),
        "prompt_max_chars": int(prompt_max),
        "completion_min_chars": int(completion_min or 0),
        "completion_max_chars": int(completion_max),
    }


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


def _build_tokenized_cache_root(
    output_dir: str,
    model_name_or_path: str,
    train_file: str,
    eval_file: str,
    eval_ratio: float,
    max_seq_length: int,
) -> Path:
    payload = {
        "model_name_or_path": model_name_or_path,
        "train_file": str(Path(train_file).resolve()),
        "eval_file": eval_file,
        "eval_ratio": float(eval_ratio),
        "max_seq_length": int(max_seq_length),
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    return Path(output_dir) / ".cache" / "tokenized_sft" / digest


def _wait_for_rank_zero_cache(ready_file: Path, error_file: Path, timeout_seconds: int = 7200) -> None:
    start = time.time()
    while True:
        if ready_file.exists():
            return
        if error_file.exists():
            raise RuntimeError(error_file.read_text(encoding="utf-8"))
        elapsed = time.time() - start
        if elapsed > timeout_seconds:
            raise TimeoutError(
                f"Timed out after {timeout_seconds}s waiting for tokenized dataset cache at {ready_file.parent}"
            )
        if int(elapsed) % 60 == 0:
            LOG.info("Rank %s still waiting for tokenized dataset cache at %s", _rank(), ready_file.parent)
        time.sleep(5)


def _tokenize_datasets_once(
    train_dataset: Dataset,
    eval_dataset: Dataset | None,
    preprocess,
    *,
    output_dir: str,
    model_name_or_path: str,
    train_file: str,
    eval_file: str,
    eval_ratio: float,
    max_seq_length: int,
) -> tuple[Dataset, Dataset | None]:
    cache_root = _build_tokenized_cache_root(
        output_dir=output_dir,
        model_name_or_path=model_name_or_path,
        train_file=train_file,
        eval_file=eval_file,
        eval_ratio=eval_ratio,
        max_seq_length=max_seq_length,
    )
    train_dir = cache_root / "train"
    eval_dir = cache_root / "eval"
    ready_file = cache_root / "READY"
    error_file = cache_root / "ERROR"

    if ready_file.exists() and train_dir.is_dir():
        _log_rank_zero("Loading tokenized datasets from cache: %s", cache_root)
        cached_train = load_from_disk(str(train_dir))
        cached_eval = load_from_disk(str(eval_dir)) if eval_dir.is_dir() else None
        return cached_train, cached_eval

    if _is_rank_zero():
        cache_root.mkdir(parents=True, exist_ok=True)
        _log_rank_zero("Tokenizing datasets into cache: %s", cache_root)
        try:
            cached_train = train_dataset.map(
                preprocess,
                remove_columns=train_dataset.column_names,
                desc="Tokenizing train dataset",
            )
            tmp_train_dir = cache_root / f"train.tmp-{os.getpid()}"
            cached_train.save_to_disk(str(tmp_train_dir))
            os.replace(tmp_train_dir, train_dir)

            cached_eval = None
            if eval_dataset is not None:
                cached_eval = eval_dataset.map(
                    preprocess,
                    remove_columns=eval_dataset.column_names,
                    desc="Tokenizing eval dataset",
                )
                tmp_eval_dir = cache_root / f"eval.tmp-{os.getpid()}"
                cached_eval.save_to_disk(str(tmp_eval_dir))
                os.replace(tmp_eval_dir, eval_dir)

            ready_file.write_text("ok\n", encoding="utf-8")
            return cached_train, cached_eval
        except Exception as exc:
            error_file.write_text(f"Rank 0 failed while tokenizing datasets: {exc}\n", encoding="utf-8")
            raise

    LOG.info("Rank %s waiting for rank 0 to prepare tokenized datasets at %s", _rank(), cache_root)
    _wait_for_rank_zero_cache(ready_file, error_file)
    return load_from_disk(str(train_dir)), load_from_disk(str(eval_dir)) if eval_dir.is_dir() else None


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
        force=True,
    )
    set_seed(args.seed)

    train_path = Path(args.train_file).expanduser()
    if not train_path.is_file():
        raise SystemExit(f"Training file not found: {train_path}")

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    _log_rank_zero("Starting LoRA SFT")
    _log_rank_zero("Train file: %s", train_path)
    _log_rank_zero("Eval file: %s", args.eval_file)
    _log_rank_zero("Output dir: %s", output_dir)
    _log_rank_zero("Model: %s", args.model_name_or_path)
    _log_rank_zero("World size (env): %s", _world_size())
    train_inspection = _inspect_csv_records(str(train_path))
    _log_rank_zero(
        "Train CSV inspection: records=%s prompt_chars=%s..%s completion_chars=%s..%s",
        train_inspection["records"],
        train_inspection["prompt_min_chars"],
        train_inspection["prompt_max_chars"],
        train_inspection["completion_min_chars"],
        train_inspection["completion_max_chars"],
    )
    _log_rank_zero("Loading CSV dataset(s)")

    train_dataset, eval_dataset = _prepare_datasets(
        train_file=args.train_file,
        eval_file=args.eval_file,
        eval_ratio=args.eval_ratio,
        seed=args.seed,
    )

    if eval_dataset is not None:
        _log_rank_zero("Loaded raw datasets: train=%s eval=%s", len(train_dataset), len(eval_dataset))
    else:
        _log_rank_zero("Loaded raw dataset: train=%s eval=disabled", len(train_dataset))
    _log_rank_zero("Loading tokenizer: %s", args.model_name_or_path)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    _log_rank_zero("Tokenizer ready. Preparing tokenized datasets")

    preprocess = lambda row: _tokenize_example(row, tokenizer, args.max_seq_length)
    train_dataset, eval_dataset = _tokenize_datasets_once(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        preprocess=preprocess,
        output_dir=str(output_dir),
        model_name_or_path=args.model_name_or_path,
        train_file=args.train_file,
        eval_file=args.eval_file,
        eval_ratio=args.eval_ratio,
        max_seq_length=args.max_seq_length,
    )
    _log_rank_zero("Tokenized datasets ready: train=%s eval=%s", len(train_dataset), len(eval_dataset) if eval_dataset else 0)

    torch_dtype = {
        "auto": "auto",
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[args.torch_dtype]
    _log_rank_zero("Loading base model with dtype=%s", args.torch_dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    _log_rank_zero("Base model loaded")

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
    _log_rank_zero("LoRA adapters attached")

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
        world_size = _world_size()
        effective_batch = args.per_device_train_batch_size * args.gradient_accumulation_steps * max(world_size, 1)
        LOG.info("Train examples: %s", len(train_dataset))
        LOG.info("Eval examples: %s", len(eval_dataset) if eval_dataset is not None else 0)
        LOG.info("World size: %s", world_size)
        LOG.info("Effective train batch size: %s", effective_batch)
        LOG.info("LoRA config: r=%s alpha=%s dropout=%s", args.lora_r, args.lora_alpha, args.lora_dropout)
        LOG.info("Trainable params: %s", _format_trainable_params(model))

    resume = _resolve_resume_from_checkpoint(args.resume_from_checkpoint, args.output_dir)
    _log_rank_zero("Starting trainer.train(resume_from_checkpoint=%s)", resume)
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
