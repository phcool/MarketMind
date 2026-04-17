from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path

from transformers import AutoConfig
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT_DIR / "dataset" / "quotes_summary_5d_2026-01-01_to_2026-04-01.csv"
DEFAULT_OUTPUT_JSON = ROOT_DIR / "test" / "outputs" / "quotes_summary_5d_vllm_eval.json"
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_ADAPTER_PATH = "/nfs/hanpeng/huggingface/models/qwen2_5_sft_cot_lora"

TRIPLE_LABEL_RE = re.compile(r"^\s*([涨跌])\s*[，,/\s]\s*([涨跌])\s*[，,/\s]\s*([涨跌])\s*$")
EXTRACTION_FAILED = "提取失败"

HEADER = """下面是一条股票日度样本，请根据该股票在目标日期当天可见的信息，预测下一个交易日相对当天是涨还是跌。

你会看到：
1. 该股票在目标日期及之前连续5个交易日的归一化K线；
2. 截止目标日期最近的news摘要；
3. 截止目标日期最近的report摘要。

其中，open 表示开盘价，high 表示最高价，low 表示最低价，close 表示收盘价，volume 表示成交量，amplitude 表示振幅，pct_change 表示涨跌幅，turnover 表示换手率。"""

TAIL = """请综合K线形态、量能变化、news摘要和report摘要，分别预测该股票相对当前交易日收盘价在以下三个阶段的涨跌：
1. 1个交易日后
2. 3个交易日后
3. 7个交易日后

【输出要求（必须严格遵守）】
请先分析，再将推理过程写在下面一对标记之间（只写推理过程）：
【思维链开始】
（在此撰写推理过程）
【思维链结束】

思维链结束后，请单独一行按顺序输出三个阶段的答案，格式必须严格为：
涨，涨，跌

三个位置分别对应：1个交易日后、3个交易日后、7个交易日后。只输出这一行，不要输出其他文字。"""


def _load_rows(dataset_path: Path) -> list[dict[str, str]]:
    with dataset_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        expected = {"date", "stock", "kline_5d", "news", "reports", "future_1_3_7_trade_day_labels"}
        actual = set(reader.fieldnames or [])
        missing = expected - actual
        if missing:
            raise SystemExit(f"Dataset missing required columns {sorted(missing)}; found {reader.fieldnames}")
        return list(reader)


def _format_kline_block(kline_rows: list[dict]) -> str:
    lines = []
    for i, row in enumerate(kline_rows, start=1):
        lines.append(
            "Day{idx} ({trade_date}): open={open}, high={high}, low={low}, close={close}, "
            "volume={volume}, amplitude={amplitude}, pct_change={pct_change}, turnover={turnover}".format(
                idx=i,
                trade_date=row.get("trade_date", ""),
                open=row.get("open", "N/A"),
                high=row.get("high", "N/A"),
                low=row.get("low", "N/A"),
                close=row.get("close", "N/A"),
                volume=row.get("volume", "N/A"),
                amplitude=row.get("amplitude", "N/A"),
                pct_change=row.get("pct_change", "N/A"),
                turnover=row.get("turnover", "N/A"),
            )
        )
    return "\n".join(lines)


def _format_summary_items(items: list[dict], *, label: str) -> str:
    if not items:
        return f"{label}：无"
    lines = [f"{label}："]
    for i, item in enumerate(items, start=1):
        lines.append(
            f"{i}. date={item.get('date', '')}\n"
            f"   title={item.get('title', '')}\n"
            f"   summary={item.get('summary', '')}"
        )
    return "\n".join(lines)


def build_prompt(row: dict[str, str]) -> str:
    kline_rows = json.loads(row["kline_5d"])
    news_items = json.loads(row["news"])
    report_items = json.loads(row["reports"])
    parts = [
        HEADER,
        "",
        f"目标日期：{row['date']}",
        f"股票代码：{row['stock']}",
        "",
        "过去5个交易日的归一化K线：",
        _format_kline_block(kline_rows),
        "",
        _format_summary_items(news_items, label="截止该日最近的news摘要"),
        "",
        _format_summary_items(report_items, label="截止该日最近的report摘要"),
        "",
        TAIL,
    ]
    return "\n".join(parts)


def _normalize_triplet(labels: tuple[str, str, str] | list[str]) -> str:
    return "，".join(labels)


def extract_predicted_label(text: str) -> str:
    if not text or not str(text).strip():
        return EXTRACTION_FAILED

    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    if not lines:
        return EXTRACTION_FAILED

    last_match = TRIPLE_LABEL_RE.match(lines[-1])
    if last_match:
        return _normalize_triplet(last_match.groups())

    for line in reversed(lines):
        match = TRIPLE_LABEL_RE.match(line)
        if match:
            return _normalize_triplet(match.groups())

    compact = "\n".join(lines)
    phrase_patterns = [
        r"(?:最终答案|答案|结论|预测结果)\s*[:：是为]?\s*([涨跌])\s*[，,/\s]\s*([涨跌])\s*[，,/\s]\s*([涨跌])",
    ]
    for pattern in phrase_patterns:
        matches = re.findall(pattern, compact)
        if matches:
            return _normalize_triplet(matches[-1])
    return EXTRACTION_FAILED


def _build_templated_prompts(tokenizer, rows: list[dict[str, str]]) -> list[str]:
    prompts = []
    for row in rows:
        messages = [{"role": "user", "content": build_prompt(row)}]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(prompt)
    return prompts


def _split_triplet(text: str) -> list[str] | None:
    match = TRIPLE_LABEL_RE.match(str(text).strip())
    if not match:
        return None
    return list(match.groups())


def _compute_accuracy(results: list[dict[str, object]], *, include_failed: bool) -> tuple[float, int, float]:
    total_samples = 0
    total_score = 0.0
    for row in results:
        pred = row.get("predicted_label")
        ref = row.get("reference_label")
        ref_parts = _split_triplet(str(ref))
        pred_parts = _split_triplet(str(pred))
        if ref_parts is None:
            continue
        if not include_failed and pred == EXTRACTION_FAILED:
            continue
        total_samples += 1
        if pred_parts is None:
            continue
        row_score = sum(1 for a, b in zip(pred_parts, ref_parts) if a == b) / 3.0
        total_score += row_score
    return total_score, total_samples, (total_score / total_samples) if total_samples else 0.0


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Qwen2.5-7B-Instruct with vLLM on quotes+summary dataset.")
    parser.add_argument("--base_model", type=str, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter_path", type=str, default=DEFAULT_ADAPTER_PATH)
    parser.add_argument("--dataset", type=str, default=str(DEFAULT_DATASET))
    parser.add_argument("--output_json", type=str, default=str(DEFAULT_OUTPUT_JSON))
    parser.add_argument("--num_samples", type=int, default=0, help="Number of random samples to evaluate. 0 means use all rows.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--use_all", action="store_true", help="Evaluate the entire dataset instead of random sampling.")
    parser.add_argument("--tensor_parallel_size", type=int, default=4)
    parser.add_argument("--pipeline_parallel_size", type=int, default=2)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_model_len", type=int, default=32768)
    parser.add_argument("--max_lora_rank", type=int, default=64)
    return parser


def main() -> None:
    args = build_argparser().parse_args()

    dataset_path = Path(args.dataset).expanduser().resolve()
    output_json = Path(args.output_json).expanduser().resolve()
    adapter_path = Path(args.adapter_path).expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)

    rows = _load_rows(dataset_path)
    if not rows:
        raise SystemExit(f"No rows found in dataset: {dataset_path}")
    if not (adapter_path / "adapter_model.safetensors").is_file():
        raise SystemExit(f"LoRA adapter not found or incomplete: {adapter_path}")

    if args.use_all or args.num_samples <= 0:
        sampled_rows = rows
    else:
        num_samples = min(args.num_samples, len(rows))
        rng = random.Random(args.seed)
        sampled_rows = rng.sample(rows, num_samples)

    tokenizer = AutoTokenizer.from_pretrained(str(adapter_path), use_fast=True, trust_remote_code=True)
    prompts = _build_templated_prompts(tokenizer, sampled_rows)

    config = AutoConfig.from_pretrained(args.base_model, trust_remote_code=True)
    num_attention_heads = int(getattr(config, "num_attention_heads", 0) or 0)
    if num_attention_heads and num_attention_heads % args.tensor_parallel_size != 0:
        valid = [d for d in range(1, num_attention_heads + 1) if num_attention_heads % d == 0]
        raise SystemExit(
            "Invalid tensor parallel setting for this model: "
            f"num_attention_heads={num_attention_heads}, tensor_parallel_size={args.tensor_parallel_size}. "
            f"Valid tensor_parallel_size values are {valid}. "
            "If you want to use 8 GPUs on Qwen2.5-7B-Instruct, use for example "
            "--tensor_parallel_size 4 --pipeline_parallel_size 2."
        )

    llm = LLM(
        model=args.base_model,
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        enable_lora=True,
        max_lora_rank=args.max_lora_rank,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
    )

    print(f"[eval] Generating with {args.base_model} + LoRA {adapter_path} on {len(prompts)} prompts ...")
    outputs = llm.generate(
        prompts,
        sampling_params,
        lora_request=LoRARequest("sft_adapter", 1, str(adapter_path)),
        use_tqdm=True,
    )

    results: list[dict[str, object]] = []
    for row, prompt, output in zip(sampled_rows, prompts, outputs):
        text = output.outputs[0].text
        pred = extract_predicted_label(text)
        results.append(
            {
                "date": row["date"],
                "stock": row["stock"],
                "reference_label": row["future_1_3_7_trade_day_labels"],
                "predicted_label": pred,
                "raw_output": text,
                "prompt": build_prompt(row),
            }
        )

    correct_all, total_all, acc_all = _compute_accuracy(results, include_failed=True)
    correct_valid, total_valid, acc_valid = _compute_accuracy(results, include_failed=False)

    payload = {
        "base_model": args.base_model,
        "adapter_path": str(adapter_path),
        "dataset": str(dataset_path),
        "num_samples": len(results),
        "accuracy_including_extraction_failures": {
            "correct": correct_all,
            "total": total_all,
            "accuracy": acc_all,
        },
        "accuracy_excluding_extraction_failures": {
            "correct": correct_valid,
            "total": total_valid,
            "accuracy": acc_valid,
        },
        "results": results,
    }
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote evaluation JSON to {output_json}")
    print(f"Average score (including extraction failures): {acc_all:.2%} ({correct_all:.4f}/{total_all})")
    print(f"Average score (excluding extraction failures): {acc_valid:.2%} ({correct_valid:.4f}/{total_valid})")


if __name__ == "__main__":
    main()
