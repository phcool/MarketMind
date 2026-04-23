"""
Build DashScope Batch JSONL from quotes_7d dataset (prompt + label), then optionally
upload and run batch inference with qwen3-max (OpenAI-compatible API).

User message format:
  - Shows the original K-line task text and the gold label (涨/跌).
  - Asks the model to write reasoning inside `</think>` ... `</think>`, then output the same correct answer on the last line.

Requires: openai package; env DASHBOARD_API_KEY or DASHSCOPE_API_KEY (repo .env loaded).
See docs/Qwen_Batch_Call.md for Batch workflow.

Usage:
  # First run: build JSONL from default dataset, upload batch job, save state (does not wait for completion)
  python scripts/batch/batch_qwen_cot_quotes_dataset.py

  # Later: poll until done, download result JSONL, merge into CoT dataset CSV (prompt + completion)
  python scripts/batch/batch_qwen_cot_quotes_dataset.py --check-status

Optional env (first run only): BATCH_COT_LIMIT, BATCH_COT_OFFSET (subset of rows for testing).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def _load_dotenv() -> None:
    """Load KEY=value from repo root .env into os.environ (no override if already set)."""
    path = ROOT_DIR / ".env"
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


def _dashscope_api_key() -> str | None:
    return os.getenv("DASHBOARD_API_KEY") or os.getenv("DASHSCOPE_API_KEY")


_load_dotenv()

DEFAULT_DATASET = ROOT_DIR / "dataset" / "quotes_7d_pre2026_dataset.csv"
DEFAULT_JSONL = ROOT_DIR / "batch" / "cot_qwen_batch_input.jsonl"
DEFAULT_RESULT = ROOT_DIR / "batch" / "cot_qwen_batch_result.jsonl"
DEFAULT_ERROR = ROOT_DIR / "batch" / "cot_qwen_batch_error.jsonl"
DEFAULT_COT_DATASET = ROOT_DIR / "dataset" / "quotes_7d_cot_from_batch.csv"
STATE_FILE = ROOT_DIR / "batch" / "cot_qwen_batch_state.json"
DEFAULT_MODEL = "qwen3-max"

DASHSCOPE_BASE_URL = os.environ.get(
    "DASHSCOPE_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
)

SYSTEM_PROMPT = """You are a helpful assistant for Chinese A-share market education. Follow the user's instructions exactly. Use Chinese for reasoning unless asked otherwise."""

# Delimiters for the reasoning block (explicit paired tags; avoid ambiguous think tokens).
REASONING_BEGIN = "【思维链开始】"
REASONING_END = "【思维链结束】"


def build_user_content(original_prompt: str, gold_label: str) -> str:
    """Wrap dataset fields into a single user message for CoT + final answer."""
    gold = (gold_label or "").strip()
    if gold not in ("涨", "跌"):
        raise ValueError(f"label must be 涨 or 跌, got {gold!r}")
    return (
        "下面是一道「根据过去7个交易日归一化K线，预测下一交易日涨跌方向」的数据集样本。\n\n"
        "【本题在数据集中的标准答案(第八个交易日相对第七个交易日涨跌方向)】\n"
        f"{gold}\n\n"
        "【原任务说明与K线特征（与训练数据中的 prompt 字段一致）】\n"
        f"{original_prompt.strip()}\n\n"
        "请你**在已经知道标准答案的前提下**，用中文写出一段**合理、简洁**的思维链：说明如何从上述K线信息（趋势、量能、波动等）出发，能够**自然推出**该涨跌方向。\n\n"
        "思维链必须写在下面一对标记之间（只写推理过程)：\n"
        f"{REASONING_BEGIN}\n"
        "（在此撰写推理过程）\n"
        f"{REASONING_END}\n\n"
        "思维链结束后，请**单独一行**输出最终答案，且必须与标准答案**完全一致**，仅一个字：「涨」或「跌」，不要引号、标点或其他文字。"
    )


# Must match scripts/dataset/build_quotes_7d_dataset.py TAIL (single-char output instruction).
_ORIGINAL_PROMPT_TAIL = """请根据上文 Day1–Day7 的归一化 K 线，只预测下一个交易日（Day8）相对前一日是涨还是跌。

【输出要求（必须严格遵守）】
只输出一个字：「涨」或「跌」，不要输出其他任何文字、数字、标点或换行。"""


def build_cot_train_prompt(original_prompt: str) -> str:
    """Replace the original one-token output instruction with CoT-style instructions (for training CSV column `prompt`)."""
    text = original_prompt.replace("\r\n", "\n").strip()
    if _ORIGINAL_PROMPT_TAIL not in text:
        print(
            "Warning: expected one-line-output tail not found in prompt; appending CoT instructions.",
            file=sys.stderr,
        )
        return (
            text
            + "\n\n"
            + "请根据上文 Day1–Day7 的归一化 K 线，预测下一个交易日（Day8）相对前一日是涨还是跌。\n\n"
            + "【输出要求（必须严格遵守）】\n"
            + "请先分析 K 线趋势、量能与波动；将推理过程写在下面一对标记之间（只写推理过程）：\n"
            + f"{REASONING_BEGIN}\n"
            + "（在此撰写推理过程）\n"
            + f"{REASONING_END}\n\n"
            + "思维链结束后，请单独一行输出最终答案，仅一个字：「涨」或「跌」，不要引号、标点或其他文字。"
        )
    cot_tail = (
        "请根据上文 Day1–Day7 的归一化 K 线，预测下一个交易日（Day8）相对前一日是涨还是跌。\n\n"
        "【输出要求（必须严格遵守）】\n"
        "请先分析 K 线趋势、量能与波动；将推理过程写在下面一对标记之间（只写推理过程）：\n"
        f"{REASONING_BEGIN}\n"
        "（在此撰写推理过程）\n"
        f"{REASONING_END}\n\n"
        "思维链结束后，请单独一行输出最终答案，仅一个字：「涨」或「跌」，不要引号、标点或其他文字。"
    )
    return text.replace(_ORIGINAL_PROMPT_TAIL, cot_tail, 1)


def _extract_completion_from_batch_line(obj: dict) -> str:
    if obj.get("error"):
        return ""
    resp = obj.get("response")
    if not isinstance(resp, dict):
        return ""
    if resp.get("status_code") != 200:
        return ""
    body = resp.get("body") or {}
    choices = body.get("choices") or []
    if not choices:
        return ""
    msg = (choices[0] or {}).get("message") or {}
    return (msg.get("content") or "").strip()


def parse_batch_result_jsonl(path: Path) -> dict[str, str]:
    """Map custom_id -> assistant text from DashScope batch output JSONL."""
    out: dict[str, str] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            cid = obj.get("custom_id")
            if cid is None:
                continue
            out[str(cid)] = _extract_completion_from_batch_line(obj)
    return out


def write_cot_dataset_csv(
    dataset_path: Path,
    batch_result_jsonl: Path,
    out_path: Path,
    *,
    offset: int,
    limit: int | None,
) -> int:
    """Write two-column CSV: prompt (CoT-style task) + completion (qwen batch output)."""
    rows, _ = iter_dataset_rows(dataset_path, offset=offset, limit=limit)
    by_id = parse_batch_result_jsonl(batch_result_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["prompt", "completion"])
        for i, row in enumerate(rows):
            cid = str(offset + i)
            prompt = build_cot_train_prompt(row["prompt"])
            completion = by_id.get(cid, "")
            w.writerow([prompt, completion])
            n += 1
    return n


def iter_dataset_rows(
    path: Path,
    *,
    offset: int,
    limit: int | None,
) -> tuple[list[dict[str, str]], int]:
    """Read CSV with multiline quoted fields. Returns list of {prompt, label} and total row count scanned."""
    rows: list[dict[str, str]] = []
    total = 0
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "prompt" not in reader.fieldnames or "label" not in reader.fieldnames:
            raise SystemExit(f"CSV must have columns prompt, label; got {reader.fieldnames}")
        for row in reader:
            total += 1
            if total <= offset:
                continue
            p = (row.get("prompt") or "").strip()
            lab = (row.get("label") or "").strip()
            if not p or lab not in ("涨", "跌"):
                continue
            rows.append({"prompt": p, "label": lab})
            if limit is not None and len(rows) >= limit:
                break
    return rows, total


def build_jsonl_line(custom_id: str, model: str, user_content: str) -> str:
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    }
    req = {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": body,
    }
    return json.dumps(req, ensure_ascii=False)


def write_batch_jsonl(
    dataset_path: Path,
    jsonl_path: Path,
    *,
    model: str,
    offset: int,
    limit: int | None,
) -> int:
    rows, _total = iter_dataset_rows(dataset_path, offset=offset, limit=limit)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with jsonl_path.open("w", encoding="utf-8") as out:
        for i, row in enumerate(rows):
            custom_id = f"{offset + i}"
            user_content = build_user_content(row["prompt"], row["label"])
            out.write(build_jsonl_line(custom_id, model, user_content) + "\n")
            n += 1
    return n


def _openai_client():
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit("Please install openai: pip install openai") from exc
    api_key = _dashscope_api_key()
    if not api_key:
        raise SystemExit("Set DASHBOARD_API_KEY or DASHSCOPE_API_KEY for batch submission.")
    return OpenAI(api_key=api_key, base_url=DASHSCOPE_BASE_URL)


def _limit_offset_from_env() -> tuple[int, int | None]:
    """Optional BATCH_COT_OFFSET (default 0) and BATCH_COT_LIMIT (subset for testing)."""

    def _one(name: str) -> int | None:
        raw = os.environ.get(name)
        if raw is None or not str(raw).strip():
            return None
        try:
            return int(str(raw).strip())
        except ValueError as exc:
            raise SystemExit(f"{name} must be an integer") from exc

    off = _one("BATCH_COT_OFFSET")
    if off is None:
        off = 0
    lim = _one("BATCH_COT_LIMIT")
    return off, lim


def submit_batch_job(jsonl_path: Path, *, model: str) -> str:
    """Upload JSONL and create batch job; return batch id (does not poll)."""
    client = _openai_client()
    print(f"Uploading {jsonl_path} ...")
    file_object = client.files.create(file=jsonl_path, purpose="batch")
    input_file_id = file_object.id
    print(f"Input file id: {input_file_id}")

    print("Creating batch job (endpoint=/v1/chat/completions) ...")
    batch = client.batches.create(
        input_file_id=input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    print(f"Batch id: {batch.id}")
    return batch.id


_TERMINAL = frozenset({"completed", "failed", "expired", "cancelled"})


def check_once_download_if_completed(
    batch_id: str,
    result_path: Path,
    error_path: Path,
) -> bool:
    """
    Query batch status once. If not terminal, print status and return False.
    If failed/expired/cancelled, print and exit 1. If completed, download files and return True.
    """
    client = _openai_client()
    batch = client.batches.retrieve(batch_id=batch_id)
    status = batch.status
    if status not in _TERMINAL:
        print(f"Current status: {status}. Not finished yet; run --check-status again later.")
        return False

    if status != "completed":
        print(f"Batch ended with status={status}, errors={getattr(batch, 'errors', None)}")
        sys.exit(1)

    result_path.parent.mkdir(parents=True, exist_ok=True)
    if batch.output_file_id:
        print(f"Downloading success output to {result_path} ...")
        content = client.files.content(batch.output_file_id)
        content.write_to_file(result_path)
        print("Done.")
    if batch.error_file_id:
        print(f"Downloading errors to {error_path} ...")
        err = client.files.content(batch.error_file_id)
        err.write_to_file(error_path)
    return True


def _write_state(payload: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_state() -> dict:
    if not STATE_FILE.is_file():
        raise SystemExit(
            f"No batch state file ({STATE_FILE}). Run this script without arguments first to submit a job."
        )
    return json.loads(STATE_FILE.read_text(encoding="utf-8"))


def run_submit_flow() -> None:
    if not DEFAULT_DATASET.is_file():
        raise SystemExit(f"Dataset not found: {DEFAULT_DATASET}")

    offset, limit = _limit_offset_from_env()
    n = write_batch_jsonl(
        DEFAULT_DATASET,
        DEFAULT_JSONL,
        model=DEFAULT_MODEL,
        offset=offset,
        limit=limit,
    )
    print(f"Wrote {n} batch lines to {DEFAULT_JSONL}")

    batch_id = submit_batch_job(DEFAULT_JSONL, model=DEFAULT_MODEL)
    _write_state(
        {
            "batch_id": batch_id,
            "input": str(DEFAULT_DATASET.resolve()),
            "jsonl_out": str(DEFAULT_JSONL.resolve()),
            "result_out": str(DEFAULT_RESULT.resolve()),
            "error_out": str(DEFAULT_ERROR.resolve()),
            "cot_dataset_out": str(DEFAULT_COT_DATASET.resolve()),
            "offset": offset,
            "limit": limit,
            "model": DEFAULT_MODEL,
        }
    )
    print(f"State saved to {STATE_FILE}")
    print("When the job finishes on the server, run: python scripts/batch/batch_qwen_cot_quotes_dataset.py --check-status")


def run_check_status_flow() -> None:
    state = _read_state()
    batch_id = state["batch_id"]
    result_path = Path(state["result_out"])
    error_path = Path(state["error_out"])
    dataset_path = Path(state["input"])
    cot_out = Path(state["cot_dataset_out"])
    offset = int(state.get("offset", 0))
    limit = state.get("limit")
    if limit is not None:
        limit = int(limit)

    print(f"Checking batch {batch_id} ...")
    if not check_once_download_if_completed(batch_id, result_path, error_path):
        return

    if not result_path.is_file():
        raise SystemExit(f"Expected result file missing: {result_path}")

    n = write_cot_dataset_csv(dataset_path, result_path, cot_out, offset=offset, limit=limit)
    print(f"Wrote CoT dataset ({n} rows) to {cot_out}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="CoT batch on quotes_7d: default = build JSONL + submit job; --check-status = one-shot status, download+merge if completed.",
    )
    ap.add_argument(
        "--check-status",
        action="store_true",
        help="Query batch once; if completed, download results and write prompt+completion CSV.",
    )
    args = ap.parse_args()

    if args.check_status:
        run_check_status_flow()
    else:
        run_submit_flow()


if __name__ == "__main__":
    main()
