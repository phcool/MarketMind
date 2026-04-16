"""
Build DashScope Batch JSONL from Content/*.txt, then optionally upload and run batch
inference with qwen3.5-flash (OpenAI-compatible API). Each request sends one full
news/report content file and asks the model to summarize its main points.

Outputs are written into Summary/ with the same relative file structure as Content/,
while preserving metadata headers and replacing the body with the summary text.

Usage:
  # First run: build JSONL from unsummarized Content files, upload batch job, save state
  python scripts/batch_qwen_summary_content.py

  # Later: poll once; if completed, download output JSONL and write Summary files
  python scripts/batch_qwen_summary_content.py --check-status

Optional env:
  BATCH_SUMMARY_LIMIT   subset size for testing
  BATCH_SUMMARY_OFFSET  skip first N eligible files
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent


def _load_dotenv() -> None:
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

CONTENT_DIR = ROOT_DIR / "Content"
SUMMARY_DIR = ROOT_DIR / "Summary"
DEFAULT_JSONL = ROOT_DIR / "batch" / "summary_qwen_batch_input.jsonl"
DEFAULT_RESULT = ROOT_DIR / "batch" / "summary_qwen_batch_result.jsonl"
DEFAULT_ERROR = ROOT_DIR / "batch" / "summary_qwen_batch_error.jsonl"
STATE_FILE = ROOT_DIR / "batch" / "summary_qwen_batch_state.json"
DEFAULT_MODEL = "qwen3.5-flash"

DASHSCOPE_BASE_URL = os.environ.get(
    "DASHSCOPE_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
)

SYSTEM_PROMPT = """You are a helpful financial text summarization assistant. Summarize the user's provided news/report in concise Chinese. Focus on the main观点、关键事实、结论与影响，不要编造，不要遗漏时间信息。"""


@dataclass(frozen=True)
class ContentRecord:
    relative_path: str
    source_path: Path
    summary_path: Path
    metadata: dict[str, str]
    body: str


def parse_content_file(path: Path) -> tuple[dict[str, str], str]:
    raw = path.read_text(encoding="utf-8")
    if "\n---\n" not in raw:
        raise ValueError(f"Invalid content file format (missing separator): {path}")
    head, _, body = raw.partition("\n---\n")
    metadata: dict[str, str] = {}
    for line in head.splitlines():
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        metadata[key.strip()] = value.strip()
    body = body.strip()
    if not body:
        raise ValueError(f"Empty content body: {path}")
    return metadata, body


def format_summary_body(metadata: dict[str, str], summary: str) -> str:
    date_s = metadata.get("DATE", "")
    title = metadata.get("TITLE", "")
    lines: list[str] = []
    if date_s:
        lines.append(f"时间：{date_s}")
    if title:
        lines.append(f"标题：{title}")
    lines.append("主要观点：")
    lines.append(summary.strip())
    return "\n".join(lines).strip() + "\n"


def build_summary_file_text(metadata: dict[str, str], summary: str) -> str:
    header_lines: list[str] = []
    for key in ("URL", "SYMBOL", "TITLE", "DATE"):
        value = metadata.get(key)
        if value:
            header_lines.append(f"{key}={value}")
    return "\n".join(header_lines) + "\n---\n" + format_summary_body(metadata, summary)


def build_user_content(record: ContentRecord) -> str:
    date_s = record.metadata.get("DATE", "")
    title = record.metadata.get("TITLE", "")
    kind = "news" if record.relative_path.startswith("news/") else "report"
    return (
        f"下面是一篇{kind}全文，请你用中文总结其主要观点。\n\n"
        "要求：\n"
        "1. 总结要准确、简洁，不要编造。\n"
        "2. 突出核心事实、主要结论、相关影响。\n"
        "3. 保留时间信息，如果原文里有明确日期/时间，请在总结中体现。\n"
        "4. 只输出总结正文，不要加前缀说明，不要输出 JSON。\n\n"
        f"原文日期：{date_s}\n"
        f"原文标题：{title}\n"
        "原文如下：\n"
        f"{record.body}"
    )


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


def iter_content_records(*, offset: int, limit: int | None) -> list[ContentRecord]:
    all_paths = sorted(p for p in CONTENT_DIR.rglob("*.txt") if p.is_file())
    eligible: list[ContentRecord] = []
    for path in all_paths:
        rel = path.relative_to(CONTENT_DIR).as_posix()
        summary_path = SUMMARY_DIR / rel
        if summary_path.is_file():
            continue
        metadata, body = parse_content_file(path)
        eligible.append(
            ContentRecord(
                relative_path=rel,
                source_path=path,
                summary_path=summary_path,
                metadata=metadata,
                body=body,
            )
        )
    if offset:
        eligible = eligible[offset:]
    if limit is not None:
        eligible = eligible[:limit]
    return eligible


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
    jsonl_path: Path,
    *,
    model: str,
    offset: int,
    limit: int | None,
) -> int:
    records = iter_content_records(offset=offset, limit=limit)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with jsonl_path.open("w", encoding="utf-8") as out:
        for record in records:
            out.write(
                build_jsonl_line(
                    custom_id=record.relative_path,
                    model=model,
                    user_content=build_user_content(record),
                )
                + "\n"
            )
            n += 1
    return n


def write_summary_files_from_batch_result(result_path: Path) -> int:
    by_id = parse_batch_result_jsonl(result_path)
    written = 0
    for relative_path, summary in by_id.items():
        if not summary.strip():
            continue
        source_path = CONTENT_DIR / relative_path
        if not source_path.is_file():
            continue
        metadata, _body = parse_content_file(source_path)
        summary_path = SUMMARY_DIR / relative_path
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            build_summary_file_text(metadata, summary),
            encoding="utf-8",
        )
        written += 1
    return written


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
    def _one(name: str) -> int | None:
        raw = os.environ.get(name)
        if raw is None or not str(raw).strip():
            return None
        try:
            return int(str(raw).strip())
        except ValueError as exc:
            raise SystemExit(f"{name} must be an integer") from exc

    off = _one("BATCH_SUMMARY_OFFSET")
    if off is None:
        off = 0
    lim = _one("BATCH_SUMMARY_LIMIT")
    return off, lim


def submit_batch_job(jsonl_path: Path) -> str:
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
    if not CONTENT_DIR.is_dir():
        raise SystemExit(f"Content directory not found: {CONTENT_DIR}")

    offset, limit = _limit_offset_from_env()
    n = write_batch_jsonl(
        DEFAULT_JSONL,
        model=DEFAULT_MODEL,
        offset=offset,
        limit=limit,
    )
    if n == 0:
        print("No unsummarized Content files found. Nothing to submit.")
        return
    print(f"Wrote {n} batch lines to {DEFAULT_JSONL}")

    batch_id = submit_batch_job(DEFAULT_JSONL)
    _write_state(
        {
            "batch_id": batch_id,
            "content_dir": str(CONTENT_DIR.resolve()),
            "summary_dir": str(SUMMARY_DIR.resolve()),
            "jsonl_out": str(DEFAULT_JSONL.resolve()),
            "result_out": str(DEFAULT_RESULT.resolve()),
            "error_out": str(DEFAULT_ERROR.resolve()),
            "offset": offset,
            "limit": limit,
            "model": DEFAULT_MODEL,
        }
    )
    print(f"State saved to {STATE_FILE}")
    print("When the job finishes on the server, run: python scripts/batch_qwen_summary_content.py --check-status")


def run_check_status_flow() -> None:
    state = _read_state()
    batch_id = state["batch_id"]
    result_path = Path(state["result_out"])
    error_path = Path(state["error_out"])

    print(f"Checking batch {batch_id} ...")
    if not check_once_download_if_completed(batch_id, result_path, error_path):
        return

    if not result_path.is_file():
        raise SystemExit(f"Expected result file missing: {result_path}")

    n = write_summary_files_from_batch_result(result_path)
    print(f"Wrote {n} summary files into {SUMMARY_DIR}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Batch summarize Content/*.txt with qwen3.5-flash and write mirrored outputs into Summary/.",
    )
    ap.add_argument(
        "--check-status",
        action="store_true",
        help="Query batch once; if completed, download results and write Summary files.",
    )
    args = ap.parse_args()

    if args.check_status:
        run_check_status_flow()
    else:
        run_submit_flow()


if __name__ == "__main__":
    main()
