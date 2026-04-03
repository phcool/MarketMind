"""
Sample one random valid row from the quotes_7d CSV and call qwen3-max once
(OpenAI-compatible chat completion). Uses the same user message template as
`batch_qwen_cot_quotes_dataset.py`.

API key resolution matches `stock_daily_dashboard.py`: load repo `.env` (no override),
then use DASHBOARD_API_KEY if set, else DASHSCOPE_API_KEY.

Requires: openai package.

Usage:
  python scripts/sample_qwen_cot_one.py
  python scripts/sample_qwen_cot_one.py --seed 42
  python scripts/sample_qwen_cot_one.py --input train/dataset/quotes_7d_pre2026_dataset.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPTS_DIR.parent
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

from batch_qwen_cot_quotes_dataset import (  # noqa: E402
    DEFAULT_DATASET,
    DASHSCOPE_BASE_URL,
    SYSTEM_PROMPT,
    build_user_content,
)


def pick_random_row(path: Path, rng: random.Random) -> dict[str, str]:
    """Uniform random among rows with valid prompt and label (reservoir k=1, one pass)."""
    chosen: dict[str, str] | None = None
    n = 0
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "prompt" not in reader.fieldnames or "label" not in reader.fieldnames:
            raise SystemExit(f"CSV must have columns prompt, label; got {reader.fieldnames}")
        for row in reader:
            p = (row.get("prompt") or "").strip()
            lab = (row.get("label") or "").strip()
            if not p or lab not in ("涨", "跌"):
                continue
            n += 1
            if n == 1 or rng.random() < 1.0 / n:
                chosen = {"prompt": p, "label": lab}
    if chosen is None:
        raise SystemExit("No valid rows found in CSV.")
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description="One-shot qwen3-max CoT sample from quotes_7d CSV.")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_DATASET,
        help="CSV with prompt, label columns",
    )
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for reproducible sample")
    parser.add_argument("--model", default="qwen3-max", help="Chat model id on DashScope")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    row = pick_random_row(args.input, rng)

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit("Please install openai: pip install openai") from exc

    api_key = _dashscope_api_key()
    if not api_key:
        raise SystemExit("Set DASHBOARD_API_KEY or DASHSCOPE_API_KEY (e.g. in repo .env).")

    user_content = build_user_content(row["prompt"], row["label"])
    print("--- sampled label (gold) ---")
    print(row["label"])
    print("--- calling model ---")

    client = OpenAI(api_key=api_key, base_url=DASHSCOPE_BASE_URL)
    resp = client.chat.completions.create(
        model=args.model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    )
    text = (resp.choices[0].message.content or "").strip()
    print("--- assistant output ---")
    print(text)


if __name__ == "__main__":
    main()
