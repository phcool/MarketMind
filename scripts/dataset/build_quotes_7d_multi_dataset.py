"""
Build a pure 7-day K-line multi-horizon direction dataset from exports/quotes.csv.

Each sample uses Day1-Day7 as normalized K-line context, then predicts whether
the close price after 1 / 3 / 7 future trading days is up or down relative to
Day7 close.

Output columns:
  - prompt
  - future_1_3_7_trade_day_labels   (e.g. 涨，跌，涨)
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from dataset.build_quotes_7d_dataset import (
    StockNormalizer,
    _as_date,
    load_quotes_until,
    series_in_label_window,
)

DEFAULT_TRAIN_OUT = ROOT_DIR / "dataset" / "quotes_7d_multi_pre2026_dataset.csv"
DEFAULT_EVAL_OUT = ROOT_DIR / "dataset" / "quotes_7d_multi_eval_20260101_20260228.csv"

HEADER = """过去7个交易日的K线数据如下(归一化后)：
其中，open 表示开盘价，high 表示最高价，low 表示最低价，close 表示收盘价，volume 表示成交量，amplitude 表示振幅，pct_change 表示涨跌幅，turnover 表示换手率。
"""

TAIL = """请根据上文 Day1–Day7 的归一化 K 线，预测未来 1 个、3 个、7 个交易日相对 Day7 收盘价是涨还是跌。

【输出要求（必须严格遵守）】
1. 先在“【思维链开始】”与“【思维链结束】”之间给出简短分析。
2. 然后严格按照下面三行格式输出最终结论：
未来1个交易日：涨/跌
未来3个交易日：涨/跌
未来7个交易日：涨/跌
3. 最终结论只能使用“涨”或“跌”，不要输出其他同义词。"""


def close_compare_to_label(current_close, future_close) -> str | None:
    if current_close in (None, "") or future_close in (None, ""):
        return None
    try:
        cur = float(current_close)
        fut = float(future_close)
    except (TypeError, ValueError):
        return None
    return "涨" if fut >= cur else "跌"


def build_prompt(norm: StockNormalizer, prev7: list[dict]) -> str:
    lines = [
        HEADER.rstrip(),
        *[norm.row_to_day_line(i + 1, prev7[i]) for i in range(7)],
        "",
        TAIL,
    ]
    return "\n".join(lines)


def build_dataset(date_start: date, label_before: date, quotes_before: date, output: Path) -> int:
    if date_start >= label_before:
        raise SystemExit("--date-start must be < --label-before")

    output.parent.mkdir(parents=True, exist_ok=True)
    groups = load_quotes_until(quotes_before)

    n_out = 0
    with output.open("w", encoding="utf-8", newline="") as f_out:
        w = csv.writer(f_out, quoting=csv.QUOTE_MINIMAL)
        w.writerow(["prompt", "future_1_3_7_trade_day_labels"])

        for _symbol, series in groups:
            if len(series) < 14:
                continue
            fit_rows = series_in_label_window(series, date_start, label_before)
            if not fit_rows:
                continue
            norm = StockNormalizer.from_series(fit_rows)
            for idx in range(6, len(series) - 7):
                current = series[idx]
                td = current["trade_date"]
                if not (date_start <= td < label_before):
                    continue
                prev7 = series[idx - 6 : idx + 1]
                if len(prev7) != 7:
                    continue
                future_1 = close_compare_to_label(current.get("close"), series[idx + 1].get("close"))
                future_3 = close_compare_to_label(current.get("close"), series[idx + 3].get("close"))
                future_7 = close_compare_to_label(current.get("close"), series[idx + 7].get("close"))
                if None in {future_1, future_3, future_7}:
                    continue
                labels = f"{future_1}，{future_3}，{future_7}"
                prompt = build_prompt(norm, prev7)
                w.writerow([prompt, labels])
                n_out += 1
    return n_out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build 7-day K-line -> future 1/3/7-day direction dataset.",
    )
    ap.add_argument("--date-start", default="2021-01-01")
    ap.add_argument("--label-before", default="2026-01-01")
    ap.add_argument(
        "--quotes-before",
        default="",
        help="Optional quote loading cutoff (exclusive). Default: label-before + 14 days.",
    )
    ap.add_argument("-o", "--output", type=Path, default=DEFAULT_TRAIN_OUT)
    args = ap.parse_args()

    date_start = _as_date(args.date_start)
    label_before = _as_date(args.label_before)
    if args.quotes_before.strip():
        quotes_before = _as_date(args.quotes_before)
    else:
        quotes_before = label_before + timedelta(days=14)
    n_out = build_dataset(date_start, label_before, quotes_before, args.output)
    print(
        f"Wrote {n_out} rows to {args.output} "
        f"(label days in [{date_start.isoformat()}, {label_before.isoformat()}), "
        "labels compare Day7 close to future 1/3/7 trading-day closes)",
    )


if __name__ == "__main__":
    main()
