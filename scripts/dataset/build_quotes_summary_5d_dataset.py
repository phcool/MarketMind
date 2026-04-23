from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from dataset.build_quotes_7d_dataset import StockNormalizer
from stock_universe import all_symbols

QUOTES_CSV = ROOT_DIR / "exports" / "quotes.csv"
REPORT_CSV = ROOT_DIR / "exports" / "report.csv"
SUMMARY_NEWS_DIR = ROOT_DIR / "Summary" / "news"
SUMMARY_REPORT_DIR = ROOT_DIR / "Summary" / "report"
DEFAULT_OUTPUT = ROOT_DIR / "dataset" / "quotes_summary_5d_2026-01-01_to_2026-04-01.csv"

DATE_START = date(2026, 1, 1)
DATE_END = date(2026, 4, 1)
NEWS_LIMIT = 30
REPORT_LIMIT = 3
LOOKBACK_DAYS = 5
MAX_AGE_DAYS = 30


@dataclass(frozen=True)
class SummaryItem:
    symbol: str
    item_date: date
    url: str
    title: str
    summary: str


def parse_date_str(raw: str) -> date | None:
    raw = (raw or "").strip()
    if not raw:
        return None
    try:
        return date.fromisoformat(raw[:10])
    except ValueError:
        return None


def pct_change_to_label(raw: str | None) -> str | None:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    try:
        value = float(s)
    except ValueError:
        return None
    return "涨" if value >= 0 else "跌"


def close_compare_to_label(current_close: str | None, future_close: str | None) -> str | None:
    if current_close is None or future_close is None:
        return None
    current_s = str(current_close).strip()
    future_s = str(future_close).strip()
    if not current_s or not future_s:
        return None
    try:
        current_v = float(current_s)
        future_v = float(future_s)
    except ValueError:
        return None
    return "涨" if future_v >= current_v else "跌"


def parse_summary_file(path: Path) -> tuple[dict[str, str], str]:
    raw = path.read_text(encoding="utf-8")
    if "\n---\n" not in raw:
        raise ValueError(f"Invalid summary file format: {path}")
    head, _, body = raw.partition("\n---\n")
    metadata: dict[str, str] = {}
    for line in head.splitlines():
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        metadata[key.strip()] = value.strip()
    return metadata, body.strip()


def load_report_symbol_by_url() -> dict[str, str]:
    mapping: dict[str, str] = {}
    with REPORT_CSV.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            url = (row.get("url") or "").strip()
            symbol = (row.get("symbol") or "").strip()
            if url and symbol and url not in mapping:
                mapping[url] = symbol
    return mapping


def load_news_summaries(symbols: set[str]) -> dict[str, list[SummaryItem]]:
    out: dict[str, list[SummaryItem]] = {}
    for path in sorted(SUMMARY_NEWS_DIR.rglob("*.txt")):
        metadata, body = parse_summary_file(path)
        symbol = (metadata.get("SYMBOL") or "").strip()
        url = (metadata.get("URL") or "").strip()
        title = (metadata.get("TITLE") or "").strip()
        item_date = parse_date_str(metadata.get("DATE") or "")
        if not symbol or symbol not in symbols or not url or item_date is None or not body:
            continue
        out.setdefault(symbol, []).append(
            SummaryItem(
                symbol=symbol,
                item_date=item_date,
                url=url,
                title=title,
                summary=body,
            )
        )
    for symbol in out:
        out[symbol].sort(key=lambda item: (item.item_date, item.url), reverse=True)
    return out


def load_report_summaries(symbols: set[str]) -> dict[str, list[SummaryItem]]:
    url_to_symbol = load_report_symbol_by_url()
    out: dict[str, list[SummaryItem]] = {}
    for path in sorted(SUMMARY_REPORT_DIR.rglob("*.txt")):
        metadata, body = parse_summary_file(path)
        url = (metadata.get("URL") or "").strip()
        title = (metadata.get("TITLE") or "").strip()
        item_date = parse_date_str(metadata.get("DATE") or "")
        symbol = url_to_symbol.get(url, "").strip()
        if not symbol or symbol not in symbols or not url or item_date is None or not body:
            continue
        out.setdefault(symbol, []).append(
            SummaryItem(
                symbol=symbol,
                item_date=item_date,
                url=url,
                title=title,
                summary=body,
            )
        )
    for symbol in out:
        out[symbol].sort(key=lambda item: (item.item_date, item.url), reverse=True)
    return out


def load_quotes() -> list[tuple[str, list[dict]]]:
    by_symbol: dict[str, list[dict]] = {}
    wanted = set(all_symbols())
    with QUOTES_CSV.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            symbol = (row.get("symbol") or "").strip()
            if symbol not in wanted:
                continue
            trade_date = parse_date_str(row.get("trade_date") or "")
            if trade_date is None:
                continue
            rec = {
                "trade_date": trade_date,
                "open": row.get("open"),
                "high": row.get("high"),
                "low": row.get("low"),
                "close": row.get("close"),
                "volume": row.get("volume"),
                "amplitude": row.get("amplitude"),
                "pct_change": row.get("pct_change"),
                "turnover": row.get("turnover"),
            }
            by_symbol.setdefault(symbol, []).append(rec)
    for symbol in by_symbol:
        by_symbol[symbol].sort(key=lambda row: row["trade_date"])
    return sorted(by_symbol.items(), key=lambda item: item[0])


def series_in_window(series: list[dict], start: date, end: date) -> list[dict]:
    return [row for row in series if start <= row["trade_date"] <= end]


def normalized_kline_rows(norm: StockNormalizer, rows: list[dict]) -> list[dict[str, object]]:
    payload: list[dict[str, object]] = []
    for row in rows:
        payload.append(
            {
                "trade_date": row["trade_date"].isoformat(),
                "open": round(norm.norm_open(row.get("open")) or 0.0, 4),
                "high": round(norm.norm_high(row.get("high")) or 0.0, 4),
                "low": round(norm.norm_low(row.get("low")) or 0.0, 4),
                "close": round(norm.norm_close(row.get("close")) or 0.0, 4),
                "volume": round(norm.norm_volume(row.get("volume")) or 0.0, 4),
                "amplitude": row.get("amplitude"),
                "pct_change": round(norm.norm_pct_change(row.get("pct_change")) or 0.0, 4),
                "turnover": round(norm.norm_turnover(row.get("turnover")) or 0.0, 4),
            }
        )
    return payload


def recent_items(items: list[SummaryItem], cutoff: date, limit: int) -> list[dict[str, str]]:
    earliest = cutoff - timedelta(days=MAX_AGE_DAYS)
    out: list[dict[str, str]] = []
    for item in items:
        if item.item_date > cutoff:
            continue
        if item.item_date < earliest:
            continue
        out.append(
            {
                "date": item.item_date.isoformat(),
                "title": item.title,
                "url": item.url,
                "summary": item.summary,
            }
        )
        if len(out) >= limit:
            break
    return out


def main() -> None:
    output = DEFAULT_OUTPUT
    output.parent.mkdir(parents=True, exist_ok=True)

    quote_groups = load_quotes()
    symbol_set = set(all_symbols())
    news_by_symbol = load_news_summaries(symbol_set)
    reports_by_symbol = load_report_summaries(symbol_set)

    n_out = 0
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(
            [
                "date",
                "stock",
                "kline_5d",
                "news",
                "reports",
                "future_1_3_7_trade_day_labels",
            ]
        )

        for symbol, series in quote_groups:
            fit_rows = series_in_window(series, DATE_START, DATE_END)
            if not fit_rows:
                continue
            norm = StockNormalizer.from_series(fit_rows)
            for idx in range(LOOKBACK_DAYS - 1, len(series) - 7):
                current = series[idx]
                current_date = current["trade_date"]
                if not (DATE_START <= current_date <= DATE_END):
                    continue
                prev5 = series[idx - (LOOKBACK_DAYS - 1) : idx + 1]
                if len(prev5) != LOOKBACK_DAYS:
                    continue
                future_1 = close_compare_to_label(current.get("close"), series[idx + 1].get("close"))
                future_3 = close_compare_to_label(current.get("close"), series[idx + 3].get("close"))
                future_7 = close_compare_to_label(current.get("close"), series[idx + 7].get("close"))
                if None in {future_1, future_3, future_7}:
                    continue
                future_labels = f"{future_1}，{future_3}，{future_7}"
                writer.writerow(
                    [
                        current_date.isoformat(),
                        symbol,
                        json.dumps(
                            normalized_kline_rows(norm, prev5),
                            ensure_ascii=False,
                        ),
                        json.dumps(
                            recent_items(news_by_symbol.get(symbol, []), current_date, NEWS_LIMIT),
                            ensure_ascii=False,
                        ),
                        json.dumps(
                            recent_items(reports_by_symbol.get(symbol, []), current_date, REPORT_LIMIT),
                            ensure_ascii=False,
                        ),
                        future_labels,
                    ]
                )
                n_out += 1

    print(f"Wrote {n_out} rows to {output}")


if __name__ == "__main__":
    main()
