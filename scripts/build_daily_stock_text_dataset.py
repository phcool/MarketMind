from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

from stock_universe import load_sectors

ROOT_DIR = Path(__file__).resolve().parent.parent
NEWS_CSV = ROOT_DIR / "exports" / "news.csv"
REPORT_CSV = ROOT_DIR / "exports" / "report.csv"
CONTENT_NEWS_DIR = ROOT_DIR / "Content" / "news"
CONTENT_REPORT_DIR = ROOT_DIR / "Content" / "report"
OUTPUT_CSV = ROOT_DIR / "exports" / "daily_stock_text_dataset_2026-01-01_to_2026-04-01.csv"

START_DATE = date(2026, 1, 1)
END_DATE = date(2026, 4, 1)
MAX_NEWS = 10
MAX_REPORTS = 3


@dataclass(frozen=True)
class ContentItem:
    symbol: str
    item_date: date
    url: str
    title: str
    content: str


def url_hash(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


def parse_date_str(raw: str) -> date | None:
    raw = (raw or "").strip()
    if not raw:
        return None
    try:
        return date.fromisoformat(raw[:10])
    except ValueError:
        return None


def news_output_path(url: str, item_date: date) -> Path:
    ym = f"{item_date.year}-{item_date.month:02d}"
    return CONTENT_NEWS_DIR / ym / f"{url_hash(url)}.txt"


def report_output_path(url: str) -> Path:
    return CONTENT_REPORT_DIR / f"{url_hash(url)}.txt"


def parse_content_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return None
    if "URL=" not in raw or "\n---\n" not in raw:
        return None
    _, _, body = raw.partition("\n---\n")
    content = body.strip()
    return content or None


def load_symbol_metadata() -> dict[str, dict[str, str]]:
    metadata: dict[str, dict[str, str]] = {}
    for sector, companies in load_sectors().items():
        for name, symbol, market in companies:
            metadata[symbol] = {
                "name": name,
                "sector": sector,
                "market": market,
            }
    return metadata


def load_news_items() -> dict[str, list[ContentItem]]:
    items_by_symbol: dict[str, list[ContentItem]] = {}
    seen_urls: set[tuple[str, date]] = set()
    with NEWS_CSV.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            symbol = (row.get("symbol") or "").strip()
            url = (row.get("url") or "").strip()
            title = (row.get("title") or "").strip()
            item_date = parse_date_str(row.get("date") or "")
            if not symbol or not url or item_date is None:
                continue
            dedupe_key = (url, item_date)
            if dedupe_key in seen_urls:
                continue
            seen_urls.add(dedupe_key)
            content = parse_content_file(news_output_path(url, item_date))
            if not content:
                continue
            items_by_symbol.setdefault(symbol, []).append(
                ContentItem(
                    symbol=symbol,
                    item_date=item_date,
                    url=url,
                    title=title,
                    content=content,
                )
            )
    for symbol in items_by_symbol:
        items_by_symbol[symbol].sort(key=lambda item: (item.item_date, item.url), reverse=True)
    return items_by_symbol


def load_report_items() -> dict[str, list[ContentItem]]:
    items_by_symbol: dict[str, list[ContentItem]] = {}
    seen_urls: set[str] = set()
    with REPORT_CSV.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            symbol = (row.get("symbol") or "").strip()
            url = (row.get("url") or "").strip()
            title = (row.get("title") or "").strip()
            item_date = parse_date_str(row.get("date") or "")
            if not symbol or not url or item_date is None:
                continue
            if url in seen_urls:
                continue
            seen_urls.add(url)
            content = parse_content_file(report_output_path(url))
            if not content:
                continue
            items_by_symbol.setdefault(symbol, []).append(
                ContentItem(
                    symbol=symbol,
                    item_date=item_date,
                    url=url,
                    title=title,
                    content=content,
                )
            )
    for symbol in items_by_symbol:
        items_by_symbol[symbol].sort(key=lambda item: (item.item_date, item.url), reverse=True)
    return items_by_symbol


def iter_dates(start: date, end: date) -> list[date]:
    dates: list[date] = []
    current = start
    while current <= end:
        dates.append(current)
        current += timedelta(days=1)
    return dates


def latest_items(items: list[ContentItem], cutoff: date, limit: int) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for item in items:
        if item.item_date > cutoff:
            continue
        out.append(
            {
                "date": item.item_date.isoformat(),
                "title": item.title,
                "url": item.url,
                "content": item.content,
            }
        )
        if len(out) >= limit:
            break
    return out


def main() -> None:
    symbol_metadata = load_symbol_metadata()
    ordered_symbols = list(symbol_metadata.keys())
    news_by_symbol = load_news_items()
    reports_by_symbol = load_report_items()
    dates = iter_dates(START_DATE, END_DATE)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["stock", "time", "news", "reports"])
        writer.writeheader()
        for current_date in dates:
            for symbol in ordered_symbols:
                writer.writerow(
                    {
                        "stock": symbol,
                        "time": current_date.isoformat(),
                        "news": json.dumps(
                            latest_items(news_by_symbol.get(symbol, []), current_date, MAX_NEWS),
                            ensure_ascii=False,
                        ),
                        "reports": json.dumps(
                            latest_items(reports_by_symbol.get(symbol, []), current_date, MAX_REPORTS),
                            ensure_ascii=False,
                        ),
                    }
                )


if __name__ == "__main__":
    main()
