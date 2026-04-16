from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from stock_universe import load_sectors

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_NEWS_CSV = ROOT_DIR / "exports" / "news.csv"
DEFAULT_REPORT_CSV = ROOT_DIR / "exports" / "report.csv"
DEFAULT_OUTPUT_JSON = ROOT_DIR / "exports" / "daily_news_report_overlap_from_2026.json"
DEFAULT_START_DATE = date(2026, 1, 1)


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


def collect_daily_counts(csv_path: Path, start_date: date) -> tuple[Counter[tuple[str, str]], date | None]:
    counts: Counter[tuple[str, str]] = Counter()
    max_seen_date: date | None = None
    with csv_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            symbol = (row.get("symbol") or "").strip()
            raw_date = (row.get("date") or "").strip()[:10]
            if not symbol or len(raw_date) != 10:
                continue
            try:
                row_date = date.fromisoformat(raw_date)
            except ValueError:
                continue
            if row_date < start_date:
                continue
            counts[(symbol, raw_date)] += 1
            if max_seen_date is None or row_date > max_seen_date:
                max_seen_date = row_date
    return counts, max_seen_date


def iter_weekdays(start_date: date, end_date: date) -> list[date]:
    days: list[date] = []
    current = start_date
    while current <= end_date:
        if current.weekday() < 5:
            days.append(current)
        current += timedelta(days=1)
    return days


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export same-day news/report overlap stats by stock for weekdays from 2026 onward."
    )
    parser.add_argument("--news-csv", type=Path, default=DEFAULT_NEWS_CSV)
    parser.add_argument("--report-csv", type=Path, default=DEFAULT_REPORT_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument(
        "--start-date",
        default=DEFAULT_START_DATE.isoformat(),
        help="Inclusive start date in YYYY-MM-DD format.",
    )
    parser.add_argument("--top-n", type=int, default=50)
    args = parser.parse_args()

    start_date = date.fromisoformat(args.start_date)
    symbol_metadata = load_symbol_metadata()

    news_counts, news_max = collect_daily_counts(args.news_csv.resolve(), start_date)
    report_counts, report_max = collect_daily_counts(args.report_csv.resolve(), start_date)
    end_date_candidates = [d for d in (news_max, report_max) if d is not None]
    if not end_date_candidates:
        raise SystemExit("No rows found on or after start date in either CSV.")
    end_date = max(end_date_candidates)

    weekdays = iter_weekdays(start_date, end_date)

    missing_both: list[dict[str, object]] = []
    overlap_rows: list[dict[str, object]] = []
    for day in weekdays:
        day_s = day.isoformat()
        for symbol in sorted(symbol_metadata):
            news_count = news_counts.get((symbol, day_s), 0)
            report_count = report_counts.get((symbol, day_s), 0)
            meta = symbol_metadata[symbol]
            if news_count == 0 and report_count == 0:
                missing_both.append(
                    {
                        "date": day_s,
                        "symbol": symbol,
                        "name": meta["name"],
                        "sector": meta["sector"],
                        "market": meta["market"],
                    }
                )
            if news_count > 0 and report_count > 0:
                overlap_rows.append(
                    {
                        "date": day_s,
                        "symbol": symbol,
                        "name": meta["name"],
                        "sector": meta["sector"],
                        "market": meta["market"],
                        "news_count": news_count,
                        "report_count": report_count,
                        "combined_count": news_count + report_count,
                    }
                )

    overlap_rows.sort(
        key=lambda item: (
            -int(item["combined_count"]),
            -int(item["news_count"]),
            -int(item["report_count"]),
            str(item["date"]),
            str(item["symbol"]),
        )
    )

    missing_by_symbol: Counter[str] = Counter()
    for row in missing_both:
        missing_by_symbol[row["symbol"]] += 1

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "weekday_range": {
            "start": start_date.isoformat(),
            "end": end_date.isoformat(),
            "weekday_count": len(weekdays),
        },
        "summary": {
            "tracked_symbol_count": len(symbol_metadata),
            "missing_both_count": len(missing_both),
            "overlap_count": len(overlap_rows),
        },
        "missing_both": {
            "all_rows": missing_both,
            "by_symbol": [
                {
                    "symbol": symbol,
                    "name": symbol_metadata[symbol]["name"],
                    "sector": symbol_metadata[symbol]["sector"],
                    "market": symbol_metadata[symbol]["market"],
                    "missing_days": count,
                }
                for symbol, count in sorted(
                    missing_by_symbol.items(),
                    key=lambda item: (-item[1], item[0]),
                )
            ],
        },
        "top_overlap_rows": overlap_rows[: max(0, args.top_n)],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
