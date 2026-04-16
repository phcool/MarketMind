from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from stock_universe import load_sectors

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_NEWS_CSV = ROOT_DIR / "exports" / "news.csv"
DEFAULT_REPORT_CSV = ROOT_DIR / "exports" / "report.csv"
DEFAULT_OUTPUT_JSON = ROOT_DIR / "exports" / "monthly_symbol_counts_from_2026.json"
DEFAULT_START_MONTH = "2026-01"


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


def collect_counts(csv_path: Path, start_month: str) -> tuple[dict[str, Counter[str]], Counter[str]]:
    counts_by_month: dict[str, Counter[str]] = defaultdict(Counter)
    totals_by_month: Counter[str] = Counter()

    with csv_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            symbol = (row.get("symbol") or "").strip()
            month = (row.get("date") or "").strip()[:7]
            if not symbol or len(month) != 7 or month < start_month:
                continue
            counts_by_month[month][symbol] += 1
            totals_by_month[month] += 1

    return dict(counts_by_month), totals_by_month


def build_source_payload(
    csv_path: Path,
    start_month: str,
    symbol_metadata: dict[str, dict[str, str]],
) -> dict[str, object]:
    counts_by_month, totals_by_month = collect_counts(csv_path, start_month)

    months_payload: dict[str, object] = {}
    for month in sorted(counts_by_month):
        month_total = totals_by_month[month]
        symbols_payload: dict[str, object] = {}
        for symbol, count in sorted(
            counts_by_month[month].items(),
            key=lambda item: (-item[1], item[0]),
        ):
            meta = symbol_metadata.get(symbol, {})
            share = count / month_total if month_total else 0.0
            symbols_payload[symbol] = {
                "name": meta.get("name", ""),
                "sector": meta.get("sector", ""),
                "market": meta.get("market", ""),
                "count": count,
                "share_of_month": round(share, 6),
                "percentage_of_month": round(share * 100, 4),
            }
        months_payload[month] = {
            "total_rows": month_total,
            "symbols": symbols_payload,
        }

    return {
        "csv_path": str(csv_path.relative_to(ROOT_DIR)),
        "start_month": start_month,
        "months": months_payload,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export per-month per-symbol counts from exports/news.csv and exports/report.csv."
    )
    parser.add_argument("--news-csv", type=Path, default=DEFAULT_NEWS_CSV)
    parser.add_argument("--report-csv", type=Path, default=DEFAULT_REPORT_CSV)
    parser.add_argument("--start-month", default=DEFAULT_START_MONTH, help="Inclusive YYYY-MM filter.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_JSON)
    args = parser.parse_args()

    symbol_metadata = load_symbol_metadata()
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "start_month": args.start_month,
        "sources": {
            "news": build_source_payload(args.news_csv.resolve(), args.start_month, symbol_metadata),
            "report": build_source_payload(args.report_csv.resolve(), args.start_month, symbol_metadata),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
