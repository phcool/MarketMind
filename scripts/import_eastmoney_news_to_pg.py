"""
Import eastmoney json files into the news table
with 6-digit stock symbol.
"""

import json
import re
import sys
from pathlib import Path

import psycopg2
from psycopg2.extras import execute_values

BASE_DIR = Path(__file__).resolve().parent.parent / "A股重点板块"
DSN = "dbname=financial_data"
BATCH = 500
STOCK_RE = re.compile(r"\(([^)]+)\)$")

SQL = """
    INSERT INTO news (url, symbol, title, date)
    VALUES %s
    ON CONFLICT (url) DO NOTHING
"""


def parse_symbol(stock_folder_name: str) -> str | None:
    m = STOCK_RE.search(stock_folder_name)
    if not m:
        return None
    raw = m.group(1).split('.')[0]
    # only keep A-share 6-digit codes
    if len(raw) == 6 and raw.isdigit():
        return raw
    return None


def iter_rows(base_dir: Path):
    for sector_dir in sorted(base_dir.iterdir()):
        if not sector_dir.is_dir():
            continue
        for stock_dir in sorted(sector_dir.iterdir()):
            if not stock_dir.is_dir():
                continue
            symbol = parse_symbol(stock_dir.name)
            if not symbol:
                continue
            news_dir = stock_dir / "news"
            if not news_dir.is_dir():
                continue
            for json_file in sorted(news_dir.rglob("eastmoney.json")):
                try:
                    entries = json.loads(json_file.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    print(f"[warn] {json_file}: {exc}", file=sys.stderr)
                    continue
                for e in entries:
                    url = (e.get("url") or "").strip()
                    if not url:
                        continue
                    yield (
                        url,
                        symbol,
                        (e.get("title") or "").strip(),
                        e.get("date") or None,
                    )


def main() -> None:
    conn = psycopg2.connect(DSN)
    conn.autocommit = False
    cur = conn.cursor()

    inserted = 0
    skipped = 0
    batch_rows = []

    def flush() -> None:
        nonlocal inserted, skipped
        if not batch_rows:
            return
        execute_values(cur, SQL, batch_rows)
        n = max(cur.rowcount, 0)
        inserted += n
        skipped += len(batch_rows) - n
        batch_rows.clear()

    print("Importing data...")
    for row in iter_rows(BASE_DIR):
        batch_rows.append(row)
        if len(batch_rows) >= BATCH:
            flush()
            conn.commit()

    flush()
    conn.commit()
    cur.close()
    conn.close()

    print(f"Done. Inserted: {inserted:,} | Skipped(duplicates): {skipped:,}")


if __name__ == "__main__":
    main()
