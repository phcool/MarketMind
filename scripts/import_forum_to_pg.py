"""
Import forum posts (2023-2026) from JSON files into the comments table.

Directory layout:
  A股重点板块/<sector>/<name(symbol)>/forum/<year>/<month>/<day>.json

Each JSON entry:
  post_id, post_title, url, post_publish_time, post_click_count, post_comment_count
"""

import json
import re
import sys
from pathlib import Path

import psycopg2
from psycopg2.extras import execute_values

BASE_DIR  = Path(__file__).resolve().parent.parent / "A股重点板块"
DSN       = "dbname=financial_data"
YEARS     = {"2023", "2024", "2025", "2026"}
BATCH     = 500   # rows per INSERT

SYMBOL_RE = re.compile(r"\((\d+)(?:\.\w+)?\)$")   # 宁德时代(300750) or 吉利汽车(00175.HK)


def extract_symbol(folder_name: str) -> str | None:
    m = SYMBOL_RE.search(folder_name)
    if not m:
        return None
    code = m.group(1)
    # Keep only A-share 6-digit stock codes
    if len(code) == 6 and code.isdigit():
        return code
    return None


def iter_rows(base_dir: Path):
    for sector_dir in sorted(base_dir.iterdir()):
        if not sector_dir.is_dir():
            continue
        for stock_dir in sorted(sector_dir.iterdir()):
            if not stock_dir.is_dir():
                continue
            symbol = extract_symbol(stock_dir.name)
            if symbol is None:
                print(f"  [skip] non-6-digit symbol: {stock_dir.name}")
                continue
            forum_dir = stock_dir / "forum"
            if not forum_dir.is_dir():
                continue
            for year_dir in sorted(forum_dir.iterdir()):
                if not year_dir.is_dir() or year_dir.name not in YEARS:
                    continue
                for json_file in sorted(year_dir.rglob("*.json")):
                    try:
                        entries = json.loads(json_file.read_text(encoding="utf-8"))
                    except (json.JSONDecodeError, OSError) as e:
                        print(f"  [warn] {json_file}: {e}", file=sys.stderr)
                        continue
                    for e in entries:
                        url = (e.get("url") or "").strip()
                        if not url:
                            continue
                        yield (
                            url,
                            str(e.get("post_id", "")).strip() or None,
                            symbol,
                            (e.get("post_title") or "").strip(),
                            e.get("post_publish_time"),
                            e.get("post_click_count"),
                            e.get("post_comment_count"),
                        )


def main():
    conn = psycopg2.connect(DSN)
    conn.autocommit = False
    cur = conn.cursor()

    SQL = """
        INSERT INTO comments
            (url, post_id, symbol, post_title, publish_time, click_count, comment_count)
        VALUES %s
        ON CONFLICT (url) DO NOTHING
    """

    total_inserted = 0
    total_skipped  = 0
    batch: list[tuple] = []

    def flush():
        nonlocal total_inserted, total_skipped
        if not batch:
            return
        before = cur.rowcount
        execute_values(cur, SQL, batch)
        inserted = cur.rowcount  # rows actually inserted (skipped = len(batch) - inserted)
        skipped  = len(batch) - max(inserted, 0)
        total_inserted += max(inserted, 0)
        total_skipped  += skipped
        batch.clear()

    print("Importing forum posts (2023-2026) …")
    for row in iter_rows(BASE_DIR):
        batch.append(row)
        if len(batch) >= BATCH:
            flush()
            conn.commit()
            print(f"  inserted so far: {total_inserted:,}  skipped: {total_skipped:,}", end="\r")

    flush()
    conn.commit()
    cur.close()
    conn.close()

    print(f"\nDone.  Inserted: {total_inserted:,}  |  Skipped (duplicates): {total_skipped:,}")


if __name__ == "__main__":
    main()
