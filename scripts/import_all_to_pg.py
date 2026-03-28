"""
Import local JSON data into PostgreSQL: comments (forum), news (eastmoney), report (sina).

Delegates row iteration to the existing import modules; runs one or all targets in a single command.

Usage (from repo root or scripts/):
  python scripts/import_all_to_pg.py
  python scripts/import_all_to_pg.py --only news
  python scripts/import_all_to_pg.py --dsn "dbname=financial_data host=localhost"
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import psycopg2
from psycopg2.extras import execute_values

# Reuse iterators from sibling scripts (same BASE_DIR layout).
import import_eastmoney_news_to_pg as _em_news
import import_forum_to_pg as _forum
import import_sina_to_pg as _sina

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_BASE = ROOT_DIR / "A股重点板块"
DEFAULT_DSN = os.environ.get("PG_DSN", "dbname=financial_data")
BATCH = 500

SQL_COMMENTS = """
    INSERT INTO comments
        (url, post_id, symbol, post_title, publish_time, click_count, comment_count)
    VALUES %s
    ON CONFLICT (url) DO NOTHING
"""

SQL_NEWS = """
    INSERT INTO news (url, symbol, title, date)
    VALUES %s
    ON CONFLICT (url) DO NOTHING
"""

SQL_REPORT = """
    INSERT INTO report (url, symbol, title, date)
    VALUES %s
    ON CONFLICT (url) DO NOTHING
"""


def _flush_batch(cur, conn, sql: str, batch: list[tuple], stats: dict[str, int], label: str) -> None:
    if not batch:
        return
    execute_values(cur, sql, batch)
    n = max(cur.rowcount, 0)
    stats["inserted"] += n
    stats["skipped"] += len(batch) - n
    batch.clear()
    conn.commit()
    print(f"  [{label}] inserted: {stats['inserted']:,}  skipped: {stats['skipped']:,}", end="\r")


def import_comments(conn, base_dir: Path) -> tuple[int, int]:
    stats = {"inserted": 0, "skipped": 0}
    batch: list[tuple] = []
    cur = conn.cursor()
    for row in _forum.iter_rows(base_dir):
        batch.append(row)
        if len(batch) >= BATCH:
            _flush_batch(cur, conn, SQL_COMMENTS, batch, stats, "comments")
    _flush_batch(cur, conn, SQL_COMMENTS, batch, stats, "comments")
    cur.close()
    print()
    return stats["inserted"], stats["skipped"]


def import_news(conn, base_dir: Path) -> tuple[int, int]:
    stats = {"inserted": 0, "skipped": 0}
    batch: list[tuple] = []
    cur = conn.cursor()
    for row in _em_news.iter_rows(base_dir):
        batch.append(row)
        if len(batch) >= BATCH:
            _flush_batch(cur, conn, SQL_NEWS, batch, stats, "news")
    _flush_batch(cur, conn, SQL_NEWS, batch, stats, "news")
    cur.close()
    print()
    return stats["inserted"], stats["skipped"]


def import_reports(conn, base_dir: Path) -> tuple[int, int]:
    stats = {"inserted": 0, "skipped": 0}
    batch: list[tuple] = []
    cur = conn.cursor()
    for row in _sina.iter_rows(base_dir):
        batch.append(row)
        if len(batch) >= BATCH:
            _flush_batch(cur, conn, SQL_REPORT, batch, stats, "report")
    _flush_batch(cur, conn, SQL_REPORT, batch, stats, "report")
    cur.close()
    print()
    return stats["inserted"], stats["skipped"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Import forum comments, eastmoney news, sina reports to PG.")
    parser.add_argument(
        "--base",
        type=Path,
        default=DEFAULT_BASE,
        help=f"Root folder of sector/stock trees (default: {DEFAULT_BASE})",
    )
    parser.add_argument("--dsn", default=DEFAULT_DSN, help="psycopg2 connection string (default: PG_DSN or dbname=financial_data)")
    parser.add_argument(
        "--only",
        choices=("all", "comments", "news", "report"),
        default="all",
        help="Which dataset to import (default: all)",
    )
    args = parser.parse_args()

    base_dir = args.base.resolve()
    if not base_dir.is_dir():
        print(f"Base directory not found: {base_dir}", file=sys.stderr)
        sys.exit(1)

    conn = psycopg2.connect(args.dsn)
    conn.autocommit = False

    summary: list[tuple[str, int, int]] = []

    try:
        if args.only in ("all", "comments"):
            print("Importing comments (forum JSON under forum/<year>/...) …")
            summary.append(("comments", *import_comments(conn, base_dir)))
        if args.only in ("all", "news"):
            print("Importing news (eastmoney.json under news/...) …")
            summary.append(("news", *import_news(conn, base_dir)))
        if args.only in ("all", "report"):
            print("Importing report (sina.json under news/...) …")
            summary.append(("report", *import_reports(conn, base_dir)))
    finally:
        conn.close()

    print("---")
    for name, ins, sk in summary:
        print(f"{name}: inserted {ins:,} | skipped (duplicates) {sk:,}")
    print("Done.")


if __name__ == "__main__":
    main()
