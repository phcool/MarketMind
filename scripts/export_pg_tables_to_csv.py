"""
Export PostgreSQL tables comments, news, report to UTF-8 CSV files with headers.

Usage:
  python scripts/export_pg_tables_to_csv.py
  python scripts/export_pg_tables_to_csv.py --out /path/to/dir

Connection: PG_DSN env or default dbname=financial_data.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import psycopg2

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DSN = os.environ.get("PG_DSN", "dbname=financial_data")

TABLES = ("comments", "news", "report", "quotes")


def export_table(cur, table: str, out_path: Path) -> int:
    cur.execute(f"SELECT * FROM {table}")
    colnames = [d[0] for d in cur.description]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(colnames)
        for row in cur:
            w.writerow(row)
            n += 1
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description="Export comments, news, report tables to CSV.")
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT_DIR / "exports",
        help=f"Output directory (default: {ROOT_DIR / 'exports'})",
    )
    parser.add_argument("--dsn", default=DEFAULT_DSN, help="psycopg2 connection string")
    args = parser.parse_args()
    out_dir = args.out.resolve()

    try:
        conn = psycopg2.connect(args.dsn)
    except Exception as exc:
        print(f"DB connection failed: {exc}", file=sys.stderr)
        sys.exit(1)

    cur = conn.cursor()
    try:
        for t in TABLES:
            path = out_dir / f"{t}.csv"
            n = export_table(cur, t, path)
            print(f"{t}: {n:,} rows -> {path}")
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    main()
