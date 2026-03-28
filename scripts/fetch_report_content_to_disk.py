"""
Fetch Sina research report plaintext for selected stocks and cache under Content/report/.

Uses the same HTTP + HTML parsing as fetch_report_content.py (div.blk_container > p, gb2312).
Save format matches stock_daily_dashboard._write_report_cache_file:

  URL=...
  TITLE=...
  DATE=YYYY-MM-DD
  ---
  <body>

File path: Content/report/{sha256(url)}.txt (same as dashboard cache — no DB writes).

Data source: UTF-8 CSV (default exports/report.csv), e.g. from export_pg_tables_to_csv.py.

Resume: skip if file exists and body after '---' is non-empty.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

from fetch_report_content import fetch_report_worker_disk
from stock_universe import all_symbols, load_sectors

ROOT_DIR = Path(__file__).resolve().parent.parent
REPORT_CACHE_DIR = ROOT_DIR / "Content" / "report"
DEFAULT_REPORT_CSV = ROOT_DIR / "exports" / "report.csv"

DEFAULT_SECTOR = "电力设备与新能源"
ALL_SINCE = "2025-01-01"  # --all: minimum report.date unless --since given

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def report_cache_path(url: str) -> Path:
    key = hashlib.sha256(url.encode("utf-8")).hexdigest()
    return REPORT_CACHE_DIR / f"{key}.txt"


def cached_body_nonempty(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return False
    if not raw.startswith("URL=") or "\n---\n" not in raw:
        return False
    _, _, body = raw.partition("\n---\n")
    return bool(body.strip())


def write_report_cache(path: Path, page_url: str, title: str, report_date, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    date_s = str(report_date) if report_date is not None else ""
    meta = f"URL={page_url}\nTITLE={title}\nDATE={date_s}\n"
    path.write_text(meta + "---\n" + body, encoding="utf-8")


def symbols_for_sector(sector: str) -> list[str]:
    sectors = load_sectors()
    if sector not in sectors:
        raise SystemExit(f"Unknown sector {sector!r}; available: {list(sectors.keys())}")
    return [t[1] for t in sectors[sector]]


def _parse_report_date(s: str) -> date | None:
    s = (s or "").strip()
    if not s:
        return None
    try:
        return date.fromisoformat(s[:10])
    except ValueError:
        return None


def _report_sort_key(t: tuple[str, str, object]) -> tuple:
    url, _title, d = t
    if d is None:
        return (1, url)
    return (0, d, url)


def load_report_rows_from_csv(
    csv_path: Path,
    symbols: set[str],
    since: str | None,
) -> list[tuple[str, str, object]]:
    if not csv_path.is_file():
        raise SystemExit(f"CSV not found: {csv_path}")
    since_d = date.fromisoformat(since[:10]) if since else None
    required = {"url", "symbol", "title", "date"}
    rows: list[tuple[str, str, object]] = []
    with csv_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
            raise SystemExit(
                f"CSV {csv_path} must have columns {sorted(required)}; got {reader.fieldnames}"
            )
        for row in reader:
            url = (row.get("url") or "").strip()
            sym = (row.get("symbol") or "").strip()
            if not url or not sym:
                continue
            if sym not in symbols:
                continue
            d = _parse_report_date(row.get("date") or "")
            if since_d is not None:
                if d is None or d < since_d:
                    continue
            title = row.get("title") or ""
            rows.append((url, str(title), d))
    rows.sort(key=_report_sort_key)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch report bodies to Content/report/ (Sina, same as dashboard cache).")
    ap.add_argument("--sector", default=DEFAULT_SECTOR, help="Sector in config/stocks.json")
    ap.add_argument("--symbols", default="", help="Comma-separated symbols; overrides --sector if set.")
    ap.add_argument(
        "--since",
        default=None,
        help="Minimum report.date (YYYY-MM-DD). Default: with --all → 2025-01-01; else → no date filter (all dates).",
    )
    ap.add_argument(
        "--all",
        action="store_true",
        help="All symbols in config (~35); report rows from 2025-01-01 onward unless --since is set.",
    )
    ap.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_REPORT_CSV,
        help=f"report CSV path (default: {DEFAULT_REPORT_CSV})",
    )
    ap.add_argument("--workers", type=int, default=1, help="Concurrent fetches (default 1 for rate limits).")
    ap.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Seconds after each worker finishes one URL (polite pacing; same as fetch_report_content.REQUEST_DELAY).",
    )
    ap.add_argument(
        "--retry-base-delay",
        type=float,
        default=None,
        help="First backoff seconds after a failed attempt; doubles each retry. Default: same as --delay.",
    )
    ap.add_argument(
        "--max-attempts",
        type=int,
        default=5,
        help="Max HTTP attempts per URL (exponential backoff between failures).",
    )
    ap.add_argument("--timeout", type=float, default=20.0, help="Per-request timeout seconds.")
    ap.add_argument("--force", action="store_true", help="Re-fetch even when cache file has body.")
    ap.add_argument("--dry-run", action="store_true", help="List rows only.")
    args = ap.parse_args()

    since_s = (args.since or "").strip()
    if args.all:
        symbols = all_symbols()
        since = since_s or ALL_SINCE
    elif args.symbols.strip():
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
        since = since_s or None
    else:
        symbols = symbols_for_sector(args.sector)
        since = since_s or None

    csv_path = args.csv.resolve()
    rows = load_report_rows_from_csv(csv_path, set(symbols), since)

    log.info("Loaded %d report rows (symbols=%s, since=%s)", len(rows), symbols, since or "—")

    to_fetch: list[tuple[str, str, object]] = []
    skipped = 0
    for url, title, d in rows:
        path = report_cache_path(url)
        if not args.force and cached_body_nonempty(path):
            skipped += 1
            continue
        to_fetch.append((url, title, d))

    log.info("Skip (cached): %d  To fetch: %d", skipped, len(to_fetch))

    if args.dry_run:
        for url, title, d in to_fetch[:50]:
            log.info("would fetch %s | %s | %s", d, title[:40] if title else "", url[:80])
        if len(to_fetch) > 50:
            log.info("... and %d more", len(to_fetch) - 50)
        return

    retry_base = args.retry_base_delay if args.retry_base_delay is not None else args.delay
    max_attempts = max(1, args.max_attempts)

    done = 0
    failed = 0
    written = 0

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {}
        for url, title, d in to_fetch:
            fut = pool.submit(
                fetch_report_worker_disk,
                url,
                timeout=args.timeout,
                retry_base=retry_base,
                max_attempts=max_attempts,
                request_delay=args.delay,
            )
            futures[fut] = (url, title, d)
        for future in as_completed(futures):
            url, title, d = futures[future]
            done += 1
            try:
                _, content = future.result()
            except Exception as exc:
                log.warning("[%d/%d] exception %s: %s", done, len(to_fetch), url[:60], exc)
                failed += 1
                continue

            if content is None:
                log.warning("[%d/%d] transient fail %s", done, len(to_fetch), url[:80])
                failed += 1
                continue
            if not (content or "").strip():
                log.debug("[%d/%d] empty/unavailable %s", done, len(to_fetch), url[:80])
                continue

            path = report_cache_path(url)
            write_report_cache(path, url, title, d, content.strip())
            written += 1
            log.info("[%d/%d] OK chars=%d %s", done, len(to_fetch), len(content), url[:70])

    log.info("Finished. written=%d failed=%d processed=%d", written, failed, done)


if __name__ == "__main__":
    main()
