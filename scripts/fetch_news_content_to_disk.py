"""
Fetch full news article body from East Money HTML (div#ContentBody) for selected stocks.

Reads URLs from UTF-8 CSV (default: exports/news.csv).
Saves text under:
  Content/news/{YYYY-MM}/{sha256(url).hex}.txt

File format (same idea as report cache):
  URL=...
  SYMBOL=...
  TITLE=...
  DATE=...
  ---
  <body paragraphs joined by newlines>

Resume: skips rows whose output file already exists and has non-empty body after '---',
and URLs already present with non-empty body anywhere under Content/news/ (full scan).

HTTP 429/456/503: exponential backoff + extra delay before retry.

Writes checkpoint/fetch_news_content_disk.json when the run finishes.

Requires: requests, beautifulsoup4.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import time
from datetime import date, datetime, timezone
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from stock_universe import all_symbols, load_sectors

ROOT_DIR = Path(__file__).resolve().parent.parent
CONTENT_NEWS_DIR = ROOT_DIR / "Content" / "news"
CHECKPOINT_DIR = ROOT_DIR / "checkpoint"
NEWS_FETCH_CHECKPOINT_JSON = CHECKPOINT_DIR / "fetch_news_content_disk.json"
DEFAULT_NEWS_CSV = ROOT_DIR / "exports" / "news.csv"

HTTP_THROTTLE_CODES = frozenset({429, 456, 503})

DEFAULT_SECTOR = "电力设备与新能源"
DEFAULT_SINCE = date(2025, 11, 1)
ALL_SINCE = date(2025, 1, 1)  # --all: from 2025-01-01 unless --since given

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Referer": "https://www.eastmoney.com/",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def url_to_relpath(url: str, news_date: date) -> Path:
    """year_month folder + hash filename."""
    ym = f"{news_date.year}-{news_date.month:02d}"
    h = hashlib.sha256(url.encode("utf-8")).hexdigest()
    return Path(ym) / f"{h}.txt"


def output_path(url: str, news_date: date) -> Path:
    return CONTENT_NEWS_DIR / url_to_relpath(url, news_date)


def scan_nonempty_news_urls(content_dir: Path) -> set[str]:
    """URLs that already have a non-empty cached body anywhere under content_dir."""
    out: set[str] = set()
    if not content_dir.is_dir():
        return out
    for p in content_dir.rglob("*.txt"):
        if not is_already_done(p):
            continue
        try:
            raw = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line in raw.splitlines()[:20]:
            s = line.strip()
            if s.startswith("URL="):
                out.add(s[4:].strip())
                break
    return out


def is_already_done(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return False
    if "URL=" not in raw or "\n---\n" not in raw:
        return False
    _, _, body = raw.partition("\n---\n")
    return bool(body.strip())


def extract_content_body(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    node = soup.select_one("#ContentBody") or soup.select_one("div.txtinfos#ContentBody")
    if not node:
        return ""
    parts: list[str] = []
    for p in node.find_all("p"):
        t = p.get_text(separator="", strip=True)
        if t:
            parts.append(t)
    if parts:
        return "\n\n".join(parts)
    return node.get_text(separator="\n", strip=True)


def fetch_news_body_with_retries(
    session: requests.Session,
    url: str,
    *,
    timeout: float,
    retry_base: float,
    max_attempts: int,
    log_label: str,
) -> str | None:
    """
    GET page and extract #ContentBody. On failure or empty body, sleep retry_base * 2^(k-1)
    before the next attempt (k = failed attempt index, 1-based).
    """
    for attempt in range(1, max_attempts + 1):
        try:
            resp = session.get(url, timeout=timeout)
            resp.raise_for_status()
            enc = resp.encoding or "utf-8"
            if resp.apparent_encoding:
                try:
                    resp.encoding = resp.apparent_encoding
                except Exception:
                    resp.encoding = enc
            html = resp.text
            body = extract_content_body(html)
            if body:
                return body
            log.warning(
                "%s empty #ContentBody (attempt %d/%d) %s",
                log_label,
                attempt,
                max_attempts,
                url[:100],
            )
        except Exception as exc:
            log.warning(
                "%s attempt %d/%d failed %s: %s",
                log_label,
                attempt,
                max_attempts,
                url[:80],
                exc,
            )
        if attempt < max_attempts:
            wait = retry_base * (2 ** (attempt - 1))
            log.info("%s retry in %.1fs", log_label, wait)
            time.sleep(wait)
    return None


def build_file_content(
    url: str,
    symbol: str,
    title: str,
    news_date: date,
    body: str,
) -> str:
    title_one = (title or "").replace("\r", " ").replace("\n", " ").strip()
    return (
        f"URL={url}\n"
        f"SYMBOL={symbol}\n"
        f"TITLE={title_one}\n"
        f"DATE={news_date.isoformat()}\n"
        f"---\n"
        f"{body.strip()}\n"
    )


def _parse_news_date(s: str) -> date | None:
    s = (s or "").strip()
    if not s:
        return None
    try:
        return date.fromisoformat(s[:10])
    except ValueError:
        return None


def load_rows_from_csv(
    csv_path: Path,
    symbols: set[str],
    since: date,
) -> list[tuple[str, str, str, date]]:
    if not csv_path.is_file():
        raise SystemExit(f"CSV not found: {csv_path}")
    required = {"url", "symbol", "title", "date"}
    out: list[tuple[str, str, str, date]] = []
    with csv_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
            raise SystemExit(
                f"CSV {csv_path} must have columns {sorted(required)}; got {reader.fieldnames}"
            )
        for row in reader:
            url = (row.get("url") or "").strip()
            sym = (row.get("symbol") or "").strip()
            title = row.get("title") or ""
            d = _parse_news_date(row.get("date") or "")
            if not url or not sym or d is None:
                continue
            if sym not in symbols:
                continue
            if d < since:
                continue
            out.append((url, sym, str(title), d))
    out.sort(key=lambda r: (r[3], r[0]))
    return out


def symbols_for_sector(sector: str) -> list[str]:
    sectors = load_sectors()
    if sector not in sectors:
        raise SystemExit(f"Unknown sector {sector!r}; available: {list(sectors.keys())}")
    return [t[1] for t in sectors[sector]]


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch news HTML body to Content/news/{YYYY-MM}/")
    ap.add_argument(
        "--sector",
        default=DEFAULT_SECTOR,
        help=f"Sector name in config/stocks.json (default: {DEFAULT_SECTOR})",
    )
    ap.add_argument(
        "--since",
        default=None,
        help="Minimum news.date (YYYY-MM-DD). Default: with --all → 2025-01-01; else → 2025-11-01.",
    )
    ap.add_argument(
        "--all",
        action="store_true",
        help="All symbols in config/stocks.json (~35); from 2025-01-01 onward unless --since is set.",
    )
    ap.add_argument(
        "--symbols",
        default="",
        help="Comma-separated symbols; if set, overrides --sector (ignored with --all).",
    )
    ap.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_NEWS_CSV,
        help=f"news CSV path (default: {DEFAULT_NEWS_CSV})",
    )
    ap.add_argument("--delay", type=float, default=1.0, help="Seconds after each finished URL (success or give-up).")
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
    ap.add_argument("--timeout", type=float, default=30.0, help="Per-request timeout seconds.")
    ap.add_argument(
        "--force",
        action="store_true",
        help="Re-fetch even if output file already has body.",
    )
    ap.add_argument("--dry-run", action="store_true", help="List work only, no HTTP writes.")
    args = ap.parse_args()

    since_s = (args.since or "").strip()
    if args.all:
        symbols = all_symbols()
        since = date.fromisoformat(since_s) if since_s else ALL_SINCE
    elif args.symbols.strip():
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
        since = date.fromisoformat(since_s) if since_s else DEFAULT_SINCE
    else:
        symbols = symbols_for_sector(args.sector)
        since = date.fromisoformat(since_s) if since_s else DEFAULT_SINCE

    CONTENT_NEWS_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = args.csv.resolve()
    rows = load_rows_from_csv(csv_path, set(symbols), since)

    log.info("Loaded %d news rows (symbols=%s, since=%s)", len(rows), symbols, since)

    session = requests.Session()
    session.headers.update(HEADERS)
    retry_base = args.retry_base_delay if args.retry_base_delay is not None else args.delay
    max_attempts = max(1, args.max_attempts)

    ok = skip = fail = 0
    for i, (url, sym, title, d) in enumerate(rows, 1):
        path = output_path(url, d)
        if not args.force and is_already_done(path):
            skip += 1
            continue
        if args.dry_run:
            log.info("[%d/%d] would fetch %s -> %s", i, len(rows), url[:80], path)
            continue

        path.parent.mkdir(parents=True, exist_ok=True)
        label = f"[{i}/{len(rows)}]"
        body = fetch_news_body_with_retries(
            session,
            url,
            timeout=args.timeout,
            retry_base=retry_base,
            max_attempts=max_attempts,
            log_label=label,
        )
        if body:
            text = build_file_content(url, sym, title, d, body)
            path.write_text(text, encoding="utf-8")
            ok += 1
            if i % 20 == 0:
                log.info("progress ok=%d skip=%d fail=%d [%d/%d]", ok, skip, fail, i, len(rows))
        else:
            log.warning("%s gave up after %d attempts %s", label, max_attempts, url[:80])
            fail += 1
        time.sleep(args.delay)

    log.info("done ok=%d skip=%d fail=%d total=%d", ok, skip, fail, len(rows))


if __name__ == "__main__":
    main()
