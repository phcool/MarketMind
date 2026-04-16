"""
Fetch news/report content to disk with a single entry point.

Modes:
  news   -> fetch East Money news body into Content/news/{YYYY-MM}/{sha256(url)}.txt
  report -> fetch Sina report body into Content/report/{sha256(url)}.txt
  all    -> run news first, then report

Special handling:
  - HTTP 404: delete the corresponding row from CSV, remove stale cache file, and skip
    immediately without retry/backoff and without the normal per-item delay.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from csv_io import delete_news_rows_by_url, delete_report_rows_by_url
from stock_universe import all_symbols, load_sectors

ROOT_DIR = Path(__file__).resolve().parent.parent
CONTENT_NEWS_DIR = ROOT_DIR / "Content" / "news"
CONTENT_REPORT_DIR = ROOT_DIR / "Content" / "report"
DEFAULT_NEWS_CSV = ROOT_DIR / "exports" / "news.csv"
DEFAULT_REPORT_CSV = ROOT_DIR / "exports" / "report.csv"

DEFAULT_SECTOR = "电力设备与新能源"
DEFAULT_NEWS_SINCE = date(2025, 11, 1)
ALL_SINCE = date(2025, 1, 1)

HTTP_THROTTLE_CODES = frozenset({429, 456, 503})

NEWS_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Referer": "https://www.eastmoney.com/",
}

REPORT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Referer": "https://stock.finance.sina.com.cn/",
    "Accept-Language": "zh-CN,zh;q=0.9",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


@dataclass(frozen=True)
class FetchResult:
    status: str
    body: str = ""
    http_status: int | None = None


def symbols_for_sector(sector: str) -> list[str]:
    sectors = load_sectors()
    if sector not in sectors:
        raise SystemExit(f"Unknown sector {sector!r}; available: {list(sectors.keys())}")
    return [t[1] for t in sectors[sector]]


def parse_date_str(s: str) -> date | None:
    s = (s or "").strip()
    if not s:
        return None
    try:
        return date.fromisoformat(s[:10])
    except ValueError:
        return None


def url_hash(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


def news_output_path(url: str, news_date: date) -> Path:
    ym = f"{news_date.year}-{news_date.month:02d}"
    return CONTENT_NEWS_DIR / ym / f"{url_hash(url)}.txt"


def report_output_path(url: str) -> Path:
    return CONTENT_REPORT_DIR / f"{url_hash(url)}.txt"


def file_has_nonempty_body(path: Path) -> bool:
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


def remove_file_if_exists(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except OSError as exc:
        log.warning("Failed to remove %s: %s", path, exc)


def extract_news_content_body(html: str) -> str:
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


def extract_report_content(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    container = soup.select_one("div.blk_container")
    if not container:
        container = soup.select_one("div.content")
    if not container:
        return ""
    parts: list[str] = []
    for p in container.find_all("p"):
        for br in p.find_all("br"):
            br.replace_with("\n")
        text = p.get_text(" ").strip()
        if text:
            parts.append(text)
    return "\n\n".join(parts)


def fetch_news_with_retries(
    session: requests.Session,
    url: str,
    *,
    timeout: float,
    retry_base: float,
    max_attempts: int,
    log_label: str,
) -> FetchResult:
    for attempt in range(1, max_attempts + 1):
        try:
            resp = session.get(url, timeout=timeout)
            code = resp.status_code
            if code == 404:
                log.warning("%s HTTP 404 delete+skip %s", log_label, url[:100])
                return FetchResult(status="not_found", http_status=404)
            if code in HTTP_THROTTLE_CODES:
                log.warning("%s HTTP %d throttle %s", log_label, code, url[:100])
            else:
                resp.raise_for_status()
                enc = resp.encoding or "utf-8"
                if resp.apparent_encoding:
                    try:
                        resp.encoding = resp.apparent_encoding
                    except Exception:
                        resp.encoding = enc
                body = extract_news_content_body(resp.text)
                if body:
                    return FetchResult(status="ok", body=body)
                log.warning("%s empty #ContentBody delete+skip %s", log_label, url[:100])
                return FetchResult(status="empty")
        except requests.HTTPError as exc:
            code = exc.response.status_code if exc.response is not None else None
            if code == 404:
                log.warning("%s HTTP 404 delete+skip %s", log_label, url[:100])
                return FetchResult(status="not_found", http_status=404)
            log.warning(
                "%s attempt %d/%d failed %s: %s",
                log_label,
                attempt,
                max_attempts,
                url[:80],
                exc,
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
    return FetchResult(status="failed")


def fetch_report_with_retries(
    session: requests.Session,
    url: str,
    *,
    timeout: float,
    retry_base: float,
    max_attempts: int,
    rate_limit_extra_delay: float,
) -> FetchResult:
    saw_empty = False
    for attempt in range(1, max_attempts + 1):
        try:
            resp = session.get(url, timeout=timeout)
            code = resp.status_code
            if code == 404:
                log.warning("HTTP 404 delete+skip %s", url[:100])
                return FetchResult(status="not_found", http_status=404)
            if code in HTTP_THROTTLE_CODES:
                log.warning("HTTP %d throttle %s", code, url[:100])
                if attempt < max_attempts:
                    wait = retry_base * (2 ** (attempt - 1)) + rate_limit_extra_delay
                    log.info("throttle wait %.1fs then retry", wait)
                    time.sleep(wait)
                continue
            if 400 <= code < 500:
                log.warning("HTTP %d skip %s", code, url[:100])
                return FetchResult(status="empty", http_status=code)
            resp.raise_for_status()
            resp.encoding = "gb2312"
            text = extract_report_content(resp.text)
            if (text or "").strip():
                return FetchResult(status="ok", body=text)
            saw_empty = True
            log.warning("empty parse delete+skip %s", url[:80])
            return FetchResult(status="empty")
        except requests.HTTPError as exc:
            code = exc.response.status_code if exc.response is not None else None
            if code == 404:
                log.warning("HTTP 404 delete+skip %s", url[:100])
                return FetchResult(status="not_found", http_status=404)
            if code is not None and 400 <= code < 500:
                return FetchResult(status="empty", http_status=code)
            log.warning("[%d/%d] HTTP error %s — %s", attempt, max_attempts, url[:80], exc)
        except Exception as exc:
            log.warning("[%d/%d] %s — %s", attempt, max_attempts, url[:80], exc)
        if attempt < max_attempts:
            wait = retry_base * (2 ** (attempt - 1))
            log.info("retry in %.1fs", wait)
            time.sleep(wait)
    if saw_empty:
        return FetchResult(status="empty")
    return FetchResult(status="failed")


def load_news_rows(csv_path: Path, symbols: set[str], since: date) -> list[tuple[str, str, str, date]]:
    if not csv_path.is_file():
        raise SystemExit(f"CSV not found: {csv_path}")
    required = {"url", "symbol", "title", "date"}
    rows: list[tuple[str, str, str, date]] = []
    with csv_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
            raise SystemExit(
                f"CSV {csv_path} must have columns {sorted(required)}; got {reader.fieldnames}"
            )
        for row in reader:
            url = (row.get("url") or "").strip()
            sym = (row.get("symbol") or "").strip()
            title = (row.get("title") or "").strip()
            row_date = parse_date_str(row.get("date") or "")
            if not url or not sym or row_date is None:
                continue
            if sym not in symbols or row_date < since:
                continue
            rows.append((url, sym, title, row_date))
    rows.sort(key=lambda r: (r[3], r[0]))

    deduped_rows: list[tuple[str, str, str, date]] = []
    seen_urls: set[str] = set()
    for row in rows:
        url = row[0]
        if url in seen_urls:
            continue
        seen_urls.add(url)
        deduped_rows.append(row)
    return deduped_rows


def load_report_rows(csv_path: Path, symbols: set[str], since: date | None) -> list[tuple[str, str, date | None]]:
    if not csv_path.is_file():
        raise SystemExit(f"CSV not found: {csv_path}")
    required = {"url", "symbol", "title", "date"}
    rows: list[tuple[str, str, date | None]] = []
    with csv_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
            raise SystemExit(
                f"CSV {csv_path} must have columns {sorted(required)}; got {reader.fieldnames}"
            )
        for row in reader:
            url = (row.get("url") or "").strip()
            sym = (row.get("symbol") or "").strip()
            if not url or not sym or sym not in symbols:
                continue
            row_date = parse_date_str(row.get("date") or "")
            if since is not None and (row_date is None or row_date < since):
                continue
            title = (row.get("title") or "").strip()
            rows.append((url, title, row_date))
    rows.sort(key=lambda r: (r[2] is None, r[2] or date.min, r[0]))

    deduped_rows: list[tuple[str, str, date | None]] = []
    seen_urls: set[str] = set()
    for row in rows:
        url = row[0]
        if url in seen_urls:
            continue
        seen_urls.add(url)
        deduped_rows.append(row)
    return deduped_rows


def build_news_file(url: str, symbol: str, title: str, news_date: date, body: str) -> str:
    title_one = title.replace("\r", " ").replace("\n", " ").strip()
    return (
        f"URL={url}\n"
        f"SYMBOL={symbol}\n"
        f"TITLE={title_one}\n"
        f"DATE={news_date.isoformat()}\n"
        f"---\n"
        f"{body.strip()}\n"
    )


def build_report_file(url: str, title: str, report_date: date | None, body: str) -> str:
    date_s = report_date.isoformat() if report_date is not None else ""
    return (
        f"URL={url}\n"
        f"TITLE={title}\n"
        f"DATE={date_s}\n"
        f"---\n"
        f"{body.strip()}\n"
    )


def process_news(args: argparse.Namespace, symbols: list[str]) -> None:
    since = date.fromisoformat(args.since) if args.since else (ALL_SINCE if args.all else DEFAULT_NEWS_SINCE)
    csv_path = args.news_csv.resolve()
    rows = load_news_rows(csv_path, set(symbols), since)
    CONTENT_NEWS_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Loaded %d news rows (symbols=%s, since=%s)", len(rows), symbols, since)

    if args.dry_run:
        for i, (url, _sym, _title, d) in enumerate(rows[:50], 1):
            log.info("[%d/%d] would fetch news %s %s", i, len(rows), d, url[:80])
        if len(rows) > 50:
            log.info("... and %d more", len(rows) - 50)
        return

    session = requests.Session()
    session.headers.update(NEWS_HEADERS)
    retry_base = args.retry_base_delay if args.retry_base_delay is not None else args.delay
    max_attempts = max(1, args.max_attempts)

    ok = skip = fail = deleted = 0
    for i, (url, sym, title, d) in enumerate(rows, 1):
        path = news_output_path(url, d)
        if not args.force and file_has_nonempty_body(path):
            skip += 1
            continue

        path.parent.mkdir(parents=True, exist_ok=True)
        label = f"[news {i}/{len(rows)}]"
        result = fetch_news_with_retries(
            session,
            url,
            timeout=args.timeout,
            retry_base=retry_base,
            max_attempts=max_attempts,
            log_label=label,
        )
        if result.status == "ok":
            path.write_text(build_news_file(url, sym, title, d, result.body), encoding="utf-8")
            ok += 1
            if args.delay > 0:
                time.sleep(args.delay)
            continue
        if result.status == "not_found":
            removed = delete_news_rows_by_url([url])
            remove_file_if_exists(path)
            deleted += removed
            continue
        if result.status == "empty":
            removed = delete_news_rows_by_url([url])
            remove_file_if_exists(path)
            deleted += removed
            continue
        fail += 1
        if args.delay > 0:
            time.sleep(args.delay)

    log.info("news done ok=%d skip=%d fail=%d deleted=%d total=%d", ok, skip, fail, deleted, len(rows))


def report_worker(
    url: str,
    *,
    timeout: float,
    retry_base: float,
    max_attempts: int,
    delay: float,
    rate_limit_extra_delay: float,
) -> tuple[str, FetchResult]:
    session = requests.Session()
    session.headers.update(REPORT_HEADERS)
    result = fetch_report_with_retries(
        session,
        url,
        timeout=timeout,
        retry_base=retry_base,
        max_attempts=max_attempts,
        rate_limit_extra_delay=rate_limit_extra_delay,
    )
    if result.status != "not_found" and delay > 0:
        time.sleep(delay)
    return url, result


def process_report(args: argparse.Namespace, symbols: list[str]) -> None:
    since = date.fromisoformat(args.since) if args.since else (ALL_SINCE if args.all else None)
    csv_path = args.report_csv.resolve()
    rows = load_report_rows(csv_path, set(symbols), since)
    CONTENT_REPORT_DIR.mkdir(parents=True, exist_ok=True)

    to_fetch: list[tuple[str, str, date | None]] = []
    skipped = 0
    for url, title, d in rows:
        path = report_output_path(url)
        if not args.force and file_has_nonempty_body(path):
            skipped += 1
            continue
        to_fetch.append((url, title, d))

    log.info("Loaded %d report rows (symbols=%s, since=%s)", len(rows), symbols, since or "—")
    log.info("Skip (cached): %d  To fetch: %d", skipped, len(to_fetch))

    if args.dry_run:
        for i, (url, title, d) in enumerate(to_fetch[:50], 1):
            log.info("[%d/%d] would fetch report %s | %s | %s", i, len(to_fetch), d, title[:40], url[:80])
        if len(to_fetch) > 50:
            log.info("... and %d more", len(to_fetch) - 50)
        return

    retry_base = args.retry_base_delay if args.retry_base_delay is not None else args.delay
    max_attempts = max(1, args.max_attempts)
    done = failed = written = deleted = empty = 0

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(
                report_worker,
                url,
                timeout=args.timeout,
                retry_base=retry_base,
                max_attempts=max_attempts,
                delay=args.delay,
                rate_limit_extra_delay=args.rate_limit_extra_delay,
            ): (url, title, d)
            for url, title, d in to_fetch
        }
        for future in as_completed(futures):
            url, title, d = futures[future]
            done += 1
            try:
                _, result = future.result()
            except Exception as exc:
                log.warning("[report %d/%d] exception %s: %s", done, len(to_fetch), url[:60], exc)
                failed += 1
                continue

            if result.status == "ok":
                path = report_output_path(url)
                path.write_text(build_report_file(url, title, d, result.body), encoding="utf-8")
                written += 1
                log.info("[report %d/%d] OK chars=%d %s", done, len(to_fetch), len(result.body), url[:70])
                continue
            if result.status == "not_found":
                removed = delete_report_rows_by_url([url])
                remove_file_if_exists(report_output_path(url))
                deleted += removed
                continue
            if result.status == "empty":
                removed = delete_report_rows_by_url([url])
                remove_file_if_exists(report_output_path(url))
                deleted += removed
                continue
            failed += 1
            log.warning("[report %d/%d] transient fail %s", done, len(to_fetch), url[:80])

    log.info(
        "report done written=%d empty=%d failed=%d deleted=%d processed=%d",
        written,
        empty,
        failed,
        deleted,
        done,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch news/report content to disk.")
    ap.add_argument("--mode", choices=["news", "report", "all"], default="all")
    ap.add_argument("--sector", default=DEFAULT_SECTOR, help="Sector in config/stocks.json")
    ap.add_argument("--symbols", default="", help="Comma-separated symbols; overrides --sector if set.")
    ap.add_argument(
        "--all",
        action="store_true",
        help="All symbols in config/stocks.json; since defaults to 2025-01-01 unless --since is set.",
    )
    ap.add_argument(
        "--since",
        default=None,
        help="Minimum date (YYYY-MM-DD). For news default is 2025-11-01 unless --all; for report default is no filter unless --all.",
    )
    ap.add_argument("--news-csv", type=Path, default=DEFAULT_NEWS_CSV)
    ap.add_argument("--report-csv", type=Path, default=DEFAULT_REPORT_CSV)
    ap.add_argument("--workers", type=int, default=1, help="Concurrent report fetches.")
    ap.add_argument("--delay", type=float, default=2.0, help="Delay after each finished URL.")
    ap.add_argument(
        "--retry-base-delay",
        type=float,
        default=None,
        help="First backoff seconds after a failed attempt; doubles each retry. Default: same as --delay.",
    )
    ap.add_argument("--max-attempts", type=int, default=5)
    ap.add_argument("--timeout", type=float, default=20.0)
    ap.add_argument("--rate-limit-extra-delay", type=float, default=20.0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.all:
        symbols = all_symbols()
    elif args.symbols.strip():
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = symbols_for_sector(args.sector)

    if args.mode in {"news", "all"}:
        process_news(args, symbols)
    if args.mode in {"report", "all"}:
        process_report(args, symbols)


if __name__ == "__main__":
    main()
