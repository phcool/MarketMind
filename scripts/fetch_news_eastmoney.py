"""
Fetch news for each stock listed in config/stocks.json from a chosen platform.

Supported platforms (--platform):
  eastmoney  Search URL: https://so.eastmoney.com/web/s?keyword=<symbol>
             Pagination is client-side: click "下一页" until it disappears.
  sina      Static HTML from stock.finance.sina.com.cn/stock/go.php/vReport_List.
             Parses table.tb_01 tr rows (GBK encoding) for title, url, date; paginate until empty.

Storage: UTF-8 CSV under exports/ — eastmoney -> news.csv, sina -> report.csv (dedupe by url).
Only A-share 6-digit symbols are appended.

Each platform has its own checkpoint file (fetch_news_<platform>_checkpoint.json).
Use --add to ignore checkpoint and re-fetch all companies.
"""

import argparse
import json
import re
import time
import logging
from datetime import datetime, date
from pathlib import Path
from typing import Callable

import requests
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout

from csv_io import merge_append_news, merge_append_report
from stock_universe import load_sectors

# ── configuration ────────────────────────────────────────────────────────────

ROOT_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = ROOT_DIR / "checkpoint"
PLATFORMS = ["eastmoney", "sina"]


def checkpoint_path(platform: str) -> Path:
    """Per-platform checkpoint file under project_root/checkpoint/."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    return CHECKPOINT_DIR / f"fetch_news_{platform}_checkpoint.json"

# eastmoney-specific
SEARCH_URL_EASTMONEY = "https://so.eastmoney.com/news/s?keyword={keyword}&sort=time"
PAGE_WAIT_MS = 3000       # wait for news list to load
NEXT_PAGE_WAIT_MS = 2000  # wait after clicking next page
MAX_PAGES: int | None = 50  # Eastmoney news search has up to 50 pages
MAX_ZERO_SAVE_STREAK = 5  # stop after this many consecutive pages with 0 new items

# URL pattern to extract publish date: e.g. /a/202603133671606557.html -> 20260313
DATE_IN_URL_RE = re.compile(r"/a/(\d{8})\d*\.html", re.I)

# sina-specific
SEARCH_URL_SINA = (
    "https://stock.finance.sina.com.cn/stock/go.php/vReport_List"
    "/kind/search/index.phtml?symbol={symbol}&t1=all&p={page}"
)
SINA_BASE = "https://stock.finance.sina.com.cn"
SINA_REQUEST_DELAY = 1.0   # seconds between page requests
SINA_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Referer": "https://stock.finance.sina.com.cn/",
    "Accept-Language": "zh-CN,zh;q=0.9",
}

SECTORS = load_sectors()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

def date_from_url(url: str) -> date | None:
    """Extract YYYYMMDD from East Money article URL; return None if not found."""
    if not url:
        return None
    m = DATE_IN_URL_RE.search(url)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y%m%d").date()
    except ValueError:
        return None

def folder_name(name: str, symbol: str, market: str) -> str:
    suffix = f"{symbol}.HK" if market == "hk" else symbol
    return f"{name}({suffix})"

def _company_key(sector: str, name: str, symbol: str, market: str) -> str:
    return f"{sector}/{folder_name(name, symbol, market)}"


def db_symbol_for_insert(symbol: str, market: str) -> str | None:
    """Same rule as import_* scripts: only mainland A-share 6-digit code."""
    if market != "a":
        return None
    if len(symbol) == 6 and symbol.isdigit():
        return symbol
    return None


def insert_news_rows(_cur, sym: str, entries: list[dict]) -> int:
    return merge_append_news(sym, entries)


def insert_report_rows(_cur, sym: str, entries: list[dict]) -> int:
    return merge_append_report(sym, entries)


# ── checkpoint (per-platform: set of completed company_key) ────────────────────

def load_checkpoint(platform: str, use_checkpoint: bool) -> set[str]:
    """Load set of company_key that are already done for this platform."""
    if not use_checkpoint:
        return set()
    path = checkpoint_path(platform)
    if not path.exists():
        return set()
    try:
        raw = path.read_text(encoding="utf-8")
        if not raw.strip():
            return set()
        data = json.loads(raw)
        if isinstance(data, dict) and "completed" in data:
            return set(data["completed"])
        return set()
    except (OSError, json.JSONDecodeError) as e:
        log.warning("Checkpoint load failed for %s: %s", platform, e)
        return set()

def save_checkpoint(platform: str, completed: set[str]) -> None:
    path = checkpoint_path(platform)
    try:
        path.write_text(
            json.dumps({"completed": sorted(completed)}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError as e:
        log.warning("Checkpoint write failed for %s: %s", platform, e)

def reset_checkpoint(platform: str) -> None:
    path = checkpoint_path(platform)
    if path.exists():
        try:
            path.unlink()
        except OSError:
            pass

# ── eastmoney fetch (platform-specific) ───────────────────────────────────────

def scrape_eastmoney(page, keyword: str) -> list[dict]:
    """
    East Money: open search page for keyword, collect all news items from all pages.
    Returns list of dicts: { "url", "title", "snippet", "date", "platform" }.
    """
    url = SEARCH_URL_EASTMONEY.format(keyword=keyword)
    page.goto(url, wait_until="domcontentloaded", timeout=30000)
    page.wait_for_timeout(PAGE_WAIT_MS)

    collected: list[dict] = []
    seen_urls: set[str] = set()
    page_num = 0
    zero_save_streak = 0

    while True:
        page_num += 1
        if MAX_PAGES is not None and page_num > MAX_PAGES:
            log.info("  Reached MAX_PAGES=%d — stop.", MAX_PAGES)
            break
        # wait for news list to be present
        try:
            page.wait_for_selector(".news_list", timeout=10000)
        except PlaywrightTimeout:
            log.warning("  No .news_list found on current page")
            break

        items = page.query_selector_all(".news_item")
        if not items:
            log.info("  No .news_item elements on this page")
            # might be last page with no results
            break

        count_before = len(collected)
        for el in items:
            try:
                link_el = el.query_selector(".news_item_t a")
                if not link_el:
                    continue
                href = link_el.get_attribute("href")
                title = (link_el.inner_text() or "").strip()
                if not href or href == "#":
                    continue

                # normalize URL (eastmoney base)
                if href.startswith("//"):
                    href = "https:" + href
                elif href.startswith("/"):
                    href = "https://so.eastmoney.com" + href

                if href in seen_urls:
                    continue
                seen_urls.add(href)

                snippet_el = el.query_selector(".news_item_c")
                snippet = (snippet_el.inner_text() or "").strip() if snippet_el else ""

                pub_date = date_from_url(href)

                collected.append({
                    "url": href,
                    "title": title,
                    "snippet": snippet,
                    "date": pub_date.isoformat() if pub_date else None,
                    "platform": "eastmoney",
                })
            except Exception as e:
                log.debug("  Skip item: %s", e)
                continue

        new_this_page = len(collected) - count_before
        if new_this_page == 0:
            zero_save_streak += 1
        else:
            zero_save_streak = 0

        log.info(
            "  Page %d: %d new items (total %d, zero-save-streak %d)",
            page_num, new_this_page, len(collected), zero_save_streak
        )
        if zero_save_streak >= MAX_ZERO_SAVE_STREAK:
            log.info("  Reached zero-save streak %d — stop.", MAX_ZERO_SAVE_STREAK)
            break

        # find "下一页" link
        next_btn = page.query_selector('div.c_pager a[title="下一页"], div.pagingnew a[title="下一页"]')
        if not next_btn:
            log.info("  No next-page button — stop.")
            break

        # check if it's disabled (last page): often the last page has no next link or it's disabled
        try:
            next_btn.click()
        except Exception as e:
            log.warning("  Next button click failed: %s", e)
            break

        page.wait_for_timeout(NEXT_PAGE_WAIT_MS)

    return collected

def fetch_eastmoney(page, cur, name: str, symbol: str, market: str, sector: str) -> None:
    """Fetch news from East Money and append to exports/news.csv."""
    log.info("━━ %s (%s) ━━", name, symbol)

    entries = scrape_eastmoney(page, name)
    if not entries:
        log.warning("  No news collected")
        return

    with_date = [e for e in entries if e.get("date")]
    without_date = len(entries) - len(with_date)
    if without_date:
        log.warning("  %d entries without date (skipped for CSV)", without_date)

    sym = db_symbol_for_insert(symbol, market)
    if not sym:
        log.warning("  Skip CSV insert: not A-share 6-digit (%s, market=%s)", symbol, market)
        return
    n = insert_news_rows(cur, sym, with_date)
    log.info("  Collected %d (%d with date), CSV new rows: %d", len(entries), len(with_date), n)

# ── sina ─────────────────────────────────────────────────────────────────────

def _sina_symbol(symbol: str, market: str) -> str:
    """Return the Sina stock symbol string, e.g. 'sz300750' or 'sh600036'."""
    if market != "a":
        return symbol
    prefix = "sh" if symbol and symbol[0] in ("6", "9") else "sz"
    return f"{prefix}{symbol}"

SINA_OLD_YEAR_THRESHOLD = 2022   # stop when this many consecutive entries are <= this year
SINA_OLD_YEAR_STREAK_MAX = 5

def scrape_sina(symbol: str, market: str) -> list[dict]:
    """Fetch all research-report pages from Sina Finance for the given stock."""
    sina_sym = _sina_symbol(symbol, market)
    session = requests.Session()
    session.headers.update(SINA_HEADERS)
    collected: list[dict] = []
    seen_urls: set[str] = set()
    old_year_streak = 0   # consecutive entries with year <= SINA_OLD_YEAR_THRESHOLD
    zero_save_streak = 0  # consecutive pages with 0 new items

    page_num = 1
    stop = False
    while not stop:
        if MAX_PAGES is not None and page_num > MAX_PAGES:
            log.info("  sina: reached MAX_PAGES=%d — stop.", MAX_PAGES)
            break

        url = SEARCH_URL_SINA.format(symbol=sina_sym, page=page_num)
        try:
            resp = session.get(url, timeout=20)
            resp.raise_for_status()
            resp.encoding = "gb2312"
        except Exception as exc:
            log.warning("  sina page %d fetch failed: %s", page_num, exc)
            break

        soup = BeautifulSoup(resp.text, "lxml")
        data_rows = [
            r for r in soup.select("table.tb_01 tr")
            if r.select_one("td.tal a")
        ]

        if not data_rows:
            log.info("  sina page %d: no data rows — stop.", page_num)
            break

        count_before = len(collected)
        for row in data_rows:
            link_el = row.select_one("td.tal a")
            if not link_el:
                continue
            href = link_el.get("href", "").strip()
            title = (link_el.get("title") or link_el.get_text(strip=True)).strip()
            if not href or not title:
                continue
            if href.startswith("//"):
                href = "https:" + href
            elif href.startswith("/"):
                href = SINA_BASE + href
            if href in seen_urls:
                continue
            seen_urls.add(href)

            tds = row.find_all("td")
            date_str = tds[3].get_text(strip=True) if len(tds) > 3 else ""
            pub_date: date | None = None
            try:
                pub_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            except ValueError:
                pass

            collected.append({
                "url": href,
                "title": title,
                "snippet": "",
                "date": pub_date.isoformat() if pub_date else None,
                "platform": "sina",
            })

            if pub_date and pub_date.year <= SINA_OLD_YEAR_THRESHOLD:
                old_year_streak += 1
                if old_year_streak >= SINA_OLD_YEAR_STREAK_MAX:
                    log.info(
                        "  sina: %d consecutive entries with year <= %d — stop.",
                        SINA_OLD_YEAR_STREAK_MAX, SINA_OLD_YEAR_THRESHOLD,
                    )
                    stop = True
                    break
            else:
                old_year_streak = 0

        new_n = len(collected) - count_before
        if new_n == 0:
            zero_save_streak += 1
        else:
            zero_save_streak = 0
        log.info(
            "  sina page %d: %d new items (total %d, old-year streak %d, zero-save-streak %d)",
            page_num, new_n, len(collected), old_year_streak, zero_save_streak,
        )
        if zero_save_streak >= MAX_ZERO_SAVE_STREAK:
            log.info("  sina: reached zero-save streak %d — stop.", MAX_ZERO_SAVE_STREAK)
            break

        page_num += 1
        time.sleep(SINA_REQUEST_DELAY)

    return collected

def fetch_sina(page, cur, name: str, symbol: str, market: str, sector: str) -> None:  # noqa: ARG001
    """Fetch Sina research list and append to exports/report.csv."""
    log.info("━━ %s (%s) ━━", name, symbol)

    entries = scrape_sina(symbol, market)
    if not entries:
        log.warning("  No entries collected")
        return

    with_date = [e for e in entries if e.get("date")]
    without_date = len(entries) - len(with_date)
    if without_date:
        log.warning("  %d entries without date (skipped for CSV)", without_date)

    sym = db_symbol_for_insert(symbol, market)
    if not sym:
        log.warning("  Skip CSV insert: not A-share 6-digit (%s, market=%s)", symbol, market)
        return
    n = insert_report_rows(cur, sym, with_date)
    log.info("  Collected %d (%d with date), CSV new rows: %d", len(entries), len(with_date), n)

# Platform registry: fetch(page, cur, name, symbol, market, sector); cur unused (CSV).
FETCHERS: dict[str, Callable[..., None]] = {
    "eastmoney": fetch_eastmoney,
    "sina": fetch_sina,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch news for each stock in config/stocks.json from a chosen platform.",
    )
    parser.add_argument(
        "--platform",
        choices=PLATFORMS,
        default="eastmoney",
        help="News platform to fetch from (default: eastmoney).",
    )
    parser.add_argument(
        "--add",
        action="store_true",
        help="Ignore checkpoint and re-fetch all companies; CSV skips duplicate urls.",
    )
    args = parser.parse_args()
    platform = args.platform
    use_checkpoint = not args.add

    if args.add:
        reset_checkpoint(platform)
        log.info("--add: checkpoint reset for %s, will fetch all companies", platform)

    fetcher = FETCHERS[platform]
    total = sum(len(v) for v in SECTORS.values())
    done = 0
    completed_set = load_checkpoint(platform, use_checkpoint)
    if use_checkpoint and completed_set:
        log.info("Checkpoint loaded for %s: %d companies already done", platform, len(completed_set))

    cur = None
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 720},
        )
        page = context.new_page()

        try:
            for sector, companies in SECTORS.items():
                for name, symbol, market in companies:
                    done += 1
                    company_key = _company_key(sector, name, symbol, market)
                    if company_key in completed_set:
                        log.info(
                            "── [%d/%d] %s / %s — skip (already done)",
                            done, total, sector, name,
                        )
                        continue
                    log.info("── [%d/%d] %s / %s", done, total, sector, name)
                    try:
                        fetcher(page, cur, name, symbol, market, sector)
                        completed_set.add(company_key)
                        save_checkpoint(platform, completed_set)
                    except Exception as exc:
                        log.exception("  Error: %s", exc)
                    time.sleep(1)
        finally:
            browser.close()

    log.info("All companies done.")

if __name__ == "__main__":
    main()
