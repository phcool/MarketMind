"""
Unified entry: fetch comments (forum), news (eastmoney), and/or report (sina) into exports/*.csv.

Comments: merged rows are trimmed to the top COMMENTS_MAX_PER_SYMBOL_DAY (200) per (symbol, calendar day)
by click_count (see csv_io.merge_append_comments and rewrite_comments_csv_trimmed).

Examples:
  python scripts/fetch_market_data.py --mode all
  python scripts/fetch_market_data.py --mode comments --add
  python scripts/fetch_market_data.py --mode news
  python scripts/fetch_market_data.py --mode report --offset 3
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait
from datetime import date, datetime
from pathlib import Path
from typing import Callable

import requests
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout

from csv_io import (
    COMMENTS_MAX_PER_SYMBOL_DAY,
    merge_append_comments,
    merge_append_news,
    merge_append_report,
    rewrite_comments_csv_trimmed,
)
from stock_universe import load_sectors

ROOT_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = ROOT_DIR / "checkpoint"

log = logging.getLogger(__name__)


# === Forum (East Money 股吧) ================================================

# ── configuration (forum) ───────────────────────────────────────────────────

CHECKPOINT_FILE = CHECKPOINT_DIR / "fetch_forum_checkpoint.json"

API_URL       = "https://gbcdn.dfcfw.com/gbapi/webarticlelist_api_Article_Articlelist.js"
LIST_PAGE_URL = "https://guba.eastmoney.com/list,{symbol}.html"  # for parsing total pages
PAGE_SIZE     = 80
WORKERS_PER_COMPANY = 4    # concurrent threads per company
REQUEST_DELAY      = 1.5   # seconds each thread sleeps between page fetches
TIMEOUT            = 20
MAX_RETRIES        = 3
MAX_POSTS_PER_COMPANY = 10_000  # stop fetching this stock once we have inserted this many new rows
MAX_ZERO_SAVE_STREAK = 20       # stop stock if 20 consecutive pages have data but CSV adds 0 new

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Referer": "https://guba.eastmoney.com/",
}

KEEP_FIELDS = [
    "post_id",
    "post_title",
    "user_id",
    "user_nickname",
    "post_click_count",
    "post_comment_count",
    "post_forward_count",
    "post_publish_time",
    "post_last_time",
    "post_type",
    "post_has_pic",
    "post_has_video",
    "art_unique_url",
]

# ── checkpoint (per-company: total_pages + completed) ─────────────────────────
# Value: int N (all 1..N done), or dict {"total_pages": N, "completed": list | int}.

_forum_checkpoint_lock = threading.Lock()
_forum_checkpoint_data: dict[str, int | dict] = {}


def _company_key(sector: str, name: str, symbol: str, market: str) -> str:
    suffix = f"{symbol}.HK" if market == "hk" else symbol
    return f"{sector}/{name}({suffix})"


def forum_load_company_checkpoint(company_key: str, use_checkpoint: bool) -> tuple[set[int], int | None]:
    """
    Load (completed page set, total_pages or None) for this company.
    If total_pages is None, caller must call get_total_pages().
    """
    if not use_checkpoint:
        return set(), None
    with _forum_checkpoint_lock:
        data = _forum_checkpoint_data.get(company_key)
    if data is None:
        return set(), None
    if isinstance(data, int):
        return set(range(1, data + 1)), data
    if isinstance(data, dict):
        total = data.get("total_pages")
        comp = data.get("completed")
        capped = data.get("capped_at")
        if capped is not None and total is not None:
            # Capped at N posts: treat as all pages done so we skip this company
            return set(range(1, total + 1)), total
        if isinstance(comp, int):
            return set(range(1, comp + 1)), total or comp
        return set(comp) if comp else set(), total
    if isinstance(data, list):
        # legacy: in-progress list without total_pages
        return set(data), None
    return set(), None


def _save_checkpoint(company_key: str, completed: set[int], total_pages: int | None = None) -> None:
    with _forum_checkpoint_lock:
        existing = _forum_checkpoint_data.get(company_key)
        if isinstance(existing, dict) and total_pages is None:
            total_pages = existing.get("total_pages")
        _forum_checkpoint_data[company_key] = {
            "total_pages": total_pages or 1,
            "completed": sorted(completed),
        }
        _write_checkpoint_file()


def _write_checkpoint_file() -> None:
    path = CHECKPOINT_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(
            json.dumps(_forum_checkpoint_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError as e:
        log.warning("Checkpoint write failed: %s", e)


def finalize_company_checkpoint(company_key: str, dispatcher: "PageDispatcher") -> None:
    """When all pages are done, store as int N (total pages) in checkpoint."""
    with dispatcher._lock:
        completed = set(dispatcher._completed)
        max_page  = dispatcher._max_page
    if max_page is not None and completed >= set(range(1, max_page + 1)):
        with _forum_checkpoint_lock:
            _forum_checkpoint_data[company_key] = max_page
            _write_checkpoint_file()
        log.debug("Checkpoint compacted for %s → %d pages", company_key, max_page)


def load_all_checkpoints(use_checkpoint: bool) -> None:
    """Load checkpoint file into _forum_checkpoint_data. Call once at startup."""
    global _forum_checkpoint_data
    if not use_checkpoint:
        _forum_checkpoint_data = {}
        return
    path = CHECKPOINT_FILE
    if not path.exists():
        _forum_checkpoint_data = {}
        return
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
        _forum_checkpoint_data = data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError) as e:
        log.warning("Checkpoint load failed: %s", e)
        _forum_checkpoint_data = {}


def forum_reset_checkpoint_for_run() -> None:
    """Clear in-memory checkpoint so all companies start from page 1 (--add)."""
    global _forum_checkpoint_data
    with _forum_checkpoint_lock:
        _forum_checkpoint_data = {}
    path = CHECKPOINT_FILE
    if path.exists():
        try:
            path.unlink()
        except OSError:
            pass


# ── thread-safe page dispatcher ───────────────────────────────────────────────

class PageDispatcher:
    """
    Hands out the smallest un-started page number to competing worker threads.
    Stops when total_pages is exhausted or when saved_count_hint >= max_posts_cap.
    """

    def __init__(self, initial_completed: set[int], total_pages: int,
                 on_page_done: None = None,
                 saved_count: list[int] | None = None,
                 max_posts_cap: int | None = None) -> None:
        self._lock            = threading.Lock()
        self._completed       = set(initial_completed)
        self._assigned        : set[int] = set()
        self._max_page        = total_pages
        self._on_page_done    = on_page_done
        self._saved_count     = saved_count  # [int] ref; read after each page
        self._max_posts_cap   = max_posts_cap
        self._cap_reached     = False
        self._stopped         = False
        self._stop_reason     = ""

    def get_next_page(self) -> int | None:
        """Return the smallest page not in completed and not in assigned, or None."""
        with self._lock:
            if self._stopped:
                return None
            if self._cap_reached:
                return None
            if self._max_posts_cap is not None and self._saved_count and self._saved_count[0] >= self._max_posts_cap:
                self._cap_reached = True
                return None
            k = 1
            while k in self._completed or k in self._assigned:
                k += 1
            if self._max_page is not None and k > self._max_page:
                return None
            self._assigned.add(k)
            return k

    def stop(self, reason: str) -> None:
        """Stop assigning new pages for this company."""
        with self._lock:
            self._stopped = True
            self._stop_reason = reason

    def mark_completed(self, page: int, company_key: str) -> None:
        """Mark page as done and persist checkpoint."""
        with self._lock:
            self._completed.add(page)
            self._assigned.discard(page)
        if self._on_page_done:
            self._on_page_done(company_key, self._completed)


# ── get total pages (from list page HTML or API fallback) ─────────────────────

def get_total_pages(symbol: str) -> int:
    """
    Get total page count for this stock bar. Tries list page HTML first (last page
    link: a.nump or a[href*="f_"], e.g. f_6011.html → 6011), then falls back to API.
    """
    url = LIST_PAGE_URL.format(symbol=symbol)
    try:
        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        resp.raise_for_status()
    except requests.RequestException:
        pass
    else:
        soup = BeautifulSoup(resp.text, "lxml")
        # Last page link: <a class="nump" href="...f_6011.html">...</a> or <span>6011</span>
        page_nums: list[int] = []
        for a in soup.select('a.nump, a[href*="f_"]'):
            href = a.get("href") or ""
            m = re.search(r"f_(\d+)\.html", href)
            if m:
                page_nums.append(int(m.group(1)))
            span = a.select_one("span")
            if span:
                try:
                    page_nums.append(int(span.get_text(strip=True)))
                except ValueError:
                    pass
        if page_nums:
            return max(page_nums)
    # Fallback: API page 1 -> count
    session = requests.Session()
    session.headers.update(HEADERS)
    data = fetch_api_page(session, symbol, 1)
    if data and isinstance(data.get("count"), (int, float)):
        return max(1, math.ceil(int(data["count"]) / PAGE_SIZE))
    return 1


# ── API helpers ───────────────────────────────────────────────────────────────

def fetch_api_page(session: requests.Session, symbol: str, page: int) -> dict | None:
    params = {"code": symbol, "p": page, "ps": PAGE_SIZE}
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.get(API_URL, params=params, timeout=TIMEOUT)
            resp.raise_for_status()
            m = re.search(r"var \w+=(\{.*\})", resp.text, re.DOTALL)
            if not m:
                return None
            return json.loads(m.group(1))
        except requests.RequestException as exc:
            log.warning("page %d attempt %d/%d: %s", page, attempt, MAX_RETRIES, exc)
            if attempt < MAX_RETRIES:
                time.sleep(attempt * 3)
    return None


def parse_posts(raw_list: list[dict]) -> tuple[list[dict], bool]:
    """
    Parse raw API posts into kept fields. No date range filter — all posts are kept.
    Returns (posts, has_more); has_more is unused (pagination uses total_pages).
    """
    posts: list[dict] = []

    for raw in raw_list:
        pub_str = raw.get("post_publish_time", "") or ""
        try:
            datetime.strptime(pub_str, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue

        unique = raw.get("art_unique_url") or ""
        if not unique:
            pid = raw.get("post_id", "")
            code = raw.get("stockbar_code", "")
            unique = f"https://guba.eastmoney.com/news,{code},{pid}.html"

        entry = {k: raw.get(k) for k in KEEP_FIELDS}
        entry["url"] = unique
        posts.append(entry)

    return posts, True


def posts_to_comment_rows(posts: list[dict], symbol: str) -> list[tuple]:
    rows: list[tuple] = []
    for p in posts:
        url = (p.get("url") or "").strip()
        if not url:
            continue
        pid = p.get("post_id")
        post_id_s = str(pid).strip() if pid is not None and str(pid).strip() else None
        title = (p.get("post_title") or "").strip()
        ts = p.get("post_publish_time")
        try:
            pub = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
        except (ValueError, TypeError):
            continue
        clicks = p.get("post_click_count")
        ccount = p.get("post_comment_count")
        rows.append((url, post_id_s, symbol, title, pub, clicks, ccount))
    return rows


def insert_comments(_cur, symbol: str, posts: list[dict]) -> int:
    """Append posts; duplicates skipped by url. Returns count of newly inserted rows."""
    rows = posts_to_comment_rows(posts, symbol)
    return merge_append_comments(rows)


# ── worker function (runs in a thread) ───────────────────────────────────────

def _worker(
    symbol: str,
    company_key: str,
    dispatcher: PageDispatcher,
    saved_counter: list[int],
    zero_save_streak: list[int],
    counter_lock: threading.Lock,
    total_pages: int,
) -> None:
    session = requests.Session()
    session.headers.update(HEADERS)

    while True:
        page = dispatcher.get_next_page()
        if page is None:
            break

        data = fetch_api_page(session, symbol, page)
        if data is None:
            log.warning("page %d: fetch failed — skipping", page)
            time.sleep(REQUEST_DELAY)
            continue

        raw_posts = data.get("re") or []
        posts, _ = parse_posts(raw_posts)

        n = 0
        if posts:
            n = insert_comments(None, symbol, posts)
        with counter_lock:
            saved_counter[0] += n
            total_saved = saved_counter[0]
            if len(posts) > 0 and n == 0:
                zero_save_streak[0] += 1
            else:
                zero_save_streak[0] = 0
            streak_now = zero_save_streak[0]

        dispatcher.mark_completed(page, company_key)
        log.info(
            "page %4d / %d | in-range %3d | new_rows %3d (total %d) | zero-insert-streak %d",
            page, total_pages, len(posts), n, total_saved, streak_now,
        )
        if total_saved >= MAX_POSTS_PER_COMPANY:
            log.info("  reached %d posts — stopping this stock", MAX_POSTS_PER_COMPANY)
        if streak_now >= MAX_ZERO_SAVE_STREAK:
            dispatcher.stop(
                f"{MAX_ZERO_SAVE_STREAK} consecutive pages had data but saved 0"
            )
            log.info(
                "  reached zero-save streak %d — skipping this stock",
                MAX_ZERO_SAVE_STREAK,
            )
        time.sleep(REQUEST_DELAY)

    session.close()


# ── per-company orchestrator ──────────────────────────────────────────────────

def fetch_company(name: str, symbol: str, market: str, sector: str,
                  use_checkpoint: bool) -> None:
    company_key    = _company_key(sector, name, symbol, market)
    initial_done, checkpoint_total = forum_load_company_checkpoint(company_key, use_checkpoint)
    total_pages    = checkpoint_total if checkpoint_total is not None else get_total_pages(symbol)
    log.info("━━ %s (%s) — total %d pages ━━", name, symbol, total_pages)
    if initial_done:
        log.info("  resuming from checkpoint (%d pages already done)", len(initial_done))

    # Persist total_pages into checkpoint so resume and logs have it
    with _forum_checkpoint_lock:
        _forum_checkpoint_data[company_key] = {"total_pages": total_pages, "completed": sorted(initial_done)}
        _write_checkpoint_file()

    saved_counter: list[int] = [0]
    zero_save_streak: list[int] = [0]
    dispatcher = PageDispatcher(
        initial_completed=initial_done,
        total_pages=total_pages,
        on_page_done=lambda ck, completed: _save_checkpoint(ck, completed, total_pages),
        saved_count=saved_counter,
        max_posts_cap=MAX_POSTS_PER_COMPANY,
    )
    counter_lock = threading.Lock()

    with ThreadPoolExecutor(
        max_workers=WORKERS_PER_COMPANY,
        thread_name_prefix=f"{symbol}",
    ) as executor:
        futures = [
            executor.submit(
                _worker,
                symbol,
                company_key,
                dispatcher,
                saved_counter,
                zero_save_streak,
                counter_lock,
                total_pages,
            )
            for _ in range(WORKERS_PER_COMPANY)
        ]
        wait(futures)
        for f in futures:
            if f.exception():
                log.error("Worker raised: %s", f.exception())

    total_saved = saved_counter[0]
    with dispatcher._lock:
        stop_reason = dispatcher._stop_reason
    if total_saved >= MAX_POSTS_PER_COMPANY:
        # Mark as fully done so resume skips; record capped_at in checkpoint
        with dispatcher._lock:
            all_pages = set(range(1, (dispatcher._max_page or 0) + 1))
            dispatcher._completed = all_pages
        with _forum_checkpoint_lock:
            _forum_checkpoint_data[company_key] = {
                "total_pages": total_pages,
                "completed": sorted(all_pages),
                "capped_at": MAX_POSTS_PER_COMPANY,
            }
            _write_checkpoint_file()
        log.info("━━ %s done — %d new rows inserted (capped at %d) ━━\n", name, total_saved, MAX_POSTS_PER_COMPANY)
    elif stop_reason:
        # Treat as finished: persist compact int form so future runs skip this stock.
        with _forum_checkpoint_lock:
            _forum_checkpoint_data[company_key] = total_pages
            _write_checkpoint_file()
        log.info("  stop reason: %s", stop_reason)
        log.info("━━ %s done — %d new rows (marked finished by stop rule) ━━\n", name, total_saved)
    else:
        finalize_company_checkpoint(company_key, dispatcher)
        log.info("━━ %s done — %d new rows inserted ━━\n", name, total_saved)




# === News / report (Eastmoney search + Sina list) ==========================

PLATFORMS = ["eastmoney", "sina"]


def news_checkpoint_path(platform: str) -> Path:
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

def news_load_platform_checkpoint(platform: str, use_checkpoint: bool) -> set[str]:
    """Load set of company_key that are already done for this platform."""
    if not use_checkpoint:
        return set()
    path = news_checkpoint_path(platform)
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

def news_save_platform_checkpoint(platform: str, completed: set[str]) -> None:
    path = news_checkpoint_path(platform)
    try:
        path.write_text(
            json.dumps({"completed": sorted(completed)}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError as e:
        log.warning("Checkpoint write failed for %s: %s", platform, e)

def news_reset_platform_checkpoint(platform: str) -> None:
    path = news_checkpoint_path(platform)
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




SECTORS = load_sectors()


def _apply_sectors() -> None:
    global SECTORS
    SECTORS = load_sectors()


def _total_stocks() -> int:
    return sum(len(v) for v in SECTORS.values())


def run_comments(args: argparse.Namespace) -> None:
    _apply_sectors()
    use_checkpoint = not args.add
    if args.add:
        forum_reset_checkpoint_for_run()
        log.info("--add: forum checkpoint reset")
    else:
        load_all_checkpoints(use_checkpoint=True)
        if _forum_checkpoint_data:
            log.info("Forum checkpoint: %d companies", len(_forum_checkpoint_data))

    total = _total_stocks()
    start_index = max(1, args.offset)
    if start_index > total:
        log.warning("offset %d > total stocks %d, nothing to do.", start_index, total)
        return
    if start_index > 1:
        log.info("offset=%d: skip first %d stocks", start_index, start_index - 1)

    done = 0
    for sector, companies in SECTORS.items():
        for name, symbol, market in companies:
            done += 1
            if done < start_index:
                log.info("── [%d/%d] %s / %s — skip (offset)", done, total, sector, name)
                continue
            log.info("── [%d/%d] %s / %s [comments]", done, total, sector, name)
            try:
                fetch_company(name, symbol, market, sector, use_checkpoint)
            except Exception as exc:
                log.exception("comments %s: %s", name, exc)

    before, after = rewrite_comments_csv_trimmed()
    if before != after:
        log.info(
            "comments.csv trim: %d -> %d rows (max %d per symbol-day by clicks)",
            before,
            after,
            COMMENTS_MAX_PER_SYMBOL_DAY,
        )


def run_news(args: argparse.Namespace) -> None:
    _apply_sectors()
    use_checkpoint = not args.add
    platform = "eastmoney"
    if args.add:
        news_reset_platform_checkpoint(platform)
        log.info("--add: checkpoint reset for %s", platform)
    completed_set = news_load_platform_checkpoint(platform, use_checkpoint)
    if use_checkpoint and completed_set:
        log.info("News checkpoint (%s): %d companies done", platform, len(completed_set))

    total = _total_stocks()
    start_index = max(1, args.offset)
    if start_index > total:
        log.warning("offset %d > total %d, skip news.", start_index, total)
        return

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
            done = 0
            for sector, companies in SECTORS.items():
                for name, symbol, market in companies:
                    done += 1
                    if done < start_index:
                        continue
                    company_key = _company_key(sector, name, symbol, market)
                    if company_key in completed_set:
                        log.info(
                            "── [%d/%d] %s / %s — skip [news, done]",
                            done, total, sector, name,
                        )
                        continue
                    log.info("── [%d/%d] %s / %s [news]", done, total, sector, name)
                    try:
                        fetch_eastmoney(page, None, name, symbol, market, sector)
                        completed_set.add(company_key)
                        news_save_platform_checkpoint(platform, completed_set)
                    except Exception as exc:
                        log.exception("news %s: %s", name, exc)
                    time.sleep(1)
        finally:
            browser.close()


def run_report(args: argparse.Namespace) -> None:
    _apply_sectors()
    use_checkpoint = not args.add
    platform = "sina"
    if args.add:
        news_reset_platform_checkpoint(platform)
        log.info("--add: checkpoint reset for %s", platform)
    completed_set = news_load_platform_checkpoint(platform, use_checkpoint)
    if use_checkpoint and completed_set:
        log.info("Report checkpoint (%s): %d companies done", platform, len(completed_set))

    total = _total_stocks()
    start_index = max(1, args.offset)
    if start_index > total:
        log.warning("offset %d > total %d, skip report.", start_index, total)
        return

    done = 0
    for sector, companies in SECTORS.items():
        for name, symbol, market in companies:
            done += 1
            if done < start_index:
                continue
            company_key = _company_key(sector, name, symbol, market)
            if use_checkpoint and company_key in completed_set:
                log.info(
                    "── [%d/%d] %s / %s — skip [report, done]",
                    done, total, sector, name,
                )
                continue
            log.info("── [%d/%d] %s / %s [report]", done, total, sector, name)
            try:
                fetch_sina(None, None, name, symbol, market, sector)
                completed_set.add(company_key)
                news_save_platform_checkpoint(platform, completed_set)
            except Exception as exc:
                log.exception("report %s: %s", name, exc)
            time.sleep(1)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )

    parser = argparse.ArgumentParser(
        description="Fetch comments / eastmoney news / sina reports into exports/*.csv (unified SECTORS).",
    )
    parser.add_argument(
        "--mode",
        choices=("all", "comments", "news", "report"),
        default="all",
        help="all=forum + news + report; comments=股吧; news=eastmoney->news.csv; report=sina->report.csv",
    )
    parser.add_argument(
        "--add",
        action="store_true",
        help="Ignore checkpoints for selected mode(s); CSV still dedupes by url.",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=1,
        help="1-based stock index to start from (applies to each mode run).",
    )
    args = parser.parse_args()

    _apply_sectors()

    if args.mode == "all":
        if args.add:
            forum_reset_checkpoint_for_run()
            news_reset_platform_checkpoint("eastmoney")
            news_reset_platform_checkpoint("sina")
            log.info("--add: reset forum + eastmoney + sina checkpoints")
        log.info("=== mode=all: comments then news then report ===")
        run_comments(args)
        run_news(args)
        run_report(args)
    elif args.mode == "comments":
        run_comments(args)
    elif args.mode == "news":
        run_news(args)
    elif args.mode == "report":
        run_report(args)

    log.info("Done (mode=%s).", args.mode)


if __name__ == "__main__":
    main()
