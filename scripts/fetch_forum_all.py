"""
Fetch all East Money stock bar (股吧) posts for every company in config/stocks.json.

Uses the JSONP CDN API endpoint:
  https://gbcdn.dfcfw.com/gbapi/webarticlelist_api_Article_Articlelist.js
    ?code=<symbol>&p=<page>&ps=<page_size>

Multi-threading per company:
  - At start, total_pages is read from the list page HTML or API; it is stored in checkpoint.
  - WORKERS_PER_COMPANY threads compete for pages via PageDispatcher. Fetch stops when
    either all pages are done or inserted row count reaches MAX_POSTS_PER_COMPANY (10k).
  - When capped at 10k, checkpoint records capped_at so the company is skipped on resume.

Storage: exports/comments.csv (dedup by url; each successful merge rewrites trimmed to the top
COMMENTS_MAX_PER_SYMBOL_DAY rows per symbol per calendar day by click_count — see csv_io.py).

Checkpoint: checkpoint/fetch_forum_checkpoint.json — completed page numbers per company.
Use --add to ignore checkpoint and re-fetch from page 1 for all companies.
"""

import argparse
import json
import math
import re
import threading
import time
import logging
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, wait

import requests
from bs4 import BeautifulSoup

from csv_io import merge_append_comments
from stock_universe import load_sectors

# ── configuration ────────────────────────────────────────────────────────────

ROOT_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = ROOT_DIR / "checkpoint"
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

SECTORS = load_sectors()

# ── logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)s] %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── checkpoint (per-company: total_pages + completed) ─────────────────────────
# Value: int N (all 1..N done), or dict {"total_pages": N, "completed": list | int}.

_checkpoint_lock = threading.Lock()
_checkpoint_data: dict[str, int | dict] = {}


def _company_key(sector: str, name: str, symbol: str, market: str) -> str:
    suffix = f"{symbol}.HK" if market == "hk" else symbol
    return f"{sector}/{name}({suffix})"


def load_checkpoint(company_key: str, use_checkpoint: bool) -> tuple[set[int], int | None]:
    """
    Load (completed page set, total_pages or None) for this company.
    If total_pages is None, caller must call get_total_pages().
    """
    if not use_checkpoint:
        return set(), None
    with _checkpoint_lock:
        data = _checkpoint_data.get(company_key)
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
    with _checkpoint_lock:
        existing = _checkpoint_data.get(company_key)
        if isinstance(existing, dict) and total_pages is None:
            total_pages = existing.get("total_pages")
        _checkpoint_data[company_key] = {
            "total_pages": total_pages or 1,
            "completed": sorted(completed),
        }
        _write_checkpoint_file()


def _write_checkpoint_file() -> None:
    path = CHECKPOINT_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(
            json.dumps(_checkpoint_data, ensure_ascii=False, indent=2),
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
        with _checkpoint_lock:
            _checkpoint_data[company_key] = max_page
            _write_checkpoint_file()
        log.debug("Checkpoint compacted for %s → %d pages", company_key, max_page)


def load_all_checkpoints(use_checkpoint: bool) -> None:
    """Load checkpoint file into _checkpoint_data. Call once at startup."""
    global _checkpoint_data
    if not use_checkpoint:
        _checkpoint_data = {}
        return
    path = CHECKPOINT_FILE
    if not path.exists():
        _checkpoint_data = {}
        return
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
        _checkpoint_data = data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError) as e:
        log.warning("Checkpoint load failed: %s", e)
        _checkpoint_data = {}


def reset_checkpoint_for_run() -> None:
    """Clear in-memory checkpoint so all companies start from page 1 (--add)."""
    global _checkpoint_data
    with _checkpoint_lock:
        _checkpoint_data = {}
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
    initial_done, checkpoint_total = load_checkpoint(company_key, use_checkpoint)
    total_pages    = checkpoint_total if checkpoint_total is not None else get_total_pages(symbol)
    log.info("━━ %s (%s) — total %d pages ━━", name, symbol, total_pages)
    if initial_done:
        log.info("  resuming from checkpoint (%d pages already done)", len(initial_done))

    # Persist total_pages into checkpoint so resume and logs have it
    with _checkpoint_lock:
        _checkpoint_data[company_key] = {"total_pages": total_pages, "completed": sorted(initial_done)}
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
        with _checkpoint_lock:
            _checkpoint_data[company_key] = {
                "total_pages": total_pages,
                "completed": sorted(all_pages),
                "capped_at": MAX_POSTS_PER_COMPANY,
            }
            _write_checkpoint_file()
        log.info("━━ %s done — %d new rows inserted (capped at %d) ━━\n", name, total_saved, MAX_POSTS_PER_COMPANY)
    elif stop_reason:
        # Treat as finished: persist compact int form so future runs skip this stock.
        with _checkpoint_lock:
            _checkpoint_data[company_key] = total_pages
            _write_checkpoint_file()
        log.info("  stop reason: %s", stop_reason)
        log.info("━━ %s done — %d new rows (marked finished by stop rule) ━━\n", name, total_saved)
    else:
        finalize_company_checkpoint(company_key, dispatcher)
        log.info("━━ %s done — %d new rows inserted ━━\n", name, total_saved)


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch East Money stock bar posts into exports/comments.csv. Use --add to re-fetch from page 1.",
    )
    parser.add_argument(
        "--add",
        action="store_true",
        help="Ignore checkpoint and re-fetch all companies from page 1; CSV skips duplicate urls.",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=1,
        help="Start from the N-th stock in the full list (1-based), skipping earlier ones.",
    )
    args = parser.parse_args()
    use_checkpoint = not args.add

    if args.add:
        reset_checkpoint_for_run()
        log.info("--add: checkpoint reset, will fetch from page 1 for all companies (CSV dedup by url)")
    else:
        load_all_checkpoints(use_checkpoint=True)
        if _checkpoint_data:
            log.info("Checkpoint loaded: %d companies", len(_checkpoint_data))

    total = sum(len(v) for v in SECTORS.values())
    start_index = max(1, args.offset)
    if start_index > total:
        log.warning("offset %d > total stocks %d, nothing to do.", start_index, total)
        return
    if start_index > 1:
        log.info("offset=%d: will skip first %d stocks", start_index, start_index - 1)
    done  = 0

    for sector, companies in SECTORS.items():
        for name, symbol, market in companies:
            done += 1
            if done < start_index:
                log.info("── [%d/%d] %s / %s — skip (offset)", done, total, sector, name)
                continue
            log.info("── [%d/%d] %s / %s", done, total, sector, name)
            try:
                fetch_company(name, symbol, market, sector, use_checkpoint)
            except Exception as exc:
                log.exception("Unexpected error for %s: %s", name, exc)


if __name__ == "__main__":
    main()
