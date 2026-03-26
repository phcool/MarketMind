"""
Fetch all East Money stock bar (股吧) posts for every company in A股重点板块.

Uses the JSONP CDN API endpoint:
  https://gbcdn.dfcfw.com/gbapi/webarticlelist_api_Article_Articlelist.js
    ?code=<symbol>&p=<page>&ps=<page_size>

Multi-threading per company:
  - At start, total_pages is read from the list page HTML or API; it is stored in checkpoint.
  - WORKERS_PER_COMPANY threads compete for pages via PageDispatcher. Fetch stops when
    either all pages are done or saved post count reaches MAX_POSTS_PER_COMPANY (10k).
  - When capped at 10k, checkpoint records capped_at so the company is skipped on resume.
  - Daily JSON files are protected by per-file locks to prevent concurrent writes.

Storage: A股重点板块/<sector>/<company>/forum/<YYYY>/<MM>/<DD>.json
Each file is a JSON array deduped by post_id (int-normalised).

Checkpoint: progress is saved to checkpoint/fetch_forum_checkpoint.json. Each
company records the list of completed page numbers so that on re-run we resume
from the next unprocessed page. Use --add to ignore checkpoint and re-fetch
from page 1 for all companies (existing data is merged, not cleared).
"""

import argparse
import json
import math
import re
import threading
import time
import logging
from collections import defaultdict
from datetime import datetime, date
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, wait

import requests
from bs4 import BeautifulSoup

# ── configuration ────────────────────────────────────────────────────────────

ROOT_DIR = Path(__file__).resolve().parent.parent
BASE_DIR = ROOT_DIR / "A股重点板块"
CHECKPOINT_DIR = ROOT_DIR / "checkpoint"
CHECKPOINT_FILE = CHECKPOINT_DIR / "fetch_forum_checkpoint.json"

API_URL       = "https://gbcdn.dfcfw.com/gbapi/webarticlelist_api_Article_Articlelist.js"
LIST_PAGE_URL = "https://guba.eastmoney.com/list,{symbol}.html"  # for parsing total pages
PAGE_SIZE     = 80
WORKERS_PER_COMPANY = 4    # concurrent threads per company
REQUEST_DELAY      = 1.5   # seconds each thread sleeps between page fetches
TIMEOUT            = 20
MAX_RETRIES        = 3
MAX_POSTS_PER_COMPANY = 10_000  # stop fetching this stock once we have saved this many posts
MAX_ZERO_SAVE_STREAK = 20       # stop stock if 20 consecutive pages have data but save 0

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

# ── company list ─────────────────────────────────────────────────────────────

SECTORS: dict[str, list[tuple[str, str, str]]] = {
    "电力设备与新能源": [
        ("宁德时代", "300750", "a"),
        ("亿纬锂能", "300014", "a"),
        ("阳光电源", "300274", "a"),
        ("隆基绿能", "601012", "a"),
        ("比亚迪",   "002594", "a"),
    ],
    "医药生物": [
        ("恒瑞医药", "600276", "a"),
        ("药明康德", "603259", "a"),
        ("复星医药", "600196", "a"),
        ("迈瑞医疗", "300760", "a"),
        ("云南白药", "000538", "a"),
    ],
    "银行": [
        ("招商银行", "600036", "a"),
        ("工商银行", "601398", "a"),
        ("平安银行", "000001", "a"),
        ("建设银行", "601939", "a"),
        ("兴业银行", "601166", "a"),
    ],
    "半导体与电子": [
        ("中微公司",   "688012", "a"),
        ("北方华创",   "002371", "a"),
        ("华虹半导体", "688347", "a"),
        ("韦尔股份",   "603501", "a"),
        ("兆易创新",   "603986", "a"),
    ],
    "食品饮料（白酒）": [
        ("贵州茅台", "600519", "a"),
        ("五粮液",   "000858", "a"),
        ("泸州老窖", "000568", "a"),
        ("洋河股份", "002646", "a"),
        ("山西汾酒", "600809", "a"),
    ],
    "汽车": [
        ("上汽集团", "600104", "a"),
        ("长城汽车", "601633", "a"),
        ("吉利汽车", "00175",  "hk"),
        ("广汽集团", "601238", "a"),
        ("江淮汽车", "600418", "a"),
    ],
    "非银金融（券商）": [
        ("中信证券", "600030", "a"),
        ("东方财富", "300059", "a"),
        ("国泰君安", "601211", "a"),
        ("华泰证券", "601688", "a"),
        ("广发证券", "000776", "a"),
    ],
}

# ── logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)s] %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── per-file write locks (prevents concurrent read-modify-write on same file) ─

_file_locks: dict[Path, threading.Lock] = {}
_file_locks_meta = threading.Lock()


def _get_file_lock(path: Path) -> threading.Lock:
    with _file_locks_meta:
        if path not in _file_locks:
            _file_locks[path] = threading.Lock()
        return _file_locks[path]


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


# ── thread-safe file write ────────────────────────────────────────────────────

def save_posts(posts: list[dict], company_dir: Path) -> int:
    """
    Group posts by publish date, acquire per-file lock, then merge + write.
    Returns total number of new entries actually written.
    """
    by_date: dict[date, list[dict]] = defaultdict(list)
    for p in posts:
        try:
            d = datetime.strptime(p["post_publish_time"], "%Y-%m-%d %H:%M:%S").date()
        except (ValueError, KeyError):
            continue
        by_date[d].append(p)

    total_new = 0
    for day, day_posts in by_date.items():
        out_dir  = company_dir / "forum" / str(day.year) / f"{day.month:02d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{day.day:02d}.json"

        with _get_file_lock(out_file):
            existing: list[dict] = []
            if out_file.exists():
                try:
                    existing = json.loads(out_file.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    existing = []

            seen_ids: set[int] = set()
            for p in existing:
                raw_id = p.get("post_id")
                if raw_id is not None:
                    try:
                        seen_ids.add(int(raw_id))
                    except (ValueError, TypeError):
                        pass

            new_entries: list[dict] = []
            for p in day_posts:
                raw_id = p.get("post_id")
                if raw_id is None:
                    continue
                try:
                    pid = int(raw_id)
                except (ValueError, TypeError):
                    continue
                if pid not in seen_ids:
                    seen_ids.add(pid)
                    new_entries.append(p)

            if not new_entries:
                continue

            merged = existing + new_entries
            out_file.write_text(
                json.dumps(merged, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            total_new += len(new_entries)

    return total_new


# ── worker function (runs in a thread) ───────────────────────────────────────

def _worker(symbol: str, company_dir: Path, company_key: str,
            dispatcher: PageDispatcher,
            saved_counter: list[int],
            zero_save_streak: list[int],
            counter_lock: threading.Lock,
            total_pages: int) -> None:
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

        n = save_posts(posts, company_dir) if posts else 0
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
            "page %4d / %d | in-range %3d | saved %3d (total %d) | zero-save-streak %d",
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

def folder_name(name: str, symbol: str, market: str) -> str:
    suffix = f"{symbol}.HK" if market == "hk" else symbol
    return f"{name}({suffix})"


def fetch_company(name: str, symbol: str, market: str, sector: str,
                  use_checkpoint: bool) -> None:
    company_key    = _company_key(sector, name, symbol, market)
    company_dir    = BASE_DIR / sector / folder_name(name, symbol, market)
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
                _worker, symbol, company_dir, company_key,
                dispatcher, saved_counter, zero_save_streak, counter_lock, total_pages,
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
        log.info("━━ %s done — %d posts saved (capped at %d) ━━\n", name, total_saved, MAX_POSTS_PER_COMPANY)
    elif stop_reason:
        # Treat as finished: persist compact int form so future runs skip this stock.
        with _checkpoint_lock:
            _checkpoint_data[company_key] = total_pages
            _write_checkpoint_file()
        log.info("  stop reason: %s", stop_reason)
        log.info("━━ %s done — %d posts saved (marked finished by stop rule) ━━\n", name, total_saved)
    else:
        finalize_company_checkpoint(company_key, dispatcher)
        log.info("━━ %s done — %d posts saved ━━\n", name, total_saved)


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch East Money stock bar posts. Use --add to re-fetch from page 1 (merge new content).",
    )
    parser.add_argument(
        "--add",
        action="store_true",
        help="Ignore checkpoint and re-fetch all companies from page 1; existing data is merged, not cleared.",
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
        log.info("--add: checkpoint reset, will fetch from page 1 for all companies")
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
