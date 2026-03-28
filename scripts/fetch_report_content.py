"""
Fetch and fill content for rows in the report table where content IS NULL.

Page structure (Sina Finance research report):
  div.blk_container > p  (each <p> is a paragraph; <br> = newline within p)

Run repeatedly to resume after interruption — only fetches rows with content IS NULL.
"""

import time
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import psycopg2
import psycopg2.extras
import requests
from bs4 import BeautifulSoup

# ── config ────────────────────────────────────────────────────────────────────
DSN              = "dbname=financial_data"
WORKERS          = 1      # single thread to avoid rate limiting
REQUEST_DELAY    = 2.0    # seconds between requests
REQUEST_TIMEOUT  = 20
BATCH_SIZE       = 50     # DB update batch size
MAX_RETRIES      = 2

HEADERS = {
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

# ── HTML parsing ──────────────────────────────────────────────────────────────

def parse_content(html: str) -> str:
    """Extract plain text from div.blk_container paragraphs (<br> -> newline)."""
    soup = BeautifulSoup(html, "lxml")
    container = soup.select_one("div.blk_container")
    if not container:
        # fallback: try the broader content div
        container = soup.select_one("div.content")
    if not container:
        return ""

    parts: list[str] = []
    for p in container.find_all("p"):
        # replace <br> with newline, then get text
        for br in p.find_all("br"):
            br.replace_with("\n")
        text = p.get_text(" ").strip()
        if text:
            parts.append(text)

    return "\n\n".join(parts)


def fetch_report_plaintext(url: str, timeout: int = REQUEST_TIMEOUT) -> str:
    """
    HTTP GET one Sina-style research report page and return body text from div.blk_container > p.
    Returns "" on 4xx or empty parse; raises on repeated network/5xx if caller handles.
    """
    session = requests.Session()
    session.headers.update(HEADERS)
    resp = session.get(url, timeout=timeout)
    if 400 <= resp.status_code < 500:
        return ""
    resp.raise_for_status()
    resp.encoding = "gb2312"
    return parse_content(resp.text)


# ── HTTP fetch ────────────────────────────────────────────────────────────────

def fetch_url(session: requests.Session, url: str) -> str | None:
    """Return parsed content string, '' if permanently unavailable, None on transient error."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.get(url, timeout=REQUEST_TIMEOUT)
            # 4xx = server rejected the request (paywall / access denied) — don't retry
            if 400 <= resp.status_code < 500:
                log.warning("  HTTP %d (skip) %s", resp.status_code, url)
                return ""
            resp.raise_for_status()
            resp.encoding = "gb2312"
            return parse_content(resp.text)
        except requests.HTTPError:
            raise
        except Exception as exc:
            log.warning("  [%d/%d] %s — %s", attempt, MAX_RETRIES, url, exc)
            if attempt < MAX_RETRIES:
                time.sleep(3)
    return None


def fetch_url_with_backoff(
    session: requests.Session,
    url: str,
    *,
    timeout: float,
    retry_base: float,
    max_attempts: int,
) -> str | None:
    """
    Like fetch_url but with exponential backoff between transient failures.
    Sleep before retry k: retry_base * 2^(k-1). Returns '' on 4xx or empty parse after
    all attempts; None if only transient errors and attempts exhausted.
    """
    empty_after_200 = False
    for attempt in range(1, max_attempts + 1):
        try:
            resp = session.get(url, timeout=timeout)
            if 400 <= resp.status_code < 500:
                log.warning("  HTTP %d (skip) %s", resp.status_code, url)
                return ""
            resp.raise_for_status()
            resp.encoding = "gb2312"
            text = parse_content(resp.text)
            if (text or "").strip():
                return text
            empty_after_200 = True
            log.warning("  [%d/%d] empty parse %s", attempt, max_attempts, url[:80])
        except requests.HTTPError as exc:
            code = exc.response.status_code if exc.response is not None else None
            if code is not None and 400 <= code < 500:
                return ""
            log.warning("  [%d/%d] HTTP error %s — %s", attempt, max_attempts, url[:80], exc)
        except Exception as exc:
            log.warning("  [%d/%d] %s — %s", attempt, max_attempts, url[:80], exc)
        if attempt < max_attempts:
            wait = retry_base * (2 ** (attempt - 1))
            log.info("  retry in %.1fs", wait)
            time.sleep(wait)
    return "" if empty_after_200 else None


# ── worker ────────────────────────────────────────────────────────────────────

def worker(url: str) -> tuple[str, str | None]:
    session = requests.Session()
    session.headers.update(HEADERS)
    content = fetch_url(session, url)
    time.sleep(REQUEST_DELAY)
    return url, content


def fetch_report_worker_disk(
    url: str,
    *,
    timeout: float,
    retry_base: float,
    max_attempts: int,
    request_delay: float,
) -> tuple[str, str | None]:
    """For fetch_report_content_to_disk: backoff retries then polite delay before next task."""
    session = requests.Session()
    session.headers.update(HEADERS)
    content = fetch_url_with_backoff(
        session,
        url,
        timeout=timeout,
        retry_base=retry_base,
        max_attempts=max_attempts,
    )
    time.sleep(request_delay)
    return url, content


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    conn = psycopg2.connect(DSN)
    conn.autocommit = False
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM report WHERE content IS NULL")
    total = cur.fetchone()[0]
    log.info("Rows with content IS NULL: %d", total)
    if total == 0:
        log.info("Nothing to do.")
        cur.close()
        conn.close()
        return

    cur.execute("SELECT url FROM report WHERE content IS NULL ORDER BY date")
    urls = [row[0] for row in cur.fetchall()]

    done = 0
    failed = 0
    pending_updates: list[tuple[str, str]] = []

    def flush_updates():
        if not pending_updates:
            return
        psycopg2.extras.execute_batch(
            cur,
            "UPDATE report SET content = %s WHERE url = %s",
            [(c, u) for u, c in pending_updates],
        )
        conn.commit()
        pending_updates.clear()

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(worker, url): url for url in urls}
        for future in as_completed(futures):
            url, content = future.result()
            done += 1
            if content is None:
                log.warning("TRANSIENT FAIL  [%d/%d] %s", done, total, url)
                failed += 1
                continue

            if content == "":
                log.debug("UNAVAILABLE  [%d/%d] %s", done, total, url)
            else:
                log.info("OK  [%d/%d] chars=%d  %s", done, total, len(content), url[:80])

            pending_updates.append((url, content))

            if len(pending_updates) >= BATCH_SIZE:
                flush_updates()

    flush_updates()
    cur.close()
    conn.close()

    log.info("Done. Fetched: %d  Failed: %d  Total: %d", done - failed, failed, total)


if __name__ == "__main__":
    main()
