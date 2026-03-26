"""
Test script: scrape East Money stock bar (股吧) posts for 云南白药 (000538).

Instead of parsing the HTML table (which only shows 12h time without AM/PM),
this script extracts the `article_list` JavaScript variable embedded in the
page source, which contains complete 24-hour timestamps.
"""

import requests
import re
import json
from pathlib import Path
from datetime import datetime
from bs4 import BeautifulSoup

ROOT_DIR = Path(__file__).resolve().parent.parent

FIRST_PAGE = "https://guba.eastmoney.com/list,{symbol}.html"
NEXT_PAGE   = "https://guba.eastmoney.com/list,{symbol},f_{page}.html"

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
    "post_publish_time",   # full 24h datetime, e.g. "2026-03-09 13:54:20"
    "post_last_time",      # time of latest reply
    "post_type",
    "post_has_pic",
    "post_has_video",
    "art_unique_url",      # direct link (caifuhao or guba)
]


def parse_article_list(html: str) -> list[dict]:
    """Extract post list from the embedded `article_list` JS variable."""
    soup = BeautifulSoup(html, "lxml")
    for script in soup.find_all("script"):
        text = script.string or ""
        if "article_list" not in text or "post_publish_time" not in text:
            continue
        m = re.search(r"var article_list=(\{.*?\});", text, re.DOTALL)
        if not m:
            continue
        data = json.loads(m.group(1))
        posts = []
        for raw in data.get("re", []):
            # build canonical URL
            unique = raw.get("art_unique_url", "")
            if not unique:
                pid  = raw.get("post_id", "")
                code = raw.get("stockbar_code", "")
                unique = f"https://guba.eastmoney.com/news,{code},{pid}.html"
            entry = {k: raw.get(k) for k in KEEP_FIELDS}
            entry["url"] = unique
            posts.append(entry)
        return posts
    return []


def fetch_forum(symbol: str, max_pages: int = 3) -> list[dict]:
    """Fetch up to max_pages pages of posts for the given stock symbol."""
    all_posts = []
    session = requests.Session()
    session.headers.update(HEADERS)

    for page in range(1, max_pages + 1):
        url = FIRST_PAGE.format(symbol=symbol) if page == 1 \
              else NEXT_PAGE.format(symbol=symbol, page=page)
        print(f"  Page {page}: {url}")
        try:
            resp = session.get(url, timeout=15)
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"  Request failed: {e}")
            break

        posts = parse_article_list(resp.text)
        if not posts:
            print("  No posts found, stopping.")
            break
        all_posts.extend(posts)
        print(f"  Got {len(posts)} posts  (total: {len(all_posts)})")

    return all_posts


def main():
    symbol = "000538"
    name   = "云南白药"
    print(f"=== Scraping forum for {name}({symbol}) ===")

    posts = fetch_forum(symbol, max_pages=3)

    out_dir  = ROOT_DIR / f"A股重点板块/医药生物/{name}({symbol})/forum"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"posts_{datetime.now().strftime('%Y%m%d')}.json"
    out_file.write_text(
        json.dumps(posts, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nSaved {len(posts)} posts -> {out_file}")

    print("\n--- Preview (first 5 posts) ---")
    for p in posts[:5]:
        print(
            f"[read:{p['post_click_count']:>5} reply:{p['post_comment_count']:>3}]"
            f"  publish:{p['post_publish_time']}  last:{p['post_last_time']}"
            f"\n  {p['user_nickname']:<20}  {p['post_title'][:45]}"
        )
        print()


if __name__ == "__main__":
    main()
