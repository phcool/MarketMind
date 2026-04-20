from __future__ import annotations

import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
import re

import requests
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

ROOT_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from fetch_content_to_disk import (  # type: ignore  # noqa: E402
    NEWS_HEADERS,
    REPORT_HEADERS,
    build_news_file,
    build_report_file,
    fetch_news_with_retries,
    fetch_report_with_retries,
    file_has_nonempty_body,
    news_output_path,
    parse_date_str,
    report_output_path,
)
from fetch_market_data import (  # type: ignore  # noqa: E402
    HEADERS as COMMENT_HEADERS,
    SEARCH_URL_EASTMONEY,
    SEARCH_URL_SINA,
    SINA_BASE,
    fetch_api_page,
    load_sectors,
    parse_posts,
)


def normalize_symbol(symbol: str) -> str:
    return (symbol or "").strip().upper()


def resolve_stock(symbol: str) -> tuple[str, str, str]:
    target = normalize_symbol(symbol)
    if not target:
        raise ValueError("请输入股票代码。")
    for _sector, companies in load_sectors().items():
        for name, sym, market in companies:
            if sym.upper() == target:
                return name, sym, market
    raise ValueError(f"未在当前股票池中找到代码 {target}。")


def parse_content_file(path: Path) -> dict[str, str]:
    raw = path.read_text(encoding="utf-8")
    header, _, body = raw.partition("\n---\n")
    meta: dict[str, str] = {}
    for line in header.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        meta[key.strip().lower()] = value.strip()
    meta["content"] = body.strip()
    return meta


def fetch_recent_comments(symbol: str, limit: int = 10, max_pages: int = 3) -> list[dict[str, str | int]]:
    session = requests.Session()
    session.headers.update(COMMENT_HEADERS)
    collected: list[dict[str, str | int]] = []
    seen_urls: set[str] = set()

    for page_num in range(1, max_pages + 1):
        data = fetch_api_page(session, symbol, page_num)
        if not data:
            continue
        raw_posts = data.get("re") or []
        posts, _ = parse_posts(raw_posts)
        for post in posts:
            url = str(post.get("url") or "").strip()
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            published_at = str(post.get("post_publish_time") or "").strip()
            collected.append(
                {
                    "title": str(post.get("post_title") or "").strip(),
                    "url": url,
                    "publish_time": published_at,
                    "click_count": int(post.get("post_click_count") or 0),
                    "comment_count": int(post.get("post_comment_count") or 0),
                }
            )
            if len(collected) >= limit:
                break
        if len(collected) >= limit:
            break

    collected.sort(key=lambda item: str(item["publish_time"]), reverse=True)
    return collected[:limit]


@contextmanager
def eastmoney_page():
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
            yield page
        finally:
            browser.close()


def scrape_recent_news(company_name: str, limit: int = 10) -> list[dict[str, str]]:
    with eastmoney_page() as page:
        url = SEARCH_URL_EASTMONEY.format(keyword=company_name)
        page.goto(url, wait_until="domcontentloaded", timeout=30000)
        page.wait_for_selector(".news_list", timeout=10000)
        items = page.query_selector_all(".news_item")

        results: list[dict[str, str]] = []
        seen_urls: set[str] = set()
        for item in items:
            if len(results) >= limit:
                break
            link_el = item.query_selector(".news_item_t a")
            if not link_el:
                continue
            href = link_el.get_attribute("href")
            title = (link_el.inner_text() or "").strip()
            if not href or not title:
                continue
            if href.startswith("//"):
                href = "https:" + href
            elif href.startswith("/"):
                href = "https://so.eastmoney.com" + href
            if href in seen_urls:
                continue
            seen_urls.add(href)
            date_match = re.search(r"/a/(\d{8})\d*\.html", href)
            date_str = ""
            if date_match:
                try:
                    date_str = datetime.strptime(date_match.group(1), "%Y%m%d").date().isoformat()
                except ValueError:
                    date_str = ""
            results.append(
                {
                    "title": title,
                    "url": href,
                    "date": date_str,
                }
            )
        return results


def ensure_news_content(symbol: str, item: dict[str, str]) -> dict[str, str]:
    date_str = item.get("date") or ""
    news_date = parse_date_str(date_str)
    if news_date is None:
        raise ValueError(f"新闻日期解析失败: {item.get('url', '')}")
    url = item["url"]
    path = news_output_path(url, news_date)
    if file_has_nonempty_body(path):
        parsed = parse_content_file(path)
        return {
            "title": parsed.get("title", item.get("title", "")),
            "url": parsed.get("url", url),
            "date": parsed.get("date", date_str),
            "content": parsed.get("content", ""),
        }

    session = requests.Session()
    session.headers.update(NEWS_HEADERS)
    result = fetch_news_with_retries(
        session,
        url,
        timeout=20.0,
        retry_base=1.5,
        max_attempts=3,
        log_label="[web-news]",
    )
    if result.status != "ok":
        raise ValueError(f"新闻抓取失败: {url}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        build_news_file(url, symbol, item.get("title", ""), news_date, result.body),
        encoding="utf-8",
    )
    return {
        "title": item.get("title", ""),
        "url": url,
        "date": news_date.isoformat(),
        "content": result.body.strip(),
    }


def fetch_recent_news(symbol: str, company_name: str, limit: int = 10) -> list[dict[str, str]]:
    items = scrape_recent_news(company_name, limit=limit)
    return [ensure_news_content(symbol, item) for item in items]


def ensure_report_content(item: dict[str, str]) -> dict[str, str]:
    url = item["url"]
    report_date = parse_date_str(item.get("date") or "")
    path = report_output_path(url)
    if file_has_nonempty_body(path):
        parsed = parse_content_file(path)
        return {
            "title": parsed.get("title", item.get("title", "")),
            "url": parsed.get("url", url),
            "date": parsed.get("date", item.get("date", "")),
            "content": parsed.get("content", ""),
        }

    session = requests.Session()
    session.headers.update(REPORT_HEADERS)
    result = fetch_report_with_retries(
        session,
        url,
        timeout=20.0,
        retry_base=1.5,
        max_attempts=3,
        rate_limit_extra_delay=5.0,
    )
    if result.status != "ok":
        raise ValueError(f"研报抓取失败: {url}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        build_report_file(url, item.get("title", ""), report_date, result.body),
        encoding="utf-8",
    )
    return {
        "title": item.get("title", ""),
        "url": url,
        "date": report_date.isoformat() if report_date else "",
        "content": result.body.strip(),
    }


def scrape_recent_reports(symbol: str, market: str, limit: int = 3) -> list[dict[str, str]]:
    session = requests.Session()
    session.headers.update(REPORT_HEADERS)
    sina_symbol = symbol if market != "a" else (f"sh{symbol}" if symbol.startswith(("6", "9")) else f"sz{symbol}")
    seen_urls: set[str] = set()
    results: list[dict[str, str]] = []

    for page_num in range(1, 4):
        url = SEARCH_URL_SINA.format(symbol=sina_symbol, page=page_num)
        resp = session.get(url, timeout=20)
        resp.raise_for_status()
        resp.encoding = "gb2312"
        soup = BeautifulSoup(resp.text, "lxml")
        rows = [row for row in soup.select("table.tb_01 tr") if row.select_one("td.tal a")]
        if not rows:
            break
        for row in rows:
            link_el = row.select_one("td.tal a")
            if not link_el:
                continue
            href = (link_el.get("href") or "").strip()
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
            results.append(
                {
                    "title": title,
                    "url": href,
                    "date": date_str,
                }
            )
            if len(results) >= limit:
                return results
    return results


def fetch_recent_reports(symbol: str, market: str, limit: int = 3) -> list[dict[str, str]]:  # noqa: ARG001
    items = scrape_recent_reports(symbol, market, limit=limit)
    return [ensure_report_content(item) for item in items[:limit]]


def fetch_stock_snapshot(symbol: str) -> dict[str, object]:
    company_name, normalized_symbol, market = resolve_stock(symbol)
    comments = fetch_recent_comments(normalized_symbol, limit=10)
    news = fetch_recent_news(normalized_symbol, company_name, limit=10)
    reports = fetch_recent_reports(normalized_symbol, market, limit=3)
    return {
        "stock": normalized_symbol,
        "name": company_name,
        "market": market,
        "comments": comments,
        "news": news,
        "reports": reports,
    }
