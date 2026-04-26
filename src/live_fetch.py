from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from pathlib import Path
import re

import pandas as pd
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from playwright.sync_api import sync_playwright

try:
    import tushare as ts
except ImportError:  # pragma: no cover - handled at runtime
    ts = None

ROOT_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
load_dotenv(ROOT_DIR / ".env")

from fetch.fetch_content_to_disk import (  # type: ignore  # noqa: E402
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
from fetch.fetch_market_data import (  # type: ignore  # noqa: E402
    HEADERS as COMMENT_HEADERS,
    SEARCH_URL_EASTMONEY,
    SEARCH_URL_SINA,
    SINA_BASE,
    fetch_api_page,
    load_sectors,
    parse_posts,
)
SERP_API_URL = "https://serpapi.com/search.json"
DASHSCOPE_BASE_URL = os.getenv(
    "DASHSCOPE_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
)
DEFAULT_ONLINE_MODEL = os.getenv(
    "QWEN_ONLINE_MODEL",
    os.getenv("QWEN_SUMMARY_MODEL", os.getenv("QWEN_SERP_FILTER_MODEL", "qwen3.6-max-preview")),
)
DEFAULT_SUMMARY_MODEL = DEFAULT_ONLINE_MODEL
DEFAULT_PREDICT_MODEL = DEFAULT_ONLINE_MODEL


def normalize_symbol(symbol: str) -> str:
    return (symbol or "").strip().upper()


def serp_api_key() -> str:
    return (os.getenv("SERP_API_KEY") or "").strip()


def dashscope_api_key() -> str:
    return (os.getenv("DASHBOARD_API_KEY") or os.getenv("DASHSCOPE_API_KEY") or "").strip()


def tushare_token() -> str:
    return (os.getenv("TUSHARE_TOKEN") or os.getenv("TUSHARE_API_TOKEN") or "").strip()


def openai_client():
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - import guard
        raise RuntimeError("当前环境未安装 openai，无法调用 Qwen Summary。") from exc
    api_key = dashscope_api_key()
    if not api_key:
        raise RuntimeError("未配置 DASHBOARD_API_KEY 或 DASHSCOPE_API_KEY。")
    return OpenAI(api_key=api_key, base_url=DASHSCOPE_BASE_URL)


def resolve_stock(symbol: str) -> tuple[str, str, str]:
    target = normalize_symbol(symbol)
    if not target:
        raise ValueError("请输入股票代码。")
    for _sector, companies in load_sectors().items():
        for name, sym, market in companies:
            if sym.upper() == target:
                return name, sym, market
    raise ValueError(f"未在当前股票池中找到代码 {target}。")


def to_tushare_code(symbol: str, market: str) -> str:
    if market != "a":
        raise ValueError("当前实时网站的 Tushare K 线仅支持 A 股股票。")
    suffix = "SH" if symbol.startswith(("6", "9")) else "SZ"
    return f"{symbol}.{suffix}"


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


def truncate_text(text: str, limit: int) -> str:
    value = (text or "").strip()
    if len(value) <= limit:
        return value
    return value[:limit].rstrip() + "……"


def build_section_summary_messages(
    section: str,
    stock: str,
    company_name: str,
    items: list[dict[str, object]],
) -> tuple[str, str]:
    if section == "comments":
        system_prompt = (
            "你是一名金融研究助理。请根据给定的评论标题及互动数据，"
            "总结市场讨论中最集中的几个观点、情绪方向和分歧点。"
            "不要编造未提供的信息，使用简洁中文输出。"
        )
        lines = [
            f"股票：{company_name}（{stock}）",
            f"评论数量：{len(items)}",
            "",
            "请基于下面这些评论标题、发布时间、点击量和评论量，总结这些评论表达的几个集中观点。",
            "输出要求：",
            "1. 先给出“整体情绪判断”；",
            "2. 再列出 3-5 个“集中观点”；",
            "3. 最后补充“主要分歧或风险点”。",
            "",
            "评论列表：",
        ]
        for idx, item in enumerate(items, start=1):
            lines.append(
                f"{idx}. 时间={item.get('publish_time', '')} "
                f"点击={item.get('click_count', 0)} 评论={item.get('comment_count', 0)} "
                f"标题={item.get('title', '')}"
            )
        return system_prompt, "\n".join(lines)

    if section == "news":
        system_prompt = (
            "你是一名金融研究助理。请根据给定的新闻标题、日期和正文内容，"
            "提炼这些新闻共同表达的几个集中观点、潜在利好利空因素与事件主线。"
            "不要编造未提供的信息，使用简洁中文输出。"
        )
        lines = [
            f"股票：{company_name}（{stock}）",
            f"新闻数量：{len(items)}",
            "",
            "请基于下面这些新闻，总结它们共同表达的几个集中观点。",
            "输出要求：",
            "1. 先给出“核心事件主线”；",
            "2. 再列出 3-5 个“集中观点”；",
            "3. 最后补充“可能的利好/利空影响”。",
            "",
            "新闻列表：",
        ]
        for idx, item in enumerate(items, start=1):
            lines.append(
                f"{idx}. 日期={item.get('date', '')}\n"
                f"标题={item.get('title', '')}\n"
                f"正文={truncate_text(str(item.get('content', '')), 1200)}\n"
            )
        return system_prompt, "\n".join(lines)

    if section == "reports":
        system_prompt = (
            "你是一名金融研究助理。请根据给定的研报标题、日期和正文内容，"
            "提炼这些研报共同表达的几个集中观点、分析逻辑和风险提示。"
            "不要编造未提供的信息，使用简洁中文输出。"
        )
        lines = [
            f"股票：{company_name}（{stock}）",
            f"研报数量：{len(items)}",
            "",
            "请基于下面这些研报，总结它们共同表达的几个集中观点。",
            "输出要求：",
            "1. 先给出“机构总体判断”；",
            "2. 再列出 3-5 个“集中观点”；",
            "3. 最后补充“主要风险提示”。",
            "",
            "研报列表：",
        ]
        for idx, item in enumerate(items, start=1):
            lines.append(
                f"{idx}. 日期={item.get('date', '')}\n"
                f"标题={item.get('title', '')}\n"
                f"正文={truncate_text(str(item.get('content', '')), 1600)}\n"
            )
        return system_prompt, "\n".join(lines)

    if section == "serp":
        system_prompt = (
            "你是一名金融研究助理。请根据给定的搜索结果标题、来源、日期和正文内容，"
            "提炼这些网页共同表达的几个集中观点、事件主线和潜在影响。"
            "不要编造未提供的信息，使用简洁中文输出。"
        )
        lines = [
            f"股票：{company_name}（{stock}）",
            f"搜索结果数量：{len(items)}",
            "",
            "请基于下面这些网页正文，总结它们共同表达的几个集中观点。",
            "输出要求：",
            "1. 先给出“外部信息主线”；",
            "2. 再列出 3-5 个“集中观点”；",
            "3. 最后补充“对股票可能的影响或需警惕的风险点”。",
            "",
            "网页列表：",
        ]
        for idx, item in enumerate(items, start=1):
            lines.append(
                f"{idx}. 来源={item.get('source', '')} 日期={item.get('date', '')}\n"
                f"标题={item.get('title', '')}\n"
                f"正文={truncate_text(str(item.get('content', '')), 1500)}\n"
            )
        return system_prompt, "\n".join(lines)

    raise ValueError(f"不支持的 summary section: {section}")


def summarize_section(
    section: str,
    stock: str,
    company_name: str,
    items: list[dict[str, object]],
    *,
    model: str = DEFAULT_SUMMARY_MODEL,
) -> str:
    normalized_section = (section or "").strip().lower()
    if normalized_section not in {"comments", "news", "reports", "serp"}:
        raise ValueError("仅支持 comments、news、reports、serp 的 Summary。")
    if not items:
        raise ValueError("当前分组没有可供总结的数据。")

    system_prompt, user_prompt = build_section_summary_messages(
        normalized_section,
        stock,
        company_name,
        items,
    )
    client = openai_client()
    response = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    text = response.choices[0].message.content if response.choices else ""
    result = (text or "").strip()
    if not result:
        raise RuntimeError("Qwen Summary 未返回有效内容。")
    return result


def format_kline_for_prompt(rows: list[dict[str, object]]) -> str:
    if not rows:
        return "无可用 K 线数据。"
    lines = []
    for idx, row in enumerate(rows, start=1):
        lines.append(
            f"{idx}. 日期={row.get('trade_date', '')} "
            f"开={row.get('open', '')} 收={row.get('close', '')} "
            f"高={row.get('high', '')} 低={row.get('low', '')} "
            f"涨跌额={row.get('change_amount', '')} 涨跌幅={row.get('pct_change', '')} "
            f"成交量={row.get('volume', '')} 成交额={row.get('amount', '')}"
        )
    return "\n".join(lines)


def build_prediction_messages(
    stock: str,
    company_name: str,
    kline: list[dict[str, object]],
    *,
    news_summary: str = "",
    reports_summary: str = "",
    serp_summary: str = "",
) -> tuple[str, str]:
    system_prompt = (
        "你是一名谨慎的金融研究助理。"
        "请结合给定的最近 7 个交易日 K 线数据，以及可用的新闻总结、研报总结、外部搜索结果总结，"
        "对未来 1 个交易日、3 个交易日、7 个交易日的股价走向做方向判断。"
        "只能使用已提供的信息，不要编造事实，不要给出投资建议。"
        "请使用简洁中文输出。"
    )
    lines = [
        f"股票：{company_name}（{stock}）",
        "",
        "最近 7 个交易日 K 线：",
        format_kline_for_prompt(kline),
        "",
        "可用的外部信息总结如下。若某部分为空，说明当前未生成该部分 Summary，请忽略它。",
        "",
        f"新闻 Summary：\n{news_summary.strip() or '未提供'}",
        "",
        f"研报 Summary：\n{reports_summary.strip() or '未提供'}",
        "",
        f"SerpAPI Summary：\n{serp_summary.strip() or '未提供'}",
        "",
        "请输出以下格式：",
        "【输出要求（必须严格遵守）】",
        "请先分析 K 线趋势、量能、波动，以及已提供的 news / reports / SerpAPI Summary；",
        "将推理过程写在下面一对标记之间（只写推理过程）：",
        "【思维链开始】",
        "（在此撰写推理过程）",
        "【思维链结束】",
        "",
        "思维链结束后，请严格按下面三行输出最终答案：",
        "未来1个交易日：上涨或下跌",
        "未来3个交易日：上涨或下跌",
        "未来7个交易日：上涨或下跌",
        "",
        "注意：",
        "1. 最终答案每一行只能出现“上涨”或“下跌”，不能输出“震荡”“持平”“不确定”等其他词；",
        "2. 不要在最终答案三行后追加其他解释；",
        "3. 如果部分 Summary 缺失，请基于剩余 Summary 与 K 线继续判断；不要因为信息缺失拒绝回答。",
        "",
    ]
    return system_prompt, "\n".join(lines)


def predict_stock_direction(
    stock: str,
    company_name: str,
    kline: list[dict[str, object]],
    *,
    news_summary: str = "",
    reports_summary: str = "",
    serp_summary: str = "",
    model: str = DEFAULT_PREDICT_MODEL,
) -> str:
    if not kline:
        raise ValueError("当前没有可用于预测的 K 线数据。")

    system_prompt, user_prompt = build_prediction_messages(
        stock,
        company_name,
        kline,
        news_summary=news_summary,
        reports_summary=reports_summary,
        serp_summary=serp_summary,
    )
    client = openai_client()
    response = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    text = response.choices[0].message.content if response.choices else ""
    result = (text or "").strip()
    if not result:
        raise RuntimeError("Qwen Predict 未返回有效内容。")
    return result


def filter_serp_results(
    stock: str,
    company_name: str,
    items: list[dict[str, object]],
) -> tuple[list[dict[str, object]], str]:
    if not items:
        raise ValueError("当前没有可供筛选的 SerpAPI 结果。")
    filtered = []
    removed_pdf = 0
    for item in items:
        url = str(item.get("url") or "").strip().lower()
        if url.endswith(".pdf"):
            removed_pdf += 1
            continue
        filtered.append(dict(item))

    note = f"已过滤掉 {removed_pdf} 条 PDF 链接，保留 {len(filtered)} / {len(items)} 条普通网页结果。"
    return filtered, note


def extract_generic_web_content(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for selector in (
        "#ContentBody",
        "article",
        "main",
        ".article",
        ".article-content",
        ".content",
        ".post-content",
        ".entry-content",
        ".newsContent",
        ".detail-content",
    ):
        node = soup.select_one(selector)
        if not node:
            continue
        paragraphs: list[str] = []
        for p in node.find_all(["p", "li"]):
            text = p.get_text(" ", strip=True)
            if text and len(text) >= 8:
                paragraphs.append(text)
        if paragraphs:
            return "\n\n".join(paragraphs[:40])
        text = node.get_text("\n", strip=True)
        if text:
            return text[:8000]

    paragraphs = []
    for p in soup.find_all("p"):
        text = p.get_text(" ", strip=True)
        if text and len(text) >= 20:
            paragraphs.append(text)
    if paragraphs:
        return "\n\n".join(paragraphs[:40])
    return ""


def fetch_serp_contents(
    items: list[dict[str, object]],
    *,
    max_attempts: int = 3,
    retry_base_delay: float = 1.5,
) -> list[dict[str, object]]:
    session = requests.Session()
    session.headers.update(NEWS_HEADERS)
    enriched: list[dict[str, object]] = []

    for item in items:
        enriched_item = dict(item)
        if str(enriched_item.get("content") or "").strip():
            enriched.append(enriched_item)
            continue
        url = str(enriched_item.get("url") or "").strip()
        if not url:
            enriched_item["content"] = ""
            enriched.append(enriched_item)
            continue

        body = ""
        last_exc: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                resp = session.get(url, timeout=20)
                resp.raise_for_status()
                if resp.apparent_encoding:
                    resp.encoding = resp.apparent_encoding
                body = extract_generic_web_content(resp.text).strip()
                break
            except Exception as exc:
                last_exc = exc
                if attempt >= max_attempts:
                    break
                wait_seconds = retry_base_delay * (2 ** (attempt - 1))
                print(
                    f"Serp content attempt {attempt}/{max_attempts} failed for {url}: {exc}; retrying after {wait_seconds:.1f}s ...",
                    flush=True,
                )
                import time

                time.sleep(wait_seconds)
        if not body and last_exc is not None:
            enriched_item["content_error"] = str(last_exc)
        enriched_item["content"] = body
        enriched.append(enriched_item)
    return enriched


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


def build_serp_queries(company_name: str, symbol: str) -> list[str]:
    return [
        f"{company_name} 股票 最新消息",
        f"{company_name} 股价 异动",
        f"{company_name} 公告",
        f"{company_name} 财报 业绩",
        f"{company_name} 研报",
        f"{symbol} {company_name}",
    ]


def serp_search_with_retry(
    session: requests.Session,
    params: dict[str, str | int],
    *,
    max_attempts: int = 3,
    retry_base_delay: float = 1.5,
) -> dict:
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            resp = session.get(SERP_API_URL, params=params, timeout=20)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            last_exc = exc
            if attempt >= max_attempts:
                break
            wait_seconds = retry_base_delay * (2 ** (attempt - 1))
            print(
                f"SerpAPI attempt {attempt}/{max_attempts} failed: {exc}; retrying after {wait_seconds:.1f}s ...",
                flush=True,
            )
            import time

            time.sleep(wait_seconds)
    assert last_exc is not None
    raise last_exc


def normalize_serp_items(data: dict, per_query: int) -> list[dict[str, str]]:
    merged: list[dict[str, str]] = []
    seen_links: set[str] = set()

    def _append_items(raw_items: list[dict]) -> None:
        for item in raw_items:
            link = str(item.get("link") or item.get("url") or "").strip()
            title = str(item.get("title") or "").strip()
            if not link or not title or link in seen_links:
                continue
            seen_links.add(link)
            merged.append(
                {
                    "title": title,
                    "url": link,
                    "snippet": str(item.get("snippet") or item.get("summary") or "").strip(),
                    "source": str(item.get("source") or "").strip(),
                    "date": str(item.get("date") or "").strip(),
                }
            )
            if len(merged) >= per_query:
                return

    _append_items(data.get("news_results") or [])
    if len(merged) < per_query:
        _append_items(data.get("organic_results") or [])
    return merged[:per_query]


def fetch_serpapi_results(
    company_name: str,
    symbol: str,
    per_query: int = 10,
) -> tuple[list[dict[str, str]], str]:
    api_key = serp_api_key()
    if not api_key:
        return [], "未配置 SERP_API_KEY，已跳过 SerpAPI 补充搜索。"

    session = requests.Session()
    merged_results: list[dict[str, str]] = []
    seen_urls: set[str] = set()
    failures: list[str] = []
    for query in build_serp_queries(company_name, symbol):
        params: dict[str, str | int] = {
            "engine": "google",
            "q": query,
            "api_key": api_key,
            "num": per_query,
            "hl": "zh-cn",
            "gl": "cn",
        }
        try:
            data = serp_search_with_retry(session, params)
            items = normalize_serp_items(data, per_query)
            for item in items:
                url = item.get("url", "").strip()
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                merged_results.append(item)
        except Exception as exc:
            print(f"SerpAPI query failed for {query}: {exc}", flush=True)
            failures.append(query)

    if failures:
        return merged_results, f"SerpAPI 有 {len(failures)} 组 query 抓取失败，其余结果已正常返回。"
    return merged_results, ""


def fetch_recent_kline(
    symbol: str,
    market: str,
    limit: int = 7,
    lookback_days: int = 45,
    max_attempts: int = 4,
    retry_base_delay: float = 2.0,
) -> tuple[list[dict[str, object]], str]:
    if ts is None:
        return [], "K 线抓取失败：当前环境未安装 tushare。"

    token = tushare_token()
    if not token:
        return [], "K 线抓取失败：未配置 TUSHARE_TOKEN。"

    try:
        ts_code = to_tushare_code(symbol, market)
    except ValueError as exc:
        return [], f"K 线抓取失败：{exc}"

    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=lookback_days)
    pro = ts.pro_api(token)
    df: pd.DataFrame | None = None
    last_exc: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            df = pro.daily(
                ts_code=ts_code,
                start_date=start.strftime("%Y%m%d"),
                end_date=end.strftime("%Y%m%d"),
            )
            break
        except Exception as exc:
            last_exc = exc
            if attempt >= max_attempts:
                break
            wait_seconds = retry_base_delay * (2 ** (attempt - 1))
            print(
                f"Tushare daily attempt {attempt}/{max_attempts} failed: {exc}; retrying after {wait_seconds:.1f}s ...",
                flush=True,
            )
            import time

            time.sleep(wait_seconds)

    if df is None:
        assert last_exc is not None
        return [], f"K 线抓取失败：{last_exc}"

    if df.empty:
        return [], "当前请求成功，但未抓到最近 7 个交易日 K 线。"

    recent_df = df.sort_values("trade_date").tail(limit)

    out: list[dict[str, object]] = []
    for _, row in recent_df.iterrows():
        trade_date_raw = str(row.get("trade_date") or "").strip()
        trade_date = trade_date_raw
        if len(trade_date_raw) == 8 and trade_date_raw.isdigit():
            trade_date = f"{trade_date_raw[:4]}-{trade_date_raw[4:6]}-{trade_date_raw[6:]}"
        out.append(
            {
                "trade_date": trade_date,
                "open": row.get("open"),
                "close": row.get("close"),
                "high": row.get("high"),
                "low": row.get("low"),
                "volume": row.get("vol"),
                "amount": row.get("amount"),
                "amplitude": None,
                "pct_change": row.get("pct_chg"),
                "change_amount": row.get("change"),
                "turnover": None,
            }
        )
    return out, ""


def fetch_stock_snapshot(symbol: str) -> dict[str, object]:
    company_name, normalized_symbol, market = resolve_stock(symbol)
    comments = fetch_recent_comments(normalized_symbol, limit=50)
    news = fetch_recent_news(normalized_symbol, company_name, limit=10)
    reports = fetch_recent_reports(normalized_symbol, market, limit=3)
    kline, kline_note = fetch_recent_kline(normalized_symbol, market, limit=7)
    serp_results, serp_note = fetch_serpapi_results(company_name, normalized_symbol, per_query=3)
    return {
        "stock": normalized_symbol,
        "name": company_name,
        "market": market,
        "comments": comments,
        "news": news,
        "reports": reports,
        "kline": kline,
        "serp_results": serp_results,
        "reports_note": "" if reports else "当前请求成功，但未抓到最近研报结果。",
        "kline_note": kline_note,
        "serp_note": serp_note,
    }
