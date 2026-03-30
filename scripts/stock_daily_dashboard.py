from __future__ import annotations

import hashlib
import json
import os
from datetime import date
from pathlib import Path

import pandas as pd
from flask import Flask, Response, jsonify, render_template_string, request, stream_with_context

from csv_io import COMMENTS_CSV, NEWS_CSV, QUOTES_CSV, REPORT_CSV
from fetch_report_content import fetch_report_plaintext
from stock_universe import load_sectors

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore[misc, assignment]

ROOT_DIR = Path(__file__).resolve().parent.parent
REPORT_CACHE_DIR = ROOT_DIR / "Content" / "report"
MAX_REPORT_BODY_CHARS = 8000


def _load_dotenv() -> None:
    """Load KEY=value pairs from repo root .env into os.environ (no override if already set)."""
    path = ROOT_DIR / ".env"
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


_load_dotenv()

# Qwen via DashScope OpenAI-compatible API (streaming for all calls)
DASHSCOPE_MODEL = os.getenv("DASHSCOPE_MODEL", os.getenv("QWEN_MODEL", "qwen3-max"))
DASHSCOPE_COMPAT_BASE_URL = os.getenv(
    "DASHSCOPE_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
)


def _dashscope_api_key() -> str | None:
    return os.getenv("DASHBOARD_API_KEY") or os.getenv("DASHSCOPE_API_KEY")


def _qwen_openai_client():
    """OpenAI-compatible client for DashScope; returns None if missing deps or key."""
    if OpenAI is None:
        return None
    key = _dashscope_api_key()
    if not key:
        return None
    return OpenAI(api_key=key, base_url=DASHSCOPE_COMPAT_BASE_URL)


def _chat_completions_stream(client, *, model: str, messages: list[dict]):
    """Stream chat completions; uses stream_options when the installed openai package supports it."""
    kwargs = dict(model=model, messages=messages, stream=True)
    try:
        return client.chat.completions.create(**kwargs, stream_options={"include_usage": True})
    except TypeError:
        return client.chat.completions.create(**kwargs)


app = Flask(__name__)


def discover_stocks() -> list[dict]:
    """Build picker entries from config/stocks.json (A-share 6-digit only for this UI)."""
    stocks: list[dict] = []
    for sector, companies in sorted(load_sectors().items()):
        for name, symbol, market in companies:
            code_text = f"{symbol}.HK" if market == "hk" else symbol
            code_digits = code_text.split(".")[0]
            if not (len(code_digits) == 6 and code_digits.isdigit()):
                continue
            stocks.append(
                {
                    "id": f"{name}|{code_text}",
                    "name": name,
                    "code_text": code_text,
                    "symbol": code_digits,
                    "sector": sector,
                }
            )

    stocks.sort(key=lambda x: (x["sector"], x["code_text"]))
    return stocks


STOCKS = discover_stocks()
STOCK_MAP = {s["id"]: s for s in STOCKS}


def query_data(stock: dict, day_str: str) -> dict:
    symbol = stock["symbol"]
    day_prefix = day_str[:10]

    comments: list[tuple] = []
    if COMMENTS_CSV.is_file():
        import csv as _csv

        with COMMENTS_CSV.open(encoding="utf-8", newline="") as f:
            for row in _csv.DictReader(f):
                if (row.get("symbol") or "").strip() != symbol:
                    continue
                pt = (row.get("publish_time") or "").strip().replace("T", " ")
                if len(pt) < 10 or pt[:10] != day_prefix:
                    continue
                comments.append(
                    (
                        (row.get("url") or "").strip(),
                        (row.get("post_title") or "").strip(),
                        pt,
                        row.get("click_count") or 0,
                        row.get("comment_count") or 0,
                    )
                )
        comments.sort(key=lambda x: str(x[2]), reverse=True)
        comments = comments[:500]

    news_rows: list[tuple] = []
    if NEWS_CSV.is_file():
        import csv as _csv

        with NEWS_CSV.open(encoding="utf-8", newline="") as f:
            for row in _csv.DictReader(f):
                if (row.get("symbol") or "").strip() != symbol:
                    continue
                ds = (row.get("date") or "").strip()[:10]
                if ds != day_prefix:
                    continue
                news_rows.append(
                    (
                        (row.get("url") or "").strip(),
                        (row.get("title") or "").strip(),
                        ds,
                    )
                )
        news_rows.sort(key=lambda x: (x[1] or ""))
        news_rows = news_rows[:500]

    report_candidates: list[tuple] = []
    if REPORT_CSV.is_file():
        import csv as _csv

        try:
            selected = pd.to_datetime(day_str).date()
        except Exception:
            selected = date.today()
        with REPORT_CSV.open(encoding="utf-8", newline="") as f:
            for row in _csv.DictReader(f):
                if (row.get("symbol") or "").strip() != symbol:
                    continue
                ds = (row.get("date") or "").strip()[:10]
                if not ds:
                    continue
                try:
                    rd = date.fromisoformat(ds)
                except ValueError:
                    continue
                if rd > selected:
                    continue
                report_candidates.append(
                    (
                        (row.get("url") or "").strip(),
                        (row.get("title") or "").strip(),
                        ds,
                    )
                )
        report_candidates.sort(key=lambda x: (x[2] or ""), reverse=True)
        report_rows = report_candidates[:3]
    else:
        report_rows = []

    return {"comments": comments, "news": news_rows, "reports": report_rows}


def _report_cache_path(page_url: str) -> Path:
    key = hashlib.sha256(page_url.encode("utf-8")).hexdigest()
    return REPORT_CACHE_DIR / f"{key}.txt"


def _read_report_cache_file(path: Path) -> str:
    """Return cached body text, or '' if missing/invalid."""
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return ""
    if raw.startswith("URL=") and "\n---\n" in raw:
        _, body = raw.split("\n---\n", 1)
        return body.strip()
    # Legacy: whole file is body
    return raw.strip()


def _write_report_cache_file(path: Path, page_url: str, title: str, report_date, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    date_s = str(report_date) if report_date is not None else ""
    meta = f"URL={page_url}\nTITLE={title}\nDATE={date_s}\n"
    path.write_text(meta + "---\n" + body, encoding="utf-8")


def load_or_fetch_report_body(page_url: str, title: str, report_date) -> str:
    """Load plaintext from Content/report/<sha256>.txt or fetch Sina page and cache."""
    path = _report_cache_path(page_url)
    if path.is_file():
        cached = _read_report_cache_file(path)
        if cached:
            return cached
    try:
        body = fetch_report_plaintext(page_url)
    except Exception:
        body = ""
    body = (body or "").strip()
    if body:
        _write_report_cache_file(path, page_url, title or "", report_date, body)
    return body


def build_recent_reports_summary_prompt(
    stock_name: str,
    day_str: str,
    reports_packed: list[tuple[int, str, str, object, str]],
) -> str:
    """
    reports_packed: (index, url, title, date, body) for each report (newest-first order).
    """
    blocks = []
    for idx, url, title, d, body in reports_packed:
        t = (title or "").strip()
        snippet = body.strip() if body else "(正文抓取失败或为空，请仅依据标题与日期推理，勿编造细节)"
        if len(snippet) > MAX_REPORT_BODY_CHARS:
            snippet = snippet[:MAX_REPORT_BODY_CHARS] + "\n...[truncated]..."
        blocks.append(
            f"--- Report #{idx} ---\n"
            f"title: {t}\n"
            f"date: {d}\n"
            f"url: {url}\n"
            f"body:\n{snippet}\n"
        )
    corpus = "\n".join(blocks)
    return (
        f"股票：{stock_name}\n"
        f"所选截止日期：{day_str}\n"
        f"以下为截至该日期的最近 {len(reports_packed)} 篇卖方/研报正文（按日期新到旧排列）。\n\n"
        f"{corpus}\n\n"
        "请用中文输出对该组研报的**综合总结**（可用 markdown 小标题分段），至少包含：\n"
        "1) 主要观点与逻辑主线；\n"
        "2) 共同提到的风险因素和潜在机会；\n"
        "3) 对投资者的简短结论要点。\n" 
        "勿编造正文中未出现的数据；若某篇仅有标题，请明确说明信息不足。\n"
    )


def stream_recent_reports_summary(stock_name: str, day_str: str, report_rows: list[tuple]):
    if not report_rows:
        yield sse_event({"type": "error", "text": "没有可总结的研报（最近3条为空）。"})
        yield sse_event({"type": "done"})
        return

    packed: list[tuple[int, str, str, object, str]] = []
    n = len(report_rows)
    for i, row in enumerate(report_rows, start=1):
        page_url, title, d = row
        yield sse_event({"type": "status", "text": f"正在获取研报正文 [{i}/{n}]（缓存优先）..."})
        body = load_or_fetch_report_body(page_url, title or "", d)
        packed.append((i, page_url, title or "", d, body))

    user_prompt = build_recent_reports_summary_prompt(stock_name, day_str, packed)
    yield from stream_qwen_sse_user_prompt(
        user_prompt,
        f"已汇总 {n} 篇研报正文，正在生成观点总结...",
    )


def sse_event(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _repair_mojibake(text: str) -> str:
    """Best-effort fix for UTF-8 text decoded as latin-1."""
    if not text:
        return text
    # Common mojibake markers like: ï¼ æ»ä½ ...
    if "ï" in text or "æ" in text or "ç" in text:
        try:
            return text.encode("latin-1").decode("utf-8")
        except Exception:
            return text
    return text


def build_summary_prompt(stock_name: str, day_str: str, comments: list[tuple]) -> str:
    """Build prompt with title + clicks + comment_count for weighting."""
    max_items = 200

    # row schema from query_data comments:
    # (url, post_title, publish_time, click_count, comment_count)
    ranked = sorted(
        comments,
        key=lambda r: ((r[3] or 0), (r[4] or 0)),
        reverse=True,
    )[:max_items]

    lines = []
    for idx, row in enumerate(ranked, 1):
        _url, title, _ts, clicks, ccount = row
        title = (title or "").strip()
        if not title:
            continue
        lines.append(f"{idx}. [clicks={clicks or 0}, comments={ccount or 0}] {title}")

    comments_text = "\n".join(lines)
    return (
        f"股票：{stock_name}\n"
        f"日期：{day_str}\n"
        f"以下是当日用户 comments（已按点击量/评论数从高到低排序，原始{len(comments)}条，提供{len(lines)}条）：\n"
        f"{comments_text}\n\n"
        "任务要求：\n"
        "1) 按观点聚类输出 2-3 个主要观点簇。\n"
        "2) 每个聚类都要给出 summary、sentiment_strength(0-1，表示情绪强度)、consensus_degree(0-1，表示共识强度，即这些观点的相似程度)。\n"
        "3) 另外输出全局情绪概率：positive_ratio（正面情绪概率） / neutral_ratio（中性情绪概率） / negative_ratio（负面情绪概率）。\n\n"
        "请只输出 JSON，不要输出 markdown 或额外解释，格式必须严格如下：\n"
        "{\n"
        "  \"clusters\": [\n"
        "    {\"summary\": \"...\", \"sentiment_strength\": 0.xx, \"consensus_degree\": 0.xx},\n"
        "    {\"summary\": \"...\", \"sentiment_strength\": 0.xx, \"consensus_degree\": 0.xx}\n"
        "  ],\n"
        "  \"positive_probability\": 0.xx,\n"
        "  \"neutral_probability\": 0.xx,\n"
        "  \"negative_probability\": 0.xx\n"
        "}\n\n"
        "约束：\n"
        "- 所有 probability/degree/strength 都在 [0,1] 区间。\n"
    )


def build_news_summary_prompt(stock_name: str, day_str: str, news_rows: list[tuple]) -> str:
    max_items = 200
    news_titles: list[str] = []
    for row in news_rows[:max_items]:
        _url, title, _d = row
        t = (title or "").strip()
        if t:
            news_titles.append(t)

    n_listed = len(news_titles)
    news_lines = [f"news_id={i} | {t}" for i, t in enumerate(news_titles, 1)]
    news_text = "\n".join(news_lines) if news_lines else "(无)"

    return (
        f"股票：{stock_name}\n"
        f"日期：{day_str}\n"
        f"以下为当日新闻标题列表（exports/news.csv 共{len(news_rows)}条，下列含标题共{n_listed}条）。"
        "每行前缀 news_id 为序号，不得编造表中不存在的 news_id。\n"
        f"{news_text}\n\n"
        "请用中文输出对当日新闻的**综合总结**（可用 markdown 小标题分段），至少包含：\n"
        "1) 主要信息主题与逻辑主线；\n"
        "2) 媒体叙述整体偏积极、中性或偏谨慎的大致倾向；\n"
        "4) 你认为与该股未来走向关联最强的新闻：按重要性列出至多十条，并标明对应 news_id；"
        "若当日新闻不足十条则全部列出；\n"
        "5) 对投资者的简短结论要点。\n"
        "勿编造正文中未出现的标题或事实；若某条仅有标题、信息不足请明确说明。\n"
    )


def stream_qwen_sse_user_prompt(user_prompt: str, status_text: str):
    client = _qwen_openai_client()
    if not _dashscope_api_key():
        yield sse_event(
            {"type": "error", "text": "未设置 DASHBOARD_API_KEY 或 DASHSCOPE_API_KEY（见项目根目录 .env）。"}
        )
        yield sse_event({"type": "done"})
        return
    if client is None:
        yield sse_event({"type": "error", "text": "未安装 openai，请执行：pip install openai"})
        yield sse_event({"type": "done"})
        return

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_prompt},
    ]

    yield sse_event({"type": "status", "text": status_text})

    try:
        stream = _chat_completions_stream(
            client, model=DASHSCOPE_MODEL, messages=messages
        )
        for chunk in stream:
            if chunk.choices:
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    yield sse_event({"type": "chunk", "text": _repair_mojibake(delta)})
    except Exception as exc:
        yield sse_event({"type": "error", "text": f"请求异常：{exc}"})

    yield sse_event({"type": "done"})


def stream_chat_summary(stock_name: str, day_str: str, comments: list[tuple]):
    if not comments:
        yield sse_event({"type": "error", "text": "该日没有 comments 数据可供总结。"})
        yield sse_event({"type": "done"})
        return

    user_prompt = build_summary_prompt(stock_name, day_str, comments)
    status_text = f"已收集评论 {len(comments)} 条，正在生成总结..."
    yield from stream_qwen_sse_user_prompt(user_prompt, status_text)


def stream_news_summary(stock_name: str, day_str: str, news_rows: list[tuple]):
    if not news_rows:
        yield sse_event({"type": "error", "text": "该日没有 news 数据可供总结。"})
        yield sse_event({"type": "done"})
        return
    if not any((row[1] or "").strip() for row in news_rows):
        yield sse_event({"type": "error", "text": "该日 news 无有效标题可供总结。"})
        yield sse_event({"type": "done"})
        return

    user_prompt = build_news_summary_prompt(stock_name, day_str, news_rows)
    status_text = f"已收集当日新闻 {len(news_rows)} 条，正在生成总结..."
    yield from stream_qwen_sse_user_prompt(user_prompt, status_text)


@app.get('/summarize_stream')
def summarize_stream() -> Response:
    selected_stock_id = request.args.get('stock', '')
    selected_day = request.args.get('day', date.today().isoformat())
    stock = STOCK_MAP.get(selected_stock_id)

    @stream_with_context
    def generate():
        if not stock:
            yield sse_event({"type": "error", "text": "股票参数无效。"})
            yield sse_event({"type": "done"})
            return
        data = query_data(stock, selected_day)
        yield from stream_chat_summary(stock['name'], selected_day, data['comments'])

    return Response(generate(), mimetype='text/event-stream')


@app.get('/summarize_news_stream')
def summarize_news_stream() -> Response:
    selected_stock_id = request.args.get('stock', '')
    selected_day = request.args.get('day', date.today().isoformat())
    stock = STOCK_MAP.get(selected_stock_id)

    @stream_with_context
    def generate():
        if not stock:
            yield sse_event({"type": "error", "text": "股票参数无效。"})
            yield sse_event({"type": "done"})
            return
        data = query_data(stock, selected_day)
        yield from stream_news_summary(stock['name'], selected_day, data['news'])

    return Response(generate(), mimetype='text/event-stream')


@app.get('/summarize_reports_stream')
def summarize_reports_stream() -> Response:
    selected_stock_id = request.args.get('stock', '')
    selected_day = request.args.get('day', date.today().isoformat())
    stock = STOCK_MAP.get(selected_stock_id)

    @stream_with_context
    def generate():
        if not stock:
            yield sse_event({"type": "error", "text": "股票参数无效。"})
            yield sse_event({"type": "done"})
            return
        data = query_data(stock, selected_day)
        yield from stream_recent_reports_summary(
            stock['name'], selected_day, data['reports']
        )

    return Response(generate(), mimetype='text/event-stream')


def _load_recent_kline(stock: dict, day_str: str, n: int = 7) -> list[dict]:
    """Load last n trading rows from exports/quotes.csv up to day_str (chronological order)."""
    try:
        day = pd.to_datetime(day_str).date()
    except Exception:
        return []

    symbol = stock["symbol"]
    raw_rows: list[tuple] = []
    if not QUOTES_CSV.is_file():
        return []

    import csv as _csv

    with QUOTES_CSV.open(encoding="utf-8", newline="") as f:
        for row in _csv.DictReader(f):
            if (row.get("symbol") or "").strip() != symbol:
                continue
            td_s = (row.get("trade_date") or "").strip()[:10]
            if not td_s:
                continue
            try:
                td = date.fromisoformat(td_s)
            except ValueError:
                continue
            if td > day:
                continue
            raw_rows.append(
                (
                    td,
                    row.get("open"),
                    row.get("high"),
                    row.get("low"),
                    row.get("close"),
                    row.get("pct_change"),
                    row.get("volume"),
                )
            )
    raw_rows.sort(key=lambda x: x[0], reverse=True)
    raw_rows = list(reversed(raw_rows[-n:]))

    out: list[dict] = []
    for row in raw_rows:
        td, o, h, l, c, pct, vol = row
        out.append(
            {
                "date": str(td),
                "open": float(o) if o not in (None, "") else None,
                "high": float(h) if h not in (None, "") else None,
                "low": float(l) if l not in (None, "") else None,
                "close": float(c) if c not in (None, "") else None,
                "pct": float(pct) if pct not in (None, "") else None,
                "volume": int(float(vol)) if vol not in (None, "") else None,
            }
        )
    return out


def _predict_with_llm(stock_name: str, day_str: str, summary_text: str, kline_rows: list[dict]) -> str:
    """Aggregate streaming completion into one string (same API as SSE paths, stream=True)."""
    client = _qwen_openai_client()
    if not _dashscope_api_key():
        return "未设置 DASHBOARD_API_KEY 或 DASHSCOPE_API_KEY（见项目根目录 .env）。"
    if client is None:
        return "未安装 openai，请执行：pip install openai"

    kline_text = "\n".join(
        f"- {r['date']}: open={r.get('open')}, high={r.get('high')}, low={r.get('low')}, close={r.get('close')}, pct={r.get('pct')}, volume={r.get('volume')}"
        for r in kline_rows
    )

    user_content = (
        f"股票：{stock_name}\n"
        f"当前日期：{day_str}\n\n"
        f"当日舆情总结：\n{summary_text}\n\n"
        f"最近7个交易日K线数据（含日期，按时间顺序）：\n{kline_text}\n\n"
        "请基于上面信息给出未来走势预测，输出三段：\n"
        "1) 1天走势预测\n"
        "2) 3天走势预测\n"
        "3) 7天走势预测\n"
        "每段包含：方向（上涨/下跌）、置信度（0-1）、一句主要依据。"
    )

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_content},
    ]

    parts: list[str] = []
    try:
        stream = _chat_completions_stream(
            client, model=DASHSCOPE_MODEL, messages=messages
        )
        for chunk in stream:
            if chunk.choices:
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    parts.append(_repair_mojibake(delta))
    except Exception as exc:
        return f"请求异常：{exc}"

    text = "".join(parts)
    return text if text.strip() else "(模型返回空内容)"


@app.post('/predict')
def predict():
    payload = request.get_json(silent=True) or {}
    selected_stock_id = payload.get('stock', '')
    selected_day = payload.get('day', date.today().isoformat())
    summary_text = (payload.get('summary') or '').strip()

    stock = STOCK_MAP.get(selected_stock_id)
    if not stock:
        return jsonify({"ok": False, "error": "股票参数无效"}), 400
    if not summary_text:
        return jsonify({"ok": False, "error": "请先生成当日观点总结"}), 400

    kline_rows = _load_recent_kline(stock, selected_day, n=7)
    if not kline_rows:
        return jsonify({"ok": False, "error": "未找到最近7天K线数据"}), 400

    result = _predict_with_llm(stock["name"], selected_day, summary_text, kline_rows)
    return jsonify({"ok": True, "prediction": result, "kline": kline_rows})


TEMPLATE = """
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <title>Stock Daily Data Viewer</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 24px; }
    h1 { margin-bottom: 8px; }
    .muted { color: #666; }
    form { display: flex; gap: 12px; align-items: end; margin: 16px 0 16px; }
    label { display: flex; flex-direction: column; font-size: 14px; gap: 6px; }
    select, input, button { padding: 8px 10px; font-size: 14px; }
    .grid { display: grid; grid-template-columns: 1fr; gap: 20px; }
    .card { border: 1px solid #ddd; border-radius: 8px; padding: 14px; }
    .card h2 { margin: 0 0 8px; font-size: 18px; }
    .count { color: #444; font-size: 13px; margin-bottom: 8px; }
    ul { margin: 0; padding-left: 20px; }
    li { margin: 6px 0; }
    a { color: #0b65c2; text-decoration: none; }
    a:hover { text-decoration: underline; }
    .small { color: #666; font-size: 12px; }
    .summary { white-space: pre-wrap; line-height: 1.6; min-height: 80px; }
    .actions { margin-bottom: 16px; }
  </style>
</head>
<body>
  <h1>股票单日数据查看</h1>
  <div class="muted">选择股票和日期：comments / news 为当日数据；reports 为截至所选日期的最近 3 条。</div>

  <form method="get">
    <label>
      股票
      <select name="stock" required>
        {% for s in stocks %}
          <option value="{{ s.id }}" {% if s.id == selected_stock_id %}selected{% endif %}>
            {{ s.sector }} / {{ s.name }} ({{ s.code_text }})
          </option>
        {% endfor %}
      </select>
    </label>

    <label>
      日期
      <input type="date" name="day" value="{{ selected_day }}" required />
    </label>

    <button type="submit">查询</button>
  </form>

  {% if stock %}
  <div class="small">当前：{{ stock.name }} ({{ stock.code_text }})，{{ selected_day }}</div>

  <div class="actions" style="display:flex; gap:10px; align-items:center; flex-wrap:wrap;">
    <button id="btn-summary" type="button">总结当日 comments 观点（Qwen）</button>
    <button id="btn-predict" type="button">基于总结+近7天K线预测（1/3/7天）</button>
    <button id="btn-news-summary" type="button">总结当日 news（Qwen）</button>
    <button id="btn-reports-summary" type="button">总结最近3条研报正文（Qwen）</button>
  </div>

  <div class="card" style="margin-bottom: 20px;">
    <h2>当日观点总结</h2>
    <div id="summary-status" class="small"></div>
    <div id="summary-text" class="summary"></div>
  </div>

  <div class="card" style="margin-bottom: 20px;">
    <h2>未来走势预测</h2>
    <div id="predict-status" class="small"></div>
    <div id="predict-text" class="summary"></div>
  </div>

  <div class="card" style="margin-bottom: 20px;">
    <h2>当日 news 总结</h2>
    <div id="news-summary-status" class="small"></div>
    <div id="news-summary-text" class="summary"></div>
  </div>

  <div class="card" style="margin-bottom: 20px;">
    <h2>最近3条研报总结（正文）</h2>
    <div id="reports-summary-status" class="small"></div>
    <div class="reports-summary-hint small" style="margin-bottom:6px;">正文缓存在项目 Content/report/（按 URL 哈希 .txt），有则直接读。</div>
    <div id="reports-summary-text" class="summary"></div>
  </div>
  {% endif %}

  <div class="grid">
    <div class="card">
      <h2>Comments</h2>
      <div class="count">{{ data.comments|length }} 条</div>
      <ul>
        {% for url, title, ts, clicks, ccount in data.comments %}
          <li>
            <a href="{{ url }}" target="_blank">{{ title }}</a>
            <div class="small">{{ ts }} | clicks={{ clicks }} | comments={{ ccount }}</div>
          </li>
        {% endfor %}
      </ul>
    </div>

    <div class="card">
      <h2>News</h2>
      <div class="count">{{ data.news|length }} 条</div>
      <ul>
        {% for url, title, d in data.news %}
          <li>
            <a href="{{ url }}" target="_blank">{{ title }}</a>
            <div class="small">{{ d }}</div>
          </li>
        {% endfor %}
      </ul>
    </div>

    <div class="card">
      <h2>Reports（最近 3 条）</h2>
      <div class="count">截至 {{ selected_day }}，共 {{ data.reports|length }} 条</div>
      <ul>
        {% for url, title, d in data.reports %}
          <li>
            <a href="{{ url }}" target="_blank">{{ title }}</a>
            <div class="small">{{ d }}</div>
          </li>
        {% endfor %}
      </ul>
    </div>
  </div>

  {% if stock %}
  <script>
    const btn = document.getElementById('btn-summary');
    const btnPredict = document.getElementById('btn-predict');
    const btnNewsSummary = document.getElementById('btn-news-summary');
    const btnReportsSummary = document.getElementById('btn-reports-summary');
    const statusEl = document.getElementById('summary-status');
    const textEl = document.getElementById('summary-text');
    const predictStatusEl = document.getElementById('predict-status');
    const predictTextEl = document.getElementById('predict-text');
    const newsSummaryStatusEl = document.getElementById('news-summary-status');
    const newsSummaryTextEl = document.getElementById('news-summary-text');
    const reportsSummaryStatusEl = document.getElementById('reports-summary-status');
    const reportsSummaryTextEl = document.getElementById('reports-summary-text');

    if (statusEl) statusEl.textContent = '就绪';

    if (btn) btn.addEventListener('click', () => {
      if (!textEl || !statusEl) { return; }
      textEl.textContent = '';
      statusEl.textContent = '正在生成总结...';
      btn.disabled = true;

      const params = new URLSearchParams({
        stock: {{ selected_stock_id|tojson }},
        day: {{ selected_day|tojson }}
      });

      const es = new EventSource('/summarize_stream?' + params.toString());

      es.onmessage = (evt) => {
        try {
          const data = JSON.parse(evt.data);
          if (data.type === 'status') {
            statusEl.textContent = data.text || '';
          } else if (data.type === 'chunk') {
            textEl.textContent += (data.text || '');
          } else if (data.type === 'error') {
            statusEl.textContent = data.text || '出错了';
          } else if (data.type === 'done') {
            if (!statusEl.textContent.startsWith('调用失败') && !statusEl.textContent.includes('异常')) {
              statusEl.textContent = '完成';
            }
            btn.disabled = false;
            es.close();
          }
        } catch (_err) {
          statusEl.textContent = '解析流式消息失败';
          btn.disabled = false;
          es.close();
        }
      };

      es.onerror = () => {
        statusEl.textContent = '流式连接中断';
        btn.disabled = false;
        es.close();
      };
    });

    if (btnPredict) btnPredict.addEventListener('click', async () => {
      const summary = (textEl.textContent || '').trim();
      if (!summary) {
        predictStatusEl.textContent = '请先生成当日观点总结';
        return;
      }

      btnPredict.disabled = true;
      predictStatusEl.textContent = '正在调用模型进行1/3/7天预测...';
      predictTextEl.textContent = '';

      try {
        const resp = await fetch('/predict', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({
            stock: {{ selected_stock_id|tojson }},
            day: {{ selected_day|tojson }},
            summary,
          }),
        });
        const data = await resp.json();
        if (!resp.ok || !data.ok) {
          predictStatusEl.textContent = data.error || '预测失败';
        } else {
          predictStatusEl.textContent = '完成';
          predictTextEl.textContent = data.prediction || '';
        }
      } catch (e) {
        predictStatusEl.textContent = '请求异常: ' + e;
      } finally {
        btnPredict.disabled = false;
      }
    });
    if (newsSummaryStatusEl) newsSummaryStatusEl.textContent = '就绪';

    if (btnNewsSummary) btnNewsSummary.addEventListener('click', () => {
      if (!newsSummaryTextEl || !newsSummaryStatusEl) { return; }
      newsSummaryTextEl.textContent = '';
      newsSummaryStatusEl.textContent = '正在生成总结...';
      btnNewsSummary.disabled = true;

      const params = new URLSearchParams({
        stock: {{ selected_stock_id|tojson }},
        day: {{ selected_day|tojson }}
      });

      const es = new EventSource('/summarize_news_stream?' + params.toString());

      es.onmessage = (evt) => {
        try {
          const data = JSON.parse(evt.data);
          if (data.type === 'status') {
            newsSummaryStatusEl.textContent = data.text || '';
          } else if (data.type === 'chunk') {
            newsSummaryTextEl.textContent += (data.text || '');
          } else if (data.type === 'error') {
            newsSummaryStatusEl.textContent = data.text || '出错了';
          } else if (data.type === 'done') {
            if (!newsSummaryStatusEl.textContent.startsWith('调用失败')
                && !newsSummaryStatusEl.textContent.includes('异常')) {
              newsSummaryStatusEl.textContent = '完成';
            }
            btnNewsSummary.disabled = false;
            es.close();
          }
        } catch (_err) {
          newsSummaryStatusEl.textContent = '解析流式消息失败';
          btnNewsSummary.disabled = false;
          es.close();
        }
      };

      es.onerror = () => {
        newsSummaryStatusEl.textContent = '流式连接中断';
        btnNewsSummary.disabled = false;
        es.close();
      };
    });

    if (reportsSummaryStatusEl) reportsSummaryStatusEl.textContent = '就绪';

    if (btnReportsSummary) btnReportsSummary.addEventListener('click', () => {
      if (!reportsSummaryTextEl || !reportsSummaryStatusEl) { return; }
      reportsSummaryTextEl.textContent = '';
      reportsSummaryStatusEl.textContent = '正在准备...';
      btnReportsSummary.disabled = true;

      const params = new URLSearchParams({
        stock: {{ selected_stock_id|tojson }},
        day: {{ selected_day|tojson }}
      });

      const es = new EventSource('/summarize_reports_stream?' + params.toString());

      es.onmessage = (evt) => {
        try {
          const data = JSON.parse(evt.data);
          if (data.type === 'status') {
            reportsSummaryStatusEl.textContent = data.text || '';
          } else if (data.type === 'chunk') {
            reportsSummaryTextEl.textContent += (data.text || '');
          } else if (data.type === 'error') {
            reportsSummaryStatusEl.textContent = data.text || '出错了';
          } else if (data.type === 'done') {
            if (!reportsSummaryStatusEl.textContent.startsWith('调用失败')
                && !reportsSummaryStatusEl.textContent.includes('异常')) {
              reportsSummaryStatusEl.textContent = '完成';
            }
            btnReportsSummary.disabled = false;
            es.close();
          }
        } catch (_err) {
          reportsSummaryStatusEl.textContent = '解析流式消息失败';
          btnReportsSummary.disabled = false;
          es.close();
        }
      };

      es.onerror = () => {
        reportsSummaryStatusEl.textContent = '流式连接中断';
        btnReportsSummary.disabled = false;
        es.close();
      };
    });

  </script>
  {% endif %}
</body>
</html>
"""


@app.get('/')
def index():
    selected_stock_id = request.args.get('stock') or (STOCKS[0]['id'] if STOCKS else '')
    selected_day = request.args.get('day') or date.today().isoformat()

    stock = STOCK_MAP.get(selected_stock_id)
    if not stock and STOCKS:
        stock = STOCKS[0]
        selected_stock_id = stock['id']

    data = {'comments': [], 'news': [], 'reports': []}
    if stock:
        data = query_data(stock, selected_day)

    return render_template_string(
        TEMPLATE,
        stocks=STOCKS,
        stock=stock,
        selected_stock_id=selected_stock_id,
        selected_day=selected_day,
        data=data,
    )


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5050, debug=False)
