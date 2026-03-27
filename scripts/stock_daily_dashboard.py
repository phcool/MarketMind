from __future__ import annotations

import json
import os
import re
from datetime import date
from pathlib import Path

import psycopg2
import requests
import pandas as pd
from flask import Flask, Response, jsonify, render_template_string, request, stream_with_context

ROOT_DIR = Path(__file__).resolve().parent.parent
BASE_DIR = ROOT_DIR / "A股重点板块"
DSN = "dbname=financial_data"

# OpenAI-compatible API settings
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.chatanywhere.tech/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5")

app = Flask(__name__)


def discover_stocks() -> list[dict]:
    """Discover stocks from folder names like: 恒瑞医药(600276), 吉利汽车(00175.HK)."""
    stocks: list[dict] = []
    pattern = re.compile(r"^(?P<name>.+)\((?P<code>[^)]+)\)$")

    for sector_dir in sorted(BASE_DIR.iterdir()):
        if not sector_dir.is_dir():
            continue
        for stock_dir in sorted(sector_dir.iterdir()):
            if not stock_dir.is_dir():
                continue
            m = pattern.match(stock_dir.name)
            if not m:
                continue
            name = m.group("name")
            code_text = m.group("code")
            code_digits = code_text.split(".")[0]
            if not (len(code_digits) == 6 and code_digits.isdigit()):
                continue
            stocks.append(
                {
                    "id": f"{name}|{code_text}",
                    "name": name,
                    "code_text": code_text,
                    "symbol": code_digits,
                    "sector": sector_dir.name,
                }
            )

    stocks.sort(key=lambda x: (x["sector"], x["code_text"]))
    return stocks


STOCKS = discover_stocks()
STOCK_MAP = {s["id"]: s for s in STOCKS}


def query_data(stock: dict, day_str: str) -> dict:
    conn = psycopg2.connect(DSN)
    cur = conn.cursor()

    symbol = stock["symbol"]

    cur.execute(
        """
        SELECT url, post_title, publish_time, click_count, comment_count
        FROM comments
        WHERE symbol = %s
          AND publish_time::date = %s::date
        ORDER BY publish_time DESC NULLS LAST
        LIMIT 500
        """,
        (symbol, day_str),
    )
    comments = cur.fetchall()

    cur.execute(
        """
        SELECT url, title, date
        FROM news
        WHERE symbol = %s
          AND date = %s::date
        ORDER BY title
        LIMIT 500
        """,
        (symbol, day_str),
    )
    news_rows = cur.fetchall()

    cur.execute(
        """
        SELECT url, title, date
        FROM report
        WHERE symbol = %s
          AND date = %s::date
        ORDER BY title
        LIMIT 500
        """,
        (symbol, day_str),
    )
    report_rows = cur.fetchall()

    cur.close()
    conn.close()

    return {"comments": comments, "news": news_rows, "reports": report_rows}


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

    news_lines = [f"{i}. {t}" for i, t in enumerate(news_titles, 1)]
    news_text = "\n".join(news_lines) if news_lines else "(无)"

    return (
        f"股票：{stock_name}\n"
        f"日期：{day_str}\n"
        f"以下为当日新闻标题（数据库共{len(news_rows)}条，列出{len(news_lines)}条）：\n"
        f"{news_text}\n\n"
        "任务要求：\n"
        "1) 按主题聚类输出 2-3 个主要信息簇。\n"
        "2) 每个聚类都要给出 summary、sentiment_strength(0-1，表示叙述在这个主题上的的强度)、"
        "consensus_degree(0-1，表示不同来源在该主题上说法的一致程度)。\n"
        "3) 另外输出全局情绪概率：positive_probability（偏多/利好观感概率） / neutral_probability（中性概率） / "
        "negative_probability（偏空/利空观感概率）。\n\n"
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


def stream_openai_sse_user_prompt(user_prompt: str, status_text: str):
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("CHATGPT_API_KEY")
    if not api_key:
        yield sse_event({"type": "error", "text": "未设置 OPENAI_API_KEY，无法调用模型。"})
        yield sse_event({"type": "done"})
        return

    base = OPENAI_BASE_URL.rstrip("/")
    if not base.endswith("/v1"):
        base = base + "/v1"
    url = base + "/chat/completions"

    payload = {
        "model": OPENAI_MODEL,
        "stream": True,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_prompt},
        ],
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    yield sse_event({"type": "status", "text": status_text})

    try:
        with requests.post(url, headers=headers, json=payload, stream=True, timeout=120) as resp:
            if resp.status_code != 200:
                txt = resp.text[:300]
                yield sse_event({"type": "error", "text": f"调用失败 HTTP={resp.status_code}: {txt}"})
                yield sse_event({"type": "done"})
                return

            resp.encoding = "utf-8"
            for raw in resp.iter_lines(decode_unicode=False):
                if not raw:
                    continue
                try:
                    line = raw.decode("utf-8").strip()
                except UnicodeDecodeError:
                    line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                    delta = obj["choices"][0].get("delta", {}).get("content", "")
                except Exception:
                    delta = ""
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
    yield from stream_openai_sse_user_prompt(user_prompt, status_text)


def stream_news_summary(stock_name: str, day_str: str, news_rows: list[tuple]):
    if not news_rows:
        yield sse_event({"type": "error", "text": "该日没有 news 数据可供总结。"})
        yield sse_event({"type": "done"})
        return

    user_prompt = build_news_summary_prompt(stock_name, day_str, news_rows)
    status_text = f"已收集当日新闻 {len(news_rows)} 条，正在生成总结..."
    yield from stream_openai_sse_user_prompt(user_prompt, status_text)


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


def _load_recent_kline(stock: dict, day_str: str, n: int = 7) -> list[dict]:
    """Load last n trading rows from local quotes files up to day_str."""
    try:
        day = pd.to_datetime(day_str).date()
    except Exception:
        return []

    company_dir = BASE_DIR / stock["sector"] / f"{stock['name']}({stock['code_text']})"
    quotes_dir = company_dir / "quotes"
    if not quotes_dir.exists():
        return []

    frames = []
    for txt in quotes_dir.rglob("*.txt"):
        try:
            df = pd.read_csv(txt, sep=r"\s+", engine="python")
            if df.empty:
                continue
            date_col = "日期" if "日期" in df.columns else df.columns[0]
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df = df[df[date_col].notna()]
            if df.empty:
                continue
            df = df[df[date_col].dt.date <= day]
            if df.empty:
                continue
            df["_date"] = df[date_col].dt.date
            frames.append(df)
        except Exception:
            continue

    if not frames:
        return []

    all_df = pd.concat(frames, ignore_index=True)
    all_df = all_df.sort_values("_date").drop_duplicates(subset=["_date"], keep="last")
    tail = all_df.tail(n)

    out = []
    for _, r in tail.iterrows():
        out.append(
            {
                "date": str(r.get("_date")),
                "open": r.get("开盘"),
                "high": r.get("最高"),
                "low": r.get("最低"),
                "close": r.get("收盘"),
                "pct": r.get("涨跌幅"),
                "volume": r.get("成交量"),
            }
        )
    return out


def _predict_with_llm(stock_name: str, day_str: str, summary_text: str, kline_rows: list[dict]) -> str:
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("CHATGPT_API_KEY")
    if not api_key:
        return "未设置 OPENAI_API_KEY，无法调用模型。"

    base = OPENAI_BASE_URL.rstrip("/")
    if not base.endswith("/v1"):
        base = base + "/v1"
    url = base + "/chat/completions"

    kline_text = "\n".join(
        f"- {r['date']}: open={r.get('open')}, high={r.get('high')}, low={r.get('low')}, close={r.get('close')}, pct={r.get('pct')}, volume={r.get('volume')}"
        for r in kline_rows
    )

    prompt = (
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

    payload = {
        "model": OPENAI_MODEL,
        "stream": False,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=120)
        if resp.status_code != 200:
            return f"调用失败 HTTP={resp.status_code}: {resp.text[:300]}"
        obj = resp.json()
        return obj["choices"][0]["message"]["content"]
    except Exception as exc:
        return f"请求异常：{exc}"


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
  <div class="muted">选择股票和日期，查看数据库中的 comments / news / reports。</div>

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
    <button id="btn-summary" type="button">总结当日 comments 观点（ChatGPT）</button>
    <button id="btn-predict" type="button">基于总结+近7天K线预测（1/3/7天）</button>
    <button id="btn-news-summary" type="button">总结当日 news（ChatGPT）</button>
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
      <h2>Reports</h2>
      <div class="count">{{ data.reports|length }} 条</div>
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
    const statusEl = document.getElementById('summary-status');
    const textEl = document.getElementById('summary-text');
    const predictStatusEl = document.getElementById('predict-status');
    const predictTextEl = document.getElementById('predict-text');
    const newsSummaryStatusEl = document.getElementById('news-summary-status');
    const newsSummaryTextEl = document.getElementById('news-summary-text');

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
