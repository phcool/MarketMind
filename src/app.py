from __future__ import annotations

import sys
from pathlib import Path

from flask import Flask, jsonify, render_template, request

ROOT_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = ROOT_DIR / "scripts"
SRC_DIR = ROOT_DIR / "src"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from stock_universe import load_sectors  # type: ignore  # noqa: E402
from live_fetch import (  # noqa: E402
    fetch_serp_contents,
    predict_stock_direction,
    fetch_stock_snapshot,
    filter_serp_results,
    normalize_symbol,
    summarize_section,
)


app = Flask(
    __name__,
    template_folder=str(Path(__file__).resolve().parent / "templates"),
    static_folder=str(Path(__file__).resolve().parent / "static"),
)


def stock_options() -> list[dict[str, str]]:
    options: list[dict[str, str]] = []
    for sector, companies in load_sectors().items():
        for name, symbol, market in companies:
            options.append(
                {
                    "sector": sector,
                    "name": name,
                    "symbol": symbol,
                    "market": market,
                }
            )
    return options


@app.get("/")
def index():
    initial_symbol = normalize_symbol(request.args.get("symbol", ""))
    initial_data = None
    initial_error = ""
    if initial_symbol:
        try:
            initial_data = fetch_stock_snapshot(initial_symbol)
        except Exception as exc:  # noqa: BLE001
            initial_error = str(exc)
    return render_template(
        "index.html",
        stock_options=stock_options(),
        initial_symbol=initial_symbol,
        initial_data=initial_data,
        initial_error=initial_error,
    )


@app.get("/api/stock/<symbol>")
def stock_api(symbol: str):
    try:
        data = fetch_stock_snapshot(symbol)
        return jsonify({"ok": True, "data": data})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.post("/api/summary/<section>")
def summary_api(section: str):
    payload = request.get_json(silent=True) or {}
    try:
        stock = normalize_symbol(payload.get("stock", ""))
        company_name = str(payload.get("name", "")).strip()
        items = payload.get("items") or []
        if not stock or not company_name:
            raise ValueError("缺少股票代码或公司名称。")
        if not isinstance(items, list):
            raise ValueError("items 必须是列表。")
        summary = summarize_section(section, stock, company_name, items)
        return jsonify({"ok": True, "summary": summary})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.post("/api/serp-content")
def serp_content_api():
    payload = request.get_json(silent=True) or {}
    try:
        items = payload.get("items") or []
        if not isinstance(items, list):
            raise ValueError("items 必须是列表。")
        enriched = fetch_serp_contents(items)
        return jsonify({"ok": True, "items": enriched})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.post("/api/serp-filter")
def serp_filter_api():
    payload = request.get_json(silent=True) or {}
    try:
        stock = normalize_symbol(payload.get("stock", ""))
        company_name = str(payload.get("name", "")).strip()
        items = payload.get("items") or []
        if not stock or not company_name:
            raise ValueError("缺少股票代码或公司名称。")
        if not isinstance(items, list):
            raise ValueError("items 必须是列表。")
        filtered, note = filter_serp_results(stock, company_name, items)
        return jsonify({"ok": True, "items": filtered, "note": note})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.post("/api/predict")
def predict_api():
    payload = request.get_json(silent=True) or {}
    try:
        stock = normalize_symbol(payload.get("stock", ""))
        company_name = str(payload.get("name", "")).strip()
        kline = payload.get("kline") or []
        news_summary = str(payload.get("news_summary", "")).strip()
        reports_summary = str(payload.get("reports_summary", "")).strip()
        serp_summary = str(payload.get("serp_summary", "")).strip()
        if not stock or not company_name:
            raise ValueError("缺少股票代码或公司名称。")
        if not isinstance(kline, list):
            raise ValueError("kline 必须是列表。")
        prediction = predict_stock_direction(
            stock,
            company_name,
            kline,
            news_summary=news_summary,
            reports_summary=reports_summary,
            serp_summary=serp_summary,
        )
        return jsonify({"ok": True, "prediction": prediction})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"ok": False, "error": str(exc)}), 400


if __name__ == "__main__":
    app.run(debug=True, port=5000)
