from __future__ import annotations

import sys
from pathlib import Path

from flask import Flask, jsonify, render_template, request

ROOT_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from stock_universe import load_sectors  # type: ignore  # noqa: E402
from src.live_fetch import fetch_stock_snapshot, normalize_symbol  # noqa: E402


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


if __name__ == "__main__":
    app.run(debug=True, port=5000)
