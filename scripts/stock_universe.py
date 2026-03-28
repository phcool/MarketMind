"""
Load the tracked stock universe (7 sectors, 35 names) from config/stocks.json.

All fetch scripts and the dashboard use this instead of scanning A股重点板块/.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
STOCKS_JSON = ROOT_DIR / "config" / "stocks.json"


def load_sectors() -> dict[str, list[tuple[str, str, str]]]:
    """
    Returns { sector_name: [(name, symbol, market), ...] }.
    market is "a" (A-share) or "hk".
    """
    if not STOCKS_JSON.is_file():
        raise FileNotFoundError(
            f"Stock universe file missing: {STOCKS_JSON}\n"
            "Create it or restore config/stocks.json from the repo."
        )
    raw = json.loads(STOCKS_JSON.read_text(encoding="utf-8"))
    blob = raw.get("sectors")
    if not isinstance(blob, dict):
        raise ValueError("stocks.json: top-level key 'sectors' must be an object")

    out: dict[str, list[tuple[str, str, str]]] = {}
    for sector_name, stocks in blob.items():
        if not isinstance(stocks, list):
            raise ValueError(f"stocks.json: sector {sector_name!r} must be a list")
        rows: list[tuple[str, str, str]] = []
        for item in stocks:
            if not isinstance(item, dict):
                raise ValueError(f"stocks.json: invalid entry under {sector_name!r}")
            name = str(item.get("name", "")).strip()
            symbol = str(item.get("symbol", "")).strip()
            market = str(item.get("market", "a")).strip().lower()
            if not name or not symbol:
                raise ValueError(f"stocks.json: name/symbol required in {sector_name!r}")
            if market not in ("a", "hk"):
                raise ValueError(f"stocks.json: market must be 'a' or 'hk', got {market!r}")
            rows.append((name, symbol, market))
        out[str(sector_name)] = rows
    return out
