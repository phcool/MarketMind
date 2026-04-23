"""
Shared UTF-8 CSV paths and merge helpers under exports/ (dedupe by url or symbol+trade_date).
"""

from __future__ import annotations

import csv
import threading
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any

# Max comment rows kept per (symbol, calendar day) after merge (forum fetch + dedupe by url).
COMMENTS_MAX_PER_SYMBOL_DAY = 200

ROOT_DIR = Path(__file__).resolve().parents[2]
EXPORTS_DIR = ROOT_DIR / "exports"

QUOTES_CSV = EXPORTS_DIR / "quotes.csv"
COMMENTS_CSV = EXPORTS_DIR / "comments.csv"
NEWS_CSV = EXPORTS_DIR / "news.csv"
REPORT_CSV = EXPORTS_DIR / "report.csv"

QUOTE_FIELDNAMES = [
    "symbol",
    "trade_date",
    "open",
    "close",
    "high",
    "low",
    "volume",
    "amount",
    "amplitude",
    "pct_change",
    "change_amount",
    "turnover",
]

NEWS_FIELDNAMES = ["url", "symbol", "title", "date"]
REPORT_FIELDNAMES = ["url", "symbol", "title", "date", "content"]
COMMENT_FIELDNAMES = [
    "url",
    "post_id",
    "symbol",
    "post_title",
    "publish_time",
    "click_count",
    "comment_count",
]

_comments_lock = threading.Lock()
_news_lock = threading.Lock()
_report_lock = threading.Lock()
_quotes_lock = threading.Lock()


def _ensure_dir() -> None:
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)


def _read_csv_dicts(path: Path, fieldnames: list[str]) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    out: list[dict[str, str]] = []
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rec = {k: (row.get(k) or "").strip() for k in fieldnames}
            out.append(rec)
    return out


def _write_csv_dicts(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    _ensure_dir()
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})
    tmp.replace(path)


def merge_append_news(
    sym: str,
    entries: list[dict],
) -> int:
    """Append news rows (url dedup). entries: date iso str in 'date' key."""
    new_rows: list[tuple[str, str, str, str]] = []
    for e in entries:
        url = (e.get("url") or "").strip()
        if not url:
            continue
        title = (e.get("title") or "").strip()
        d = e.get("date")
        if not d:
            continue
        try:
            day = datetime.fromisoformat(str(d)).date()
        except (ValueError, TypeError):
            continue
        new_rows.append((url, sym, title, day.isoformat()))
    if not new_rows:
        return 0
    with _news_lock:
        _ensure_dir()
        existing = _read_csv_dicts(NEWS_CSV, NEWS_FIELDNAMES)
        have = {r["url"] for r in existing if r.get("url")}
        added = 0
        for url, s, title, ds in new_rows:
            if url in have:
                continue
            have.add(url)
            existing.append({"url": url, "symbol": s, "title": title, "date": ds})
            added += 1
        if added:
            _write_csv_dicts(NEWS_CSV, NEWS_FIELDNAMES, existing)
        return added


def merge_append_report(sym: str, entries: list[dict]) -> int:
    """Append report list rows; preserve existing content for same url."""
    new_tuples: list[tuple[str, str, str, str]] = []
    for e in entries:
        url = (e.get("url") or "").strip()
        if not url:
            continue
        title = (e.get("title") or "").strip()
        d = e.get("date")
        if not d:
            continue
        try:
            day = datetime.fromisoformat(str(d)).date()
        except (ValueError, TypeError):
            continue
        new_tuples.append((url, sym, title, day.isoformat()))
    if not new_tuples:
        return 0
    with _report_lock:
        _ensure_dir()
        by_url: dict[str, dict[str, str]] = {}
        if REPORT_CSV.is_file():
            for r in _read_csv_dicts(REPORT_CSV, REPORT_FIELDNAMES):
                u = r.get("url", "")
                if u:
                    by_url[u] = dict(r)
        added = 0
        for url, s, title, ds in new_tuples:
            if url in by_url:
                continue
            by_url[url] = {
                "url": url,
                "symbol": s,
                "title": title,
                "date": ds,
                "content": "",
            }
            added += 1
        rows = sorted(by_url.values(), key=lambda x: (x.get("symbol", ""), x.get("date", ""), x.get("url", "")))
        if added:
            _write_csv_dicts(REPORT_CSV, REPORT_FIELDNAMES, rows)
        return added


def _comment_row_clicks(row: dict[str, str]) -> int:
    ck = (row.get("click_count") or "").strip()
    if not ck:
        return 0
    try:
        return int(float(ck))
    except ValueError:
        return 0


def _comment_row_calendar_day(row: dict[str, str]) -> str:
    pt = (row.get("publish_time") or "").strip().replace("T", " ")
    if len(pt) < 10:
        return ""
    ds = pt[:10]
    try:
        datetime.strptime(ds, "%Y-%m-%d")
        return ds
    except ValueError:
        return ""


def trim_comment_rows_per_symbol_day(
    rows: list[dict[str, str]],
    top_n: int = COMMENTS_MAX_PER_SYMBOL_DAY,
) -> list[dict[str, str]]:
    """Keep at most top_n rows per (symbol, calendar day) by click_count descending."""
    by_key: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        sym = (r.get("symbol") or "").strip()
        day = _comment_row_calendar_day(r)
        by_key[(sym, day)].append(r)
    out: list[dict[str, str]] = []
    for key in sorted(by_key.keys()):
        bucket = by_key[key]
        bucket.sort(
            key=lambda x: (
                -_comment_row_clicks(x),
                x.get("url") or "",
                x.get("post_id") or "",
            )
        )
        out.extend(bucket[:top_n])
    return out


def merge_append_comments(rows: list[tuple]) -> int:
    """
    rows: (url, post_id, symbol, post_title, publish_time_datetime, click_count, comment_count)

    After merging new urls, the whole table is trimmed to COMMENTS_MAX_PER_SYMBOL_DAY rows
    per (symbol, calendar day) by click_count (same rule as one-off CSV cleanup).
    """
    if not rows:
        return 0
    with _comments_lock:
        _ensure_dir()
        by_url: dict[str, dict[str, str]] = {}
        if COMMENTS_CSV.is_file():
            for r in _read_csv_dicts(COMMENTS_CSV, COMMENT_FIELDNAMES):
                u = r.get("url", "")
                if u:
                    by_url[u] = dict(r)
        added = 0
        for tup in rows:
            url, post_id, sym, title, pub, clicks, ccount = tup
            url = (url or "").strip()
            if not url or url in by_url:
                continue
            if isinstance(pub, datetime):
                pts = pub.strftime("%Y-%m-%d %H:%M:%S")
            else:
                pts = str(pub) if pub is not None else ""
            by_url[url] = {
                "url": url,
                "post_id": str(post_id) if post_id is not None else "",
                "symbol": str(sym),
                "post_title": title or "",
                "publish_time": pts,
                "click_count": str(int(clicks)) if clicks is not None and str(clicks).strip() != "" else "",
                "comment_count": str(int(ccount)) if ccount is not None and str(ccount).strip() != "" else "",
            }
            added += 1
        trimmed = trim_comment_rows_per_symbol_day(
            list(by_url.values()),
            COMMENTS_MAX_PER_SYMBOL_DAY,
        )
        out = sorted(
            trimmed,
            key=lambda x: (x.get("symbol", ""), x.get("publish_time", ""), x.get("url", "")),
        )
        # Write when new urls merged OR global trim removed rows (fixes legacy oversized CSV).
        if added or len(out) < len(by_url):
            _write_csv_dicts(COMMENTS_CSV, COMMENT_FIELDNAMES, out)
        return added


def rewrite_comments_csv_trimmed() -> tuple[int, int]:
    """
    Re-read exports/comments.csv and keep top COMMENTS_MAX_PER_SYMBOL_DAY rows per
    (symbol, calendar day) by click_count. Rewrites file if row count changes.
    Returns (row_count_before, row_count_after).
    """
    with _comments_lock:
        if not COMMENTS_CSV.is_file():
            return 0, 0
        rows = _read_csv_dicts(COMMENTS_CSV, COMMENT_FIELDNAMES)
        n_before = len(rows)
        trimmed = trim_comment_rows_per_symbol_day(rows, COMMENTS_MAX_PER_SYMBOL_DAY)
        out = sorted(
            trimmed,
            key=lambda x: (x.get("symbol", ""), x.get("publish_time", ""), x.get("url", "")),
        )
        n_after = len(out)
        if n_after != n_before:
            _write_csv_dicts(COMMENTS_CSV, COMMENT_FIELDNAMES, out)
        return n_before, n_after


def _fmt_num(x: Any) -> str:
    if x is None:
        return ""
    try:
        v = float(x)
        if v == int(v) and abs(v) < 1e12:
            return str(int(v))
        return f"{v:.6f}".rstrip("0").rstrip(".")
    except (TypeError, ValueError):
        return str(x)


def upsert_quotes_rows(rows: list[tuple]) -> None:
    """
    rows: (symbol, trade_date, open, close, high, low, volume, amount,
           amplitude, pct_change, change_amount, turnover) — same as exports/quotes.csv rows.
    trade_date may be date or string YYYY-MM-DD.
    """
    if not rows:
        return
    with _quotes_lock:
        _ensure_dir()
        key_to_row: dict[tuple[str, str], dict[str, str]] = {}
        if QUOTES_CSV.is_file() and QUOTES_CSV.stat().st_size > 0:
            for r in _read_csv_dicts(QUOTES_CSV, QUOTE_FIELDNAMES):
                sym, td = r.get("symbol", ""), r.get("trade_date", "")
                if sym and td:
                    key_to_row[(sym, td[:10])] = dict(r)
        for tup in rows:
            sym, td, o, c, h, l_, vol, amt, amp, pct, chg, turn = tup
            sym = str(sym).strip()
            if isinstance(td, date):
                tds = td.isoformat()
            else:
                tds = str(td)[:10]
            key_to_row[(sym, tds)] = {
                "symbol": sym,
                "trade_date": tds,
                "open": _fmt_num(o),
                "close": _fmt_num(c),
                "high": _fmt_num(h),
                "low": _fmt_num(l_),
                "volume": _fmt_num(vol),
                "amount": _fmt_num(amt),
                "amplitude": _fmt_num(amp),
                "pct_change": _fmt_num(pct),
                "change_amount": _fmt_num(chg),
                "turnover": _fmt_num(turn),
            }
        ordered = sorted(key_to_row.values(), key=lambda r: (r["symbol"], r["trade_date"]))
        _write_csv_dicts(QUOTES_CSV, QUOTE_FIELDNAMES, ordered)


def ensure_quotes_csv_header() -> None:
    with _quotes_lock:
        _ensure_dir()
        if not QUOTES_CSV.is_file() or QUOTES_CSV.stat().st_size == 0:
            _write_csv_dicts(QUOTES_CSV, QUOTE_FIELDNAMES, [])


def update_report_content_by_url(updates: list[tuple[str, str]]) -> None:
    """Batch set content for url; rewrites report.csv."""
    if not updates:
        return
    umap = {u: c for u, c in updates}
    with _report_lock:
        rows = _read_csv_dicts(REPORT_CSV, REPORT_FIELDNAMES)
        for r in rows:
            u = r.get("url", "")
            if u in umap:
                r["content"] = umap[u]
        _write_csv_dicts(REPORT_CSV, REPORT_FIELDNAMES, rows)


def delete_news_rows_by_url(urls: list[str]) -> int:
    """Delete news.csv rows whose url is in urls. Returns deleted row count."""
    if not urls:
        return 0
    targets = {u.strip() for u in urls if u and u.strip()}
    if not targets:
        return 0
    with _news_lock:
        rows = _read_csv_dicts(NEWS_CSV, NEWS_FIELDNAMES)
        out = [r for r in rows if (r.get("url", "") not in targets)]
        deleted = len(rows) - len(out)
        if deleted:
            _write_csv_dicts(NEWS_CSV, NEWS_FIELDNAMES, out)
        return deleted


def delete_report_rows_by_url(urls: list[str]) -> int:
    """Delete report.csv rows whose url is in urls. Returns deleted row count."""
    if not urls:
        return 0
    targets = {u.strip() for u in urls if u and u.strip()}
    if not targets:
        return 0
    with _report_lock:
        rows = _read_csv_dicts(REPORT_CSV, REPORT_FIELDNAMES)
        out = [r for r in rows if (r.get("url", "") not in targets)]
        deleted = len(rows) - len(out)
        if deleted:
            _write_csv_dicts(REPORT_CSV, REPORT_FIELDNAMES, out)
        return deleted
