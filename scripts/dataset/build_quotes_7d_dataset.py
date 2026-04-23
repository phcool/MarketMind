"""
Build a single training CSV from exports/quotes.csv.

Loads all rows with trade_date < --train-before (default 2026-01-01) so that the
7-day lookback before the first label in [--date-start, train_before) is available.

Only **emits** samples whose label day (Day8) satisfies:
  --date-start <= trade_date < --train-before  (default: 2021-01-01 .. 2026-01-01).

Per symbol, normalization (min-max, z-score, etc.) is fit **only** on rows in that
same [date-start, train_before) window.

Each row: 7-day normalized K-line prompt -> label is the next day's direction:
  「涨」 if that day's pct_change >= 0, else 「跌」.

Outputs: UTF-8 CSV, columns prompt, label.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from fetch.csv_io import QUOTES_CSV

DEFAULT_OUT = ROOT_DIR / "dataset" / "quotes_7d_pre2026_dataset.csv"

HEADER = """过去7个交易日的K线数据如下(归一化后)：
"""

TAIL = """请根据上文 Day1–Day7 的归一化 K 线，只预测下一个交易日（Day8）相对前一日是涨还是跌。

【输出要求（必须严格遵守）】
只输出一个字：「涨」或「跌」，不要输出其他任何文字、数字、标点或换行。"""


def _to_float(x: Any) -> float | None:
    if x is None:
        return None
    return float(x)


def _fmt_norm(x: float | None, *, nd: int = 4) -> str:
    if x is None:
        return "N/A"
    return f"{x:.{nd}f}"


def _fmt_z(x: float | None, *, nd: int = 4) -> str:
    if x is None:
        return "N/A"
    return f"{x:+.{nd}f}"


def _fmt_amp(x: Any) -> str:
    if x is None:
        return "N/A"
    d = Decimal(str(x)) if not isinstance(x, Decimal) else x
    s = format(d, "f")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s or "0"


def _min_max_bounds(vals: list[float]) -> tuple[float, float]:
    if not vals:
        return 0.0, 1.0
    lo, hi = min(vals), max(vals)
    if hi <= lo:
        hi = lo + 1e-9
    return lo, hi


def pct_change_to_label(pc: Any) -> str | None:
    """「涨」 if pct_change >= 0, else 「跌」; None if missing."""
    v = _to_float(pc)
    if v is None:
        return None
    return "涨" if v >= 0 else "跌"


@dataclass
class StockNormalizer:
    o_lo: float
    o_hi: float
    h_lo: float
    h_hi: float
    l_lo: float
    l_hi: float
    c_lo: float
    c_hi: float
    vol_log_lo: float
    vol_log_hi: float
    pc_mean: float
    pc_std: float
    to_lo: float
    to_hi: float

    @classmethod
    def from_series(cls, series: list[dict]) -> StockNormalizer:
        opens: list[float] = []
        highs: list[float] = []
        lows: list[float] = []
        closes: list[float] = []
        log_vols: list[float] = []
        pcs: list[float] = []
        turns: list[float] = []

        for r in series:
            o = _to_float(r.get("open"))
            hi = _to_float(r.get("high"))
            lo = _to_float(r.get("low"))
            c = _to_float(r.get("close"))
            if o is not None:
                opens.append(o)
            if hi is not None:
                highs.append(hi)
            if lo is not None:
                lows.append(lo)
            if c is not None:
                closes.append(c)

            v = r.get("volume")
            if v is not None:
                log_vols.append(math.log1p(max(0.0, float(v))))

            pc = _to_float(r.get("pct_change"))
            if pc is not None:
                pcs.append(pc)

            t = _to_float(r.get("turnover"))
            if t is not None:
                turns.append(t)

        o_lo, o_hi = _min_max_bounds(opens)
        h_lo, h_hi = _min_max_bounds(highs)
        l_lo, l_hi = _min_max_bounds(lows)
        c_lo, c_hi = _min_max_bounds(closes)

        vol_log_lo = min(log_vols) if log_vols else 0.0
        vol_log_hi = max(log_vols) if log_vols else 1.0
        if vol_log_hi <= vol_log_lo:
            vol_log_hi = vol_log_lo + 1e-9

        pc_mean = sum(pcs) / len(pcs) if pcs else 0.0
        if len(pcs) >= 2:
            var = sum((x - pc_mean) ** 2 for x in pcs) / len(pcs)
            pc_std = math.sqrt(var)
        else:
            pc_std = 0.0
        if pc_std < 1e-12:
            pc_std = 1.0

        to_lo = min(turns) if turns else 0.0
        to_hi = max(turns) if turns else 1.0
        if to_hi <= to_lo:
            to_hi = to_lo + 1e-9

        return cls(
            o_lo=o_lo,
            o_hi=o_hi,
            h_lo=h_lo,
            h_hi=h_hi,
            l_lo=l_lo,
            l_hi=l_hi,
            c_lo=c_lo,
            c_hi=c_hi,
            vol_log_lo=vol_log_lo,
            vol_log_hi=vol_log_hi,
            pc_mean=pc_mean,
            pc_std=pc_std,
            to_lo=to_lo,
            to_hi=to_hi,
        )

    def _norm_mm(self, x: Any, lo: float, hi: float) -> float | None:
        v = _to_float(x)
        if v is None:
            return None
        return (v - lo) / (hi - lo)

    def norm_open(self, x: Any) -> float | None:
        return self._norm_mm(x, self.o_lo, self.o_hi)

    def norm_high(self, x: Any) -> float | None:
        return self._norm_mm(x, self.h_lo, self.h_hi)

    def norm_low(self, x: Any) -> float | None:
        return self._norm_mm(x, self.l_lo, self.l_hi)

    def norm_close(self, x: Any) -> float | None:
        return self._norm_mm(x, self.c_lo, self.c_hi)

    def norm_volume(self, x: Any) -> float | None:
        if x is None:
            return None
        lv = math.log1p(max(0.0, float(x)))
        return (lv - self.vol_log_lo) / (self.vol_log_hi - self.vol_log_lo)

    def norm_pct_change(self, x: Any) -> float | None:
        v = _to_float(x)
        if v is None:
            return None
        return (v - self.pc_mean) / self.pc_std

    def norm_turnover(self, x: Any) -> float | None:
        v = _to_float(x)
        if v is None:
            return None
        return (v - self.to_lo) / (self.to_hi - self.to_lo)

    def row_to_day_line(self, day_idx: int, r: dict) -> str:
        o = self.norm_open(r.get("open"))
        h = self.norm_high(r.get("high"))
        l = self.norm_low(r.get("low"))
        c = self.norm_close(r.get("close"))
        vol = self.norm_volume(r.get("volume"))
        pcz = self.norm_pct_change(r.get("pct_change"))
        to = self.norm_turnover(r.get("turnover"))
        amp = _fmt_amp(r.get("amplitude"))
        return (
            f"Day{day_idx}: open={_fmt_norm(o)}, high={_fmt_norm(h)}, low={_fmt_norm(l)}, "
            f"close={_fmt_norm(c)}, volume={_fmt_norm(vol)}, amplitude={amp}, "
            f"pct_change={_fmt_z(pcz)}, turnover={_fmt_norm(to)}"
        )


def build_prompt(norm: StockNormalizer, prev7: list[dict]) -> str:
    lines = [
        HEADER.rstrip(),
        *[norm.row_to_day_line(i + 1, prev7[i]) for i in range(7)],
        "",
        TAIL,
    ]
    return "\n".join(lines)


def load_quotes_until(train_before: date) -> list[tuple[str, list[dict]]]:
    """Load quotes with trade_date < train_before (for lookback before date-start)."""
    if not QUOTES_CSV.is_file():
        return []
    by_symbol: dict[str, list[dict]] = {}
    with QUOTES_CSV.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sym = (row.get("symbol") or "").strip()
            td_s = (row.get("trade_date") or "").strip()[:10]
            if not sym or not td_s:
                continue
            try:
                td = date.fromisoformat(td_s)
            except ValueError:
                continue
            if td >= train_before:
                continue
            rec = {
                "trade_date": td,
                "open": row.get("open"),
                "high": row.get("high"),
                "low": row.get("low"),
                "close": row.get("close"),
                "volume": row.get("volume"),
                "amplitude": row.get("amplitude"),
                "pct_change": row.get("pct_change"),
                "turnover": row.get("turnover"),
            }
            by_symbol.setdefault(sym, []).append(rec)
    for sym in by_symbol:
        by_symbol[sym].sort(key=lambda x: x["trade_date"])
    return sorted(by_symbol.items(), key=lambda x: x[0])


def series_in_label_window(
    series: list[dict], date_start: date, train_before: date
) -> list[dict]:
    """Rows used to fit normalization: date_start <= trade_date < train_before."""
    return [r for r in series if date_start <= r["trade_date"] < train_before]


def _as_date(d: Any) -> date:
    if isinstance(d, date):
        return d
    return date.fromisoformat(str(d))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build 7-day K-line -> next-day 涨/跌 classification dataset (label days in [date-start, train_before)).",
    )
    ap.add_argument(
        "--date-start",
        default="2021-01-01",
        help="Emit samples only when Day8 trade_date >= this (YYYY-MM-DD). Default: 2021-01-01.",
    )
    ap.add_argument(
        "--train-before",
        default="2026-01-01",
        help="Upper bound: trade_date < this; also end of label window. Default: 2026-01-01.",
    )
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUT,
        help="Output CSV path (columns: prompt, label).",
    )
    args = ap.parse_args()

    date_start = _as_date(args.date_start)
    train_before = _as_date(args.train_before)
    if date_start >= train_before:
        raise SystemExit("--date-start must be < --train-before")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    groups = load_quotes_until(train_before)

    n_out = 0
    with args.output.open("w", encoding="utf-8", newline="") as f_out:
        w = csv.writer(f_out, quoting=csv.QUOTE_MINIMAL)
        w.writerow(["prompt", "label"])

        for _symbol, series in groups:
            if len(series) < 8:
                continue
            fit_rows = series_in_label_window(series, date_start, train_before)
            if not fit_rows:
                continue
            norm = StockNormalizer.from_series(fit_rows)
            for k in range(7, len(series)):
                target = series[k]
                td = target["trade_date"]
                if not (date_start <= td < train_before):
                    continue
                lab = pct_change_to_label(target.get("pct_change"))
                if lab is None:
                    continue
                prev7 = series[k - 7 : k]
                prompt = build_prompt(norm, prev7)
                w.writerow([prompt, lab])
                n_out += 1

    print(
        f"Wrote {n_out} rows to {args.output} "
        f"(label days in [{date_start.isoformat()}, {train_before.isoformat()}); "
        "涨 if pct_change>=0 else 跌)",
    )


if __name__ == "__main__":
    main()
