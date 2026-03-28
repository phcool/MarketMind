"""
Build train + validation CSVs from PostgreSQL `quotes`.

Loads all rows with trade_date <= data_end (default 2026-03-28). Per symbol,
normalization stats use the FULL loaded series (train + validation periods combined).

Training rows: label trade_date < train_before (default 2026-01-01).
Validation rows: label trade_date in [val_start, val_end] (default 2026-01-01 .. 2026-03-28).

Normalization (per stock, on merged history through data_end):
  open/high/low/close — independent Min-Max [0,1]; volume — log1p then Min-Max;
  pct_change — Z-score; turnover — Min-Max; amplitude — raw.

Outputs: UTF-8 CSV, columns prompt, pct_change.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

import psycopg2

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_TRAIN_OUT = ROOT_DIR / "train" / "dataset" / "quotes_7d_pre2026_dataset.csv"
DEFAULT_VAL_OUT = ROOT_DIR / "train" / "dataset" / "quotes_7d_val_20260101_20260328_dataset.csv"
DSN = os.environ.get("PG_DSN", "dbname=financial_data")

HEADER = """过去7个交易日的K线数据如下(归一化后)：
"""

TAIL = """请结合上文的 Day1–Day7，简要判断趋势、成交与多空力量，并预测下一个交易日（Day8）的收盘涨跌幅（pct_change，单位%）写到最后一行。

【输出格式（必须严格遵守）】
1. 你可以有文字思考过程，写在前面,但是在最后一行必须输出你对下一个交易日的收盘涨跌幅的预测(pct_change，单位%)。
2. 全文的最后一行必须是单独一行，写你对涨跌幅的预测，表示 Day8 的预测涨跌幅数值；

【格式示例】

<你的思考过程>
pct_change_prediction: 0.**%

现在请按上述格式输出结论：最后一行必须形如「pct_change_prediction: 数字%」（数字可带负号与小数点）。"""


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


def load_quotes(cur, data_end: str) -> list[tuple[str, list[dict]]]:
    cur.execute(
        """
        SELECT symbol, trade_date, open, high, low, close, volume,
               amplitude, pct_change, turnover
        FROM quotes
        WHERE trade_date <= %s::date
        ORDER BY symbol, trade_date
        """,
        (data_end,),
    )
    cols = [d[0] for d in cur.description]
    rows = cur.fetchall()
    by_symbol: dict[str, list[dict]] = {}
    for tup in rows:
        rec = dict(zip(cols, tup))
        sym = rec.pop("symbol")
        by_symbol.setdefault(sym, []).append(rec)
    return sorted(by_symbol.items(), key=lambda x: x[0])


def _as_date(d: Any) -> date:
    if isinstance(d, date):
        return d
    return date.fromisoformat(str(d))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build train + val 7-day K-line prompt datasets (joint per-stock normalization)."
    )
    ap.add_argument(
        "--data-end",
        default="2026-03-28",
        help="Load quotes with trade_date <= this (YYYY-MM-DD); also used to fit normalization.",
    )
    ap.add_argument(
        "--train-before",
        default="2026-01-01",
        help="Training samples: label trade_date < this date.",
    )
    ap.add_argument(
        "--val-start",
        default="2026-01-01",
        help="Validation samples: label trade_date >= this date.",
    )
    ap.add_argument(
        "--val-end",
        default="2026-03-28",
        help="Validation samples: label trade_date <= this date.",
    )
    ap.add_argument("-o", "--output", type=Path, default=DEFAULT_TRAIN_OUT, help="Training CSV path.")
    ap.add_argument("--val-output", type=Path, default=DEFAULT_VAL_OUT, help="Validation CSV path.")
    args = ap.parse_args()

    train_before = _as_date(args.train_before)
    val_start = _as_date(args.val_start)
    val_end = _as_date(args.val_end)
    if val_start > val_end:
        raise SystemExit("val-start must be <= val-end")
    if train_before > val_start:
        raise SystemExit("train-before must be <= val-start so train/val label dates do not overlap")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.val_output.parent.mkdir(parents=True, exist_ok=True)

    conn = psycopg2.connect(DSN)
    try:
        cur = conn.cursor()
        groups = load_quotes(cur, args.data_end)
        cur.close()
    finally:
        conn.close()

    n_train = 0
    n_val = 0
    with (
        args.output.open("w", encoding="utf-8", newline="") as f_train,
        args.val_output.open("w", encoding="utf-8", newline="") as f_val,
    ):
        w_train = csv.writer(f_train, quoting=csv.QUOTE_MINIMAL)
        w_val = csv.writer(f_val, quoting=csv.QUOTE_MINIMAL)
        w_train.writerow(["prompt", "pct_change"])
        w_val.writerow(["prompt", "pct_change"])

        for _symbol, series in groups:
            if len(series) < 8:
                continue
            norm = StockNormalizer.from_series(series)
            for k in range(7, len(series)):
                target = series[k]
                td = _as_date(target["trade_date"])
                pc = target.get("pct_change")
                if pc is None:
                    continue
                prev7 = series[k - 7 : k]
                prompt = build_prompt(norm, prev7)
                row = [prompt, float(pc)]
                if td < train_before:
                    w_train.writerow(row)
                    n_train += 1
                elif val_start <= td <= val_end:
                    w_val.writerow(row)
                    n_val += 1

    print(f"Wrote {n_train} train rows to {args.output}")
    print(f"Wrote {n_val} val rows to {args.val_output}")


if __name__ == "__main__":
    main()
