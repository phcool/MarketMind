import argparse
import json
from datetime import date, timedelta
from pathlib import Path

import akshare as ak
import pandas as pd

ROOT_DIR       = Path(__file__).resolve().parent.parent
BASE_DIR       = ROOT_DIR / "A股重点板块"
CHECKPOINT_DIR = ROOT_DIR / "checkpoint"
CHECKPOINT_FILE = CHECKPOINT_DIR / "fetch_stocks_checkpoint.json"

DEFAULT_START_DATE = "20250101"   # used only when no checkpoint entry exists
END_DATE = date.today().strftime("%Y%m%d")

SECTORS: dict[str, list[tuple[str, str, str]]] = {
    "电力设备与新能源": [
        ("宁德时代", "300750", "a"),
        ("亿纬锂能", "300014", "a"),
        ("阳光电源", "300274", "a"),
        ("隆基绿能", "601012", "a"),
        ("比亚迪",   "002594", "a"),
    ],
    "医药生物": [
        ("恒瑞医药", "600276", "a"),
        ("药明康德", "603259", "a"),
        ("复星医药", "600196", "a"),
        ("迈瑞医疗", "300760", "a"),
        ("云南白药", "000538", "a"),
    ],
    "银行": [
        ("招商银行", "600036", "a"),
        ("工商银行", "601398", "a"),
        ("平安银行", "000001", "a"),
        ("建设银行", "601939", "a"),
        ("兴业银行", "601166", "a"),
    ],
    "半导体与电子": [
        ("中微公司",   "688012", "a"),
        ("北方华创",   "002371", "a"),
        ("华虹半导体", "688347", "a"),
        ("韦尔股份",   "603501", "a"),
        ("兆易创新",   "603986", "a"),
    ],
    "食品饮料（白酒）": [
        ("贵州茅台", "600519", "a"),
        ("五粮液",   "000858", "a"),
        ("泸州老窖", "000568", "a"),
        ("洋河股份", "002646", "a"),
        ("山西汾酒", "600809", "a"),
    ],
    "汽车": [
        ("上汽集团", "600104", "a"),
        ("长城汽车", "601633", "a"),
        ("吉利汽车", "00175",  "hk"),
        ("广汽集团", "601238", "a"),
        ("江淮汽车", "600418", "a"),
    ],
    "非银金融（券商）": [
        ("中信证券", "600030", "a"),
        ("东方财富", "300059", "a"),
        ("国泰君安", "601211", "a"),
        ("华泰证券", "601688", "a"),
        ("广发证券", "000776", "a"),
    ],
}


# ── checkpoint helpers ────────────────────────────────────────────────────────

def load_checkpoint() -> dict[str, str]:
    """Return {symbol: latest_date_str} e.g. {'300750': '2026-03-24'}."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    if not CHECKPOINT_FILE.exists():
        return {}
    text = CHECKPOINT_FILE.read_text(encoding="utf-8").strip()
    if not text:
        return {}
    return json.loads(text)


def save_checkpoint(ckpt: dict[str, str]) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_FILE.write_text(
        json.dumps(ckpt, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# ── fetch & save ──────────────────────────────────────────────────────────────

def fetch_stock(symbol: str, market: str, start: str, end: str) -> pd.DataFrame:
    if market == "a":
        return ak.stock_zh_a_hist(
            symbol=symbol,
            period="daily",
            start_date=start,
            end_date=end,
            adjust="",
        )
    else:
        return ak.stock_hk_hist(
            symbol=symbol,
            period="daily",
            start_date=start,
            end_date=end,
            adjust="",
        )


def save_by_month(df: pd.DataFrame, company_dir: Path) -> None:
    date_col = "日期" if "日期" in df.columns else df.columns[0]
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["_year"]  = df[date_col].dt.year
    df["_month"] = df[date_col].dt.month

    for (year, month), group in df.groupby(["_year", "_month"]):
        year_dir = company_dir / "quotes" / str(year)
        year_dir.mkdir(parents=True, exist_ok=True)
        out_path = year_dir / f"{month:02d}.txt"

        clean = group.drop(columns=["_year", "_month"])

        # append-merge: if file exists, merge on date column to avoid duplicates
        if out_path.exists():
            existing = pd.read_csv(out_path, sep=r"\s+", engine="python")
            existing[date_col] = pd.to_datetime(existing[date_col])
            merged = (
                pd.concat([existing, clean])
                .drop_duplicates(subset=[date_col])
                .sort_values(date_col)
            )
            out_path.write_text(merged.to_string(index=False), encoding="utf-8")
        else:
            out_path.write_text(clean.to_string(index=False), encoding="utf-8")


def folder_name(name: str, symbol: str, market: str) -> str:
    suffix = f"{symbol}.HK" if market == "hk" else symbol
    return f"{name}({suffix})"


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch daily stock quotes.")
    parser.add_argument(
        "--add",
        action="store_true",
        help="Ignore checkpoint and fetch from DEFAULT_START_DATE",
    )
    args = parser.parse_args()

    ckpt = {} if args.add else load_checkpoint()

    for sector, companies in SECTORS.items():
        for name, symbol, market in companies:
            # determine start date
            if symbol in ckpt and not args.add:
                # resume from the day after the last saved date
                last_date = date.fromisoformat(ckpt[symbol])
                start = (last_date + timedelta(days=1)).strftime("%Y%m%d")
            else:
                start = DEFAULT_START_DATE

            end = END_DATE
            if start > end:
                print(f"  {name} ({symbol}): already up to date ({ckpt.get(symbol)}), skip.")
                continue

            company_dir = BASE_DIR / sector / folder_name(name, symbol, market)
            print(f"Fetching {name} ({symbol})  {start} → {end} ...", end=" ", flush=True)
            try:
                df = fetch_stock(symbol, market, start, end)
                if df is None or df.empty:
                    print("no data returned, skipped.")
                    continue

                save_by_month(df, company_dir)

                # update checkpoint with the latest date in this batch
                date_col = "日期" if "日期" in df.columns else df.columns[0]
                latest = pd.to_datetime(df[date_col]).max().date().isoformat()
                ckpt[symbol] = latest
                save_checkpoint(ckpt)

                print(f"saved {len(df)} rows, latest={latest}.")
            except Exception as exc:
                print(f"ERROR – {exc}")


if __name__ == "__main__":
    main()
