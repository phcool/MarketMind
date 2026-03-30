"""
Unified entry: fetch comments (forum), news (eastmoney search), and/or report (sina) into exports/*.csv.

Reuses logic from fetch_forum_all.py and fetch_news_eastmoney.py; injects SECTORS from config/stocks.json.

Examples:
  python scripts/fetch_market_data.py --mode all
  python scripts/fetch_market_data.py --mode comments --add
  python scripts/fetch_market_data.py --mode news
  python scripts/fetch_market_data.py --mode report --offset 3

Checkpoints: same as the underlying scripts (forum + fetch_news_eastmoney per platform).
"""

from __future__ import annotations

import argparse
import logging
import time

import fetch_forum_all as forum
import fetch_news_eastmoney as nem
from playwright.sync_api import sync_playwright

from stock_universe import load_sectors

# Injected into both submodules (same as their default from config/stocks.json).
SECTORS = load_sectors()

log = logging.getLogger(__name__)


def _apply_sectors() -> None:
    forum.SECTORS = SECTORS
    nem.SECTORS = SECTORS


def _total_stocks() -> int:
    return sum(len(v) for v in SECTORS.values())


def run_comments(args: argparse.Namespace) -> None:
    _apply_sectors()
    use_checkpoint = not args.add
    if args.add:
        forum.reset_checkpoint_for_run()
        log.info("--add: forum checkpoint reset")
    else:
        forum.load_all_checkpoints(use_checkpoint=True)
        if forum._checkpoint_data:
            log.info("Forum checkpoint: %d companies", len(forum._checkpoint_data))

    total = _total_stocks()
    start_index = max(1, args.offset)
    if start_index > total:
        log.warning("offset %d > total stocks %d, nothing to do.", start_index, total)
        return
    if start_index > 1:
        log.info("offset=%d: skip first %d stocks", start_index, start_index - 1)

    done = 0
    for sector, companies in SECTORS.items():
        for name, symbol, market in companies:
            done += 1
            if done < start_index:
                log.info("── [%d/%d] %s / %s — skip (offset)", done, total, sector, name)
                continue
            log.info("── [%d/%d] %s / %s [comments]", done, total, sector, name)
            try:
                forum.fetch_company(name, symbol, market, sector, use_checkpoint)
            except Exception as exc:
                log.exception("comments %s: %s", name, exc)


def run_news(args: argparse.Namespace) -> None:
    _apply_sectors()
    use_checkpoint = not args.add
    platform = "eastmoney"
    if args.add:
        nem.reset_checkpoint(platform)
        log.info("--add: checkpoint reset for %s", platform)
    completed_set = nem.load_checkpoint(platform, use_checkpoint)
    if use_checkpoint and completed_set:
        log.info("News checkpoint (%s): %d companies done", platform, len(completed_set))

    total = _total_stocks()
    start_index = max(1, args.offset)
    if start_index > total:
        log.warning("offset %d > total %d, skip news.", start_index, total)
        return

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
            done = 0
            for sector, companies in SECTORS.items():
                for name, symbol, market in companies:
                    done += 1
                    if done < start_index:
                        continue
                    company_key = nem._company_key(sector, name, symbol, market)
                    if company_key in completed_set:
                        log.info(
                            "── [%d/%d] %s / %s — skip [news, done]",
                            done, total, sector, name,
                        )
                        continue
                    log.info("── [%d/%d] %s / %s [news]", done, total, sector, name)
                    try:
                        nem.fetch_eastmoney(page, None, name, symbol, market, sector)
                        completed_set.add(company_key)
                        nem.save_checkpoint(platform, completed_set)
                    except Exception as exc:
                        log.exception("news %s: %s", name, exc)
                    time.sleep(1)
        finally:
            browser.close()


def run_report(args: argparse.Namespace) -> None:
    _apply_sectors()
    use_checkpoint = not args.add
    platform = "sina"
    if args.add:
        nem.reset_checkpoint(platform)
        log.info("--add: checkpoint reset for %s", platform)
    completed_set = nem.load_checkpoint(platform, use_checkpoint)
    if use_checkpoint and completed_set:
        log.info("Report checkpoint (%s): %d companies done", platform, len(completed_set))

    total = _total_stocks()
    start_index = max(1, args.offset)
    if start_index > total:
        log.warning("offset %d > total %d, skip report.", start_index, total)
        return

    done = 0
    for sector, companies in SECTORS.items():
        for name, symbol, market in companies:
            done += 1
            if done < start_index:
                continue
            company_key = nem._company_key(sector, name, symbol, market)
            if use_checkpoint and company_key in completed_set:
                log.info(
                    "── [%d/%d] %s / %s — skip [report, done]",
                    done, total, sector, name,
                )
                continue
            log.info("── [%d/%d] %s / %s [report]", done, total, sector, name)
            try:
                nem.fetch_sina(None, None, name, symbol, market, sector)
                completed_set.add(company_key)
                nem.save_checkpoint(platform, completed_set)
            except Exception as exc:
                log.exception("report %s: %s", name, exc)
            time.sleep(1)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )

    parser = argparse.ArgumentParser(
        description="Fetch comments / eastmoney news / sina reports into exports/*.csv (unified SECTORS).",
    )
    parser.add_argument(
        "--mode",
        choices=("all", "comments", "news", "report"),
        default="all",
        help="all=forum + news + report; comments=股吧; news=eastmoney->news.csv; report=sina->report.csv",
    )
    parser.add_argument(
        "--add",
        action="store_true",
        help="Ignore checkpoints for selected mode(s); CSV still dedupes by url.",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=1,
        help="1-based stock index to start from (applies to each mode run).",
    )
    args = parser.parse_args()

    _apply_sectors()

    if args.mode == "all":
        if args.add:
            forum.reset_checkpoint_for_run()
            nem.reset_checkpoint("eastmoney")
            nem.reset_checkpoint("sina")
            log.info("--add: reset forum + eastmoney + sina checkpoints")
        log.info("=== mode=all: comments then news then report ===")
        run_comments(args)
        run_news(args)
        run_report(args)
    elif args.mode == "comments":
        run_comments(args)
    elif args.mode == "news":
        run_news(args)
    elif args.mode == "report":
        run_report(args)

    log.info("Done (mode=%s).", args.mode)


if __name__ == "__main__":
    main()
