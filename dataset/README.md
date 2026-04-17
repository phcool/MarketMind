---
pretty_name: MarketMind Datasets
license: mit
task_categories:
- text-classification
- time-series-forecasting
language:
- zh
tags:
- finance
- stocks
- chinese
- marketmind
---

# MarketMind Datasets

This dataset repository contains the CSV datasets currently used in the `MarketMind` project.

## Files

- `quotes_7d_pre2026_dataset.csv`: pre-2026 training-style quote dataset.
- `quotes_7d_eval_20260101_20260228.csv`: 2026-01-01 to 2026-02-28 evaluation split.
- `quotes_7d_cot_from_batch.csv`: chain-of-thought style dataset generated from batch processing.
- `quotes_summary_5d_2026-01-01_to_2026-04-01.csv`: daily stock dataset with 5-day normalized K-line context, recent news/report summaries, and 1/3/7-trading-day movement labels.

## `quotes_summary_5d_2026-01-01_to_2026-04-01.csv`

Columns:

- `date`: trading date.
- `stock`: stock code.
- `kline_5d`: normalized K-line information for the current trading day and the previous 4 trading days.
- `news`: up to 30 recent summarized news items as JSON, restricted to within the previous month.
- `reports`: up to 3 recent summarized report items as JSON, restricted to within the previous month.
- `future_1_3_7_trade_day_labels`: labels such as `涨，涨，跌`, comparing the close price on the current day with the close price after 1, 3, and 7 future trading days.

## Notes

- Text content is primarily in Chinese.
- This repository is intended for research and internal experimentation around stock movement prediction and financial text conditioning.
- Users should validate data quality, licensing, and suitability before downstream production use.
