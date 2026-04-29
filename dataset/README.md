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
- `quotes_7d_multi_pre2026_dataset.csv`: 7-day K-line multi-horizon GRPO training dataset.
- `quotes_7d_multi_eval_20260101_20260228.csv`: 2026 evaluation split for the multi-horizon GRPO task.
- `quotes_summary_5d_2026-01-01_to_2026-04-01.csv`: daily stock dataset with 5-day normalized K-line context, recent news/report summaries, and 1/3/7-trading-day movement labels.

## `quotes_7d_multi_pre2026_dataset.csv`

Columns:

- `prompt`: seven normalized daily K-line rows plus instructions to analyze first and then output three final direction labels.
- `future_1_3_7_trade_day_labels`: labels such as `涨，跌，涨`, comparing the Day7 close with the close after 1, 3, and 7 future trading days.

This dataset is used by `train/scripts/grpo/train_grpo_qwen.py`. The GRPO reward parses the final 1/3/7-day direction answers and scores the average per-horizon accuracy; unparsable outputs receive 0 reward.

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
