# A股重点板块数据仓库

这个项目围绕 7 大行业板块股票，维护了三类核心数据（行情、论坛、资讯/研报），数据落在 **`exports/*.csv`**（UTF-8），并提供可视化分析页面；**不再使用 PostgreSQL**。

**股票清单**：`config/stocks.json`（`sectors` → 每板块内 `name` / `symbol` / `market`）。抓取脚本与 `stock_daily_dashboard` 均从此文件读取，不再扫描本地板块目录。

## 信息源总览

当前主要信息源：

- **东方财富-行情接口（akshare 封装）**：日线 K 线行情（A 股 + 部分港股）
- **东方财富-股吧**：帖子标题、URL、发布时间、点击量、评论数
- **东方财富-新闻搜索**：新闻标题、URL、日期（eastmoney platform）
- **新浪财经-研报列表**：研报标题、URL、日期（sina platform）
- **新浪财经-研报详情页**：研报正文内容（填充 `exports/report.csv` 的 `content` 列）

## 本地配置与缓存

- **`config/stocks.json`**：7 个板块、35 只股票的唯一来源。
- 行情/股吧/新闻/研报数据在 **`exports/quotes.csv`、`exports/comments.csv`、`exports/news.csv`、`exports/report.csv`**；`stock_daily_dashboard` 的 K 线来自 **`exports/quotes.csv`**。
- 研报正文抓取可能使用本地缓存目录（见 `fetch_report_content.py`），与股票列表无关。

## `exports/` CSV 结构（主数据）

合并与路径逻辑见 **`scripts/csv_io.py`**；行情字段映射见 **`scripts/quotes_db.py`**。

| 文件 | 说明 |
|------|------|
| `exports/comments.csv` | `url,post_id,symbol,post_title,publish_time,click_count,comment_count`（股吧；按 `url` 去重） |
| `exports/news.csv` | `url,symbol,title,date`（东财新闻列表） |
| `exports/report.csv` | `url,symbol,title,date,content`（新浪研报；正文可由 `fetch_report_content.py` 回填） |
| `exports/quotes.csv` | `symbol,trade_date,open,close,high,low,volume,amount,amplitude,pct_change,change_amount,turnover`（日线；`(symbol, trade_date)` 唯一） |

### 从 `quotes` 导出「7 日 K 线 → 下一日涨/跌」训练集

脚本：`scripts/build_quotes_7d_dataset.py`  

只读取 **`trade_date < --train-before`**（默认 `2026-01-01`）的日线；按股票**滑动窗口**（Day1–Day7 特征 → 预测 Day8）。归一化只在上述截止前的序列上估计（与 [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) §9.2 思路一致）。

输出**单个** UTF-8 CSV，列 **`prompt`, `label`**：`label` 为「涨」或「跌」——若 Day8 当日 `pct_change >= 0` 为「涨」，否则为「跌」。Prompt 要求模型**只输出**「涨」或「跌」。

默认输出：`train/dataset/quotes_7d_pre2026_dataset.csv`。

```bash
python scripts/build_quotes_7d_dataset.py
python scripts/build_quotes_7d_dataset.py -o train/dataset/my_train.csv
python scripts/build_quotes_7d_dataset.py --train-before 2026-01-01
```

### GRPO 强化学习训练（Qwen2.5-7B-Instruct）

使用 **Hugging Face TRL** 的 **GRPO** + **Accelerate + DeepSpeed ZeRO-3**，默认 **8 卡**（`CUDA_VISIBLE_DEVICES` 可改）。**Rollout 默认启用 vLLM**（`vllm_mode=colocate`，与训练同机共享 GPU；依赖 `trl[vllm]`）。入口：`train/train_grpo_qwen.py`；启动：`train/run_grpo_8gpu.sh`；DeepSpeed：`train/ds_zero3.json`；Accelerate：`train/accelerate_deepspeed_zero3.yaml`。架构与 reward 细节见 [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) **§10**。

**数据**：`train_grpo_qwen.py` 当前示例仍按 **`prompt` + `pct_change`（回归）** 设计；若改用上述「涨/跌」数据集，需自行调整数据列与 reward（例如分类准确率）。

**Reward（回归示例）**：补全**最后一行**解析浮点数（去 `%`），`reward = exp(-|pred - label| / 100)`；解析失败 **0**。

```bash
pip install -r train/requirements.txt
# 可选：huggingface-cli login
bash train/run_grpo_8gpu.sh
# 关闭 vLLM（慢，仅调试）：加 --no_vllm
# 独立 vLLM 服务：trl vllm-serve ... 后 train_grpo_qwen.py --vllm_mode server --vllm_server_base_url http://...
```

常用参数：`--vllm_gpu_memory_utilization`（colocate 显存比例）、`--num_generations`、`--max_prompt_length`、`--max_completion_length`、`--report_to tensorboard`。

## 核心脚本与使用方式

以下命令默认在项目根目录执行。

---

### A. 行情抓取：`scripts/fetch_stocks.py`

作用：通过 akshare 抓取日线行情并 **合并写入 `exports/quotes.csv`**（按 `symbol` + `trade_date` 去重覆盖）。

特性：

- checkpoint：`checkpoint/fetch_stocks_checkpoint.json`（按股票代码记录已抓到的最新交易日）
- 默认增量（从 checkpoint 次日抓到今日）
- `--add` 从 `DEFAULT_START_DATE` 全量重拉（仍按主键去重更新）
- 依赖：`akshare`、`pandas`

用法：

```bash
python scripts/fetch_stocks.py
python scripts/fetch_stocks.py --add
```

---

### B. 统一抓取：`scripts/fetch_market_data.py`（股吧 + 东财新闻 + 新浪研报列表）

作用：按 `config/stocks.json` 依次抓取并写入 **`exports/comments.csv`**、**`exports/news.csv`**、**`exports/report.csv`**（均为按 `url` 去重合并）。

- **`--mode comments`**：东方财富股吧 API → `comments.csv`。每只股票多线程分页；单股最多约 1w 条新插入后封顶；连续多页有数据但 CSV 零新增则停。合并写入后，**每个 `symbol` 每个自然日只保留 `click_count` 最高的 200 条**（`csv_io.COMMENTS_MAX_PER_SYMBOL_DAY`，由 `merge_append_comments` 与 `run_comments` 结束时的全表裁剪共同保证）。
- **`--mode news`**：Playwright 东财新闻搜索 → `news.csv`（最多约 50 页、zero-save-streak）。
- **`--mode report`**：新浪研报列表 → `report.csv`（同上策略）。
- **`--mode all`**：依次执行 comments → news → report。

Checkpoint：`checkpoint/fetch_forum_checkpoint.json`（股吧按公司分页）、`checkpoint/fetch_news_<platform>_checkpoint.json`（新闻/研报按公司已完成的 `company_key`）。

```bash
python scripts/fetch_market_data.py --mode all
python scripts/fetch_market_data.py --mode comments --add
python scripts/fetch_market_data.py --mode news
python scripts/fetch_market_data.py --mode report --offset 5
```

**新闻正文落盘**：`scripts/fetch_news_content_to_disk.py` 从 **`exports/news.csv`**（默认路径，可由 `--csv` 指定）读取 `url,symbol,title,date`，抓取东财正文（`#ContentBody`），写入 `Content/news/{YYYY-MM}/{sha256(url)}.txt`，支持断点续传（已存在且非空则跳过）。失败时按指数退避重试：首次等待 `--retry-base-delay`（默认与 `--delay` 相同），每次再失败则等待时间翻倍，最多 `--max-attempts` 次（默认 5）。默认板块「电力设备与新能源」、默认 `--since 2025-11-01`。**`--all`**：`config/stocks.json` 全部约 35 只股票、`news.date >= 2025-01-01`（可用 `--since` 覆盖起始日）。

```bash
python scripts/fetch_news_content_to_disk.py
python scripts/fetch_news_content_to_disk.py --all
python scripts/fetch_news_content_to_disk.py --symbols 300750,300014 --since 2025-11-01 --force
```

**研报正文落盘**：`scripts/fetch_report_content_to_disk.py` 从 **`exports/report.csv`**（默认，`--csv` 可改）读取行，复用 `fetch_report_content.py` 的抓取与 `div.blk_container` 解析，写入 **`Content/report/{sha256(url)}.txt`**，头格式与看板缓存一致。失败时指数退避重试：`--retry-base-delay`（默认同 `--delay`，默认 2s）、`--max-attempts`（默认 5）、`--timeout`。默认板块「电力设备与新能源」；无 `--since` 时不按日期过滤。**`--all`**：全部约 35 只股票、`report.date >= 2025-01-01`（可用 `--since` 覆盖）。断点续传：已有非空正文则跳过。

```bash
python scripts/fetch_report_content_to_disk.py
python scripts/fetch_report_content_to_disk.py --all
python scripts/fetch_report_content_to_disk.py --workers 2 --force
```

---

### C. 新浪研报正文回填：`scripts/fetch_report_content.py`

对 **`exports/report.csv` 中 `content` 为空** 的行抓正文并回写该 CSV。

```bash
python scripts/fetch_report_content.py
```

## 可视化与LLM分析页面

脚本：`scripts/stock_daily_dashboard.py`

功能：

- 选股票 + 选日期查看当日 `comments / news / reports`
- 一键总结当日评论观点（流式输出）
- 基于“总结 + 最近 7 交易日 K 线（`exports/quotes.csv`）”预测 1/3/7 天走势
- 总结最近 3 条研报正文、当日 news 等（见页面按钮）

启动：

```bash
python scripts/stock_daily_dashboard.py
```

访问：

- [http://127.0.0.1:5050](http://127.0.0.1:5050)

### 页面 LLM（阿里云 DashScope / 通义，OpenAI 兼容流式）

看板通过 **`openai`** 客户端连接 **`https://dashscope.aliyuncs.com/compatible-mode/v1`**，**全部使用流式** `chat.completions.create(..., stream=True)`（与官方示例一致）。`.env` 示例：

- **`DASHSCOPE_API_KEY`** 或 **`DASHBOARD_API_KEY`**：与地域绑定的 API Key
- **`DASHSCOPE_BASE_URL`**（可选）：默认上述 compatible-mode 地址
- **`DASHSCOPE_MODEL`** / **`QWEN_MODEL`**（可选）：默认 `qwen3-max`

依赖：`pip install openai`。项目根目录 `.env` 会由 `stock_daily_dashboard.py` 自动加载。

## 备注

- 数据落盘与入库都以 `url` 作为去重主键，避免重复采集
- `symbol` 字段统一为 6 位字符串，保留前导零
- 如果中途中断，绝大多数任务可直接重跑继续
