# A股重点板块数据仓库

这个项目围绕 7 大行业板块股票，维护了三类核心数据（行情、论坛、资讯/研报），并提供 PostgreSQL 入库与可视化分析页面。

**股票清单**：`config/stocks.json`（`sectors` → 每板块内 `name` / `symbol` / `market`）。抓取脚本与 `stock_daily_dashboard` 均从此文件读取，不再扫描本地板块目录。

## 信息源总览

当前主要信息源：

- **东方财富-行情接口（akshare 封装）**：日线 K 线行情（A 股 + 部分港股）
- **东方财富-股吧**：帖子标题、URL、发布时间、点击量、评论数
- **东方财富-新闻搜索**：新闻标题、URL、日期（eastmoney platform）
- **新浪财经-研报列表**：研报标题、URL、日期（sina platform）
- **新浪财经-研报详情页**：研报正文内容（填充 report.content）

## 本地配置与缓存

- **`config/stocks.json`**：7 个板块、35 只股票的唯一来源。
- 行情/股吧/新闻/研报数据在 **PostgreSQL**；`stock_daily_dashboard` 的 K 线来自 **`quotes` 表**。
- 研报正文抓取可能使用本地缓存目录（见 `fetch_report_content.py`），与股票列表无关。

## 数据库结构（PostgreSQL: `financial_data`）

当前主表：

### 1) `comments`

字段：

- `url` `TEXT` PRIMARY KEY
- `post_id` `VARCHAR(50)`
- `symbol` `VARCHAR(6)`
- `post_title` `TEXT`
- `publish_time` `TIMESTAMP`
- `click_count` `INTEGER`
- `comment_count` `INTEGER`

索引：

- `idx_comments_symbol(symbol)`
- `idx_comments_publish_time(publish_time)`

### 2) `news`

字段：

- `url` `TEXT` PRIMARY KEY
- `symbol` `VARCHAR(6)`
- `title` `TEXT`
- `date` `DATE`

索引：

- `idx_news_symbol(symbol)`
- `idx_news_date(date)`

### 3) `report`

字段：

- `url` `TEXT` PRIMARY KEY
- `symbol` `VARCHAR(6)`
- `title` `TEXT`
- `date` `DATE`
- `content` `TEXT`（研报正文）

索引：

- `idx_report_symbol(symbol)`
- `idx_report_date(date)`

### 4) `quotes`

日线行情（由 `fetch_stocks.py` 写入；列语义与 akshare 日线一致）。

字段：

- `symbol` `VARCHAR(10)` — 股票代码（与 `trade_date` 联合主键）
- `trade_date` `DATE`
- `open` / `close` / `high` / `low` `NUMERIC(14,4)`
- `volume` `BIGINT` — 成交量
- `amount` `NUMERIC(22,6)` — 成交额
- `amplitude` `NUMERIC(10,4)` — 振幅
- `pct_change` `NUMERIC(10,4)` — 涨跌幅
- `change_amount` `NUMERIC(14,4)` — 涨跌额
- `turnover` `NUMERIC(10,4)` — 换手率

主键：`PRIMARY KEY (symbol, trade_date)`

索引：`idx_quotes_symbol(symbol)`、`idx_quotes_trade_date(trade_date)`

建表 SQL：`scripts/sql/create_quotes_table.sql` · 共享逻辑：`scripts/quotes_db.py`

### 从 `quotes` 导出「7 日 K 线 → prompt」训练集

脚本：`scripts/build_quotes_7d_dataset.py`  

从库中读取 `trade_date <= data_end`（默认 `2026-03-28`）的日线，按股票做**滑动窗口**（7 日特征 → 第 8 日涨跌幅标签）。**归一化**在每只股票的「训练 + 验证」合并序列上估计（与 [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) §9.2 一致），再分别写入：

- **训练集**：标签日 `< 2026-01-01` → 默认 `exports/quotes_7d_pre2026_dataset.csv`
- **验证集**：标签日 `2026-01-01`～`2026-03-28` → 默认 `exports/quotes_7d_val_20260101_20260328_dataset.csv`

**CSV 第二列仍为真实涨跌幅（%）**，便于算 MAE。

```bash
python scripts/build_quotes_7d_dataset.py
python scripts/build_quotes_7d_dataset.py -o exports/train.csv --val-output exports/val.csv
python scripts/build_quotes_7d_dataset.py --data-end 2026-03-28 --train-before 2026-01-01 --val-start 2026-01-01 --val-end 2026-03-28
```

环境变量：`PG_DSN`（可选，默认 `dbname=financial_data`）。

### GRPO 强化学习训练（Qwen2.5-7B-Instruct）

使用 **Hugging Face TRL** 的 **GRPO** + **Accelerate + DeepSpeed ZeRO-3**，默认 **8 卡**（`CUDA_VISIBLE_DEVICES` 可改）。**Rollout 默认启用 vLLM**（`vllm_mode=colocate`，与训练同机共享 GPU；依赖 `trl[vllm]`）。入口：`train/train_grpo_qwen.py`；启动：`train/run_grpo_8gpu.sh`；DeepSpeed：`train/ds_zero3.json`；Accelerate：`train/accelerate_deepspeed_zero3.yaml`。架构与 reward 细节见 [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) **§10**。

**数据**：CSV 需含 `prompt`、`pct_change`。默认 `train/dataset/quotes_7d_pre2026_dataset.csv`。

**Reward**：补全**最后一行**解析浮点数（去 `%`），`reward = exp(-|pred - label| / 100)`；解析失败 **0**。

```bash
pip install -r train/requirements.txt
# 可选：huggingface-cli login
bash train/run_grpo_8gpu.sh
# 关闭 vLLM（慢，仅调试）：加 --no_vllm
# 独立 vLLM 服务：trl vllm-serve ... 后 train_grpo_qwen.py --vllm_mode server --vllm_server_base_url http://...
```

常用参数：`--vllm_gpu_memory_utilization`（colocate 显存比例）、`--num_generations`、`--max_prompt_length`、`--max_completion_length`、`--report_to tensorboard`。

## 数据量（当前库快照）

- `comments`: **3,301,111**
- `news`: **5,372**
- `report`: **9,562**

> 注：数据量会随抓取与导入任务持续变化。

## 核心脚本与使用方式

以下命令默认在项目根目录执行。

---

### A. 行情抓取：`scripts/fetch_stocks.py`

作用：通过 akshare 抓取日线行情并 **upsert 到 PostgreSQL `quotes` 表**（主键 `symbol` + `trade_date`）。

特性：

- checkpoint：`checkpoint/fetch_stocks_checkpoint.json`（按股票代码记录已抓到的最新交易日）
- 默认增量（从 checkpoint 次日抓到今日）
- `--add` 从 `DEFAULT_START_DATE` 全量重拉（仍按主键去重更新）
- 依赖：`akshare`、`pandas`、`psycopg2`；DSN 同 `PG_DSN` / `dbname=financial_data`

用法：

```bash
python scripts/fetch_stocks.py
python scripts/fetch_stocks.py --add
```

---

### B. 论坛抓取：`scripts/fetch_forum_all.py`

作用：抓取股吧帖子并写入 PostgreSQL **`comments`** 表（按 `url` 去重）。

特性（已实现）：

- checkpoint 续跑
- `--add` 全量重跑
- `--offset` 跳过前 N-1 只股票
- 每股票上限 1w 条
- 连续零新增页停止（zero-save-streak）

用法：

```bash
python scripts/fetch_forum_all.py
python scripts/fetch_forum_all.py --add
python scripts/fetch_forum_all.py --offset 10
```

---

### C. 新闻/研报抓取：`scripts/fetch_news_eastmoney.py`

支持平台：

- `eastmoney`
- `sina`

特性：

- checkpoint 按平台独立管理（`checkpoint/fetch_news_<platform>_checkpoint.json`）
- `--add` 全量重抓（数据库按 `url` 去重 upsert/跳过重复）
- Eastmoney 使用新闻搜索 URL（按股票名 + 时间排序，最多 50 页）
- 已支持 zero-save-streak（连续 5 页无新增则停止）

用法：

```bash
python scripts/fetch_news_eastmoney.py --platform eastmoney
python scripts/fetch_news_eastmoney.py --platform sina
python scripts/fetch_news_eastmoney.py --platform eastmoney --add
```

---

### D. 统一抓取入口（可选）

`scripts/fetch_market_data.py`：按 `config/stocks.json` 依次跑股吧 / 东财新闻 / 新浪研报列表（与分别运行 `fetch_forum_all.py`、`fetch_news_eastmoney.py` 等价，checkpoint 行为一致）。

```bash
python scripts/fetch_market_data.py --mode all
python scripts/fetch_market_data.py --mode comments --offset 5
```

### E. 新浪研报正文抓取：`scripts/fetch_report_content.py`

对 `report.content IS NULL` 的记录抓正文并回写数据库。

```bash
python scripts/fetch_report_content.py
```

> **`scripts/import_all_to_pg.py` 已弃用**：仓库不再提供按本地目录树批量导入的模块；请使用上述 `fetch_*` 脚本直写数据库。

## 可视化与LLM分析页面

脚本：`scripts/stock_daily_dashboard.py`

功能：

- 选股票 + 选日期查看当日 `comments / news / reports`
- 一键总结当日评论观点（流式输出）
- 基于“总结 + 最近 7 交易日 K 线（`quotes` 表）”预测 1/3/7 天走势
- 总结最近 3 条研报正文、当日 news 等（见页面按钮）

启动：

```bash
python scripts/stock_daily_dashboard.py
```

访问：

- [http://127.0.0.1:5050](http://127.0.0.1:5050)

### 页面LLM相关环境变量

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`（默认 `https://api.chatanywhere.tech/v1`）
- `OPENAI_MODEL`（默认 `gpt-5`）

可放在 `.env`，或直接 `export` 后再启动页面。

## 备注

- 数据落盘与入库都以 `url` 作为去重主键，避免重复采集
- `symbol` 字段统一为 6 位字符串，保留前导零
- 如果中途中断，绝大多数任务可直接重跑继续
