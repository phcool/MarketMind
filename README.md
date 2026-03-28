# A股重点板块数据仓库

这个项目围绕 7 大行业板块股票，维护了三类核心数据（行情、论坛、资讯/研报），并提供 PostgreSQL 入库与可视化分析页面。

## 信息源总览

当前主要信息源：

- **东方财富-行情接口（akshare 封装）**：日线 K 线行情（A 股 + 部分港股）
- **东方财富-股吧**：帖子标题、URL、发布时间、点击量、评论数
- **东方财富-新闻搜索**：新闻标题、URL、日期（eastmoney platform）
- **新浪财经-研报列表**：研报标题、URL、日期（sina platform）
- **新浪财经-研报详情页**：研报正文内容（填充 report.content）

## 本地数据目录结构

```text
A股重点板块/
└── <板块>/
    └── <公司(代码)>/
        └── （可选）其它本地文件；行情/股吧/新闻/研报均在 PostgreSQL
```

`stock_daily_dashboard` 与抓取脚本通过 **文件夹名** `公司(代码)` 发现股票列表；K 线来自 **`quotes` 表**，不再读取本地 `quotes/*.txt`。

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

### D. 入库脚本

#### 1) 论坛入库：`scripts/import_forum_to_pg.py`

将历史 **`forum/*.json`** 导入 `comments`（按 `url` 去重）；日常抓取已直写库，本脚本多用于一次性迁移。

```bash
python scripts/import_forum_to_pg.py
```

#### 2) 东财新闻入库：`scripts/import_eastmoney_news_to_pg.py`

将历史 **`eastmoney.json`** 导入 `news`；日常抓取已直写库。

```bash
python scripts/import_eastmoney_news_to_pg.py
```

#### 3) 新浪研报列表入库：`scripts/import_sina_to_pg.py`

将历史 **`sina.json`** 导入 `report`；日常抓取已直写库。

```bash
python scripts/import_sina_to_pg.py
```

#### 4) 新浪研报正文抓取：`scripts/fetch_report_content.py`

对 `report.content IS NULL` 的记录抓正文并回写数据库。

```bash
python scripts/fetch_report_content.py
```

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
