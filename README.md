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
        ├── quotes/                    # 行情（按年/月）
        │   └── <YYYY>/<MM>.txt
        ├── forum/                     # 论坛帖子（按年/月/日）
        │   └── <YYYY>/<MM>/<DD>.json
        └── news/                      # 资讯/研报（按年/月/日）
            └── <YYYY>/<MM>/<DD>/
                ├── eastmoney.json
                └── sina.json
```

当前文件规模（本地统计）：

- `forum/*.json`: **126,812**
- `news/*/eastmoney.json`: **2,762**
- `news/*/sina.json`: **4,803**
- `quotes/*.txt`: **1,518**

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

## 数据量（当前库快照）

- `comments`: **3,301,111**
- `news`: **5,372**
- `report`: **9,562**

> 注：数据量会随抓取与导入任务持续变化。

## 核心脚本与使用方式

以下命令默认在项目根目录执行。

---

### A. 行情抓取：`scripts/fetch_stocks.py`

作用：抓取日线行情并写入 `quotes/`。

特性：

- 支持 checkpoint：`checkpoint/fetch_stocks_checkpoint.json`
- 默认增量更新（从上次最新日期+1开始）
- `--add` 可忽略 checkpoint 重拉

用法：

```bash
python scripts/fetch_stocks.py
python scripts/fetch_stocks.py --add
```

---

### B. 论坛抓取：`scripts/fetch_forum_all.py`

作用：抓取股吧帖子并写入 `forum/YYYY/MM/DD.json`。

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
- `--add` 全量重抓（不清空历史文件，按 URL 去重追加）
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

将 `forum` 数据导入 `comments` 表（按 `url` 去重）。

```bash
python scripts/import_forum_to_pg.py
```

#### 2) 东财新闻入库：`scripts/import_eastmoney_news_to_pg.py`

将 `eastmoney.json` 导入 `news` 表（`symbol` 为 6 位代码）。

```bash
python scripts/import_eastmoney_news_to_pg.py
```

#### 3) 新浪研报列表入库：`scripts/import_sina_to_pg.py`

将 `sina.json` 导入 `report` 表（`symbol` 为 6 位代码）。

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
- 基于“总结 + 最近7天K线”预测 1/3/7 天走势
- 一键提取当日前最近 3 条 `report` 与 3 条 `news`

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
