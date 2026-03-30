# A股重点板块数据仓库

这个项目围绕 7 大行业板块股票，维护了三类核心数据（行情、论坛、资讯/研报），全部落在项目根目录 **`exports/*.csv`**（UTF-8），并提供可视化分析页面。

**股票清单**：`config/stocks.json`（`sectors` → 每板块内 `name` / `symbol` / `market`）。抓取脚本与 `stock_daily_dashboard` 均从此文件读取，不再扫描本地板块目录。

## 信息源总览

当前主要信息源：

- **东方财富-行情接口（akshare 封装）**：日线 K 线行情（A 股 + 部分港股）
- **东方财富-股吧**：帖子标题、URL、发布时间、点击量、评论数
- **东方财富-新闻搜索**：新闻标题、URL、日期（eastmoney platform）
- **新浪财经-研报列表**：研报标题、URL、日期（sina platform）
- **新浪财经-研报详情页**：研报正文内容（填充 `exports/report.csv` 的 `content` 列，或由 `fetch_report_content_to_disk.py` 写入 `Content/report/`）

## 本地配置与缓存

- **`config/stocks.json`**：7 个板块、35 只股票的唯一来源。
- 行情/股吧/新闻/研报数据在 **`exports/`** 下对应 CSV；`stock_daily_dashboard` 的 K 线与列表数据均从这些文件读取。
- 研报正文还可使用本地缓存目录 `Content/report/`（见 `fetch_report_content_to_disk.py` / 看板按需抓取）。

## `exports/` CSV 布局（唯一数据源）

| 文件 | 写入脚本 | 说明 |
|------|----------|------|
| `quotes.csv` | `fetch_stocks.py` | 主键语义：`symbol` + `trade_date`（合并时后者覆盖） |
| `comments.csv` | `fetch_forum_all.py` | 列同上；按 `url` 去重；写入后全局裁剪为每只股票每个自然日 **`click_count` 最高的 200 条**（`csv_io.COMMENTS_MAX_PER_SYMBOL_DAY`） |
| `news.csv` | `fetch_news_eastmoney.py --platform eastmoney` | 列：`url,symbol,title,date`；按 `url` 去重 |
| `report.csv` | `fetch_news_eastmoney.py --platform sina`、`fetch_report_content.py` | 列：`url,symbol,title,date,content`；列表抓取追加空 `content`，正文脚本回填 |

共享合并逻辑：`scripts/csv_io.py`；akshare → 行情行映射：`scripts/quotes_db.py`。

### 从 `exports/quotes.csv` 构建「7 日 K 线 → prompt」训练集

脚本：`scripts/build_quotes_7d_dataset.py`  

从 **`exports/quotes.csv`**（可用 `--quotes-csv` 指定）读取 `trade_date <= data_end`（默认 `2026-03-28`）的日线，按股票做**滑动窗口**（7 日特征 → 第 8 日涨跌幅标签）。**归一化**在每只股票的「训练 + 验证」合并序列上估计，再分别写入：

- **训练集**：标签日 `< 2026-01-01` → 默认 `exports/quotes_7d_pre2026_dataset.csv`
- **验证集**：标签日 `2026-01-01`～`2026-03-28` → 默认 `exports/quotes_7d_val_20260101_20260328_dataset.csv`

**CSV 第二列仍为真实涨跌幅（%）**，便于算 MAE。

```bash
python scripts/build_quotes_7d_dataset.py
python scripts/build_quotes_7d_dataset.py -o exports/train.csv --val-output exports/val.csv
python scripts/build_quotes_7d_dataset.py --data-end 2026-03-28 --train-before 2026-01-01 --val-start 2026-01-01 --val-end 2026-03-28
```

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

### B. 论坛抓取：`scripts/fetch_forum_all.py`

作用：抓取股吧帖子并写入 **`exports/comments.csv`**（按 `url` 去重；每次合并写入后按「股票 + 自然日」只保留 **`click_count` 最高的 200 条**，与离线裁剪规则一致）。

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
- `--add` 全量重抓（CSV 按 `url` 去重合并）
- Eastmoney 使用新闻搜索 URL（按股票名 + 时间排序，最多 50 页）
- 已支持 zero-save-streak（连续 5 页无新增则停止）

用法：

```bash
python scripts/fetch_news_eastmoney.py --platform eastmoney
python scripts/fetch_news_eastmoney.py --platform sina
python scripts/fetch_news_eastmoney.py --platform eastmoney --add
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

### D. 统一抓取入口（可选）

`scripts/fetch_market_data.py`：按 `config/stocks.json` 依次跑股吧 / 东财新闻 / 新浪研报列表（与分别运行 `fetch_forum_all.py`、`fetch_news_eastmoney.py` 等价，checkpoint 行为一致）。

```bash
python scripts/fetch_market_data.py --mode all
python scripts/fetch_market_data.py --mode comments --offset 5
```

### E. 新浪研报正文回填 CSV：`scripts/fetch_report_content.py`

对 **`exports/report.csv`** 中 `content` 为空的行抓取正文并回写该列（适合与列表抓取脚本配合）。

```bash
python scripts/fetch_report_content.py
```

### F. 删除旧 PostgreSQL 库（一次性）

若曾使用数据库 `financial_data` 且已迁到 CSV，可安装 `psycopg2-binary` 后执行（会要求输入 `YES` 确认）：

```bash
pip install psycopg2-binary
python scripts/drop_financial_data_database.py
```

默认使用管理连接串 `PG_ADMIN_DSN`（未设置时为 `dbname=postgres`），**不要**连到 `financial_data` 本身。

## 可视化与LLM分析页面

脚本：`scripts/stock_daily_dashboard.py`

功能：

- 选股票 + 选日期查看当日 `comments / news / reports`（来自 `exports/*.csv`）
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

- 数据写入 CSV 时以 `url`（或行情 `symbol+trade_date`）作为去重主键，避免重复采集
- `symbol` 字段统一为 6 位字符串，保留前导零
- 如果中途中断，绝大多数任务可直接重跑继续
