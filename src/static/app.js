function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

const LAST_FETCH_CACHE_KEY = "marketmind:last-fetch";
let currentData = null;

function predictionSummaryLine(data) {
  if (!data) {
    return '<p class="meta">点击 Predict，基于最近 7 个交易日 K 线和已生成的 news / reports / SerpAPI Summary，先分析再预测未来 1 日、3 日、7 日走向。最终结论每一项只会输出“上涨”或“下跌”。</p>';
  }
  return `<p class="content">${escapeHtml(data)}</p>`;
}

function saveLastFetch(data) {
  try {
    localStorage.setItem(LAST_FETCH_CACHE_KEY, JSON.stringify(data));
  } catch (error) {
    console.warn("Failed to cache last fetch result:", error);
  }
}

function loadLastFetch() {
  try {
    const raw = localStorage.getItem(LAST_FETCH_CACHE_KEY);
    if (!raw) {
      return null;
    }
    return JSON.parse(raw);
  } catch (error) {
    console.warn("Failed to load cached fetch result:", error);
    return null;
  }
}

function renderGroupWrapper(title, countLabel, body, options = {}) {
  const summaryButton = options.summarySection
    ? `<button type="button" class="summary-trigger" data-summary-section="${escapeHtml(options.summarySection)}">Summary</button>`
    : "";
  const contentButton = options.contentSection
    ? `<button type="button" class="summary-trigger secondary" data-content-section="${escapeHtml(options.contentSection)}">Content</button>`
    : "";
  const filterButton = options.filterSection
    ? `<button type="button" class="summary-trigger tertiary" data-filter-section="${escapeHtml(options.filterSection)}">Filter</button>`
    : "";
  const summaryContent = options.summarySection
    ? `
      <div class="llm-summary ${options.summary ? "has-content" : ""}" data-summary-box="${escapeHtml(options.summarySection)}">
        ${options.summary ? `<p class="content">${escapeHtml(options.summary)}</p>` : '<p class="meta">点击 Summary 生成该分组的集中观点总结。</p>'}
      </div>
    `
    : "";

  if (!options.collapsible) {
    return `
      <section class="group">
        <div class="group-head">
          <h2>${title}</h2>
          ${filterButton}
          ${contentButton}
          ${summaryButton}
          <span>${countLabel}</span>
        </div>
        ${summaryContent}
        ${body}
      </section>
    `;
  }

  return `
    <details class="group group-collapsible">
      <summary class="group-head">
        <h2>${title}</h2>
        ${filterButton}
        ${contentButton}
        ${summaryButton}
        <span>${countLabel}</span>
      </summary>
      <div class="group-body">
        ${summaryContent}
        ${body}
      </div>
    </details>
  `;
}

function renderSection(title, items, type, options = {}) {
  const sourceItems = items || [];
  const visibleItems = sourceItems.slice(0, options.displayLimit || sourceItems.length);
  const cards = visibleItems.map((item) => {
    const meta = [];
    if (item.publish_time) meta.push(`时间：${escapeHtml(item.publish_time)}`);
    if (item.date) meta.push(`日期：${escapeHtml(item.date)}`);
    if (item.click_count !== undefined) meta.push(`点击：${escapeHtml(item.click_count)}`);
    if (item.comment_count !== undefined) meta.push(`评论：${escapeHtml(item.comment_count)}`);
    const content = item.content
      ? `<p class="content">${escapeHtml(item.content)}</p>`
      : "";
    return `
      <article class="card ${type}">
        <a href="${escapeHtml(item.url)}" target="_blank" rel="noreferrer">${escapeHtml(item.title || item.url)}</a>
        <p class="meta">${meta.join(" ｜ ")}</p>
        ${content}
      </article>
    `;
  }).join("");

  return renderGroupWrapper(
    title,
    `${visibleItems.length} / ${sourceItems.length} 条`,
    `<div class="card-list">${cards || '<p class="empty">暂无结果</p>'}</div>`,
    options,
  );
}

function renderKlineSection(rows) {
  const cards = (rows || []).map((row) => `
    <article class="card kline">
      <p class="meta">交易日：${escapeHtml(row.trade_date)}</p>
      <p class="content">
开：${escapeHtml(row.open)} ｜ 收：${escapeHtml(row.close)}
高：${escapeHtml(row.high)} ｜ 低：${escapeHtml(row.low)}
涨跌幅：${escapeHtml(row.pct_change)} ｜ 成交量：${escapeHtml(row.volume)}
      </p>
    </article>
  `).join("");

  return renderGroupWrapper(
    "最近 7 个交易日 K 线",
    `${rows.length} 条`,
    `<div class="card-list">${cards || '<p class="empty">暂无 K 线结果</p>'}</div>`,
    { collapsible: true },
  );
}

function renderSerpSection(items) {
  const cards = (items || []).map((item) => `
    <article class="card serp">
      <a href="${escapeHtml(item.url)}" target="_blank" rel="noreferrer">${escapeHtml(item.title)}</a>
      <p class="meta">${escapeHtml(item.source || "SerpAPI")} ${item.date ? `｜ ${escapeHtml(item.date)}` : ""}</p>
      ${item.content_error ? `<p class="meta">正文抓取失败：${escapeHtml(item.content_error)}</p>` : ""}
      ${item.content ? `<p class="content">${escapeHtml(item.content)}</p>` : ""}
      ${item.snippet ? `<p class="content">${escapeHtml(item.snippet)}</p>` : ""}
    </article>
  `).join("");

  return renderGroupWrapper(
    "SerpAPI 补充搜索结果",
    `${(items || []).length} 条`,
    `<div class="card-list">${cards || '<p class="empty">暂无 SerpAPI 结果</p>'}</div>`,
    {
      collapsible: true,
      filterSection: "serp",
      contentSection: "serp",
      summarySection: "serp",
      summary: currentData?.serp_summary || "",
    },
  );
}

function renderResult(data) {
  currentData = data;
  const root = document.getElementById("result-root");
  const reportsNote = data.reports_note
    ? `<p class="meta">${escapeHtml(data.reports_note)}</p>`
    : "";
  const klineNote = data.kline_note
    ? `<p class="meta">${escapeHtml(data.kline_note)}</p>`
    : "";
  const serpNote = data.serp_note
    ? `<p class="meta">${escapeHtml(data.serp_note)}</p>`
    : "";
  root.innerHTML = `
    <section class="summary">
      <div class="summary-head">
        <h2>${escapeHtml(data.name)} <span>${escapeHtml(data.stock)}</span></h2>
        <button type="button" class="summary-trigger predict-trigger" id="predict-button">Predict</button>
      </div>
      <p>市场：${escapeHtml(data.market)}</p>
      ${reportsNote}
      ${klineNote}
      ${serpNote}
      <div class="llm-summary prediction-box ${data.prediction ? "has-content" : ""}" id="prediction-box">
        ${predictionSummaryLine(data.prediction || "")}
      </div>
    </section>
    ${renderKlineSection(data.kline || [])}
    ${renderSection("最新评论", data.comments || [], "comments", {
      displayLimit: 10,
      collapsible: true,
      summarySection: "comments",
      summary: data.comments_summary || "",
    })}
    ${renderSection("最新新闻", data.news || [], "news", {
      collapsible: true,
      summarySection: "news",
      summary: data.news_summary || "",
    })}
    ${renderSection("最新研报", data.reports || [], "reports", {
      collapsible: true,
      summarySection: "reports",
      summary: data.reports_summary || "",
    })}
    ${renderSerpSection(data.serp_results || [])}
  `;
  bindSummaryButtons();
}

function setPredictionBox(text, { loading = false, error = false } = {}) {
  const box = document.getElementById("prediction-box");
  if (!box) {
    return;
  }
  box.classList.toggle("has-content", Boolean(text));
  box.classList.toggle("is-loading", loading);
  box.classList.toggle("is-error", error);
  box.innerHTML = text ? `<p class="${error ? "meta" : "content"}">${escapeHtml(text)}</p>` : predictionSummaryLine("");
}

function setSummaryBox(section, text, { loading = false, error = false } = {}) {
  const box = document.querySelector(`[data-summary-box="${section}"]`);
  if (!box) {
    return;
  }
  box.classList.toggle("has-content", Boolean(text));
  box.classList.toggle("is-loading", loading);
  box.classList.toggle("is-error", error);
  if (!text) {
    box.innerHTML = '<p class="meta">点击 Summary 生成该分组的集中观点总结。</p>';
    return;
  }
  const cls = error ? "meta" : "content";
  box.innerHTML = `<p class="${cls}">${escapeHtml(text)}</p>`;
}

function sectionItems(section) {
  if (!currentData) {
    return [];
  }
  if (section === "comments") return currentData.comments || [];
  if (section === "news") return currentData.news || [];
  if (section === "reports") return currentData.reports || [];
  if (section === "serp") return currentData.serp_results || [];
  return [];
}

async function fetchSerpContent() {
  if (!currentData) {
    return [];
  }
  const filterNote = currentData.serp_filter_note || "";
  if (!filterNote) {
    await fetchSerpFilter();
  }
  setSummaryBox("serp", "正在抓取 SerpAPI 结果正文内容...", { loading: true });
  const resp = await fetch("/api/serp-content", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ items: sectionItems("serp") }),
  });
  const payload = await resp.json();
  if (!resp.ok || !payload.ok) {
    throw new Error(payload.error || "SerpAPI 正文抓取失败");
  }
  currentData.serp_results = payload.items || [];
  saveLastFetch(currentData);
  renderResult(currentData);
  setSummaryBox("serp", "", {});
  return currentData.serp_results;
}

async function fetchSerpFilter() {
  if (!currentData) {
    return [];
  }
  setSummaryBox("serp", "正在按规则过滤 PDF 链接...", { loading: true });
  const resp = await fetch("/api/serp-filter", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      stock: currentData.stock,
      name: currentData.name,
      items: currentData.serp_results || [],
    }),
  });
  const payload = await resp.json();
  if (!resp.ok || !payload.ok) {
    throw new Error(payload.error || "SerpAPI 结果过滤失败");
  }
  currentData.serp_results = payload.items || [];
  currentData.serp_filter_note = payload.note || "";
  currentData.serp_summary = "";
  saveLastFetch(currentData);
  renderResult(currentData);
  setSummaryBox("serp", currentData.serp_filter_note || "过滤完成");
  return currentData.serp_results;
}

async function fetchSectionSummary(section) {
  if (!currentData) {
    return;
  }
  setSummaryBox(section, "正在调用 Qwen3.6-max-preview 生成总结...", { loading: true });
  try {
    let items = sectionItems(section);
    if (section === "serp") {
      const filterNote = currentData.serp_filter_note || "";
      if (!filterNote) {
        items = await fetchSerpFilter();
      }
      const hasContent = items.some((item) => String(item.content || "").trim());
      if (!hasContent) {
        items = await fetchSerpContent();
      }
      if (!items.some((item) => String(item.content || "").trim())) {
        throw new Error("SerpAPI 结果尚未抓到可用于总结的正文内容");
      }
      setSummaryBox(section, "正在调用 Qwen3.6-max-preview 生成总结...", { loading: true });
    }
    const resp = await fetch(`/api/summary/${encodeURIComponent(section)}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        stock: currentData.stock,
        name: currentData.name,
        items,
      }),
    });
    const payload = await resp.json();
    if (!resp.ok || !payload.ok) {
      throw new Error(payload.error || "Summary 生成失败");
    }
    currentData[`${section}_summary`] = payload.summary;
    saveLastFetch(currentData);
    setSummaryBox(section, payload.summary);
  } catch (error) {
    setSummaryBox(section, error.message || "Summary 生成失败", { error: true });
  }
}

async function fetchPrediction() {
  if (!currentData) {
    return;
  }
  setPredictionBox("正在调用 Qwen3.6-max-preview 先分析再生成未来 1 日、3 日、7 日涨跌判断...", { loading: true });
  try {
    const resp = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        stock: currentData.stock,
        name: currentData.name,
        kline: currentData.kline || [],
        news_summary: currentData.news_summary || "",
        reports_summary: currentData.reports_summary || "",
        serp_summary: currentData.serp_summary || "",
      }),
    });
    const payload = await resp.json();
    if (!resp.ok || !payload.ok) {
      throw new Error(payload.error || "预测失败");
    }
    currentData.prediction = payload.prediction || "";
    saveLastFetch(currentData);
    setPredictionBox(currentData.prediction);
  } catch (error) {
    setPredictionBox(error.message || "预测失败", { error: true });
  }
}

function bindSummaryButtons() {
  document.querySelectorAll(".summary-trigger").forEach((button) => {
    button.addEventListener("click", (event) => {
      event.preventDefault();
      event.stopPropagation();
      const section = button.dataset.summarySection;
      if (!section) {
        return;
      }
      fetchSectionSummary(section);
    });
  });
  document.querySelectorAll("[data-content-section]").forEach((button) => {
    button.addEventListener("click", async (event) => {
      event.preventDefault();
      event.stopPropagation();
      const section = button.dataset.contentSection;
      if (section !== "serp") {
        return;
      }
      try {
        await fetchSerpContent();
      } catch (error) {
        setSummaryBox("serp", error.message || "SerpAPI 正文抓取失败", { error: true });
      }
    });
  });
  document.querySelectorAll("[data-filter-section]").forEach((button) => {
    button.addEventListener("click", async (event) => {
      event.preventDefault();
      event.stopPropagation();
      const section = button.dataset.filterSection;
      if (section !== "serp") {
        return;
      }
      try {
        await fetchSerpFilter();
      } catch (error) {
        setSummaryBox("serp", error.message || "SerpAPI 结果过滤失败", { error: true });
      }
    });
  });
  const predictButton = document.getElementById("predict-button");
  if (predictButton) {
    predictButton.addEventListener("click", async () => {
      await fetchPrediction();
    });
  }
}

async function fetchSymbol(symbol) {
  const status = document.getElementById("status");
  status.className = "status loading";
  status.textContent = `正在实时抓取 ${symbol} 的 comments、news、reports、K线和 SerpAPI 补充搜索结果...`;
  try {
    const resp = await fetch(`/api/stock/${encodeURIComponent(symbol)}`);
    const payload = await resp.json();
    if (!resp.ok || !payload.ok) {
      throw new Error(payload.error || "抓取失败");
    }
    renderResult(payload.data);
    saveLastFetch(payload.data);
    status.className = "status success";
    status.textContent = `抓取完成：${payload.data.name} ${payload.data.stock}`;
  } catch (error) {
    status.className = "status error";
    status.textContent = error.message || "抓取失败";
  }
}

document.getElementById("fetch-form").addEventListener("submit", (event) => {
  event.preventDefault();
  const symbol = document.getElementById("symbol-input").value.trim();
  if (!symbol) {
    return;
  }
  fetchSymbol(symbol);
});

document.querySelectorAll(".chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    const symbol = chip.dataset.symbol;
    document.getElementById("symbol-input").value = symbol;
    fetchSymbol(symbol);
  });
});

if (window.__INITIAL_DATA__) {
  renderResult(window.__INITIAL_DATA__);
  saveLastFetch(window.__INITIAL_DATA__);
} else {
  const cachedData = loadLastFetch();
  if (cachedData) {
    renderResult(cachedData);
    const input = document.getElementById("symbol-input");
    if (input && cachedData.stock) {
      input.value = cachedData.stock;
    }
    const status = document.getElementById("status");
    status.className = "status success";
    status.textContent = `已加载上次缓存结果：${cachedData.name || ""} ${cachedData.stock || ""}`.trim();
  }
}
