function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function renderSection(title, items, type) {
  const cards = (items || []).map((item) => {
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

  return `
    <section class="group">
      <div class="group-head">
        <h2>${title}</h2>
        <span>${items.length} 条</span>
      </div>
      <div class="card-list">${cards || '<p class="empty">暂无结果</p>'}</div>
    </section>
  `;
}

function renderResult(data) {
  const root = document.getElementById("result-root");
  root.innerHTML = `
    <section class="summary">
      <h2>${escapeHtml(data.name)} <span>${escapeHtml(data.stock)}</span></h2>
      <p>市场：${escapeHtml(data.market)}</p>
    </section>
    ${renderSection("最新评论", data.comments || [], "comments")}
    ${renderSection("最新新闻", data.news || [], "news")}
    ${renderSection("最新研报", data.reports || [], "reports")}
  `;
}

async function fetchSymbol(symbol) {
  const status = document.getElementById("status");
  status.className = "status loading";
  status.textContent = `正在实时抓取 ${symbol} 的 comments、news 和 reports...`;
  try {
    const resp = await fetch(`/api/stock/${encodeURIComponent(symbol)}`);
    const payload = await resp.json();
    if (!resp.ok || !payload.ok) {
      throw new Error(payload.error || "抓取失败");
    }
    renderResult(payload.data);
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
}
