// =========================================================================
// Config
// =========================================================================
const API_BASE = window.location.protocol === "file:" || window.location.port === "5500"
  ? "http://127.0.0.1:8000"
  : window.location.origin;

const LABEL_COLORS = {
  Positive: { bg: "bg-emerald-500", text: "text-emerald-700", soft: "bg-emerald-50", icon: "🟢" },
  Negative: { bg: "bg-red-500", text: "text-red-700", soft: "bg-red-50", icon: "🔴" },
  Neutral: { bg: "bg-amber-500", text: "text-amber-700", soft: "bg-amber-50", icon: "🟡" },
  None: { bg: "bg-gray-400", text: "text-gray-600", soft: "bg-gray-50", icon: "⚪" },
};

const EXAMPLES = [
  { rating: 5, content: "quần kaki chất vải rất tốt, mặc đứng form, đường may chắc chắn" },
  { rating: 1, content: "vải rất xấu, mặc vào nhăn nhúm, đường may bung tung, không đáng tiền chút nào" },
  { rating: 3, content: "form hơi rộng so với size, màu thì ổn, chất vải bình thường, tạm chấp nhận được" },
];

async function fetchWithTimeout(url, options = {}, timeoutMs = 5000) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { ...options, signal: controller.signal });
  } finally {
    clearTimeout(timeoutId);
  }
}

// =========================================================================
// Tabs
// =========================================================================
function setupTabs() {
  const tabs = [
    { btn: "tab-demo",  panel: "panel-demo" },
    { btn: "tab-batch", panel: "panel-batch" },
    { btn: "tab-info",  panel: "panel-info" },
  ];
  tabs.forEach(t => {
    document.getElementById(t.btn).addEventListener("click", () => {
      tabs.forEach(x => {
        document.getElementById(x.panel).classList.toggle("hidden", x.panel !== t.panel);
        const btn = document.getElementById(x.btn);
        const active = (x.btn === t.btn);
        btn.classList.toggle("text-brand", active);
        btn.classList.toggle("border-brand", active);
        btn.classList.toggle("font-medium", active);
        btn.classList.toggle("text-gray-500", !active);
        btn.classList.toggle("border-transparent", !active);
      });
    });
  });
}

// =========================================================================
// Health polling
// =========================================================================
async function pollHealth() {
  const badge = document.getElementById("status-badge");
  try {
    const r = await fetchWithTimeout(`${API_BASE}/health`, {}, 1500);
    const j = await r.json();
    if (j.model_loaded) {
      badge.textContent = "🟢 Model Ready";
      badge.className = "w-fit rounded-full bg-emerald-50 px-3 py-1 text-sm text-emerald-700";
      return true;
    }
    badge.textContent = "🟡 Model loading...";
    badge.className = "w-fit rounded-full bg-amber-50 px-3 py-1 text-sm text-amber-700";
  } catch (e) {
    badge.textContent = "🔴 Backend offline";
    badge.className = "w-fit rounded-full bg-red-50 px-3 py-1 text-sm text-red-700";
  }
  return false;
}

async function healthLoop() {
  while (!(await pollHealth())) {
    await new Promise(r => setTimeout(r, 2000));
  }
}

// =========================================================================
// Demo tab
// =========================================================================
function setupDemo() {
  const ratingInput = document.getElementById("rating-input");
  const starDisplay = document.getElementById("star-display");
  ratingInput.addEventListener("input", () => {
    const n = parseInt(ratingInput.value, 10);
    starDisplay.textContent = "★".repeat(n) + "☆".repeat(5 - n);
  });

  const ul = document.getElementById("example-list");
  EXAMPLES.forEach(ex => {
    const li = document.createElement("li");
    li.className = "cursor-pointer text-brand hover:underline";
    li.textContent = `★${ex.rating} — ${ex.content.slice(0, 60)}…`;
    li.addEventListener("click", () => {
      ratingInput.value = ex.rating;
      ratingInput.dispatchEvent(new Event("input"));
      document.getElementById("content-input").value = ex.content;
    });
    ul.appendChild(li);
  });

  document.getElementById("btn-predict").addEventListener("click", onPredict);
}

async function onPredict() {
  const btn = document.getElementById("btn-predict");
  const rating = parseInt(document.getElementById("rating-input").value, 10);
  const content = document.getElementById("content-input").value.trim();
  if (!content) {
    alert("Vui lòng nhập nội dung review.");
    return;
  }

  btn.disabled = true;
  btn.textContent = "Đang phân tích...";
  const cards = document.getElementById("result-cards");
  cards.innerHTML = '<p class="text-slate-400">⏳ Đang chạy model...</p>';

  try {
    const r = await fetchWithTimeout(`${API_BASE}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ rating, content }),
    });
    if (!r.ok) throw new Error(`HTTP ${r.status}: ${await r.text()}`);
    const data = await r.json();
    renderCards(data.predictions);
    document.getElementById("inference-time").textContent = `⏱️ Inference: ${data.inference_ms}ms`;
  } catch (e) {
    cards.innerHTML = `<p class="text-red-600">Lỗi: ${e.message}</p>`;
    document.getElementById("inference-time").textContent = "";
  } finally {
    btn.disabled = false;
    btn.textContent = "Phân tích →";
  }
}

function renderCards(predictions) {
  const cards = document.getElementById("result-cards");
  cards.innerHTML = "";
  predictions.forEach(p => {
    const c = LABEL_COLORS[p.label];
    const probTooltip = Object.entries(p.probs)
      .map(([k, v]) => `${k}: ${(v * 100).toFixed(1)}%`)
      .join(" | ");
    const card = document.createElement("div");
    card.className = `flex items-center justify-between ${c.soft} rounded-xl p-3`;
    card.title = probTooltip;
    card.innerHTML = `
      <div class="flex min-w-0 flex-1 items-center gap-3">
        <span class="text-xl">${c.icon}</span>
        <div class="min-w-0 flex-1">
          <p class="truncate font-medium">${p.aspect}</p>
          <div class="mt-1 h-2 overflow-hidden rounded-full bg-white">
            <div class="${c.bg} h-2" style="width:${(p.confidence * 100).toFixed(0)}%"></div>
          </div>
        </div>
      </div>
      <div class="ml-3 text-right">
        <p class="${c.text} font-semibold">${p.label}</p>
        <p class="text-xs text-slate-500">${(p.confidence * 100).toFixed(0)}%</p>
      </div>
    `;
    cards.appendChild(card);
  });
}

// =========================================================================
// Batch tab
// =========================================================================
const ASPECT_COLS = ["Chất Liệu", "Kích Cỡ/Form", "Thiết Kế", "Gia Công", "Giá Trị Thực Tế"];

function setupBatch() {
  const fileInput = document.getElementById("file-input");
  const fileLabel = document.getElementById("file-label");
  const btnBatch = document.getElementById("btn-batch");
  const dropZone = fileInput.parentElement;

  ["dragenter", "dragover"].forEach(ev =>
    dropZone.addEventListener(ev, e => {
      e.preventDefault();
      dropZone.classList.add("border-brand");
    })
  );
  ["dragleave", "drop"].forEach(ev =>
    dropZone.addEventListener(ev, e => {
      e.preventDefault();
      dropZone.classList.remove("border-brand");
    })
  );
  dropZone.addEventListener("drop", e => {
    if (e.dataTransfer.files[0]) {
      fileInput.files = e.dataTransfer.files;
      fileInput.dispatchEvent(new Event("change"));
    }
  });

  fileInput.addEventListener("change", () => {
    const f = fileInput.files[0];
    if (!f) return;
    fileLabel.textContent = `✓ ${f.name} (${(f.size / 1024).toFixed(1)} KB)`;
    btnBatch.disabled = false;
  });

  btnBatch.addEventListener("click", onBatchSubmit);
  document.getElementById("filter-search").addEventListener("input", renderBatchTable);
  document.getElementById("filter-label").addEventListener("change", renderBatchTable);
  document.getElementById("filter-aspect").addEventListener("change", renderBatchTable);
  document.getElementById("btn-download").addEventListener("click", downloadBatchCSV);
}

async function onBatchSubmit() {
  const file = document.getElementById("file-input").files[0];
  if (!file) return;

  const btn = document.getElementById("btn-batch");
  const progress = document.getElementById("batch-progress");
  const progressText = document.getElementById("batch-progress-text");
  btn.disabled = true;
  btn.textContent = "Đang xử lý...";
  progressText.textContent = "Đang phân tích...";
  progress.classList.remove("hidden");

  try {
    const fd = new FormData();
    fd.append("file", file);
    const t0 = performance.now();
    const estRows = Math.min(5000, Math.floor(file.size / 250));
    const estSec = Math.round(estRows * 0.13);
    progressText.textContent = `Đang phân tích ~${estRows} dòng — dự kiến ~${estSec}s...`;
    const r = await fetch(`${API_BASE}/predict_batch`, { method: "POST", body: fd });
    if (!r.ok) throw new Error(`HTTP ${r.status}: ${await r.text()}`);
    const data = await r.json();
    const elapsed = ((performance.now() - t0) / 1000).toFixed(1);

    window.BATCH = data;
    document.getElementById("batch-table-wrap").classList.remove("hidden");
    document.getElementById("batch-charts").classList.remove("hidden");
    renderBatchTable();

    if (window.renderBatchCharts) window.renderBatchCharts(data);

    progressText.textContent = `Hoàn thành ${data.summary.total} dòng trong ${elapsed}s.`;
    progress.classList.add("hidden");
  } catch (e) {
    progress.classList.add("hidden");
    alert("Lỗi: " + e.message);
  } finally {
    btn.disabled = false;
    btn.textContent = "Phân tích toàn bộ";
  }
}

function renderBatchTable() {
  if (!window.BATCH) return;
  const tbody = document.getElementById("batch-tbody");
  const search = document.getElementById("filter-search").value.toLowerCase();
  const lblFilter = document.getElementById("filter-label").value;
  const aspFilter = document.getElementById("filter-aspect").value;

  const rows = window.BATCH.rows.filter(row => {
    if (search && !row.content.toLowerCase().includes(search)) return false;
    if (lblFilter === "any-neg") {
      if (!ASPECT_COLS.some(a => row[a] === "Negative")) return false;
    }
    if (lblFilter === "all-neg") {
      const mentioned = ASPECT_COLS.filter(a => row[a] !== "None");
      if (mentioned.length === 0) return false;
      if (!mentioned.every(a => row[a] === "Negative")) return false;
    }
    if (aspFilter !== "all" && row[aspFilter] === "None") return false;
    return true;
  });

  tbody.innerHTML = rows.slice(0, 500).map(row => {
    const cells = ASPECT_COLS.map(a => {
      const lbl = row[a];
      const c = LABEL_COLORS[lbl];
      return `<td class="p-2"><span class="${c.soft} ${c.text} rounded px-2 py-0.5 text-xs">${c.icon} ${lbl}</span></td>`;
    }).join("");
    const star = "★".repeat(row.rating) + "☆".repeat(5 - row.rating);
    const escapedContent = escapeHTML(row.content);
    const contentShort = row.content.length > 100 ? `${escapeHTML(row.content.slice(0, 100))}…` : escapedContent;
    return `
      <tr class="border-t hover:bg-slate-50">
        <td class="whitespace-nowrap p-2 text-amber-500">${star}</td>
        <td class="max-w-md p-2" title="${escapedContent}">${contentShort}</td>
        ${cells}
      </tr>`;
  }).join("");

  document.getElementById("batch-table-summary").textContent =
    `Hiển thị ${Math.min(rows.length, 500)} / ${rows.length} dòng (lọc từ ${window.BATCH.summary.total}).`;
}

function downloadBatchCSV() {
  if (!window.BATCH) return;
  const headers = ["rating", "content", ...ASPECT_COLS.flatMap(a => [a, `${a}_conf`])];
  const lines = [headers.join(",")];
  for (const row of window.BATCH.rows) {
    const values = headers.map(h => {
      const v = row[h] ?? "";
      const s = String(v).replace(/"/g, '""');
      return /[",\n]/.test(s) ? `"${s}"` : s;
    });
    lines.push(values.join(","));
  }
  const blob = new Blob(["\ufeff" + lines.join("\n")], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "absa_results.csv";
  a.click();
  URL.revokeObjectURL(url);
}

function escapeHTML(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

// =========================================================================
// Charts
// =========================================================================
const CHART_COLORS = {
  Positive: "#10b981",
  Negative: "#ef4444",
  Neutral: "#f59e0b",
  None: "#9ca3af",
};

let chartDonut = null;
let chartRating = null;
let chartStacked = null;

window.renderBatchCharts = function (data) {
  drawDonut(data.summary.overall_sentiment);
  drawRating(data.rows);
  drawStacked(data.summary.by_aspect);
  drawHeatmap(data.summary.by_rating_aspect_neg);
  renderInsights(data.summary.insights || []);
};

function drawDonut(overall) {
  const labels = ["Positive", "Negative", "Neutral", "None"];
  const dataArr = labels.map(l => overall[l] || 0);
  const ctx = document.getElementById("chart-donut").getContext("2d");
  if (chartDonut) chartDonut.destroy();
  chartDonut = new Chart(ctx, {
    type: "doughnut",
    data: {
      labels,
      datasets: [{
        data: dataArr,
        backgroundColor: labels.map(l => CHART_COLORS[l]),
        borderWidth: 0,
      }],
    },
    options: {
      responsive: true,
      plugins: { legend: { position: "right" } },
    },
  });
}

function drawRating(rows) {
  const counts = [1, 2, 3, 4, 5].map(r => rows.filter(x => x.rating === r).length);
  const ctx = document.getElementById("chart-rating").getContext("2d");
  if (chartRating) chartRating.destroy();
  chartRating = new Chart(ctx, {
    type: "bar",
    data: {
      labels: ["1★", "2★", "3★", "4★", "5★"],
      datasets: [{ data: counts, backgroundColor: ["#bfdbfe", "#93c5fd", "#60a5fa", "#3b82f6", "#1d4ed8"], borderRadius: 6 }],
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false } },
      scales: { y: { beginAtZero: true, ticks: { precision: 0 } } },
    },
  });
}

function drawStacked(byAspect) {
  const aspects = Object.keys(byAspect);
  const labels = ["Positive", "Negative", "Neutral", "None"];
  const datasets = labels.map(l => ({
    label: l,
    data: aspects.map(a => byAspect[a][l] || 0),
    backgroundColor: CHART_COLORS[l],
  }));
  const ctx = document.getElementById("chart-stacked").getContext("2d");
  if (chartStacked) chartStacked.destroy();
  chartStacked = new Chart(ctx, {
    type: "bar",
    data: { labels: aspects, datasets },
    options: {
      indexAxis: "y",
      responsive: true,
      interaction: { mode: "index", axis: "y", intersect: false },
      plugins: {
        legend: { position: "top" },
        tooltip: {
          mode: "index",
          axis: "y",
          intersect: false,
          callbacks: {
            title: items => aspects[items[0]?.dataIndex] || "",
          },
        },
      },
      scales: { x: { stacked: true }, y: { stacked: true } },
    },
  });
}

function drawHeatmap(byRatingAspectNeg) {
  const aspects = ["Chất Liệu", "Kích Cỡ/Form", "Thiết Kế", "Gia Công", "Giá Trị Thực Tế"];
  const ratings = ["1", "2", "3", "4", "5"];
  const container = document.getElementById("chart-heatmap");

  let html = '<table class="w-full border-separate text-sm" style="border-spacing:4px">';
  html += '<thead><tr><th class="p-2 text-left">Rating ↓ / Khía cạnh →</th>';
  for (const a of aspects) html += `<th class="p-2 text-center text-xs font-medium text-slate-600">${a}</th>`;
  html += "</tr></thead><tbody>";

  function cellColor(rate) {
    if (rate <= 0) return "#f3f4f6";
    if (rate < 0.1) return "#fee2e2";
    if (rate < 0.2) return "#fecaca";
    if (rate < 0.3) return "#fca5a5";
    if (rate < 0.4) return "#f87171";
    if (rate < 0.5) return "#ef4444";
    return "#b91c1c";
  }

  for (const r of ratings) {
    html += `<tr><td class="p-2 font-medium">${r}★</td>`;
    for (const a of aspects) {
      const rate = byRatingAspectNeg[r]?.[a] ?? 0;
      const pct = (rate * 100).toFixed(1);
      const dark = rate > 0.3 ? "text-white" : "text-slate-800";
      html += `<td class="rounded p-3 text-center ${dark}" style="background:${cellColor(rate)}" title="${rate * 100}%">${pct}%</td>`;
    }
    html += "</tr>";
  }
  html += "</tbody></table>";
  html += '<p class="mt-2 text-xs text-slate-500">Tỉ lệ review Negative trong nhóm có cùng rating (số càng cao = càng nhiều phàn nàn).</p>';
  container.innerHTML = html;
}

function renderInsights(insights) {
  const ul = document.getElementById("insight-list");
  if (!insights || insights.length === 0) {
    ul.innerHTML = '<li class="italic text-slate-500">Không phát hiện insight nổi bật.</li>';
    return;
  }
  ul.innerHTML = insights.map(t => {
    const html = escapeHTML(t).replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
    return `<li class="flex gap-2"><span>•</span><span>${html}</span></li>`;
  }).join("");
}

// =========================================================================
// Boot
// =========================================================================
document.addEventListener("DOMContentLoaded", () => {
  setupTabs();
  setupDemo();
  setupBatch();
  healthLoop();
});
