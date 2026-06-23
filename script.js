// ── จัดกลุ่มสัญลักษณ์ ──────────────────────────────────────────────────────
const SYMBOL_GROUPS = {
    "ดัชนีตลาด":              ["S&P 500", "Nasdaq"],
    "กองทุน ETF":             ["IVV", "SMH"],
    "หุ้นเทคโนโลยี":         ["Google", "NVDA", "AMD", "TSM", "MU", "WDC", "TSLA", "RKLB"],
    "สินค้าโภคภัณฑ์ / คริปโต": ["Gold", "Bitcoin"],
    "หุ้นไทย":                ["SCB", "TQM"],
};

// ── ป้ายอธิบายศัพท์ ─────────────────────────────────────────────────────────
const TERM_TIPS = {
    "RF":       "Random Forest — ML ที่ใช้ต้นไม้ตัดสินใจ 100 ต้น",
    "ARIMA":    "ARIMA — โมเดลสถิติ Time Series แบบคลาสสิก",
    "ข่าว":     "News Sentiment — วิเคราะห์อารมณ์ข่าวจากแหล่ง ≥80% น่าเชื่อถือ",
    "ML ราคา":  "Machine Learning จากราคา (logistic/RF/GB/stacking) ผสมแบบปรับน้ำหนักตามฟอร์มล่าสุด",
    "Ensemble": "ผลรวมถ่วงน้ำหนักจากทุกแขน ปรับตามผลถูก/ผิดที่ผ่านมา (recency-aware)",
};

const CONFIDENCE_INFO = {
    high: {
        label: "สูง",
        short: "หลายแขนเห็นตรงกัน และสัญญาณค่อนข้างชัด",
        tooltip: "สูง: ตั้งแต่ 3 แขนขึ้นไป (RF/ARIMA/ข่าว/ML ราคา) ทายทางเดียวกัน และสัญญาณแรง",
    },
    medium: {
        label: "กลาง",
        short: "แขนส่วนใหญ่เห็นตรงกัน แต่สัญญาณยังไม่แรงมาก",
        tooltip: "กลาง: อย่างน้อย 2 แขนทายตรงกัน แต่ % ทำนายยังไม่แรงพอ",
    },
    low: {
        label: "ต่ำ",
        short: "แขนทำนายยังขัดแย้งกัน จึงควรใช้ข้อมูลอื่นประกอบ",
        tooltip: "ต่ำ: แขนทำนายขัดแย้งกัน สัญญาณยังไม่มั่นคง",
    },
};

document.addEventListener("DOMContentLoaded", () => {
    document.querySelectorAll("[data-dashboard-tab]").forEach(button => {
        button.addEventListener("click", () => setDashboardView(button.dataset.dashboardTab));
    });
    fetchData();
});

// ── เปิด/ปิด dropdown รายละเอียดถูก-ผิดของแต่ละวัน ────────────────────────────
function toggleDayDetail(rowId, triggerRow) {
    const detail = document.getElementById(rowId);
    if (!detail) return;
    const open = detail.style.display === "none";
    detail.style.display = open ? "table-row" : "none";
    const caret = triggerRow.querySelector(".row-caret");
    if (caret) caret.textContent = open ? "▾" : "▸";
    triggerRow.classList.toggle("summary-row-open", open);
}
window.toggleDayDetail = toggleDayDetail;

async function fetchData() {
    try {
        const res = await fetch("dashboard_data.json");
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const dashboardData = await res.json();
        renderDashboard(dashboardData);
        renderBacktest(await fetchBacktestData(), dashboardData.model_validation);
    } catch (err) {
        console.error("โหลดข้อมูลไม่ได้:", err);
        document.getElementById("last-updated").innerText =
            "⚠️ โหลดข้อมูลไม่สำเร็จ — กรุณารอให้ script ทำงานก่อน";
    }
}

async function fetchBacktestData() {
    try {
        const res = await fetch("backtest_data.json");
        return res.ok ? await res.json() : null;
    } catch (_) {
        return null;
    }
}

function setDashboardView(view) {
    document.querySelectorAll("[data-dashboard-tab]").forEach(button => {
        const active = button.dataset.dashboardTab === view;
        button.classList.toggle("is-active", active);
        button.setAttribute("aria-selected", String(active));
    });
    document.getElementById("today-view").hidden = view !== "today";
    document.getElementById("backtest-view").hidden = view !== "backtest";
}

// ── Render ───────────────────────────────────────────────────────────────────
function renderDashboard(data) {
    // Header
    const d = new Date(data.last_updated);
    document.getElementById("last-updated").innerText =
        `อัปเดตล่าสุด: ${d.toLocaleString("th-TH")}`;

    if (data.prediction_for_date) {
        document.getElementById("prediction-for-date").innerText =
            `(${data.prediction_for_date})`;
        document.getElementById("hero-date").innerText = data.prediction_for_date;
    }

    const preds   = data.tomorrow_predictions || {};
    const details = data.tomorrow_details     || {};
    const weights = data.model_weights        || {};

    // Hero summary
    renderHero(preds, data.stats);

    // Prediction cards (grouped)
    renderPredictions(preds, details, weights);

    // Accuracy evaluation
    renderEvaluation(data.evaluation);

    // Cumulative stats
    renderStats(data.stats);

    // News section
    renderNews(data.news, data.news_fetch_stats);
}

// ── Hero summary ─────────────────────────────────────────────────────────────
function renderHero(preds, stats) {
    let upCount = 0, downCount = 0;
    for (const pct of Object.values(preds)) {
        pct > 0 ? upCount++ : downCount++;
    }
    document.getElementById("hero-up").innerText   = `📈 ${upCount} ตัว`;
    document.getElementById("hero-down").innerText = `📉 ${downCount} ตัว`;

    const acc    = stats?.overall_accuracy_pct;
    const streak = stats?.all_correct_streak || 0;
    const accEl  = document.getElementById("hero-accuracy");
    accEl.innerText = acc !== null && acc !== undefined ? `${acc}%` : "ยังไม่มีข้อมูล";
    if (acc >= 60)      accEl.classList.add("up");
    else if (acc < 40)  accEl.classList.add("down");

    const streakEl = document.getElementById("hero-streak");
    streakEl.innerText = streak > 0 ? `${streak} วัน 🔥` : "0 วัน";
    if (streak >= 3) streakEl.classList.add("up");
}

// ── Prediction cards ─────────────────────────────────────────────────────────
function renderPredictions(preds, details, weights) {
    const container = document.getElementById("predictions-container");
    container.innerHTML = "";

    for (const [groupName, symbols] of Object.entries(SYMBOL_GROUPS)) {
        const groupSymbols = symbols.filter(s => s in preds);
        if (groupSymbols.length === 0) continue;

        // กลุ่ม header
        const groupHeader = document.createElement("div");
        groupHeader.className = "group-header";
        groupHeader.innerHTML = `<span>${groupName}</span>`;
        container.appendChild(groupHeader);

        const grid = document.createElement("div");
        grid.className = "card-grid";

        for (const symbol of groupSymbols) {
            const pct    = preds[symbol];
            const detail = details[symbol] || {};
            const w      = weights[symbol] || {};
            grid.appendChild(buildPredCard(symbol, pct, detail, w));
        }
        container.appendChild(grid);
    }
}

function buildPredCard(symbol, pct, detail, w) {
    const isUp        = pct > 0;
    const sign        = isUp ? "+" : "";
    const dirClass    = isUp ? "up" : "down";
    const icon        = isUp ? "📈" : "📉";
    const dirText     = isUp ? "ขึ้น" : "ลง";
    const confidence  = getConfidence(detail);
    const confInfo    = CONFIDENCE_INFO[confidence];
    const confLabel   = confInfo.label;
    const confClass   = `confidence-${confidence}`;

    // News indicator
    const newsInfo    = detail.news_info || {};
    const newsDir     = newsInfo.news_direction || newsInfo.direction || "Neutral";
    const newsCount   = newsInfo.article_count || 0;
    const newsIcon    = newsDir === "Up" ? "📈" : (newsDir === "Down" ? "📉" : "➖");
    const newsText    = newsDir === "Up" ? "บวก" : (newsDir === "Down" ? "ลบ" : "เป็นกลาง");
    const newsHtml    = newsCount > 0
        ? `<div class="news-indicator">
               <span class="news-dot news-dot--${newsDir.toLowerCase()}"></span>
               ข่าวประกอบ: ${newsIcon} ${newsText}
               <span class="news-count">${newsCount} ข่าว</span>
           </div>`
        : `<div class="news-indicator news-indicator--none">➖ ไม่มีข่าวที่เกี่ยวข้อง</div>`;

    // Adaptive weight bar (4 แขน: RF, ARIMA, ข่าว, ML ราคา)
    const rfW    = Math.round((w.rf       || 0) * 100);
    const arW    = Math.round((w.arima    || 0) * 100);
    const nwW    = Math.round((w.news     || 0) * 100);
    const mlW    = Math.round((w.price_ml || 0) * 100);
    const sampls = w.samples || 0;
    const weightHtml = `
        <div class="weight-section">
            <div class="weight-bar">
                <div class="weight-segment weight-rf"       style="width:${rfW}%" title="RF ${rfW}%"></div>
                <div class="weight-segment weight-arima"    style="width:${arW}%" title="ARIMA ${arW}%"></div>
                <div class="weight-segment weight-news"     style="width:${nwW}%" title="ข่าว ${nwW}%"></div>
                <div class="weight-segment weight-price-ml" style="width:${mlW}%" title="ML ราคา ${mlW}%"></div>
            </div>
            <div class="weight-labels">
                <span class="wlabel wlabel-rf"        data-tooltip="${TERM_TIPS['RF']}">RF ${rfW}%</span>
                <span class="wlabel wlabel-arima"     data-tooltip="${TERM_TIPS['ARIMA']}">ARIMA ${arW}%</span>
                <span class="wlabel wlabel-news"      data-tooltip="${TERM_TIPS['ข่าว']}">ข่าว ${nwW}%</span>
                <span class="wlabel wlabel-price-ml"  data-tooltip="${TERM_TIPS['ML ราคา'] || 'โมเดล Machine Learning จากราคา (ผสมหลายตัวแบบปรับน้ำหนักตามฟอร์ม)'}">ML ราคา ${mlW}%</span>
                <span class="wlabel wlabel-samples">${sampls} วัน</span>
            </div>
        </div>`;

    const card = document.createElement("div");
    card.className = "card glass-card pred-card";
    card.innerHTML = `
        <div class="card-top">
            <span class="symbol-name">${symbol}</span>
            <span class="confidence-badge ${confClass}" tabindex="0" data-tooltip="${confInfo.tooltip}">ความเชื่อมั่น: ${confLabel}</span>
        </div>
        <div class="prediction-value ${dirClass}">
            ${icon} ${sign}${(pct * 100).toFixed(2)}%
        </div>
        <p class="pred-label">AI คาดว่าราคาจะ <strong>${dirText}</strong> ในวันทำนาย</p>
        <div class="confidence-note ${confClass}">
            <strong>${confLabel}</strong>
            <span>${confInfo.short}</span>
        </div>
        ${newsHtml}
        ${weightHtml}
    `;
    return card;
}

function getConfidence(detail) {
    if (!detail) return "low";
    // ความเชื่อมั่นจากความสอดคล้องของแขนทำนาย (RF, ARIMA, ข่าว, ML ราคา)
    const rfDir    = detail.rf_pct       ? (detail.rf_pct       > 0 ? 1 : -1) : 0;
    const arimaDir = detail.arima_pct    ? (detail.arima_pct    > 0 ? 1 : -1) : 0;
    const newsDir  = detail.news_pct     ? (detail.news_pct     > 0 ? 1 : -1) : 0;
    const mlDir    = detail.price_ml_pct ? (detail.price_ml_pct > 0 ? 1 : -1) : 0;
    const votes    = [rfDir, arimaDir, newsDir, mlDir].filter(d => d !== 0);
    if (votes.length === 0) return "low";
    const pos      = votes.filter(d => d > 0).length;
    const neg      = votes.filter(d => d < 0).length;
    const allAgree = pos === votes.length || neg === votes.length;

    // ถ้ามีสัญญาณ ML ราคา ใช้ระยะห่างความน่าจะเป็นเสริม
    const probability = Number(detail.probability_up);
    const probStrong  = Number.isFinite(probability) && Math.abs(probability - 0.5) >= 0.15;

    if (allAgree && votes.length >= 3 && (probStrong || Math.abs(detail.predicted_pct) > 0.003)) return "high";
    if (allAgree && votes.length >= 2) return "medium";
    return "low";
}

// ── Evaluation ───────────────────────────────────────────────────────────────
function renderEvaluation(evaluation) {
    const el = document.getElementById("evaluation-content");
    if (!evaluation?.results || Object.keys(evaluation.results).length === 0) {
        el.innerHTML = `<p class="muted-text">ยังไม่มีข้อมูลประเมิน — จะปรากฏหลังจากครบ 1 วงรอบ (ทำนาย → เช็คผลวันถัดไป)</p>`;
        return;
    }

    const { prediction_was_for, made_on, results } = evaluation;
    let correct = 0, total = 0;
    for (const r of Object.values(results)) { total++; if (r.correct) correct++; }

    let html = `
        <div class="eval-header">
            <span>ทำนายเมื่อ <strong>${made_on}</strong> สำหรับวันที่ <strong>${prediction_was_for}</strong></span>
            <span class="eval-score">${correct}/${total} ถูก</span>
        </div>
        <div class="eval-grid">`;

    for (const [symbol, res] of Object.entries(results)) {
        const ok        = res.correct;
        const predDir   = res.predicted_dir === "Up" ? "ขึ้น 📈" : "ลง 📉";
        const actDir    = res.actual_dir    === "Up" ? "ขึ้น 📈" : "ลง 📉";
        const predPct   = (res.predicted_pct * 100).toFixed(2);
        const actPct    = (res.actual_pct    * 100).toFixed(2);
        html += `
            <div class="eval-item ${ok ? "eval-correct" : "eval-wrong"}">
                <div class="eval-item-top">
                    <strong>${symbol}</strong>
                    <span class="eval-badge">${ok ? "✅ ถูก" : "❌ ผิด"}</span>
                </div>
                <div class="eval-detail">
                    <span>ทำนาย: ${predDir} (${predPct}%)</span>
                    <span>จริง: ${actDir} (${actPct}%)</span>
                </div>
            </div>`;
    }
    html += `</div>`;
    el.innerHTML = html;
}

// ── Stats ─────────────────────────────────────────────────────────────────────
function renderStats(stats) {
    if (!stats) return;

    const grid        = document.getElementById("stats-grid");
    const overallPct  = stats.overall_accuracy_pct;
    const totalDays   = stats.total_evaluated || 0;
    const streak      = stats.all_correct_streak || 0;

    grid.innerHTML = "";

    const baselines = stats.baselines || {};
    const alwaysUp  = baselines.always_up_pct;
    const momentum  = baselines.momentum_pct;
    const beatBase  = (a, b) => (a !== null && a !== undefined && b !== null && b !== undefined && a > b);

    // Summary cards
    [
        {
            label: "ความแม่นยำรวม (ระบบ)",
            value: overallPct !== null && overallPct !== undefined ? `${overallPct}%` : "N/A",
            sub:   `${totalDays} วัน ที่ประเมินแล้ว`,
            color: overallPct >= 60 ? "var(--accent-up)" : overallPct >= 40 ? "#f0c040" : "var(--accent-down)",
        },
        {
            label: "Baseline: ทายขึ้นตลอด",
            value: alwaysUp !== null && alwaysUp !== undefined ? `${alwaysUp}%` : "N/A",
            sub:   beatBase(overallPct, alwaysUp) ? "✅ ระบบเก่งกว่า" : "เกณฑ์เทียบขั้นต่ำ",
            color: "var(--text-muted)",
        },
        {
            label: "Baseline: Momentum",
            value: momentum !== null && momentum !== undefined ? `${momentum}%` : "N/A",
            sub:   beatBase(overallPct, momentum) ? "✅ ระบบเก่งกว่า" : "ทายตามทิศทางวันก่อน",
            color: "var(--text-muted)",
        },
        {
            label: "ทายถูกติดต่อกัน",
            value: streak > 0 ? `${streak} 🔥` : "0",
            sub:   "วันที่ทายถูกทุก symbol",
            color: streak >= 3 ? "var(--accent-up)" : "var(--text-muted)",
        },
    ].forEach(c => {
        const card = document.createElement("div");
        card.className = "card glass-card";
        card.innerHTML = `
            <div class="symbol-name">${c.label}</div>
            <div class="prediction-value" style="color:${c.color}; font-size:2rem;">${c.value}</div>
            <p class="muted-text" style="margin-top:0.5rem;">${c.sub}</p>`;
        grid.appendChild(card);
    });

    // Per-symbol cards
    if (stats.per_symbol) {
        for (const [symbol, s] of Object.entries(stats.per_symbol)) {
            const pct   = s.accuracy_pct;
            const color = pct >= 60 ? "var(--accent-up)" : pct >= 40 ? "#f0c040" : "var(--accent-down)";
            const card  = document.createElement("div");
            card.className = "card glass-card";
            card.innerHTML = `
                <div class="symbol-name">${symbol}</div>
                <div class="prediction-value" style="color:${color}; font-size:2rem;">
                    ${pct !== null && pct !== undefined ? pct + "%" : "N/A"}
                </div>
                <p class="muted-text" style="margin-top:0.5rem;">${s.correct} / ${s.total} ถูก</p>`;
            grid.appendChild(card);
        }
    }

    // History table
    const tableEl = document.getElementById("history-table-container");
    const history = stats.recent_history;
    if (!history || history.length === 0) {
        tableEl.innerHTML = `<p class="muted-text" style="padding:1rem;">ยังไม่มีประวัติ — จะปรากฏหลังครบ 1 วงรอบ</p>`;
        return;
    }

    const symbols    = Object.keys(history[history.length - 1].symbols || {});
    const headerCols = symbols.map(s => `<th>${s}</th>`).join("");

    // ── ตารางสรุปผลรายวัน: การ Predictive ถูก/ผิด แต่ละวัน ──────────────────
    let summaryRows = "";
    let dayIdx = 0;
    for (const row of [...history].reverse()) {
        const entries = Object.entries(row.symbols || {});
        const total = entries.length;
        const correct = entries.filter(([, d]) => d && d.correct).length;
        const wrong = total - correct;
        if (total === 0) continue;
        const pct = Math.round((correct / total) * 100);

        let verdict, vClass;
        if (pct >= 70)      { verdict = "แม่นยำ";      vClass = "verdict-good"; }
        else if (pct >= 50) { verdict = "พอใช้";       vClass = "verdict-mid";  }
        else                { verdict = "ต้องปรับปรุง"; vClass = "verdict-bad";  }

        const chip = (sym, d) => {
            const dir = d.actual_dir === "Up" ? "📈" : "📉";
            const actualPct = Number.isFinite(d.actual_pct) ? (d.actual_pct * 100).toFixed(2) : "N/A";
            const pdir = d.predicted_dir === "Up" ? "ขึ้น" : "ลง";
            return `<span class="result-chip ${d.correct ? "chip-correct" : "chip-wrong"}"
                        data-tooltip="ทำนาย: ${pdir} · จริง: ${dir} ${actualPct}%">
                        ${d.correct ? "✅" : "❌"} ${sym}
                    </span>`;
        };
        const correctChips = entries.filter(([, d]) => d && d.correct).map(([s, d]) => chip(s, d)).join("");
        const wrongChips = entries.filter(([, d]) => d && !d.correct).map(([s, d]) => chip(s, d)).join("");

        const rowId = `day-detail-${dayIdx++}`;
        summaryRows += `
            <tr class="summary-row" onclick="toggleDayDetail('${rowId}', this)">
                <td class="date-cell"><span class="row-caret">▸</span> ${row.for_date}</td>
                <td class="summary-bar-cell">
                    <div class="summary-bar">
                        <div class="summary-bar-correct" style="width:${pct}%"></div>
                    </div>
                </td>
                <td class="summary-num summary-correct">✅ ${correct}</td>
                <td class="summary-num summary-wrong">❌ ${wrong}</td>
                <td class="summary-num">${correct}/${total}</td>
                <td class="summary-pct">${pct}%</td>
                <td><span class="verdict-badge ${vClass}">${verdict}</span></td>
            </tr>
            <tr id="${rowId}" class="day-detail-row" style="display:none;">
                <td colspan="7">
                    <div class="day-detail">
                        ${correctChips ? `<div class="detail-group">
                            <span class="detail-label detail-label-correct">ทายถูก (${correct})</span>
                            <div class="chip-wrap">${correctChips}</div>
                        </div>` : ""}
                        ${wrongChips ? `<div class="detail-group">
                            <span class="detail-label detail-label-wrong">ทายผิด (${wrong})</span>
                            <div class="chip-wrap">${wrongChips}</div>
                        </div>` : ""}
                    </div>
                </td>
            </tr>`;
    }

    let rows = "";
    for (const row of [...history].reverse()) {
        let cols = "";
        for (const sym of symbols) {
            const d = row.symbols[sym];
            if (!d) { cols += "<td>—</td>"; continue; }
            const ok  = d.correct;
            const dir = d.actual_dir === "Up" ? "📈" : "📉";
            const pct = (d.actual_pct * 100).toFixed(2);
            cols += `<td class="history-cell ${ok ? "cell-correct" : "cell-wrong"}">
                ${ok ? "✅" : "❌"} ${dir} ${pct}%<br>
                <span class="muted-text" style="font-size:0.7rem;">ทำนาย: ${d.predicted_dir === "Up" ? "ขึ้น" : "ลง"}</span>
            </td>`;
        }
        rows += `<tr><td class="date-cell">${row.for_date}</td>${cols}</tr>`;
    }

    tableEl.innerHTML = `
        <h3 class="table-subtitle">📋 สรุปผลการทำนายรายวัน</h3>
        <p class="muted-text" style="margin-bottom:0.75rem; font-size:0.85rem;">นับรวมทุก symbol ในแต่ละวันว่าทายถูกกี่ตัว ผิดกี่ตัว (ใหม่ก่อน)</p>
        <div style="overflow-x:auto;">
        <table class="history-table summary-table">
            <thead>
                <tr>
                    <th style="text-align:left;">วันที่</th>
                    <th style="min-width:120px;">ความแม่นยำ</th>
                    <th>ทายถูก</th>
                    <th>ทายผิด</th>
                    <th>รวม</th>
                    <th>%</th>
                    <th>ผลรวม</th>
                </tr>
            </thead>
            <tbody>${summaryRows}</tbody>
        </table>
        </div>

        <h3 class="table-subtitle" style="margin-top:2rem;">🔍 รายละเอียดต่อ symbol</h3>
        <p class="muted-text" style="margin-bottom:0.75rem; font-size:0.85rem;">${history.length} วันล่าสุด (ใหม่ก่อน)</p>
        <div style="overflow-x:auto;">
        <table class="history-table">
            <thead>
                <tr>
                    <th style="text-align:left;">วันที่</th>
                    ${headerCols}
                </tr>
            </thead>
            <tbody>${rows}</tbody>
        </table>
        </div>`;
}

// ── News ─────────────────────────────────────────────────────────────────────
function renderNews(news, fetchStats) {
    // Filter bar
    const filterBar = document.getElementById("news-filter-bar");
    if (fetchStats && (fetchStats.total_fetched > 0 || fetchStats.accepted > 0)) {
        filterBar.style.display = "flex";
        filterBar.innerHTML = `
            <span class="filter-stat">
                📥 ดึงมาทั้งหมด <strong>${fetchStats.total_fetched}</strong> ข่าว
            </span>
            <span class="filter-sep">→</span>
            <span class="filter-stat filter-accepted">
                ✅ ผ่านกรอง ≥${fetchStats.min_credibility || 80}% : <strong>${fetchStats.accepted}</strong> ข่าว
            </span>
            <span class="filter-sep">·</span>
            <span class="filter-stat filter-rejected">
                🗑 ทิ้ง: <strong>${fetchStats.rejected}</strong> ข่าว
            </span>`;
    }

    // News list
    const list = document.getElementById("news-list");
    list.innerHTML = "";

    if (!news || news.length === 0) {
        list.innerHTML = `<li class="muted-text">ไม่มีข่าวในขณะนี้</li>`;
        return;
    }

    for (const item of news) {
        const li   = document.createElement("li");
        // item อาจเป็น string (legacy) หรือ object ใหม่ {title, domain, credibility}
        if (typeof item === "string") {
            li.innerHTML = `<span class="news-title">${item}</span>`;
        } else {
            const credColor = item.credibility >= 95 ? "var(--accent-up)" : "var(--text-muted)";
            li.innerHTML = `
                <span class="news-title">${item.title}</span>
                <span class="news-source" style="color:${credColor};">
                    ${item.domain} · ${item.credibility}/100
                </span>`;
        }
        list.appendChild(li);
    }
}

function formatAccuracy(value) {
    return value === null || value === undefined ? "N/A" : `${value}%`;
}

function renderBacktest(backtest, validation = {}) {
    const container = document.getElementById("backtest-content");
    const priceShadow = validation.price_shadow || {};
    const newsShadow = validation.news_shadow || {};
    if (!backtest?.summary) {
        container.innerHTML = `
            <div class="backtest-empty glass-card">
                <strong>กำลังรอผล Backtest ราคา</strong>
                <p class="muted-text">ระบบจะสร้างผลทดสอบแบบไม่ใช้ข้อมูลอนาคต แล้วแสดงผลจริงในหน้านี้</p>
            </div>`;
        return;
    }

    const threeYear = backtest.summary.three_year || {};
    const recent = backtest.summary.recent_60 || {};
    const ensemble = backtest.summary.adaptive_ensemble || {};
    const ensemble3y = ensemble.three_year || {};
    const ensemble60 = ensemble.recent_60 || {};
    const baselines = backtest.summary.baselines || {};
    const models = backtest.models || {};
    const latestRecords = [...(backtest.records || [])].reverse().slice(0, 120);
    const modelRows = Object.entries(models).map(([symbol, model]) => `
        <tr>
            <td>${symbol}</td>
            <td>${model.champion}</td>
            <td>${formatAccuracy(model.three_year_accuracy_pct)}</td>
            <td>${formatAccuracy(model.recent_accuracy_pct)}</td>
            <td>${model.recent_samples || 0}</td>
        </tr>`).join("");
    const recordRows = latestRecords.map(record => `
        <tr>
            <td>${record.as_of_date || "-"}</td>
            <td>${record.market_date}</td>
            <td>${record.symbol}</td>
            <td>${record.model}</td>
            <td>${record.predicted_dir === "Up" ? "📈 ขึ้น" : "📉 ลง"}</td>
            <td>${record.actual_dir === "Up" ? "📈 ขึ้น" : "📉 ลง"} ${(record.actual_pct * 100).toFixed(2)}%<br><small>${record.actual_close ? record.actual_close.toFixed(2) : "-"}</small></td>
            <td>${record.market_gap_reason ? "ข้ามวันตลาดปิด" : "-"}</td>
            <td class="${record.correct ? "summary-correct" : "summary-wrong"}">${record.correct ? "ถูก" : "ผิด"}</td>
        </tr>`).join("");

    const targetPct = recent.target_accuracy_pct || (validation.target_accuracy_pct ?? 75);
    container.innerHTML = `
        <div class="backtest-metric-grid">
            <article class="glass-card backtest-metric"><span>Ensemble (เรียนรู้-ปรับน้ำหนัก) 3 ปี</span><strong>${formatAccuracy(ensemble3y.accuracy_pct)}</strong><small>${ensemble3y.correct || 0}/${ensemble3y.samples || 0} สัญญาณ · วิธีเดียวกับที่ใช้ทายจริง</small></article>
            <article class="glass-card backtest-metric"><span>Ensemble 60 วันล่าสุด</span><strong>${formatAccuracy(ensemble60.accuracy_pct)}</strong><small>เป้าหมายที่ไล่ตาม ${targetPct}%</small></article>
            <article class="glass-card backtest-metric"><span>Champion รายตัว 3 ปี</span><strong>${formatAccuracy(threeYear.accuracy_pct)}</strong><small>${threeYear.correct || 0}/${threeYear.samples || 0} สัญญาณ</small></article>
        </div>
        <section class="backtest-band">
            <h3>เทียบกับเกณฑ์ขั้นต่ำ (Baseline)${ensemble.params ? ` · ปรับอัตโนมัติ β=${ensemble.params.beta} window=${ensemble.params.window} floor=${ensemble.params.floor}` : ""}</h3>
            <div class="shadow-status-grid">
                <span>ทายขึ้นตลอด: ${formatAccuracy(baselines.always_up_pct)}</span>
                <span>Momentum (ตามวันก่อน): ${formatAccuracy(baselines.momentum_pct)}</span>
                <span>Ensemble 3 ปี: <strong>${formatAccuracy(ensemble3y.accuracy_pct)}</strong></span>
                <span>${(ensemble3y.accuracy_pct ?? 0) > Math.max(baselines.always_up_pct ?? 0, baselines.momentum_pct ?? 0) ? "✅ ระบบเก่งกว่าการเดา" : "ยังไม่ชนะ baseline ชัดเจน"}</span>
            </div>
        </section>
        <section class="backtest-band">
            <h3>ความก้าวหน้าสู่เป้าหมาย ${targetPct}% (ไม่ใช่ประตูปิดกั้น — ทุกโมเดลทายจริงทุกวัน)</h3>
            <div class="shadow-status-grid">
                <span>ราคา: ${formatAccuracy(priceShadow.accuracy_pct)} · ${priceShadow.market_days || 0}/60 วัน</span>
                <span>ราคา + ข่าว: ${formatAccuracy(newsShadow.accuracy_pct)} · ${newsShadow.market_days || 0}/60 วัน</span>
                <span>ครอบคลุมข่าว: ${formatAccuracy(newsShadow.coverage_pct)} · ${newsShadow.days_with_eligible_news || 0}/${newsShadow.days_collected || 0} วัน</span>
                <span>${newsShadow.gap_to_target_pct != null ? `ห่างเป้าหมายอีก ${newsShadow.gap_to_target_pct}%` : "กำลังเก็บข้อมูล"}</span>
            </div>
        </section>
        <section class="backtest-band">
            <h3>Champion รายสินทรัพย์</h3>
            <div class="backtest-table-wrap"><table class="history-table"><thead><tr><th>สินทรัพย์</th><th>โมเดล</th><th>3 ปี</th><th>60 วัน</th><th>ตัวอย่าง</th></tr></thead><tbody>${modelRows}</tbody></table></div>
        </section>
        <section class="backtest-band">
            <h3>บันทึกทำนายเทียบผลจริง</h3>
            <div class="backtest-table-wrap"><table class="history-table"><thead><tr><th>อ้างอิง ณ</th><th>วันที่จริง</th><th>สินทรัพย์</th><th>โมเดล</th><th>ทำนาย</th><th>ผลจริง / ราคาปิด</th><th>วันตลาดปิด</th><th>ผล</th></tr></thead><tbody>${recordRows}</tbody></table></div>
        </section>`;
}
