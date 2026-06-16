document.addEventListener("DOMContentLoaded", () => {
    fetchData();
});

async function fetchData() {
    try {
        // Fetch the local JSON file (this will work on Vercel)
        const response = await fetch('dashboard_data.json');
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        renderDashboard(data);
    } catch (error) {
        console.error("Could not load dashboard data:", error);
        document.getElementById('last-updated').innerText = "Error loading data. Please check if script has run.";
    }
}

function renderDashboard(data) {
    // 1. Update Last Updated
    const dateObj = new Date(data.last_updated + 'Z'); // treat as UTC
    document.getElementById('last-updated').innerText = `Last Updated: ${dateObj.toLocaleString()}`;

    // 2. Show prediction target date
    if (data.prediction_for_date) {
        document.getElementById('prediction-for-date').innerText = `(${data.prediction_for_date})`;
    }

    // 3. Render Tomorrow's Predictions
    const grid = document.getElementById('predictions-grid');
    grid.innerHTML = '';
    const predictions = data.tomorrow_predictions || data.today_predictions || {};
    const details     = data.tomorrow_details || {};
    const weights     = data.model_weights || {};
    for (const [symbol, pct] of Object.entries(predictions)) {
        const isUp = pct > 0;
        const dirClass = isUp ? 'up' : 'down';
        const icon = isUp ? '📈' : '📉';
        const sign = isUp ? '+' : '';
        const formattedPct = (pct * 100).toFixed(2);

        // Show the adaptive weights that produced this forecast
        const w = weights[symbol];
        let weightHtml = '';
        if (w) {
            const rfW = Math.round(w.rf * 100);
            const arW = Math.round(w.arima * 100);
            weightHtml = `
                <div style="margin-top:0.75rem; font-size:0.75rem; color:var(--text-muted);">
                    <div style="display:flex; height:6px; border-radius:3px; overflow:hidden; margin-bottom:0.35rem;">
                        <div style="width:${rfW}%; background:var(--accent-up);"></div>
                        <div style="width:${arW}%; background:#6c8cff;"></div>
                    </div>
                    RF ${rfW}% · ARIMA ${arW}% &nbsp;<span style="opacity:0.6;">(${w.samples} day${w.samples !== 1 ? 's' : ''} learned)</span>
                </div>`;
        }

        const card = document.createElement('div');
        card.className = 'card glass-card';
        card.innerHTML = `
            <div class="symbol-name">${symbol}</div>
            <div class="prediction-value ${dirClass}">
                ${icon} ${sign}${formattedPct}%
            </div>
            <p style="margin-top: 1rem; color: var(--text-muted); font-size: 0.9rem;">
                AI predicts market will go ${isUp ? 'UP' : 'DOWN'} tomorrow.
            </p>
            ${weightHtml}
        `;
        grid.appendChild(card);
    }

    // 4. Render Accuracy Evaluation
    const evalContainer = document.getElementById('evaluation-content');
    const hasResults = data.evaluation && data.evaluation.results && Object.keys(data.evaluation.results).length > 0;
    if (!hasResults) {
        evalContainer.innerHTML = `<p style="color:var(--text-muted);">No evaluation data yet — will appear after the first full cycle (prediction → next day check).</p>`;
    } else {
        const { prediction_was_for, made_on, results } = data.evaluation;
        let html = `
            <p style="margin-bottom: 1rem; color: var(--text-muted);">
                Prediction made on <strong>${made_on}</strong> for <strong>${prediction_was_for}</strong>
            </p>`;
        for (const [symbol, res] of Object.entries(results)) {
            const statusColor = res.correct ? 'var(--accent-up)' : 'var(--accent-down)';
            const statusText  = res.correct ? '✅ CORRECT' : '❌ INCORRECT';
            html += `
                <div class="eval-item">
                    <strong style="font-size: 1.1rem;">${symbol}</strong>
                    <span style="float: right; color: ${statusColor}; font-weight: bold;">${statusText}</span>
                    <br>
                    <span style="font-size: 0.9rem; color: var(--text-muted);">
                        Predicted: ${res.predicted_dir} (${(res.predicted_percent * 100).toFixed(2)}%)
                        | Actual: ${res.actual_dir} (${(res.actual_percent * 100).toFixed(2)}%)
                    </span>
                </div>`;
        }
        evalContainer.innerHTML = html;
    }

    // 5. Render Cumulative Stats
    renderStats(data.stats);

    // 6. Render News
    const newsList = document.getElementById('news-list');
    newsList.innerHTML = '';
    if (data.news && data.news.length > 0) {
        data.news.forEach(newsItem => {
            const li = document.createElement('li');
            li.innerText = newsItem;
            newsList.appendChild(li);
        });
    } else {
        newsList.innerHTML = '<li>No recent news available.</li>';
    }
}

function renderStats(stats) {
    if (!stats) return;

    // ── Summary cards ──
    const grid = document.getElementById('stats-grid');
    grid.innerHTML = '';

    const overallPct = stats.overall_accuracy_pct;
    const totalDays  = stats.total_evaluated || 0;
    const streak     = stats.all_correct_streak || 0;

    const summaryCards = [
        {
            label: 'Overall Accuracy',
            value: overallPct !== null ? `${overallPct}%` : 'N/A',
            sub:   `${totalDays} day${totalDays !== 1 ? 's' : ''} evaluated`,
            color: overallPct >= 60 ? 'var(--accent-up)' : overallPct >= 40 ? '#f0c040' : 'var(--accent-down)',
        },
        {
            label: 'All-Correct Streak',
            value: streak > 0 ? `${streak} 🔥` : '0',
            sub:   'consecutive days all symbols correct',
            color: streak >= 3 ? 'var(--accent-up)' : 'var(--text-muted)',
        },
    ];

    summaryCards.forEach(c => {
        const card = document.createElement('div');
        card.className = 'card glass-card';
        card.innerHTML = `
            <div class="symbol-name">${c.label}</div>
            <div class="prediction-value" style="color:${c.color}; font-size:2rem;">${c.value}</div>
            <p style="margin-top:0.5rem; color:var(--text-muted); font-size:0.85rem;">${c.sub}</p>
        `;
        grid.appendChild(card);
    });

    // Per-symbol cards
    if (stats.per_symbol) {
        for (const [symbol, s] of Object.entries(stats.per_symbol)) {
            const pct   = s.accuracy_pct;
            const color = pct >= 60 ? 'var(--accent-up)' : pct >= 40 ? '#f0c040' : 'var(--accent-down)';
            const card  = document.createElement('div');
            card.className = 'card glass-card';
            card.innerHTML = `
                <div class="symbol-name">${symbol}</div>
                <div class="prediction-value" style="color:${color}; font-size:2rem;">
                    ${pct !== null ? pct + '%' : 'N/A'}
                </div>
                <p style="margin-top:0.5rem; color:var(--text-muted); font-size:0.85rem;">
                    ${s.correct} / ${s.total} correct
                </p>
            `;
            grid.appendChild(card);
        }
    }

    // ── History table ──
    const tableContainer = document.getElementById('history-table-container');
    const history = stats.recent_history;
    if (!history || history.length === 0) {
        tableContainer.innerHTML = '<p style="color:var(--text-muted); padding:1rem;">No history yet — data will appear after the first full prediction cycle.</p>';
        return;
    }

    const symbols = Object.keys(history[history.length - 1].symbols || {});
    let headerCols = symbols.map(s => `<th>${s}</th>`).join('');

    let rows = '';
    for (const row of [...history].reverse()) {
        let cols = '';
        for (const sym of symbols) {
            const d = row.symbols[sym];
            if (!d) { cols += '<td>—</td>'; continue; }
            const icon  = d.correct ? '✅' : '❌';
            const dir   = d.actual_dir === 'Up' ? '📈' : '📉';
            const pct   = (d.actual_pct * 100).toFixed(2);
            cols += `<td style="font-size:0.8rem;">${icon} ${dir} ${pct}%<br><span style="color:var(--text-muted);font-size:0.7rem;">pred: ${d.predicted_dir}</span></td>`;
        }
        rows += `<tr><td style="font-size:0.8rem; white-space:nowrap;">${row.for_date}</td>${cols}</tr>`;
    }

    tableContainer.innerHTML = `
        <p style="margin-bottom:0.75rem; color:var(--text-muted); font-size:0.85rem;">Last ${history.length} evaluated days (newest first)</p>
        <table style="width:100%; border-collapse:collapse; font-size:0.85rem;">
            <thead>
                <tr style="border-bottom:1px solid rgba(255,255,255,0.1);">
                    <th style="text-align:left; padding:0.4rem 0.6rem; color:var(--text-muted);">Date</th>
                    ${headerCols.replace(/<th>/g, '<th style="text-align:center; padding:0.4rem 0.6rem; color:var(--text-muted);">')}
                </tr>
            </thead>
            <tbody>
                ${rows.replace(/<tr>/g, '<tr style="border-bottom:1px solid rgba(255,255,255,0.05);">')}
            </tbody>
        </table>
    `;
}
