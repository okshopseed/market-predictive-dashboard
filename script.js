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
    for (const [symbol, pct] of Object.entries(predictions)) {
        const isUp = pct > 0;
        const dirClass = isUp ? 'up' : 'down';
        const icon = isUp ? '📈' : '📉';
        const sign = isUp ? '+' : '';
        const formattedPct = (pct * 100).toFixed(2);

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

    // 4. Render News
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
