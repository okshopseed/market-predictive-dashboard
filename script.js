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
    const dateObj = new Date(data.last_updated);
    document.getElementById('last-updated').innerText = `Last Updated: ${dateObj.toLocaleString()}`;

    // 2. Render Predictions
    const grid = document.getElementById('predictions-grid');
    grid.innerHTML = ''; // clear
    for (const [symbol, pct] of Object.entries(data.today_predictions)) {
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
                AI predicts market will go ${isUp ? 'UP' : 'DOWN'}.
            </p>
        `;
        grid.appendChild(card);
    }

    // 3. Render Evaluation
    const evalContainer = document.getElementById('evaluation-content');
    if (!data.evaluation || !data.evaluation.last_evaluated_date) {
        evalContainer.innerHTML = `<p>No previous evaluation data available yet.</p>`;
    } else {
        let html = `<p style="margin-bottom: 1rem; color: var(--text-muted);">Predictions made on: <strong>${data.evaluation.last_evaluated_date}</strong></p>`;
        for (const [symbol, res] of Object.entries(data.evaluation.results)) {
            const statusColor = res.correct ? 'var(--accent-up)' : 'var(--accent-down)';
            const statusText = res.correct ? '✅ CORRECT' : '❌ INCORRECT';
            
            html += `
                <div class="eval-item">
                    <strong style="font-size: 1.1rem;">${symbol}</strong>
                    <span style="float: right; color: ${statusColor}; font-weight: bold;">${statusText}</span>
                    <br>
                    <span style="font-size: 0.9rem; color: var(--text-muted);">
                        Predicted: ${res.predicted_dir} | Actual: ${res.actual_dir} (${(res.actual_percent * 100).toFixed(2)}%)
                    </span>
                </div>
            `;
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
