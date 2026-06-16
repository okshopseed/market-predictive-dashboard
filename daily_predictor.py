import yfinance as yf
import pandas as pd
import numpy as np
import feedparser
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA
import json
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

DATA_FILE = "prediction_history.json"
DASHBOARD_FILE = "dashboard_data.json"
SYMBOLS = {
    "S&P 500": "^GSPC",
    "Nasdaq": "^IXIC",
    "Gold": "GC=F"
}

def fetch_data(symbol, period="5y"):
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period)
    if df.empty:
        return None
    df['Return'] = df['Close'].pct_change()
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['Volatility'] = df['Return'].rolling(window=20).std()
    # Target: next trading day's return
    df['Target_Return'] = df['Return'].shift(-1)
    df = df.dropna()
    return df

def train_predict_rf(df):
    features = ['Return', 'SMA_10', 'SMA_50', 'Volatility']
    X = df[features].values
    y = df['Target_Return'].values
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    last_row = df[features].iloc[-1].values.reshape(1, -1)
    return model.predict(last_row)[0]

def train_predict_arima(df):
    y = df['Close'].values
    try:
        model = ARIMA(y, order=(5,1,0))
        model_fit = model.fit()
        pred_close = model_fit.forecast(steps=1)[0]
        last_close = y[-1]
        return (pred_close - last_close) / last_close
    except:
        return 0.0

def fetch_news():
    feeds = [
        "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664",
        "https://feeds.a.dj.com/rss/RSSMarketsMain.xml"
    ]
    news = []
    for f in feeds:
        try:
            parsed = feedparser.parse(f)
            for entry in parsed.entries[:5]:
                news.append(entry.title)
        except Exception as e:
            print(f"Error fetching news: {e}")
    return news

def load_history():
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
    return {}

def save_history(history):
    with open(DATA_FILE, 'w') as f:
        json.dump(history, f, indent=4)

def main():
    print("========================================")
    print("🚀 DAILY MARKET PREDICTIVE SYSTEM 🚀")
    print("========================================\n")

    # Script runs at 23:00 UTC (06:00 ICT). US markets closed 3h ago at 20:00 UTC.
    # today_str (UTC) = the trading day whose results are now available.
    # tomorrow_str     = the next trading day we are predicting FOR.
    today_str    = datetime.utcnow().strftime("%Y-%m-%d")
    tomorrow_str = (datetime.utcnow() + timedelta(days=1)).strftime("%Y-%m-%d")

    history = load_history()

    # ──────────────────────────────────────────────────────────────
    # STEP 1 — Check accuracy of the previous prediction FOR today
    # Predictions are stored under the date they were FOR (not made on).
    # So we look for history[today_str] (stored yesterday, for today).
    # If missing (weekend gap), scan back up to 3 days.
    # ──────────────────────────────────────────────────────────────
    print("--- 📊 Accuracy Check: Was Yesterday's Prediction Correct? ---")

    eval_target_date = None
    for i in range(0, 4):
        check_date = (datetime.utcnow() - timedelta(days=i)).strftime("%Y-%m-%d")
        entry = history.get(check_date, {})
        if entry and not entry.get("evaluated"):
            eval_target_date = check_date
            break

    eval_results = {}
    if eval_target_date:
        prev_preds = history[eval_target_date]
        made_on = prev_preds.get("made_on", "unknown")
        print(f"Prediction was made on: {made_on}  |  Was for: {eval_target_date}")
        for name, sym in SYMBOLS.items():
            df = yf.Ticker(sym).history(period="5d")
            if not df.empty and len(df) >= 2:
                # Most recent close = today's US session (closed 3h ago)
                actual_return = (df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]
                pred = prev_preds.get(name, 0)
                actual_dir = "Up" if actual_return > 0 else "Down"
                pred_dir   = "Up" if pred > 0 else "Down"
                correct    = actual_dir == pred_dir
                print(f"[{name}] Predicted: {pred*100:.2f}% ({pred_dir}) | Actual: {actual_return*100:.2f}% ({actual_dir}) | {'✅ CORRECT' if correct else '❌ WRONG'}")
                eval_results[name] = {
                    "predicted_percent": pred,
                    "actual_percent":    float(actual_return),
                    "predicted_dir":     pred_dir,
                    "actual_dir":        actual_dir,
                    "correct":           correct
                }
        history[eval_target_date]["evaluated"] = True
        history[eval_target_date]["eval_date"] = today_str
    else:
        print("No unevaluated prediction found for today or recent past.")

    # ──────────────────────────────────────────────────────────────
    # STEP 2 — Predict for TOMORROW (next trading day)
    # Store the prediction under tomorrow_str so the next run can
    # find it by looking for history[their_today_str].
    # ──────────────────────────────────────────────────────────────
    print(f"\n--- 🔮 Prediction for TOMORROW ({tomorrow_str}) ---")

    tomorrow_preds = {}
    for name, sym in SYMBOLS.items():
        try:
            df = fetch_data(sym)
            if df is not None:
                pred_rf    = train_predict_rf(df)
                pred_arima = train_predict_arima(df)
                ensemble   = (pred_rf + pred_arima) / 2
                tomorrow_preds[name] = float(ensemble)
                direction  = "Up 📈" if ensemble > 0 else "Down 📉"
                print(f"[{name}] Direction: {direction} | Expected Change: {ensemble*100:.2f}%")
        except Exception as e:
            print(f"[{name}] Error predicting: {e}")

    # Store under tomorrow_str — the date this prediction is FOR
    history[tomorrow_str] = {
        **tomorrow_preds,
        "made_on":  today_str,
        "for_date": tomorrow_str,
        "evaluated": False
    }
    save_history(history)

    # ──────────────────────────────────────────────────────────────
    # STEP 3 — Fetch News
    # ──────────────────────────────────────────────────────────────
    print("\n--- 📰 Latest Market News ---")
    news = fetch_news()
    for i, n in enumerate(news, 1):
        print(f"{i}. {n}")
    if not news:
        print("No news fetched.")

    # ──────────────────────────────────────────────────────────────
    # STEP 4 — Write dashboard_data.json
    # ──────────────────────────────────────────────────────────────
    dashboard_data = {
        "last_updated":        datetime.utcnow().isoformat(),
        "prediction_for_date": tomorrow_str,
        "tomorrow_predictions": tomorrow_preds,
        "evaluation": {
            "prediction_was_for": eval_target_date,
            "made_on":            history.get(eval_target_date, {}).get("made_on") if eval_target_date else None,
            "results":            eval_results
        },
        "news": news
    }
    with open(DASHBOARD_FILE, 'w') as f:
        json.dump(dashboard_data, f, indent=4)

if __name__ == "__main__":
    main()
