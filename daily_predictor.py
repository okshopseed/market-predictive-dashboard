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
    # Calculate basic technical indicators
    df['Return'] = df['Close'].pct_change()
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['Volatility'] = df['Return'].rolling(window=20).std()
    
    # Target: next day's return
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
    pred_return = model.predict(last_row)[0]
    return pred_return

def train_predict_arima(df):
    y = df['Close'].values
    try:
        model = ARIMA(y, order=(5,1,0))
        model_fit = model.fit()
        pred_close = model_fit.forecast(steps=1)[0]
        last_close = y[-1]
        pred_return = (pred_close - last_close) / last_close
        return pred_return
    except:
        return 0.0

def fetch_news():
    feeds = [
        "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664",
        "https://feeds.a.dj.com/rss/RSSMarketsMain.xml" # WSJ Markets
    ]
    news = []
    for f in feeds:
        try:
            parsed = feedparser.parse(f)
            for entry in parsed.entries[:5]: # top 5 from each
                news.append(entry.title)
        except Exception as e:
            print(f"Error fetching news: {e}")
            pass
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
    
    today_str = datetime.now().strftime("%Y-%m-%d")
    history = load_history()
    
    # 1. Evaluate yesterday's predictions if any
    print("--- 📊 Evaluation of Previous Predictions ---")
    
    # Look back up to 3 days (in case of weekends) to evaluate the last run
    last_run_date = None
    for i in range(1, 4):
        check_date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
        if check_date in history and 'evaluated' not in history[check_date]:
            last_run_date = check_date
            break
            
    if last_run_date:
        print(f"Evaluating predictions made on: {last_run_date}")
        prev_preds = history[last_run_date]
        eval_results = {}
        for name, sym in SYMBOLS.items():
            df = yf.Ticker(sym).history(period="5d")
            if not df.empty and len(df) >= 2:
                actual_return = (df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]
                pred = prev_preds.get(name, 0)
                actual_dir = "Up" if actual_return > 0 else "Down"
                pred_dir = "Up" if pred > 0 else "Down"
                correct = actual_dir == pred_dir
                print(f"[{name}] Predicted: {pred*100:.2f}% ({pred_dir}) | Actual: {actual_return*100:.2f}% ({actual_dir}) | Correct: {correct}")
                eval_results[name] = {
                    "predicted_percent": pred,
                    "actual_percent": float(actual_return),
                    "predicted_dir": pred_dir,
                    "actual_dir": actual_dir,
                    "correct": correct
                }
        history[last_run_date]['evaluated'] = True
    else:
        print("No recent unevaluated predictions found.")
    
    # 2. Make today's predictions
    print("\n--- 🔮 Today's Predictions (Quantitative Models) ---")
    today_preds = {}
    for name, sym in SYMBOLS.items():
        try:
            df = fetch_data(sym)
            if df is not None:
                pred_rf = train_predict_rf(df)
                pred_arima = train_predict_arima(df)
                
                # Ensemble (average)
                ensemble_pred = (pred_rf + pred_arima) / 2
                today_preds[name] = float(ensemble_pred)
                
                direction = "Up 📈" if ensemble_pred > 0 else "Down 📉"
                print(f"[{name}] Direction: {direction} | Expected Change: {ensemble_pred*100:.2f}%")
        except Exception as e:
            print(f"[{name}] Error predicting: {e}")
            
    history[today_str] = today_preds
    save_history(history)
    
    # 3. Fetch News
    print("\n--- 📰 Latest Market News (for Qualitative Analysis) ---")
    news = fetch_news()
    if news:
        for i, n in enumerate(news, 1):
            print(f"{i}. {n}")
    else:
        print("No news fetched via RSS. Falling back...")

    # 4. Export to Dashboard Data
    dashboard_data = {
        "last_updated": datetime.now().isoformat(),
        "today_predictions": today_preds,
        "evaluation": {
            "last_evaluated_date": last_run_date,
            "results": eval_results if 'eval_results' in locals() else {}
        },
        "news": news
    }
    with open(DASHBOARD_FILE, 'w') as f:
        json.dump(dashboard_data, f, indent=4)
        
if __name__ == "__main__":
    main()
