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

DATA_FILE      = "prediction_history.json"
DASHBOARD_FILE = "dashboard_data.json"
SYMBOLS = {
    "S&P 500": "^GSPC",
    "Nasdaq":  "^IXIC",
    "Gold":    "GC=F"
}

# ─────────────────────────────────────────────
# Data / Model helpers
# ─────────────────────────────────────────────

def fetch_data(symbol, period="5y"):
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period)
    if df.empty:
        return None
    df['Return']      = df['Close'].pct_change()
    df['SMA_10']      = df['Close'].rolling(window=10).mean()
    df['SMA_50']      = df['Close'].rolling(window=50).mean()
    df['Volatility']  = df['Return'].rolling(window=20).std()
    df['Target_Return'] = df['Return'].shift(-1)
    return df.dropna()

def train_predict_rf(df):
    features = ['Return', 'SMA_10', 'SMA_50', 'Volatility']
    X, y = df[features].values, df['Target_Return'].values
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return float(model.predict(df[features].iloc[-1].values.reshape(1, -1))[0])

def train_predict_arima(df):
    try:
        model_fit = ARIMA(df['Close'].values, order=(5, 1, 0)).fit()
        pred_close = model_fit.forecast(steps=1)[0]
        last_close = df['Close'].iloc[-1]
        return float((pred_close - last_close) / last_close)
    except Exception:
        return 0.0

def fetch_news():
    feeds = [
        "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664",
        "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
    ]
    news = []
    for url in feeds:
        try:
            for entry in feedparser.parse(url).entries[:5]:
                news.append(entry.title)
        except Exception as e:
            print(f"News fetch error: {e}")
    return news

# ─────────────────────────────────────────────
# History helpers — new schema
#
# prediction_history.json structure:
# {
#   "YYYY-MM-DD": {           ← the date this prediction is FOR
#     "for_date":  "...",
#     "made_on":   "...",
#     "predictions": {
#       "S&P 500": {"predicted_pct": 0.006, "predicted_dir": "Up",
#                   "rf_pct": ..., "arima_pct": ...},
#       ...
#     },
#     "actuals": {            ← filled in during evaluation
#       "S&P 500": {"actual_pct": 0.016, "actual_dir": "Up", "correct": true},
#       ...
#     },
#     "evaluated": false,
#     "eval_date":  null
#   }
# }
# ─────────────────────────────────────────────

def load_history():
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def save_history(history):
    with open(DATA_FILE, 'w') as f:
        json.dump(history, f, indent=4)

def compute_stats(history):
    """Aggregate accuracy stats across all evaluated entries."""
    per_symbol = {name: {"total": 0, "correct": 0} for name in SYMBOLS}
    overall    = {"total": 0, "correct": 0}
    records    = []  # for recent-history table

    for date_str in sorted(history.keys()):
        entry = history[date_str]
        if not isinstance(entry, dict) or not entry.get("evaluated"):
            continue

        actuals     = entry.get("actuals", {})
        predictions = entry.get("predictions", {})
        made_on     = entry.get("made_on", "?")
        row = {"for_date": date_str, "made_on": made_on, "symbols": {}}

        for name in SYMBOLS:
            if name not in actuals:
                continue
            a = actuals[name]
            p = predictions.get(name, {})
            per_symbol[name]["total"]   += 1
            overall["total"]            += 1
            if a["correct"]:
                per_symbol[name]["correct"] += 1
                overall["correct"]          += 1
            row["symbols"][name] = {
                "predicted_dir": p.get("predicted_dir"),
                "predicted_pct": p.get("predicted_pct"),
                "actual_dir":    a["actual_dir"],
                "actual_pct":    a["actual_pct"],
                "correct":       a["correct"],
            }
        records.append(row)

    # Per-symbol accuracy %
    symbol_stats = {}
    for name, s in per_symbol.items():
        symbol_stats[name] = {
            "total":        s["total"],
            "correct":      s["correct"],
            "accuracy_pct": round(s["correct"] / s["total"] * 100, 1) if s["total"] else None,
        }

    overall_accuracy = (
        round(overall["correct"] / overall["total"] * 100, 1)
        if overall["total"] else None
    )

    # Current winning streak (most recent consecutive correct across ALL symbols)
    streak = 0
    for row in reversed(records):
        if all(row["symbols"].get(n, {}).get("correct") for n in SYMBOLS if n in row["symbols"]):
            streak += 1
        else:
            break

    return {
        "overall_accuracy_pct": overall_accuracy,
        "total_evaluated":      overall["total"] // len(SYMBOLS) if overall["total"] else 0,
        "per_symbol":           symbol_stats,
        "all_correct_streak":   streak,
        "recent_history":       records[-30:],  # last 30 evaluated days
    }

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    print("========================================")
    print("🚀 DAILY MARKET PREDICTIVE SYSTEM 🚀")
    print("========================================\n")

    # Script runs at 23:00 UTC (06:00 ICT). US markets closed 3h ago.
    today_str    = datetime.utcnow().strftime("%Y-%m-%d")
    tomorrow_str = (datetime.utcnow() + timedelta(days=1)).strftime("%Y-%m-%d")

    history = load_history()

    # ── STEP 1: Migrate any legacy flat entries (old schema → new schema) ──
    for date_str, entry in list(history.items()):
        if isinstance(entry, dict) and "predictions" not in entry:
            # Old flat format: {"S&P 500": 0.006, "made_on": ..., "evaluated": ...}
            preds = {k: v for k, v in entry.items()
                     if k not in ("made_on", "for_date", "evaluated", "eval_date")}
            new_entry = {
                "for_date":    date_str,
                "made_on":     entry.get("made_on", "unknown"),
                "predictions": {
                    name: {
                        "predicted_pct": float(pct),
                        "predicted_dir": "Up" if float(pct) > 0 else "Down",
                        "rf_pct":        float(pct),
                        "arima_pct":     float(pct),
                    }
                    for name, pct in preds.items() if isinstance(pct, (int, float))
                },
                "actuals":   {},
                "evaluated": entry.get("evaluated", False),
                "eval_date": entry.get("eval_date", None),
            }
            history[date_str] = new_entry

    # ── STEP 2: Evaluate the most recent unevaluated prediction ──
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
        entry   = history[eval_target_date]
        made_on = entry.get("made_on", "unknown")
        print(f"Prediction made on: {made_on}  |  Was for: {eval_target_date}")

        for name, sym in SYMBOLS.items():
            df = yf.Ticker(sym).history(period="5d")
            if df.empty or len(df) < 2:
                continue
            actual_pct = float((df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2])
            pred_info  = entry["predictions"].get(name, {})
            pred_pct   = pred_info.get("predicted_pct", 0)
            actual_dir = "Up" if actual_pct > 0 else "Down"
            pred_dir   = pred_info.get("predicted_dir", "Up" if pred_pct > 0 else "Down")
            correct    = actual_dir == pred_dir

            print(f"[{name}] Predicted: {pred_pct*100:.2f}% ({pred_dir}) "
                  f"| Actual: {actual_pct*100:.2f}% ({actual_dir}) "
                  f"| {'✅ CORRECT' if correct else '❌ WRONG'}")

            # Persist actuals back into history
            entry["actuals"][name] = {
                "actual_pct": actual_pct,
                "actual_dir": actual_dir,
                "correct":    correct,
            }
            eval_results[name] = {
                "predicted_pct": pred_pct,
                "predicted_dir": pred_dir,
                "actual_pct":    actual_pct,
                "actual_dir":    actual_dir,
                "correct":       correct,
            }

        entry["evaluated"] = True
        entry["eval_date"] = today_str
    else:
        print("No unevaluated prediction found for today or recent past.")

    # ── STEP 3: Predict for TOMORROW ──
    print(f"\n--- 🔮 Prediction for TOMORROW ({tomorrow_str}) ---")

    tomorrow_preds = {}
    pred_details   = {}
    for name, sym in SYMBOLS.items():
        try:
            df = fetch_data(sym)
            if df is None:
                continue
            rf_pct    = train_predict_rf(df)
            arima_pct = train_predict_arima(df)
            ensemble  = (rf_pct + arima_pct) / 2
            direction = "Up" if ensemble > 0 else "Down"

            tomorrow_preds[name] = ensemble
            pred_details[name]   = {
                "predicted_pct": ensemble,
                "predicted_dir": direction,
                "rf_pct":        rf_pct,
                "arima_pct":     arima_pct,
            }
            icon = "📈" if ensemble > 0 else "📉"
            print(f"[{name}] {direction} {icon} | RF: {rf_pct*100:.2f}%  "
                  f"ARIMA: {arima_pct*100:.2f}%  Ensemble: {ensemble*100:.2f}%")
        except Exception as e:
            print(f"[{name}] Error: {e}")

    # Store prediction FOR tomorrow
    history[tomorrow_str] = {
        "for_date":    tomorrow_str,
        "made_on":     today_str,
        "predictions": pred_details,
        "actuals":     {},
        "evaluated":   False,
        "eval_date":   None,
    }
    save_history(history)

    # ── STEP 4: Fetch News ──
    print("\n--- 📰 Latest Market News ---")
    news = fetch_news()
    for i, n in enumerate(news, 1):
        print(f"{i}. {n}")
    if not news:
        print("No news fetched.")

    # ── STEP 5: Compute cumulative stats ──
    stats = compute_stats(history)
    print(f"\n--- 📈 Cumulative Stats ---")
    print(f"Overall accuracy: {stats['overall_accuracy_pct']}%  "
          f"({stats['total_evaluated']} days evaluated)")
    for name, s in stats["per_symbol"].items():
        print(f"  {name}: {s['accuracy_pct']}%  ({s['correct']}/{s['total']})")

    # ── STEP 6: Write dashboard_data.json ──
    dashboard_data = {
        "last_updated":         datetime.utcnow().isoformat(),
        "prediction_for_date":  tomorrow_str,
        "tomorrow_predictions": tomorrow_preds,
        "tomorrow_details":     pred_details,
        "evaluation": {
            "prediction_was_for": eval_target_date,
            "made_on":            history.get(eval_target_date, {}).get("made_on") if eval_target_date else None,
            "results":            eval_results,
        },
        "stats":  stats,
        "news":   news,
    }
    with open(DASHBOARD_FILE, 'w') as f:
        json.dump(dashboard_data, f, indent=4)

    print("\n✅ dashboard_data.json updated.")

if __name__ == "__main__":
    main()
