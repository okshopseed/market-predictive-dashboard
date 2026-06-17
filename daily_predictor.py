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
    "Gold":    "GC=F",
    "SCB":     "SCB.BK",
    "TQM":     "TQM.BK",
    "IVV":     "IVV",
    "Google":  "GOOGL",
    "NVDA":    "NVDA",
    "AMD":     "AMD",
    "TSM":     "TSM",
    "SMH":     "SMH",
    "MU":      "MU",
    "WDC":     "WDC",
    "TSLA":    "TSLA",
    "RKLB":    "RKLB",
    "Bitcoin": "BTC-USD"
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

def sanitize(obj):
    """Recursively replace NaN/Infinity with None.

    Python's json.dump emits bare `NaN`/`Infinity` tokens by default, which are
    valid for Python's json.load but are REJECTED by the browser's JSON.parse
    (and by JSON.parse in Node). A single NaN anywhere in the payload makes the
    whole dashboard fail with "Error loading data". This guarantees every value
    we write is standards-compliant JSON the browser can parse.
    """
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize(v) for v in obj]
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj

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
        json.dump(sanitize(history), f, indent=4)

def compute_model_weights(history):
    """
    FEEDBACK LOOP — the core of self-improvement.

    For each symbol, look at every past evaluated prediction and measure how
    often RF vs ARIMA pointed in the correct DIRECTION. Models that have been
    more accurate historically get a larger weight in the next ensemble.

    Uses Laplace smoothing so a model is never fully silenced and cold-start
    (no history) defaults to a balanced 50/50.

    Returns: { symbol: {"rf": w, "arima": w, "rf_accuracy": %, "arima_accuracy": %, "samples": n} }
    """
    perf = {name: {"rf_correct": 0, "rf_total": 0, "arima_correct": 0, "arima_total": 0}
            for name in SYMBOLS}

    for date_str, entry in history.items():
        if not isinstance(entry, dict) or not entry.get("evaluated"):
            continue
        actuals = entry.get("actuals", {})
        preds   = entry.get("predictions", {})
        for name in SYMBOLS:
            if name not in actuals or name not in preds:
                continue
            actual_dir = actuals[name]["actual_dir"]
            p  = preds[name]
            rf = p.get("rf_pct")
            ar = p.get("arima_pct")
            if rf is not None:
                perf[name]["rf_total"] += 1
                if ("Up" if rf > 0 else "Down") == actual_dir:
                    perf[name]["rf_correct"] += 1
            if ar is not None:
                perf[name]["arima_total"] += 1
                if ("Up" if ar > 0 else "Down") == actual_dir:
                    perf[name]["arima_correct"] += 1

    weights = {}
    for name, s in perf.items():
        # Laplace-smoothed directional accuracy (never 0, never 1 with tiny n)
        rf_acc = (s["rf_correct"] + 1) / (s["rf_total"] + 2)
        ar_acc = (s["arima_correct"] + 1) / (s["arima_total"] + 2)
        total  = rf_acc + ar_acc
        weights[name] = {
            "rf":             round(rf_acc / total, 3),
            "arima":          round(ar_acc / total, 3),
            "rf_accuracy":    round(s["rf_correct"] / s["rf_total"] * 100, 1) if s["rf_total"] else None,
            "arima_accuracy": round(s["arima_correct"] / s["arima_total"] * 100, 1) if s["arima_total"] else None,
            "samples":        s["rf_total"],
        }
    return weights


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
            prev_close = df['Close'].iloc[-2]
            actual_pct = float((df['Close'].iloc[-1] - prev_close) / prev_close)
            # Skip symbols with no valid move (e.g. market holiday / missing data) —
            # otherwise we'd store a NaN and log a bogus "correct" result.
            if pd.isna(actual_pct) or prev_close == 0:
                print(f"[{name}] Skipped — no valid price change (holiday/missing data)")
                continue
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

    # ── STEP 3: Compute ADAPTIVE WEIGHTS from past performance ──
    # This is where accumulated history feeds back into the next forecast.
    model_weights = compute_model_weights(history)
    print("\n--- ⚖️  Adaptive Model Weights (learned from history) ---")
    for name, w in model_weights.items():
        rf_a = f"{w['rf_accuracy']}%" if w['rf_accuracy'] is not None else "n/a"
        ar_a = f"{w['arima_accuracy']}%" if w['arima_accuracy'] is not None else "n/a"
        print(f"[{name}] RF w={w['rf']} (acc {rf_a}) | ARIMA w={w['arima']} (acc {ar_a}) "
              f"| samples={w['samples']}")

    # ── STEP 4: Predict for TOMORROW using weighted ensemble ──
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

            # Weighted ensemble — better historical model gets more say
            w        = model_weights[name]
            ensemble = rf_pct * w["rf"] + arima_pct * w["arima"]
            direction = "Up" if ensemble > 0 else "Down"

            tomorrow_preds[name] = ensemble
            pred_details[name]   = {
                "predicted_pct": ensemble,
                "predicted_dir": direction,
                "rf_pct":        rf_pct,
                "arima_pct":     arima_pct,
                "weights":       {"rf": w["rf"], "arima": w["arima"]},
            }
            icon = "📈" if ensemble > 0 else "📉"
            print(f"[{name}] {direction} {icon} | RF: {rf_pct*100:.2f}%(w{w['rf']})  "
                  f"ARIMA: {arima_pct*100:.2f}%(w{w['arima']})  → Ensemble: {ensemble*100:.2f}%")
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

    # ── STEP 5: Fetch News ──
    print("\n--- 📰 Latest Market News ---")
    news = fetch_news()
    for i, n in enumerate(news, 1):
        print(f"{i}. {n}")
    if not news:
        print("No news fetched.")

    # ── STEP 6: Compute cumulative stats ──
    stats = compute_stats(history)
    print(f"\n--- 📈 Cumulative Stats ---")
    print(f"Overall accuracy: {stats['overall_accuracy_pct']}%  "
          f"({stats['total_evaluated']} days evaluated)")
    for name, s in stats["per_symbol"].items():
        print(f"  {name}: {s['accuracy_pct']}%  ({s['correct']}/{s['total']})")

    # ── STEP 7: Write dashboard_data.json ──
    dashboard_data = {
        "last_updated":         datetime.utcnow().isoformat(),
        "prediction_for_date":  tomorrow_str,
        "tomorrow_predictions": tomorrow_preds,
        "tomorrow_details":     pred_details,
        "model_weights":        model_weights,
        "evaluation": {
            "prediction_was_for": eval_target_date,
            "made_on":            history.get(eval_target_date, {}).get("made_on") if eval_target_date else None,
            "results":            eval_results,
        },
        "stats":  stats,
        "news":   news,
    }
    with open(DASHBOARD_FILE, 'w') as f:
        json.dump(sanitize(dashboard_data), f, indent=4)

    print("\n✅ dashboard_data.json updated.")

if __name__ == "__main__":
    main()
