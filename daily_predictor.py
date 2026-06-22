import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA
import json
import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import warnings
warnings.filterwarnings('ignore')

# สูตรที่ 4 (News) — โหลดแบบ optional ถ้า vaderSentiment ไม่ได้ติดตั้งยังรันได้
try:
    import news_analyzer as _news_mod
    _NEWS_AVAILABLE = True
except ImportError:
    _NEWS_AVAILABLE = False
    print("[News] vaderSentiment ยังไม่ได้ติดตั้ง — รัน: pip install vaderSentiment")

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
    "Bitcoin": "BTC-USD",
}

BANGKOK_TZ = ZoneInfo("Asia/Bangkok")


def is_trading_day(day):
    """Return whether the Bangkok calendar day is Monday through Friday."""
    if isinstance(day, datetime):
        day = day.date()
    return day.weekday() < 5


def next_trading_day(day):
    """Return the next weekday after a Bangkok calendar day."""
    if isinstance(day, datetime):
        day = day.date()
    day += timedelta(days=1)
    while not is_trading_day(day):
        day += timedelta(days=1)
    return day


def previous_trading_day(day):
    """Return the previous weekday before a Bangkok calendar day."""
    if isinstance(day, datetime):
        day = day.date()
    day -= timedelta(days=1)
    while not is_trading_day(day):
        day -= timedelta(days=1)
    return day


def get_run_dates(now):
    """Return the run, evaluation, and prediction dates, or None on weekends."""
    bangkok_day = now.astimezone(BANGKOK_TZ).date()
    if not is_trading_day(bangkok_day):
        return None
    return {
        "run_date": bangkok_day,
        "evaluation_date": previous_trading_day(bangkok_day),
        "prediction_date": next_trading_day(bangkok_day),
    }


def price_change_for_market_date(df, market_date):
    """Return the close-to-close change for one completed market date."""
    positions = [
        i for i, timestamp in enumerate(df.index)
        if pd.Timestamp(timestamp).date() == market_date
    ]
    if not positions:
        return None

    position = positions[-1]
    if position == 0:
        return None

    close = df["Close"].iloc[position]
    previous_close = df["Close"].iloc[position - 1]
    if pd.isna(close) or pd.isna(previous_close) or previous_close == 0:
        return None

    return {
        "market_date": market_date,
        "previous_market_date": pd.Timestamp(df.index[position - 1]).date(),
        "actual_pct": float((close - previous_close) / previous_close),
    }


def preserve_historical_evaluations(history):
    """Keep prior prediction results intact; only future runs use the new safeguards."""
    return history


def remove_weekend_target_entries(history):
    """Remove only records whose prediction date is Saturday or Sunday."""
    cleaned = {}
    for date_str, entry in history.items():
        try:
            prediction_day = datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            cleaned[date_str] = entry
            continue
        if is_trading_day(prediction_day):
            cleaned[date_str] = entry
    return cleaned

# ─── สูตรที่ 1: Random Forest ──────────────────────────────────────────────────

def fetch_data(symbol, period="5y"):
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period)
    if df.empty:
        return None
    df['Return']        = df['Close'].pct_change()
    df['SMA_10']        = df['Close'].rolling(window=10).mean()
    df['SMA_50']        = df['Close'].rolling(window=50).mean()
    df['Volatility']    = df['Return'].rolling(window=20).std()
    df['Target_Return'] = df['Return'].shift(-1)
    return df.dropna()


def train_predict_rf(df):
    features = ['Return', 'SMA_10', 'SMA_50', 'Volatility']
    X, y = df[features].values, df['Target_Return'].values
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return float(model.predict(df[features].iloc[-1].values.reshape(1, -1))[0])


# ─── สูตรที่ 2: ARIMA ──────────────────────────────────────────────────────────

def train_predict_arima(df):
    try:
        model_fit = ARIMA(df['Close'].values, order=(5, 1, 0)).fit()
        pred_close = model_fit.forecast(steps=1)[0]
        last_close = df['Close'].iloc[-1]
        return float((pred_close - last_close) / last_close)
    except Exception:
        return 0.0


# ─── History helpers ──────────────────────────────────────────────────────────

def sanitize(obj):
    """แทนที่ NaN/Inf ด้วย None เพื่อให้ JSON.parse ของ browser รับได้"""
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


# ─── สูตรที่ 3: Adaptive Weighted Ensemble (3-way) ────────────────────────────

def compute_model_weights(history):
    """
    FEEDBACK LOOP — หัวใจของการเรียนรู้ตัวเอง

    นับว่าแต่ละโมเดล (RF, ARIMA, News) ทายทิศทางถูกกี่ครั้งในอดีต
    แล้วแปลงเป็นน้ำหนัก (w) ด้วย Laplace Smoothing
    โมเดลที่แม่นกว่าจะได้ออกเสียงมากกว่าในรอบถัดไป

    news_total=0 → ใช้ค่า default 0.45 (ต่ำกว่า Laplace 0.5 เล็กน้อย
    เพื่อให้ RF/ARIMA มีน้ำหนักนำก่อนจนกว่าข่าวจะพิสูจน์ตัวเอง)
    """
    perf = {
        name: {"rf_correct": 0, "rf_total": 0,
               "arima_correct": 0, "arima_total": 0,
               "news_correct": 0, "news_total": 0}
        for name in SYMBOLS
    }

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
            nw = p.get("news_pct")

            if rf is not None:
                perf[name]["rf_total"] += 1
                if ("Up" if rf > 0 else "Down") == actual_dir:
                    perf[name]["rf_correct"] += 1
            if ar is not None:
                perf[name]["arima_total"] += 1
                if ("Up" if ar > 0 else "Down") == actual_dir:
                    perf[name]["arima_correct"] += 1
            # news_pct=0 หมายถึง "ไม่มีข่าว" ไม่นับเป็นการทาย
            if nw is not None and nw != 0.0:
                perf[name]["news_total"] += 1
                if ("Up" if nw > 0 else "Down") == actual_dir:
                    perf[name]["news_correct"] += 1

    weights = {}
    for name, s in perf.items():
        rf_acc   = (s["rf_correct"]   + 1) / (s["rf_total"]   + 2)
        ar_acc   = (s["arima_correct"] + 1) / (s["arima_total"] + 2)
        # News: ถ้ายังไม่มีประวัติให้ default 0.45 (conservative prior)
        if s["news_total"] > 0:
            news_acc = (s["news_correct"] + 1) / (s["news_total"] + 2)
        else:
            news_acc = 0.45
        total = rf_acc + ar_acc + news_acc

        weights[name] = {
            "rf":             round(rf_acc   / total, 3),
            "arima":          round(ar_acc   / total, 3),
            "news":           round(news_acc / total, 3),
            "rf_accuracy":    round(s["rf_correct"]    / s["rf_total"]    * 100, 1) if s["rf_total"]    else None,
            "arima_accuracy": round(s["arima_correct"] / s["arima_total"] * 100, 1) if s["arima_total"] else None,
            "news_accuracy":  round(s["news_correct"]  / s["news_total"]  * 100, 1) if s["news_total"]  else None,
            "samples":        s["rf_total"],
        }
    return weights


def compute_stats(history):
    """รวมสถิติความแม่นยำสะสม"""
    per_symbol = {name: {"total": 0, "correct": 0} for name in SYMBOLS}
    overall    = {"total": 0, "correct": 0}
    records    = []

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
        "recent_history":       records[-30:],
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 50)
    print("🚀 DAILY MARKET PREDICTIVE SYSTEM (4-Formula)")
    print("=" * 50 + "\n")

    run_dates = get_run_dates(datetime.now(BANGKOK_TZ))
    if run_dates is None:
        print("⏸️ วันนี้เป็นวันหยุดสุดสัปดาห์ — ไม่ดึงข้อมูล ไม่ทำนาย และไม่บันทึกผล")
        return

    today_str    = run_dates["run_date"].isoformat()
    tomorrow_str = run_dates["prediction_date"].isoformat()

    history = remove_weekend_target_entries(preserve_historical_evaluations(load_history()))

    # ── STEP 1: Migrate legacy entries ────────────────────────────────────────
    for date_str, entry in list(history.items()):
        if isinstance(entry, dict) and "predictions" not in entry:
            preds = {k: v for k, v in entry.items()
                     if k not in ("made_on", "for_date", "evaluated", "eval_date")}
            history[date_str] = {
                "for_date":    date_str,
                "made_on":     entry.get("made_on", "unknown"),
                "predictions": {
                    name: {
                        "predicted_pct": float(pct),
                        "predicted_dir": "Up" if float(pct) > 0 else "Down",
                        "rf_pct":        float(pct),
                        "arima_pct":     float(pct),
                        "news_pct":      0.0,
                    }
                    for name, pct in preds.items() if isinstance(pct, (int, float))
                },
                "actuals":   {},
                "evaluated": entry.get("evaluated", False),
                "eval_date": entry.get("eval_date", None),
            }

    # ── STEP 2: Evaluate recent unevaluated prediction ─────────────────────────
    print("--- 📊 ตรวจสอบความแม่นยำ: ทำนายเมื่อวานถูกไหม? ---")

    eval_target_date = run_dates["evaluation_date"].isoformat()
    entry = history.get(eval_target_date, {})
    if not entry or entry.get("evaluated"):
        eval_target_date = None

    eval_results = {}
    if eval_target_date:
        entry   = history[eval_target_date]
        made_on = entry.get("made_on", "unknown")
        print(f"  ทำนายเมื่อ: {made_on}  |  สำหรับวัน: {eval_target_date}")

        for name, sym in SYMBOLS.items():
            df = yf.Ticker(sym).history(period="5d")
            if df.empty or len(df) < 2:
                continue
            expected_close_date = run_dates["evaluation_date"]
            market_result = price_change_for_market_date(df, expected_close_date)
            if market_result is None:
                print(f"  [{name}] ข้าม — ไม่มีราคาปิดของวันประเมิน {expected_close_date}")
                continue
            actual_pct = market_result["actual_pct"]
            pred_info  = entry["predictions"].get(name, {})
            pred_pct   = pred_info.get("predicted_pct", 0)
            actual_dir = "Up" if actual_pct > 0 else "Down"
            pred_dir   = pred_info.get("predicted_dir", "Up" if pred_pct > 0 else "Down")
            correct    = actual_dir == pred_dir

            print(f"  [{name}] ทำนาย: {pred_pct*100:.2f}% ({pred_dir}) "
                  f"| จริง: {actual_pct*100:.2f}% ({actual_dir}) "
                  f"| {'✅ ถูก' if correct else '❌ ผิด'}")

            entry["actuals"][name] = {
                "actual_pct": actual_pct,
                "actual_dir": actual_dir,
                "correct":    correct,
                "market_date": market_result["market_date"].isoformat(),
                "previous_market_date": market_result["previous_market_date"].isoformat(),
            }
            eval_results[name] = {
                "predicted_pct": pred_pct,
                "predicted_dir": pred_dir,
                "actual_pct":    actual_pct,
                "actual_dir":    actual_dir,
                "correct":       correct,
            }
        if eval_results:
            entry["evaluated"] = True
            entry["eval_date"] = today_str
        else:
            print("  ยังไม่มีราคาปิดของวันประเมิน จึงยังไม่บันทึกผล")
    else:
        print("  ไม่พบการทำนายที่ยังไม่ได้ประเมิน")

    # ── STEP 3: Compute adaptive 3-way weights ────────────────────────────────
    model_weights = compute_model_weights(history)
    print("\n--- ⚖️  น้ำหนักโมเดล (เรียนรู้จากประวัติ) ---")
    for name, w in model_weights.items():
        rf_a   = f"{w['rf_accuracy']}%"   if w['rf_accuracy']   is not None else "n/a"
        ar_a   = f"{w['arima_accuracy']}%" if w['arima_accuracy'] is not None else "n/a"
        nw_a   = f"{w['news_accuracy']}%"  if w['news_accuracy']  is not None else "n/a"
        print(f"  [{name}] RF={w['rf']}({rf_a}) ARIMA={w['arima']}({ar_a}) "
              f"News={w['news']}({nw_a}) | {w['samples']} วัน")

    # ── STEP 3.5: สูตรที่ 4 — News Sentiment ─────────────────────────────────
    news_results = {}
    news_meta    = {}
    if _NEWS_AVAILABLE:
        try:
            payload      = _news_mod.analyze_all(SYMBOLS)
            news_meta    = payload.pop("_meta", {})
            news_results = payload
        except Exception as e:
            print(f"\n[News] Error: {e}")
    else:
        print("\n[News] ข้ามสูตรที่ 4 (vaderSentiment ไม่ได้ติดตั้ง)")

    # ── STEP 4: Predict for the next trading day (3-way weighted ensemble) ───
    print(f"\n--- 🔮 ทำนายสำหรับวันทำการถัดไป ({tomorrow_str}) ---")

    tomorrow_preds = {}
    pred_details   = {}
    for name, sym in SYMBOLS.items():
        try:
            df = fetch_data(sym)
            if df is None:
                continue
            rf_pct    = train_predict_rf(df)
            arima_pct = train_predict_arima(df)

            news_data = news_results.get(name, {})
            news_pct  = news_data.get("news_score", 0.0) or 0.0

            w = model_weights[name]

            # ถ้าไม่มีสัญญาณข่าว → กระจายน้ำหนัก news ให้ RF+ARIMA แบบ proportional
            if news_pct == 0.0:
                denom    = w["rf"] + w["arima"]
                rf_w     = w["rf"]    / denom if denom > 0 else 0.5
                ar_w     = w["arima"] / denom if denom > 0 else 0.5
                ensemble = rf_pct * rf_w + arima_pct * ar_w
            else:
                ensemble = rf_pct * w["rf"] + arima_pct * w["arima"] + news_pct * w["news"]

            direction = "Up" if ensemble > 0 else "Down"
            tomorrow_preds[name] = ensemble
            pred_details[name]   = {
                "predicted_pct": ensemble,
                "predicted_dir": direction,
                "rf_pct":        rf_pct,
                "arima_pct":     arima_pct,
                "news_pct":      news_pct,
                "news_info": {
                    "direction":     news_data.get("news_direction", "Neutral"),
                    "article_count": news_data.get("article_count", 0),
                    "avg_credibility": news_data.get("avg_credibility"),
                    "sentiment_raw": news_data.get("sentiment_raw"),
                },
                "weights": {"rf": w["rf"], "arima": w["arima"], "news": w["news"]},
            }
            icon = "📈" if ensemble > 0 else "📉"
            nw_icon = "📈" if news_pct > 0 else ("📉" if news_pct < 0 else "➖")
            print(f"  [{name}] {direction} {icon} | "
                  f"RF:{rf_pct*100:.2f}%(w{w['rf']})  "
                  f"ARIMA:{arima_pct*100:.2f}%(w{w['arima']})  "
                  f"News:{nw_icon}{news_pct*100:.3f}%(w{w['news']})  "
                  f"→ Ensemble:{ensemble*100:.2f}%")
        except Exception as e:
            print(f"  [{name}] Error: {e}")

    history[tomorrow_str] = {
        "for_date":    tomorrow_str,
        "made_on":     today_str,
        "predictions": pred_details,
        "actuals":     {},
        "evaluated":   False,
        "eval_date":   None,
    }
    save_history(history)

    # ── STEP 5: Cumulative stats ───────────────────────────────────────────────
    stats = compute_stats(history)
    print(f"\n--- 📈 สถิติสะสม ---")
    print(f"  ความแม่นยำรวม: {stats['overall_accuracy_pct']}% "
          f"({stats['total_evaluated']} วันที่ประเมินแล้ว)")
    for name, s in stats["per_symbol"].items():
        print(f"    {name}: {s['accuracy_pct']}%  ({s['correct']}/{s['total']})")

    # ── STEP 6: Write dashboard_data.json ─────────────────────────────────────
    dashboard_data = {
        "last_updated":         datetime.now(BANGKOK_TZ).isoformat(),
        "prediction_for_date":  tomorrow_str,
        "tomorrow_predictions": tomorrow_preds,
        "tomorrow_details":     pred_details,
        "model_weights":        model_weights,
        "evaluation": {
            "prediction_was_for": eval_target_date,
            "made_on": history.get(eval_target_date, {}).get("made_on") if eval_target_date else None,
            "results": eval_results,
        },
        "stats":  stats,
        # ข่าว: เฉพาะแหล่งน่าเชื่อถือ ≥ 90%
        "news": news_meta.get("headlines", []),
        "news_fetch_stats": {
            "total_fetched":   news_meta.get("total_fetched", 0),
            "accepted":        news_meta.get("accepted", 0),
            "rejected":        news_meta.get("rejected", 0),
            "min_credibility": news_meta.get("min_credibility", 90),
        },
    }
    with open(DASHBOARD_FILE, 'w') as f:
        json.dump(sanitize(dashboard_data), f, indent=4)

    print("\n✅ dashboard_data.json อัปเดตเรียบร้อย")


if __name__ == "__main__":
    main()
