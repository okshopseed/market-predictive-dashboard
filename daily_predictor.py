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

import shadow_model
import market_model

# สูตรที่ 4 (News) — โหลดแบบ optional ถ้า vaderSentiment ไม่ได้ติดตั้งยังรันได้
try:
    import news_analyzer as _news_mod
    _NEWS_AVAILABLE = True
except ImportError:
    _NEWS_AVAILABLE = False
    print("[News] vaderSentiment ยังไม่ได้ติดตั้ง — รัน: pip install vaderSentiment")

DATA_FILE      = "prediction_history.json"
DASHBOARD_FILE = "dashboard_data.json"
NEWS_HISTORY_FILE = "news_history.json"
MODEL_REGISTRY_FILE = "model_registry.json"
PRICE_HISTORY_FILE = "price_history.json"
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
    """Return the current prediction date and the prior closed market date."""
    bangkok_day = now.astimezone(BANGKOK_TZ).date()
    if not is_trading_day(bangkok_day):
        return None
    return {
        "run_date": bangkok_day,
        "evaluation_date": previous_trading_day(bangkok_day),
        "prediction_date": bangkok_day,
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


def ohlcv_for_market_date(df, market_date):
    """Extract the OHLCV snapshot for one completed market date (for the permanent archive)."""
    positions = [
        i for i, timestamp in enumerate(df.index)
        if pd.Timestamp(timestamp).date() == market_date
    ]
    if not positions:
        return None
    row = df.iloc[positions[-1]]

    def value(column):
        if column not in df.columns or pd.isna(row[column]):
            return None
        return float(row[column])

    return {
        "open":   value("Open"),
        "high":   value("High"),
        "low":    value("Low"),
        "close":  value("Close"),
        "volume": value("Volume"),
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

def prepare_market_data(prices):
    """Create legacy features while retaining the newest closed row for prediction."""
    df = prices.copy()
    df['Return']        = df['Close'].pct_change()
    df['SMA_10']        = df['Close'].rolling(window=10).mean()
    df['SMA_50']        = df['Close'].rolling(window=50).mean()
    df['Volatility']    = df['Return'].rolling(window=20).std()
    df['Target_Return'] = df['Return'].shift(-1)
    return df.dropna(subset=['Return', 'SMA_10', 'SMA_50', 'Volatility'])


def fetch_data(symbol, period="5y"):
    ticker = yf.Ticker(symbol)
    prices = ticker.history(period=period)
    if prices.empty:
        return None
    return prepare_market_data(prices)


def train_predict_rf(df):
    features = ['Return', 'SMA_10', 'SMA_50', 'Volatility']
    training = df.dropna(subset=['Target_Return'])
    if len(training) < 20:
        return 0.0
    X, y = training[features].values, training['Target_Return'].values
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return float(model.predict(df[features].iloc[-1].values.reshape(1, -1))[0])


# ─── สูตรที่ 2: ARIMA ──────────────────────────────────────────────────────────

def train_predict_arima(df):
    try:
        closes = df['Close'].dropna()
        model_fit = ARIMA(closes.values, order=(5, 1, 0)).fit()
        pred_close = model_fit.forecast(steps=1)[0]
        last_close = closes.iloc[-1]
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


def load_json_file(path, default):
    try:
        with open(path, "r") as file:
            return json.load(file)
    except (OSError, ValueError):
        return default


def save_json_file(path, payload):
    with open(path, "w") as file:
        json.dump(sanitize(payload), file, indent=4)


def evaluate_shadow_predictions(entry, symbol, actual_pct, market_date=None, previous_market_date=None):
    """Store the result of each shadow signal without touching Active results."""
    actual_dir = "Up" if actual_pct > 0 else "Down"
    shadow_actuals = entry.setdefault("shadow_actuals", {})
    for shadow_name, predictions in entry.get("shadow_predictions", {}).items():
        prediction = predictions.get(symbol)
        if not prediction:
            continue
        predicted_dir = prediction.get("predicted_dir", "Up")
        result = {
            "actual_pct": actual_pct,
            "actual_dir": actual_dir,
            "correct": actual_dir == predicted_dir,
        }
        if market_date is not None:
            result["market_date"] = market_date
        if previous_market_date is not None:
            result["previous_market_date"] = previous_market_date
        shadow_actuals.setdefault(shadow_name, {})[symbol] = result


def compute_shadow_progress(history, shadow_name):
    """Report a shadow track's recent accuracy vs the 75% target (informational only)."""
    records = []
    for for_date, entry in history.items():
        for actual in entry.get("shadow_actuals", {}).get(shadow_name, {}).values():
            records.append({"for_date": for_date, "correct": actual.get("correct", False)})
    return shadow_model.target_progress(records)


def compute_news_coverage(news_history, window_days=shadow_model.PROMOTION_WINDOW_DAYS):
    """Summarize how often the live news collector found eligible public sources."""
    dates = sorted(news_history.keys())[-window_days:]
    days_with_eligible_news = sum(
        bool(news_history[date_str].get("stats", {}).get("eligible", 0))
        for date_str in dates
        if isinstance(news_history.get(date_str), dict)
    )
    days_collected = len(dates)
    return {
        "days_collected": days_collected,
        "days_with_eligible_news": days_with_eligible_news,
        "coverage_pct": round(days_with_eligible_news / days_collected * 100, 1)
        if days_collected
        else None,
        "remaining_decision_days": max(0, window_days - days_collected),
    }


def build_validation_panel(history):
    """Progress-only panel for the dashboard — no gate, every arm always votes live."""
    return {
        "price_shadow": compute_shadow_progress(history, "price"),
        "news_shadow": compute_shadow_progress(history, "price_news"),
        "target_accuracy_pct": shadow_model.TARGET_ACCURACY_PCT,
    }


def registry_champion(registry, symbol):
    return registry.get("symbols", {}).get(symbol, {}).get("champion")


def _probability_to_pct(probability_up):
    """Keep the existing percentage-oriented dashboard card meaningful for classifiers."""
    return float((probability_up - 0.5) * 0.04)


def build_shadow_predictions(registry, symbol, price_signal, news_signal):
    """Return auditable price-only and price-plus-news shadow variants."""
    if not registry_champion(registry, symbol) or not price_signal:
        return {}

    price_prediction = dict(price_signal)
    price_prediction["predicted_pct"] = _probability_to_pct(price_prediction["probability_up"])
    price_prediction["predicted_dir"] = "Up" if price_prediction["probability_up"] >= 0.5 else "Down"

    price_news_prediction = shadow_model.build_news_shadow_prediction(price_prediction, news_signal)
    price_news_prediction["predicted_pct"] = _probability_to_pct(
        price_news_prediction["probability_up"]
    )
    return {"price": price_prediction, "price_news": price_news_prediction}


# ─── สูตรที่ 3: Adaptive N-arm Ensemble (recency-aware / Hedge) ────────────────

# แขนทำนายทั้งหมดที่ร่วมโหวตทุกวัน และคีย์ % ที่เก็บไว้ในประวัติของแต่ละแขน
ARM_NAMES = ("legacy_rf", "legacy_arima", "news", "price_ml")
ARM_PCT_KEYS = {
    "legacy_rf":    "rf_pct",
    "legacy_arima": "arima_pct",
    "news":         "news_pct",
    "price_ml":     "price_ml_pct",
}
# ป้ายสั้นสำหรับแสดงผลบนการ์ด (คงคีย์เดิม rf/arima/news ไว้เพื่อความเข้ากันได้)
ARM_SHORT_KEYS = {
    "legacy_rf":    "rf",
    "legacy_arima": "arima",
    "news":         "news",
    "price_ml":     "price_ml",
}


def _arm_voted(arm, value):
    """แขนนี้ถือว่า 'ออกเสียง' วันนั้นไหม (news=0 คือไม่มีข่าว ไม่นับ)."""
    if value is None:
        return False
    if arm == "news" and value == 0.0:
        return False
    return True


def compute_arm_weights(history, params=None):
    """
    FEEDBACK LOOP — หัวใจของการเรียนรู้ตัวเอง (เวอร์ชัน recency-aware + auto-tuned)

    ทุกแขน (legacy_rf, legacy_arima, news, price_ml) ทายทุกวัน หลังรู้ผลจริง
    จะ "คูณปรับน้ำหนัก" ด้วย Multiplicative-Weights/Hedge: แขนที่ผิดช่วงนี้โดนลด
    แขนที่ฟอร์มดีช่วงนี้ได้ออกเสียงมากกว่ารอบถัดไป — แยกคำนวณราย symbol

    params = {beta, window, floor} ที่ auto-tune มาจาก backtest (ถ้าไม่มีใช้ค่า default)

    คืนค่า (weights, accuracy):
      weights[symbol]  = {arm: น้ำหนัก 0..1 รวม=1} (เฉพาะแขนที่เคยออกเสียง)
      accuracy[symbol] = {arm: {recent_accuracy_pct, samples}}
    """
    params = params or {}
    beta = params.get("beta", market_model.HEDGE_BETA)
    window = params.get("window", market_model.HEDGE_WINDOW)
    floor = params.get("floor", market_model.HEDGE_FLOOR)
    weights = {}
    accuracy = {}
    for name in SYMBOLS:
        correctness = {arm: [] for arm in ARM_NAMES}
        for date_str in sorted(history.keys()):
            entry = history[date_str]
            if not isinstance(entry, dict) or not entry.get("evaluated"):
                continue
            actuals = entry.get("actuals", {})
            preds = entry.get("predictions", {})
            if name not in actuals or name not in preds:
                continue
            actual_dir = actuals[name]["actual_dir"]
            p = preds[name]
            for arm, key in ARM_PCT_KEYS.items():
                value = p.get(key)
                if not _arm_voted(arm, value):
                    continue
                correctness[arm].append(("Up" if value > 0 else "Down") == actual_dir)

        sequences = {arm: seq for arm, seq in correctness.items() if seq}
        weights[name] = market_model.adaptive_weights(sequences, beta=beta, floor=floor, window=window)
        accuracy[name] = {
            arm: {
                "recent_accuracy_pct": round(
                    sum(seq[-window:]) / len(seq[-window:]) * 100, 1
                ),
                "samples": len(seq),
            }
            for arm, seq in sequences.items()
        }
    return weights, accuracy


def resolve_today_weights(symbol_weights, present_arms):
    """น้ำหนักสำหรับแขนที่ออกเสียงวันนี้ พร้อม cold-start fallback สำหรับแขนใหม่."""
    if not present_arms:
        return {}
    known = {arm: symbol_weights.get(arm) for arm in present_arms if symbol_weights.get(arm) is not None}
    if not known:
        return {arm: 1.0 / len(present_arms) for arm in present_arms}
    # แขนใหม่ที่ยังไม่มีประวัติ → ให้เสียงแบบระมัดระวัง (เท่าแขนที่อ่อนสุดที่รู้จัก)
    default = min(known.values())
    resolved = {arm: symbol_weights.get(arm, default) if symbol_weights.get(arm) is not None else default
                for arm in present_arms}
    total = sum(resolved.values())
    return {arm: value / total for arm, value in resolved.items()}


# ─── สูตรที่ 1.5: แขน price_ml = adaptive blend ของ candidate ML models ────────

def price_ml_candidate_weights(registry_entry):
    """น้ำหนักแต่ละ candidate model จากความแม่นล่าสุดใน registry (วิธีเดียวกับ backtest)."""
    candidates = registry_entry.get("candidates")
    if not candidates:
        champion = registry_entry.get("champion")
        if not champion:
            return {}
        candidates = [{"model": champion, "recent_accuracy_pct": registry_entry.get("recent_accuracy_pct")}]
    raw = {
        c["model"]: max(market_model.HEDGE_FLOOR, (c.get("recent_accuracy_pct") or 50.0) / 100.0)
        for c in candidates if c.get("model")
    }
    total = sum(raw.values())
    return {model: value / total for model, value in raw.items()} if total else {}


def predict_price_ml_arm(raw_prices, registry_entry):
    """ทายด้วยทุก candidate ML model แล้วผสมความน่าจะเป็นตามน้ำหนัก (live = backtest)."""
    weights = price_ml_candidate_weights(registry_entry)
    if not weights:
        return None
    probabilities = {}
    for model in weights:
        signal = market_model.predict_price_signal(raw_prices, model_name=model, train_window=252)
        if signal:
            probabilities[model] = signal["probability_up"]
    if not probabilities:
        return None
    blended = market_model.blend_probabilities(probabilities, weights)
    if blended is None:
        blended = sum(probabilities.values()) / len(probabilities)
    return {
        "probability_up": float(blended),
        "predicted_dir": "Up" if blended >= 0.5 else "Down",
        "per_model": {model: round(prob, 4) for model, prob in probabilities.items()},
        "weights": {model: round(weights[model], 4) for model in probabilities},
    }


def compute_baselines(history):
    """Baseline เทียบผล: 'ทายขึ้นตลอด' และ 'momentum' (ทายตามทิศทางวันก่อน)."""
    always_up = {"total": 0, "correct": 0}
    momentum = {"total": 0, "correct": 0}
    previous_dir = {}
    for date_str in sorted(history.keys()):
        entry = history[date_str]
        if not isinstance(entry, dict) or not entry.get("evaluated"):
            continue
        actuals = entry.get("actuals", {})
        for name in SYMBOLS:
            if name not in actuals:
                continue
            actual_dir = actuals[name]["actual_dir"]
            always_up["total"] += 1
            if actual_dir == "Up":
                always_up["correct"] += 1
            if name in previous_dir:
                momentum["total"] += 1
                if previous_dir[name] == actual_dir:
                    momentum["correct"] += 1
            previous_dir[name] = actual_dir

    def pct(counter):
        return round(counter["correct"] / counter["total"] * 100, 1) if counter["total"] else None

    return {
        "always_up_pct": pct(always_up),
        "momentum_pct": pct(momentum),
        "samples": always_up["total"],
    }


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
        "total_evaluated":      len(records),
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

    today_str           = run_dates["run_date"].isoformat()
    prediction_date_str = run_dates["prediction_date"].isoformat()

    history = remove_weekend_target_entries(preserve_historical_evaluations(load_history()))
    model_registry = load_json_file(MODEL_REGISTRY_FILE, {})
    price_history = load_json_file(PRICE_HISTORY_FILE, {})

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
                        "price_ml_pct":  None,
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
            # เก็บราคาดิบของวันที่ประเมินไว้ถาวร (audit trail ไม่ขึ้นกับ yfinance สด)
            ohlcv = ohlcv_for_market_date(df, expected_close_date)
            if ohlcv:
                price_history.setdefault(eval_target_date, {})[name] = ohlcv
            evaluate_shadow_predictions(
                entry,
                name,
                actual_pct,
                market_date=market_result["market_date"].isoformat(),
                previous_market_date=market_result["previous_market_date"].isoformat(),
            )
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
            save_json_file(PRICE_HISTORY_FILE, price_history)
        else:
            print("  ยังไม่มีราคาปิดของวันประเมิน จึงยังไม่บันทึกผล")
    else:
        print("  ไม่พบการทำนายที่ยังไม่ได้ประเมิน")

    validation_panel = build_validation_panel(history)

    # ── STEP 3: Compute adaptive N-arm weights (recency-aware + auto-tuned) ───
    ensemble_params = model_registry.get("ensemble_params") or {}
    arm_weights, arm_accuracy = compute_arm_weights(history, ensemble_params)
    if ensemble_params:
        print(f"  (auto-tuned: β={ensemble_params.get('beta')} "
              f"window={ensemble_params.get('window')} floor={ensemble_params.get('floor')})")
    print("\n--- ⚖️  น้ำหนักแขนทำนาย (เรียนรู้จากฟอร์มล่าสุด) ---")
    for name in SYMBOLS:
        parts = []
        for arm in ARM_NAMES:
            weight = arm_weights.get(name, {}).get(arm)
            if weight is None:
                continue
            acc = arm_accuracy.get(name, {}).get(arm, {})
            parts.append(f"{ARM_SHORT_KEYS[arm]}={weight:.2f}({acc.get('recent_accuracy_pct','n/a')}%)")
        if parts:
            print(f"  [{name}] " + "  ".join(parts))

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

    news_history = load_json_file(NEWS_HISTORY_FILE, {})
    news_snapshot = news_meta.get("snapshot")
    if news_snapshot:
        news_history[prediction_date_str] = news_snapshot
        save_json_file(NEWS_HISTORY_FILE, news_history)
    validation_panel["news_shadow"].update(compute_news_coverage(news_history))

    # ── STEP 4: Predict for the current Bangkok trading day ───────────────────
    print(f"\n--- 🔮 ทำนายสำหรับวันที่ {prediction_date_str} ---")

    prediction_values = {}
    pred_details      = {}
    display_weights   = {}
    shadow_predictions = {"price": {}, "price_news": {}}
    for name, sym in SYMBOLS.items():
        try:
            df = fetch_data(sym)
            if df is None:
                continue
            rf_pct    = train_predict_rf(df)
            arima_pct = train_predict_arima(df)

            news_data = news_results.get(name, {})
            news_pct  = news_data.get("news_score", 0.0) or 0.0

            # ── แขน price_ml = adaptive blend ของ candidate ML models ──────────
            price_ml_pct = None
            price_ml_info = None
            registry_entry = model_registry.get("symbols", {}).get(name, {})
            if registry_entry:
                raw_prices = yf.Ticker(sym).history(period="7y")
                price_ml_info = predict_price_ml_arm(raw_prices, registry_entry)
                if price_ml_info:
                    price_ml_pct = _probability_to_pct(price_ml_info["probability_up"])
                # คงไว้: shadow tracks (price / price+news) สำหรับเทียบใน backtest section
                champion = registry_entry.get("champion")
                if champion:
                    champion_signal = market_model.predict_price_signal(
                        raw_prices, model_name=champion, train_window=252,
                    )
                    for shadow_name, prediction in build_shadow_predictions(
                        model_registry, name, champion_signal, news_data,
                    ).items():
                        shadow_predictions[shadow_name][name] = prediction

            # ── ผสมทุกแขนด้วยน้ำหนัก adaptive (recency-aware) ─────────────────
            arm_pcts = {
                "legacy_rf":    rf_pct,
                "legacy_arima": arima_pct,
                "news":         news_pct,
                "price_ml":     price_ml_pct,
            }
            present = [arm for arm in ARM_NAMES if _arm_voted(arm, arm_pcts[arm])]
            today_weights = resolve_today_weights(arm_weights.get(name, {}), present)
            ensemble = sum(arm_pcts[arm] * today_weights[arm] for arm in present) if present else 0.0
            direction = "Up" if ensemble > 0 else "Down"

            prediction_values[name] = ensemble
            pred_details[name]      = {
                "predicted_pct": ensemble,
                "predicted_dir": direction,
                "rf_pct":        rf_pct,
                "arima_pct":     arima_pct,
                "news_pct":      news_pct,
                "price_ml_pct":  price_ml_pct,
                "price_ml_dir":  price_ml_info["predicted_dir"] if price_ml_info else None,
                "probability_up": price_ml_info["probability_up"] if price_ml_info else None,
                "price_ml_models": price_ml_info["per_model"] if price_ml_info else None,
                "news_info": {
                    "direction":     news_data.get("news_direction", "Neutral"),
                    "article_count": news_data.get("article_count", 0),
                    "avg_credibility": news_data.get("avg_credibility"),
                    "sentiment_raw": news_data.get("sentiment_raw"),
                },
                "weights": {ARM_SHORT_KEYS[arm]: round(today_weights.get(arm, 0.0), 3) for arm in ARM_NAMES},
                "model_source": "adaptive_ensemble",
            }
            # น้ำหนักจริงที่ใช้ทายวันนี้ (normalize รวม=100% เฉพาะแขนที่ออกเสียง) สำหรับการ์ด
            display_weights[name] = {
                **{ARM_SHORT_KEYS[arm]: round(today_weights.get(arm, 0.0), 3) for arm in ARM_NAMES},
                "samples": max(
                    (arm_accuracy.get(name, {}).get(arm, {}).get("samples", 0) for arm in ARM_NAMES),
                    default=0,
                ),
            }
            icon = "📈" if ensemble > 0 else "📉"
            pml = f"{price_ml_pct*100:.2f}%" if price_ml_pct is not None else "n/a"
            print(f"  [{name}] {direction} {icon} | "
                  f"RF:{rf_pct*100:.2f}%  ARIMA:{arima_pct*100:.2f}%  "
                  f"News:{news_pct*100:.3f}%  priceML:{pml}  → Ensemble:{ensemble*100:.2f}%")
        except Exception as e:
            print(f"  [{name}] Error: {e}")

    history[prediction_date_str] = {
        "for_date":    prediction_date_str,
        "made_on":     today_str,
        "predictions": pred_details,
        "actuals":     {},
        "shadow_predictions": shadow_predictions,
        "shadow_actuals": {},
        "evaluated":   False,
        "eval_date":   None,
    }
    save_history(history)

    # ── STEP 5: Cumulative stats + baselines ──────────────────────────────────
    stats = compute_stats(history)
    stats["baselines"] = compute_baselines(history)
    print(f"\n--- 📈 สถิติสะสม ---")
    print(f"  ความแม่นยำรวม: {stats['overall_accuracy_pct']}% "
          f"({stats['total_evaluated']} วันที่ประเมินแล้ว)")
    print(f"  Baseline: ทายขึ้นตลอด={stats['baselines']['always_up_pct']}%  "
          f"momentum={stats['baselines']['momentum_pct']}%")
    for name, s in stats["per_symbol"].items():
        print(f"    {name}: {s['accuracy_pct']}%  ({s['correct']}/{s['total']})")

    # ── STEP 6: Write dashboard_data.json ─────────────────────────────────────
    dashboard_data = {
        "last_updated":         datetime.now(BANGKOK_TZ).isoformat(),
        "prediction_for_date":  prediction_date_str,
        "tomorrow_predictions": prediction_values,
        "tomorrow_details":     pred_details,
        "model_weights":        display_weights,
        "arm_accuracy":         arm_accuracy,
        "evaluation": {
            "prediction_was_for": eval_target_date,
            "made_on": history.get(eval_target_date, {}).get("made_on") if eval_target_date else None,
            "results": eval_results,
        },
        "model_validation":     validation_panel,
        "ensemble_params":      ensemble_params,
        "stats":  stats,
        # ข่าว: เฉพาะแหล่งน่าเชื่อถือ ≥ 80%
        "news": news_meta.get("headlines", []),
        "news_fetch_stats": {
            "total_fetched":   news_meta.get("total_fetched", 0),
            "accepted":        news_meta.get("accepted", 0),
            "rejected":        news_meta.get("rejected", 0),
            "min_credibility": news_meta.get("min_credibility", 80),
        },
    }
    with open(DASHBOARD_FILE, 'w') as f:
        json.dump(sanitize(dashboard_data), f, indent=4)

    print("\n✅ dashboard_data.json อัปเดตเรียบร้อย")


if __name__ == "__main__":
    main()
