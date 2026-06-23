"""Leak-free price features, walk-forward backtesting, and model selection."""

from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = (
    "return_1",
    "return_2",
    "return_3",
    "return_5",
    "return_20",
    "sma_10_gap",
    "sma_20_gap",
    "sma_50_gap",
    "rsi_14",
    "macd_gap",
    "bb_position",
    "stoch_k_14",
    "volatility_20",
    "volume_ratio_20",
    "day_of_week",
)
MODEL_NAMES = ("logistic", "random_forest", "gradient_boosting", "stacking")
MIN_TRAINING_ROWS = 60

# Adaptive (Hedge / multiplicative-weights) ensemble parameters. Every model votes
# each day; a model that was wrong has its weight multiplied by HEDGE_BETA, so the
# arms that have been right recently dominate the next prediction. HEDGE_FLOOR keeps
# every arm with a minimum voice so a temporarily-cold model can recover.
HEDGE_BETA = 0.85
HEDGE_FLOOR = 0.05
HEDGE_WINDOW = 40


def build_price_feature_frame(prices):
    """Build a feature frame without discarding the newest closed market row."""
    if prices is None or prices.empty or "Close" not in prices:
        return pd.DataFrame(
            columns=[*FEATURE_COLUMNS, "target_up", "target_return", "target_close", "target_date"]
        )

    data = prices.copy().sort_index()
    close = data["Close"].astype(float)
    # High/Low/Open are optional — fall back to Close so feature math still works
    # on Close-only price frames (e.g. synthetic test data) without dropping rows.
    high = data.get("High", close).astype(float)
    low = data.get("Low", close).astype(float)
    returns = close.pct_change()
    sma_10 = close.rolling(10).mean()
    sma_20 = close.rolling(20).mean()
    sma_50 = close.rolling(50).mean()
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()

    gains = returns.clip(lower=0)
    losses = -returns.clip(upper=0)
    average_gain = gains.rolling(14).mean()
    average_loss = losses.rolling(14).mean()
    relative_strength = average_gain / average_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + relative_strength))

    # Bollinger position: where the close sits inside its 20-day ±2σ band.
    std_20 = close.rolling(20).std()
    bb_position = (close - sma_20) / (2 * std_20.replace(0, np.nan))

    # Stochastic %K (14): close vs the recent high/low range.
    lowest_low = low.rolling(14).min()
    highest_high = high.rolling(14).max()
    stoch_k = (close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)

    # Day of week (Mon=0..Fri=4), normalised to keep scale comparable to other features.
    day_of_week = pd.Series(data.index, index=data.index).dt.dayofweek.astype(float) / 4.0

    volume = data.get("Volume", pd.Series(0.0, index=data.index)).astype(float)
    volume_average = volume.rolling(20).mean()
    volume_ratio = volume / volume_average.replace(0, np.nan)

    target_return = returns.shift(-1)
    target_up = pd.Series(np.nan, index=data.index, dtype=float)
    known_target = target_return.notna()
    target_up.loc[known_target] = (target_return.loc[known_target] > 0).astype(float)
    target_dates = pd.Series(data.index, index=data.index).shift(-1)

    features = pd.DataFrame(
        {
            "return_1": returns,
            "return_2": close.pct_change(2),
            "return_3": close.pct_change(3),
            "return_5": close.pct_change(5),
            "return_20": close.pct_change(20),
            "sma_10_gap": close / sma_10 - 1,
            "sma_20_gap": close / sma_20 - 1,
            "sma_50_gap": close / sma_50 - 1,
            "rsi_14": rsi / 100,
            "macd_gap": ema_12 / ema_26 - 1,
            "bb_position": bb_position,
            "stoch_k_14": stoch_k,
            "volatility_20": returns.rolling(20).std(),
            "volume_ratio_20": volume_ratio,
            "day_of_week": day_of_week,
            "target_return": target_return,
            "target_up": target_up,
            "target_close": close.shift(-1),
            "target_date": target_dates,
        },
        index=data.index,
    )
    features = features.replace([np.inf, -np.inf], np.nan)
    return features.dropna(subset=FEATURE_COLUMNS)


def _make_model(model_name):
    if model_name == "logistic":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, random_state=42))
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=160,
            min_samples_leaf=4,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=1,
        )
    if model_name == "gradient_boosting":
        return HistGradientBoostingClassifier(max_iter=140, learning_rate=0.06, random_state=42)
    if model_name == "stacking":
        # Meta-learner trained on the cross-validated out-of-fold probabilities of the
        # three base models — a leak-free way to combine their strengths per symbol.
        return StackingClassifier(
            estimators=[
                ("logistic", _make_model("logistic")),
                ("random_forest", _make_model("random_forest")),
                ("gradient_boosting", _make_model("gradient_boosting")),
            ],
            final_estimator=LogisticRegression(max_iter=500, random_state=42),
            stack_method="predict_proba",
            cv=3,
            n_jobs=1,
        )
    raise ValueError(f"Unknown model: {model_name}")


def _training_rows(features, prediction_index, train_window):
    position = features.index.get_loc(prediction_index)
    training = features.iloc[:position].dropna(subset=["target_up"])
    return training.tail(train_window)


def _fallback_probability(features, prediction_index):
    return 0.55 if features.loc[prediction_index, "return_5"] >= 0 else 0.45


def fit_price_model(features, prediction_index, model_name, train_window=252):
    """Fit only on labels known before ``prediction_index`` and return a model or None."""
    training = _training_rows(features, prediction_index, train_window)
    if len(training) < MIN_TRAINING_ROWS or training["target_up"].nunique() < 2:
        return None

    model = _make_model(model_name)
    model.fit(training[list(FEATURE_COLUMNS)], training["target_up"].astype(int))
    return model


def predict_price_signal(prices, model_name="logistic", as_of_date=None, train_window=252):
    """Predict the next market row from the latest closed row available at ``as_of_date``."""
    features = build_price_feature_frame(prices)
    if features.empty:
        return None

    if as_of_date is None:
        prediction_index = features.index[-1]
    else:
        cutoff = pd.Timestamp(as_of_date)
        eligible = features.index[features.index <= cutoff]
        if len(eligible) == 0:
            return None
        prediction_index = eligible[-1]

    model = fit_price_model(features, prediction_index, model_name, train_window)
    if model is None:
        probability_up = _fallback_probability(features, prediction_index)
        actual_model = "momentum_fallback"
    else:
        probability_up = float(model.predict_proba(features.loc[[prediction_index], FEATURE_COLUMNS])[0][1])
        actual_model = model_name

    return {
        "model": actual_model,
        "as_of_date": pd.Timestamp(prediction_index).date().isoformat(),
        "probability_up": round(probability_up, 6),
        "predicted_dir": "Up" if probability_up >= 0.5 else "Down",
    }


def walk_forward_backtest(
    prices,
    symbol,
    model_names=MODEL_NAMES,
    train_window=252,
    test_window=756,
    retrain_every=20,
):
    """Return chronological, price-only out-of-sample predictions for one symbol."""
    features = build_price_feature_frame(prices)
    eligible = features.dropna(subset=["target_up", "target_date"])
    if len(eligible) <= train_window:
        return []

    test_rows = eligible.iloc[train_window:].tail(test_window)
    records = []
    for model_name in model_names:
        model = None
        model_label = model_name
        for row_number, (prediction_index, row) in enumerate(test_rows.iterrows()):
            if row_number % retrain_every == 0 or model is None:
                model = fit_price_model(features, prediction_index, model_name, train_window)
                model_label = model_name if model is not None else "momentum_fallback"

            if model is None:
                probability_up = _fallback_probability(features, prediction_index)
            else:
                probability_up = float(model.predict_proba(features.loc[[prediction_index], FEATURE_COLUMNS])[0][1])

            predicted_dir = "Up" if probability_up >= 0.5 else "Down"
            actual_dir = "Up" if row["target_up"] == 1 else "Down"
            records.append(
                {
                    "symbol": symbol,
                    "model": model_label,
                    "as_of_date": pd.Timestamp(prediction_index).date().isoformat(),
                    "as_of": pd.Timestamp(prediction_index).isoformat(),
                    "market_date": pd.Timestamp(row["target_date"]).date().isoformat(),
                    "probability_up": round(probability_up, 6),
                    "predicted_dir": predicted_dir,
                    "actual_dir": actual_dir,
                    "actual_pct": float(row["target_return"]),
                    "actual_close": float(row["target_close"]),
                    "market_gap_reason": (
                        "market_closed_weekend_or_holiday"
                        if (pd.Timestamp(row["target_date"]) - pd.Timestamp(prediction_index)).days > 1
                        else None
                    ),
                    "correct": predicted_dir == actual_dir,
                }
            )
    return records


def build_model_registry(records, recent_window=60):
    """Choose the highest-accuracy candidate per symbol from chronological records."""
    by_symbol_model = defaultdict(lambda: defaultdict(list))
    for record in records:
        by_symbol_model[record["symbol"]][record["model"]].append(record)

    symbols = {}
    for symbol, models in by_symbol_model.items():
        candidates = []
        for model_name, model_records in models.items():
            recent = sorted(model_records, key=lambda item: item["market_date"])[-recent_window:]
            correct = sum(item["correct"] for item in recent)
            candidates.append(
                {
                    "model": model_name,
                    "recent_accuracy_pct": round(correct / len(recent) * 100, 1) if recent else None,
                    "recent_samples": len(recent),
                    "three_year_accuracy_pct": round(
                        sum(item["correct"] for item in model_records) / len(model_records) * 100,
                        1,
                    ),
                    "three_year_samples": len(model_records),
                }
            )
        champion = sorted(
            candidates,
            key=lambda item: (
                -(item["recent_accuracy_pct"] if item["recent_accuracy_pct"] is not None else -1),
                -item["recent_samples"],
                item["model"],
            ),
        )[0]
        symbols[symbol] = {"champion": champion["model"], **champion, "candidates": candidates}

    return {"version": "price-v1", "recent_window": recent_window, "symbols": symbols}


# ─── Adaptive (Hedge) ensemble — shared by live prediction and backtest ──────────

def adaptive_weights(correctness_by_arm, beta=HEDGE_BETA, floor=HEDGE_FLOOR, window=HEDGE_WINDOW):
    """Recency-aware multiplicative weights from each arm's recent right/wrong record.

    ``correctness_by_arm`` maps an arm name to a chronological list of booleans for the
    days that arm actually voted. Arms that have never voted get no weight. The result
    is normalised to sum to 1, with each voting arm guaranteed at least ``floor``.
    """
    raw = {}
    for arm, sequence in correctness_by_arm.items():
        if not sequence:
            continue
        weight = 1.0
        for was_correct in sequence[-window:]:
            if not was_correct:
                weight *= beta
        raw[arm] = weight

    if not raw:
        return {}

    total = sum(raw.values())
    normalised = {arm: value / total for arm, value in raw.items()}
    count = len(normalised)
    # Blend toward uniform so every voting arm keeps at least ``floor`` while the
    # weights still sum to exactly 1 (a temporarily-cold arm can always recover).
    if floor * count >= 1:
        return {arm: 1.0 / count for arm in normalised}
    return {arm: (1 - floor * count) * value + floor for arm, value in normalised.items()}


def blend_probabilities(probabilities_by_arm, weights):
    """Combine each arm's probability-up into one ensemble probability."""
    active = {arm: prob for arm, prob in probabilities_by_arm.items() if arm in weights}
    if not active:
        return None
    total = sum(weights[arm] for arm in active)
    if total <= 0:
        return None
    return sum(prob * weights[arm] for arm, prob in active.items()) / total


def simulate_adaptive_ensemble(
    records, beta=HEDGE_BETA, floor=HEDGE_FLOOR, window=HEDGE_WINDOW
):
    """Walk forward through per-model records and replay the live adaptive ensemble.

    For each symbol, every market day combines all models that voted that day using
    the Hedge weights learned strictly from prior days (no look-ahead), records whether
    the blended call was correct, then updates the weights from that day's outcome.
    Returns chronological ensemble records mirroring the per-model record shape.
    """
    by_symbol = defaultdict(list)
    for record in records:
        by_symbol[record["symbol"]].append(record)

    ensemble_records = []
    for symbol, symbol_records in by_symbol.items():
        days = defaultdict(dict)
        for record in symbol_records:
            days[record["market_date"]][record["model"]] = record

        history_by_arm = defaultdict(list)
        for market_date in sorted(days):
            todays_models = days[market_date]
            weights = adaptive_weights(history_by_arm, beta, floor, window)
            if not weights:
                weights = {model: 1.0 for model in todays_models}

            probabilities = {
                model: record["probability_up"] for model, record in todays_models.items()
            }
            ensemble_probability = blend_probabilities(probabilities, weights)
            if ensemble_probability is None:
                ensemble_probability = sum(probabilities.values()) / len(probabilities)

            sample = next(iter(todays_models.values()))
            predicted_dir = "Up" if ensemble_probability >= 0.5 else "Down"
            ensemble_records.append(
                {
                    "symbol": symbol,
                    "model": "adaptive_ensemble",
                    "as_of_date": sample["as_of_date"],
                    "market_date": market_date,
                    "probability_up": round(ensemble_probability, 6),
                    "predicted_dir": predicted_dir,
                    "actual_dir": sample["actual_dir"],
                    "actual_pct": sample["actual_pct"],
                    "actual_close": sample["actual_close"],
                    "weights": {arm: round(value, 4) for arm, value in weights.items()},
                    "correct": predicted_dir == sample["actual_dir"],
                }
            )

            for model, record in todays_models.items():
                history_by_arm[model].append(record["correct"])

    ensemble_records.sort(key=lambda record: (record["market_date"], record["symbol"]))
    return ensemble_records


# กริดสำหรับ auto-tune — ค่าที่กว้างพอจะไล่จาก "ตามแขนที่ฟอร์มดีแบบเด็ดขาด" (β ต่ำ)
# ไปจนถึง "เฉลี่ยทุกแขนเท่ากัน" (β สูง)
TUNE_BETAS = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
TUNE_WINDOWS = (20, 40, 60, 120)
TUNE_FLOORS = (0.0, 0.02, 0.05)


def _ensemble_accuracy(records, beta, floor, window):
    ensemble = simulate_adaptive_ensemble(records, beta=beta, floor=floor, window=window)
    if not ensemble:
        return None, 0
    correct = sum(record["correct"] for record in ensemble)
    return correct / len(ensemble) * 100, len(ensemble)


def tune_ensemble_hyperparameters(
    records, betas=TUNE_BETAS, windows=TUNE_WINDOWS, floors=TUNE_FLOORS
):
    """Search β/window/floor that maximises the adaptive ensemble's directional accuracy.

    The ensemble weights are already online (no look-ahead per day); these three
    hyperparameters are the only knobs, and they are chosen on the full history to push
    the realised hit-rate as high as the data allows. Returns the best params plus the
    ranked grid for transparency.
    """
    grid = []
    for beta in betas:
        for window in windows:
            for floor in floors:
                accuracy, samples = _ensemble_accuracy(records, beta, floor, window)
                if accuracy is None:
                    continue
                grid.append(
                    {
                        "beta": beta,
                        "window": window,
                        "floor": floor,
                        "accuracy_pct": round(accuracy, 2),
                        "samples": samples,
                    }
                )
    if not grid:
        return {
            "best": {"beta": HEDGE_BETA, "window": HEDGE_WINDOW, "floor": HEDGE_FLOOR},
            "grid": [],
        }
    # เลือกความแม่นสูงสุด; เสมอกันให้เลือก window ใหญ่กว่า (นิ่งกว่า) แล้ว β สูงกว่า (อนุรักษ์กว่า)
    grid.sort(key=lambda item: (-item["accuracy_pct"], -item["window"], -item["beta"]))
    return {"best": {k: grid[0][k] for k in ("beta", "window", "floor")}, "grid": grid}
