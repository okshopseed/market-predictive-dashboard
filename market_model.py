"""Leak-free price features, walk-forward backtesting, and model selection."""

from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = (
    "return_1",
    "return_5",
    "return_20",
    "sma_10_gap",
    "sma_20_gap",
    "sma_50_gap",
    "rsi_14",
    "macd_gap",
    "volatility_20",
    "volume_ratio_20",
)
MODEL_NAMES = ("logistic", "random_forest", "gradient_boosting")
MIN_TRAINING_ROWS = 60


def build_price_feature_frame(prices):
    """Build a feature frame without discarding the newest closed market row."""
    if prices is None or prices.empty or "Close" not in prices:
        return pd.DataFrame(
            columns=[*FEATURE_COLUMNS, "target_up", "target_return", "target_close", "target_date"]
        )

    data = prices.copy().sort_index()
    close = data["Close"].astype(float)
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
            "return_5": close.pct_change(5),
            "return_20": close.pct_change(20),
            "sma_10_gap": close / sma_10 - 1,
            "sma_20_gap": close / sma_20 - 1,
            "sma_50_gap": close / sma_50 - 1,
            "rsi_14": rsi / 100,
            "macd_gap": ema_12 / ema_26 - 1,
            "volatility_20": returns.rolling(20).std(),
            "volume_ratio_20": volume_ratio,
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
