"""Build and persist leak-free, price-only backtest artifacts."""

from datetime import datetime, timezone
import json
from pathlib import Path

import market_model


BACKTEST_DATA_FILE = Path("backtest_data.json")
MODEL_REGISTRY_FILE = Path("model_registry.json")
PROMOTION_TARGET_PCT = 75.0
PROMOTION_WINDOW_DAYS = 60


def _metrics(records):
    correct = sum(record["correct"] for record in records)
    return {
        "samples": len(records),
        "correct": correct,
        "accuracy_pct": round(correct / len(records) * 100, 1) if records else None,
    }


def summarize_backtest(records, recent_days=PROMOTION_WINDOW_DAYS):
    chronological = sorted(records, key=lambda record: record["market_date"])
    market_dates = sorted({record["market_date"] for record in chronological})
    recent_dates = set(market_dates[-recent_days:])
    recent = [record for record in chronological if record["market_date"] in recent_dates]

    per_symbol = {}
    for symbol in sorted({record["symbol"] for record in chronological}):
        per_symbol[symbol] = _metrics([record for record in chronological if record["symbol"] == symbol])

    recent_metrics = _metrics(recent)
    recent_metrics["market_days"] = len(recent_dates)
    recent_metrics["target_accuracy_pct"] = PROMOTION_TARGET_PCT
    recent_metrics["promotion_ready"] = (
        len(recent_dates) >= recent_days
        and (recent_metrics["accuracy_pct"] or 0) >= PROMOTION_TARGET_PCT
    )
    return {
        "three_year": _metrics(chronological),
        "recent_60": recent_metrics,
        "per_symbol": per_symbol,
    }


def build_price_backtest(
    symbols,
    fetch_prices,
    model_names=market_model.MODEL_NAMES,
    train_window=252,
    test_window=756,
    retrain_every=20,
):
    """Run candidate models and retain each symbol's selected champion predictions."""
    candidate_records = []
    for name, ticker in symbols.items():
        prices = fetch_prices(ticker)
        candidate_records.extend(
            market_model.walk_forward_backtest(
                prices,
                symbol=name,
                model_names=model_names,
                train_window=train_window,
                test_window=test_window,
                retrain_every=retrain_every,
            )
        )

    registry = market_model.build_model_registry(candidate_records)
    champions = {
        name: entry["champion"]
        for name, entry in registry["symbols"].items()
    }
    records = [
        record for record in candidate_records
        if champions.get(record["symbol"]) == record["model"]
    ]
    records.sort(key=lambda record: (record["market_date"], record["symbol"]))
    summary = summarize_backtest(records)

    registry["promotion_target_pct"] = PROMOTION_TARGET_PCT
    registry["promotion_window_days"] = PROMOTION_WINDOW_DAYS
    registry["price_shadow_promotion_ready"] = summary["recent_60"]["promotion_ready"]
    registry["active_model"] = "price"
    artifact = {
        "schema_version": 1,
        "mode": "price_only",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "models": registry["symbols"],
        "records": records,
    }
    return artifact, registry


def write_backtest_artifacts(artifact, registry, data_file=BACKTEST_DATA_FILE, registry_file=MODEL_REGISTRY_FILE):
    try:
        previous_registry = json.loads(registry_file.read_text())
    except (OSError, ValueError):
        previous_registry = {}
    registry = dict(registry)
    registry["active_model"] = previous_registry.get(
        "active_model", registry.get("active_model", "price")
    )
    data_file.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n")
    registry_file.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n")
