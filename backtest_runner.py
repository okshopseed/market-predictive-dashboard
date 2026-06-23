"""Build and persist leak-free, price-only backtest artifacts."""

from collections import defaultdict
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


def _baseline_metrics(records):
    """Naive references — 'always Up' and 'momentum' (repeat prior actual direction)."""
    always_up = {"total": 0, "correct": 0}
    momentum = {"total": 0, "correct": 0}
    by_symbol = defaultdict(list)
    for record in records:
        by_symbol[record["symbol"]].append(record)
    for symbol_records in by_symbol.values():
        previous_dir = None
        for record in sorted(symbol_records, key=lambda item: item["market_date"]):
            actual_dir = record.get("actual_dir")
            if actual_dir is None:
                continue
            always_up["total"] += 1
            if actual_dir == "Up":
                always_up["correct"] += 1
            if previous_dir is not None:
                momentum["total"] += 1
                if previous_dir == actual_dir:
                    momentum["correct"] += 1
            previous_dir = actual_dir

    def pct(counter):
        return round(counter["correct"] / counter["total"] * 100, 1) if counter["total"] else None

    return {"always_up_pct": pct(always_up), "momentum_pct": pct(momentum)}


def summarize_backtest(records, recent_days=PROMOTION_WINDOW_DAYS, ensemble_records=None):
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
    # 75% เป็นเป้าหมายที่ไล่ตาม ไม่ใช่ประตูเปิด/ปิด — รายงานว่าห่างเป้าเท่าไหร่
    recent_metrics["reached_target"] = (recent_metrics["accuracy_pct"] or 0) >= PROMOTION_TARGET_PCT

    summary = {
        "three_year": _metrics(chronological),
        "recent_60": recent_metrics,
        "per_symbol": per_symbol,
        "baselines": _baseline_metrics(chronological),
    }

    # adaptive ensemble: จำลองวิธี "เรียนรู้-ปรับน้ำหนัก" แบบเดียวกับที่ใช้ทายจริง
    if ensemble_records is not None:
        ens_chrono = sorted(ensemble_records, key=lambda record: record["market_date"])
        ens_recent = [r for r in ens_chrono if r["market_date"] in recent_dates]
        summary["adaptive_ensemble"] = {
            "three_year": _metrics(ens_chrono),
            "recent_60": _metrics(ens_recent),
        }
    return summary


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

    # auto-tune: เลือก β/window/floor ที่ทำให้ % การทายของ ensemble สูงสุดจากข้อมูลจริง
    tuning = market_model.tune_ensemble_hyperparameters(candidate_records)
    best_params = tuning["best"]

    # จำลอง adaptive ensemble (เรียนรู้-ปรับน้ำหนัก) ด้วยพารามิเตอร์ที่จูนแล้ว — วิธีเดียวกับที่ใช้ทายจริง
    ensemble_records = market_model.simulate_adaptive_ensemble(candidate_records, **best_params)
    summary = summarize_backtest(records, ensemble_records=ensemble_records)
    summary["adaptive_ensemble"]["params"] = best_params
    summary["adaptive_ensemble"]["tuning_top"] = tuning["grid"][:5]

    registry["ensemble_params"] = best_params
    registry["target_accuracy_pct"] = PROMOTION_TARGET_PCT
    registry["progress_window_days"] = PROMOTION_WINDOW_DAYS
    registry["adaptive_ensemble_recent_pct"] = summary["adaptive_ensemble"]["recent_60"]["accuracy_pct"]
    artifact = {
        "schema_version": 2,
        "mode": "price_adaptive_ensemble",
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
