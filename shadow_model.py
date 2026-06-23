"""News blending and target-progress reporting for live daily predictions.

The 75% figure is a *target to track*, not a gate. Every model/arm always votes in the
adaptive ensemble; this module only reports how close recent accuracy is to the target
so the dashboard can show progress over time.
"""


TARGET_ACCURACY_PCT = 75.0
PROGRESS_WINDOW_DAYS = 60
MAX_NEWS_ADJUSTMENT = 0.08

# Backwards-compatible aliases (older imports referenced these names).
PROMOTION_TARGET_PCT = TARGET_ACCURACY_PCT
PROMOTION_WINDOW_DAYS = PROGRESS_WINDOW_DAYS


def build_news_shadow_prediction(price_signal, news_signal):
    """Blend the current news sentiment into a price probability (bounded adjustment)."""
    base_probability = float(price_signal.get("probability_up", 0.5))
    news_score = float(news_signal.get("news_score", 0.0) or 0.0)
    adjustment = max(-MAX_NEWS_ADJUSTMENT, min(MAX_NEWS_ADJUSTMENT, news_score * 2.5))
    probability_up = max(0.01, min(0.99, base_probability + adjustment))
    return {
        "model": "price_news_shadow",
        "probability_up": round(probability_up, 6),
        "predicted_dir": "Up" if probability_up >= 0.5 else "Down",
        "price_model": price_signal.get("model"),
        "news_score": news_score,
        "article_count": int(news_signal.get("article_count", 0) or 0),
    }


def target_progress(records, target_pct=TARGET_ACCURACY_PCT, window_days=PROGRESS_WINDOW_DAYS):
    """Report recent accuracy and how far it is from the target (informational only)."""
    market_days = sorted({record["for_date"] for record in records})[-window_days:]
    eligible = [record for record in records if record["for_date"] in set(market_days)]
    correct = sum(record.get("correct", False) for record in eligible)
    accuracy_pct = round(correct / len(eligible) * 100, 1) if eligible else None
    gap = round(target_pct - accuracy_pct, 1) if accuracy_pct is not None else None
    return {
        "market_days": len(market_days),
        "samples": len(eligible),
        "correct": correct,
        "accuracy_pct": accuracy_pct,
        "target_accuracy_pct": target_pct,
        "gap_to_target_pct": gap,
        "reached_target": accuracy_pct is not None and accuracy_pct >= target_pct,
    }
