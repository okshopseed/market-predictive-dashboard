"""News-shadow scoring and promotion gates for live daily predictions."""


PROMOTION_TARGET_PCT = 75.0
PROMOTION_WINDOW_DAYS = 60
MAX_NEWS_ADJUSTMENT = 0.08


def build_news_shadow_prediction(price_signal, news_signal):
    """Blend the current news sentiment into a shadow-only probability."""
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


def shadow_promotion_status(records, target_pct=PROMOTION_TARGET_PCT, window_days=PROMOTION_WINDOW_DAYS):
    """Calculate a pooled accuracy gate using the last distinct market days."""
    market_days = sorted({record["for_date"] for record in records})[-window_days:]
    eligible = [record for record in records if record["for_date"] in set(market_days)]
    correct = sum(record.get("correct", False) for record in eligible)
    accuracy_pct = round(correct / len(eligible) * 100, 1) if eligible else None
    return {
        "market_days": len(market_days),
        "samples": len(eligible),
        "correct": correct,
        "accuracy_pct": accuracy_pct,
        "target_accuracy_pct": target_pct,
        "promotion_ready": (
            len(market_days) >= window_days
            and accuracy_pct is not None
            and accuracy_pct >= target_pct
        ),
    }
