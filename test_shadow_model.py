import unittest

try:
    import shadow_model
except ModuleNotFoundError:
    shadow_model = None


class NewsShadowTests(unittest.TestCase):
    def test_news_shadow_adjusts_probability_without_becoming_active_early(self):
        self.assertIsNotNone(shadow_model)

        shadow = shadow_model.build_news_shadow_prediction(
            {"probability_up": 0.49, "predicted_dir": "Down", "model": "logistic"},
            {"news_score": 0.02, "article_count": 3},
        )

        self.assertGreater(shadow["probability_up"], 0.49)
        self.assertEqual(shadow["model"], "price_news_shadow")
        self.assertFalse(shadow_model.shadow_promotion_status([
            {"for_date": "2026-01-01", "correct": True},
        ])["promotion_ready"])

    def test_shadow_requires_60_market_days_and_75_percent_accuracy(self):
        self.assertIsNotNone(shadow_model)
        records = []
        for day in range(60):
            date = f"2026-03-{day + 1:02d}" if day < 31 else f"2026-04-{day - 30:02d}"
            records.append({"for_date": date, "correct": day < 45})

        status = shadow_model.shadow_promotion_status(records)

        self.assertEqual(status["market_days"], 60)
        self.assertEqual(status["accuracy_pct"], 75.0)
        self.assertTrue(status["promotion_ready"])


if __name__ == "__main__":
    unittest.main()
