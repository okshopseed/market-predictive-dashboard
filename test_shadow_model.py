import unittest

try:
    import shadow_model
except ModuleNotFoundError:
    shadow_model = None


class NewsShadowTests(unittest.TestCase):
    def test_news_shadow_adjusts_probability_as_a_bounded_blend(self):
        self.assertIsNotNone(shadow_model)

        shadow = shadow_model.build_news_shadow_prediction(
            {"probability_up": 0.49, "predicted_dir": "Down", "model": "logistic"},
            {"news_score": 0.02, "article_count": 3},
        )

        self.assertGreater(shadow["probability_up"], 0.49)
        self.assertEqual(shadow["model"], "price_news_shadow")

    def test_target_progress_reports_accuracy_and_gap_without_gating(self):
        self.assertIsNotNone(shadow_model)
        records = []
        for day in range(60):
            date = f"2026-03-{day + 1:02d}" if day < 31 else f"2026-04-{day - 30:02d}"
            records.append({"for_date": date, "correct": day < 45})

        status = shadow_model.target_progress(records)

        self.assertEqual(status["market_days"], 60)
        self.assertEqual(status["accuracy_pct"], 75.0)
        self.assertEqual(status["gap_to_target_pct"], 0.0)
        self.assertTrue(status["reached_target"])

    def test_target_progress_handles_short_history(self):
        self.assertIsNotNone(shadow_model)

        status = shadow_model.target_progress([{"for_date": "2026-01-01", "correct": True}])

        self.assertEqual(status["accuracy_pct"], 100.0)
        self.assertIn("gap_to_target_pct", status)


if __name__ == "__main__":
    unittest.main()
