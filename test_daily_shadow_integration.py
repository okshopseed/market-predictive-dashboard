import unittest

import daily_predictor


class DailyShadowIntegrationTests(unittest.TestCase):
    def test_shadow_predictions_use_registry_champion_and_keep_price_and_news_variants(self):
        registry = {"symbols": {"TEST": {"champion": "logistic"}}}
        price_signal = {"model": "logistic", "probability_up": 0.61, "predicted_dir": "Up"}
        news_signal = {"news_score": -0.02, "article_count": 2}

        predictions = daily_predictor.build_shadow_predictions(
            registry,
            "TEST",
            price_signal,
            news_signal,
        )

        self.assertEqual(daily_predictor.registry_champion(registry, "TEST"), "logistic")
        self.assertEqual(predictions["price"]["model"], "logistic")
        self.assertEqual(predictions["price"]["predicted_dir"], "Up")
        self.assertEqual(predictions["price_news"]["model"], "price_news_shadow")
        self.assertIn("predicted_pct", predictions["price_news"])

    def test_shadow_actuals_are_recorded_separately_from_active_actuals(self):
        entry = {
            "actuals": {},
            "shadow_predictions": {
                "price": {"TEST": {"predicted_dir": "Up", "probability_up": 0.61}},
                "price_news": {"TEST": {"predicted_dir": "Down", "probability_up": 0.44}},
            },
            "shadow_actuals": {},
        }

        daily_predictor.evaluate_shadow_predictions(entry, "TEST", actual_pct=0.01)

        self.assertEqual(entry["shadow_actuals"]["price"]["TEST"]["correct"], True)
        self.assertEqual(entry["shadow_actuals"]["price_news"]["TEST"]["correct"], False)
        self.assertEqual(entry["actuals"], {})

    def test_shadow_progress_is_calculated_from_history_records(self):
        history = {
            "2026-01-01": {
                "shadow_actuals": {"price_news": {"TEST": {"correct": True}}},
            },
            "2026-01-02": {
                "shadow_actuals": {"price_news": {"TEST": {"correct": False}}},
            },
        }

        progress = daily_predictor.compute_shadow_progress(history, "price_news")

        self.assertEqual(progress["market_days"], 2)
        self.assertEqual(progress["samples"], 2)
        self.assertEqual(progress["accuracy_pct"], 50.0)

    def test_validation_panel_reports_progress_without_a_gate(self):
        history = {
            "2026-01-01": {"shadow_actuals": {
                "price": {"TEST": {"correct": True}},
                "price_news": {"TEST": {"correct": True}},
            }},
            "2026-01-02": {"shadow_actuals": {
                "price": {"TEST": {"correct": False}},
                "price_news": {"TEST": {"correct": True}},
            }},
        }

        panel = daily_predictor.build_validation_panel(history)

        self.assertEqual(panel["target_accuracy_pct"], 75.0)
        self.assertEqual(panel["price_shadow"]["accuracy_pct"], 50.0)
        self.assertEqual(panel["news_shadow"]["accuracy_pct"], 100.0)
        self.assertNotIn("active_model", panel)

    def test_news_coverage_reports_collected_days_and_remaining_decision_days(self):
        news_history = {
            "2026-01-01": {"stats": {"eligible": 3}},
            "2026-01-02": {"stats": {"eligible": 0}},
            "2026-01-03": {"stats": {"eligible": 1}},
        }

        coverage = daily_predictor.compute_news_coverage(news_history)

        self.assertEqual(coverage["days_collected"], 3)
        self.assertEqual(coverage["days_with_eligible_news"], 2)
        self.assertEqual(coverage["coverage_pct"], 66.7)
        self.assertEqual(coverage["remaining_decision_days"], 57)


if __name__ == "__main__":
    unittest.main()
