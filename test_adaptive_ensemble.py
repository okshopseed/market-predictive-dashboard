import unittest

import daily_predictor
import market_model
from test_market_model import make_prices


class AdaptiveWeightTests(unittest.TestCase):
    def test_recent_winner_gets_more_weight_than_recent_loser(self):
        weights = market_model.adaptive_weights(
            {
                "winner": [True] * 20,
                "loser": [False] * 20,
            }
        )

        self.assertAlmostEqual(sum(weights.values()), 1.0, places=6)
        self.assertGreater(weights["winner"], weights["loser"])
        # floor keeps the cold arm in the game
        self.assertGreaterEqual(weights["loser"], market_model.HEDGE_FLOOR - 1e-9)

    def test_arms_that_never_voted_get_no_weight(self):
        weights = market_model.adaptive_weights({"a": [True, False], "silent": []})

        self.assertIn("a", weights)
        self.assertNotIn("silent", weights)

    def test_blend_probabilities_is_weighted_average(self):
        blended = market_model.blend_probabilities(
            {"a": 0.8, "b": 0.4}, {"a": 0.75, "b": 0.25}
        )
        self.assertAlmostEqual(blended, 0.8 * 0.75 + 0.4 * 0.25, places=6)


class SimulateAdaptiveEnsembleTests(unittest.TestCase):
    def test_ensemble_is_leak_free_and_covers_each_market_day_once(self):
        prices = make_prices(420)
        records = market_model.walk_forward_backtest(
            prices, "TEST", train_window=150, test_window=40, retrain_every=20
        )

        ensemble = market_model.simulate_adaptive_ensemble(records)

        market_days = {record["market_date"] for record in records}
        self.assertEqual(len(ensemble), len(market_days))
        for record in ensemble:
            self.assertEqual(record["model"], "adaptive_ensemble")
            self.assertLess(record["as_of_date"], record["market_date"])
            self.assertIn(record["predicted_dir"], {"Up", "Down"})


class TuneEnsembleTests(unittest.TestCase):
    def test_tuning_returns_valid_params_and_does_not_underperform_defaults(self):
        prices = make_prices(420)
        records = market_model.walk_forward_backtest(
            prices, "TEST", train_window=150, test_window=40, retrain_every=20
        )

        tuning = market_model.tune_ensemble_hyperparameters(
            records, betas=(0.5, 0.8, 1.0), windows=(20, 40), floors=(0.0, 0.05)
        )
        best = tuning["best"]
        self.assertIn("beta", best)
        self.assertIn("window", best)
        self.assertIn("floor", best)

        # the chosen params must be the grid's maximum accuracy
        best_acc = max(row["accuracy_pct"] for row in tuning["grid"])
        chosen = market_model.simulate_adaptive_ensemble(records, **best)
        chosen_acc = round(sum(r["correct"] for r in chosen) / len(chosen) * 100, 2)
        self.assertAlmostEqual(chosen_acc, best_acc, places=2)


class ArmWeightAndBaselineTests(unittest.TestCase):
    def _history(self):
        # price_ml is right every day, legacy_rf wrong every day → price_ml should dominate.
        history = {}
        for day in range(1, 6):
            date = f"2026-02-0{day}"
            history[date] = {
                "evaluated": True,
                "actuals": {"NVDA": {"actual_dir": "Up", "correct": True}},
                "predictions": {
                    "NVDA": {
                        "rf_pct": -0.01,        # legacy_rf says Down (wrong)
                        "arima_pct": 0.01,      # legacy_arima says Up (right)
                        "news_pct": 0.0,        # no news → abstains
                        "price_ml_pct": 0.02,   # price_ml says Up (right)
                    }
                },
            }
        return history

    def test_compute_arm_weights_favours_the_recent_winner(self):
        weights, accuracy = daily_predictor.compute_arm_weights(self._history())

        nvda = weights["NVDA"]
        self.assertGreater(nvda["price_ml"], nvda["legacy_rf"])
        self.assertNotIn("news", nvda)  # abstained every day
        self.assertEqual(accuracy["NVDA"]["price_ml"]["recent_accuracy_pct"], 100.0)
        self.assertEqual(accuracy["NVDA"]["legacy_rf"]["recent_accuracy_pct"], 0.0)

    def test_compute_baselines(self):
        baselines = daily_predictor.compute_baselines(self._history())

        self.assertEqual(baselines["always_up_pct"], 100.0)   # every actual was Up
        self.assertEqual(baselines["momentum_pct"], 100.0)    # direction never changed
        self.assertEqual(baselines["samples"], 5)


class NewFeatureTests(unittest.TestCase):
    def test_new_features_and_stacking_are_available(self):
        for column in ("return_2", "bb_position", "stoch_k_14", "day_of_week"):
            self.assertIn(column, market_model.FEATURE_COLUMNS)
        self.assertIn("stacking", market_model.MODEL_NAMES)

        signal = market_model.predict_price_signal(
            make_prices(420), model_name="stacking", train_window=200
        )
        self.assertIn(signal["predicted_dir"], {"Up", "Down"})


if __name__ == "__main__":
    unittest.main()
