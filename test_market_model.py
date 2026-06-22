from datetime import date
import unittest

import numpy as np
import pandas as pd

try:
    import market_model
except ModuleNotFoundError:
    market_model = None


def make_prices(rows=360):
    index = pd.bdate_range("2024-01-02", periods=rows)
    returns = np.resize(np.array([0.012, -0.007, 0.009, -0.004, 0.006]), rows)
    close = 100 * np.cumprod(1 + returns)
    return pd.DataFrame(
        {"Close": close, "Volume": np.linspace(1_000_000, 1_500_000, rows)},
        index=index,
    )


class PriceFeatureTests(unittest.TestCase):
    def test_feature_frame_keeps_latest_closed_observation_for_prediction(self):
        self.assertIsNotNone(market_model)
        prices = make_prices()

        features = market_model.build_price_feature_frame(prices)

        self.assertEqual(features.index[-1], prices.index[-1])
        self.assertTrue(pd.isna(features.iloc[-1]["target_up"]))
        self.assertTrue(features.iloc[-1][list(market_model.FEATURE_COLUMNS)].notna().all())


class WalkForwardBacktestTests(unittest.TestCase):
    def test_backtest_records_only_use_information_before_actual_market_date(self):
        self.assertIsNotNone(market_model)
        prices = make_prices(380)

        records = market_model.walk_forward_backtest(
            prices,
            symbol="TEST",
            model_names=("logistic",),
            train_window=120,
            test_window=30,
            retrain_every=10,
        )

        self.assertGreater(len(records), 0)
        for record in records:
            self.assertLess(record["as_of_date"], record["market_date"])
            self.assertNotIn("news", record)
            self.assertIn(record["predicted_dir"], {"Up", "Down"})
            self.assertIn(record["actual_dir"], {"Up", "Down"})
            self.assertIn("as_of", record)
            self.assertGreater(record["actual_close"], 0)

    def test_registry_uses_recent_accuracy_to_choose_each_symbol_champion(self):
        self.assertIsNotNone(market_model)
        records = [
            {"symbol": "TEST", "model": "logistic", "correct": False, "market_date": "2026-01-01"},
            {"symbol": "TEST", "model": "random_forest", "correct": True, "market_date": "2026-01-01"},
            {"symbol": "TEST", "model": "logistic", "correct": False, "market_date": "2026-01-02"},
            {"symbol": "TEST", "model": "random_forest", "correct": True, "market_date": "2026-01-02"},
        ]

        registry = market_model.build_model_registry(records, recent_window=2)

        self.assertEqual(registry["symbols"]["TEST"]["champion"], "random_forest")
        self.assertEqual(registry["symbols"]["TEST"]["recent_accuracy_pct"], 100.0)


if __name__ == "__main__":
    unittest.main()
