import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from test_market_model import make_prices

try:
    import backtest_runner
except ModuleNotFoundError:
    backtest_runner = None


class BacktestArtifactTests(unittest.TestCase):
    def test_price_backtest_artifact_keeps_only_champion_records_without_news(self):
        self.assertIsNotNone(backtest_runner)

        artifact, registry = backtest_runner.build_price_backtest(
            {"TEST": "TEST"},
            fetch_prices=lambda symbol: make_prices(380),
            model_names=("logistic", "random_forest"),
            train_window=120,
            test_window=30,
            retrain_every=10,
        )

        self.assertEqual(artifact["mode"], "price_only")
        self.assertEqual(registry["symbols"]["TEST"]["champion"], artifact["records"][0]["model"])
        self.assertEqual(artifact["models"]["TEST"]["champion"], registry["symbols"]["TEST"]["champion"])
        self.assertGreater(artifact["summary"]["three_year"]["samples"], 0)
        self.assertTrue(all("news" not in record for record in artifact["records"]))
        self.assertTrue(all(record["as_of_date"] < record["market_date"] for record in artifact["records"]))

    def test_summary_counts_recent_market_days_not_model_rows(self):
        self.assertIsNotNone(backtest_runner)
        records = [
            {"market_date": "2026-01-01", "correct": True, "symbol": "A"},
            {"market_date": "2026-01-01", "correct": False, "symbol": "B"},
            {"market_date": "2026-01-02", "correct": True, "symbol": "A"},
        ]

        summary = backtest_runner.summarize_backtest(records, recent_days=2)

        self.assertEqual(summary["recent_60"]["market_days"], 2)
        self.assertEqual(summary["recent_60"]["samples"], 3)
        self.assertAlmostEqual(summary["recent_60"]["accuracy_pct"], 66.7)

    def test_refresh_keeps_active_price_or_news_mode(self):
        self.assertIsNotNone(backtest_runner)
        with tempfile.TemporaryDirectory() as directory:
            data_file = Path(directory) / "backtest_data.json"
            registry_file = Path(directory) / "model_registry.json"
            registry_file.write_text(json.dumps({"active_model": "price_news"}))

            backtest_runner.write_backtest_artifacts(
                {"summary": {}},
                {"symbols": {}},
                data_file=data_file,
                registry_file=registry_file,
            )

            refreshed = json.loads(registry_file.read_text())

        self.assertEqual(refreshed["active_model"], "price_news")


if __name__ == "__main__":
    unittest.main()
