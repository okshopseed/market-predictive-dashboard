from datetime import date, datetime
import json
from pathlib import Path
import unittest
from zoneinfo import ZoneInfo

import daily_predictor
import pandas as pd


class DailyPredictionScheduleTests(unittest.TestCase):
    def test_workflow_runs_only_on_bangkok_weekdays(self):
        workflow = Path(".github/workflows/daily_prediction.yml").read_text()

        self.assertIn('cron: "30 0 * * 1-5"', workflow)


class DailyPredictionCalendarTests(unittest.TestCase):
    def test_run_after_monday_close_evaluates_monday_and_predicts_tuesday(self):
        tuesday_morning = datetime(2026, 6, 23, 7, 30, tzinfo=ZoneInfo("Asia/Bangkok"))

        run_dates = daily_predictor.get_run_dates(tuesday_morning)

        self.assertEqual(run_dates["evaluation_date"], date(2026, 6, 22))
        self.assertEqual(run_dates["prediction_date"], date(2026, 6, 23))

    def test_weekend_run_is_skipped(self):
        sunday_in_bangkok = datetime(2026, 6, 21, 6, tzinfo=ZoneInfo("Asia/Bangkok"))
        run_dates = getattr(daily_predictor, "get_run_dates", lambda now: "missing")

        self.assertIsNone(run_dates(sunday_in_bangkok))

    def test_only_weekend_target_dates_are_removed_from_history(self):
        history = {
            "2026-06-19": {"made_on": "2026-06-18"},
            "2026-06-20": {"made_on": "2026-06-19"},
            "2026-06-21": {"made_on": "2026-06-20"},
            "2026-06-22": {"made_on": "2026-06-21"},
        }
        remove_weekends = getattr(
            daily_predictor, "remove_weekend_target_entries", lambda records: "missing"
        )

        cleaned = remove_weekends(history)

        self.assertIsInstance(cleaned, dict)
        self.assertEqual(set(cleaned), {"2026-06-19", "2026-06-22"})

    def test_evaluation_uses_the_target_market_day_when_a_newer_row_exists(self):
        prices = pd.DataFrame(
            {"Close": [100.0, 102.0, 101.0]},
            index=pd.to_datetime(["2026-06-17", "2026-06-18", "2026-06-19"]),
        )
        result_for_date = getattr(
            daily_predictor, "price_change_for_market_date", lambda frame, day: "missing"
        )

        result = result_for_date(prices, date(2026, 6, 18))

        self.assertIsInstance(result, dict)
        self.assertEqual(result["market_date"], date(2026, 6, 18))
        self.assertEqual(result["previous_market_date"], date(2026, 6, 17))
        self.assertAlmostEqual(result["actual_pct"], 0.02)

    def test_historical_evaluations_are_preserved(self):
        history = {
            "2026-06-19": {
                "evaluated": True,
                "eval_date": "2026-06-19",
                "actuals": {"S&P 500": {"correct": True}},
            },
            "2026-06-23": {
                "evaluated": True,
                "eval_date": "2026-06-24",
                "actuals": {"S&P 500": {"correct": True}},
            },
        }
        preserve = getattr(
            daily_predictor, "preserve_historical_evaluations", lambda records: "missing"
        )

        cleaned = preserve(history)

        self.assertIsInstance(cleaned, dict)
        self.assertTrue(cleaned["2026-06-19"]["evaluated"])
        self.assertEqual(cleaned["2026-06-19"]["eval_date"], "2026-06-19")
        self.assertEqual(cleaned["2026-06-19"]["actuals"], {"S&P 500": {"correct": True}})
        self.assertTrue(cleaned["2026-06-23"]["evaluated"])


class HistoricalDataIntegrityTests(unittest.TestCase):
    def test_evaluated_actuals_use_the_prediction_date(self):
        history = json.loads(Path("prediction_history.json").read_text())

        for prediction_date, entry in history.items():
            if not entry.get("evaluated"):
                continue
            for symbol, actual in entry.get("actuals", {}).items():
                with self.subTest(prediction_date=prediction_date, symbol=symbol):
                    self.assertEqual(actual.get("market_date"), prediction_date)


class CumulativeStatsTests(unittest.TestCase):
    def test_total_evaluated_counts_days_with_partial_market_closures(self):
        history = {
            "2026-06-18": {
                "evaluated": True,
                "actuals": {
                    "S&P 500": {"correct": True, "actual_dir": "Up", "actual_pct": 0.01},
                },
                "predictions": {"S&P 500": {"predicted_dir": "Up", "predicted_pct": 0.01}},
            },
            "2026-06-19": {
                "evaluated": True,
                "actuals": {
                    "SCB": {"correct": False, "actual_dir": "Down", "actual_pct": -0.01},
                },
                "predictions": {"SCB": {"predicted_dir": "Up", "predicted_pct": 0.01}},
            },
        }

        stats = daily_predictor.compute_stats(history)

        self.assertEqual(stats["total_evaluated"], 2)


if __name__ == "__main__":
    unittest.main()
