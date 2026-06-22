from datetime import date, datetime
from pathlib import Path
import unittest
from zoneinfo import ZoneInfo

import daily_predictor


class DailyPredictionScheduleTests(unittest.TestCase):
    def test_workflow_runs_only_on_bangkok_weekdays(self):
        workflow = Path(".github/workflows/daily_prediction.yml").read_text()

        self.assertIn('cron: "0 23 * * 0-4"', workflow)


class DailyPredictionCalendarTests(unittest.TestCase):
    def test_weekend_run_is_skipped(self):
        sunday_in_bangkok = datetime(2026, 6, 21, 6, tzinfo=ZoneInfo("Asia/Bangkok"))
        run_dates = getattr(daily_predictor, "get_run_dates", lambda now: "missing")

        self.assertIsNone(run_dates(sunday_in_bangkok))

    def test_weekend_history_entries_are_removed(self):
        history = {
            "2026-06-19": {"made_on": "2026-06-18"},
            "2026-06-20": {"made_on": "2026-06-19"},
            "2026-06-22": {"made_on": "2026-06-21"},
            "2026-06-23": {"made_on": "2026-06-22"},
        }
        prune = getattr(daily_predictor, "prune_non_trading_history", lambda records: "missing")

        cleaned = prune(history)

        self.assertIsInstance(cleaned, dict)
        self.assertEqual(set(cleaned), {"2026-06-19", "2026-06-23"})

    def test_stale_price_is_not_used_for_a_newer_evaluation_day(self):
        matches_expected_date = getattr(
            daily_predictor, "matches_evaluation_date", lambda actual, expected: "missing"
        )

        self.assertFalse(matches_expected_date(date(2026, 6, 19), date(2026, 6, 22)))

    def test_same_day_evaluations_are_reset(self):
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
        reset = getattr(daily_predictor, "reset_same_day_evaluations", lambda records: "missing")

        cleaned = reset(history)

        self.assertIsInstance(cleaned, dict)
        self.assertFalse(cleaned["2026-06-19"]["evaluated"])
        self.assertIsNone(cleaned["2026-06-19"]["eval_date"])
        self.assertEqual(cleaned["2026-06-19"]["actuals"], {})
        self.assertTrue(cleaned["2026-06-23"]["evaluated"])


if __name__ == "__main__":
    unittest.main()
