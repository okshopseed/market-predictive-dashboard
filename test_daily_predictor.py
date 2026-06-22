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

    def test_stale_price_is_not_used_for_a_newer_evaluation_day(self):
        matches_expected_date = getattr(
            daily_predictor, "matches_evaluation_date", lambda actual, expected: "missing"
        )

        self.assertFalse(matches_expected_date(date(2026, 6, 19), date(2026, 6, 22)))

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


if __name__ == "__main__":
    unittest.main()
