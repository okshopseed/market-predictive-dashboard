from pathlib import Path
import unittest


class BacktestDashboardTests(unittest.TestCase):
    def test_dashboard_has_top_level_backtest_tab_and_renderer(self):
        html = Path("index.html").read_text()
        script = Path("script.js").read_text()

        self.assertIn('data-dashboard-tab="backtest"', html)
        self.assertIn('id="backtest-view"', html)
        self.assertIn("function renderBacktest", script)
        self.assertIn("function setDashboardView", script)
        self.assertIn("coverage_pct", script)
        self.assertIn("actual_close", script)


if __name__ == "__main__":
    unittest.main()
