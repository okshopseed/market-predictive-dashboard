from datetime import datetime, timezone
import unittest

try:
    import news_analyzer
except ImportError:
    news_analyzer = None


class OnlineNewsSnapshotTests(unittest.TestCase):
    def test_snapshot_uses_sources_at_or_above_80_percent_only_for_model_input(self):
        self.assertIsNotNone(news_analyzer)
        articles = [
            {
                "title": "Nvidia shares rise after strong chip demand",
                "url": "https://www.cnbc.com/example-nvidia",
                "domain": "cnbc.com",
                "published_at": "2026-06-22T00:20:00+00:00",
            },
            {
                "title": "Nvidia rumor spreads online",
                "url": "https://unrated.example/nvidia",
                "domain": "unrated.example",
                "published_at": "2026-06-22T00:25:00+00:00",
            },
        ]

        snapshot = news_analyzer.score_online_articles(
            {"NVDA": "NVDA"},
            articles,
            collected_at=datetime(2026, 6, 22, 7, 30, tzinfo=timezone.utc),
        )

        self.assertEqual(news_analyzer.MIN_CREDIBILITY, 80)
        self.assertEqual(snapshot["stats"]["discovered"], 2)
        self.assertEqual(snapshot["stats"]["eligible"], 1)
        self.assertEqual(snapshot["symbols"]["NVDA"]["article_count"], 1)
        self.assertTrue(snapshot["articles"][0]["eligible_for_model"])
        self.assertFalse(snapshot["articles"][1]["eligible_for_model"])
        self.assertIn("published_at", snapshot["articles"][0])
        self.assertNotIn("text", snapshot["articles"][0])


if __name__ == "__main__":
    unittest.main()
