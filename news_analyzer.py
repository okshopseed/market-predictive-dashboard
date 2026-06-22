"""Public online news collection and sentiment snapshots for the shadow model."""

from datetime import datetime, timezone
from urllib.parse import quote_plus, urlparse

import feedparser
import requests as _requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


_RSS_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; MarketSentimentBot/2.0)"}

# Scores are only used to decide whether a discovered public source can affect a model.
SOURCE_CREDIBILITY = {
    "reuters.com": 100,
    "apnews.com": 100,
    "economist.com": 98,
    "bloomberg.com": 95,
    "wsj.com": 95,
    "ft.com": 95,
    "barrons.com": 94,
    "bbc.com": 94,
    "bbc.co.uk": 94,
    "cnbc.com": 92,
    "nytimes.com": 91,
    "marketwatch.com": 90,
    "theguardian.com": 90,
    "investing.com": 85,
    "yahoo.com": 85,
    "bangkokpost.com": 80,
    "nationthailand.com": 80,
}
PUBLISHER_DOMAIN = {
    "reuters": "reuters.com",
    "associated press": "apnews.com",
    "ap news": "apnews.com",
    "bloomberg": "bloomberg.com",
    "wall street journal": "wsj.com",
    "wsj": "wsj.com",
    "financial times": "ft.com",
    "barron": "barrons.com",
    "bbc": "bbc.com",
    "cnbc": "cnbc.com",
    "marketwatch": "marketwatch.com",
    "new york times": "nytimes.com",
    "guardian": "theguardian.com",
    "economist": "economist.com",
    "bangkok post": "bangkokpost.com",
    "nation thailand": "nationthailand.com",
}

MIN_CREDIBILITY = 80
SCALE_FACTOR = 0.02

# These are syndication endpoints and public search feeds. The collector never fetches
# article bodies, which keeps the stored audit trail to short public metadata only.
RSS_FEEDS = [
    ("reuters.com", "https://feeds.reuters.com/reuters/businessNews"),
    ("reuters.com", "https://feeds.reuters.com/reuters/technologyNews"),
    ("wsj.com", "https://feeds.a.dj.com/rss/RSSMarketsMain.xml"),
    ("cnbc.com", "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664"),
    ("marketwatch.com", "https://feeds.marketwatch.com/marketwatch/topstories/"),
    ("barrons.com", "https://www.barrons.com/xml/rss/3_7510.xml"),
]

SYMBOL_KEYWORDS = {
    "S&P 500": ["s&p", "s&p 500", "sp500", "stock market", "wall street", "equities", "dow jones", "market rally"],
    "Nasdaq": ["nasdaq", "tech stocks", "technology stocks", "nasdaq 100", "nasdaq composite"],
    "Gold": ["gold", "precious metal", "xau", "bullion", "safe haven", "gold price"],
    "SCB": ["scb", "siam commercial", "thai bank", "thai banking"],
    "TQM": ["tqm", "thai quality", "thai insurance"],
    "IVV": ["s&p 500", "ishares", "index fund", "ivv etf"],
    "Google": ["google", "alphabet", "googl", "youtube", "android", "waymo", "gemini ai", "google cloud"],
    "NVDA": ["nvidia", "nvda", "gpu", "ai chip", "jensen huang", "blackwell", "cuda", "nvidia earnings"],
    "AMD": ["amd", "advanced micro devices", "ryzen", "radeon", "lisa su", "mi300"],
    "TSM": ["tsmc", "taiwan semiconductor", "chip foundry", "3nm", "2nm", "taiwan chip"],
    "SMH": ["semiconductor", "chip industry", "chip stocks", "smh", "vaneck semi"],
    "MU": ["micron", "dram", "nand flash", "hbm memory", "memory chip", "micron earnings"],
    "WDC": ["western digital", "wdc", "hard drive", "flash storage"],
    "TSLA": ["tesla", "tsla", "elon musk", "electric vehicle", "gigafactory", "cybertruck"],
    "RKLB": ["rocket lab", "rklb", "small launch", "neutron rocket", "satellite launch"],
    "Bitcoin": ["bitcoin", "btc", "cryptocurrency", "crypto market", "spot bitcoin etf", "digital asset"],
}


def _domain(url):
    try:
        return urlparse(url).netloc.lower().removeprefix("www.")
    except Exception:
        return ""


def _publisher_to_domain(publisher):
    publisher = (publisher or "").lower()
    for key, domain in PUBLISHER_DOMAIN.items():
        if key in publisher:
            return domain
    return ""


def _published_at(entry):
    parsed = entry.get("published_parsed") or entry.get("updated_parsed")
    if parsed:
        return datetime(*parsed[:6], tzinfo=timezone.utc).isoformat()
    return None


def _entry_domain(entry, fallback_domain):
    source = entry.get("source", {})
    if hasattr(source, "get"):
        source_domain = _domain(source.get("href", "")) or _publisher_to_domain(source.get("title", ""))
        if source_domain:
            return source_domain
    return _domain(entry.get("link", "")) or fallback_domain


def _feed_entries(url, fallback_domain, limit=15):
    try:
        response = _requests.get(url, headers=_RSS_HEADERS, timeout=10)
        response.raise_for_status()
    except Exception as error:
        print(f"  [News] {fallback_domain}: {error}")
        return []

    entries = []
    for entry in feedparser.parse(response.content).entries[:limit]:
        title = entry.get("title", "").strip()
        if not title:
            continue
        entries.append(
            {
                "title": title,
                "url": entry.get("link", ""),
                "domain": _entry_domain(entry, fallback_domain),
                "published_at": _published_at(entry),
                "summary": entry.get("summary", "").strip(),
            }
        )
    return entries


def collect_online_articles(symbols):
    """Collect short public metadata from direct RSS feeds and public search feeds."""
    articles = []
    for domain, url in RSS_FEEDS:
        articles.extend(_feed_entries(url, domain))

    for name in symbols:
        keywords = SYMBOL_KEYWORDS.get(name, [name])[:2]
        query = " OR ".join(f'"{keyword}"' for keyword in keywords)
        url = "https://news.google.com/rss/search?q=" + quote_plus(query) + "&hl=en-US&gl=US&ceid=US:en"
        articles.extend(_feed_entries(url, "news.google.com", limit=10))

    unique = []
    seen = set()
    for article in articles:
        key = article["url"] or (article["domain"], article["title"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(article)
    return unique


def _symbol_score(name, articles):
    keywords = [keyword.lower() for keyword in SYMBOL_KEYWORDS.get(name, [name.lower()])]
    relevant = [
        article for article in articles
        if article["eligible_for_model"]
        and any(keyword in article["title"].lower() for keyword in keywords)
    ]
    if not relevant:
        return {
            "news_score": 0.0,
            "news_direction": "Neutral",
            "article_count": 0,
            "avg_credibility": None,
            "sentiment_raw": None,
        }

    total_weight = sum(article["credibility"] for article in relevant)
    weighted_sentiment = sum(article["sentiment"] * article["credibility"] for article in relevant) / total_weight
    score = weighted_sentiment * SCALE_FACTOR
    return {
        "news_score": float(score),
        "news_direction": "Up" if score > 1e-5 else ("Down" if score < -1e-5 else "Neutral"),
        "article_count": len(relevant),
        "avg_credibility": round(total_weight / len(relevant), 1),
        "sentiment_raw": round(weighted_sentiment, 4),
    }


def score_online_articles(symbols, articles, collected_at=None):
    """Create a compact, auditable snapshot without persisting article bodies."""
    analyzer = SentimentIntensityAnalyzer()
    snapshot_articles = []
    for article in articles:
        domain = (article.get("domain") or _domain(article.get("url", ""))).lower()
        credibility = SOURCE_CREDIBILITY.get(domain, 0)
        title = article.get("title", "").strip()
        summary = article.get("summary", "").strip()
        eligible = bool(title) and credibility >= MIN_CREDIBILITY
        snapshot_articles.append(
            {
                "title": title,
                "url": article.get("url", ""),
                "domain": domain,
                "published_at": article.get("published_at"),
                "credibility": credibility,
                "sentiment": analyzer.polarity_scores(f"{title} {summary}")["compound"] if title else 0.0,
                "eligible_for_model": eligible,
            }
        )

    results = {name: _symbol_score(name, snapshot_articles) for name in symbols}
    eligible = sum(article["eligible_for_model"] for article in snapshot_articles)
    return {
        "collected_at": (collected_at or datetime.now(timezone.utc)).isoformat(),
        "articles": snapshot_articles,
        "symbols": results,
        "stats": {
            "discovered": len(snapshot_articles),
            "eligible": eligible,
            "rejected": len(snapshot_articles) - eligible,
            "min_credibility": MIN_CREDIBILITY,
        },
    }


def analyze_all(symbols):
    """Collect public online news and return current per-symbol sentiment values."""
    print(f"\n--- 📰 Online News Sentiment (credibility >= {MIN_CREDIBILITY}) ---")
    snapshot = score_online_articles(symbols, collect_online_articles(symbols))
    for name, result in snapshot["symbols"].items():
        print(
            f"  [{name}] {result['news_direction']:8s} | {result['news_score'] * 100:+.3f}% "
            f"| {result['article_count']:2d} articles"
        )

    results = dict(snapshot["symbols"])
    results["_meta"] = {
        "total_fetched": snapshot["stats"]["discovered"],
        "accepted": snapshot["stats"]["eligible"],
        "rejected": snapshot["stats"]["rejected"],
        "min_credibility": MIN_CREDIBILITY,
        "headlines": [
            {
                "title": article["title"],
                "domain": article["domain"],
                "credibility": article["credibility"],
                "url": article["url"],
                "published_at": article["published_at"],
            }
            for article in snapshot["articles"]
            if article["eligible_for_model"]
        ][:25],
        "snapshot": snapshot,
    }
    return results
