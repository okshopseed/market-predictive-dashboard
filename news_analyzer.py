"""
news_analyzer.py — สูตรที่ 4: News Sentiment Prediction
เก็บข่าวเฉพาะจากแหล่งน่าเชื่อถือ ≥ 90% (NewsGuard / Media Bias-Fact Check)
แล้ววิเคราะห์ sentiment ด้วย VADER → คืนค่า news_score (%) ต่อ symbol
"""

from urllib.parse import urlparse
import feedparser
import requests as _requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import yfinance as yf

_RSS_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; MarketSentimentBot/1.0)"}

# ─── คะแนนความน่าเชื่อถือของแหล่งข่าว (0–100) ────────────────────────────────
# อ้างอิง: NewsGuard (newsguardtech.com) และ Media Bias/Fact Check (mediabiasfactcheck.com)
SOURCE_CREDIBILITY = {
    "reuters.com":       100,
    "apnews.com":        100,
    "economist.com":      98,
    "bloomberg.com":      95,
    "wsj.com":            95,
    "ft.com":             95,
    "barrons.com":        94,
    "bbc.com":            94,
    "bbc.co.uk":          94,
    "cnbc.com":           92,
    "nytimes.com":        91,
    "marketwatch.com":    90,
    "theguardian.com":    90,
}

# ชื่อสำนักพิมพ์ (จาก yfinance.news) → domain สำหรับเช็คคะแนน
PUBLISHER_DOMAIN = {
    "reuters":              "reuters.com",
    "associated press":     "apnews.com",
    "ap news":              "apnews.com",
    "bloomberg":            "bloomberg.com",
    "wall street journal":  "wsj.com",
    "wsj":                  "wsj.com",
    "financial times":      "ft.com",
    "barron":               "barrons.com",
    "bbc":                  "bbc.com",
    "cnbc":                 "cnbc.com",
    "marketwatch":          "marketwatch.com",
    "new york times":       "nytimes.com",
    "guardian":             "theguardian.com",
    "economist":            "economist.com",
}

MIN_CREDIBILITY = 90    # เกณฑ์บังคับ: ข่าวจากแหล่งต่ำกว่านี้ถูกทิ้งทั้งหมด
SCALE_FACTOR    = 0.02  # sentiment ±1.0 → ±2% price estimate (calibration)

# RSS feeds พร้อม domain ที่รู้ล่วงหน้า
RSS_FEEDS = [
    ("reuters.com",     "https://feeds.reuters.com/reuters/businessNews"),
    ("reuters.com",     "https://feeds.reuters.com/reuters/technologyNews"),
    ("wsj.com",         "https://feeds.a.dj.com/rss/RSSMarketsMain.xml"),
    ("cnbc.com",        "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664"),
    ("marketwatch.com", "https://feeds.marketwatch.com/marketwatch/topstories/"),
    ("barrons.com",     "https://www.barrons.com/xml/rss/3_7510.xml"),
]

# คำสำคัญที่ใช้จับความเกี่ยวข้องของข่าวกับแต่ละ symbol
SYMBOL_KEYWORDS = {
    "S&P 500": ["s&p", "s&p 500", "sp500", "stock market", "wall street", "equities", "dow jones", "market rally"],
    "Nasdaq":  ["nasdaq", "tech stocks", "technology stocks", "nasdaq 100", "nasdaq composite"],
    "Gold":    ["gold", "precious metal", "xau", "bullion", "safe haven", "gold price"],
    "SCB":     ["scb", "siam commercial", "thai bank", "thai banking"],
    "TQM":     ["tqm", "thai quality", "thai insurance"],
    "IVV":     ["s&p 500", "ishares", "index fund", "ivv etf"],
    "Google":  ["google", "alphabet", "googl", "youtube", "android", "waymo", "gemini ai", "google cloud"],
    "NVDA":    ["nvidia", "nvda", "gpu", "ai chip", "jensen huang", "blackwell", "cuda", "nvidia earnings"],
    "AMD":     ["amd", "advanced micro devices", "ryzen", "radeon", "lisa su", "mi300"],
    "TSM":     ["tsmc", "taiwan semiconductor", "chip foundry", "3nm", "2nm", "taiwan chip"],
    "SMH":     ["semiconductor", "chip industry", "chip stocks", "smh", "vaneck semi"],
    "MU":      ["micron", " dram", "nand flash", "hbm memory", "memory chip", "micron earnings"],
    "WDC":     ["western digital", "wdc", "hard drive", "flash storage", "wd "],
    "TSLA":    ["tesla", "tsla", "elon musk", "electric vehicle", " ev ", "gigafactory", "cybertruck"],
    "RKLB":    ["rocket lab", "rklb", "small launch", "neutron rocket", "satellite launch"],
    "Bitcoin": ["bitcoin", " btc ", "cryptocurrency", "crypto market", "spot bitcoin etf", "digital asset"],
}


# ─── Helper functions ─────────────────────────────────────────────────────────

def _domain(url: str) -> str:
    try:
        host = urlparse(url).netloc.lower()
        return host.removeprefix("www.")
    except Exception:
        return ""


def _publisher_to_domain(pub: str) -> str:
    pub_l = pub.lower()
    for key, domain in PUBLISHER_DOMAIN.items():
        if key in pub_l:
            return domain
    return ""


# ─── Fetch functions ──────────────────────────────────────────────────────────

def _fetch_rss(analyzer: SentimentIntensityAnalyzer) -> tuple:
    """ดึงข่าว RSS → กรองเฉพาะแหล่งที่ credibility ≥ MIN_CREDIBILITY"""
    accepted, total_raw, rejected = [], 0, 0
    for src_domain, url in RSS_FEEDS:
        if SOURCE_CREDIBILITY.get(src_domain, 0) < MIN_CREDIBILITY:
            continue
        try:
            raw  = _requests.get(url, headers=_RSS_HEADERS, timeout=10)
            feed = feedparser.parse(raw.content)
            for entry in feed.entries[:15]:
                total_raw += 1
                link         = entry.get("link", "")
                entry_domain = _domain(link) or src_domain
                cred         = SOURCE_CREDIBILITY.get(entry_domain,
                               SOURCE_CREDIBILITY.get(src_domain, 0))
                if cred < MIN_CREDIBILITY:
                    rejected += 1
                    continue
                title   = entry.get("title", "").strip()
                if not title:
                    rejected += 1
                    continue
                summary = entry.get("summary", "").strip()
                text    = f"{title} {summary}"
                accepted.append({
                    "title":       title,
                    "domain":      entry_domain,
                    "credibility": cred,
                    "sentiment":   analyzer.polarity_scores(text)["compound"],
                    "text":        text.lower(),
                })
        except Exception as e:
            print(f"  [RSS] {src_domain}: {e}")
    return accepted, total_raw, rejected


def _fetch_ticker_news(ticker_sym: str, analyzer: SentimentIntensityAnalyzer) -> list:
    """ข่าวรายหุ้นจาก yfinance — รองรับทั้ง schema เก่า (flat) และใหม่ (nested content)"""
    try:
        news_list = yf.Ticker(ticker_sym).news or []
    except Exception:
        return []
    result = []
    for item in news_list[:15]:
        if not isinstance(item, dict):
            continue

        # Schema ใหม่ (yfinance ≥ 0.2.50): ข้อมูลอยู่ใน item["content"]
        content = item.get("content", {})
        if content:
            title      = content.get("title", "").strip()
            source_id  = content.get("provider", {}).get("sourceId", "")
            # sourceId มักเป็น "reuters.com" หรือ "yahoofinance.com" อยู่แล้ว
            domain     = _domain(f"https://{source_id}") if source_id else ""
            if not domain:
                pub_name = content.get("provider", {}).get("displayName", "")
                domain   = _publisher_to_domain(pub_name)
        else:
            # Schema เก่า (flat)
            title     = item.get("title", "").strip()
            publisher = item.get("publisher", "")
            domain    = _publisher_to_domain(publisher)

        if not title or not domain:
            continue
        cred = SOURCE_CREDIBILITY.get(domain, 0)
        if cred < MIN_CREDIBILITY:
            continue
        result.append({
            "title":       title,
            "domain":      domain,
            "credibility": cred,
            "sentiment":   analyzer.polarity_scores(title)["compound"],
            "text":        title.lower(),
        })
    return result


# ─── Scoring ──────────────────────────────────────────────────────────────────

def _score_symbol(name: str, ticker: str, global_news: list,
                  analyzer: SentimentIntensityAnalyzer) -> dict:
    """คำนวณ news_score ของ 1 symbol จากข่าวรวม + ข่าวรายหุ้น"""
    keywords = [k.lower() for k in SYMBOL_KEYWORDS.get(name, [name.lower()])]
    relevant = [n for n in global_news if any(kw in n["text"] for kw in keywords)]
    relevant += _fetch_ticker_news(ticker, analyzer)

    if not relevant:
        return {
            "news_score":      0.0,
            "news_direction":  "Neutral",
            "article_count":   0,
            "avg_credibility": None,
            "sentiment_raw":   None,
        }

    # ถ่วงน้ำหนักตามคะแนนความน่าเชื่อถือ: แหล่ง 100 มีน้ำหนักมากกว่าแหล่ง 90
    total_w            = sum(n["credibility"] for n in relevant)
    weighted_sentiment = sum(n["sentiment"] * n["credibility"] for n in relevant) / total_w
    news_pct           = weighted_sentiment * SCALE_FACTOR

    return {
        "news_score":      float(news_pct),
        "news_direction":  "Up" if news_pct > 1e-5 else ("Down" if news_pct < -1e-5 else "Neutral"),
        "article_count":   len(relevant),
        "avg_credibility": round(total_w / len(relevant), 1),
        "sentiment_raw":   round(weighted_sentiment, 4),
    }


# ─── Main entry point ─────────────────────────────────────────────────────────

def analyze_all(symbols: dict) -> dict:
    """
    วิเคราะห์ข่าวทุก symbol
    คืนค่า dict{name: scores} + "_meta" key สำหรับ dashboard stats/headlines
    """
    print("\n--- 📰 สูตรที่ 4: News Sentiment (แหล่งน่าเชื่อถือ ≥ 90%) ---")
    analyzer                         = SentimentIntensityAnalyzer()
    global_news, total_raw, rejected = _fetch_rss(analyzer)
    print(f"  RSS ดึงมา {total_raw} | ผ่านกรอง {len(global_news)} | ทิ้ง {rejected}")

    results = {}
    for name, ticker in symbols.items():
        r    = _score_symbol(name, ticker, global_news, analyzer)
        results[name] = r
        icon = "📈" if r["news_direction"] == "Up" else ("📉" if r["news_direction"] == "Down" else "➖")
        cred_str = f"avg_cred={r['avg_credibility']}" if r["avg_credibility"] else "ไม่มีข่าวที่เกี่ยวข้อง"
        print(f"  [{name}] {icon} {r['news_direction']:8s} | {r['news_score']*100:+.3f}% | "
              f"{r['article_count']:2d} ข่าว | {cred_str}")

    # _meta: ข้อมูลสรุป + พาดหัวสำหรับแสดงบน dashboard
    results["_meta"] = {
        "total_fetched":   total_raw,
        "accepted":        len(global_news),
        "rejected":        rejected,
        "min_credibility": MIN_CREDIBILITY,
        "headlines": [
            {"title": n["title"], "domain": n["domain"], "credibility": n["credibility"]}
            for n in global_news[:25]
        ],
    }
    return results
