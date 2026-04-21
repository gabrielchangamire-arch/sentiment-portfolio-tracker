"""
Sentiment analysis module.

Fetches recent news articles for a given stock ticker via NewsAPI,
then scores each headline using VADER (no API key needed for VADER).
Returns per-article scores and an overall average sentiment.

When NEWSAPI_KEY is not set, falls back to built-in sample headlines
so the app can run in demo mode on Streamlit Cloud without secrets.
"""

import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from config import NEWSAPI_KEY, DEMO_MODE

NEWSAPI_URL = "https://newsapi.org/v2/everything"

analyzer = SentimentIntensityAnalyzer()

# Sample headlines used in demo mode (no API key required)
_DEMO_ARTICLES = {
    "AAPL": [
        {"title": "Apple Reports Record Q2 Revenue Driven by iPhone Sales", "publishedAt": "2026-04-18T10:00:00Z"},
        {"title": "Apple Stock Rises on Strong Services Growth", "publishedAt": "2026-04-17T14:30:00Z"},
        {"title": "Analysts Upgrade Apple Price Target Amid AI Optimism", "publishedAt": "2026-04-16T09:15:00Z"},
        {"title": "Apple Faces Regulatory Scrutiny in EU Over App Store Policies", "publishedAt": "2026-04-15T11:00:00Z"},
        {"title": "Apple Announces New MacBook Pro Lineup at Spring Event", "publishedAt": "2026-04-14T16:45:00Z"},
    ],
    "TSLA": [
        {"title": "Tesla Deliveries Beat Expectations in Q1 2026", "publishedAt": "2026-04-18T08:00:00Z"},
        {"title": "Tesla Stock Drops After CEO Comments Spark Controversy", "publishedAt": "2026-04-17T12:00:00Z"},
        {"title": "Tesla Expands Supercharger Network Across Europe", "publishedAt": "2026-04-16T10:30:00Z"},
        {"title": "Analysts Warn Tesla Margins Under Pressure From Price Cuts", "publishedAt": "2026-04-15T14:00:00Z"},
        {"title": "Tesla Unveils Next-Gen Battery Technology at Investor Day", "publishedAt": "2026-04-14T09:00:00Z"},
    ],
    "_DEFAULT": [
        {"title": "Stock Market Rallies on Strong Economic Data", "publishedAt": "2026-04-18T10:00:00Z"},
        {"title": "Federal Reserve Signals Cautious Approach to Rate Changes", "publishedAt": "2026-04-17T13:00:00Z"},
        {"title": "Tech Sector Leads Gains as Earnings Season Kicks Off", "publishedAt": "2026-04-16T11:00:00Z"},
        {"title": "Investors Weigh Inflation Concerns Against Corporate Profits", "publishedAt": "2026-04-15T09:30:00Z"},
        {"title": "Global Markets Mixed Amid Trade Policy Uncertainty", "publishedAt": "2026-04-14T15:00:00Z"},
    ],
}


def _demo_articles(ticker: str, page_size: int) -> list[dict]:
    """Return sample articles for demo mode."""
    articles = _DEMO_ARTICLES.get(ticker.upper(), _DEMO_ARTICLES["_DEFAULT"])
    return [
        {
            "title": a["title"],
            "description": "",
            "publishedAt": a["publishedAt"],
            "url": "",
        }
        for a in articles[:page_size]
    ]


def fetch_articles(query: str, page_size: int = 20) -> list[dict]:
    """
    Fetch recent English-language articles from NewsAPI.
    Falls back to sample data in demo mode.
    """
    if DEMO_MODE:
        ticker = query.split()[0] if query else "DEMO"
        return _demo_articles(ticker, page_size)

    params = {
        "q": query,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": page_size,
        "apiKey": NEWSAPI_KEY,
    }
    resp = requests.get(NEWSAPI_URL, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    articles = []
    for art in data.get("articles", []):
        articles.append({
            "title": art.get("title", ""),
            "description": art.get("description", ""),
            "publishedAt": art.get("publishedAt", ""),
            "url": art.get("url", ""),
        })
    return articles


def score_text(text: str) -> dict:
    """Return VADER polarity scores for a single piece of text."""
    return analyzer.polarity_scores(text)


def analyse_sentiment(ticker: str, company_name: str = "",
                      page_size: int = 20) -> dict:
    """
    End-to-end sentiment analysis for a stock.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol, e.g. "AAPL".
    company_name : str
        Optional company name to improve search relevance.
    page_size : int
        Number of articles to fetch.

    Returns
    -------
    dict with keys:
        ticker, query, article_count,
        articles  — list of {title, date, url, sentiment{neg, neu, pos, compound}},
        avg_compound — float mean compound score across all headlines.
    """
    query = f"{ticker} {company_name}".strip()
    articles = fetch_articles(query, page_size=page_size)

    scored = []
    compound_sum = 0.0

    for art in articles:
        text = art["title"] or art["description"] or ""
        scores = score_text(text)
        compound_sum += scores["compound"]
        scored.append({
            "title": art["title"],
            "date": art["publishedAt"][:10] if art["publishedAt"] else "",
            "url": art["url"],
            "sentiment": scores,
        })

    avg_compound = (compound_sum / len(scored)) if scored else 0.0

    return {
        "ticker": ticker,
        "query": query,
        "article_count": len(scored),
        "articles": scored,
        "avg_compound": round(avg_compound, 4),
    }
