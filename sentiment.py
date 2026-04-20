"""
Sentiment analysis module.

Fetches recent news articles for a given stock ticker via NewsAPI,
then scores each headline using VADER (no API key needed for VADER).
Returns per-article scores and an overall average sentiment.
"""

import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from config import NEWSAPI_KEY

NEWSAPI_URL = "https://newsapi.org/v2/everything"

analyzer = SentimentIntensityAnalyzer()


def fetch_articles(query: str, page_size: int = 20) -> list[dict]:
    """
    Fetch recent English-language articles from NewsAPI.

    Parameters
    ----------
    query : str
        Search term — typically a company name or ticker (e.g. "AAPL Apple").
    page_size : int
        Max articles to return (free tier caps at 100).

    Returns
    -------
    list[dict]  Each dict has 'title', 'description', 'publishedAt', 'url'.
    """
    if not NEWSAPI_KEY:
        raise RuntimeError("NEWSAPI_KEY is not configured. See README for setup.")

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
