"""
Sentiment analysis module — DistilBERT version.

Fetches recent news for a given stock ticker via NewsAPI, then scores each
article using `distilbert-base-uncased-finetuned-sst-2-english` (HuggingFace
Transformers). Unlike the VADER version on `master`, this branch:

- reads BOTH the headline and the description (concatenated for context),
- uses a transformer model fine-tuned on SST-2 for binary sentiment,
- maps the POSITIVE/NEGATIVE label + confidence into a [-1, +1] compound
  score, so downstream code (charts, correlations) keeps the same shape.

When NEWSAPI_KEY is not set, falls back to built-in sample articles so the
app can run in demo mode on Streamlit Cloud without secrets.
"""

from __future__ import annotations

import requests

from config import NEWSAPI_KEY, DEMO_MODE

NEWSAPI_URL = "https://newsapi.org/v2/everything"
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

# Sample articles used in demo mode (no API key required). These mirror the
# shape of NewsAPI results so the pipeline doesn't have to branch.
_DEMO_ARTICLES = {
    "AAPL": [
        {
            "title": "Apple Reports Record Q2 Revenue Driven by iPhone Sales",
            "description": "Apple posted its strongest second quarter on record, beating analyst estimates as iPhone demand stayed strong across major markets.",
            "publishedAt": "2026-04-18T10:00:00Z",
        },
        {
            "title": "Apple Stock Rises on Strong Services Growth",
            "description": "Services revenue grew double-digits, lifting margins and pushing the stock to a new 52-week high.",
            "publishedAt": "2026-04-17T14:30:00Z",
        },
        {
            "title": "Analysts Upgrade Apple Price Target Amid AI Optimism",
            "description": "Several Wall Street firms raised their price targets, citing the rollout of new on-device AI features.",
            "publishedAt": "2026-04-16T09:15:00Z",
        },
        {
            "title": "Apple Faces Regulatory Scrutiny in EU Over App Store Policies",
            "description": "European regulators opened a formal probe into Apple's commission structure, raising the prospect of fines.",
            "publishedAt": "2026-04-15T11:00:00Z",
        },
        {
            "title": "Apple Announces New MacBook Pro Lineup at Spring Event",
            "description": "The refreshed lineup adds the next-gen M-series chip and longer battery life, drawing positive early reviews.",
            "publishedAt": "2026-04-14T16:45:00Z",
        },
    ],
    "TSLA": [
        {
            "title": "Tesla Deliveries Beat Expectations in Q1 2026",
            "description": "Quarterly deliveries came in above consensus, easing concerns about demand softness.",
            "publishedAt": "2026-04-18T08:00:00Z",
        },
        {
            "title": "Tesla Stock Drops After CEO Comments Spark Controversy",
            "description": "Shares fell sharply after the CEO's social media posts unsettled investors.",
            "publishedAt": "2026-04-17T12:00:00Z",
        },
        {
            "title": "Tesla Expands Supercharger Network Across Europe",
            "description": "The buildout adds hundreds of new stalls and opens charging access to other manufacturers.",
            "publishedAt": "2026-04-16T10:30:00Z",
        },
        {
            "title": "Analysts Warn Tesla Margins Under Pressure From Price Cuts",
            "description": "Several analysts cut earnings estimates, citing aggressive pricing that has hurt automotive margins.",
            "publishedAt": "2026-04-15T14:00:00Z",
        },
        {
            "title": "Tesla Unveils Next-Gen Battery Technology at Investor Day",
            "description": "The new cells promise lower cost per kWh and were well-received by investors at the live demo.",
            "publishedAt": "2026-04-14T09:00:00Z",
        },
    ],
    "_DEFAULT": [
        {
            "title": "Stock Market Rallies on Strong Economic Data",
            "description": "Equities closed broadly higher after upbeat job growth and steady inflation data.",
            "publishedAt": "2026-04-18T10:00:00Z",
        },
        {
            "title": "Federal Reserve Signals Cautious Approach to Rate Changes",
            "description": "Officials struck a measured tone in recent remarks, balancing growth concerns with inflation risk.",
            "publishedAt": "2026-04-17T13:00:00Z",
        },
        {
            "title": "Tech Sector Leads Gains as Earnings Season Kicks Off",
            "description": "Several large-cap tech names reported above-consensus results, lifting the broader index.",
            "publishedAt": "2026-04-16T11:00:00Z",
        },
        {
            "title": "Investors Weigh Inflation Concerns Against Corporate Profits",
            "description": "Mixed economic data left traders cautious, even as profit growth held up across most sectors.",
            "publishedAt": "2026-04-15T09:30:00Z",
        },
        {
            "title": "Global Markets Mixed Amid Trade Policy Uncertainty",
            "description": "Markets traded sideways as investors awaited clarity on incoming tariff announcements.",
            "publishedAt": "2026-04-14T15:00:00Z",
        },
    ],
}


# ── Lazy model loader ────────────────────────────────────────────────────
# DistilBERT-SST2 is ~250MB. We load it once per process and reuse.
_pipeline = None


def _get_pipeline():
    """Lazy-load the HuggingFace pipeline. Imports are inside this function
    so the module can be imported in environments that don't have torch."""
    global _pipeline
    if _pipeline is None:
        from transformers import pipeline
        _pipeline = pipeline(
            task="sentiment-analysis",
            model=MODEL_NAME,
            tokenizer=MODEL_NAME,
            truncation=True,
            top_k=None,  # return all class scores so we can build a compound
        )
    return _pipeline


def _demo_articles(ticker: str, page_size: int) -> list[dict]:
    articles = _DEMO_ARTICLES.get(ticker.upper(), _DEMO_ARTICLES["_DEFAULT"])
    return [
        {
            "title": a["title"],
            "description": a["description"],
            "publishedAt": a["publishedAt"],
            "url": "",
        }
        for a in articles[:page_size]
    ]


def fetch_articles(query: str, page_size: int = 20) -> list[dict]:
    """Fetch recent English-language articles, falling back to demo data."""
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
            "title": art.get("title", "") or "",
            "description": art.get("description", "") or "",
            "publishedAt": art.get("publishedAt", "") or "",
            "url": art.get("url", "") or "",
        })
    return articles


def _build_input(title: str, description: str) -> str:
    """Combine title + description for richer context.

    The tokenizer truncates at 512 tokens, so we just concatenate and let
    truncation handle long article bodies.
    """
    title = (title or "").strip()
    description = (description or "").strip()
    if title and description:
        return f"{title}. {description}"
    return title or description


def _to_compound(scores: list[dict]) -> tuple[float, dict]:
    """Map DistilBERT-SST2 output into a VADER-style compound + per-class dict.

    DistilBERT returns:  [{"label": "POSITIVE", "score": 0.97},
                          {"label": "NEGATIVE", "score": 0.03}]
    We:
      - read the POSITIVE probability `p`,
      - return compound = 2*p - 1   (so it lives in [-1, +1]),
      - and expose pos/neg/neu like VADER for downstream compatibility.
    """
    by_label = {item["label"].upper(): float(item["score"]) for item in scores}
    p_pos = by_label.get("POSITIVE", 0.5)
    p_neg = by_label.get("NEGATIVE", 1.0 - p_pos)
    compound = round(2 * p_pos - 1, 4)
    return compound, {
        "compound": compound,
        "pos": round(p_pos, 4),
        "neg": round(p_neg, 4),
        # SST-2 has no "neutral" class; keep the key so the table never breaks.
        "neu": 0.0,
        "label": "POSITIVE" if p_pos >= 0.5 else "NEGATIVE",
    }


def score_text(text: str) -> dict:
    """Return DistilBERT polarity scores for a single piece of text."""
    if not text or not text.strip():
        return {"compound": 0.0, "pos": 0.5, "neg": 0.5, "neu": 0.0, "label": "NEUTRAL"}
    pipe = _get_pipeline()
    raw = pipe(text)[0]  # top_k=None returns list of all class scores per input
    _, payload = _to_compound(raw)
    return payload


def analyse_sentiment(ticker: str, company_name: str = "",
                      page_size: int = 20) -> dict:
    """End-to-end sentiment analysis for a stock ticker."""
    query = f"{ticker} {company_name}".strip()
    articles = fetch_articles(query, page_size=page_size)

    scored = []
    compound_sum = 0.0

    if articles:
        pipe = _get_pipeline()
        # Batch the inputs through the pipeline for speed.
        inputs = [_build_input(a["title"], a["description"]) for a in articles]
        outputs = pipe(inputs)  # list of [ {label, score}, {label, score} ]

        for art, out in zip(articles, outputs):
            _, payload = _to_compound(out)
            compound_sum += payload["compound"]
            scored.append({
                "title": art["title"],
                "description": art["description"],
                "date": art["publishedAt"][:10] if art["publishedAt"] else "",
                "url": art["url"],
                "sentiment": payload,
            })

    avg_compound = (compound_sum / len(scored)) if scored else 0.0

    return {
        "ticker": ticker,
        "query": query,
        "model": MODEL_NAME,
        "article_count": len(scored),
        "articles": scored,
        "avg_compound": round(avg_compound, 4),
    }
