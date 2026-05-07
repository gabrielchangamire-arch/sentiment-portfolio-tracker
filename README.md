# Sentiment Portfolio Tracker — DistilBERT branch

> You are looking at the **`distilbert`** branch.
> The original VADER version lives on **`master`**.

A Python tool that tracks your stock portfolio and correlates news sentiment with portfolio performance. This branch swaps VADER for HuggingFace's `distilbert-base-uncased-finetuned-sst-2-english`, a transformer model that reads **both the headline and the article description** for richer context.

## What's different on this branch

| | `master` (VADER) | `distilbert` (this branch) |
|---|---|---|
| Engine | VADER lexicon | DistilBERT (HuggingFace Transformers) |
| Reads | Headline only | Headline **+** description |
| Model size | ~1 MB | ~250 MB (downloaded on first run) |
| Speed | Instant | ~0.1–0.3 s per article on CPU |
| Strength | Fast, no setup | Understands negation, sarcasm, and context |
| Output | `compound` ∈ [-1, +1] | Same `compound` ∈ [-1, +1] (mapped from POSITIVE/NEGATIVE probability) |

Both branches produce the same downstream API (`avg_compound`, `articles[].sentiment.compound`, etc.), so the chart and correlation code is unchanged.

## Features

- **Portfolio management** — add/remove holdings with cost-basis tracking
- **Live prices** — fetches current stock prices via Yahoo Finance
- **Sentiment analysis** — scores recent news per ticker using DistilBERT-SST2
- **Headline + description scoring** — concatenates both before tokenizing for richer context than headlines alone
- **Correlation** — scatter plot with Pearson correlation between sentiment and gain/loss %
- **Charts** — bar chart per ticker, scatter with trend line

## Setup

```bash
git clone -b distilbert https://github.com/gabrielchangamire-arch/sentiment-portfolio-tracker.git
cd sentiment-portfolio-tracker

pip install -r requirements.txt

# Optional: NewsAPI key for live data. Without it, the app runs in demo mode.
echo "NEWSAPI_KEY=your_api_key_here" > .env
```

### Getting a NewsAPI Key

1. Go to [https://newsapi.org](https://newsapi.org) and sign up for a free key
2. Copy the key into `.env`

## Run the Streamlit app

```bash
streamlit run streamlit_app.py
```

The first run downloads the DistilBERT model (~250 MB) and caches it. Subsequent runs are instant.

## Run the CLI

```bash
python main.py
```

## How the score is mapped

DistilBERT-SST2 returns `{"label": "POSITIVE", "score": p}` where `p` is the model's confidence. To stay compatible with the rest of the app (which expects a VADER-style compound), we compute:

```
compound = 2 * P(positive) - 1
```

So a confident POSITIVE → close to +1, confident NEGATIVE → close to -1, an uncertain one → close to 0. SST-2 has no neutral class; a `neu` field of 0 is kept in the output for shape compatibility.

## Project structure

```
sentiment-portfolio-tracker/
├── main.py             CLI menu and entry point
├── streamlit_app.py    Streamlit web UI (this branch wires up DistilBERT)
├── config.py           Loads NEWSAPI_KEY from .env
├── sentiment.py        NewsAPI fetching + DistilBERT scoring
├── portfolio.py        Portfolio CRUD + yfinance price lookup
├── visualize.py        Charts and correlation
├── requirements.txt    Adds transformers + torch on this branch
└── .gitignore
```

## Why a separate branch instead of replacing master

VADER is genuinely useful: it's free, lightweight, and starts in milliseconds — perfect for a "first try" portfolio piece. DistilBERT is the upgrade once you've shown the basic pipeline works. Keeping both lets you:

- Demo the original quickly (no model download)
- Show the DistilBERT version in interviews when the question is "how would you handle nuance?"
- Compare the two on the same data and talk about the tradeoff

## Deploy

Both versions can run as separate Streamlit Community Cloud apps from the same repo by selecting the branch when you create the app. See the README on `master` for the original deployment, and create a second Streamlit app pointing to the `distilbert` branch for this one.
