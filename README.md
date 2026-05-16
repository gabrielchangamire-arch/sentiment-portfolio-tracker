# Sentiment Portfolio Tracker

A Python CLI tool for tracking a stock portfolio and comparing recent news sentiment with portfolio performance. It uses NewsAPI for headlines, VADER for sentiment scoring, and yfinance for current prices. The sentiment/performance chart is meant as an exploratory signal, not a trading recommendation.

## Features

- **Portfolio management** - add/remove holdings with cost basis tracking
- **Live prices** - fetches current stock prices via Yahoo Finance
- **Sentiment analysis** - scores recent news headlines per ticker using VADER
- **Correlation** - scatter plot with Pearson correlation between sentiment and gain/loss %
- **Charts** - generates PNGs from the holdings and sentiment data:
  - Sentiment bar chart per ticker
  - Portfolio allocation pie chart
  - Sentiment vs performance scatter with trend line

## Setup

```bash
# Clone the repo
git clone https://github.com/gabrielchangamire-arch/sentiment-portfolio-tracker.git
cd sentiment-portfolio-tracker

# Install dependencies
pip install -r requirements.txt

# Add your own NewsAPI key
echo "NEWSAPI_KEY=your_api_key_here" > .env
```

### Getting a NewsAPI Key

1. Go to [https://newsapi.org](https://newsapi.org)
2. Create a free account
3. Copy your API key from the dashboard
4. Paste it into the `.env` file as shown above

## Usage

```bash
python main.py
```

The interactive menu lets you manage holdings, run sentiment analysis, and generate charts from the portfolio saved on your machine.

## Project structure

```
sentiment-portfolio-tracker/
├── main.py             # CLI menu and entry point
├── config.py           # Loads NEWSAPI_KEY from .env
├── sentiment.py        # NewsAPI fetching + VADER sentiment scoring
├── portfolio.py        # Portfolio CRUD + yfinance price lookup
├── visualize.py        # Charts and sentiment-vs-performance correlation
├── requirements.txt    # Python dependencies
└── .gitignore
```

## Data storage

- Portfolio holdings are stored locally in `portfolio.json` (auto-created, git-ignored)
- The NewsAPI key is read from a `.env` file in the project root (git-ignored)
