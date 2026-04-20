# Sentiment Portfolio Tracker

A Python CLI tool that tracks your stock portfolio and correlates news sentiment with portfolio performance. Uses NewsAPI for headlines, VADER for sentiment scoring, and yfinance for live prices.

## Features

- **Portfolio management** — add/remove holdings with cost basis tracking
- **Live prices** — fetches current stock prices via Yahoo Finance
- **Sentiment analysis** — scores recent news headlines per ticker using VADER
- **Correlation** — scatter plot with Pearson correlation between sentiment and gain/loss %
- **Charts** — auto-generated PNGs:
  - Sentiment bar chart per ticker
  - Portfolio allocation pie chart
  - Sentiment vs performance scatter with trend line

## Setup

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/sentiment-portfolio-tracker.git
cd sentiment-portfolio-tracker

# Install dependencies
pip install -r requirements.txt

# Add your NewsAPI key
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

Follow the interactive menu to manage your portfolio, run sentiment analysis, and generate charts.

## Project Structure

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

## Data Storage

- Portfolio holdings are stored locally in `portfolio.json` (auto-created, git-ignored)
- The NewsAPI key is read from a `.env` file in the project root (git-ignored)
