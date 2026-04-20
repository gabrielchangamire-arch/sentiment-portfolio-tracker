"""
Portfolio module.

Manages a simple stock portfolio (ticker, shares, cost basis) stored in
a local JSON file, and fetches current prices via the free Yahoo Finance
endpoint (yfinance).
"""

import json
import os
import yfinance as yf

PORTFOLIO_FILE = "portfolio.json"


def _load_portfolio() -> list[dict]:
    if not os.path.exists(PORTFOLIO_FILE):
        return []
    with open(PORTFOLIO_FILE, "r") as f:
        return json.load(f)


def _save_portfolio(portfolio: list[dict]) -> None:
    with open(PORTFOLIO_FILE, "w") as f:
        json.dump(portfolio, f, indent=2)


def add_holding(ticker: str, shares: float, cost_per_share: float) -> dict:
    """Add or update a holding. If the ticker already exists, averages in."""
    portfolio = _load_portfolio()
    ticker = ticker.upper()

    for h in portfolio:
        if h["ticker"] == ticker:
            total_cost = h["shares"] * h["cost_per_share"] + shares * cost_per_share
            h["shares"] += shares
            h["cost_per_share"] = round(total_cost / h["shares"], 4)
            _save_portfolio(portfolio)
            return h

    holding = {
        "ticker": ticker,
        "shares": shares,
        "cost_per_share": round(cost_per_share, 4),
    }
    portfolio.append(holding)
    _save_portfolio(portfolio)
    return holding


def remove_holding(ticker: str) -> bool:
    """Remove a ticker entirely. Returns True if found."""
    portfolio = _load_portfolio()
    ticker = ticker.upper()
    before = len(portfolio)
    portfolio = [h for h in portfolio if h["ticker"] != ticker]
    if len(portfolio) < before:
        _save_portfolio(portfolio)
        return True
    return False


def get_portfolio() -> list[dict]:
    """Return the stored portfolio list."""
    return _load_portfolio()


def get_current_prices(tickers: list[str]) -> dict[str, float | None]:
    """
    Fetch the latest closing price for each ticker via yfinance.

    Returns {ticker: price} where price is None if lookup fails.
    """
    prices: dict[str, float | None] = {}
    for t in tickers:
        try:
            info = yf.Ticker(t)
            hist = info.history(period="1d")
            if not hist.empty:
                prices[t] = round(float(hist["Close"].iloc[-1]), 2)
            else:
                prices[t] = None
        except Exception:
            prices[t] = None
    return prices


def portfolio_summary() -> list[dict]:
    """
    Enrich each holding with current price, market value, and gain/loss.

    Returns a list of dicts, each with:
        ticker, shares, cost_per_share, current_price,
        market_value, cost_basis, gain_loss, gain_loss_pct
    """
    portfolio = _load_portfolio()
    if not portfolio:
        return []

    tickers = [h["ticker"] for h in portfolio]
    prices = get_current_prices(tickers)

    enriched = []
    for h in portfolio:
        price = prices.get(h["ticker"])
        cost_basis = round(h["shares"] * h["cost_per_share"], 2)
        market_value = round(h["shares"] * price, 2) if price else None
        gain_loss = round(market_value - cost_basis, 2) if market_value else None
        gain_loss_pct = (
            round((gain_loss / cost_basis) * 100, 2)
            if gain_loss is not None and cost_basis != 0
            else None
        )
        enriched.append({
            "ticker": h["ticker"],
            "shares": h["shares"],
            "cost_per_share": h["cost_per_share"],
            "current_price": price,
            "market_value": market_value,
            "cost_basis": cost_basis,
            "gain_loss": gain_loss,
            "gain_loss_pct": gain_loss_pct,
        })
    return enriched
