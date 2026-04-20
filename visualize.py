"""
Sentiment-Portfolio Tracker — visualisation & correlation module.

Charts:
  1. Sentiment bar chart per ticker
  2. Portfolio allocation pie chart
  3. Sentiment vs gain/loss scatter (correlation)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sentiment import analyse_sentiment
from portfolio import portfolio_summary


def plot_sentiment_bars(sentiment_results: list[dict],
                        save_path: str = "sentiment_bars.png") -> str:
    """
    Horizontal bar chart of average compound sentiment per ticker.

    Parameters
    ----------
    sentiment_results : list[dict]
        Each dict is the return value of analyse_sentiment().
    """
    if not sentiment_results:
        print("No sentiment data to plot.")
        return ""

    tickers = [r["ticker"] for r in sentiment_results]
    compounds = [r["avg_compound"] for r in sentiment_results]
    colors = ["#2ecc71" if c >= 0 else "#e74c3c" for c in compounds]

    fig, ax = plt.subplots(figsize=(8, max(4, len(tickers) * 0.6)))
    ax.barh(tickers, compounds, color=colors)
    ax.set_xlabel("Avg Compound Sentiment")
    ax.set_title("News Sentiment by Ticker")
    ax.axvline(0, color="grey", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")
    return save_path


def plot_portfolio_allocation(save_path: str = "portfolio_allocation.png") -> str:
    """Pie chart of portfolio allocation by market value."""
    summary = portfolio_summary()
    valid = [s for s in summary if s["market_value"] is not None]

    if not valid:
        print("No portfolio data with current prices — nothing to plot.")
        return ""

    labels = [s["ticker"] for s in valid]
    values = [s["market_value"] for s in valid]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=140)
    ax.set_title("Portfolio Allocation (by Market Value)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")
    return save_path


def plot_sentiment_vs_gain(sentiment_results: list[dict],
                           save_path: str = "sentiment_vs_gain.png") -> str:
    """
    Scatter plot: x = avg compound sentiment, y = gain/loss %.

    Computes Pearson correlation and prints it to the console.
    """
    summary = {s["ticker"]: s for s in portfolio_summary()}
    xs, ys, labels = [], [], []

    for r in sentiment_results:
        t = r["ticker"]
        if t in summary and summary[t]["gain_loss_pct"] is not None:
            xs.append(r["avg_compound"])
            ys.append(summary[t]["gain_loss_pct"])
            labels.append(t)

    if len(xs) < 2:
        print("Need at least 2 tickers with both sentiment and price data to plot correlation.")
        return ""

    # Pearson correlation
    corr = float(np.corrcoef(xs, ys)[0, 1])
    print(f"Pearson correlation (sentiment vs gain): {corr:.4f}")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(xs, ys, s=100, color="#3498db", edgecolors="black", zorder=3)
    for i, lbl in enumerate(labels):
        ax.annotate(lbl, (xs[i], ys[i]), textcoords="offset points",
                    xytext=(8, 4), fontsize=9)

    # Trend line
    m, b = np.polyfit(xs, ys, 1)
    x_line = np.linspace(min(xs) - 0.05, max(xs) + 0.05, 50)
    ax.plot(x_line, m * x_line + b, "--", color="grey", alpha=0.7,
            label=f"r = {corr:.3f}")

    ax.set_xlabel("Avg Compound Sentiment")
    ax.set_ylabel("Gain / Loss (%)")
    ax.set_title("Sentiment vs Portfolio Performance")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")
    return save_path


def generate_all_charts(sentiment_results: list[dict]) -> None:
    """Convenience: generate every chart in one call."""
    plot_sentiment_bars(sentiment_results)
    plot_portfolio_allocation()
    plot_sentiment_vs_gain(sentiment_results)
