"""
Sentiment Portfolio Tracker — Streamlit web interface.

Run:  streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sentiment import analyse_sentiment
from portfolio import add_holding, remove_holding, get_portfolio, portfolio_summary
from config import DEMO_MODE

st.set_page_config(page_title="Sentiment Portfolio Tracker", layout="wide")
st.title("Sentiment Portfolio Tracker")

if DEMO_MODE:
    st.info("Running in **demo mode** with sample headlines. "
            "Set a NEWSAPI_KEY secret for live data.")

# ── Sidebar: Portfolio Management ───────────────────────────────────
st.sidebar.header("Manage Portfolio")
with st.sidebar.form("add_holding_form", clear_on_submit=True):
    ticker = st.text_input("Ticker Symbol", placeholder="e.g. AAPL").strip().upper()
    shares = st.number_input("Shares", min_value=0.01, step=0.01, format="%.2f")
    cost = st.number_input("Cost per Share ($)", min_value=0.01, step=0.01, format="%.2f")
    add_submitted = st.form_submit_button("Add to Portfolio")

    if add_submitted and ticker:
        h = add_holding(ticker, shares, cost)
        st.success(f"Added {h['ticker']} — {h['shares']} shares @ ${h['cost_per_share']:.2f}")
        st.rerun()

portfolio = get_portfolio()
if portfolio:
    with st.sidebar.expander("Remove a holding"):
        tickers_list = [h["ticker"] for h in portfolio]
        rm_ticker = st.selectbox("Select ticker to remove", tickers_list)
        if st.button("Remove"):
            if remove_holding(rm_ticker):
                st.success(f"{rm_ticker} removed.")
                st.rerun()

# ── Portfolio Table ─────────────────────────────────────────────────
if portfolio:
    st.subheader("Your Portfolio")
    with st.spinner("Fetching live prices..."):
        summary = portfolio_summary()
    df = pd.DataFrame(summary)
    df.columns = ["Ticker", "Shares", "Cost/Share", "Price", "Market Value",
                   "Cost Basis", "Gain/Loss", "G/L %"]
    st.dataframe(df, use_container_width=True, hide_index=True)
else:
    st.info("Portfolio is empty. Use the sidebar to add holdings.")

# ── Sentiment Analysis ──────────────────────────────────────────────
st.subheader("Sentiment Analysis")

col_input, col_opts = st.columns([2, 1])
with col_input:
    query_ticker = st.text_input("Enter a ticker to analyse",
                                 placeholder="e.g. AAPL, TSLA, MSFT").strip().upper()
with col_opts:
    num_articles = st.slider("Articles to fetch", 5, 50, 15)

if st.button("Run Sentiment Analysis") and query_ticker:
    with st.spinner(f"Fetching news and scoring sentiment for {query_ticker}..."):
        result = analyse_sentiment(query_ticker, page_size=num_articles)

    # Headline metric
    label = ("Positive" if result["avg_compound"] >= 0.05
             else "Negative" if result["avg_compound"] <= -0.05
             else "Neutral")
    m1, m2 = st.columns(2)
    m1.metric("Avg Sentiment", f"{result['avg_compound']:+.4f}")
    m2.metric("Articles Analysed", result["article_count"])
    st.caption(f"Overall tone: **{label}**")

    # Articles table
    if result["articles"]:
        art_data = []
        for a in result["articles"]:
            art_data.append({
                "Date": a["date"],
                "Headline": a["title"],
                "Compound": round(a["sentiment"]["compound"], 4),
                "Pos": round(a["sentiment"]["pos"], 3),
                "Neg": round(a["sentiment"]["neg"], 3),
            })
        st.dataframe(pd.DataFrame(art_data), use_container_width=True,
                     hide_index=True)

    # ── Sentiment bar chart for this ticker ──────────────────────
    compounds = [a["sentiment"]["compound"] for a in result["articles"]]
    titles = [a["title"][:50] + "..." if len(a["title"]) > 50
              else a["title"] for a in result["articles"]]
    colors = ["#2ecc71" if c >= 0 else "#e74c3c" for c in compounds]

    fig1, ax1 = plt.subplots(figsize=(8, max(3, len(titles) * 0.35)))
    ax1.barh(range(len(titles)), compounds, color=colors)
    ax1.set_yticks(range(len(titles)))
    ax1.set_yticklabels(titles, fontsize=7)
    ax1.set_xlabel("Compound Sentiment Score")
    ax1.set_title(f"Per-Article Sentiment: {query_ticker}")
    ax1.axvline(0, color="grey", linewidth=0.8)
    ax1.invert_yaxis()
    fig1.tight_layout()
    st.pyplot(fig1)
    plt.close(fig1)

    # ── Sentiment vs portfolio gain/loss (if ticker is in portfolio) ─
    if portfolio:
        port_map = {s["ticker"]: s for s in portfolio_summary()}
        if query_ticker in port_map and port_map[query_ticker]["gain_loss_pct"] is not None:
            st.subheader("Sentiment vs Portfolio Performance")

            # Gather all portfolio tickers with sentiment
            all_results = [result]
            other_tickers = [h["ticker"] for h in portfolio
                             if h["ticker"] != query_ticker]

            if other_tickers:
                with st.spinner("Fetching sentiment for remaining portfolio tickers..."):
                    for t in other_tickers:
                        all_results.append(analyse_sentiment(t, page_size=10))

            xs, ys, labels = [], [], []
            for r in all_results:
                t = r["ticker"]
                if t in port_map and port_map[t]["gain_loss_pct"] is not None:
                    xs.append(r["avg_compound"])
                    ys.append(port_map[t]["gain_loss_pct"])
                    labels.append(t)

            if len(xs) >= 2:
                corr = float(np.corrcoef(xs, ys)[0, 1])

                fig2, ax2 = plt.subplots(figsize=(8, 5))
                ax2.scatter(xs, ys, s=120, color="#3498db",
                            edgecolors="black", zorder=3)
                for i, lbl in enumerate(labels):
                    ax2.annotate(lbl, (xs[i], ys[i]),
                                 textcoords="offset points",
                                 xytext=(8, 4), fontsize=10)
                m, b = np.polyfit(xs, ys, 1)
                x_line = np.linspace(min(xs) - 0.05, max(xs) + 0.05, 50)
                ax2.plot(x_line, m * x_line + b, "--", color="grey",
                         alpha=0.7, label=f"r = {corr:.3f}")
                ax2.set_xlabel("Avg Compound Sentiment")
                ax2.set_ylabel("Gain / Loss (%)")
                ax2.set_title("Sentiment vs Portfolio Performance")
                ax2.legend()
                fig2.tight_layout()
                st.pyplot(fig2)
                plt.close(fig2)

                st.caption(f"Pearson correlation: **{corr:.4f}**")
            else:
                st.info("Need at least 2 portfolio tickers with price data "
                        "to plot correlation.")
