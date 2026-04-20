"""
Sentiment-Portfolio Tracker — CLI entry point.

Run:  python main.py
"""

from portfolio import add_holding, remove_holding, get_portfolio, portfolio_summary
from sentiment import analyse_sentiment
from visualize import generate_all_charts


def print_menu():
    print("\n===== Sentiment Portfolio Tracker =====")
    print("1. Add stock to portfolio")
    print("2. Remove stock from portfolio")
    print("3. View portfolio")
    print("4. View portfolio with live prices")
    print("5. Run sentiment analysis on portfolio")
    print("6. Generate all charts (sentiment + correlation)")
    print("7. Exit")
    print("=======================================")


def handle_add():
    ticker = input("Ticker symbol (e.g. AAPL): ").strip().upper()
    try:
        shares = float(input("Number of shares: "))
        cost = float(input("Cost per share ($): "))
    except ValueError:
        print("Invalid number.")
        return
    h = add_holding(ticker, shares, cost)
    print(f"Portfolio updated: {h['ticker']} — {h['shares']} shares @ ${h['cost_per_share']:.2f}")


def handle_remove():
    ticker = input("Ticker to remove: ").strip().upper()
    if remove_holding(ticker):
        print(f"{ticker} removed.")
    else:
        print(f"{ticker} not found in portfolio.")


def handle_view():
    portfolio = get_portfolio()
    if not portfolio:
        print("Portfolio is empty.")
        return
    print(f"\n{'Ticker':<8} {'Shares':>8} {'Cost/Share':>12}")
    print("-" * 30)
    for h in portfolio:
        print(f"{h['ticker']:<8} {h['shares']:>8.2f} ${h['cost_per_share']:>10.2f}")


def handle_summary():
    print("\nFetching current prices...")
    summary = portfolio_summary()
    if not summary:
        print("Portfolio is empty.")
        return

    print(f"\n{'Ticker':<8} {'Shares':>8} {'Price':>10} {'Value':>12} "
          f"{'Gain/Loss':>12} {'G/L %':>8}")
    print("-" * 64)
    for s in summary:
        price = f"${s['current_price']:.2f}" if s['current_price'] else "N/A"
        value = f"${s['market_value']:.2f}" if s['market_value'] else "N/A"
        gl = f"${s['gain_loss']:+.2f}" if s['gain_loss'] is not None else "N/A"
        pct = f"{s['gain_loss_pct']:+.1f}%" if s['gain_loss_pct'] is not None else "N/A"
        print(f"{s['ticker']:<8} {s['shares']:>8.2f} {price:>10} {value:>12} "
              f"{gl:>12} {pct:>8}")


def handle_sentiment():
    portfolio = get_portfolio()
    if not portfolio:
        print("Portfolio is empty — add stocks first.")
        return

    results = []
    for h in portfolio:
        ticker = h["ticker"]
        print(f"  Analysing {ticker}...")
        result = analyse_sentiment(ticker, page_size=15)
        results.append(result)
        label = "positive" if result["avg_compound"] >= 0.05 else (
            "negative" if result["avg_compound"] <= -0.05 else "neutral"
        )
        print(f"    {result['article_count']} articles — "
              f"avg sentiment: {result['avg_compound']:+.4f} ({label})")

    return results


def handle_charts():
    print("Running sentiment analysis for all holdings...")
    portfolio = get_portfolio()
    if not portfolio:
        print("Portfolio is empty — add stocks first.")
        return

    results = []
    for h in portfolio:
        ticker = h["ticker"]
        print(f"  Fetching news for {ticker}...")
        results.append(analyse_sentiment(ticker, page_size=15))

    print("\nGenerating charts...")
    generate_all_charts(results)
    print("Done. Check the PNG files in the current directory.")


def main():
    while True:
        print_menu()
        choice = input("Choose an option (1-7): ").strip()

        if choice == "1":
            handle_add()
        elif choice == "2":
            handle_remove()
        elif choice == "3":
            handle_view()
        elif choice == "4":
            handle_summary()
        elif choice == "5":
            handle_sentiment()
        elif choice == "6":
            handle_charts()
        elif choice == "7":
            print("Goodbye!")
            break
        else:
            print("Invalid choice — enter 1-7.")


if __name__ == "__main__":
    main()
