"""Command entry point for the scheduled price-only three-year backtest."""

import yfinance as yf

import backtest_runner
from daily_predictor import SYMBOLS


def fetch_prices(symbol):
    return yf.Ticker(symbol).history(period="7y")


def main():
    artifact, registry = backtest_runner.build_price_backtest(SYMBOLS, fetch_prices)
    backtest_runner.write_backtest_artifacts(artifact, registry)
    summary = artifact["summary"]
    print(
        "Price backtest complete: "
        f"{summary['three_year']['accuracy_pct']}% over {summary['three_year']['samples']} signals; "
        f"latest 60 days {summary['recent_60']['accuracy_pct']}%"
    )


if __name__ == "__main__":
    main()
