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
    ensemble = summary.get("adaptive_ensemble", {})
    baselines = summary.get("baselines", {})
    print(
        "Price backtest complete: "
        f"champion {summary['three_year']['accuracy_pct']}% over {summary['three_year']['samples']} signals; "
        f"latest 60 days {summary['recent_60']['accuracy_pct']}%"
    )
    print(
        "Adaptive ensemble (auto-tuned "
        f"{ensemble.get('params')}): "
        f"3yr {ensemble.get('three_year', {}).get('accuracy_pct')}% | "
        f"60d {ensemble.get('recent_60', {}).get('accuracy_pct')}%  ||  "
        f"baselines: always-up {baselines.get('always_up_pct')}%  momentum {baselines.get('momentum_pct')}%"
    )


if __name__ == "__main__":
    main()
