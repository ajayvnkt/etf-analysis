"""Run a sample ETF portfolio health check."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from etfengine import DEFAULT_CONFIG, analyze_portfolio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review an ETF portfolio using the intelligence engine signals")
    parser.add_argument("--portfolio", required=True, help="CSV file with columns Symbol,Shares,CostBasis")
    parser.add_argument("--output", default="examples/example2/output", help="Directory to save analysis artifacts")
    parser.add_argument("--lookback", type=int, default=None, help="Override lookback years for analytics")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.portfolio)
    config = DEFAULT_CONFIG
    if args.lookback:
        config = config.__class__(**{**config.__dict__, "lookback_years": args.lookback})
    results, summary = analyze_portfolio(df, config, output_dir=args.output)

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    summary_path = output_path / "portfolio_summary.json"
    summary_path.write_text(json.dumps(summary.to_dict(), indent=2))

    print("=== Portfolio Review ===")
    display_cols = [
        "Symbol",
        "Shares",
        "CostBasis",
        "LastPrice",
        "ReturnPct",
        "RiskReward",
        "HoldPeriodDays",
        "Action",
    ]
    print(results[display_cols].to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
    print("\nSummary:")
    for key, value in summary.to_dict().items():
        print(f"  {key}: {value:,.2f}")
    print(f"\nDetailed CSV saved to: {output_path / 'portfolio_review.csv'}")
    print(f"Summary JSON saved to: {summary_path}")


if __name__ == "__main__":
    main()
