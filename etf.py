"""Super-intelligent ETF analysis runner."""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd

from etfengine import DEFAULT_CONFIG, ETFIntelligenceEngine
from etfengine.dashboard import build_dashboard
from etfengine.utils import ensure_directory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("etf_scanner.log", mode="a"),
    ],
)
logger = logging.getLogger("etf-runner")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the ETF super-intelligence engine")
    parser.add_argument("--csv", default="etfdb_screener.csv", help="Path to ETFDB screener CSV")
    parser.add_argument("--rf", type=float, default=DEFAULT_CONFIG.risk_free_rate, help="Risk-free rate")
    parser.add_argument("--min-aum", type=float, default=DEFAULT_CONFIG.min_aum_m, help="Minimum AUM (millions)")
    parser.add_argument("--alpha-key", help="Alpha Vantage API key", default=os.getenv("ALPHAVANTAGE_API_KEY"))
    parser.add_argument("--polygon-key", help="Polygon API key", default=os.getenv("POLYGON_API_KEY"))
    parser.add_argument("--top", type=int, default=DEFAULT_CONFIG.topn_shortlist, help="Number of buy ideas to surface")
    parser.add_argument("--output", default=DEFAULT_CONFIG.output_dir, help="Output directory")
    parser.add_argument("--train-ml", action="store_true", help="Force retrain ML model even if cached")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = DEFAULT_CONFIG
    config = config.__class__(
        **{**config.__dict__,
           "risk_free_rate": args.rf,
           "min_aum_m": args.min_aum,
           "alpha_vantage_api_key": args.alpha_key,
           "polygon_api_key": args.polygon_key,
           "topn_shortlist": args.top,
           "output_dir": args.output,
        }
    )

    ensure_directory(config.output_dir)
    logger.info("Running ETF intelligence engine with %s", config)
    engine = ETFIntelligenceEngine(config)
    results = engine.run(args.csv)

    ranked = results["ranked"]
    shortlist = results["shortlist"]
    recommendations = results["recommendations"]

    if config.enable_plotly:
        build_dashboard(ranked, recommendations, config.dashboard_output)

    summary_path = Path(config.output_dir) / "run_summary.json"
    results["summary"]["dashboard"] = config.dashboard_output
    results["summary"]["recommendation_count"] = len(shortlist)
    summary_path.write_text(pd.Series(results["summary"]).to_json(indent=2))

    logger.info("Generated %d ranked ETFs and %d buy ideas", len(ranked), len(shortlist))
    logger.info("Top ideas:\n%s", shortlist[["Symbol", "predicted_return", "hold_period_days", "risk_reward_ratio"]].head())


if __name__ == "__main__":
    main()
