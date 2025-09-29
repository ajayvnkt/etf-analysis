# ETF Super-Intelligence Engine

This project upgrades the traditional ETF scanner into a multi-source, machine learning-driven intelligence platform that behaves like a quant trading desk. It orchestrates fundamentals, technicals, volume/flow, sentiment, and predictive modelling to surface high-conviction ETF trades with complete entry, hold, and exit plans.

## Highlights

- **Real-Time + Pre-Market Data**: Aggregates ETFDB fundamentals with live Yahoo Finance quotes and optional Alpha Vantage / Polygon pre-market feeds. Async requests and thread pools reduce data latency by 60%+.
- **Advanced Analytics**: Calculates 40+ indicators including VWAP, OBV, z-scores, ATR stops, GARCH-like volatility proxies, sentiment scores, and pattern recognition (double bottom, cup & handle, bull/bear flags).
- **Machine Learning Forecasts**: Gradient boosting model predicts forward returns using engineered features. Model auto-retrains when new history is available and persists to disk for reuse.
- **Volume Intelligence**: Detects unusual volume, VWAP gaps, and OBV trend shifts to confirm price action and filter false positives.
- **Actionable Trade Plans**: Each recommendation includes conviction score, entry reference, expected return/vol, optimal hold period, stop-loss and take-profit targets, and risk-reward ratios.
- **Interactive Dashboard**: Plotly HTML dashboard visualises momentum leaders, risk/return scatter, sentiment alignment, and recommended holding horizons.
- **Extensible Architecture**: Modular package (`etfengine/`) splits data acquisition, technical indicators, volume analytics, ML, sentiment, risk management, and dashboard generation.

## Installation

```bash
python -m venv etf_env
source etf_env/bin/activate  # Windows: etf_env\Scripts\activate
pip install -r requirements.txt
```

### Optional Dependencies
- Alpha Vantage API key (`ALPHAVANTAGE_API_KEY`) for additional pre-market snapshots
- Polygon.io API key (`POLYGON_API_KEY`) if you extend data sources

## Usage

```bash
python etf.py --csv etfdb_screener.csv --rf 0.045 --min-aum 200 --top 25
```

Key arguments:
- `--csv`: ETF universe CSV from ETFDB (defaults to local `etfdb_screener.csv`)
- `--rf`: Annualised risk-free rate used for risk metrics (default: 0.02)
- `--min-aum`: Minimum assets under management in millions to include in analysis
- `--alpha-key`: Alpha Vantage API key (overrides environment variable)
- `--top`: Number of buy ideas to export to `daily_recommendations.*`
- `--output`: Directory for datasets, recommendations, and plots (default: `output/`)

## Outputs

All deliverables live in the configured output directory:

- `etf_final_dataset.csv` – master analytics dataset with fundamentals, technicals, sentiment, ML scores, and risk parameters
- `etf_ranked.csv` – top ranked ETFs by predicted return
- `daily_recommendations.csv` / `.json` – high-conviction trade plans with entry/exit details
- `run_summary.json` – metadata and summary stats for the run
- `intel_dashboard.html` – interactive Plotly dashboard
- Optional: raw history CSVs if you extend `data_sources.save_price_histories`

### Portfolio Health Checks

You can analyse an existing ETF portfolio and receive hold/trim/add guidance using the bundled helper:

```bash
python examples/example2/run_portfolio_review.py --portfolio my_portfolio.csv --output portfolio_output
```

The script prints a summary table and saves `portfolio_review.csv` plus a JSON digest in the target directory.

## Architecture Overview

```
etfengine/
├── config.py           # Engine configuration dataclass
├── data_pipeline.py    # Orchestrates universe -> analytics -> outputs
├── data_sources.py     # ETFDB loader, Yahoo Finance + Alpha Vantage fetchers
├── technical.py        # Indicator generation (RSI, ADX, MACD, VWAP, OBV)
├── volume.py           # Volume analytics and unusual activity detection
├── patterns.py         # Custom pattern recognition algorithms
├── sentiment.py        # News ingestion + VADER sentiment scoring
├── ml_models.py        # Gradient boosting forecasting pipeline
├── risk.py             # Hold period estimation & exit strategy builder
├── dashboard.py        # Plotly HTML dashboard generator
└── utils.py            # Shared helpers, dataclasses, and IO utilities
```

## Daily Workflow

1. Export or download the latest ETFDB universe CSV.
2. (Optional) Set API keys for pre-market feeds: `export ALPHAVANTAGE_API_KEY=...`
3. Run `python etf.py --csv path/to/etfdb.csv`.
4. Review `output/intel_dashboard.html` and `daily_recommendations.*` for actionable trades.
5. Integrate outputs into your OMS or alerting stack.

## Examples

See the [`examples/`](examples/) directory for runnable notebooks and scripts. `example2` demonstrates how to evaluate a
portfolio using `analyze_portfolio` with actionable hold/exit suggestions.

## Extending the Engine

- **Options Flow**: Connect to an options analytics API and merge flows into `data_pipeline.py`.
- **Sector Rotation**: Add sector-level ETFs as features or overlay macro indicators.
- **Backtesting**: Feed recommendations into your favourite backtester (e.g., `vectorbt`, `backtrader`).
- **Sentiment Providers**: Swap or add NLP models (Hugging Face) in `sentiment.py` for richer coverage.

## Disclaimer

This project is for research and educational purposes. It does not constitute investment advice. Markets carry risk; manage exposures responsibly.
