# ETF Analysis & Scanner

A comprehensive ETF (Exchange-Traded Fund) analysis and screening tool that downloads market data, computes financial metrics, and generates investment shortlists.

## Features

- **Data Source**: Uses ETFDB screener data and Yahoo Finance for real-time market data
- **Comprehensive Metrics**: Calculates 20+ financial metrics including Sharpe ratio, Sortino ratio, maximum drawdown, alpha, beta, and technical indicators
- **Multiple Shortlists**: Generates specialized shortlists for different investment strategies:
  - Momentum leaders
  - Pullback opportunities
  - Low volatility defensive ETFs
  - High-yield bond ETFs
- **Performance Visualization**: Creates performance charts for top-ranked ETFs
- **Correlation Analysis**: Computes pairwise correlations between ETFs

## Installation

1. Clone this repository:
```bash
git clone <your-repo-url>
cd Etfs
```

2. Create a virtual environment:
```bash
python -m venv etf_env
source etf_env/bin/activate  # On Windows: etf_env\Scripts\activate
```

3. Install dependencies:
```bash
pip install pandas numpy yfinance matplotlib requests-cache
```

## Usage

### Command Line Interface

Run the main analysis script:

```bash
python etf.py --csv etfdb_screener.csv --rf 0.045 --min-aum 100
```

**Parameters:**
- `--csv`: Path to ETF screener CSV file (default: auto-detect)
- `--rf`: Risk-free rate (default: 0.0)
- `--min-aum`: Minimum AUM in millions (default: 100)

### Jupyter Notebook

For interactive analysis, use the Jupyter notebook:

```bash
jupyter notebook etf_analysis.ipynb
```

## Output Files

The analysis generates several CSV files:

- `etf_universe_used.csv`: List of ETFs analyzed
- `etf_fundamentals.csv`: Basic ETF information and fundamentals
- `etf_monthly_metrics.csv`: Comprehensive metrics for all ETFs
- `etf_final_dataset.csv`: Complete dataset with all available data
- `etf_ranked.csv`: ETFs ranked by composite score
- `etf_pairwise_corr.csv`: Correlation matrix between ETFs
- `shortlist_momo_leaders.csv`: Momentum leaders
- `shortlist_pullbacks_above_10mo.csv`: Pullback opportunities
- `shortlist_lowvol_defensive.csv`: Low volatility defensive ETFs
- `shortlist_bonds_high_yield_low_vol.csv`: High-yield bond ETFs

## Metrics Calculated

### Performance Metrics
- **Returns**: 1-month, 3-month, 6-month, 12-month, 5-year CAGR
- **Risk-Adjusted Returns**: Sharpe ratio, Sortino ratio, Treynor ratio
- **Risk Metrics**: Annualized volatility, maximum drawdown, beta
- **Advanced Metrics**: Alpha, up/down capture ratios, information ratio

### Technical Indicators
- **RSI**: 14-period Relative Strength Index
- **ADX**: 14-period Average Directional Index
- **Moving Averages**: 10-month moving average signals

### Fundamental Data
- **Costs**: Expense ratio, bid-ask spread proxy
- **Size**: Assets under management
- **Income**: Dividend yield
- **Liquidity**: Average dollar volume

## Composite Scoring

ETFs are ranked using a weighted composite score based on:
- 12-1 month momentum (25%)
- 6-month returns (15%)
- Sharpe ratio (12%)
- Sortino ratio (10%)
- Treynor ratio (8%)
- Up capture ratio (7%)
- Maximum drawdown (8%)
- 10-month MA signal (5%)
- Skewness (5%)
- RSI deviation (5%)

## Data Requirements

The tool requires an ETF screener CSV file with the following columns:
- Symbol
- Name
- Asset Class
- Assets (AUM)
- ER (Expense Ratio)
- Dividend Yield
- Returns (1 Month, 1 Year, 5 Year)
- Standard Deviation
- Beta
- RSI
- Average Daily Volume
- Price

## Configuration

Key parameters can be modified in `etf.py`:

```python
LOOKBACK_YEARS = 5      # Years of historical data
MIN_AUM_M = 100         # Minimum AUM in millions
TOPN_LIST = 15          # Number of ETFs in shortlists
TOPN_RANKED = 150       # Number of ETFs in final ranking
```

## Performance Charts

The tool generates performance charts for the top 20 ranked ETFs, saved in the `plots/` directory.

## Logging

Analysis progress and errors are logged to `etf_scanner.log`.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is open source. Please check the license file for details.

## Disclaimer

This tool is for educational and research purposes only. It does not constitute financial advice. Always do your own research before making investment decisions.

## Requirements

- Python 3.7+
- pandas
- numpy
- yfinance
- matplotlib
- requests-cache

## Support

For issues and questions, please open an issue in the GitHub repository.
