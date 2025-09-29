# Example 2 – ETF Portfolio Health Check

This example demonstrates how to review an existing ETF portfolio and determine whether positions should be held, trimmed, or
added to.

## Files

- `sample_portfolio.csv` – Example holdings with share counts and cost basis
- `run_portfolio_review.py` – Script that runs the portfolio analysis helper and stores the output table in `output/`

## Usage

```bash
python examples/example2/run_portfolio_review.py --portfolio examples/example2/sample_portfolio.csv --output examples/example2/output
```

The script prints a table with actionable guidance and saves `portfolio_review.csv` plus a JSON summary in the specified
`--output` directory.
