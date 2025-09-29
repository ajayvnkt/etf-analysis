"""Portfolio analysis helpers for ETF holdings."""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .config import EngineConfig
from .data_sources import fetch_latest_quote, fetch_price_history
from .risk import build_exit_plan
from .sentiment import sentiment_for_symbol
from .technical import compute_indicators, technical_snapshot
from .utils import ensure_directory
from .volume import compute_volume_signals


@dataclass
class PortfolioSummary:
    total_cost: float
    total_value: float
    total_pnl: float
    total_return_pct: float
    risk_weighted_return: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "total_cost": self.total_cost,
            "total_value": self.total_value,
            "total_pnl": self.total_pnl,
            "total_return_pct": self.total_return_pct,
            "risk_weighted_return": self.risk_weighted_return,
        }


def _action_from_signals(return_pct: float, momentum: float, risk_reward: float, trend: float, sentiment: float) -> Tuple[str, str]:
    """Translate quantitative signals into a suggested action."""
    if risk_reward >= 1.5 and momentum > 0 and trend > 0:
        return "Hold / Consider adding", "Trend and momentum positive with favourable risk-reward."
    if return_pct < -0.07 and momentum < 0:
        return "Reduce / Exit", "Drawdown beyond comfort with negative momentum."
    if risk_reward < 1.0 and trend < 0:
        return "Tighten stops", "Risk-reward unattractive while trend is weak."
    if sentiment < -0.2 and momentum <= 0:
        return "Monitor closely", "Bearish sentiment could pressure price."
    return "Hold", "Signals mixed; maintain but monitor."


def analyze_portfolio(portfolio: pd.DataFrame, config: EngineConfig, output_dir: str | None = None) -> Tuple[pd.DataFrame, PortfolioSummary]:
    """Evaluate an ETF portfolio for hold/trim/add decisions."""
    if "Symbol" not in portfolio.columns:
        raise ValueError("portfolio must include a 'Symbol' column")

    portfolio = portfolio.copy()
    portfolio["Shares"] = pd.to_numeric(portfolio.get("Shares", np.nan), errors="coerce")
    portfolio["CostBasis"] = pd.to_numeric(portfolio.get("CostBasis", np.nan), errors="coerce")

    symbols = portfolio["Symbol"].dropna().unique().tolist()
    if not symbols:
        raise ValueError("portfolio contains no symbols")

    end = dt.datetime.utcnow()
    start = end - dt.timedelta(days=365 * config.lookback_years)

    histories = fetch_price_history(symbols, start, end, interval="1d", max_workers=config.max_concurrent_requests)

    rows = []
    total_cost = 0.0
    total_value = 0.0
    total_risk_weight = 0.0
    weighted_returns = 0.0

    for _, row in portfolio.iterrows():
        symbol = row["Symbol"]
        history = histories.get(symbol)
        if history is None or history.empty:
            continue
        enriched = compute_indicators(history)
        risk = build_exit_plan(enriched)
        tech = technical_snapshot(enriched)
        volume = compute_volume_signals(enriched)
        sentiment = sentiment_for_symbol(symbol)
        latest_quote = fetch_latest_quote(symbol)
        last_price = float(latest_quote.get("last_price", enriched["Close"].iloc[-1]) if latest_quote else enriched["Close"].iloc[-1])
        cost_basis = float(row.get("CostBasis") or enriched["Close"].iloc[-1])
        shares = float(row.get("Shares") or 0.0)
        market_value = last_price * shares
        cost_value = cost_basis * shares
        pnl = market_value - cost_value
        return_pct = (last_price / cost_basis - 1.0) if cost_basis else float("nan")
        momentum_20d = float(enriched["Close"].pct_change(20).iloc[-1]) if len(enriched) > 20 else float("nan")
        trend = float(tech.get("ema55_vs_price", np.nan))
        action, rationale = _action_from_signals(return_pct, momentum_20d, risk.risk_reward_ratio, trend, sentiment.get("sentiment_score", 0.0))

        rows.append(
            {
                "Symbol": symbol,
                "Shares": shares,
                "CostBasis": cost_basis,
                "LastPrice": last_price,
                "UnrealizedPnL": pnl,
                "ReturnPct": return_pct,
                "Momentum20d": momentum_20d,
                "RiskReward": risk.risk_reward_ratio,
                "HoldPeriodDays": risk.hold_period_days,
                "VolumeZ": volume.volume_zscore,
                "UnusualVolume": volume.unusual_volume,
                "Sentiment": sentiment.get("sentiment_score", np.nan),
                "Action": action,
                "Rationale": rationale,
            }
        )

        total_cost += cost_value
        total_value += market_value
        if not np.isnan(risk.risk_reward_ratio):
            total_risk_weight += abs(risk.risk_reward_ratio)
            weighted_returns += abs(risk.risk_reward_ratio) * return_pct

    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("no analytics could be generated for portfolio symbols")

    total_pnl = total_value - total_cost
    total_return_pct = (total_value / total_cost - 1.0) if total_cost else float("nan")
    risk_weighted_return = (weighted_returns / total_risk_weight) if total_risk_weight else float("nan")
    summary = PortfolioSummary(total_cost, total_value, total_pnl, total_return_pct, risk_weighted_return)

    if output_dir:
        ensure_directory(output_dir)
        result.to_csv(f"{output_dir}/portfolio_review.csv", index=False)

    return result, summary
