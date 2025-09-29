"""Configuration defaults for the ETF intelligence engine."""
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class EngineConfig:
    """Runtime configuration for the ETF engine."""

    benchmark: str = "SPY"
    risk_free_rate: float = 0.02
    min_aum_m: float = 100.0
    lookback_years: int = 7
    topn_ranked: int = 200
    topn_shortlist: int = 20
    premarket_providers: List[str] = field(default_factory=lambda: ["yfinance", "alphavantage"])
    alpha_vantage_api_key: str | None = None
    polygon_api_key: str | None = None
    yahoo_api_key: str | None = None
    cache_expiry_minutes: int = 30
    risk_target_vol: float = 0.18
    max_concurrent_requests: int = 12
    ml_model_path: str = "models/etf_ml_model.joblib"
    dashboard_output: str = "intel_dashboard.html"
    enable_plotly: bool = True
    news_sources: List[str] = field(default_factory=lambda: ["yfinance"])
    run_date: _dt.date = field(default_factory=lambda: _dt.datetime.utcnow().date())

    # Files
    output_dir: str = "output"
    ranked_csv: str = "output/etf_ranked.csv"
    shortlist_csv: str = "output/daily_recommendations.csv"
    shortlist_json: str = "output/daily_recommendations.json"
    fundamentals_csv: str = "output/etf_fundamentals.csv"
    dataset_csv: str = "output/etf_final_dataset.csv"
    corr_csv: str = "output/etf_pairwise_corr.csv"


DEFAULT_CONFIG = EngineConfig()
