"""Core orchestration for the ETF intelligence engine."""
from __future__ import annotations

import asyncio
import datetime as dt
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .config import EngineConfig
from .data_sources import (
    fetch_benchmark,
    fetch_latest_quote,
    fetch_premarket_quotes,
    fetch_price_history,
    load_etf_universe,
)
from .ml_models import FEATURE_COLUMNS, load_or_train_model, predict_returns
from .patterns import PatternSignal, detect_patterns
from .risk import RiskProfile, build_exit_plan
from .sentiment import sentiment_for_symbol
from .technical import compute_indicators, technical_snapshot
from .utils import Recommendation, describe_statistically, ensure_directory, timer, write_csv, write_json
from .volume import VolumeSignals, compute_volume_signals

logger = logging.getLogger(__name__)


@dataclass
class SymbolAnalytics:
    symbol: str
    fundamentals: dict
    history: pd.DataFrame
    enriched: pd.DataFrame
    technicals: dict
    volume: VolumeSignals
    patterns: List[PatternSignal]
    sentiment: dict
    risk: RiskProfile
    latest_quote: dict | None


class ETFIntelligenceEngine:
    def __init__(self, config: EngineConfig):
        self.config = config

    def run(self, csv_path: str) -> Dict[str, any]:
        with timer("load-universe"):
            universe = load_etf_universe(csv_path, self.config.min_aum_m)
        symbols = universe.index.tolist()
        end = dt.datetime.utcnow()
        start = end - dt.timedelta(days=365 * self.config.lookback_years)

        with timer("fetch-history"):
            histories = fetch_price_history(symbols, start, end, interval="1d", max_workers=self.config.max_concurrent_requests)
        with timer("fetch-benchmark"):
            benchmark = fetch_benchmark(self.config.benchmark, start, end)

        analytics: Dict[str, SymbolAnalytics] = {}
        historical_features: List[dict] = []

        for symbol, hist in histories.items():
            enriched = compute_indicators(hist)
            tech = technical_snapshot(enriched)
            volume = compute_volume_signals(enriched)
            patterns = detect_patterns(enriched)
            sentiment = sentiment_for_symbol(symbol)
            risk = build_exit_plan(enriched)
            latest_quote = fetch_latest_quote(symbol)
            analytics[symbol] = SymbolAnalytics(
                symbol=symbol,
                fundamentals=universe.loc[symbol].to_dict(),
                history=hist,
                enriched=enriched,
                technicals=tech,
                volume=volume,
                patterns=patterns,
                sentiment=sentiment,
                risk=risk,
                latest_quote=latest_quote,
            )

            returns = enriched["Close"].pct_change().dropna()
            monthly = returns.resample("M").apply(lambda x: (1 + x).prod() - 1)
            hist_features = {
                "Symbol": symbol,
                "momentum_1m": float(monthly.tail(1).mean()),
                "momentum_3m": float(monthly.tail(3).mean()),
                "momentum_6m": float(monthly.tail(6).mean()),
                "momentum_12m": float(monthly.tail(12).mean()),
                "volatility": float(returns.std() * np.sqrt(252)),
                "max_drawdown": float(((enriched["Close"].cummax() - enriched["Close"]) / enriched["Close"].cummax()).max()),
                "sharpe": float(monthly.mean() / (monthly.std() + 1e-9) * np.sqrt(12)),
                "sortino": float(monthly.mean() / (monthly[monthly < 0].std() + 1e-9) * np.sqrt(12)),
                "vwap_gap": volume.vwap_deviation,
                "volume_z": volume.volume_zscore,
                "obv_trend": volume.obv_trend,
                "pattern_confidence": float(max((p.confidence for p in patterns), default=0.0)),
                "sentiment_score": sentiment.get("sentiment_score", 0.0),
                "forward_return": float(monthly.shift(-1).iloc[-1]) if len(monthly) > 1 else np.nan,
            }
            historical_features.append(hist_features)

        features_df = pd.DataFrame(historical_features)

        with timer("ml-model"):
            try:
                model, score = load_or_train_model(features_df, self.config.ml_model_path)
                logger.info("ML model R2: %s", score)
            except ValueError:
                model, score = load_or_train_model(features_df.fillna(0), self.config.ml_model_path)

        inference_features = features_df.set_index("Symbol")
        predictions = predict_returns(model, inference_features)
        inference_features["predicted_return"] = predictions

        with timer("premarket"):
            premarket_quotes = asyncio.run(fetch_premarket_quotes(symbols, self.config))

        ranked_records = []
        recs: List[Recommendation] = []
        for symbol in symbols:
            if symbol not in analytics:
                continue
            data = analytics[symbol]
            feat = inference_features.loc[symbol]
            patterns = data.patterns
            pattern_desc = ", ".join(f"{p.name} ({p.confidence:.2f})" for p in patterns) if patterns else "None"
            rec = Recommendation(
                symbol=symbol,
                thesis=f"Momentum {feat['momentum_3m']:.2%}, pattern: {pattern_desc}",
                conviction=float(np.clip(feat["predicted_return"] * 10, 0, 1)),
                expected_return=float(feat["predicted_return"]),
                expected_vol=float(feat["volatility"]),
                hold_period_days=data.risk.hold_period_days,
                entry_price=float(data.latest_quote.get("last_price", data.enriched["Close"].iloc[-1]) if data.latest_quote else data.enriched["Close"].iloc[-1]),
                stop_loss=data.risk.stop_loss,
                take_profit=data.risk.take_profit,
            )
            recs.append(rec)

            ranked_records.append(
                {
                    "Symbol": symbol,
                    **data.fundamentals,
                    **feat.to_dict(),
                    **data.technicals,
                    **data.volume.to_dict(),
                    **data.sentiment,
                    "predicted_return": feat["predicted_return"],
                    "premarket": premarket_quotes.get(symbol, {}),
                    "patterns": pattern_desc,
                    "hold_period_days": data.risk.hold_period_days,
                    "stop_loss": data.risk.stop_loss,
                    "take_profit": data.risk.take_profit,
                    "risk_reward_ratio": data.risk.risk_reward_ratio,
                }
            )

        ranked_df = pd.DataFrame(ranked_records).sort_values("predicted_return", ascending=False)
        shortlist_df = ranked_df.head(self.config.topn_shortlist)
        recs_data = [rec.to_dict() for rec in recs if rec.symbol in shortlist_df["Symbol"].values]
        recommendations_df = pd.DataFrame(recs_data)

        ensure_directory(self.config.output_dir)
        write_csv(self.config.dataset_csv, ranked_df)
        write_csv(self.config.shortlist_csv, shortlist_df)
        write_csv(self.config.ranked_csv, ranked_df.head(self.config.topn_ranked))
        write_json(self.config.shortlist_json, recs_data)

        summary = {
            "as_of": dt.datetime.utcnow().isoformat(),
            "universe": len(symbols),
            "top_recommendations": recs_data,
            "momentum_stats": describe_statistically(ranked_df["momentum_3m"]),
        }
        return {
            "ranked": ranked_df,
            "shortlist": shortlist_df,
            "recommendations": recommendations_df,
            "summary": summary,
            "premarket": premarket_quotes,
        }
