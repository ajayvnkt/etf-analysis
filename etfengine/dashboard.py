"""Plotly dashboard generation."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .utils import ensure_directory

logger = logging.getLogger(__name__)


def build_dashboard(ranked_df: pd.DataFrame, recommendations: pd.DataFrame, output_path: str) -> None:
    ensure_directory(Path(output_path).parent)
    fig = make_subplots(rows=2, cols=2, subplot_titles=(
        "Top Momentum ETFs",
        "Risk/Reward Distribution",
        "Sentiment vs Predicted Return",
        "Hold Period Recommendations",
    ))

    top_momo = ranked_df.sort_values("momentum_12m", ascending=False).head(10)
    fig.add_trace(go.Bar(x=top_momo["Symbol"], y=top_momo["momentum_12m"], name="12m Momentum"), row=1, col=1)

    fig.add_trace(
        go.Scatter(
            x=recommendations["risk_reward_ratio"],
            y=recommendations["expected_return"],
            mode="markers",
            text=recommendations["Symbol"],
            name="Risk/Reward",
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Scatter(
            x=recommendations["sentiment_score"],
            y=recommendations["expected_return"],
            mode="markers",
            text=recommendations["Symbol"],
            name="Sentiment vs Return",
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Bar(x=recommendations["Symbol"], y=recommendations["hold_period_days"], name="Hold Days"),
        row=2,
        col=2,
    )

    fig.update_layout(title="ETF Super-Intelligence Dashboard", height=900, showlegend=False)
    fig.write_html(output_path)
    logger.info("Dashboard saved to %s", output_path)
