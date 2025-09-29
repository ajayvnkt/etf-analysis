"""Pattern recognition for ETFs."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PatternSignal:
    name: str
    confidence: float
    description: str


def detect_double_bottom(df: pd.DataFrame, lookback: int = 120) -> PatternSignal | None:
    if len(df) < lookback:
        return None
    window = df.tail(lookback)["Close"]
    min_idx = window.idxmin()
    min_price = window.min()
    before = window[:min_idx]
    after = window[min_idx:]
    if before.empty or after.empty:
        return None
    second_min = after.min()
    diff = abs(second_min - min_price) / min_price
    if diff < 0.03:
        neckline = max(before.max(), after.max())
        confidence = float((neckline - min_price) / neckline)
        return PatternSignal("Double Bottom", confidence, "Potential reversal after double trough")
    return None


def detect_cup_handle(df: pd.DataFrame, lookback: int = 180) -> PatternSignal | None:
    if len(df) < lookback:
        return None
    window = df.tail(lookback)["Close"]
    peak = window.idxmax()
    trough = window.idxmin()
    if trough < peak:
        return None
    peak_price = window.loc[peak]
    trough_price = window.loc[trough]
    depth = (peak_price - trough_price) / peak_price
    if depth < 0.1 or depth > 0.5:
        return None
    recent = window.tail(30)
    handle_depth = (recent.max() - recent.min()) / recent.max()
    if handle_depth < 0.05:
        confidence = float((recent.iloc[-1] - recent.min()) / (recent.max() - recent.min() + 1e-9))
        return PatternSignal("Cup & Handle", confidence, "Bullish continuation setup")
    return None


def detect_flags(df: pd.DataFrame, lookback: int = 60) -> PatternSignal | None:
    if len(df) < lookback:
        return None
    window = df.tail(lookback)
    price = window["Close"]
    returns = price.pct_change()
    if returns.tail(5).mean() > 0.02 and returns.tail(5).std() < 0.01:
        return PatternSignal("Bull Flag", 0.6, "Strong run-up followed by tight consolidation")
    if returns.tail(5).mean() < -0.02 and returns.tail(5).std() < 0.01:
        return PatternSignal("Bear Flag", 0.6, "Sharp drop with sideways drift")
    return None


def detect_patterns(df: pd.DataFrame) -> List[PatternSignal]:
    patterns: List[PatternSignal] = []
    for detector in (detect_double_bottom, detect_cup_handle, detect_flags):
        signal = detector(df)
        if signal:
            patterns.append(signal)
    return patterns
