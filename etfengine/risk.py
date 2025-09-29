"""Risk management utilities."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from .utils import atr, risk_reward, trading_days_between


@dataclass
class RiskProfile:
    hold_period_days: int
    stop_loss: float
    take_profit: float
    risk_reward_ratio: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "hold_period_days": self.hold_period_days,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "risk_reward_ratio": self.risk_reward_ratio,
        }


def estimate_hold_period(momentum: float, volatility: float) -> int:
    base = 5 + 20 * max(momentum, 0)
    vol_adj = 15 * (0.3 - min(volatility, 0.3))
    return int(max(3, base + vol_adj))


def build_exit_plan(df: pd.DataFrame) -> RiskProfile:
    if df.empty:
        return RiskProfile(5, float("nan"), float("nan"), float("nan"))
    close = df["Close"].iloc[-1]
    atr14 = atr(df, 14).iloc[-1]
    if math.isnan(atr14):
        atr14 = df["Close"].pct_change().std() * close
    stop = close - 1.5 * atr14
    target = close + 3 * atr14
    ratio = risk_reward(close, stop, target)
    recent_mom = df["Close"].pct_change(20).iloc[-1]
    vol = df["Close"].pct_change().std() * math.sqrt(252)
    hold_days = estimate_hold_period(max(recent_mom, 0), vol)
    return RiskProfile(hold_days, float(stop), float(target), float(ratio))
