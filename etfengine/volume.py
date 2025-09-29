"""Volume intelligence layer."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from .utils import rolling_vwap


@dataclass
class VolumeSignals:
    avg_volume: float
    volume_zscore: float
    obv_trend: float
    vwap_deviation: float
    unusual_volume: bool

    def to_dict(self) -> Dict[str, float | bool]:
        return {
            "avg_volume": self.avg_volume,
            "volume_zscore": self.volume_zscore,
            "obv_trend": self.obv_trend,
            "vwap_deviation": self.vwap_deviation,
            "unusual_volume": self.unusual_volume,
        }


def compute_volume_signals(df: pd.DataFrame) -> VolumeSignals:
    if df.empty:
        return VolumeSignals(0.0, 0.0, 0.0, 0.0, False)
    volume = df["Volume"].astype(float)
    avg_volume = float(volume.tail(20).mean())
    vol_z = float(((volume - volume.rolling(60).mean()) / (volume.rolling(60).std() + 1e-9)).iloc[-1])
    close = df["Close"].astype(float)
    obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    obv_trend = float(obv.diff().tail(10).sum())
    vwap = rolling_vwap(df, 30)
    vwap_dev = float((close.iloc[-1] - vwap.iloc[-1]) / vwap.iloc[-1]) if not np.isnan(vwap.iloc[-1]) else 0.0
    unusual = vol_z > 2.5
    return VolumeSignals(avg_volume, vol_z, obv_trend, vwap_dev, unusual)
