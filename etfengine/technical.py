"""Technical analysis utilities."""
from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import pandas as pd
import pandas_ta as ta

from .utils import atr, rolling_vwap, rolling_zscore

logger = logging.getLogger(__name__)


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["RSI14"] = ta.rsi(out["Close"], length=14)
    adx = ta.adx(out["High"], out["Low"], out["Close"], length=14)
    if adx is not None and not adx.empty:
        out["ADX14"] = adx.get("ADX_14")
    macd = ta.macd(out["Close"])
    if macd is not None and not macd.empty:
        out["MACD"] = macd.get("MACD_12_26_9")
        out["MACD_SIGNAL"] = macd.get("MACDs_12_26_9")
    stoch = ta.stoch(out["High"], out["Low"], out["Close"])
    if stoch is not None and not stoch.empty:
        out["STOCHK"] = stoch.get("STOCHk_14_3_3")
        out["STOCHD"] = stoch.get("STOCHd_14_3_3")
    out["EMA21"] = ta.ema(out["Close"], length=21)
    out["EMA55"] = ta.ema(out["Close"], length=55)
    out["ATR14"] = atr(out, 14)
    out["VWAP20"] = rolling_vwap(out, 20)
    out["ZScore20"] = rolling_zscore(out["Close"], 20)
    out["OBV"] = ta.obv(out["Close"], out["Volume"])
    out["MOM126"] = ta.mom(out["Close"], length=126)
    return out


def technical_snapshot(df: pd.DataFrame) -> Dict[str, float]:
    if df.empty:
        return {}
    latest = df.iloc[-1]
    snapshot = {
        "rsi14": float(latest.get("RSI14", np.nan)),
        "adx14": float(latest.get("ADX14", np.nan)),
        "ema21_vs_price": float((latest.get("Close") - latest.get("EMA21")) / latest.get("EMA21")),
        "ema55_vs_price": float((latest.get("Close") - latest.get("EMA55")) / latest.get("EMA55")),
        "macd_hist": float(latest.get("MACD", np.nan) - latest.get("MACD_SIGNAL", np.nan)),
        "stoch_k": float(latest.get("STOCHK", np.nan)),
        "atr_pct": float(latest.get("ATR14", np.nan) / latest.get("Close", np.nan)),
        "zscore20": float(latest.get("ZScore20", np.nan)),
        "obv_trend": float(df["OBV"].diff().tail(20).sum()),
        "vwap_gap": float((latest.get("Close") - latest.get("VWAP20")) / latest.get("VWAP20")),
    }
    return snapshot
