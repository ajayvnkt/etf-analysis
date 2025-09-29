"""Utility helpers for ETF intelligence engine."""
from __future__ import annotations

import asyncio
import contextlib
import datetime as dt
import functools
import json
import logging
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable, List, Sequence

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


def ensure_directory(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def chunked(seq: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


async def gather_with_concurrency(limit: int, *tasks: Awaitable[Any]) -> List[Any]:
    semaphore = asyncio.Semaphore(limit)

    async def sem_task(task: Awaitable[Any]) -> Any:
        async with semaphore:
            return await task

    return await asyncio.gather(*(sem_task(t) for t in tasks))


def annualize_volatility(returns: pd.Series, periods: int = 252) -> float:
    if returns.empty:
        return float("nan")
    return float(returns.std(ddof=1) * math.sqrt(periods))


def trading_days_between(start: dt.datetime, end: dt.datetime) -> int:
    days = np.busday_count(start.date(), end.date())
    return int(days)


def rolling_vwap(df: pd.DataFrame, window: int = 20) -> pd.Series:
    if {"Close", "Volume"}.issubset(df.columns):
        pv = df["Close"] * df["Volume"]
        return pv.rolling(window=window).sum() / df["Volume"].rolling(window=window).sum()
    return pd.Series(index=df.index, dtype=float)


def normalize(series: pd.Series) -> pd.Series:
    if series.empty:
        return series
    return (series - series.min()) / (series.max() - series.min() + 1e-9)


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=str)


def write_csv(path: str | Path, df: pd.DataFrame) -> None:
    path = Path(path)
    ensure_directory(path.parent)
    df.to_csv(path, index=False)


def read_csv_if_exists(path: str | Path) -> pd.DataFrame | None:
    path = Path(path)
    if path.exists():
        return pd.read_csv(path)
    return None


@dataclass
class Recommendation:
    symbol: str
    thesis: str
    conviction: float
    expected_return: float
    expected_vol: float
    hold_period_days: int
    entry_price: float
    stop_loss: float
    take_profit: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "thesis": self.thesis,
            "conviction": round(self.conviction, 3),
            "expected_return": round(self.expected_return, 4),
            "expected_vol": round(self.expected_vol, 4),
            "hold_period_days": self.hold_period_days,
            "entry_price": round(self.entry_price, 4),
            "stop_loss": round(self.stop_loss, 4),
            "take_profit": round(self.take_profit, 4),
        }


def describe_statistically(series: pd.Series) -> dict[str, float]:
    series = series.dropna()
    if series.empty:
        return {}
    return {
        "mean": float(series.mean()),
        "median": float(series.median()),
        "std": float(series.std(ddof=1)),
        "skew": float(series.skew()),
        "kurtosis": float(series.kurtosis()),
        "min": float(series.min()),
        "max": float(series.max()),
    }


def apply_asyncio(event_loop: asyncio.AbstractEventLoop, coro: Awaitable[Any]) -> Any:
    if event_loop.is_running():
        return asyncio.ensure_future(coro)
    return event_loop.run_until_complete(coro)


@contextlib.contextmanager
def timer(name: str) -> Iterable[None]:
    start = dt.datetime.now()
    logger.info("Starting %s", name)
    yield
    elapsed = dt.datetime.now() - start
    logger.info("Finished %s in %.2fs", name, elapsed.total_seconds())


def rolling_zscore(series: pd.Series, window: int = 20) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std(ddof=1)
    return (series - mean) / (std + 1e-9)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def risk_reward(entry: float, stop: float, target: float) -> float:
    risk = entry - stop
    reward = target - entry
    if risk <= 0:
        return float("nan")
    return reward / risk
