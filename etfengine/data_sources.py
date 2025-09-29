"""Data acquisition layer for ETF intelligence engine."""
from __future__ import annotations

import asyncio
import datetime as dt
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Iterable, List, Optional

import aiohttp
import numpy as np
import pandas as pd
import yfinance as yf

from .config import EngineConfig
from .utils import ensure_directory

logger = logging.getLogger(__name__)


def load_etf_universe(csv_path: str, min_aum: float) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "Symbol" not in df.columns:
        raise ValueError("CSV must contain Symbol column")
    df = df.rename(columns={col: col.strip() for col in df.columns})
    if "Assets" in df.columns:
        df["Assets"] = (
            df["Assets"].astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False)
        )
        df["Assets"] = pd.to_numeric(df["Assets"], errors="coerce")
        df = df[df["Assets"] >= min_aum * 1_000_000]
    df = df.drop_duplicates("Symbol").set_index("Symbol")
    return df


def _fetch_history(symbol: str, start: dt.datetime, end: dt.datetime, interval: str = "1d") -> pd.DataFrame:
    logger.debug("Fetching history %s %s-%s %s", symbol, start, end, interval)
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(start=start, end=end, interval=interval, auto_adjust=False, prepost=True)
        data = data.rename(columns=str.title)
        data["Symbol"] = symbol
        return data
    except Exception as exc:
        logger.warning("History fetch failed for %s: %s", symbol, exc)
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume", "Symbol"])


def fetch_price_history(symbols: Iterable[str], start: dt.datetime, end: dt.datetime, interval: str = "1d",
                        max_workers: int = 12) -> Dict[str, pd.DataFrame]:
    histories: Dict[str, pd.DataFrame] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(lambda s: _fetch_history(s, start, end, interval), symbols))
    for df in results:
        if not df.empty:
            histories[df["Symbol"].iloc[0]] = df
    return histories


def fetch_latest_quote(symbol: str) -> Optional[dict]:
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.fast_info
        return {
            "symbol": symbol,
            "last_price": info.get("last_price"),
            "prev_close": info.get("previous_close"),
            "last_volume": info.get("last_volume"),
            "currency": info.get("currency"),
        }
    except Exception as exc:
        logger.warning("Failed to fetch fast info for %s: %s", symbol, exc)
        return None


async def fetch_premarket_quotes(symbols: List[str], config: EngineConfig) -> Dict[str, dict]:
    results: Dict[str, dict] = {}
    loop = asyncio.get_event_loop()

    async def fetch_from_yfinance(sym: str) -> Optional[dict]:
        return await loop.run_in_executor(None, _get_premarket_yf, sym)

    async def fetch_from_alpha_vantage(session: aiohttp.ClientSession, sym: str) -> Optional[dict]:
        if not config.alpha_vantage_api_key:
            return None
        url = (
            "https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol="
            f"{sym}&interval=5min&datatype=json&apikey={config.alpha_vantage_api_key}"
        )
        try:
            async with session.get(url, timeout=20) as resp:
                payload = await resp.json()
        except Exception as exc:
            logger.warning("Alpha Vantage request failed for %s: %s", sym, exc)
            return None
        series = payload.get("Time Series (5min)")
        if not series:
            return None
        latest_time = sorted(series.keys())[-1]
        latest = series[latest_time]
        return {
            "symbol": sym,
            "timestamp": latest_time,
            "open": float(latest.get("1. open", np.nan)),
            "high": float(latest.get("2. high", np.nan)),
            "low": float(latest.get("3. low", np.nan)),
            "close": float(latest.get("4. close", np.nan)),
            "volume": float(latest.get("5. volume", np.nan)),
            "provider": "alphavantage",
        }

    async with aiohttp.ClientSession() as session:
        tasks = []
        for sym in symbols:
            if "yfinance" in config.premarket_providers:
                tasks.append(fetch_from_yfinance(sym))
            if "alphavantage" in config.premarket_providers and config.alpha_vantage_api_key:
                tasks.append(fetch_from_alpha_vantage(session, sym))
        responses = await asyncio.gather(*tasks)

    for resp in responses:
        if isinstance(resp, dict):
            results[resp["symbol"]] = resp
    return results


def _get_premarket_yf(symbol: str) -> Optional[dict]:
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="1d", interval="1m", prepost=True)
        if df.empty:
            return None
        premarket = df[df.index.get_level_values(0).tz_convert("US/Eastern").time < dt.time(9, 30)]
        if premarket.empty:
            return None
        last_row = premarket.iloc[-1]
        return {
            "symbol": symbol,
            "timestamp": last_row.name.isoformat(),
            "open": float(last_row.get("Open", np.nan)),
            "high": float(last_row.get("High", np.nan)),
            "low": float(last_row.get("Low", np.nan)),
            "close": float(last_row.get("Close", np.nan)),
            "volume": float(last_row.get("Volume", np.nan)),
            "provider": "yfinance",
        }
    except Exception as exc:
        logger.debug("Premarket not available for %s: %s", symbol, exc)
        return None


def save_price_histories(histories: Dict[str, pd.DataFrame], output_dir: str) -> None:
    ensure_directory(output_dir)
    for symbol, df in histories.items():
        df.to_csv(f"{output_dir}/{symbol}_history.csv")


def fetch_benchmark(benchmark: str, start: dt.datetime, end: dt.datetime) -> pd.DataFrame:
    return _fetch_history(benchmark, start, end, interval="1d")
