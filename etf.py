# ====================== ETF Monthly Scanner & Picker (CSV + yfinance Monthly) ======================
# Uses etfdb_screener.csv for universe and fundamentals
# Fetches monthly OHLC data from yfinance for full metrics (max drawdown, alpha, ADX, etc.)
# Computes metrics: Sharpe, Sortino, Treynor, RSI, ADX, skewness, etc.
# Shortlists: momentum, pullbacks, low-vol, high-yield bonds
# Outputs: CSVs with metrics, rankings, shortlists
#
# Requirements:
#   pip install yfinance pandas numpy requests-cache matplotlib
#
# Usage examples:
#   python etf_scanner_csv_yfinance_monthly.py --csv etfdb_screener.csv --rf 0.045 --min-aum 50
# =============================================================================

from __future__ import annotations
import os, sys, math, time, argparse, logging, multiprocessing as mp
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Caching
try:
    import requests_cache
    requests_cache.install_cache("yf_cache", expire_after=3600*24)  # 1 day
except Exception:
    pass

pd.options.mode.copy_on_write = True
logging.basicConfig(filename='etf_scanner.log', level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# ------------------------------ Config ---------------------------------------
LOOKBACK_YEARS   = 5
EXTRA_BACK_DAYS  = 30
CHUNK_SIZE       = 60
THROTTLE_S       = 0.5
BENCH            = "SPY"
RISK_FREE        = 0.00
MIN_AUM_M        = 100
TOPN_LIST        = 15
TOPN_RANKED      = 150

# Weights for composite scoring
W_MOM12_1  = 0.25
W_MOM6     = 0.15
W_SHARPE   = 0.12
W_SORTINO  = 0.10
W_TREYNOR  = 0.08
W_UPCAP    = 0.07
W_MAXDD    = 0.08
W_10MO     = 0.05
W_SKEW     = 0.05
W_RSI      = 0.05

# Autodetect CSV path helper
def resolve_csv_path(user_csv: Optional[str]) -> Optional[str]:
    """Resolve the ETF CSV path. Prefer user-provided, else local etfdb_screener.csv near this script or CWD."""
    candidates = []
    if user_csv:
        candidates.append(user_csv)
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        candidates.extend([
            os.path.join(script_dir, "etfdb_screener.csv"),
            os.path.join(os.getcwd(), "etfdb_screener.csv"),
        ])
    except Exception:
        candidates.append("etfdb_screener.csv")
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None

# ---------------------------- Helper Functions -------------------------------
def nz(series, default=0):
    """Null-safe series handling"""
    if series is None or (isinstance(series, (float, int)) and math.isnan(series)):
        return default
    if isinstance(series, pd.Series):
        return series.fillna(default)
    return series

def pct(val, base=100):
    """Convert to percentage"""
    try:
        return float(val) * base
    except:
        return np.nan

def parse_currency(value):
    """Convert currency string (e.g., '$653,311,000,000') to float"""
    try:
        return float(value.replace('$', '').replace(',', ''))
    except:
        return np.nan

def monthly_resample(series):
    """Resample data to monthly (already monthly from yfinance)"""
    if series.empty or not isinstance(series.index, pd.DatetimeIndex):
        series.index = pd.to_datetime(series.index)
    return series.resample('ME').last()

def returns_from_prices(prices):
    """Calculate returns from price series"""
    return prices.pct_change().dropna()

def annualize_vol(returns, periods=12):
    """Annualize volatility (monthly data)"""
    std = returns.std(ddof=1)
    return std * math.sqrt(periods) if not pd.isna(std) else np.nan

def sharpe_ratio(returns, risk_free=0, periods=12):
    """Calculate Sharpe ratio"""
    if len(returns) < 3:
        return np.nan
    excess_returns = returns - risk_free / periods
    std = excess_returns.std(ddof=1)
    return (excess_returns.mean() / std) * math.sqrt(periods) if std != 0 else np.nan

def sortino_ratio(returns, risk_free=0, periods=12):
    """Calculate Sortino ratio"""
    if len(returns) < 3:
        return np.nan
    excess_returns = returns - risk_free / periods
    downside_returns = excess_returns[excess_returns < 0]
    downside_std = downside_returns.std(ddof=1)
    return (excess_returns.mean() / downside_std) * math.sqrt(periods) if downside_std != 0 else np.nan

def max_drawdown(prices):
    """Calculate maximum drawdown (positive %)"""
    if prices.empty:
        return np.nan
    peak = prices.cummax()
    drawdown = (peak - prices) / peak * 100
    return drawdown.max() if not drawdown.empty else np.nan

def cagr_from_prices(prices):
    """Calculate CAGR from price series (monthly)"""
    if len(prices) < 12:
        return np.nan
    years = len(prices) / 12
    start, end = prices.iloc[0], prices.iloc[-1]
    if start <= 0 or pd.isna(start) or pd.isna(end):
        return np.nan
    return ((end / start) ** (1 / years) - 1) * 100

def beta_alpha(returns, benchmark_returns):
    """Calculate beta and alpha (monthly, annualized alpha)"""
    if len(returns) < 12 or len(benchmark_returns) < 12:
        return np.nan, np.nan
    aligned = pd.concat([returns, benchmark_returns], axis=1, join='inner').dropna()
    if len(aligned) < 12:
        return np.nan, np.nan
    ret_vals = aligned.iloc[:, 0].values
    bench_vals = aligned.iloc[:, 1].values
    covariance = np.cov(ret_vals, bench_vals, ddof=1)[0, 1]
    benchmark_variance = np.var(bench_vals, ddof=1)
    beta = covariance / benchmark_variance if benchmark_variance != 0 else np.nan
    alpha = (np.mean(ret_vals) - beta * np.mean(bench_vals)) * 12 * 100 if not pd.isna(beta) else np.nan
    return alpha, beta

def up_down_capture(mrets: pd.Series, bmr: pd.Series) -> Tuple[float, float]:
    """Calculate up/down capture ratios"""
    if len(mrets) < 12 or len(bmr) < 12:
        return np.nan, np.nan
    common = pd.concat([mrets, bmr], axis=1).dropna()
    if len(common) < 12:
        return np.nan, np.nan
    up = common[common.iloc[:, 1] > 0]
    down = common[common.iloc[:, 1] < 0]
    upcap = (1 + up.iloc[:, 0]).prod() ** (12 / len(up)) - 1 if len(up) > 0 else np.nan
    downcap = (1 + down.iloc[:, 0]).prod() ** (12 / len(down)) - 1 if len(down) > 0 else np.nan
    return upcap * 100, downcap * 100

def treynor_ratio(mrets: pd.Series, rf_annual: float, beta: float) -> float:
    """Calculate Treynor ratio"""
    if pd.isna(beta) or beta == 0 or len(mrets) < 12:
        return np.nan
    rf_m = rf_annual / 12.0
    ex_mean = (mrets - rf_m).mean() * 12
    return ex_mean / beta

def information_ratio(mrets: pd.Series, bmr: pd.Series) -> float:
    """Calculate Information ratio"""
    if len(mrets) < 12:
        return np.nan
    ex = mrets - bmr
    std = ex.std(ddof=1)
    return ex.mean() / std * math.sqrt(12) if std != 0 else np.nan

def rsi(series: pd.Series, period: int = 14) -> float:
    """Calculate RSI"""
    if len(series) < period + 1:
        return 50
    delta = series.diff()
    gain = nz(delta.where(delta > 0, 0), 0)
    loss = nz(-delta.where(delta < 0, 0), 0)
    avg_gain = gain.rolling(period).mean().iloc[-1]
    avg_loss = loss.rolling(period).mean().iloc[-1]
    if avg_loss == 0:
        return 100
    return 100 - (100 / (1 + avg_gain / avg_loss))

def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
    """Calculate ADX (monthly)"""
    if len(high) < period + 1 or len(low) < period + 1 or len(close) < period + 1:
        return np.nan
    tr = np.maximum(high - low, np.maximum(abs(high - close.shift()), abs(low - close.shift())))
    dm_plus = high - high.shift()
    dm_minus = low.shift() - low
    dm_plus[dm_plus < dm_minus] = 0
    dm_minus[dm_minus < dm_plus] = 0
    atr = tr.rolling(period).mean()
    di_plus = 100 * (dm_plus.rolling(period).mean() / atr)
    di_minus = 100 * (dm_minus.rolling(period).mean() / atr)
    dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus + 1e-6)
    return dx.rolling(period).mean().iloc[-1]

def winsorize(series, lower=0.05, upper=0.95):
    """Winsorize series to remove outliers"""
    if series.dropna().empty:
        return series
    return series.clip(lower=series.quantile(lower), upper=series.quantile(upper))

def zscore(series):
    """Calculate z-score"""
    if series.std(ddof=1) == 0:
        return pd.Series(0, index=series.index)
    return (series - series.mean()) / series.std(ddof=1)

# ---------------------------- Data Download ----------------------------------
def download_chunked(symbols: List[str], period: str, interval: str = "1mo") -> Tuple[pd.DataFrame, List[str]]:
    """Download monthly data in chunks with error handling"""
    logging.info(f"Downloading monthly data for {len(symbols)} symbols...")
    all_data = []
    successful_symbols = []
    for i in range(0, len(symbols), CHUNK_SIZE):
        chunk = symbols[i:i+CHUNK_SIZE]
        try:
            data = yf.download(chunk, period=period, interval=interval, group_by='ticker',
                             progress=False, threads=True, auto_adjust=False, prepost=False)
            if data.empty:
                continue
            if len(chunk) == 1:
                symbol = chunk[0]
                if len(data.dropna()) > 12:
                    all_data.append(pd.concat({symbol: data}, axis=1))
                    successful_symbols.append(symbol)
            else:
                for symbol in chunk:
                    try:
                        symbol_data = data.xs(symbol, level=0, axis=1)
                        if len(symbol_data.dropna()) > 12:
                            all_data.append(pd.concat({symbol: symbol_data}, axis=1))
                            successful_symbols.append(symbol)
                    except:
                        continue
        except Exception as e:
            logging.warning(f"Failed to download chunk {i//CHUNK_SIZE + 1}: {e}")
        time.sleep(THROTTLE_S)
    if all_data:
        return pd.concat(all_data, axis=1), sorted(set(successful_symbols))
    # Fallback: try single-symbol downloads for a subset
    fallback_limit = min(300, len(symbols))
    for symbol in symbols[:fallback_limit]:
        try:
            data = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=False, prepost=False)
            if not data.empty and len(data.dropna()) > 12:
                all_data.append(pd.concat({symbol: data}, axis=1))
                successful_symbols.append(symbol)
            time.sleep(0.05)
        except Exception as e:
            continue
    if all_data:
        return pd.concat(all_data, axis=1), sorted(set(successful_symbols))
    return pd.DataFrame(), []

# ---------------------------- Universe Fetch --------------------------------
def build_universe(csv_file: Optional[str] = None) -> Dict[str, Dict[str, str]]:
    """Build universe from etfdb_screener.csv or default list"""
    if csv_file and os.path.exists(csv_file):
        # Detect header row by scanning for a line that contains 'Symbol'
        header_row = 0
        try:
            with open(csv_file, 'r', encoding='utf-8') as fh:
                for i in range(10):
                    pos = fh.tell()
                    line = fh.readline()
                    if not line:
                        break
                    if 'Symbol' in line and 'Name' in line:
                        header_row = i
                        break
                    header_row = i + 1
        except Exception:
            header_row = 0
        df = pd.read_csv(csv_file, skiprows=header_row)
        # Normalize column names
        cols = {c: c.strip() for c in df.columns}
        df.rename(columns=cols, inplace=True)
        # Helper parsers
        def parse_pct(x):
            try:
                s = str(x).strip()
                if s.endswith('%'):
                    s = s[:-1]
                return float(s) / 100.0
            except:
                return np.nan
        def parse_float(x):
            try:
                return float(str(x).replace(',', '').replace('$', ''))
            except:
                return np.nan
        uni = {}
        for _, row in df.iterrows():
            raw_sym = str(row.get('Symbol', '')).strip().upper()
            if not raw_sym:
                continue
            sym = raw_sym.replace('.', '-')
            uni[sym] = {
                "name": str(row.get('Name', '')),
                "category": str(row.get('Asset Class', '')),
                "aum_m": parse_float(row.get('Assets', np.nan)) / 1e6,
                "expense_%": parse_pct(row.get('ER', np.nan)) * 100,
                "div_yield_%": parse_float(row.get('Dividend, Annual Dividend Yield %', row.get('Annual Dividend Yield %', np.nan))),
                "ret_1m_%": parse_pct(row.get('1 Month Returns', np.nan)) * 100,
                "ret_12m_%": parse_pct(row.get('1 Year Returns', np.nan)) * 100,
                "ret_5y_%": parse_pct(row.get('5 Year Returns', np.nan)) * 100,
                "vol_ann_%": parse_pct(row.get('Standard Deviation', np.nan)) * 100 if 'Standard Deviation' in df.columns else np.nan,
                "beta": parse_float(row.get('Beta', np.nan)),
                "rsi": parse_float(row.get('RSI', np.nan)),
                "avg_dollar_vol": parse_float(row.get('Avg. Daily Volume', np.nan)) * parse_float(row.get('Price', np.nan)),
            }
        return uni
    default_etfs = [
        "SPY", "QQQ", "VTI", "VOO", "IVV", "VEA", "VWO", "AGG", "TLT", "GLD",
        "VNQ", "EFA", "IWM", "XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLY",
        "BND", "HYG", "LQD", "TIP", "SHY", "IEF", "SGOV", "BIL", "SMH", "SOXX"
    ]
    return {sym: {"name": f"{sym} ETF", "category": "ETF"} for sym in default_etfs}

# ---------------------------- Fundamentals -----------------------------------
def fetch_fundamentals(ticker: str, universe: Dict) -> Dict:
    """Fetch fundamentals from universe (CSV data)"""
    data = universe.get(ticker, {})
    return {
        "expense_%": data.get("expense_%", np.nan),
        "aum_m": data.get("aum_m", np.nan),
        "div_yield_%": data.get("div_yield_%", np.nan),
        "bidask_proxy_%": 0.1  # Placeholder, no bid/ask in CSV
    }

# ---------------------------- Main Computation -------------------------------
def compute_all(universe: Dict[str, Dict[str, str]], rf_annual: float, min_aum_m: float, csv_path: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Main computation using CSV fundamentals and yfinance monthly OHLC"""
    logging.info(f"Starting computation for {len(universe)} symbols...")
    symbols = sorted(set(universe.keys()) | {BENCH})
    uni_df = pd.DataFrame([
        {"symbol": s, "name": universe.get(s, {}).get("name", ""), "category": universe.get(s, {}).get("category", "")}
        for s in symbols if s != BENCH
    ])
    uni_df.to_csv("etf_universe_used.csv", index=False)

    # Pre-filter by AUM if available to avoid downloading tiny ETFs
    filtered_symbols = [s for s in symbols if (s == BENCH) or (universe.get(s, {}).get('aum_m', np.nan) >= min_aum_m)]
    if len(filtered_symbols) < 2:
        filtered_symbols = symbols

    # Download monthly OHLC
    years = LOOKBACK_YEARS
    days = int(years * 365 + EXTRA_BACK_DAYS)
    period_str = f"{days}d"
    D, present = download_chunked(filtered_symbols, period=period_str, interval="1mo")
    if D.empty or not present:
        raise RuntimeError("No OHLC data returned.")

    logging.info(f"Downloaded data for {len(present)} symbols")

    # Fetch fundamentals from CSV
    logging.info("Fetching fundamentals from CSV...")
    # Ensure we do not include the benchmark in subsequent loops
    present = [t for t in present if t != BENCH]
    with mp.Pool(mp.cpu_count()) as pool:
        fund_list = pool.starmap(fetch_fundamentals, [(t, universe) for t in present])
    fund_df = pd.DataFrame(fund_list, index=present).reset_index().rename(columns={'index': 'symbol'})
    fund_df.to_csv("etf_fundamentals.csv", index=False)

    logging.info(f"After download & fundamentals: {len(present)} symbols")

    # Benchmark data
    bench_data = D[BENCH]['Adj Close'].dropna() if BENCH in D.columns.get_level_values(0) else pd.Series(dtype=float)
    bench_mrets = returns_from_prices(bench_data) if not bench_data.empty else pd.Series(dtype=float)

    # Compute metrics
    rows = []
    rets_dict = {}
    for t in present:
        if t == BENCH:
            continue
        try:
            if t not in D.columns.get_level_values(0):
                continue
            sub = D[t]
            adj_close = sub['Adj Close'].dropna()
            if len(adj_close) < 12:
                continue
            mclose = monthly_resample(adj_close)
            mrets = returns_from_prices(mclose)
            mhigh = monthly_resample(sub["High"].dropna())
            mlow = monthly_resample(sub["Low"].dropna())
            if len(mrets) < 12:
                continue

            # Store for correlation
            rets_dict[t] = mrets

            # Basic metrics
            vol_ann = annualize_vol(mrets, 12) * 100
            sharpe = sharpe_ratio(mrets, rf_annual/12, 12)
            sortino = sortino_ratio(mrets, rf_annual/12, 12)
            max_dd = max_drawdown(mclose)
            cagr_5y = cagr_from_prices(mclose)

            # Growth metrics
            ret_1m = mrets.iloc[-1] * 100 if len(mrets) > 0 else np.nan
            ret_3m = (mclose.iloc[-1] / mclose.iloc[-4] - 1) * 100 if len(mclose) >= 4 else np.nan
            ret_6m = (mclose.iloc[-1] / mclose.iloc[-7] - 1) * 100 if len(mclose) >= 7 else np.nan
            ret_12m = (mclose.iloc[-1] / mclose.iloc[-13] - 1) * 100 if len(mclose) >= 13 else np.nan
            mom_12_1 = (mclose.iloc[-2] / mclose.iloc[-13] - 1) * 100 if len(mclose) >= 13 else np.nan

            # Advanced metrics
            alpha, beta = beta_alpha(mrets, bench_mrets)
            upcap, downcap = up_down_capture(mrets, bench_mrets)
            treynor = treynor_ratio(mrets, rf_annual, beta)
            info_r = information_ratio(mrets, bench_mrets)
            skew = mrets.skew() if len(mrets) >= 12 else np.nan
            kurt = mrets.kurtosis() if len(mrets) >= 12 else np.nan

            # Technical indicators
            rsi_m = rsi(mclose, 14)
            adx_m = adx(mhigh, mlow, mclose, 14)

            # 10-month MA signal
            ma_10 = mclose.rolling(10).mean()
            above_10mo = 1 if len(ma_10) > 0 and mclose.iloc[-1] > ma_10.iloc[-1] else 0
            pct_above_10mo = (mclose.iloc[-1] / ma_10.iloc[-1] - 1) * 100 if len(ma_10) > 0 and ma_10.iloc[-1] != 0 else np.nan

            # Fundamentals from CSV
            fund_row = fund_df[fund_df['symbol'] == t]
            expense_pct = fund_row['expense_%'].values[0] if len(fund_row) > 0 else universe.get(t, {}).get("expense_%", np.nan)
            aum_m = fund_row['aum_m'].values[0] if len(fund_row) > 0 else universe.get(t, {}).get("aum_m", np.nan)
            div_yield_pct = fund_row['div_yield_%'].values[0] if len(fund_row) > 0 else universe.get(t, {}).get("div_yield_%", np.nan)
            bidask_pct = fund_row['bidask_proxy_%'].values[0] if len(fund_row) > 0 else 0.1

            # Liquidity (60d avg dollar volume, approximate from CSV)
            ddv = universe.get(t, {}).get("avg_dollar_vol", np.nan)

            rows.append({
                "symbol": t,
                "name": universe.get(t, {}).get("name", ""),
                "category": universe.get(t, {}).get("category", ""),
                "last_m_close": float(mclose.iloc[-1]),
                "ret_1m_%": float(ret_1m),
                "ret_3m_%": float(ret_3m),
                "ret_6m_%": float(ret_6m),
                "ret_12m_%": float(ret_12m),
                "mom_12_1_%": float(mom_12_1),
                "CAGR_5y_%": float(cagr_5y),
                "Vol_ann_%": float(vol_ann),
                "Sharpe": float(sharpe),
                "Sortino": float(sortino),
                "MaxDD_%": float(max_dd),
                "Alpha_%": float(alpha),
                "Beta": float(beta),
                "UpCapture_%": float(upcap),
                "DownCapture_%": float(downcap),
                "Treynor": float(treynor),
                "Info_Ratio": float(info_r),
                "Skewness": float(skew),
                "Kurtosis": float(kurt),
                "RSI14_monthly": float(rsi_m),
                "ADX14_monthly": float(adx_m),
                "Above_10MO": int(above_10mo),
                "%_Above_10MO": float(pct_above_10mo),
                "Expense_%": float(expense_pct),
                "AUM_M": float(aum_m),
                "Div_Yield_%": float(div_yield_pct),
                "BidAsk_Proxy_%": float(bidask_pct),
                "AvgDollarVol_60d": float(ddv)
            })
        except Exception as e:
            logging.error(f"{t} computation failed: {e}")
            continue

    M = pd.DataFrame(rows)
    if not M.empty:
        M = M.sort_values("symbol")
    M.to_csv("etf_monthly_metrics.csv", index=False)
    logging.info(f"Computed metrics for {len(M)} ETFs")

    # Correlation matrix
    corr_pairwise = pd.DataFrame()
    if rets_dict and not M.empty:
        R = pd.DataFrame(rets_dict).dropna(how="all")
        corr_pairwise = R.corr(min_periods=12)
        corr_pairwise.to_csv("etf_pairwise_corr.csv")
        if BENCH in corr_pairwise.columns:
            corr_pairwise[[BENCH]].rename(columns={BENCH: "Corr_vs_SPY"}).to_csv("etf_corr_vs_SPY.csv")

    # Append Corr_vs_SPY to metrics
    if not M.empty and not corr_pairwise.empty and BENCH in corr_pairwise.columns:
        corr_vs_spy = corr_pairwise[BENCH].rename("Corr_vs_SPY").reset_index().rename(columns={"index": "symbol"})
        M = M.merge(corr_vs_spy, on="symbol", how="left")
    elif not M.empty:
        M["Corr_vs_SPY"] = np.nan

    # Re-save metrics with Corr_vs_SPY
    M.to_csv("etf_monthly_metrics.csv", index=False)

    # Build final dataset merging raw CSV fundamentals (full) if available
    final_df = M.copy()
    try:
        if csv_path and os.path.exists(csv_path):
            raw_csv = pd.read_csv(csv_path)
            if 'Symbol' in raw_csv.columns:
                raw_csv['Symbol'] = raw_csv['Symbol'].astype(str).str.replace('.', '-', regex=False)
            if not final_df.empty:
                final_df = final_df.merge(raw_csv, left_on='symbol', right_on='Symbol', how='left')
            else:
                final_df = raw_csv
    except Exception as e:
        logging.warning(f"Final dataset merge with CSV failed: {e}")
    final_df.to_csv("etf_final_dataset.csv", index=False)

    # Plots for top ranked
    os.makedirs("plots", exist_ok=True)
    if not M.empty and "symbol" in M.columns:
        for t in M["symbol"].head(20):
            try:
                sub = D[t]["Adj Close"].dropna()
                mclose = monthly_resample(sub)
                plt.plot(mclose.index, mclose.values, label=t)
                plt.title(f"{t} Monthly Adj Close")
                plt.xlabel("Date")
                plt.ylabel("Price")
                plt.legend()
                plt.savefig(f"plots/{t}_perf.png")
                plt.close()
            except:
                continue

    return M, corr_pairwise

# ---------------------------- Shortlists & Ranking ---------------------------
def composite_rank(M: pd.DataFrame) -> pd.DataFrame:
    """Create composite ranking"""
    if M.empty:
        return M
    M = M.copy()
    Z = {}
    for f in ["mom_12_1_%", "ret_6m_%", "Sharpe", "Sortino", "Treynor", "UpCapture_%", "MaxDD_%", "%_Above_10MO", "Skewness", "RSI14_monthly"]:
        M[f] = winsorize(M[f].astype(float).replace([np.inf, -np.inf], np.nan))
    Z["MOM12_1"] = zscore(M["mom_12_1_%"].fillna(0))
    Z["MOM6"] = zscore(M["ret_6m_%"].fillna(0))
    Z["SHARPE"] = zscore(M["Sharpe"].fillna(0))
    Z["SORTINO"] = zscore(M["Sortino"].fillna(0))
    Z["TREYNOR"] = zscore(M["Treynor"].fillna(0))
    Z["UPCAP"] = zscore(M["UpCapture_%"].fillna(0))
    Z["MAXDD"] = zscore(-M["MaxDD_%"].fillna(0))
    Z["10MO"] = zscore(M["%_Above_10MO"].fillna(0))
    Z["SKEW"] = zscore(M["Skewness"].fillna(0))
    Z["RSI"] = zscore(-(M["RSI14_monthly"] - 50).abs().fillna(0))

    score = (
        W_MOM12_1 * Z["MOM12_1"] +
        W_MOM6 * Z["MOM6"] +
        W_SHARPE * Z["SHARPE"] +
        W_SORTINO * Z["SORTINO"] +
        W_TREYNOR * Z["TREYNOR"] +
        W_UPCAP * Z["UPCAP"] +
        W_MAXDD * Z["MAXDD"] +
        W_10MO * Z["10MO"] +
        W_SKEW * Z["SKEW"] +
        W_RSI * Z["RSI"]
    )
    M["score"] = score
    M = M.sort_values("score", ascending=False).head(TOPN_RANKED)
    M.to_csv("etf_ranked.csv", index=False)
    return M

def shortlists(M: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create various shortlists"""
    if M.empty:
        return M, M, M
    leaders = M[
        (nz(M["mom_12_1_%"], -1e9) > 5) &
        (nz(M["Sharpe"], -1e9) > 0.5) &
        (nz(M["Above_10MO"], 0) == 1)
    ].sort_values(["mom_12_1_%", "Sharpe"], ascending=False).head(TOPN_LIST)
    leaders.to_csv("shortlist_momo_leaders.csv", index=False)

    pullbacks = M[
        (nz(M["ret_1m_%"], 1e9) < -2) &
        (nz(M["ret_6m_%"], -1e9) > 0) &
        (nz(M["Above_10MO"], 0) == 1)
    ].sort_values(["ret_1m_%", "mom_12_1_%"], ascending=[True, False]).head(TOPN_LIST)
    pullbacks.to_csv("shortlist_pullbacks_above_10mo.csv", index=False)

    low_vol = M[
        (nz(M["Vol_ann_%"], 1e9) < M["Vol_ann_%"].median()) &
        (nz(M["MaxDD_%"], 1e9) < -10) &
        (nz(M["Sharpe"], -1e9) > 0)
    ].sort_values(["Vol_ann_%", "Sharpe"], ascending=[True, False]).head(TOPN_LIST)
    low_vol.to_csv("shortlist_lowvol_defensive.csv", index=False)

    return leaders, pullbacks, low_vol

def shortlist_bonds(M: pd.DataFrame) -> pd.DataFrame:
    """Create bonds shortlist"""
    if M.empty:
        return M
    bonds = M[M["category"].str.contains("bond|tips|fixed|municipal", case=False, na=False)]
    high_yield_low_vol = bonds[
        (nz(bonds["Div_Yield_%"], -1e9) > 3) &
        (nz(bonds["Vol_ann_%"], 1e9) < 15) &
        (nz(bonds["Expense_%"], 1e9) < 0.5)
    ].sort_values(["Div_Yield_%", "Vol_ann_%"], ascending=[False, True]).head(TOPN_LIST)
    high_yield_low_vol.to_csv("shortlist_bonds_high_yield_low_vol.csv", index=False)
    return high_yield_low_vol

# ------------------------------- CLI -----------------------------------------
def main():
    global RISK_FREE, MIN_AUM_M
    ap = argparse.ArgumentParser(description="ETF Monthly Scanner")
    ap.add_argument("--csv", help="CSV file with ETF universe (etfdb_screener.csv)", default=None)
    ap.add_argument("--rf", type=float, default=0.0, help="Risk-free rate")
    ap.add_argument("--min-aum", type=float, default=100, help="Min AUM in millions")
    args = ap.parse_args()

    RISK_FREE = args.rf
    MIN_AUM_M = args.min_aum

    print("=" * 70)
    print("ETF Monthly Scanner - Starting Analysis")
    print("=" * 70)

    try:
        csv_path = resolve_csv_path(args.csv)
        print(f"Building universe from CSV: {csv_path or 'default list'}")
        universe = build_universe(csv_path)
        print(f"Universe size: {len(universe)} ETFs")
        M, corr = compute_all(universe, RISK_FREE, MIN_AUM_M, csv_path)
        if M.empty:
            print("No data available for analysis")
            return 1

        print(f"Analyzed {len(M)} ETFs successfully")
        R = composite_rank(M)
        L, P, LV = shortlists(M)
        B = shortlist_bonds(M)

        print("\n" + "=" * 70)
        print("TOP 12 ETFs BY COMPOSITE SCORE")
        print("=" * 70)
        display_cols = ["symbol", "name", "category", "score", "mom_12_1_%", "ret_6m_%", "Sharpe", "Treynor", "Expense_%", "Skewness", "RSI14_monthly"]
        available_cols = [col for col in display_cols if col in R.columns]
        print(R[available_cols].head(12).to_string(index=False, float_format="%.2f"))

        print(f"\nFiles created:")
        for f in ["etf_universe_used.csv", "etf_fundamentals.csv", "etf_monthly_metrics.csv", "etf_final_dataset.csv", "etf_ranked.csv",
                  "shortlist_momo_leaders.csv", "shortlist_pullbacks_above_10mo.csv", "shortlist_lowvol_defensive.csv",
                  "etf_pairwise_corr.csv", "etf_corr_vs_SPY.csv"]:
            if os.path.exists(f):
                print(f"- {f} ({pd.read_csv(f).shape[0]} rows)")
        if not B.empty:
            print(f"- shortlist_bonds_high_yield_low_vol.csv ({len(B)} ETFs)")
        print("\nAnalysis complete!")
        return 0
    except Exception as e:
        logging.error(f"Analysis failed: {e}")
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())