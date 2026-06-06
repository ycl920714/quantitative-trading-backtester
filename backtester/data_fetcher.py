"""
Data fetching module — downloads OHLCV price data for US stocks via yfinance.
"""
import pandas as pd
import yfinance as yf
from typing import List, Optional


def fetch_stock_data(
    ticker: str,
    start: str,
    end: str,
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Download adjusted OHLCV data for a single ticker.

    Returns a DataFrame with columns: Open, High, Low, Close, Volume, Adj Close.
    Raises ValueError if no data is returned.
    """
    df = yf.download(ticker, start=start, end=end, interval=interval,
                     auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {ticker} ({start} → {end})")
    df.index = pd.to_datetime(df.index)
    # Flatten MultiIndex columns if present (yfinance ≥ 0.2.x)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.dropna(inplace=True)
    return df


def fetch_multiple(
    tickers: List[str],
    start: str,
    end: str,
    interval: str = "1d",
) -> dict[str, pd.DataFrame]:
    """Download data for multiple tickers; returns {ticker: DataFrame}."""
    result = {}
    for t in tickers:
        try:
            result[t] = fetch_stock_data(t, start, end, interval)
        except ValueError as e:
            print(f"[WARNING] {e}")
    return result


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Attach common technical indicators to a price DataFrame (in-place copy)."""
    df = df.copy()
    close = df["Close"]

    # Moving averages
    for w in (10, 20, 50, 200):
        df[f"SMA_{w}"] = close.rolling(w).mean()
        df[f"EMA_{w}"] = close.ewm(span=w, adjust=False).mean()

    # Bollinger Bands (20-day, 2σ)
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df["BB_upper"] = bb_mid + 2 * bb_std
    df["BB_mid"]   = bb_mid
    df["BB_lower"] = bb_mid - 2 * bb_std

    # RSI-14
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs   = gain / loss.replace(0, float("nan"))
    df["RSI_14"] = 100 - (100 / (1 + rs))

    # MACD (12/26/9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"]        = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"]   = df["MACD"] - df["MACD_signal"]

    # Average True Range (14-day)
    high, low = df["High"], df["Low"]
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    df["ATR_14"] = tr.rolling(14).mean()

    # Volume SMA
    df["Vol_SMA_20"] = df["Volume"].rolling(20).mean()

    return df
