"""
Built-in quantitative trading strategies.

Each strategy subclasses BaseStrategy and implements `generate_signals`,
which returns the input DataFrame with an added 'signal' column
(+1 = long, -1 = short, 0 = flat/exit).
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Any


# ─── Base ─────────────────────────────────────────────────────────────────────

class BaseStrategy(ABC):
    name: str = "base"

    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return df with a 'signal' column added."""


# ─── 1. Dual Moving-Average Crossover ─────────────────────────────────────────

class SMACrossover(BaseStrategy):
    """
    Buy when the fast SMA crosses above the slow SMA; sell on the reverse.

    Parameters
    ----------
    fast : int  (default 50)
    slow : int  (default 200)
    """
    name = "SMA Crossover"

    def __init__(self, fast: int = 50, slow: int = 200) -> None:
        self.fast = fast
        self.slow = slow

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df[f"SMA_fast"] = df["Close"].rolling(self.fast).mean()
        df[f"SMA_slow"] = df["Close"].rolling(self.slow).mean()

        df["signal"] = 0
        df.loc[df["SMA_fast"] > df["SMA_slow"], "signal"] = 1
        df.loc[df["SMA_fast"] < df["SMA_slow"], "signal"] = -1
        return df


# ─── 2. EMA Crossover ─────────────────────────────────────────────────────────

class EMACrossover(BaseStrategy):
    """
    Buy on EMA-fast > EMA-slow; sell on the reverse.

    Parameters
    ----------
    fast : int  (default 12)
    slow : int  (default 26)
    """
    name = "EMA Crossover"

    def __init__(self, fast: int = 12, slow: int = 26) -> None:
        self.fast = fast
        self.slow = slow

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df["EMA_fast"] = df["Close"].ewm(span=self.fast, adjust=False).mean()
        df["EMA_slow"] = df["Close"].ewm(span=self.slow, adjust=False).mean()

        df["signal"] = 0
        df.loc[df["EMA_fast"] > df["EMA_slow"], "signal"] = 1
        df.loc[df["EMA_fast"] < df["EMA_slow"], "signal"] = -1
        return df


# ─── 3. RSI Mean-Reversion ────────────────────────────────────────────────────

class RSIMeanReversion(BaseStrategy):
    """
    Buy when RSI crosses below oversold; sell when RSI crosses above overbought.

    Parameters
    ----------
    period     : int   (default 14)
    oversold   : float (default 30)
    overbought : float (default 70)
    """
    name = "RSI Mean Reversion"

    def __init__(self, period: int = 14, oversold: float = 30, overbought: float = 70) -> None:
        self.period     = period
        self.oversold   = oversold
        self.overbought = overbought

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        delta = df["Close"].diff()
        gain  = delta.clip(lower=0).rolling(self.period).mean()
        loss  = (-delta.clip(upper=0)).rolling(self.period).mean()
        rs    = gain / loss.replace(0, np.nan)
        df["RSI"] = 100 - (100 / (1 + rs))

        signal = pd.Series(0, index=df.index)
        signal[df["RSI"] < self.oversold]   = 1   # oversold → buy
        signal[df["RSI"] > self.overbought] = -1  # overbought → sell/short
        df["signal"] = signal
        return df


# ─── 4. MACD Strategy ─────────────────────────────────────────────────────────

class MACDStrategy(BaseStrategy):
    """
    Buy when MACD histogram turns positive; sell when it turns negative.

    Parameters
    ----------
    fast   : int (default 12)
    slow   : int (default 26)
    signal : int (default 9)
    """
    name = "MACD"

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9) -> None:
        self.fast   = fast
        self.slow   = slow
        self.signal = signal

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        ema_fast = df["Close"].ewm(span=self.fast,   adjust=False).mean()
        ema_slow = df["Close"].ewm(span=self.slow,   adjust=False).mean()
        macd     = ema_fast - ema_slow
        sig_line = macd.ewm(span=self.signal, adjust=False).mean()
        df["MACD_hist"] = macd - sig_line

        df["signal"] = 0
        df.loc[df["MACD_hist"] > 0, "signal"] = 1
        df.loc[df["MACD_hist"] < 0, "signal"] = -1
        return df


# ─── 5. Bollinger Band Breakout ───────────────────────────────────────────────

class BollingerBandBreakout(BaseStrategy):
    """
    Buy when price closes above the upper band (momentum breakout).
    Sell when price closes below the lower band.

    Parameters
    ----------
    window : int   (default 20)
    num_std: float (default 2.0)
    """
    name = "Bollinger Band Breakout"

    def __init__(self, window: int = 20, num_std: float = 2.0) -> None:
        self.window  = window
        self.num_std = num_std

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        mid  = df["Close"].rolling(self.window).mean()
        std  = df["Close"].rolling(self.window).std()
        df["BB_upper"] = mid + self.num_std * std
        df["BB_lower"] = mid - self.num_std * std

        df["signal"] = 0
        df.loc[df["Close"] > df["BB_upper"], "signal"] = 1
        df.loc[df["Close"] < df["BB_lower"], "signal"] = -1
        return df


# ─── 6. Bollinger Band Mean Reversion ─────────────────────────────────────────

class BollingerBandMeanReversion(BaseStrategy):
    """
    Buy when price closes below the lower band; sell when it crosses the mid.

    Parameters
    ----------
    window : int   (default 20)
    num_std: float (default 2.0)
    """
    name = "Bollinger Band Mean Reversion"

    def __init__(self, window: int = 20, num_std: float = 2.0) -> None:
        self.window  = window
        self.num_std = num_std

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        mid  = df["Close"].rolling(self.window).mean()
        std  = df["Close"].rolling(self.window).std()
        upper = mid + self.num_std * std
        lower = mid - self.num_std * std

        signal = pd.Series(0, index=df.index)
        in_trade = False
        for i in range(len(df)):
            price = df["Close"].iloc[i]
            if not in_trade and price < lower.iloc[i]:
                signal.iloc[i] = 1
                in_trade = True
            elif in_trade and price >= mid.iloc[i]:
                signal.iloc[i] = 0
                in_trade = False
            elif in_trade:
                signal.iloc[i] = 1

        df["signal"] = signal
        df["BB_upper"] = upper
        df["BB_lower"] = lower
        df["BB_mid"]   = mid
        return df


# ─── 7. Momentum / Rate-of-Change ────────────────────────────────────────────

class MomentumStrategy(BaseStrategy):
    """
    Buy the top-performing (highest N-day ROC) if ROC > 0; sell when ROC < 0.

    Parameters
    ----------
    lookback : int (default 20)
    """
    name = "Momentum (ROC)"

    def __init__(self, lookback: int = 20) -> None:
        self.lookback = lookback

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df["ROC"] = df["Close"].pct_change(self.lookback)

        df["signal"] = 0
        df.loc[df["ROC"] > 0, "signal"] = 1
        df.loc[df["ROC"] < 0, "signal"] = -1
        return df


# ─── 8. Dual-Thrust (Volatility Breakout) ────────────────────────────────────

class DualThrust(BaseStrategy):
    """
    Intraday-style volatility breakout adapted to daily bars.

    Each day, compute a range from the prior N bars' high/low and open/close
    extremes. Go long if today's close breaks out above open + k*range.

    Parameters
    ----------
    lookback : int   (default 5)
    k        : float (default 0.5)
    """
    name = "Dual Thrust"

    def __init__(self, lookback: int = 5, k: float = 0.5) -> None:
        self.lookback = lookback
        self.k        = k

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        hh = df["High"].rolling(self.lookback).max()
        lc = df["Close"].rolling(self.lookback).min()
        hc = df["Close"].rolling(self.lookback).max()
        ll = df["Low"].rolling(self.lookback).min()

        rng = pd.concat([hh - lc, hc - ll], axis=1).max(axis=1)
        df["DT_upper"] = df["Open"] + self.k * rng
        df["DT_lower"] = df["Open"] - self.k * rng

        df["signal"] = 0
        df.loc[df["Close"] > df["DT_upper"], "signal"] = 1
        df.loc[df["Close"] < df["DT_lower"], "signal"] = -1
        return df


# ─── 9. Buy-and-Hold benchmark ────────────────────────────────────────────────

class BuyAndHold(BaseStrategy):
    """Always hold long — the simple buy-and-hold benchmark."""
    name = "Buy & Hold"

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df["signal"] = 1
        return df


# ─── Registry ─────────────────────────────────────────────────────────────────

STRATEGY_REGISTRY: dict[str, type[BaseStrategy]] = {
    "sma_crossover":       SMACrossover,
    "ema_crossover":       EMACrossover,
    "rsi_mean_reversion":  RSIMeanReversion,
    "macd":                MACDStrategy,
    "bb_breakout":         BollingerBandBreakout,
    "bb_mean_reversion":   BollingerBandMeanReversion,
    "momentum":            MomentumStrategy,
    "dual_thrust":         DualThrust,
    "buy_and_hold":        BuyAndHold,
}
