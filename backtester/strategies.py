"""
Trading strategies.

Each strategy implements `generate_signals(df)`, which adds a 'signal' column:
+1 = want to be long, -1 = want to be short, 0 = want to be flat.

Timing convention
-----------------
A signal on day t only uses information available at the close of day t.
The backtest engines never trade on the same bar: a signal from day t is
executed at the OPEN of day t+1. Strategies therefore do not shift their
own signals; the engines do it in one place so the rule is applied
consistently.

`params` on each class describes the tunable parameters for the web app:
    name -> (label, min, max, default)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd


def _hold_until_opposite(entry_long: pd.Series, entry_short: pd.Series) -> pd.Series:
    """Turn one-day entry events into a held position.

    Once a long (short) condition fires, the position is kept until the
    opposite condition fires. Before the first event the position is flat.
    """
    s = pd.Series(np.nan, index=entry_long.index)
    s[entry_long.fillna(False).astype(bool)] = 1
    s[entry_short.fillna(False).astype(bool)] = -1
    return s.ffill().fillna(0)


class BaseStrategy(ABC):
    name: str = "base"
    style: str = ""
    params: dict[str, tuple] = {}

    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return df with a 'signal' column added."""


# 1. SMA crossover ------------------------------------------------------------

class SMACrossover(BaseStrategy):
    """Long when the fast simple moving average is above the slow one."""
    name = "SMA Crossover"
    style = "Trend following"
    params = {"fast": ("Fast MA (days)", 5, 100, 50),
              "slow": ("Slow MA (days)", 20, 250, 200)}

    def __init__(self, fast: int = 50, slow: int = 200) -> None:
        self.fast, self.slow = fast, slow

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df["SMA_fast"] = df["Close"].rolling(self.fast).mean()
        df["SMA_slow"] = df["Close"].rolling(self.slow).mean()
        df["signal"] = 0
        df.loc[df["SMA_fast"] > df["SMA_slow"], "signal"] = 1
        df.loc[df["SMA_fast"] < df["SMA_slow"], "signal"] = -1
        return df


# 2. EMA crossover ------------------------------------------------------------

class EMACrossover(BaseStrategy):
    """Long when the fast exponential moving average is above the slow one."""
    name = "EMA Crossover"
    style = "Trend following"
    params = {"fast": ("Fast EMA (days)", 5, 50, 12),
              "slow": ("Slow EMA (days)", 10, 200, 26)}

    def __init__(self, fast: int = 12, slow: int = 26) -> None:
        self.fast, self.slow = fast, slow

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df["EMA_fast"] = df["Close"].ewm(span=self.fast, adjust=False).mean()
        df["EMA_slow"] = df["Close"].ewm(span=self.slow, adjust=False).mean()
        df["signal"] = 0
        df.loc[df["EMA_fast"] > df["EMA_slow"], "signal"] = 1
        df.loc[df["EMA_fast"] < df["EMA_slow"], "signal"] = -1
        return df


# 3. RSI mean reversion -------------------------------------------------------

class RSIMeanReversion(BaseStrategy):
    """Buy when RSI falls below the oversold level and hold until it rises
    above the overbought level (then short, if shorting is allowed)."""
    name = "RSI Mean Reversion"
    style = "Mean reversion"
    params = {"period": ("RSI period (days)", 5, 30, 14),
              "oversold": ("Oversold level", 10, 40, 30),
              "overbought": ("Overbought level", 60, 90, 70)}

    def __init__(self, period: int = 14, oversold: float = 30, overbought: float = 70) -> None:
        self.period, self.oversold, self.overbought = period, oversold, overbought

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        delta = df["Close"].diff()
        gain = delta.clip(lower=0).rolling(self.period).mean()
        loss = (-delta.clip(upper=0)).rolling(self.period).mean()
        rs = gain / loss.replace(0, np.nan)
        df["RSI"] = 100 - (100 / (1 + rs))
        df["signal"] = _hold_until_opposite(df["RSI"] < self.oversold,
                                            df["RSI"] > self.overbought)
        return df


# 4. MACD ---------------------------------------------------------------------

class MACDStrategy(BaseStrategy):
    """Long when the MACD histogram is positive, short/flat when negative."""
    name = "MACD"
    style = "Trend following"
    params = {"fast": ("Fast EMA (days)", 5, 30, 12),
              "slow": ("Slow EMA (days)", 15, 60, 26),
              "signal": ("Signal line (days)", 3, 20, 9)}

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9) -> None:
        self.fast, self.slow, self.signal = fast, slow, signal

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        ema_fast = df["Close"].ewm(span=self.fast, adjust=False).mean()
        ema_slow = df["Close"].ewm(span=self.slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        df["MACD_hist"] = macd - macd.ewm(span=self.signal, adjust=False).mean()
        df["signal"] = 0
        df.loc[df["MACD_hist"] > 0, "signal"] = 1
        df.loc[df["MACD_hist"] < 0, "signal"] = -1
        return df


# 5. Bollinger Band breakout --------------------------------------------------

class BollingerBandBreakout(BaseStrategy):
    """Go long after a close above the upper band and hold until a close
    below the lower band."""
    name = "Bollinger Band Breakout"
    style = "Momentum"
    params = {"window": ("Window (days)", 10, 60, 20),
              "num_std": ("Band width (std)", 1.0, 3.0, 2.0)}

    def __init__(self, window: int = 20, num_std: float = 2.0) -> None:
        self.window, self.num_std = window, num_std

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        mid = df["Close"].rolling(self.window).mean()
        std = df["Close"].rolling(self.window).std()
        df["BB_upper"] = mid + self.num_std * std
        df["BB_lower"] = mid - self.num_std * std
        df["BB_mid"] = mid
        df["signal"] = _hold_until_opposite(df["Close"] > df["BB_upper"],
                                            df["Close"] < df["BB_lower"])
        return df


# 6. Bollinger Band mean reversion --------------------------------------------

class BollingerBandMeanReversion(BaseStrategy):
    """Buy after a close below the lower band; exit when price gets back to
    the moving average. Long only by design."""
    name = "Bollinger Band Mean Reversion"
    style = "Mean reversion"
    params = {"window": ("Window (days)", 10, 60, 20),
              "num_std": ("Band width (std)", 1.0, 3.0, 2.0)}

    def __init__(self, window: int = 20, num_std: float = 2.0) -> None:
        self.window, self.num_std = window, num_std

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        mid = df["Close"].rolling(self.window).mean()
        std = df["Close"].rolling(self.window).std()
        lower = mid - self.num_std * std
        close = df["Close"].to_numpy()
        lo, md = lower.to_numpy(), mid.to_numpy()

        signal = np.zeros(len(df))
        in_trade = False
        for i in range(len(df)):
            if not in_trade and close[i] < lo[i]:
                in_trade = True
            elif in_trade and close[i] >= md[i]:
                in_trade = False
            signal[i] = 1 if in_trade else 0

        df["signal"] = signal.astype(int)
        df["BB_upper"] = mid + self.num_std * std
        df["BB_lower"] = lower
        df["BB_mid"] = mid
        return df


# 7. Time-series momentum -----------------------------------------------------

class MomentumStrategy(BaseStrategy):
    """Long when the trailing N-day return is positive (time-series momentum)."""
    name = "Momentum (ROC)"
    style = "Momentum"
    params = {"lookback": ("Lookback (days)", 10, 250, 60)}

    def __init__(self, lookback: int = 60) -> None:
        self.lookback = lookback

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df["ROC"] = df["Close"].pct_change(self.lookback)
        df["signal"] = 0
        df.loc[df["ROC"] > 0, "signal"] = 1
        df.loc[df["ROC"] < 0, "signal"] = -1
        return df


# 8. Dual Thrust --------------------------------------------------------------

class DualThrust(BaseStrategy):
    """Volatility breakout adapted to daily bars.

    The range is built from the previous N days only (so today's high and
    low are not used to set today's trigger). A close above
    open + k * range opens a long; a close below open - k * range closes it
    (or opens a short). The position is held between triggers.
    """
    name = "Dual Thrust"
    style = "Volatility breakout"
    params = {"lookback": ("Lookback (days)", 2, 20, 5),
              "k": ("Trigger width k", 0.1, 1.0, 0.5)}

    def __init__(self, lookback: int = 5, k: float = 0.5) -> None:
        self.lookback, self.k = lookback, k

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        prev = df.shift(1)
        hh = prev["High"].rolling(self.lookback).max()
        lc = prev["Close"].rolling(self.lookback).min()
        hc = prev["Close"].rolling(self.lookback).max()
        ll = prev["Low"].rolling(self.lookback).min()
        rng = pd.concat([hh - lc, hc - ll], axis=1).max(axis=1)
        df["DT_upper"] = df["Open"] + self.k * rng
        df["DT_lower"] = df["Open"] - self.k * rng
        df["signal"] = _hold_until_opposite(df["Close"] > df["DT_upper"],
                                            df["Close"] < df["DT_lower"])
        return df


# 9. Buy and hold benchmark ---------------------------------------------------

class BuyAndHold(BaseStrategy):
    """Always long. Used as the benchmark."""
    name = "Buy & Hold"
    style = "Benchmark"
    params = {}

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df["signal"] = 1
        return df


STRATEGY_REGISTRY: dict[str, type[BaseStrategy]] = {
    "sma_crossover":      SMACrossover,
    "ema_crossover":      EMACrossover,
    "rsi_mean_reversion": RSIMeanReversion,
    "macd":               MACDStrategy,
    "bb_breakout":        BollingerBandBreakout,
    "bb_mean_reversion":  BollingerBandMeanReversion,
    "momentum":           MomentumStrategy,
    "dual_thrust":        DualThrust,
    "buy_and_hold":       BuyAndHold,
}
