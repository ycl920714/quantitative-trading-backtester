"""
Vectorised backtest used by the web app.

Execution rule (same as the event-driven engine in engine.py):
    signal computed at the close of day t  ->  trade at the open of day t+1

So on day t the position held overnight (from the close of t-1 to the open
of t) is the OLD position, and the position held during the day (open of t
to close of t) is the NEW one:

    gap_t      = Open_t / Close_{t-1} - 1
    intraday_t = Close_t / Open_t - 1
    r_t        = (1 + p_old * gap_t) * (1 + p_new * intraday_t) - 1 - cost * |p_new - p_old|

where p_new = signal_{t-1} and p_old = signal_{t-2}.
Transaction cost is charged on every unit of position change (entering a
long from flat costs 1 unit, flipping from long to short costs 2).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def positions_from_signal(signal: pd.Series, allow_short: bool = False) -> pd.Series:
    """Clip the raw signal to the allowed position set."""
    s = signal.fillna(0).astype(float)
    return s if allow_short else s.clip(lower=0)


def run_vectorised(
    df: pd.DataFrame,
    initial_capital: float = 10_000.0,
    cost_bps: float = 5.0,
    allow_short: bool = False,
) -> pd.DataFrame:
    """Backtest a DataFrame that already has a 'signal' column.

    Returns a copy with columns:
        position          position held during day t (after the open)
        returns           close-to-close return of the asset
        strategy_returns  net daily return of the strategy
        strategy_equity   equity curve of the strategy
        buyhold_equity    equity curve of buying at the first close and holding
    """
    o = df.copy()
    sig = positions_from_signal(o["signal"], allow_short)

    p_new = sig.shift(1).fillna(0)
    p_old = sig.shift(2).fillna(0)

    prev_close = o["Close"].shift(1)
    gap = (o["Open"] / prev_close - 1).fillna(0)
    intraday = (o["Close"] / o["Open"] - 1).fillna(0)
    cost = (p_new - p_old).abs() * cost_bps / 10_000.0

    o["position"] = p_new
    o["returns"] = o["Close"].pct_change().fillna(0)
    o["trade_cost"] = cost
    o["strategy_returns"] = (1 + p_old * gap) * (1 + p_new * intraday) - 1 - cost
    o["strategy_equity"] = initial_capital * (1 + o["strategy_returns"]).cumprod()
    o["buyhold_equity"] = initial_capital * (1 + o["returns"]).cumprod()
    return o


def compute_metrics(returns: pd.Series, equity: pd.Series, rf: float = 0.0, ann: int = 252,
                    position: pd.Series | None = None) -> dict:
    """Performance metrics from a daily return series.

    rf is an annual risk-free rate. Sharpe and Sortino use excess returns.
    Sortino uses downside deviation: sqrt(mean(min(r - rf_daily, 0)^2)).
    """
    r = returns.dropna()
    keys = ["Total Return", "CAGR", "Annual Volatility", "Sharpe Ratio", "Sortino Ratio",
            "Max Drawdown", "Calmar Ratio", "Time in Market"]
    if len(r) < 2:
        return {k: np.nan for k in keys}

    growth = (1 + r).prod()
    years = len(r) / ann
    cagr = growth ** (1 / years) - 1 if years > 0 else np.nan
    vol = r.std() * np.sqrt(ann)

    rf_d = (1 + rf) ** (1 / ann) - 1
    ex = r - rf_d
    sharpe = ex.mean() / r.std() * np.sqrt(ann) if r.std() > 0 else np.nan
    dd_dev = np.sqrt((np.minimum(ex, 0) ** 2).mean()) * np.sqrt(ann)
    sortino = ex.mean() * ann / dd_dev if dd_dev > 0 else np.nan

    eq = equity.loc[r.index]
    mdd = (eq / eq.cummax() - 1).min()
    calmar = cagr / abs(mdd) if mdd < 0 else np.nan

    return {"Total Return": growth - 1, "CAGR": cagr, "Annual Volatility": vol,
            "Sharpe Ratio": sharpe, "Sortino Ratio": sortino, "Max Drawdown": mdd,
            "Calmar Ratio": calmar,
            "Time in Market": (position.loc[r.index] != 0).mean() if position is not None else 1.0}


def var_cvar(returns: pd.Series, confidence: float = 0.95) -> tuple[float, float]:
    """Historical one-day VaR and CVaR (expected shortfall), as returns."""
    r = returns.dropna()
    if len(r) < 10:
        return np.nan, np.nan
    var = np.percentile(r, (1 - confidence) * 100)
    return var, r[r <= var].mean()
