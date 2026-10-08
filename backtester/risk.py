"""
Value at Risk backtesting.

A VaR model is only useful if its forecasts are right about as often as
they claim. This module produces one-day VaR forecasts out of sample and
checks them with the standard tests used in bank risk management.

Forecasts
---------
The VaR for day t is estimated only from returns up to day t-1, then
compared with the actual return on day t. Three methods:

    historical   empirical quantile of the last `window` returns
    normal       mean - z * std of the last `window` returns
    ewma         RiskMetrics: z * sigma_t, where
                 sigma_t^2 = lam * sigma_{t-1}^2 + (1 - lam) * r_{t-1}^2

VaR is reported as a negative return (e.g. -0.031 = a 3.1% loss). An
"exception" is a day when the actual return is below the VaR forecast.

Tests
-----
Kupiec (1995) proportion-of-failures test
    H0: the exception rate equals 1 - confidence.
Christoffersen (1998) independence test
    H0: an exception today does not make one tomorrow more likely.
    Rejection means exceptions cluster, i.e. the model reacts too slowly
    when volatility rises.
Basel traffic light (99% VaR, last 250 days)
    0-4 exceptions green, 5-9 yellow, 10 or more red.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

METHODS = {
    "historical": "Historical simulation",
    "normal": "Normal (parametric)",
    "ewma": "EWMA (RiskMetrics, lambda 0.94)",
}


def var_forecasts(returns: pd.Series, method: str = "historical", confidence: float = 0.99,
                  window: int = 250, lam: float = 0.94) -> pd.Series:
    """One-day-ahead VaR forecasts. Value at t uses returns up to t-1 only."""
    r = returns.dropna()
    alpha = 1 - confidence
    z = stats.norm.ppf(alpha)  # negative number, e.g. -2.33 at 99%

    if method == "historical":
        var = r.rolling(window).quantile(alpha)
    elif method == "normal":
        var = r.rolling(window).mean() + z * r.rolling(window).std()
    elif method == "ewma":
        sigma2 = np.empty(len(r))
        sigma2[0] = r.iloc[:window].var()
        rv = r.to_numpy()
        for t in range(1, len(r)):
            sigma2[t] = lam * sigma2[t - 1] + (1 - lam) * rv[t - 1] ** 2
        # sigma2[t] already only uses returns up to t-1, so no extra shift below
        var = pd.Series(z * np.sqrt(sigma2), index=r.index)
        var.iloc[:window] = np.nan
        return var.rename("VaR")
    else:
        raise ValueError(f"Unknown method {method}")

    # the rolling estimate at t includes r_t, so shift to make it a forecast for t+1
    return var.shift(1).rename("VaR")


def kupiec_pof(exceptions: int, n: int, p: float) -> tuple[float, float]:
    """Kupiec proportion-of-failures likelihood ratio and its p-value (chi2, 1 df)."""
    if n == 0:
        return np.nan, np.nan
    x = exceptions
    phat = x / n
    def ll(prob):
        prob = min(max(prob, 1e-12), 1 - 1e-12)
        return (n - x) * np.log(1 - prob) + x * np.log(prob)
    lr = -2 * (ll(p) - ll(phat))
    return lr, 1 - stats.chi2.cdf(lr, 1)


def christoffersen_independence(hits: np.ndarray) -> tuple[float, float]:
    """Christoffersen independence likelihood ratio and its p-value (chi2, 1 df)."""
    h = np.asarray(hits, dtype=int)
    prev, curr = h[:-1], h[1:]
    n00 = np.sum((prev == 0) & (curr == 0)); n01 = np.sum((prev == 0) & (curr == 1))
    n10 = np.sum((prev == 1) & (curr == 0)); n11 = np.sum((prev == 1) & (curr == 1))
    if n01 + n11 == 0 or (n00 + n01) == 0 or (n10 + n11) == 0:
        return np.nan, np.nan
    pi0 = n01 / (n00 + n01)
    pi1 = n11 / (n10 + n11)
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    def xlogy(a, b):
        return a * np.log(b) if a > 0 else 0.0
    ll_null = xlogy(n00 + n10, 1 - pi) + xlogy(n01 + n11, pi)
    ll_alt = xlogy(n00, 1 - pi0) + xlogy(n01, pi0) + xlogy(n10, 1 - pi1) + xlogy(n11, pi1)
    lr = -2 * (ll_null - ll_alt)
    return lr, 1 - stats.chi2.cdf(lr, 1)


def basel_zone(exceptions_250: int) -> str:
    """Basel traffic light for 99% VaR over 250 trading days."""
    if exceptions_250 <= 4:
        return "Green"
    if exceptions_250 <= 9:
        return "Yellow"
    return "Red"


def backtest_var(returns: pd.Series, method: str = "historical", confidence: float = 0.99,
                 window: int = 250) -> tuple[pd.DataFrame, dict]:
    """Run forecasts and all tests. Returns (daily frame, summary dict)."""
    r = returns.dropna()
    var = var_forecasts(r, method, confidence, window)
    df = pd.DataFrame({"return": r, "VaR": var}).dropna()
    df["exception"] = df["return"] < df["VaR"]

    n = len(df)
    x = int(df["exception"].sum())
    p = 1 - confidence
    lr_pof, p_pof = kupiec_pof(x, n, p)
    lr_ind, p_ind = christoffersen_independence(df["exception"].to_numpy())

    rolling_250 = df["exception"].rolling(250).sum()
    last_250 = int(df["exception"].iloc[-250:].sum()) if n >= 250 else np.nan
    worst_250 = int(rolling_250.max()) if n >= 250 else np.nan
    worst_end = rolling_250.idxmax() if n >= 250 else None

    summary = {
        "Method": METHODS[method],
        "Days tested": n,
        "Expected exceptions": round(n * p, 1),
        "Actual exceptions": x,
        "Exception rate": x / n if n else np.nan,
        "Kupiec p-value": p_pof,
        "Independence p-value": p_ind,
        "Exceptions, last 250 days": last_250,
        "Basel zone, last 250 days": basel_zone(last_250) if n >= 250 and confidence == 0.99 else "n/a",
        "Worst 250-day window": worst_250,
        "Worst window ends": worst_end.date() if worst_end is not None else None,
    }
    return df, summary
