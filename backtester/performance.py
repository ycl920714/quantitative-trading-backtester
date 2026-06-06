"""
Performance metrics and report generation for BacktestResult objects.
"""
from __future__ import annotations

import math
import pandas as pd
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .engine import BacktestResult


# ─── Core metrics ─────────────────────────────────────────────────────────────

def compute_metrics(result: "BacktestResult", risk_free_rate: float = 0.04) -> dict:
    """
    Compute a comprehensive set of performance metrics for a BacktestResult.

    Parameters
    ----------
    result          : BacktestResult
    risk_free_rate  : float  Annual risk-free rate (default 4%)

    Returns
    -------
    dict of metric_name → value
    """
    eq  = result.equity_curve
    rets = eq.pct_change().dropna()
    trades = result.trades

    # ── Return metrics ────────────────────────────────────────────────
    total_return   = (eq.iloc[-1] / eq.iloc[0]) - 1
    n_years        = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr           = (1 + total_return) ** (1 / max(n_years, 1e-6)) - 1

    # ── Risk metrics ──────────────────────────────────────────────────
    ann_factor = 252
    ann_vol    = rets.std() * math.sqrt(ann_factor)
    rfr_daily  = (1 + risk_free_rate) ** (1 / ann_factor) - 1
    excess     = rets - rfr_daily
    sharpe     = (excess.mean() / rets.std() * math.sqrt(ann_factor)
                  if rets.std() > 0 else 0.0)

    downside = rets[rets < rfr_daily]
    sortino  = (excess.mean() / downside.std() * math.sqrt(ann_factor)
                if len(downside) > 0 and downside.std() > 0 else 0.0)

    # ── Drawdown ──────────────────────────────────────────────────────
    rolling_max = eq.cummax()
    drawdown    = (eq - rolling_max) / rolling_max
    max_dd      = drawdown.min()

    # Calmar ratio
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0

    # Max drawdown duration (bars)
    in_dd   = drawdown < 0
    dd_start = None
    max_dur  = 0
    for i, val in enumerate(in_dd):
        if val:
            if dd_start is None:
                dd_start = i
            max_dur = max(max_dur, i - dd_start + 1)
        else:
            dd_start = None

    # ── Trade statistics ──────────────────────────────────────────────
    n_trades   = len(trades)
    if n_trades:
        pnls       = [t.pnl for t in trades]
        pnl_pcts   = [t.pnl_pct for t in trades]
        wins       = [p for p in pnls if p > 0]
        losses     = [p for p in pnls if p <= 0]
        win_rate   = len(wins) / n_trades
        avg_win    = np.mean(wins)    if wins   else 0.0
        avg_loss   = np.mean(losses)  if losses else 0.0
        profit_factor = (sum(wins) / abs(sum(losses))
                         if losses and sum(losses) != 0 else float("inf"))
        avg_pnl_pct = np.mean(pnl_pcts)
        best_trade  = max(pnl_pcts)
        worst_trade = min(pnl_pcts)
        total_comm  = sum(t.commission for t in trades)
    else:
        win_rate = avg_win = avg_loss = profit_factor = 0.0
        avg_pnl_pct = best_trade = worst_trade = total_comm = 0.0

    # ── Value at Risk (historical 95 %) ──────────────────────────────
    var_95  = float(np.percentile(rets, 5)) if len(rets) > 0 else 0.0
    cvar_95 = float(rets[rets <= var_95].mean()) if len(rets[rets <= var_95]) > 0 else 0.0

    return {
        # Return
        "Total Return (%)":      round(total_return * 100, 2),
        "CAGR (%)":              round(cagr * 100, 2),
        "Final Equity ($)":      round(eq.iloc[-1], 2),
        # Risk
        "Ann. Volatility (%)":   round(ann_vol * 100, 2),
        "Max Drawdown (%)":      round(max_dd * 100, 2),
        "Max DD Duration (days)":max_dur,
        # Ratios
        "Sharpe Ratio":          round(sharpe, 3),
        "Sortino Ratio":         round(sortino, 3),
        "Calmar Ratio":          round(calmar, 3),
        # Tail risk
        "VaR 95% (daily %)":     round(var_95 * 100, 4),
        "CVaR 95% (daily %)":    round(cvar_95 * 100, 4),
        # Trades
        "# Trades":              n_trades,
        "Win Rate (%)":          round(win_rate * 100, 2),
        "Profit Factor":         round(profit_factor, 3),
        "Avg Win (%)":           round(avg_win / result.initial_capital * 100, 4) if n_trades else 0,
        "Avg Loss (%)":          round(avg_loss / result.initial_capital * 100, 4) if n_trades else 0,
        "Avg Trade PnL (%)":     round(avg_pnl_pct * 100, 4),
        "Best Trade (%)":        round(best_trade * 100, 2),
        "Worst Trade (%)":       round(worst_trade * 100, 2),
        "Total Commission ($)":  round(total_comm, 2),
    }


# ─── Comparison table ─────────────────────────────────────────────────────────

def compare_results(results: list["BacktestResult"], risk_free_rate: float = 0.04) -> pd.DataFrame:
    """Return a DataFrame comparing key metrics across multiple BacktestResults."""
    rows = []
    key_metrics = [
        "Total Return (%)", "CAGR (%)", "Sharpe Ratio",
        "Sortino Ratio", "Max Drawdown (%)", "Calmar Ratio",
        "Win Rate (%)", "Profit Factor", "# Trades",
    ]
    for r in results:
        m = compute_metrics(r, risk_free_rate)
        row = {"Strategy": r.strategy_name, "Ticker": r.ticker}
        row.update({k: m[k] for k in key_metrics})
        rows.append(row)
    df = pd.DataFrame(rows).set_index(["Ticker", "Strategy"])
    return df


# ─── Text report ──────────────────────────────────────────────────────────────

def print_report(result: "BacktestResult", risk_free_rate: float = 0.04) -> None:
    """Print a formatted performance report to stdout."""
    m = compute_metrics(result, risk_free_rate)
    result.metrics = m

    sep = "─" * 55
    print(f"\n{'═'*55}")
    print(f"  Backtest Report │ {result.ticker} │ {result.strategy_name}")
    print(f"{'═'*55}")
    print(f"  Period : {result.equity_curve.index[0].date()} → "
          f"{result.equity_curve.index[-1].date()}")
    print(f"  Capital: ${result.initial_capital:,.0f}")
    print(sep)

    sections = {
        "Returns": ["Total Return (%)", "CAGR (%)", "Final Equity ($)"],
        "Risk":    ["Ann. Volatility (%)", "Max Drawdown (%)",
                    "Max DD Duration (days)", "VaR 95% (daily %)", "CVaR 95% (daily %)"],
        "Ratios":  ["Sharpe Ratio", "Sortino Ratio", "Calmar Ratio"],
        "Trades":  ["# Trades", "Win Rate (%)", "Profit Factor",
                    "Avg Trade PnL (%)", "Best Trade (%)", "Worst Trade (%)",
                    "Total Commission ($)"],
    }

    for section, keys in sections.items():
        print(f"\n  [{section}]")
        for k in keys:
            print(f"    {k:<30} {m[k]:>12}")
    print(f"\n{'═'*55}\n")
