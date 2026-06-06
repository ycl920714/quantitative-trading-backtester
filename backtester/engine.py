"""
Core backtesting engine.

Usage
-----
bt = Backtester(initial_capital=100_000, commission=0.001, slippage=0.001)
result = bt.run(strategy, price_df)
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Optional


# ─── Data containers ─────────────────────────────────────────────────────────

@dataclass
class Trade:
    entry_date:  pd.Timestamp
    exit_date:   Optional[pd.Timestamp]
    entry_price: float
    exit_price:  Optional[float]
    shares:      float
    direction:   str          # "long" | "short"
    pnl:         float = 0.0
    pnl_pct:     float = 0.0
    commission:  float = 0.0


@dataclass
class BacktestResult:
    ticker:         str
    strategy_name:  str
    initial_capital: float
    equity_curve:   pd.Series
    trades:         list[Trade]
    signals:        pd.DataFrame
    metrics:        dict = field(default_factory=dict)

    # convenience
    @property
    def final_equity(self) -> float:
        return float(self.equity_curve.iloc[-1])

    @property
    def total_return(self) -> float:
        return (self.final_equity / self.initial_capital) - 1


# ─── Engine ───────────────────────────────────────────────────────────────────

class Backtester:
    """
    Event-driven backtester supporting long/short signals, commission, and
    fixed-fractional position sizing.

    Parameters
    ----------
    initial_capital : float
        Starting portfolio cash (USD).
    commission : float
        Round-trip commission rate per trade (e.g. 0.001 = 0.1%).
    slippage : float
        One-way slippage as a fraction of price.
    position_size : float
        Fraction of current equity to deploy per trade (default 1.0 = 100%).
    allow_short : bool
        Whether to take short positions when signal == -1.
    """

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        commission: float = 0.001,
        slippage: float = 0.0005,
        position_size: float = 1.0,
        allow_short: bool = False,
    ) -> None:
        self.initial_capital = initial_capital
        self.commission      = commission
        self.slippage        = slippage
        self.position_size   = position_size
        self.allow_short     = allow_short

    # ------------------------------------------------------------------
    def run(
        self,
        strategy: "BaseStrategy",
        df: pd.DataFrame,
        ticker: str = "UNKNOWN",
    ) -> BacktestResult:
        """
        Run a strategy on price data and return a BacktestResult.

        The strategy's `generate_signals` method must return a DataFrame
        that includes a 'signal' column: +1 = long, -1 = short, 0 = flat.
        """
        signals_df = strategy.generate_signals(df.copy())
        signals_df = signals_df.dropna(subset=["signal"])

        cash       = self.initial_capital
        shares     = 0.0
        direction  = 0   # +1 long, -1 short, 0 flat
        entry_date = None
        entry_price= None
        trades: list[Trade] = []
        equity_values = []

        prices = signals_df["Close"].values
        dates  = signals_df.index
        sigs   = signals_df["signal"].values

        for i in range(len(signals_df)):
            price = prices[i]
            sig   = int(sigs[i])
            date  = dates[i]

            # ── close existing position if signal changed ──────────────
            if direction != 0 and sig != direction:
                slip  = price * self.slippage
                exit_px = price - slip if direction == 1 else price + slip
                comm  = abs(shares) * exit_px * self.commission

                if direction == 1:      # close long
                    pnl = shares * (exit_px - entry_price) - comm
                    cash += shares * exit_px - comm
                else:                   # close short
                    pnl = shares * (entry_price - exit_px) - comm
                    cash += shares * entry_price + pnl

                pnl_pct = pnl / (abs(shares) * entry_price)
                trades.append(Trade(
                    entry_date=entry_date, exit_date=date,
                    entry_price=entry_price, exit_price=exit_px,
                    shares=abs(shares), direction="long" if direction==1 else "short",
                    pnl=pnl, pnl_pct=pnl_pct, commission=comm,
                ))
                shares = 0.0
                direction = 0

            # ── open new position ──────────────────────────────────────
            if sig != 0 and direction == 0:
                if sig == -1 and not self.allow_short:
                    pass
                else:
                    slip     = price * self.slippage
                    entry_px = price + slip if sig == 1 else price - slip
                    invest   = cash * self.position_size
                    shares   = invest / entry_px
                    comm     = shares * entry_px * self.commission
                    shares  -= comm / entry_px   # reduce shares by commission cost
                    cash    -= invest
                    direction  = sig
                    entry_date = date
                    entry_price= entry_px

            # ── mark-to-market equity ──────────────────────────────────
            if direction == 1:
                mkt_value = shares * price
            elif direction == -1:
                mkt_value = shares * (2 * entry_price - price)   # short P&L
            else:
                mkt_value = 0.0
            equity_values.append(cash + mkt_value)

        # close any open trade at the last price
        if direction != 0:
            price = prices[-1]
            date  = dates[-1]
            slip  = price * self.slippage
            exit_px = price - slip if direction == 1 else price + slip
            comm  = abs(shares) * exit_px * self.commission
            if direction == 1:
                pnl = shares * (exit_px - entry_price) - comm
            else:
                pnl = shares * (entry_price - exit_px) - comm
            pnl_pct = pnl / (abs(shares) * entry_price)
            trades.append(Trade(
                entry_date=entry_date, exit_date=date,
                entry_price=entry_price, exit_price=exit_px,
                shares=abs(shares), direction="long" if direction==1 else "short",
                pnl=pnl, pnl_pct=pnl_pct, commission=comm,
            ))

        equity_curve = pd.Series(equity_values, index=dates, name="equity")

        result = BacktestResult(
            ticker=ticker,
            strategy_name=strategy.name,
            initial_capital=self.initial_capital,
            equity_curve=equity_curve,
            trades=trades,
            signals=signals_df,
        )
        return result
