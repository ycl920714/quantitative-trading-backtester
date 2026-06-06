#!/usr/bin/env python3
"""
Auto Quantitative Trading Strategy Backtester for US Stocks
============================================================

Usage examples
--------------
# Run all strategies on AAPL for the past 5 years
python backtest_runner.py --ticker AAPL

# Compare strategies on multiple tickers
python backtest_runner.py --ticker AAPL MSFT NVDA --strategy all

# Run a specific strategy with custom parameters
python backtest_runner.py --ticker TSLA --strategy sma_crossover --fast 20 --slow 100

# Custom date range, $50k capital, allow short selling
python backtest_runner.py --ticker SPY --start 2018-01-01 --end 2023-12-31 \\
    --capital 50000 --short --strategy macd rsi_mean_reversion

Available strategies
--------------------
  sma_crossover       Dual SMA crossover (golden/death cross)
  ema_crossover       Dual EMA crossover (faster signals)
  rsi_mean_reversion  RSI oversold/overbought mean reversion
  macd                MACD histogram direction
  bb_breakout         Bollinger Band breakout (momentum)
  bb_mean_reversion   Bollinger Band mean reversion
  momentum            N-day Rate-of-Change momentum
  dual_thrust         Volatility range breakout
  buy_and_hold        Buy-and-hold benchmark
  all                 Run every strategy (default)
"""
from __future__ import annotations

import argparse
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless-safe; swap to "TkAgg" for interactive
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter

# ── path setup so this script can be run from any directory ───────────────────
sys.path.insert(0, str(Path(__file__).parent))

from backtester import (
    Backtester, BacktestResult,
    STRATEGY_REGISTRY,
    fetch_stock_data, add_indicators,
    compute_metrics, compare_results, print_report,
)
from backtester.strategies import (
    SMACrossover, EMACrossover, RSIMeanReversion, MACDStrategy,
    BollingerBandBreakout, BollingerBandMeanReversion,
    MomentumStrategy, DualThrust, BuyAndHold,
)


# ─── Plotting ─────────────────────────────────────────────────────────────────

COLORS = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63",
          "#9C27B0", "#00BCD4", "#F44336", "#8BC34A", "#795548"]


def _pct_fmt(x, _):
    return f"{x*100:.0f}%"


def plot_single(result: BacktestResult, df: pd.DataFrame, out_path: str) -> None:
    """6-panel chart for a single strategy run."""
    signals = result.signals
    eq      = result.equity_curve
    metrics = compute_metrics(result)

    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor("#0d1117")
    gs  = gridspec.GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.3)

    ax_price  = fig.add_subplot(gs[0, :])
    ax_equity = fig.add_subplot(gs[1, :])
    ax_dd     = fig.add_subplot(gs[2, 0])
    ax_rets   = fig.add_subplot(gs[2, 1])
    ax_rsi    = fig.add_subplot(gs[3, 0])
    ax_macd   = fig.add_subplot(gs[3, 1])

    dark_style = dict(facecolor="#0d1117", labelcolor="#c9d1d9")
    for ax in [ax_price, ax_equity, ax_dd, ax_rets, ax_rsi, ax_macd]:
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="#c9d1d9", labelsize=8)
        ax.spines[:].set_color("#30363d")
        ax.title.set_color("#c9d1d9")
        ax.xaxis.label.set_color("#c9d1d9")
        ax.yaxis.label.set_color("#c9d1d9")

    # ── Panel 1: Price + signals ───────────────────────────────────────
    ax_price.plot(signals.index, signals["Close"], color="#58a6ff", lw=1.2,
                  label="Close")
    for col, color, lbl in [
        ("SMA_fast", "#ffd700", "SMA fast"),
        ("SMA_slow", "#ff6b6b", "SMA slow"),
        ("EMA_fast", "#ffd700", "EMA fast"),
        ("EMA_slow", "#ff6b6b", "EMA slow"),
        ("BB_upper", "#aaa",    "BB upper"),
        ("BB_lower", "#aaa",    "BB lower"),
    ]:
        if col in signals.columns:
            ax_price.plot(signals.index, signals[col], color=color,
                          lw=0.7, ls="--", alpha=0.7, label=lbl)

    # buy / sell markers
    longs  = signals[signals["signal"] == 1]
    shorts = signals[signals["signal"] == -1]
    ax_price.scatter(longs.index,  longs["Close"],  marker="^", color="#4caf50",
                     s=30, zorder=5, label="Long signal")
    ax_price.scatter(shorts.index, shorts["Close"], marker="v", color="#f44336",
                     s=30, zorder=5, label="Short signal")
    ax_price.set_title(f"{result.ticker} — {result.strategy_name}", fontsize=12, pad=6)
    ax_price.legend(loc="upper left", fontsize=7, facecolor="#161b22",
                    labelcolor="#c9d1d9", framealpha=0.8)
    ax_price.set_ylabel("Price ($)")

    # ── Panel 2: Equity curve ─────────────────────────────────────────
    bh_eq = None
    if "buy_and_hold" in STRATEGY_REGISTRY:
        bh = BuyAndHold()
        bh_res = Backtester(result.initial_capital).run(bh, df, result.ticker)
        bh_eq  = bh_res.equity_curve
        ax_equity.plot(bh_eq.index, bh_eq / result.initial_capital - 1,
                       color="#aaa", lw=1, ls="--", label="Buy & Hold")

    ax_equity.plot(eq.index, eq / result.initial_capital - 1,
                   color="#4fc3f7", lw=1.5, label=result.strategy_name)
    ax_equity.axhline(0, color="#555", lw=0.8)
    ax_equity.yaxis.set_major_formatter(FuncFormatter(_pct_fmt))
    ax_equity.set_title("Equity Curve (% return)", fontsize=10)
    ax_equity.legend(fontsize=7, facecolor="#161b22", labelcolor="#c9d1d9")
    ax_equity.fill_between(eq.index, eq / result.initial_capital - 1, 0,
                            where=(eq / result.initial_capital - 1) >= 0,
                            alpha=0.15, color="#4caf50")
    ax_equity.fill_between(eq.index, eq / result.initial_capital - 1, 0,
                            where=(eq / result.initial_capital - 1) < 0,
                            alpha=0.15, color="#f44336")

    # ── Panel 3: Drawdown ─────────────────────────────────────────────
    dd = (eq - eq.cummax()) / eq.cummax()
    ax_dd.fill_between(dd.index, dd, 0, color="#f44336", alpha=0.5)
    ax_dd.plot(dd.index, dd, color="#f44336", lw=0.8)
    ax_dd.yaxis.set_major_formatter(FuncFormatter(_pct_fmt))
    ax_dd.set_title("Drawdown", fontsize=10)
    ax_dd.set_ylabel("DD (%)")

    # ── Panel 4: Daily returns histogram ─────────────────────────────
    rets = eq.pct_change().dropna()
    ax_rets.hist(rets, bins=50, color="#4fc3f7", edgecolor="#0d1117", alpha=0.8)
    ax_rets.axvline(rets.mean(), color="#ffd700", lw=1.2,
                    label=f"Mean {rets.mean()*100:.3f}%")
    ax_rets.axvline(np.percentile(rets, 5), color="#f44336", lw=1.2, ls="--",
                    label=f"VaR 5% {np.percentile(rets,5)*100:.3f}%")
    ax_rets.set_title("Daily Returns Distribution", fontsize=10)
    ax_rets.legend(fontsize=7, facecolor="#161b22", labelcolor="#c9d1d9")

    # ── Panel 5: RSI ─────────────────────────────────────────────────
    if "RSI" in signals.columns:
        ax_rsi.plot(signals.index, signals["RSI"], color="#ff9800", lw=1)
        ax_rsi.axhline(70, color="#f44336", lw=0.8, ls="--", label="70")
        ax_rsi.axhline(30, color="#4caf50", lw=0.8, ls="--", label="30")
        ax_rsi.fill_between(signals.index, signals["RSI"], 70,
                             where=(signals["RSI"] >= 70), alpha=0.2, color="#f44336")
        ax_rsi.fill_between(signals.index, signals["RSI"], 30,
                             where=(signals["RSI"] <= 30), alpha=0.2, color="#4caf50")
        ax_rsi.set_ylim(0, 100)
        ax_rsi.legend(fontsize=7, facecolor="#161b22", labelcolor="#c9d1d9")
    elif "ROC" in signals.columns:
        ax_rsi.plot(signals.index, signals["ROC"], color="#ff9800", lw=1)
        ax_rsi.axhline(0, color="#555", lw=0.8)
        ax_rsi.yaxis.set_major_formatter(FuncFormatter(_pct_fmt))
    ax_rsi.set_title("RSI / Momentum Indicator", fontsize=10)

    # ── Panel 6: MACD ────────────────────────────────────────────────
    if "MACD_hist" in signals.columns:
        hist = signals["MACD_hist"]
        colors_hist = ["#4caf50" if v >= 0 else "#f44336" for v in hist]
        ax_macd.bar(signals.index, hist, color=colors_hist, width=1, alpha=0.8)
        ax_macd.axhline(0, color="#555", lw=0.8)
        ax_macd.set_title("MACD Histogram", fontsize=10)
    else:
        # fallback: volume
        ax_macd.bar(signals.index, df.loc[signals.index, "Volume"] / 1e6,
                    color="#4fc3f7", alpha=0.6, width=1)
        ax_macd.set_title("Volume (M shares)", fontsize=10)

    fig.suptitle(
        f"Backtest │ {result.ticker} │ {result.strategy_name}  "
        f"│  Return: {metrics['Total Return (%)']:+.1f}%  "
        f"│  Sharpe: {metrics['Sharpe Ratio']:.2f}  "
        f"│  MaxDD: {metrics['Max Drawdown (%)']:.1f}%",
        fontsize=11, color="#c9d1d9", y=0.995,
    )

    plt.savefig(out_path, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Chart saved → {out_path}")


def plot_comparison(results: list[BacktestResult], out_path: str) -> None:
    """Overlay equity curves and key metrics for multiple strategies."""
    fig, axes = plt.subplots(2, 1, figsize=(16, 10),
                             gridspec_kw={"height_ratios": [3, 1]})
    fig.patch.set_facecolor("#0d1117")
    for ax in axes:
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="#c9d1d9", labelsize=8)
        ax.spines[:].set_color("#30363d")
        ax.title.set_color("#c9d1d9")
        ax.yaxis.label.set_color("#c9d1d9")

    ax_eq, ax_bar = axes
    capital = results[0].initial_capital

    for i, r in enumerate(results):
        col = COLORS[i % len(COLORS)]
        norm = r.equity_curve / capital - 1
        ax_eq.plot(r.equity_curve.index, norm, color=col, lw=1.5,
                   label=f"{r.strategy_name}")
    ax_eq.axhline(0, color="#555", lw=0.8)
    ax_eq.yaxis.set_major_formatter(FuncFormatter(_pct_fmt))
    ax_eq.set_title(f"Strategy Comparison — {results[0].ticker}", fontsize=13,
                    color="#c9d1d9")
    ax_eq.legend(fontsize=8, facecolor="#161b22", labelcolor="#c9d1d9",
                 framealpha=0.9, loc="upper left")

    names   = [r.strategy_name for r in results]
    rets    = [compute_metrics(r)["Total Return (%)"] for r in results]
    sharpes = [compute_metrics(r)["Sharpe Ratio"]     for r in results]
    bar_colors = ["#4caf50" if v >= 0 else "#f44336" for v in rets]
    bars = ax_bar.bar(names, rets, color=bar_colors, alpha=0.8, edgecolor="#0d1117")
    for bar, s in zip(bars, sharpes):
        ax_bar.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (0.3 if bar.get_height() >= 0 else -1.5),
                    f"SR:{s:.2f}", ha="center", va="bottom",
                    fontsize=7, color="#c9d1d9")
    ax_bar.axhline(0, color="#555", lw=0.8)
    ax_bar.set_title("Total Return (%) + Sharpe Ratio", fontsize=10, color="#c9d1d9")
    ax_bar.set_ylabel("Return (%)", color="#c9d1d9")
    ax_bar.tick_params(axis="x", rotation=20, labelsize=7)

    fig.tight_layout(pad=2)
    plt.savefig(out_path, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Comparison chart saved → {out_path}")


# ─── Strategy factory with CLI param overrides ────────────────────────────────

def build_strategy(name: str, args: argparse.Namespace):
    cls = STRATEGY_REGISTRY[name]
    kwargs = {}

    if name == "sma_crossover":
        if args.fast: kwargs["fast"] = args.fast
        if args.slow: kwargs["slow"] = args.slow

    elif name == "ema_crossover":
        if args.fast: kwargs["fast"] = args.fast
        if args.slow: kwargs["slow"] = args.slow

    elif name == "rsi_mean_reversion":
        if args.rsi_period:     kwargs["period"]     = args.rsi_period
        if args.rsi_oversold:   kwargs["oversold"]   = args.rsi_oversold
        if args.rsi_overbought: kwargs["overbought"] = args.rsi_overbought

    elif name == "momentum":
        if args.lookback: kwargs["lookback"] = args.lookback

    return cls(**kwargs)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Auto Quantitative Trading Strategy Backtester",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Universe
    parser.add_argument("--ticker",  nargs="+", default=["AAPL"],
                        help="One or more US stock tickers (default: AAPL)")
    parser.add_argument("--start",   default=None,
                        help="Start date YYYY-MM-DD (default: 5 years ago)")
    parser.add_argument("--end",     default=None,
                        help="End date YYYY-MM-DD (default: today)")

    # Capital / execution
    parser.add_argument("--capital",  type=float, default=100_000.0,
                        help="Initial capital in USD (default: 100000)")
    parser.add_argument("--commission", type=float, default=0.001,
                        help="Commission rate per trade (default: 0.001 = 0.1%%)")
    parser.add_argument("--slippage",  type=float, default=0.0005,
                        help="Slippage per side (default: 0.0005 = 0.05%%)")
    parser.add_argument("--short",    action="store_true",
                        help="Allow short selling")
    parser.add_argument("--position-size", type=float, default=1.0,
                        help="Fraction of equity per trade (default: 1.0)")

    # Strategies
    parser.add_argument("--strategy", nargs="+",
                        default=["all"],
                        choices=list(STRATEGY_REGISTRY.keys()) + ["all"],
                        help="Strategy or 'all' (default: all)")

    # Strategy params
    parser.add_argument("--fast",           type=int,   default=None)
    parser.add_argument("--slow",           type=int,   default=None)
    parser.add_argument("--lookback",       type=int,   default=None)
    parser.add_argument("--rsi-period",     type=int,   default=None)
    parser.add_argument("--rsi-oversold",   type=float, default=None)
    parser.add_argument("--rsi-overbought", type=float, default=None)

    # Output
    parser.add_argument("--outdir", default="backtest_results",
                        help="Output directory for charts (default: backtest_results/)")
    parser.add_argument("--no-charts", action="store_true",
                        help="Skip chart generation")
    parser.add_argument("--rfr", type=float, default=0.04,
                        help="Annual risk-free rate for Sharpe/Sortino (default: 0.04)")

    args = parser.parse_args()

    # ── Resolve dates ──────────────────────────────────────────────────
    end   = args.end   or datetime.today().strftime("%Y-%m-%d")
    start = args.start or (datetime.today() - timedelta(days=5*365)).strftime("%Y-%m-%d")

    # ── Resolve strategy list ──────────────────────────────────────────
    if "all" in args.strategy:
        strategy_names = list(STRATEGY_REGISTRY.keys())
    else:
        strategy_names = args.strategy

    # ── Output directory ──────────────────────────────────────────────
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    engine = Backtester(
        initial_capital=args.capital,
        commission=args.commission,
        slippage=args.slippage,
        position_size=args.position_size,
        allow_short=args.short,
    )

    all_results: list[BacktestResult] = []

    # ── Run for each ticker ────────────────────────────────────────────
    for ticker in args.ticker:
        print(f"\n{'━'*60}")
        print(f"  Fetching data for {ticker}  ({start} → {end})")
        print(f"{'━'*60}")

        try:
            df = fetch_stock_data(ticker, start, end)
        except ValueError as e:
            print(f"  [ERROR] {e}")
            continue

        print(f"  {len(df)} trading days loaded.")

        ticker_results: list[BacktestResult] = []

        for sname in strategy_names:
            strategy = build_strategy(sname, args)
            try:
                result = engine.run(strategy, df, ticker=ticker)
                result.metrics = compute_metrics(result, args.rfr)
                ticker_results.append(result)
                all_results.append(result)
                print_report(result, risk_free_rate=args.rfr)
            except Exception as e:
                print(f"  [ERROR] Strategy '{sname}' failed on {ticker}: {e}")
                continue

            # per-strategy chart
            if not args.no_charts:
                chart_path = outdir / f"{ticker}_{sname}.png"
                try:
                    plot_single(result, df, str(chart_path))
                except Exception as e:
                    print(f"  [WARN] Could not save chart: {e}")

        # comparison chart for this ticker
        if len(ticker_results) > 1 and not args.no_charts:
            cmp_path = outdir / f"{ticker}_comparison.png"
            try:
                plot_comparison(ticker_results, str(cmp_path))
            except Exception as e:
                print(f"  [WARN] Could not save comparison chart: {e}")

        # Summary table for this ticker
        if ticker_results:
            print(f"\n  ── Summary Table: {ticker} ──")
            cmp_df = compare_results(ticker_results, args.rfr)
            pd.set_option("display.max_columns", None)
            pd.set_option("display.width", 160)
            print(cmp_df.to_string())

    # ── Cross-ticker summary CSV ───────────────────────────────────────
    if all_results:
        csv_path = outdir / "backtest_summary.csv"
        try:
            summary = compare_results(all_results, args.rfr).reset_index()
            summary.to_csv(csv_path, index=False)
            print(f"\n  Summary CSV saved → {csv_path}")
        except Exception as e:
            print(f"  [WARN] Could not save summary CSV: {e}")

    print(f"\n  Done. Results in: {outdir}/\n")


if __name__ == "__main__":
    main()
