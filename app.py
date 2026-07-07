"""
Quantitative Trading Strategy Backtester
==========================================
A research-grade backtesting tool for evaluating trading strategies against
historical price data, with risk-adjusted performance metrics, benchmark
comparison, parameter sensitivity analysis, and out-of-sample validation.

Author: (your name)
Built with: Streamlit, yfinance, Plotly, NumPy, Pandas
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# ----------------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Quantitative Trading Strategy Backtester",
    page_icon="📈",
    layout="wide",
)

# ----------------------------------------------------------------------------
# DATA LAYER
# ----------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_price_data(ticker: str, period: str) -> pd.DataFrame:
    """Download OHLCV data and flatten any MultiIndex columns from yfinance."""
    df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna()
    return df


# ----------------------------------------------------------------------------
# STRATEGY LAYER
# Each strategy function takes a price DataFrame + params and returns
# a 'signal' column: 1 = long, 0 = flat (long-only, no shorting, for simplicity
# and so results are interpretable to a non-specialist admissions reader).
# ----------------------------------------------------------------------------

def strategy_sma_crossover(df: pd.DataFrame, fast: int = 20, slow: int = 50) -> pd.DataFrame:
    out = df.copy()
    out["fast_ma"] = out["Close"].rolling(fast).mean()
    out["slow_ma"] = out["Close"].rolling(slow).mean()
    out["signal"] = np.where(out["fast_ma"] > out["slow_ma"], 1, 0)
    out["signal"] = out["signal"].shift(1).fillna(0)  # avoid look-ahead bias
    return out


def strategy_rsi_mean_reversion(df: pd.DataFrame, period: int = 14,
                                 oversold: int = 30, overbought: int = 70) -> pd.DataFrame:
    out = df.copy()
    delta = out["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out["rsi"] = 100 - (100 / (1 + rs))

    signal = pd.Series(0, index=out.index, dtype=float)
    position = 0
    rsi_vals = out["rsi"].values
    for i in range(len(out)):
        if np.isnan(rsi_vals[i]):
            signal.iloc[i] = position
            continue
        if rsi_vals[i] < oversold:
            position = 1
        elif rsi_vals[i] > overbought:
            position = 0
        signal.iloc[i] = position
    out["signal"] = signal.shift(1).fillna(0)
    return out


def strategy_bollinger_bands(df: pd.DataFrame, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    out = df.copy()
    out["mid"] = out["Close"].rolling(window).mean()
    std = out["Close"].rolling(window).std()
    out["upper"] = out["mid"] + num_std * std
    out["lower"] = out["mid"] - num_std * std

    signal = pd.Series(0, index=out.index, dtype=float)
    position = 0
    close = out["Close"].values
    lower = out["lower"].values
    upper = out["upper"].values
    mid = out["mid"].values
    for i in range(len(out)):
        if np.isnan(lower[i]):
            signal.iloc[i] = position
            continue
        if close[i] < lower[i]:
            position = 1
        elif close[i] > mid[i]:
            position = 0
        signal.iloc[i] = position
    out["signal"] = signal.shift(1).fillna(0)
    return out


def strategy_momentum(df: pd.DataFrame, lookback: int = 60) -> pd.DataFrame:
    out = df.copy()
    out["momentum"] = out["Close"].pct_change(lookback)
    out["signal"] = np.where(out["momentum"] > 0, 1, 0)
    out["signal"] = out["signal"].shift(1).fillna(0)
    return out


STRATEGIES = {
    "📈 Trend Following — SMA Crossover": {
        "fn": strategy_sma_crossover,
        "params": {"fast": ("Fast MA window", 5, 100, 20), "slow": ("Slow MA window", 10, 250, 50)},
    },
    "🔄 Mean Reversion — RSI": {
        "fn": strategy_rsi_mean_reversion,
        "params": {"period": ("RSI period", 5, 30, 14), "oversold": ("Oversold level", 10, 40, 30),
                   "overbought": ("Overbought level", 60, 90, 70)},
    },
    "📊 Mean Reversion — Bollinger Bands": {
        "fn": strategy_bollinger_bands,
        "params": {"window": ("Window", 10, 60, 20), "num_std": ("Std Dev multiplier", 1.0, 3.0, 2.0)},
    },
    "🚀 Momentum": {
        "fn": strategy_momentum,
        "params": {"lookback": ("Lookback window (days)", 10, 250, 60)},
    },
}


# ----------------------------------------------------------------------------
# BACKTEST ENGINE
# ----------------------------------------------------------------------------
def run_backtest(df: pd.DataFrame, initial_capital: float, cost_bps: float = 5.0) -> pd.DataFrame:
    """
    Vectorized backtest with transaction cost / slippage modeling.
    cost_bps: round-trip cost in basis points applied whenever position changes.
    """
    out = df.copy()
    out["returns"] = out["Close"].pct_change().fillna(0)
    out["position_change"] = out["signal"].diff().abs().fillna(0)
    out["trade_cost"] = out["position_change"] * (cost_bps / 10000.0)
    out["strategy_returns"] = out["signal"] * out["returns"] - out["trade_cost"]
    out["strategy_equity"] = initial_capital * (1 + out["strategy_returns"]).cumprod()
    out["buyhold_equity"] = initial_capital * (1 + out["returns"]).cumprod()
    return out


# ----------------------------------------------------------------------------
# RISK / PERFORMANCE METRICS
# ----------------------------------------------------------------------------
def compute_metrics(returns: pd.Series, equity: pd.Series, periods_per_year: int = 252) -> dict:
    returns = returns.dropna()
    if len(returns) < 2 or equity.iloc[0] == 0:
        return {k: np.nan for k in
                ["Total Return", "CAGR", "Annual Volatility", "Sharpe Ratio", "Sortino Ratio",
                 "Max Drawdown", "Calmar Ratio", "Win Rate", "Number of Trades"]}

    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    n_years = len(returns) / periods_per_year
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / n_years) - 1 if n_years > 0 else np.nan

    ann_vol = returns.std() * np.sqrt(periods_per_year)
    ann_return = returns.mean() * periods_per_year
    sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan

    downside = returns[returns < 0]
    downside_vol = downside.std() * np.sqrt(periods_per_year) if len(downside) > 0 else np.nan
    sortino = ann_return / downside_vol if downside_vol and downside_vol > 0 else np.nan

    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    max_dd = drawdown.min()

    calmar = cagr / abs(max_dd) if max_dd != 0 else np.nan
    win_rate = (returns > 0).sum() / (returns != 0).sum() if (returns != 0).sum() > 0 else np.nan

    return {
        "Total Return": total_return,
        "CAGR": cagr,
        "Annual Volatility": ann_vol,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Max Drawdown": max_dd,
        "Calmar Ratio": calmar,
        "Win Rate": win_rate,
    }


def fmt_pct(x):
    return "—" if pd.isna(x) else f"{x * 100:.2f}%"


def fmt_num(x):
    return "—" if pd.isna(x) else f"{x:.2f}"


# ----------------------------------------------------------------------------
# CHARTING
# ----------------------------------------------------------------------------
def plot_equity_curve(result: pd.DataFrame, strategy_name: str, split_date=None):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3],
                         vertical_spacing=0.05,
                         subplot_titles=("Equity Curve: Strategy vs Buy & Hold", "Drawdown"))

    fig.add_trace(go.Scatter(x=result.index, y=result["strategy_equity"], name=strategy_name,
                              line=dict(color="#E63946", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=result.index, y=result["buyhold_equity"], name="Buy & Hold",
                              line=dict(color="#457B9D", width=2, dash="dash")), row=1, col=1)

    running_max = result["strategy_equity"].cummax()
    drawdown = (result["strategy_equity"] - running_max) / running_max
    fig.add_trace(go.Scatter(x=result.index, y=drawdown * 100, fill="tozeroy", name="Drawdown %",
                              line=dict(color="#E63946")), row=2, col=1)

    if split_date is not None:
        fig.add_vline(x=split_date, line_dash="dot", line_color="gray", row=1, col=1)
        fig.add_annotation(x=split_date, y=1, yref="paper", text="In-sample | Out-of-sample",
                            showarrow=False, row=1, col=1, font=dict(size=10, color="gray"))

    fig.update_layout(height=550, hovermode="x unified", legend=dict(orientation="h", y=1.08))
    fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
    return fig


def plot_price_with_signals(result: pd.DataFrame, strategy_key: str):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=result.index, y=result["Close"], name="Close Price",
                              line=dict(color="#1d3557", width=1.5)))

    if "fast_ma" in result.columns:
        fig.add_trace(go.Scatter(x=result.index, y=result["fast_ma"], name="Fast MA",
                                  line=dict(color="#f4a261", width=1)))
        fig.add_trace(go.Scatter(x=result.index, y=result["slow_ma"], name="Slow MA",
                                  line=dict(color="#2a9d8f", width=1)))
    if "upper" in result.columns:
        fig.add_trace(go.Scatter(x=result.index, y=result["upper"], name="Upper Band",
                                  line=dict(color="gray", width=1, dash="dot")))
        fig.add_trace(go.Scatter(x=result.index, y=result["lower"], name="Lower Band",
                                  line=dict(color="gray", width=1, dash="dot")))
        fig.add_trace(go.Scatter(x=result.index, y=result["mid"], name="Mid Band",
                                  line=dict(color="orange", width=1, dash="dash")))

    buys = result[result["signal"].diff() == 1]
    sells = result[result["signal"].diff() == -1]
    fig.add_trace(go.Scatter(x=buys.index, y=buys["Close"], mode="markers", name="Buy",
                              marker=dict(symbol="triangle-up", color="green", size=10)))
    fig.add_trace(go.Scatter(x=sells.index, y=sells["Close"], mode="markers", name="Sell",
                              marker=dict(symbol="triangle-down", color="red", size=10)))

    fig.update_layout(height=450, title="Price Chart with Entry / Exit Signals",
                       hovermode="x unified", legend=dict(orientation="h", y=1.1))
    return fig


# ----------------------------------------------------------------------------
# SIDEBAR — SETTINGS
# ----------------------------------------------------------------------------
st.sidebar.markdown("## ⚙️ Settings")

mode = st.sidebar.radio("Mode", ["Single", "Multi-Strategy", "Multi-Stock", "Parameter Sensitivity"])

if mode in ("Single", "Multi-Strategy", "Parameter Sensitivity"):
    tickers_input = st.sidebar.text_input("Stock Ticker", value="AAPL")
    tickers = [tickers_input.strip().upper()]
else:
    tickers_input = st.sidebar.text_input("Stock Tickers (comma-separated)", value="AAPL, MSFT, GOOGL")
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

if mode in ("Single", "Multi-Stock", "Parameter Sensitivity"):
    strategy_name = st.sidebar.selectbox("Strategy", list(STRATEGIES.keys()))
    strategy_names = [strategy_name]
else:
    strategy_names = st.sidebar.multiselect("Strategies to compare", list(STRATEGIES.keys()),
                                             default=list(STRATEGIES.keys())[:2])

period = st.sidebar.selectbox("Period", ["1y", "2y", "5y", "10y", "max"], index=2)
initial_capital = st.sidebar.number_input("Initial Capital ($)", min_value=100, value=10000, step=100)
cost_bps = st.sidebar.slider("Transaction Cost (bps per trade)", 0, 50, 5,
                              help="Round-trip trading cost in basis points, applied whenever the "
                                   "position changes. Set to 0 to ignore costs.")
oos_split = st.sidebar.slider("Out-of-sample split (%)", 50, 95, 70,
                               help="Percentage of the data used for in-sample analysis. The "
                                    "remainder is held out to check the strategy isn't overfit.")

# Strategy-specific parameter controls (only shown for Single / Multi-Stock / Sensitivity modes)
custom_params = {}
if mode in ("Single", "Multi-Stock"):
    st.sidebar.markdown("#### Strategy Parameters")
    spec = STRATEGIES[strategy_names[0]]
    for pkey, (label, lo, hi, default) in spec["params"].items():
        if isinstance(default, int):
            custom_params[pkey] = st.sidebar.slider(label, lo, hi, default)
        else:
            custom_params[pkey] = st.sidebar.slider(label, float(lo), float(hi), float(default), step=0.1)

run_btn = st.sidebar.button("🚀 Run Backtest", use_container_width=True, type="primary")

st.sidebar.markdown("---")
st.sidebar.caption(
    "⚠️ Educational tool. Backtested performance does not guarantee future results. "
    "Long-only strategies, no shorting. Adjusted close prices via yfinance."
)

# ----------------------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------------------
st.markdown("# 📈 Quantitative Trading Strategy Backtester")
st.markdown(
    "Backtest trading strategies on historical data with **risk-adjusted metrics**, "
    "**benchmark comparison**, **transaction costs**, and **out-of-sample validation**."
)

if not run_btn:
    st.info("👈 Choose a mode and click **Run Backtest**")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("### 🎯 Single")
        st.write("One stock, one strategy. Full chart with entry/exit signals and risk metrics.")
    with c2:
        st.markdown("### 📊 Multi-Strategy")
        st.write("One stock, several strategies side by side — see which approach works best.")
    with c3:
        st.markdown("### 🌍 Multi-Stock")
        st.write("Same strategy across multiple tickers — test robustness across companies.")
    with c4:
        st.markdown("### 🔬 Sensitivity")
        st.write("Heatmap of performance across a strategy's parameter space — checks for overfitting.")
    st.stop()

# ----------------------------------------------------------------------------
# EXECUTION
# ----------------------------------------------------------------------------

def get_data_or_warn(ticker, period):
    df = fetch_price_data(ticker, period)
    if df.empty:
        st.error(f"No data found for ticker **{ticker}**. Check the symbol and try again.")
        return None
    return df


def split_index(df):
    n = len(df)
    split_i = int(n * oos_split / 100)
    return df.index[split_i] if 0 < split_i < n else None


# ---------- MODE: SINGLE ----------
if mode == "Single":
    ticker = tickers[0]
    df = get_data_or_warn(ticker, period)
    if df is not None:
        spec = STRATEGIES[strategy_names[0]]
        sig_df = spec["fn"](df, **custom_params)
        result = run_backtest(sig_df, initial_capital, cost_bps)
        split_dt = split_index(result)

        full_metrics = compute_metrics(result["strategy_returns"], result["strategy_equity"])
        bh_metrics = compute_metrics(result["returns"], result["buyhold_equity"])

        if split_dt is not None:
            is_res = result.loc[:split_dt]
            oos_res = result.loc[split_dt:]
            is_metrics = compute_metrics(is_res["strategy_returns"], is_res["strategy_equity"])
            oos_metrics = compute_metrics(oos_res["strategy_returns"], oos_res["strategy_equity"])
        else:
            is_metrics = oos_metrics = None

        st.markdown(f"## {ticker} — {strategy_names[0]}")

        cols = st.columns(6)
        labels = ["Total Return", "CAGR", "Sharpe Ratio", "Max Drawdown", "Sortino Ratio", "Win Rate"]
        for c, lbl in zip(cols, labels):
            val = full_metrics[lbl]
            display = fmt_pct(val) if lbl in ("Total Return", "CAGR", "Max Drawdown", "Win Rate") else fmt_num(val)
            c.metric(lbl, display)

        st.plotly_chart(plot_equity_curve(result, strategy_names[0], split_dt), use_container_width=True)
        st.plotly_chart(plot_price_with_signals(result, strategy_names[0]), use_container_width=True)

        st.markdown("### 📋 Strategy vs Buy & Hold")
        comp_df = pd.DataFrame({"Strategy": full_metrics, "Buy & Hold": bh_metrics}).T
        comp_df_display = comp_df.copy()
        for col in ["Total Return", "CAGR", "Annual Volatility", "Max Drawdown", "Win Rate"]:
            comp_df_display[col] = comp_df_display[col].apply(fmt_pct)
        for col in ["Sharpe Ratio", "Sortino Ratio", "Calmar Ratio"]:
            comp_df_display[col] = comp_df_display[col].apply(fmt_num)
        st.dataframe(comp_df_display, use_container_width=True)

        if is_metrics is not None:
            st.markdown("### 🔬 In-Sample vs Out-of-Sample (Overfitting Check)")
            st.caption(
                f"In-sample: start → {split_dt.date()} ({oos_split}% of data). "
                f"Out-of-sample: {split_dt.date()} → end. A strategy that performs well in-sample "
                "but poorly out-of-sample is likely overfit to historical noise."
            )
            oos_df = pd.DataFrame({"In-Sample": is_metrics, "Out-of-Sample": oos_metrics}).T
            oos_df_display = oos_df.copy()
            for col in ["Total Return", "CAGR", "Annual Volatility", "Max Drawdown", "Win Rate"]:
                oos_df_display[col] = oos_df_display[col].apply(fmt_pct)
            for col in ["Sharpe Ratio", "Sortino Ratio", "Calmar Ratio"]:
                oos_df_display[col] = oos_df_display[col].apply(fmt_num)
            st.dataframe(oos_df_display, use_container_width=True)

        with st.expander("📖 How this strategy works"):
            st.write({
                "📈 Trend Following — SMA Crossover": "Goes long when the fast moving average crosses "
                    "above the slow moving average (an uptrend signal), and exits when it crosses back below.",
                "🔄 Mean Reversion — RSI": "Buys when RSI falls below the oversold threshold (price has "
                    "fallen 'too far, too fast') and sells when RSI rises above the overbought threshold.",
                "📊 Mean Reversion — Bollinger Bands": "Buys when price dips below the lower band "
                    "(statistically cheap) and exits when price reverts to the moving average.",
                "🚀 Momentum": "Goes long when trailing N-day returns are positive, betting that recent "
                    "winners keep winning over the lookback horizon.",
            }[strategy_names[0]])

# ---------- MODE: MULTI-STRATEGY ----------
elif mode == "Multi-Strategy":
    ticker = tickers[0]
    df = get_data_or_warn(ticker, period)
    if df is not None and strategy_names:
        st.markdown(f"## {ticker} — Strategy Comparison")
        rows = []
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=initial_capital * (1 + df["Close"].pct_change().fillna(0)).cumprod(),
                                  name="Buy & Hold", line=dict(color="gray", dash="dash")))

        for name in strategy_names:
            spec = STRATEGIES[name]
            default_params = {k: v[3] for k, v in spec["params"].items()}
            sig_df = spec["fn"](df, **default_params)
            result = run_backtest(sig_df, initial_capital, cost_bps)
            metrics = compute_metrics(result["strategy_returns"], result["strategy_equity"])
            metrics["Strategy"] = name
            rows.append(metrics)
            fig.add_trace(go.Scatter(x=result.index, y=result["strategy_equity"], name=name))

        fig.update_layout(height=500, title="Equity Curves", hovermode="x unified",
                           legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, use_container_width=True)

        comp = pd.DataFrame(rows).set_index("Strategy")
        comp_display = comp.copy()
        for col in ["Total Return", "CAGR", "Annual Volatility", "Max Drawdown", "Win Rate"]:
            comp_display[col] = comp_display[col].apply(fmt_pct)
        for col in ["Sharpe Ratio", "Sortino Ratio", "Calmar Ratio"]:
            comp_display[col] = comp_display[col].apply(fmt_num)
        st.markdown("### 📋 Performance Comparison")
        st.dataframe(comp_display, use_container_width=True)

        best = comp["Sharpe Ratio"].idxmax()
        st.success(f"🏆 Best risk-adjusted performer (highest Sharpe Ratio): **{best}**")

# ---------- MODE: MULTI-STOCK ----------
elif mode == "Multi-Stock":
    spec = STRATEGIES[strategy_names[0]]
    st.markdown(f"## {strategy_names[0]} — Across {len(tickers)} Stocks")
    rows = []
    fig = go.Figure()

    for ticker in tickers:
        df = get_data_or_warn(ticker, period)
        if df is None:
            continue
        sig_df = spec["fn"](df, **custom_params)
        result = run_backtest(sig_df, initial_capital, cost_bps)
        metrics = compute_metrics(result["strategy_returns"], result["strategy_equity"])
        metrics["Ticker"] = ticker
        rows.append(metrics)
        # normalize to 100 for comparability across price levels
        normalized = 100 * result["strategy_equity"] / result["strategy_equity"].iloc[0]
        fig.add_trace(go.Scatter(x=result.index, y=normalized, name=ticker))

    if rows:
        fig.update_layout(height=500, title="Normalized Equity Curves (Base = 100)",
                           hovermode="x unified", legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, use_container_width=True)

        comp = pd.DataFrame(rows).set_index("Ticker")
        comp_display = comp.copy()
        for col in ["Total Return", "CAGR", "Annual Volatility", "Max Drawdown", "Win Rate"]:
            comp_display[col] = comp_display[col].apply(fmt_pct)
        for col in ["Sharpe Ratio", "Sortino Ratio", "Calmar Ratio"]:
            comp_display[col] = comp_display[col].apply(fmt_num)
        st.markdown("### 📋 Performance by Stock")
        st.dataframe(comp_display, use_container_width=True)

        best = comp["Sharpe Ratio"].idxmax()
        st.success(f"🏆 Strategy works best on: **{best}** (highest Sharpe Ratio)")

# ---------- MODE: PARAMETER SENSITIVITY ----------
elif mode == "Parameter Sensitivity":
    ticker = tickers[0]
    df = get_data_or_warn(ticker, period)
    spec = STRATEGIES[strategy_names[0]]
    param_keys = list(spec["params"].keys())

    st.markdown(f"## 🔬 Parameter Sensitivity — {strategy_names[0]} on {ticker}")
    st.caption(
        "This heatmap shows the Sharpe Ratio across a grid of parameter values. "
        "A strategy that only performs well at one exact parameter combination — surrounded by poor "
        "performance everywhere else — is a red flag for overfitting rather than a genuine edge."
    )

    if df is not None and len(param_keys) >= 2:
        p1_key, p2_key = param_keys[0], param_keys[1]
        p1_label, p1_lo, p1_hi, p1_default = spec["params"][p1_key]
        p2_label, p2_lo, p2_hi, p2_default = spec["params"][p2_key]

        n_steps = 8
        p1_values = np.linspace(p1_lo, p1_hi, n_steps)
        p2_values = np.linspace(p2_lo, p2_hi, n_steps)
        if isinstance(p1_default, int):
            p1_values = sorted(set(int(v) for v in p1_values))
        if isinstance(p2_default, int):
            p2_values = sorted(set(int(v) for v in p2_values))

        other_params = {k: v[3] for k, v in spec["params"].items() if k not in (p1_key, p2_key)}

        sharpe_grid = np.full((len(p2_values), len(p1_values)), np.nan)
        progress = st.progress(0, text="Running parameter grid...")
        total = len(p1_values) * len(p2_values)
        i = 0
        for r, p2v in enumerate(p2_values):
            for c, p1v in enumerate(p1_values):
                params = {p1_key: p1v, p2_key: p2v, **other_params}
                try:
                    sig_df = spec["fn"](df, **params)
                    result = run_backtest(sig_df, initial_capital, cost_bps)
                    m = compute_metrics(result["strategy_returns"], result["strategy_equity"])
                    sharpe_grid[r, c] = m["Sharpe Ratio"]
                except Exception:
                    sharpe_grid[r, c] = np.nan
                i += 1
                progress.progress(i / total, text="Running parameter grid...")
        progress.empty()

        heat = go.Figure(data=go.Heatmap(
            z=sharpe_grid, x=[str(v) for v in p1_values], y=[str(v) for v in p2_values],
            colorscale="RdYlGn", colorbar=dict(title="Sharpe"),
        ))
        heat.update_layout(
            height=500, title=f"Sharpe Ratio Grid: {p1_label} vs {p2_label}",
            xaxis_title=p1_label, yaxis_title=p2_label,
        )
        st.plotly_chart(heat, use_container_width=True)

        flat = sharpe_grid.flatten()
        flat = flat[~np.isnan(flat)]
        if len(flat) > 0:
            st.write(
                f"**Grid summary:** Sharpe ranges from {flat.min():.2f} to {flat.max():.2f} "
                f"(mean {flat.mean():.2f}, std {flat.std():.2f}). "
                + ("⚠️ High variance across the grid suggests results are sensitive to exact parameter "
                   "choice — be cautious about reading too much into any single backtest."
                   if flat.std() > 0.5 else
                   "✅ Relatively stable performance across the grid is a good sign of robustness.")
            )
    else:
        st.warning("This strategy needs at least two tunable parameters for a 2D sensitivity grid.")
