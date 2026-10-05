"""
Trading Strategy Backtester (Streamlit web app)

Tests nine rule-based trading strategies against buy-and-hold on daily US
stock data, with transaction costs, next-day execution and an
in-sample / out-of-sample split.

Author: Yun-Chen Lin
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots
from scipy import stats

import importlib

import backtester.strategies
import backtester.vectorised

# Streamlit re-runs app.py on every change but keeps imported modules in
# memory. Reloading makes sure an updated strategy file is actually used.
importlib.reload(backtester.strategies)
importlib.reload(backtester.vectorised)

from backtester.strategies import STRATEGY_REGISTRY
from backtester.vectorised import compute_metrics, run_vectorised, var_cvar

st.set_page_config(
    page_title="Trading Strategy Backtester",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).parent

# ---------------------------------------------------------------------------
# Style: deliberately plain
# ---------------------------------------------------------------------------
st.markdown("""
<style>
#MainMenu, footer, .stDeployButton { display: none !important; }
.main .block-container { max-width: 1150px; padding-top: 2rem; }
h1 { font-weight: 600 !important; letter-spacing: -0.01em; }
.lead { color: #475569; font-size: 1.02rem; line-height: 1.6; max-width: 760px; }
.small { color: #64748B; font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)

INK, GREY, BLUE, RED, GREEN = "#1F2937", "#94A3B8", "#1D4ED8", "#B91C1C", "#15803D"
PALETTE = ["#1D4ED8", "#B45309", "#15803D", "#7C3AED", "#B91C1C", "#0F766E",
           "#BE185D", "#4B5563", "#CA8A04"]
LAYOUT = dict(
    paper_bgcolor="white", plot_bgcolor="white",
    font=dict(family="Inter, Arial, sans-serif", color="#374151", size=12),
    xaxis=dict(gridcolor="#E5E7EB"), yaxis=dict(gridcolor="#E5E7EB"),
    legend=dict(orientation="h", y=1.14),
    hovermode="x unified", margin=dict(t=60, b=40, l=60, r=20),
)

# Strategies shown in the app (buy and hold is always shown as the benchmark)
STRATS = {cls.name: cls for key, cls in STRATEGY_REGISTRY.items() if key != "buy_and_hold"}
DEFAULT_UNIVERSE = "SPY, QQQ, AAPL, MSFT, JPM, XOM, JNJ, KO"


# ---------------------------------------------------------------------------
# Data and helpers
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_price_data(ticker: str, period: str) -> pd.DataFrame:
    df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna()


def backtest(df, strat_name, params, capital, cost_bps, allow_short):
    sig = STRATS[strat_name](**params).generate_signals(df.copy())
    return run_vectorised(sig, capital, cost_bps, allow_short)


def metrics(res, rf, col="strategy"):
    if col == "strategy":
        return compute_metrics(res["strategy_returns"], res["strategy_equity"], rf,
                               position=res["position"])
    return compute_metrics(res["returns"], res["buyhold_equity"], rf)


def default_params(name):
    return {k: v[3] for k, v in STRATS[name].params.items()}


def pct(x):
    return "n/a" if pd.isna(x) else f"{x * 100:.1f}%"


def num(x):
    return "n/a" if pd.isna(x) else f"{x:.2f}"


def fmt_table(rows: dict) -> pd.DataFrame:
    d = pd.DataFrame(rows).T
    out = d.copy()
    for c in ["Total Return", "CAGR", "Annual Volatility", "Max Drawdown", "Time in Market"]:
        if c in out:
            out[c] = d[c].apply(pct)
    for c in ["Sharpe Ratio", "Sortino Ratio", "Calmar Ratio"]:
        if c in out:
            out[c] = d[c].apply(num)
    return out


def split_date(df, pct_in):
    i = int(len(df) * pct_in / 100)
    return df.index[i] if 0 < i < len(df) else None


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def plot_equity(res, label, split_dt=None):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3],
                        vertical_spacing=0.05, subplot_titles=("Equity curve", "Drawdown (%)"))
    fig.add_trace(go.Scatter(x=res.index, y=res["strategy_equity"], name=label,
                             line=dict(color=BLUE, width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=res.index, y=res["buyhold_equity"], name="Buy & Hold",
                             line=dict(color=GREY, width=1.5, dash="dot")), row=1, col=1)
    eq = res["strategy_equity"]
    fig.add_trace(go.Scatter(x=res.index, y=(eq / eq.cummax() - 1) * 100, name="Drawdown",
                             fill="tozeroy", line=dict(color=RED, width=1),
                             fillcolor="rgba(185,28,28,0.08)"), row=2, col=1)
    if split_dt is not None:
        for r in (1, 2):
            fig.add_vline(x=split_dt, line_dash="dot", line_color=GREY, row=r, col=1)
    fig.update_layout(height=540, **LAYOUT)
    return fig


def plot_signals(res):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=res.index, y=res["Close"], name="Price", line=dict(color=INK, width=1.3)))
    for col, nm, c in [("SMA_fast", "Fast SMA", "#B45309"), ("SMA_slow", "Slow SMA", "#7C3AED"),
                       ("EMA_fast", "Fast EMA", "#B45309"), ("EMA_slow", "Slow EMA", "#7C3AED"),
                       ("BB_upper", "Upper band", GREY), ("BB_lower", "Lower band", GREY),
                       ("BB_mid", "Middle band", "#B45309")]:
        if col in res:
            fig.add_trace(go.Scatter(x=res.index, y=res[col], name=nm, line=dict(color=c, width=1)))
    chg = res["position"].diff()
    buys, sells = res[chg > 0], res[chg < 0]
    fig.add_trace(go.Scatter(x=buys.index, y=buys["Open"], mode="markers", name="Buy (at open)",
                             marker=dict(symbol="triangle-up", color=GREEN, size=8)))
    fig.add_trace(go.Scatter(x=sells.index, y=sells["Open"], mode="markers", name="Sell (at open)",
                             marker=dict(symbol="triangle-down", color=RED, size=8)))
    fig.update_layout(height=420, title="Trades", **LAYOUT)
    return fig


def plot_return_distribution(r):
    r = r.dropna()
    r = r[r != 0]  # days with no position carry no information about the strategy's risk
    if len(r) < 20:
        return None, None, None
    x = np.linspace(r.quantile(0.001), r.quantile(0.999), 300)
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=r * 100, histnorm="probability density", nbinsx=80,
                               name="Daily returns (days in market)", marker_color=BLUE, opacity=0.6))
    fig.add_trace(go.Scatter(x=x * 100, y=stats.norm.pdf(x, r.mean(), r.std()) / 100,
                             name="Normal with same mean and std", line=dict(color="#B45309", dash="dash")))
    v, _ = var_cvar(r)
    fig.add_vline(x=v * 100, line_color=RED, line_dash="dot",
                  annotation_text=f"VaR 95%: {v * 100:.2f}%", annotation_font_color=RED)
    fig.update_layout(height=400, xaxis_title="Daily return (%)", yaxis_title="Density", **LAYOUT)
    return fig, float(stats.skew(r)), float(stats.kurtosis(r))


def plot_rolling_sharpe(r, window=126):
    rs = (r.rolling(window).mean() / r.rolling(window).std() * np.sqrt(252)).replace([np.inf, -np.inf], np.nan)
    fig = go.Figure(go.Scatter(x=rs.index, y=rs, name=f"Rolling Sharpe ({window}d)", line=dict(color=BLUE)))
    fig.add_hline(y=0, line_color=GREY, line_dash="dot")
    fig.update_layout(height=340, yaxis_title="Sharpe ratio (annualised)", **LAYOUT)
    return fig


def plot_monte_carlo(r, capital, n_sims=500, horizon=252):
    """Bootstrap of daily strategy returns, INCLUDING days out of the market,
    so the simulated paths keep the strategy's real exposure."""
    r = r.dropna().to_numpy()
    if len(r) < 60:
        return None, None
    rng = np.random.default_rng(42)
    paths = capital * np.cumprod(1 + rng.choice(r, size=(n_sims, horizon), replace=True), axis=1)
    paths = np.hstack([np.full((n_sims, 1), capital), paths])
    days = np.arange(horizon + 1)
    p5, p50, p95 = (np.percentile(paths, q, axis=0) for q in (5, 50, 95))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.r_[days, days[::-1]], y=np.r_[p95, p5[::-1]], fill="toself",
                             fillcolor="rgba(29,78,216,0.12)", line=dict(width=0), name="5th to 95th percentile"))
    fig.add_trace(go.Scatter(x=days, y=p50, name="Median", line=dict(color=BLUE, width=2)))
    fig.add_hline(y=capital, line_color=GREY, line_dash="dot")
    fig.update_layout(height=400, xaxis_title="Trading days ahead", yaxis_title="Portfolio value ($)", **LAYOUT)
    return fig, (paths[:, -1] > capital).mean()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
MODES = ["About", "Single backtest", "Compare strategies", "Compare stocks",
         "Parameter sensitivity", "Strategy study"]

with st.sidebar:
    st.markdown("### Settings")
    mode = st.radio("Mode", MODES, index=MODES.index(st.session_state.get("mode", "About")))
    st.session_state.mode = mode

    if mode in ("Compare stocks", "Strategy study"):
        default = DEFAULT_UNIVERSE if mode == "Strategy study" else "AAPL, MSFT, JPM"
        raw = st.text_input("Tickers (comma separated)", value=default)
        tickers = [t.strip().upper() for t in raw.split(",") if t.strip()]
    else:
        tickers = [st.text_input("Ticker", value="SPY").strip().upper()]

    if mode in ("Single backtest", "Compare stocks", "Parameter sensitivity"):
        strat_names = [st.selectbox("Strategy", list(STRATS))]
    elif mode == "Compare strategies":
        strat_names = st.multiselect("Strategies", list(STRATS), default=list(STRATS))
    else:
        strat_names = list(STRATS)

    period = st.selectbox("History", ["2y", "5y", "10y", "20y", "max"],
                          index=2 if mode == "Strategy study" else 1)
    capital = st.number_input("Starting capital ($)", min_value=100, value=10_000, step=1_000)
    cost_bps = st.slider("Transaction cost per trade (bps)", 0, 50, 10,
                         help="Charged on each unit of position change. 10 bps = 0.10%.")
    in_sample = st.slider("In-sample share of history (%)", 50, 90, 70)
    allow_short = st.checkbox("Allow short selling", value=False)
    rf = st.number_input("Risk-free rate for Sharpe (% per year)", 0.0, 10.0, 0.0, 0.5) / 100

    params = {}
    if mode in ("Single backtest", "Compare stocks"):
        spec = STRATS[strat_names[0]].params
        if spec:
            st.markdown("**Strategy parameters**")
        for k, (label, lo, hi, d) in spec.items():
            params[k] = (st.slider(label, lo, hi, d) if isinstance(d, int)
                         else st.slider(label, float(lo), float(hi), float(d), step=0.1))

    st.caption("Educational project. Historical results do not predict future returns.")


# ---------------------------------------------------------------------------
# About page
# ---------------------------------------------------------------------------
if mode == "About":
    st.title("Trading Strategy Backtester")
    st.markdown(
        '<p class="lead">I built this to answer a simple question: do common technical trading '
        "rules beat buying and holding once you account for trading costs and test them on data "
        "they were not tuned on? It runs nine rule-based strategies on daily US stock and ETF data "
        "and compares each one with buy-and-hold.</p>"
        '<p class="small">Yun-Chen Lin · BSc Finance and Business, University of Sussex · '
        '<a href="https://github.com/ycl920714/quantitative-trading-backtester">Source code on GitHub</a></p>',
        unsafe_allow_html=True,
    )

    findings = ROOT / "FINDINGS.md"
    if findings.exists():
        st.markdown(findings.read_text(encoding="utf-8"))

    st.markdown("## How to use it")
    st.markdown(
        "- **Single backtest**: one stock, one strategy, with trades, drawdowns, VaR/CVaR, "
        "return distribution, rolling Sharpe and a bootstrap simulation.\n"
        "- **Compare strategies**: all strategies on one stock.\n"
        "- **Compare stocks**: one strategy across several stocks.\n"
        "- **Parameter sensitivity**: picks the best parameters on the in-sample period, then "
        "checks how they do out of sample.\n"
        "- **Strategy study**: every strategy on a basket of stocks, summarised in one table. "
        "This is the mode behind the findings above."
    )

    st.markdown("## Methodology")
    st.markdown(
        "- **Data**: daily prices from Yahoo Finance via `yfinance`, adjusted for splits and dividends.\n"
        "- **No look-ahead**: a signal uses information up to the close of day t and is traded at "
        "the open of day t+1. The overnight move belongs to the old position and the intraday move "
        "to the new one.\n"
        "- **Costs**: a fixed cost in basis points on every unit of position change "
        "(default 10 bps).\n"
        "- **Out-of-sample test**: history is split by date (default 70/30). In *Parameter "
        "sensitivity* the parameters are chosen on the first part only, then evaluated on the second.\n"
        "- **Metrics**: CAGR, volatility, Sharpe and Sortino (excess of the chosen risk-free rate, "
        "Sortino uses downside deviation), maximum drawdown, Calmar, time in market, and historical "
        "one-day VaR and CVaR at 95% and 99%.\n"
        "- **Benchmark**: buying at the first close and holding, with no costs."
    )

    st.markdown("## Limitations")
    st.markdown(
        "- **Survivorship bias**: the default tickers are large companies that exist today, which "
        "flatters buy-and-hold.\n"
        "- **Idle cash earns nothing**: when a strategy is out of the market its cash earns 0%. With "
        "a positive interest rate, low-exposure strategies would look slightly better.\n"
        "- **Simple cost model**: no bid-ask spread modelling, market impact, borrow fees for shorts "
        "or taxes.\n"
        "- **One period, one market regime**: results over a mostly rising US market may not hold "
        "in other periods or markets.\n"
        "- **Indicator warm-up**: strategies stay in cash until their indicators have enough "
        "history (up to 200 days for the slow moving average), while buy-and-hold is invested "
        "from day one.\n"
        "- **Daily bars only**, single-asset positions, full allocation when in the market."
    )
    st.stop()


# ---------------------------------------------------------------------------
# Single backtest
# ---------------------------------------------------------------------------
def load(ticker):
    df = fetch_price_data(ticker, period)
    if df.empty:
        st.error(f"No data for {ticker}. Check the symbol.")
        return None
    return df


if mode == "Single backtest":
    ticker, name = tickers[0], strat_names[0]
    df = load(ticker)
    if df is None:
        st.stop()
    res = backtest(df, name, params, capital, cost_bps, allow_short)
    sd = split_date(res, in_sample)

    st.title(f"{name} on {ticker}")
    st.caption(STRATS[name].__doc__.strip().split("\n")[0])

    m, bh = metrics(res, rf), metrics(res, rf, "bh")
    c = st.columns(6)
    for col, k, f in zip(c, ["CAGR", "Sharpe Ratio", "Sortino Ratio", "Max Drawdown",
                              "Annual Volatility", "Time in Market"],
                         [pct, num, num, pct, pct, pct]):
        col.metric(k, f(m[k]))

    st.plotly_chart(plot_equity(res, name, sd), width="stretch")

    st.markdown("#### Strategy vs buy and hold")
    st.dataframe(fmt_table({name: m, "Buy & Hold": bh}), width="stretch")

    if sd is not None:
        st.markdown("#### First part vs second part of the history")
        st.caption(f"Split at {sd.date()}. With default parameters nothing is fitted, so this mainly "
                   "shows whether results are stable over time. Use *Parameter sensitivity* for a "
                   "proper out-of-sample test.")
        a, b = res.loc[:sd], res.loc[sd:]
        st.dataframe(fmt_table({
            "Strategy, first part": metrics(a, rf), "Buy & Hold, first part": metrics(a, rf, "bh"),
            "Strategy, second part": metrics(b, rf), "Buy & Hold, second part": metrics(b, rf, "bh"),
        }), width="stretch")

    st.markdown("#### Tail risk (historical, one day)")
    v1 = st.columns(4)
    for i, conf in enumerate((0.95, 0.99)):
        v, cv = var_cvar(res["strategy_returns"], conf)
        v1[2 * i].metric(f"VaR {int(conf * 100)}%", pct(v))
        v1[2 * i + 1].metric(f"CVaR {int(conf * 100)}%", pct(cv))

    st.plotly_chart(plot_signals(res), width="stretch")

    with st.expander("More analysis: return distribution, rolling Sharpe, bootstrap simulation"):
        fig, sk, ku = plot_return_distribution(res["strategy_returns"])
        if fig:
            st.plotly_chart(fig, width="stretch")
            st.caption(f"Skewness {sk:.2f}, excess kurtosis {ku:.2f}. Excess kurtosis above 0 "
                       "means extreme days happen more often than a normal distribution implies.")
        st.plotly_chart(plot_rolling_sharpe(res["strategy_returns"]), width="stretch")
        fig, p_profit = plot_monte_carlo(res["strategy_returns"], capital)
        if fig:
            st.plotly_chart(fig, width="stretch")
            st.caption(f"500 one-year paths resampled from the strategy's daily returns. "
                       f"Share of paths ending above the starting capital: {p_profit:.0%}. "
                       "Resampling assumes returns are independent from day to day, which "
                       "ignores volatility clustering.")


# ---------------------------------------------------------------------------
# Compare strategies
# ---------------------------------------------------------------------------
elif mode == "Compare strategies":
    ticker = tickers[0]
    df = load(ticker)
    if df is None or not strat_names:
        st.stop()
    st.title(f"Strategies compared on {ticker}")
    fig, rows = go.Figure(), {}
    first = None
    for i, name in enumerate(strat_names):
        res = backtest(df, name, default_params(name), capital, cost_bps, allow_short)
        first = res if first is None else first
        rows[name] = metrics(res, rf)
        fig.add_trace(go.Scatter(x=res.index, y=res["strategy_equity"], name=name,
                                 line=dict(color=PALETTE[i % len(PALETTE)], width=1.6)))
    fig.add_trace(go.Scatter(x=first.index, y=first["buyhold_equity"], name="Buy & Hold",
                             line=dict(color=INK, width=2.2, dash="dot")))
    rows["Buy & Hold"] = metrics(first, rf, "bh")
    fig.update_layout(height=520, title="Equity curves (default parameters)", **LAYOUT)
    st.plotly_chart(fig, width="stretch")
    order = sorted(rows, key=lambda k: -np.nan_to_num(rows[k]["Sharpe Ratio"], nan=-9))
    st.dataframe(fmt_table({k: rows[k] for k in order}), width="stretch")
    st.caption("Sorted by Sharpe ratio.")


# ---------------------------------------------------------------------------
# Compare stocks
# ---------------------------------------------------------------------------
elif mode == "Compare stocks":
    name = strat_names[0]
    st.title(f"{name} across stocks")
    fig, rows = go.Figure(), {}
    for i, t in enumerate(tickers):
        df = load(t)
        if df is None:
            continue
        res = backtest(df, name, params, capital, cost_bps, allow_short)
        m, bh = metrics(res, rf), metrics(res, rf, "bh")
        rows[t] = {"Strategy CAGR": pct(m["CAGR"]), "B&H CAGR": pct(bh["CAGR"]),
                   "Strategy Sharpe": num(m["Sharpe Ratio"]), "B&H Sharpe": num(bh["Sharpe Ratio"]),
                   "Strategy max DD": pct(m["Max Drawdown"]), "B&H max DD": pct(bh["Max Drawdown"]),
                   "Time in market": pct(m["Time in Market"])}
        fig.add_trace(go.Scatter(x=res.index, y=100 * res["strategy_equity"] / capital, name=t,
                                 line=dict(color=PALETTE[i % len(PALETTE)])))
    if rows:
        fig.update_layout(height=480, title="Strategy equity, start = 100", **LAYOUT)
        st.plotly_chart(fig, width="stretch")
        st.dataframe(pd.DataFrame(rows).T, width="stretch")


# ---------------------------------------------------------------------------
# Parameter sensitivity with a real out-of-sample test
# ---------------------------------------------------------------------------
elif mode == "Parameter sensitivity":
    ticker, name = tickers[0], strat_names[0]
    df = load(ticker)
    spec = STRATS[name].params
    keys = list(spec)
    st.title(f"Parameter sensitivity: {name} on {ticker}")
    st.markdown(
        "Every parameter pair on the grid is backtested. The best pair is chosen using **only the "
        "in-sample period**, then tested on the later out-of-sample period. A big drop from "
        "in-sample to out-of-sample is a sign of overfitting.")
    if df is None or len(keys) < 2:
        st.warning("This strategy needs at least two parameters for a 2D grid.")
        st.stop()

    (k1, (l1, lo1, hi1, d1)), (k2, (l2, lo2, hi2, d2)) = list(spec.items())[:2]
    rest = {k: v[3] for k, v in spec.items() if k not in (k1, k2)}
    g1 = np.linspace(lo1, hi1, 8)
    g2 = np.linspace(lo2, hi2, 8)
    g1 = sorted({int(x) for x in g1}) if isinstance(d1, int) else [round(x, 2) for x in g1]
    g2 = sorted({int(x) for x in g2}) if isinstance(d2, int) else [round(x, 2) for x in g2]

    sd = split_date(df, in_sample)
    grid_is = np.full((len(g2), len(g1)), np.nan)
    grid_oos = np.full_like(grid_is, np.nan)
    bar = st.progress(0.0, text="Running grid")
    for r, v2 in enumerate(g2):
        for c, v1 in enumerate(g1):
            if k1 in ("fast",) and k2 in ("slow",) and v1 >= v2:
                continue  # a fast average must be shorter than the slow one
            res = backtest(df, name, {k1: v1, k2: v2, **rest}, capital, cost_bps, allow_short)
            grid_is[r, c] = metrics(res.loc[:sd], rf)["Sharpe Ratio"]
            grid_oos[r, c] = metrics(res.loc[sd:], rf)["Sharpe Ratio"]
        bar.progress((r + 1) / len(g2), text="Running grid")
    bar.empty()

    heat = go.Figure(go.Heatmap(z=grid_is, x=[str(v) for v in g1], y=[str(v) for v in g2],
                                colorscale="RdBu", zmid=0, colorbar=dict(title="Sharpe")))
    heat.update_layout(height=480, title=f"In-sample Sharpe ratio (until {sd.date()})",
                       xaxis_title=l1, yaxis_title=l2, **LAYOUT)
    st.plotly_chart(heat, width="stretch")

    if np.isfinite(grid_is).any():
        r, c = np.unravel_index(np.nanargmax(grid_is), grid_is.shape)
        best_is, best_oos = grid_is[r, c], grid_oos[r, c]
        bh_oos = metrics(run_vectorised(STRATEGY_REGISTRY["buy_and_hold"]().generate_signals(df.copy()),
                                        capital, 0).loc[sd:], rf, "bh")["Sharpe Ratio"]
        a, b, cc = st.columns(3)
        a.metric(f"Best in-sample pair ({l1} / {l2})", f"{g1[c]} / {g2[r]}")
        b.metric("Sharpe in sample", num(best_is))
        cc.metric("Same parameters out of sample", num(best_oos),
                  delta=num(best_oos - best_is) if np.isfinite(best_oos) else None)
        st.caption(f"Buy-and-hold Sharpe in the out-of-sample period: {num(bh_oos)}. "
                   f"In-sample Sharpe across the grid ranges from {np.nanmin(grid_is):.2f} to "
                   f"{np.nanmax(grid_is):.2f}.")


# ---------------------------------------------------------------------------
# Strategy study: all strategies x all tickers
# ---------------------------------------------------------------------------
elif mode == "Strategy study":
    st.title("Strategy study")
    st.markdown(
        f"Every strategy with its default parameters on each ticker, {period} of history, "
        f"{cost_bps} bps per trade, {'long/short' if allow_short else 'long only'}. "
        "The table shows averages across tickers. *Beats B&H* counts on how many tickers the "
        "strategy had a higher Sharpe ratio than buy-and-hold over the same dates.")

    per_ticker, bh_rows, data = [], {}, {}
    bar = st.progress(0.0, text="Downloading and backtesting")
    for i, t in enumerate(tickers):
        df = fetch_price_data(t, period)
        if df.empty:
            continue
        data[t] = df
        for name in strat_names:
            res0 = backtest(df, name, default_params(name), capital, 0, allow_short)
            res = backtest(df, name, default_params(name), capital, cost_bps, allow_short)
            sd = split_date(res, in_sample)
            m, bh = metrics(res, rf), metrics(res, rf, "bh")
            m_oos, bh_oos = metrics(res.loc[sd:], rf), metrics(res.loc[sd:], rf, "bh")
            per_ticker.append({
                "Ticker": t, "Strategy": name,
                "CAGR": m["CAGR"], "CAGR before costs": metrics(res0, rf)["CAGR"],
                "Sharpe": m["Sharpe Ratio"], "B&H Sharpe": bh["Sharpe Ratio"],
                "Max DD": m["Max Drawdown"], "B&H Max DD": bh["Max Drawdown"],
                "Time in market": m["Time in Market"],
                "Position changes per year": res["position"].diff().abs().gt(0).sum() / (len(res) / 252),
                "Beats B&H": m["Sharpe Ratio"] > bh["Sharpe Ratio"],
                "Beats B&H (second part)": m_oos["Sharpe Ratio"] > bh_oos["Sharpe Ratio"],
            })
            bh_rows[t] = bh
        bar.progress((i + 1) / len(tickers), text="Downloading and backtesting")
    bar.empty()

    if not per_ticker:
        st.error("No data downloaded.")
        st.stop()

    long_df = pd.DataFrame(per_ticker)
    n = long_df["Ticker"].nunique()
    summary = long_df.groupby("Strategy").agg(
        CAGR=("CAGR", "mean"), CAGR_gross=("CAGR before costs", "mean"),
        Sharpe=("Sharpe", "mean"), MaxDD=("Max DD", "mean"),
        TiM=("Time in market", "mean"), TPY=("Position changes per year", "mean"),
        Beats=("Beats B&H", "sum"), Beats2=("Beats B&H (second part)", "sum"),
    ).sort_values("Sharpe", ascending=False)
    bh_df = pd.DataFrame(bh_rows).T
    bh_line = pd.DataFrame({"CAGR": [bh_df["CAGR"].mean()], "CAGR_gross": [bh_df["CAGR"].mean()],
                            "Sharpe": [bh_df["Sharpe Ratio"].mean()], "MaxDD": [bh_df["Max Drawdown"].mean()],
                            "TiM": [1.0], "TPY": [0.0], "Beats": [np.nan], "Beats2": [np.nan]},
                           index=["Buy & Hold"])
    table = pd.concat([summary, bh_line])

    shown = pd.DataFrame({
        "Avg CAGR": table["CAGR"].apply(pct),
        "Avg CAGR before costs": table["CAGR_gross"].apply(pct),
        "Avg Sharpe": table["Sharpe"].apply(num),
        "Avg max drawdown": table["MaxDD"].apply(pct),
        "Time in market": table["TiM"].apply(pct),
        "Position changes per year": table["TPY"].apply(lambda x: f"{x:.1f}"),
        f"Beats B&H (of {n})": table["Beats"].apply(lambda x: "" if pd.isna(x) else f"{int(x)}"),
        f"Beats B&H, second part (of {n})": table["Beats2"].apply(lambda x: "" if pd.isna(x) else f"{int(x)}"),
    })
    st.dataframe(shown, width="stretch")

    total = len(long_df)
    wins = int(long_df["Beats B&H"].sum())
    st.markdown(f"**{wins} of {total}** strategy and ticker combinations had a higher Sharpe ratio "
                f"than buy-and-hold. Average buy-and-hold Sharpe: {num(bh_df['Sharpe Ratio'].mean())}.")

    with st.expander("Full results by ticker"):
        out = long_df.copy()
        for c in ["CAGR", "CAGR before costs", "Max DD", "B&H Max DD", "Time in market"]:
            out[c] = out[c].apply(pct)
        for c in ["Sharpe", "B&H Sharpe", "Position changes per year"]:
            out[c] = out[c].apply(num)
        st.dataframe(out, width="stretch", hide_index=True)
