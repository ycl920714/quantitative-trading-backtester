"""
Quantitative Trading Strategy Backtester
==========================================
A research-grade backtesting tool for evaluating trading strategies against
historical price data, with risk-adjusted performance metrics, benchmark
comparison, parameter sensitivity analysis, and out-of-sample validation.
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ----------------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="QTB — Quantitative Trading Backtester",
    page_icon="📈",
    layout="wide",
)

# ----------------------------------------------------------------------------
# DESIGN SYSTEM  — Deep navy fintech terminal aesthetic
# Palette:  Ink #0A0E1A  |  Surface #111827  |  Panel #1A2332
#           Mint #00D4A4  |  Amber #F97316   |  Muted #6B7A94
# Type:     Space Grotesk (display)  |  Inter (body)  |  JetBrains Mono (data)
# ----------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;600&display=swap');

/* BASE */
.stApp {
    background-color: #0A0E1A;
    color: #E8EDF4;
    font-family: 'Inter', sans-serif;
}
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1400px;
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background-color: #0F1623 !important;
    border-right: 1px solid #1A2332;
}
section[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.5rem;
}
section[data-testid="stSidebar"] .stMarkdown p {
    color: #6B7A94;
    font-size: 0.78rem;
    line-height: 1.5;
}

/* HEADINGS */
h1, h2, h3, h4 {
    font-family: 'Space Grotesk', sans-serif !important;
    color: #E8EDF4 !important;
}
h1 { font-size: 2.6rem !important; font-weight: 700 !important; letter-spacing: -0.03em !important; line-height: 1.1 !important; }
h2 { font-size: 1.5rem !important; font-weight: 600 !important; letter-spacing: -0.01em !important; margin-top: 2rem !important; }
h3 { font-size: 1.1rem !important; font-weight: 500 !important; color: #A0AEBF !important; }

/* METRIC CARDS — the signature element: data-terminal readouts */
[data-testid="metric-container"] {
    background: #111827;
    border: 1px solid #1A2332;
    border-radius: 12px;
    padding: 1.1rem 1.3rem !important;
    transition: border-color 0.25s ease, box-shadow 0.25s ease;
    position: relative;
    overflow: hidden;
}
[data-testid="metric-container"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #00D4A4, transparent);
    opacity: 0;
    transition: opacity 0.25s;
}
[data-testid="metric-container"]:hover {
    border-color: #00D4A4;
    box-shadow: 0 0 20px rgba(0, 212, 164, 0.08);
}
[data-testid="metric-container"]:hover::before { opacity: 1; }
[data-testid="metric-container"] label,
[data-testid="metric-container"] [data-testid="stMetricLabel"] p {
    color: #6B7A94 !important;
    font-size: 0.68rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #00D4A4 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.55rem !important;
    font-weight: 600 !important;
    letter-spacing: -0.02em !important;
}

/* BUTTONS */
.stButton > button {
    background: linear-gradient(135deg, #00D4A4 0%, #00B388 100%) !important;
    color: #0A0E1A !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.04em !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.65rem 1.5rem !important;
    transition: opacity 0.18s, transform 0.12s !important;
    box-shadow: 0 4px 14px rgba(0, 212, 164, 0.25) !important;
}
.stButton > button:hover {
    opacity: 0.88 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(0, 212, 164, 0.35) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* FORM INPUTS */
.stTextInput > div > div > input,
.stNumberInput > div > div > input {
    background-color: #1A2332 !important;
    border: 1px solid #2D3A50 !important;
    border-radius: 8px !important;
    color: #E8EDF4 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.9rem !important;
}
.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus {
    border-color: #00D4A4 !important;
    box-shadow: 0 0 0 2px rgba(0,212,164,0.15) !important;
}
.stSelectbox > div > div {
    background-color: #1A2332 !important;
    border: 1px solid #2D3A50 !important;
    border-radius: 8px !important;
    color: #E8EDF4 !important;
}

/* SLIDER */
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background-color: #00D4A4 !important;
    border-color: #00D4A4 !important;
}
.stSlider [data-baseweb="slider"] div[class*="Track"] div:first-child {
    background-color: #00D4A4 !important;
}

/* RADIO */
.stRadio > div { gap: 0.4rem !important; }
.stRadio label { color: #C4CEDB !important; font-size: 0.88rem !important; }

/* LABELS */
.stSelectbox label, .stTextInput label, .stNumberInput label,
.stSlider label, .stRadio label[data-baseweb="form-control-label"] > span:first-child,
p[class*="label"] {
    color: #6B7A94 !important;
    font-size: 0.72rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    font-weight: 500 !important;
}

/* DATAFRAME */
.stDataFrame {
    border: 1px solid #1A2332 !important;
    border-radius: 10px !important;
    overflow: hidden;
}
[data-testid="stDataFrame"] th {
    background-color: #111827 !important;
    color: #6B7A94 !important;
    font-size: 0.72rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    border-bottom: 1px solid #1A2332 !important;
}
[data-testid="stDataFrame"] td {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.85rem !important;
    color: #C4CEDB !important;
    border-bottom: 1px solid #131B28 !important;
}

/* ALERT / INFO BOXES */
[data-testid="stAlert"] {
    background-color: #111827 !important;
    border-radius: 10px !important;
    border: 1px solid #1A2332 !important;
}
[data-testid="stAlert"][kind="info"] {
    border-left: 3px solid #3B82F6 !important;
}
[data-testid="stAlert"][kind="success"] {
    background-color: #0D1F1A !important;
    border: 1px solid #1A3A2E !important;
    border-left: 3px solid #00D4A4 !important;
}
[data-testid="stAlert"][kind="warning"] {
    border-left: 3px solid #F97316 !important;
}

/* EXPANDER */
[data-testid="stExpander"] {
    background-color: #111827 !important;
    border: 1px solid #1A2332 !important;
    border-radius: 10px !important;
}
[data-testid="stExpander"] summary {
    color: #A0AEBF !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
}

/* PROGRESS BAR */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #00D4A4, #00B388) !important;
    border-radius: 4px !important;
}
.stProgress > div > div > div {
    background-color: #1A2332 !important;
    border-radius: 4px !important;
}

/* DIVIDER */
hr { border-color: #1A2332 !important; }

/* CAPTION */
.stCaptionContainer p, caption {
    color: #6B7A94 !important;
    font-size: 0.78rem !important;
}

/* SCROLLBAR */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0A0E1A; }
::-webkit-scrollbar-thumb { background: #1A2332; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #2D3A50; }

/* SIDEBAR SECTION LABELS */
.sidebar-label {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.65rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #00D4A4;
    margin: 1.2rem 0 0.5rem 0;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid #1A2332;
}

/* HERO section */
.hero-badge {
    display: inline-block;
    background: #0D2A22;
    color: #00D4A4;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    border: 1px solid #1A3A2E;
    margin-bottom: 1rem;
}
.hero-subtitle {
    color: #6B7A94;
    font-size: 1rem;
    font-family: 'Inter', sans-serif;
    line-height: 1.6;
    max-width: 680px;
    margin-bottom: 2rem;
}
.mode-card {
    background: #111827;
    border: 1px solid #1A2332;
    border-radius: 14px;
    padding: 1.4rem 1.5rem;
    transition: border-color 0.2s, transform 0.15s;
    height: 100%;
}
.mode-card:hover {
    border-color: #2D3A50;
    transform: translateY(-2px);
}
.mode-card-icon { font-size: 1.6rem; margin-bottom: 0.6rem; }
.mode-card-title {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    font-size: 1rem;
    color: #E8EDF4;
    margin-bottom: 0.4rem;
}
.mode-card-desc {
    font-size: 0.82rem;
    color: #6B7A94;
    line-height: 1.5;
}
.section-divider {
    border: none;
    border-top: 1px solid #1A2332;
    margin: 2rem 0;
}
.ticker-tag {
    display: inline-block;
    background: #1A2332;
    color: #00D4A4;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    font-weight: 600;
    padding: 0.25rem 0.7rem;
    border-radius: 6px;
    border: 1px solid #2D3A50;
    margin-right: 0.4rem;
    margin-bottom: 0.3rem;
}
.best-badge {
    display: inline-block;
    background: #0D2A22;
    color: #00D4A4;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    font-weight: 600;
    padding: 0.2rem 0.6rem;
    border-radius: 6px;
    border: 1px solid #1A3A2E;
    margin-left: 0.5rem;
    vertical-align: middle;
}
</style>
"""


# ----------------------------------------------------------------------------
# DATA LAYER
# ----------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_price_data(ticker: str, period: str) -> pd.DataFrame:
    df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna()


# ----------------------------------------------------------------------------
# STRATEGY LAYER
# ----------------------------------------------------------------------------
def strategy_sma_crossover(df, fast=20, slow=50):
    out = df.copy()
    out["fast_ma"] = out["Close"].rolling(fast).mean()
    out["slow_ma"] = out["Close"].rolling(slow).mean()
    out["signal"] = np.where(out["fast_ma"] > out["slow_ma"], 1, 0)
    out["signal"] = out["signal"].shift(1).fillna(0)
    return out


def strategy_rsi(df, period=14, oversold=30, overbought=70):
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


def strategy_bollinger(df, window=20, num_std=2.0):
    out = df.copy()
    out["mid"] = out["Close"].rolling(window).mean()
    std = out["Close"].rolling(window).std()
    out["upper"] = out["mid"] + num_std * std
    out["lower"] = out["mid"] - num_std * std
    signal = pd.Series(0, index=out.index, dtype=float)
    position = 0
    close, lower, upper, mid = out["Close"].values, out["lower"].values, out["upper"].values, out["mid"].values
    for i in range(len(out)):
        if np.isnan(lower[i]):
            signal.iloc[i] = position; continue
        if close[i] < lower[i]:
            position = 1
        elif close[i] > mid[i]:
            position = 0
        signal.iloc[i] = position
    out["signal"] = signal.shift(1).fillna(0)
    return out


def strategy_momentum(df, lookback=60):
    out = df.copy()
    out["momentum"] = out["Close"].pct_change(lookback)
    out["signal"] = np.where(out["momentum"] > 0, 1, 0)
    out["signal"] = out["signal"].shift(1).fillna(0)
    return out


STRATEGIES = {
    "📈 SMA Crossover": {
        "fn": strategy_sma_crossover,
        "short": "Trend Following",
        "params": {"fast": ("Fast MA", 5, 100, 20), "slow": ("Slow MA", 10, 250, 50)},
    },
    "🔄 RSI Mean Reversion": {
        "fn": strategy_rsi,
        "short": "Mean Reversion",
        "params": {
            "period": ("RSI Period", 5, 30, 14),
            "oversold": ("Oversold", 10, 40, 30),
            "overbought": ("Overbought", 60, 90, 70),
        },
    },
    "📊 Bollinger Bands": {
        "fn": strategy_bollinger,
        "short": "Mean Reversion",
        "params": {"window": ("Window", 10, 60, 20), "num_std": ("Std Dev", 1.0, 3.0, 2.0)},
    },
    "🚀 Momentum": {
        "fn": strategy_momentum,
        "short": "Momentum",
        "params": {"lookback": ("Lookback (days)", 10, 250, 60)},
    },
}

STRATEGY_DESCRIPTIONS = {
    "📈 SMA Crossover": "Goes long when the fast moving average crosses above the slow moving average "
        "(a classic trend signal), and exits when it crosses back below.",
    "🔄 RSI Mean Reversion": "Buys when RSI falls below the oversold threshold — the asset has fallen "
        "'too far, too fast' — and exits when RSI climbs above overbought.",
    "📊 Bollinger Bands": "Enters long when price dips below the lower band (statistically cheap) "
        "and exits once price reverts to the rolling mean.",
    "🚀 Momentum": "Goes long when trailing N-day returns are positive, betting recent winners "
        "keep winning over the lookback horizon.",
}


# ----------------------------------------------------------------------------
# BACKTEST ENGINE
# ----------------------------------------------------------------------------
def run_backtest(df, initial_capital, cost_bps=5.0):
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
def compute_metrics(returns, equity, periods=252):
    returns = returns.dropna()
    if len(returns) < 2 or equity.iloc[0] == 0:
        return {k: np.nan for k in ["Total Return", "CAGR", "Annual Volatility",
                                     "Sharpe Ratio", "Sortino Ratio", "Max Drawdown",
                                     "Calmar Ratio", "Win Rate"]}
    total_ret = equity.iloc[-1] / equity.iloc[0] - 1
    n_years = len(returns) / periods
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / n_years) - 1 if n_years > 0 else np.nan
    ann_vol = returns.std() * np.sqrt(periods)
    ann_ret = returns.mean() * periods
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    down = returns[returns < 0]
    down_vol = down.std() * np.sqrt(periods) if len(down) > 0 else np.nan
    sortino = ann_ret / down_vol if down_vol and down_vol > 0 else np.nan
    running_max = equity.cummax()
    dd = (equity - running_max) / running_max
    max_dd = dd.min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else np.nan
    win_rate = (returns > 0).sum() / (returns != 0).sum() if (returns != 0).sum() > 0 else np.nan
    return {"Total Return": total_ret, "CAGR": cagr, "Annual Volatility": ann_vol,
            "Sharpe Ratio": sharpe, "Sortino Ratio": sortino, "Max Drawdown": max_dd,
            "Calmar Ratio": calmar, "Win Rate": win_rate}


def fp(x):
    return "—" if pd.isna(x) else f"{x * 100:.2f}%"


def fn(x):
    return "—" if pd.isna(x) else f"{x:.2f}"


# ----------------------------------------------------------------------------
# CHARTS  — dark-mode Plotly theme to match the CSS
# ----------------------------------------------------------------------------
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#0F1623",
    font=dict(family="Inter, sans-serif", color="#A0AEBF", size=12),
    xaxis=dict(gridcolor="#1A2332", linecolor="#1A2332", zerolinecolor="#1A2332"),
    yaxis=dict(gridcolor="#1A2332", linecolor="#1A2332", zerolinecolor="#1A2332"),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#1A2332", borderwidth=1,
                orientation="h", y=1.08),
    hovermode="x unified",
    hoverlabel=dict(bgcolor="#111827", bordercolor="#2D3A50",
                    font=dict(family="JetBrains Mono, monospace", size=12, color="#E8EDF4")),
    margin=dict(t=60, b=40, l=60, r=20),
)


def plot_equity_curve(result, strategy_name, split_dt=None):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                         row_heights=[0.68, 0.32], vertical_spacing=0.04,
                         subplot_titles=("Equity Curve", "Drawdown (%)"))
    for ax in fig.layout:
        if ax.startswith("xaxis") or ax.startswith("yaxis"):
            fig.layout[ax].update(gridcolor="#1A2332", linecolor="#1A2332")

    fig.add_trace(go.Scatter(x=result.index, y=result["strategy_equity"],
                              name=strategy_name,
                              line=dict(color="#00D4A4", width=2.2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=result.index, y=result["buyhold_equity"],
                              name="Buy & Hold",
                              line=dict(color="#6B7A94", width=1.5, dash="dot")), row=1, col=1)

    running_max = result["strategy_equity"].cummax()
    drawdown = (result["strategy_equity"] - running_max) / running_max * 100
    fig.add_trace(go.Scatter(x=result.index, y=drawdown,
                              fill="tozeroy", name="Drawdown",
                              line=dict(color="#F97316", width=1),
                              fillcolor="rgba(249,115,22,0.12)"), row=2, col=1)

    if split_dt is not None:
        for row in [1, 2]:
            fig.add_vline(x=split_dt, line_dash="dot", line_color="#2D3A50",
                           annotation_text="OOS Split", annotation_font_color="#6B7A94",
                           annotation_font_size=10, row=row, col=1)

    fig.update_layout(height=560, **PLOT_LAYOUT)
    fig.update_annotations(font_color="#6B7A94", font_size=11)
    fig.update_yaxes(title_text="Value ($)", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown %", row=2, col=1)
    return fig


def plot_price_signals(result, strategy_name):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=result.index, y=result["Close"], name="Price",
                              line=dict(color="#E8EDF4", width=1.4)))
    if "fast_ma" in result.columns:
        fig.add_trace(go.Scatter(x=result.index, y=result["fast_ma"], name="Fast MA",
                                  line=dict(color="#F97316", width=1.2)))
        fig.add_trace(go.Scatter(x=result.index, y=result["slow_ma"], name="Slow MA",
                                  line=dict(color="#3B82F6", width=1.2)))
    if "upper" in result.columns:
        fig.add_trace(go.Scatter(x=result.index, y=result["upper"], name="Upper Band",
                                  line=dict(color="#2D3A50", width=1, dash="dot")))
        fig.add_trace(go.Scatter(x=result.index, y=result["lower"], name="Lower Band",
                                  line=dict(color="#2D3A50", width=1, dash="dot"),
                                  fill="tonexty", fillcolor="rgba(45,58,80,0.2)"))
        fig.add_trace(go.Scatter(x=result.index, y=result["mid"], name="Mid Band",
                                  line=dict(color="#F97316", width=1, dash="dash")))
    buys = result[result["signal"].diff() == 1]
    sells = result[result["signal"].diff() == -1]
    fig.add_trace(go.Scatter(x=buys.index, y=buys["Close"], mode="markers", name="Buy",
                              marker=dict(symbol="triangle-up", color="#00D4A4", size=9,
                                          line=dict(width=0))))
    fig.add_trace(go.Scatter(x=sells.index, y=sells["Close"], mode="markers", name="Sell",
                              marker=dict(symbol="triangle-down", color="#F97316", size=9,
                                          line=dict(width=0))))
    fig.update_layout(height=440, title="Entry / Exit Signals", **PLOT_LAYOUT)
    return fig


# ----------------------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------------------
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<p class="sidebar-label">Mode</p>', unsafe_allow_html=True)
    mode = st.radio("", ["Single", "Multi-Strategy", "Multi-Stock", "Parameter Sensitivity"],
                    label_visibility="collapsed")

    st.markdown('<p class="sidebar-label">Universe</p>', unsafe_allow_html=True)
    if mode in ("Single", "Multi-Strategy", "Parameter Sensitivity"):
        ticker_input = st.text_input("Ticker", value="AAPL", label_visibility="collapsed",
                                      placeholder="Ticker, e.g. AAPL")
        tickers = [ticker_input.strip().upper()]
    else:
        tickers_input = st.text_input("Tickers (comma-separated)", value="AAPL, MSFT, GOOGL",
                                       label_visibility="collapsed",
                                       placeholder="e.g. AAPL, MSFT, TSLA")
        tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

    st.markdown('<p class="sidebar-label">Strategy</p>', unsafe_allow_html=True)
    if mode in ("Single", "Multi-Stock", "Parameter Sensitivity"):
        strategy_name = st.selectbox("Strategy", list(STRATEGIES.keys()),
                                      label_visibility="collapsed")
        strategy_names = [strategy_name]
    else:
        strategy_names = st.multiselect("Strategies", list(STRATEGIES.keys()),
                                         default=list(STRATEGIES.keys())[:2],
                                         label_visibility="collapsed")

    st.markdown('<p class="sidebar-label">Backtest Settings</p>', unsafe_allow_html=True)
    period = st.selectbox("Period", ["1y", "2y", "5y", "10y", "max"],
                           index=2, label_visibility="collapsed")
    initial_capital = st.number_input("Capital ($)", min_value=100, value=10000, step=100)
    cost_bps = st.slider("Transaction cost (bps)", 0, 50, 5,
                          help="Round-trip cost per trade in basis points.")
    oos_split = st.slider("In-sample split (%)", 50, 95, 70,
                           help="% of data used in-sample. The rest tests out-of-sample robustness.")

    custom_params = {}
    if mode in ("Single", "Multi-Stock"):
        spec = STRATEGIES[strategy_names[0]]
        if spec["params"]:
            st.markdown('<p class="sidebar-label">Parameters</p>', unsafe_allow_html=True)
            for pkey, (label, lo, hi, default) in spec["params"].items():
                if isinstance(default, int):
                    custom_params[pkey] = st.slider(label, lo, hi, default)
                else:
                    custom_params[pkey] = st.slider(label, float(lo), float(hi),
                                                      float(default), step=0.1)

    st.markdown('<div style="height:1rem"></div>', unsafe_allow_html=True)
    run_btn = st.button("▶  Run Backtest", use_container_width=True, type="primary")

    st.markdown('<div style="height:0.5rem"></div>', unsafe_allow_html=True)
    st.caption(
        "Educational tool only. Past performance does not indicate future results. "
        "Long-only strategies. Adjusted close prices via yfinance."
    )


# ----------------------------------------------------------------------------
# HERO (pre-run state)
# ----------------------------------------------------------------------------
if not run_btn:
    st.markdown("""
    <div class="hero-badge">Portfolio Project · Finance & Analytics</div>
    <h1>Quantitative<br>Trading Backtester</h1>
    <p class="hero-subtitle">
        Research-grade backtesting with risk-adjusted performance metrics,
        buy &amp; hold benchmarking, transaction cost modelling,
        and out-of-sample validation.
    </p>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4, gap="small")
    cards = [
        ("🎯", "Single", "One stock, one strategy. Full equity curve with entry/exit signals, drawdown, and six risk metrics."),
        ("📊", "Multi-Strategy", "Compare all four strategies on one stock side-by-side. Sharpe-ranked summary table."),
        ("🌍", "Multi-Stock", "One strategy across many tickers. Tests whether the edge is stock-specific or broadly robust."),
        ("🔬", "Sensitivity", "Parameter grid heatmap. Flags overfitting when performance only appears at one exact setting."),
    ]
    for col, (icon, title, desc) in zip([c1, c2, c3, c4], cards):
        with col:
            st.markdown(f"""
            <div class="mode-card">
                <div class="mode-card-icon">{icon}</div>
                <div class="mode-card-title">{title}</div>
                <div class="mode-card-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown("##### Built-in strategies")
    r1, r2, r3, r4 = st.columns(4, gap="small")
    strat_info = [
        ("📈", "SMA Crossover", "Trend", "Fast/slow moving average crossover — classic trend signal"),
        ("🔄", "RSI", "Mean Reversion", "Overbought/oversold oscillator — buy dips, sell rips"),
        ("📊", "Bollinger Bands", "Mean Reversion", "Statistical range — enter below, exit at mean"),
        ("🚀", "Momentum", "Momentum", "Trailing N-day returns — buy winners, ignore losers"),
    ]
    for col, (icon, name, style, desc) in zip([r1, r2, r3, r4], strat_info):
        with col:
            st.markdown(f"""
            <div class="mode-card" style="padding:1rem 1.2rem">
                <div style="font-size:1.2rem;margin-bottom:0.4rem">{icon} <strong style="color:#E8EDF4;font-family:'Space Grotesk',sans-serif">{name}</strong></div>
                <div style="font-size:0.68rem;text-transform:uppercase;letter-spacing:0.1em;color:#00D4A4;margin-bottom:0.4rem">{style}</div>
                <div class="mode-card-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
    st.stop()


# ----------------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------------
def get_data(ticker):
    df = fetch_price_data(ticker, period)
    if df.empty:
        st.error(f"No data found for **{ticker}**. Check the symbol and try again.")
        return None
    return df


def split_index(df):
    n = len(df)
    i = int(n * oos_split / 100)
    return df.index[i] if 0 < i < n else None


def render_metrics_row(m):
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Return", fp(m["Total Return"]))
    c2.metric("CAGR", fp(m["CAGR"]))
    c3.metric("Sharpe Ratio", fn(m["Sharpe Ratio"]))
    c4.metric("Sortino Ratio", fn(m["Sortino Ratio"]))
    c5.metric("Max Drawdown", fp(m["Max Drawdown"]))
    c6.metric("Win Rate", fp(m["Win Rate"]))


def comparison_table(rows_dict):
    comp = pd.DataFrame(rows_dict).T
    display = comp.copy()
    for col in ["Total Return", "CAGR", "Annual Volatility", "Max Drawdown", "Win Rate"]:
        display[col] = display[col].apply(fp)
    for col in ["Sharpe Ratio", "Sortino Ratio", "Calmar Ratio"]:
        display[col] = display[col].apply(fn)
    return display


# ============================================================================
# MODES
# ============================================================================

# ---------- SINGLE ----------
if mode == "Single":
    ticker = tickers[0]
    df = get_data(ticker)
    if df is not None:
        spec = STRATEGIES[strategy_names[0]]
        sig_df = spec["fn"](df, **custom_params)
        result = run_backtest(sig_df, initial_capital, cost_bps)
        split_dt = split_index(result)

        strat_m = compute_metrics(result["strategy_returns"], result["strategy_equity"])
        bh_m = compute_metrics(result["returns"], result["buyhold_equity"])

        if split_dt is not None:
            is_m = compute_metrics(result.loc[:split_dt, "strategy_returns"],
                                    result.loc[:split_dt, "strategy_equity"])
            oos_m = compute_metrics(result.loc[split_dt:, "strategy_returns"],
                                     result.loc[split_dt:, "strategy_equity"])
        else:
            is_m = oos_m = None

        st.markdown(
            f'<span class="ticker-tag">{ticker}</span>'
            f'<span class="ticker-tag">{strategy_names[0]}</span>'
            f'<span class="ticker-tag">{period}</span>',
            unsafe_allow_html=True,
        )
        st.markdown('<div style="height:0.8rem"></div>', unsafe_allow_html=True)
        render_metrics_row(strat_m)
        st.markdown('<div style="height:1rem"></div>', unsafe_allow_html=True)

        st.plotly_chart(plot_equity_curve(result, strategy_names[0], split_dt),
                        use_container_width=True)
        st.plotly_chart(plot_price_signals(result, strategy_names[0]),
                        use_container_width=True)

        st.markdown("### Strategy vs Buy & Hold")
        st.dataframe(comparison_table({"Strategy": strat_m, "Buy & Hold": bh_m}),
                     use_container_width=True)

        if is_m is not None:
            st.markdown("### In-Sample vs Out-of-Sample")
            st.caption(
                f"In-sample: first {oos_split}% of data (until {split_dt.date()}). "
                "Out-of-sample is held back entirely during strategy design — "
                "a large drop in Sharpe ratio between the two windows signals overfitting."
            )
            st.dataframe(comparison_table({"In-Sample": is_m, "Out-of-Sample": oos_m}),
                         use_container_width=True)

        with st.expander("How this strategy works"):
            st.write(STRATEGY_DESCRIPTIONS[strategy_names[0]])


# ---------- MULTI-STRATEGY ----------
elif mode == "Multi-Strategy":
    ticker = tickers[0]
    df = get_data(ticker)
    if df is not None and strategy_names:
        st.markdown(
            f'<span class="ticker-tag">{ticker}</span>'
            f'<span class="ticker-tag">All Strategies</span>'
            f'<span class="ticker-tag">{period}</span>',
            unsafe_allow_html=True,
        )
        st.markdown('<div style="height:0.8rem"></div>', unsafe_allow_html=True)

        fig = go.Figure()
        bh_r = initial_capital * (1 + df["Close"].pct_change().fillna(0)).cumprod()
        fig.add_trace(go.Scatter(x=df.index, y=bh_r, name="Buy & Hold",
                                  line=dict(color="#6B7A94", width=1.4, dash="dot")))

        colors = ["#00D4A4", "#3B82F6", "#F97316", "#A78BFA"]
        metrics_rows = {}
        for i, name in enumerate(strategy_names):
            spec = STRATEGIES[name]
            default_p = {k: v[3] for k, v in spec["params"].items()}
            sig_df = spec["fn"](df, **default_p)
            result = run_backtest(sig_df, initial_capital, cost_bps)
            m = compute_metrics(result["strategy_returns"], result["strategy_equity"])
            metrics_rows[name] = m
            fig.add_trace(go.Scatter(x=result.index, y=result["strategy_equity"], name=name,
                                      line=dict(color=colors[i % len(colors)], width=2)))

        fig.update_layout(height=520, title="Equity Curves", **PLOT_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Performance Comparison")
        best = max(metrics_rows, key=lambda k: metrics_rows[k].get("Sharpe Ratio", -np.inf))
        st.markdown(
            f'Best risk-adjusted performer: <span class="best-badge">🏆 {best}</span>',
            unsafe_allow_html=True,
        )
        st.markdown('<div style="height:0.5rem"></div>', unsafe_allow_html=True)
        st.dataframe(comparison_table(metrics_rows), use_container_width=True)


# ---------- MULTI-STOCK ----------
elif mode == "Multi-Stock":
    spec = STRATEGIES[strategy_names[0]]
    st.markdown(
        f'<span class="ticker-tag">{strategy_names[0]}</span>'
        + "".join(f'<span class="ticker-tag">{t}</span>' for t in tickers)
        + f'<span class="ticker-tag">{period}</span>',
        unsafe_allow_html=True,
    )
    st.markdown('<div style="height:0.8rem"></div>', unsafe_allow_html=True)

    fig = go.Figure()
    colors = ["#00D4A4", "#3B82F6", "#F97316", "#A78BFA", "#F43F5E", "#FBBF24"]
    metrics_rows = {}

    for i, ticker in enumerate(tickers):
        df = get_data(ticker)
        if df is None:
            continue
        sig_df = spec["fn"](df, **custom_params)
        result = run_backtest(sig_df, initial_capital, cost_bps)
        m = compute_metrics(result["strategy_returns"], result["strategy_equity"])
        metrics_rows[ticker] = m
        norm = 100 * result["strategy_equity"] / result["strategy_equity"].iloc[0]
        fig.add_trace(go.Scatter(x=result.index, y=norm, name=ticker,
                                  line=dict(color=colors[i % len(colors)], width=2)))

    if metrics_rows:
        fig.update_layout(height=520, title="Normalised Equity (Base = 100)", **PLOT_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Performance by Stock")
        best = max(metrics_rows, key=lambda k: metrics_rows[k].get("Sharpe Ratio", -np.inf))
        st.markdown(
            f'Strategy works best on: <span class="best-badge">🏆 {best}</span>',
            unsafe_allow_html=True,
        )
        st.markdown('<div style="height:0.5rem"></div>', unsafe_allow_html=True)
        st.dataframe(comparison_table(metrics_rows), use_container_width=True)


# ---------- PARAMETER SENSITIVITY ----------
elif mode == "Parameter Sensitivity":
    ticker = tickers[0]
    df = get_data(ticker)
    spec = STRATEGIES[strategy_names[0]]
    param_keys = list(spec["params"].keys())

    st.markdown(
        f'<span class="ticker-tag">{ticker}</span>'
        f'<span class="ticker-tag">Sensitivity Analysis</span>'
        f'<span class="ticker-tag">{strategy_names[0]}</span>',
        unsafe_allow_html=True,
    )
    st.markdown('<div style="height:0.5rem"></div>', unsafe_allow_html=True)
    st.caption(
        "A good strategy should work across a broad range of parameter values, not just one "
        "exact combination. If the Sharpe ratio is high only at a single point — surrounded by poor "
        "performance everywhere else — that's a strong overfitting warning."
    )

    if df is not None and len(param_keys) >= 2:
        p1_key, p2_key = param_keys[0], param_keys[1]
        p1_label, p1_lo, p1_hi, p1_def = spec["params"][p1_key]
        p2_label, p2_lo, p2_hi, p2_def = spec["params"][p2_key]
        n = 8
        p1_vals = np.linspace(p1_lo, p1_hi, n)
        p2_vals = np.linspace(p2_lo, p2_hi, n)
        if isinstance(p1_def, int):
            p1_vals = sorted(set(int(v) for v in p1_vals))
        if isinstance(p2_def, int):
            p2_vals = sorted(set(int(v) for v in p2_vals))
        other_p = {k: v[3] for k, v in spec["params"].items()
                   if k not in (p1_key, p2_key)}

        grid = np.full((len(p2_vals), len(p1_vals)), np.nan)
        prog = st.progress(0, text="Computing parameter grid…")
        total = len(p1_vals) * len(p2_vals)
        count = 0
        for r, p2v in enumerate(p2_vals):
            for c, p1v in enumerate(p1_vals):
                try:
                    sig_df = spec["fn"](df, **{p1_key: p1v, p2_key: p2v, **other_p})
                    result = run_backtest(sig_df, initial_capital, cost_bps)
                    m = compute_metrics(result["strategy_returns"], result["strategy_equity"])
                    grid[r, c] = m["Sharpe Ratio"]
                except Exception:
                    pass
                count += 1
                prog.progress(count / total, text="Computing parameter grid…")
        prog.empty()

        heat = go.Figure(data=go.Heatmap(
            z=grid,
            x=[str(v) for v in p1_vals],
            y=[str(v) for v in p2_vals],
            colorscale="RdYlGn",
            colorbar=dict(title="Sharpe", tickfont=dict(family="JetBrains Mono", color="#A0AEBF")),
            hoverongaps=False,
        ))
        heat.update_layout(
            height=520,
            title=f"Sharpe Ratio — {p1_label} × {p2_label}",
            xaxis_title=p1_label,
            yaxis_title=p2_label,
            **PLOT_LAYOUT,
        )
        st.plotly_chart(heat, use_container_width=True)

        flat = grid.flatten()
        flat = flat[~np.isnan(flat)]
        if len(flat) > 0:
            is_robust = flat.std() <= 0.5
            icon = "✅" if is_robust else "⚠️"
            msg = (
                f"{icon}  Sharpe ranges **{flat.min():.2f} → {flat.max():.2f}** "
                f"(mean {flat.mean():.2f}, σ {flat.std():.2f}).  "
            )
            if is_robust:
                msg += "Relatively stable across the grid — a positive sign of robustness."
            else:
                msg += "High variance across the grid — be cautious about reading too much into any single backtest."
            st.info(msg)
