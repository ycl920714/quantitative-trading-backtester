"""
Quantitative Trading Strategy Backtester
==========================================
Research-grade backtesting with risk metrics, Monte Carlo simulation,
rolling Sharpe, return distribution analysis, and VaR/CVaR.
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

st.set_page_config(
    page_title="QTB — Quantitative Trading Backtester",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
#  DESIGN  ·  Light fintech  (think Robinhood / modern Bloomberg web)
#  Palette:  Cloud #F8FAFC  |  White #FFFFFF  |  Frost #F1F5F9
#            Sky #0EA5E9    |  Emerald #10B981  |  Rose #EF4444
#            Ink #0F172A    |  Slate #64748B    |  Border #E2E8F0
# ─────────────────────────────────────────────────────────────────────────────
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;600&display=swap');

/* ── HIDE STREAMLIT CHROME (safe — do NOT touch header) ── */
#MainMenu { display: none !important; }
.stDeployButton { display: none !important; }
footer { display: none !important; }

/* ── BASE ── */
.stApp { background: #F8FAFC; color: #0F172A; font-family: 'Inter', sans-serif; }
.main .block-container { padding-top: 1rem; padding-bottom: 3rem; max-width: 1400px; }

/* ── SIDEBAR ── */
section[data-testid="stSidebar"] { background: #FFFFFF !important; border-right: 1px solid #E2E8F0; }
section[data-testid="stSidebar"] > div:first-child { padding-top: 1.5rem; }
section[data-testid="stSidebar"] .stMarkdown p { color: #94A3B8; font-size: 0.78rem; line-height: 1.5; }

/* ── SIDEBAR TOGGLE — fixed position, always on top ── */
[data-testid="collapsedControl"] {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    position: fixed !important;
    top: 0.5rem !important;
    left: 0.3rem !important;
    z-index: 99999 !important;
    background: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 8px !important;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1) !important;
    color: #0EA5E9 !important;
    padding: 0.25rem !important;
}
[data-testid="collapsedControl"]:hover {
    background: #F0F9FF !important;
    border-color: #0EA5E9 !important;
}

/* ── HEADINGS ── */
h1,h2,h3,h4 { font-family: 'Space Grotesk', sans-serif !important; color: #0F172A !important; }
h1 { font-size: 2.4rem !important; font-weight: 700 !important; letter-spacing: -0.03em !important; line-height: 1.1 !important; }
h2 { font-size: 1.4rem !important; font-weight: 600 !important; letter-spacing: -0.01em !important; margin-top: 2rem !important; }
h3 { font-size: 1rem !important; font-weight: 500 !important; color: #475569 !important; }

/* ── METRIC CARDS ── */
[data-testid="metric-container"] {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 1.1rem 1.3rem !important;
    transition: border-color 0.2s, box-shadow 0.2s;
    position: relative; overflow: hidden;
}
[data-testid="metric-container"]::after {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #0EA5E9, #38BDF8);
    opacity: 0; transition: opacity 0.2s;
}
[data-testid="metric-container"]:hover { border-color: #BAE6FD; box-shadow: 0 4px 16px rgba(14,165,233,0.1); }
[data-testid="metric-container"]:hover::after { opacity: 1; }
[data-testid="metric-container"] label,
[data-testid="metric-container"] [data-testid="stMetricLabel"] p {
    color: #94A3B8 !important; font-size: 0.68rem !important;
    text-transform: uppercase !important; letter-spacing: 0.1em !important;
    font-weight: 500 !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #0369A1 !important; font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.5rem !important; font-weight: 600 !important; letter-spacing: -0.02em !important;
}

/* ── BUTTON ── */
.stButton > button {
    background: linear-gradient(135deg, #0EA5E9 0%, #0284C7 100%) !important;
    color: #FFFFFF !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700 !important; font-size: 0.9rem !important;
    letter-spacing: 0.04em !important; border: none !important;
    border-radius: 10px !important; padding: 0.65rem 1.5rem !important;
    box-shadow: 0 4px 14px rgba(14,165,233,0.3) !important;
    transition: opacity 0.18s, transform 0.12s, box-shadow 0.18s !important;
}
.stButton > button:hover {
    opacity: 0.9 !important; transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(14,165,233,0.4) !important;
}

/* ── INPUTS ── */
.stTextInput > div > div > input,
.stNumberInput > div > div > input {
    background: #FFFFFF !important; border: 1px solid #E2E8F0 !important;
    border-radius: 8px !important; color: #0F172A !important;
    font-family: 'JetBrains Mono', monospace !important; font-size: 0.9rem !important;
}
.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus {
    border-color: #0EA5E9 !important; box-shadow: 0 0 0 3px rgba(14,165,233,0.12) !important;
}
.stSelectbox > div > div {
    background: #FFFFFF !important; border: 1px solid #E2E8F0 !important;
    border-radius: 8px !important; color: #0F172A !important;
}

/* ── SLIDER ── */
.stSlider [data-baseweb="slider"] div[role="slider"] { background: #0EA5E9 !important; border-color: #0EA5E9 !important; }

/* ── LABELS ── */
.stSelectbox label, .stTextInput label, .stNumberInput label, .stSlider label {
    color: #94A3B8 !important; font-size: 0.72rem !important;
    text-transform: uppercase !important; letter-spacing: 0.08em !important; font-weight: 500 !important;
}

/* ── RADIO ── */
.stRadio label { color: #334155 !important; font-size: 0.88rem !important; }

/* ── DATAFRAME ── */
.stDataFrame { border: 1px solid #E2E8F0 !important; border-radius: 10px !important; overflow: hidden; }
[data-testid="stDataFrame"] th {
    background: #F8FAFC !important; color: #94A3B8 !important;
    font-size: 0.7rem !important; text-transform: uppercase !important; letter-spacing: 0.08em !important;
    border-bottom: 1px solid #E2E8F0 !important;
}
[data-testid="stDataFrame"] td {
    font-family: 'JetBrains Mono', monospace !important; font-size: 0.85rem !important;
    color: #334155 !important; border-bottom: 1px solid #F1F5F9 !important;
}

/* ── ALERTS ── */
[data-testid="stAlert"] { background: #F8FAFC !important; border-radius: 10px !important; border: 1px solid #E2E8F0 !important; }
[data-testid="stAlert"][kind="success"] { background: #F0FDF4 !important; border-left: 3px solid #10B981 !important; border-color: #D1FAE5 !important; }
[data-testid="stAlert"][kind="warning"] { border-left: 3px solid #F59E0B !important; }
[data-testid="stAlert"][kind="info"] { border-left: 3px solid #0EA5E9 !important; }

/* ── EXPANDER ── */
[data-testid="stExpander"] { background: #FFFFFF !important; border: 1px solid #E2E8F0 !important; border-radius: 10px !important; }

/* ── PROGRESS ── */
.stProgress > div > div > div > div { background: linear-gradient(90deg, #0EA5E9, #38BDF8) !important; border-radius: 4px !important; }
.stProgress > div > div > div { background: #E2E8F0 !important; border-radius: 4px !important; }

/* ── DIVIDER / CAPTION ── */
hr { border-color: #E2E8F0 !important; }
.stCaptionContainer p { color: #94A3B8 !important; font-size: 0.78rem !important; }

/* ── CUSTOM COMPONENTS ── */
.sidebar-label {
    font-family: 'Space Grotesk', sans-serif; font-size: 0.63rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.12em; color: #0EA5E9;
    margin: 1.2rem 0 0.5rem 0; padding-bottom: 0.4rem; border-bottom: 1px solid #E2E8F0;
}
.hero-badge {
    display: inline-block; background: #EFF6FF; color: #0369A1;
    font-family: 'Space Grotesk', sans-serif; font-size: 0.7rem; font-weight: 600;
    letter-spacing: 0.1em; text-transform: uppercase;
    padding: 0.3rem 0.8rem; border-radius: 20px; border: 1px solid #BFDBFE; margin-bottom: 1rem;
}
.hero-subtitle { color: #64748B; font-size: 1rem; line-height: 1.6; max-width: 680px; margin-bottom: 2rem; }
.mode-card {
    background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 14px;
    padding: 1.4rem 1.5rem; transition: border-color 0.2s, box-shadow 0.15s, transform 0.15s; height: 100%;
}
.mode-card:hover { border-color: #BAE6FD; box-shadow: 0 4px 20px rgba(14,165,233,0.08); transform: translateY(-2px); }
.mode-card-icon { font-size: 1.5rem; margin-bottom: 0.6rem; }
.mode-card-title { font-family: 'Space Grotesk', sans-serif; font-weight: 600; font-size: 0.95rem; color: #0F172A; margin-bottom: 0.35rem; }
.mode-card-desc { font-size: 0.8rem; color: #64748B; line-height: 1.55; }
.ticker-tag {
    display: inline-block; background: #EFF6FF; color: #0369A1;
    font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; font-weight: 600;
    padding: 0.22rem 0.65rem; border-radius: 6px; border: 1px solid #BFDBFE;
    margin-right: 0.4rem; margin-bottom: 0.3rem;
}
.best-badge {
    display: inline-block; background: #F0FDF4; color: #059669;
    font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; font-weight: 600;
    padding: 0.2rem 0.6rem; border-radius: 6px; border: 1px solid #A7F3D0; margin-left: 0.4rem;
}
.section-rule { border: none; border-top: 1px solid #E2E8F0; margin: 2rem 0; }
.insight-box {
    background: #F0F9FF; border: 1px solid #BAE6FD; border-left: 3px solid #0EA5E9;
    border-radius: 10px; padding: 1rem 1.2rem; margin: 0.8rem 0;
    font-size: 0.84rem; color: #0369A1; line-height: 1.6;
}
</style>
"""

# ─────────────────────────────────────────────────────────────────────────────
#  DATA
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_price_data(ticker: str, period: str) -> pd.DataFrame:
    df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna()


# ─────────────────────────────────────────────────────────────────────────────
#  STRATEGIES
# ─────────────────────────────────────────────────────────────────────────────
def strategy_sma(df, fast=20, slow=50):
    o = df.copy()
    o["fast_ma"] = o["Close"].rolling(fast).mean()
    o["slow_ma"] = o["Close"].rolling(slow).mean()
    o["signal"] = np.where(o["fast_ma"] > o["slow_ma"], 1, 0)
    o["signal"] = o["signal"].shift(1).fillna(0)
    return o

def strategy_rsi(df, period=14, oversold=30, overbought=70):
    o = df.copy()
    d = o["Close"].diff()
    ag = d.clip(lower=0).rolling(period).mean()
    al = (-d.clip(upper=0)).rolling(period).mean()
    o["rsi"] = 100 - 100 / (1 + ag / al.replace(0, np.nan))
    # Vectorised signal: buy when RSI crosses below oversold, sell when above overbought
    long  = (o["rsi"] < oversold).astype(float)
    short = (o["rsi"] > overbought).astype(float)
    sig   = long.copy()
    sig[short == 1] = 0
    # Forward-fill to hold position between signals
    sig = sig.replace(0, np.nan)
    sig = sig.ffill().fillna(0)
    sig[o["rsi"] > overbought] = 0
    o["signal"] = sig.shift(1).fillna(0)
    return o

def strategy_bb(df, window=20, num_std=2.0):
    o = df.copy()
    o["mid"] = o["Close"].rolling(window).mean()
    s = o["Close"].rolling(window).std()
    o["upper"], o["lower"] = o["mid"] + num_std * s, o["mid"] - num_std * s
    # Entry: price below lower band; Exit: price above mid band
    entry = (o["Close"] < o["lower"]).astype(float)
    exit_ = (o["Close"] > o["mid"]).astype(float)
    sig = entry.copy().replace(0, np.nan)
    sig[exit_ == 1] = 0
    sig = sig.ffill().fillna(0)
    sig[exit_ == 1] = 0
    o["signal"] = sig.shift(1).fillna(0)
    return o

def strategy_mom(df, lookback=60):
    o = df.copy()
    o["signal"] = np.where(o["Close"].pct_change(lookback) > 0, 1, 0)
    o["signal"] = o["signal"].shift(1).fillna(0)
    return o

STRATEGIES = {
    "📈 SMA Crossover":      {"fn": strategy_sma, "short": "Trend",    "params": {"fast": ("Fast MA", 5, 100, 20), "slow": ("Slow MA", 10, 250, 50)}},
    "🔄 RSI Mean Reversion": {"fn": strategy_rsi, "short": "MeanRev",  "params": {"period": ("RSI Period", 5, 30, 14), "oversold": ("Oversold", 10, 40, 30), "overbought": ("Overbought", 60, 90, 70)}},
    "📊 Bollinger Bands":    {"fn": strategy_bb,  "short": "MeanRev",  "params": {"window": ("Window", 10, 60, 20), "num_std": ("Std Dev", 1.0, 3.0, 2.0)}},
    "🚀 Momentum":           {"fn": strategy_mom, "short": "Momentum", "params": {"lookback": ("Lookback (days)", 10, 250, 60)}},
}

STRAT_DESC = {
    "📈 SMA Crossover": "Goes long when the fast moving average crosses above the slow MA, signalling an uptrend. Exits when the signal reverses.",
    "🔄 RSI Mean Reversion": "Buys when RSI falls below the oversold threshold (price has dropped 'too far, too fast') and exits when RSI climbs above overbought.",
    "📊 Bollinger Bands": "Enters long when price dips below the lower band (statistically cheap relative to recent history) and exits when price reverts to the rolling mean.",
    "🚀 Momentum": "Goes long when trailing N-day returns are positive, betting that recent winners continue to outperform over the lookback horizon.",
}


# ─────────────────────────────────────────────────────────────────────────────
#  BACKTEST ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def run_backtest(df, initial_capital, cost_bps=5.0):
    o = df.copy()
    o["returns"] = o["Close"].pct_change().fillna(0)
    o["position_change"] = o["signal"].diff().abs().fillna(0)
    o["trade_cost"] = o["position_change"] * (cost_bps / 10000.0)
    o["strategy_returns"] = o["signal"] * o["returns"] - o["trade_cost"]
    o["strategy_equity"] = initial_capital * (1 + o["strategy_returns"]).cumprod()
    o["buyhold_equity"]   = initial_capital * (1 + o["returns"]).cumprod()
    return o


# ─────────────────────────────────────────────────────────────────────────────
#  METRICS
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(returns, equity, ann=252):
    r = returns.dropna()
    if len(r) < 2: return {k: np.nan for k in ["Total Return","CAGR","Annual Volatility","Sharpe Ratio","Sortino Ratio","Max Drawdown","Calmar Ratio","Win Rate"]}
    total = equity.iloc[-1] / equity.iloc[0] - 1
    ny = len(r) / ann
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1/ny) - 1 if ny > 0 else np.nan
    vol = r.std() * np.sqrt(ann)
    ar  = r.mean() * ann
    sharpe  = ar / vol if vol > 0 else np.nan
    dv = r[r<0].std() * np.sqrt(ann)
    sortino = ar / dv if dv and dv > 0 else np.nan
    mdd = ((equity - equity.cummax()) / equity.cummax()).min()
    calmar = cagr / abs(mdd) if mdd else np.nan
    wr = (r > 0).sum() / (r != 0).sum() if (r != 0).sum() > 0 else np.nan
    return {"Total Return": total, "CAGR": cagr, "Annual Volatility": vol,
            "Sharpe Ratio": sharpe, "Sortino Ratio": sortino, "Max Drawdown": mdd,
            "Calmar Ratio": calmar, "Win Rate": wr}

def compute_var_cvar(returns, confidence=0.95):
    r = returns.dropna()
    if len(r) < 10: return np.nan, np.nan
    var  = np.percentile(r, (1 - confidence) * 100)
    cvar = r[r <= var].mean()
    return var, cvar

def fp(x): return "—" if pd.isna(x) else f"{x*100:.2f}%"
def fn(x): return "—" if pd.isna(x) else f"{x:.2f}"


# ─────────────────────────────────────────────────────────────────────────────
#  CHART THEME
# ─────────────────────────────────────────────────────────────────────────────
LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#F8FAFC",
    font=dict(family="Inter, sans-serif", color="#64748B", size=12),
    xaxis=dict(gridcolor="#E2E8F0", linecolor="#E2E8F0", zerolinecolor="#E2E8F0"),
    yaxis=dict(gridcolor="#E2E8F0", linecolor="#E2E8F0", zerolinecolor="#E2E8F0"),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#E2E8F0", borderwidth=1,
                orientation="h", y=1.08),
    hovermode="x unified",
    hoverlabel=dict(bgcolor="#FFFFFF", bordercolor="#E2E8F0",
                    font=dict(family="JetBrains Mono, monospace", size=12, color="#0F172A")),
    margin=dict(t=60, b=40, l=60, r=20),
)

SKY   = "#0EA5E9"
SLATE = "#94A3B8"
ROSE  = "#EF4444"
EMRLD = "#10B981"
AMBER = "#F59E0B"
PURP  = "#8B5CF6"


# ─────────────────────────────────────────────────────────────────────────────
#  CHARTS
# ─────────────────────────────────────────────────────────────────────────────
def plot_equity(result, strat_name, split_dt=None):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.68, 0.32],
                         vertical_spacing=0.04, subplot_titles=("Equity Curve", "Drawdown (%)"))
    fig.add_trace(go.Scatter(x=result.index, y=result["strategy_equity"], name=strat_name,
                              line=dict(color=SKY, width=2.2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=result.index, y=result["buyhold_equity"], name="Buy & Hold",
                              line=dict(color=SLATE, width=1.5, dash="dot")), row=1, col=1)
    running_max = result["strategy_equity"].cummax()
    dd = (result["strategy_equity"] - running_max) / running_max * 100
    fig.add_trace(go.Scatter(x=result.index, y=dd, fill="tozeroy", name="Drawdown",
                              line=dict(color=ROSE, width=1), fillcolor="rgba(239,68,68,0.1)"), row=2, col=1)
    if split_dt:
        for r in [1, 2]:
            fig.add_vline(x=split_dt, line_dash="dot", line_color="#CBD5E1",
                           annotation_text="OOS Split" if r == 1 else "",
                           annotation_font_color=SLATE, annotation_font_size=10, row=r, col=1)
    fig.update_layout(height=560, **LAYOUT)
    fig.update_annotations(font_color=SLATE, font_size=11)
    fig.update_yaxes(title_text="Value ($)", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown %", row=2, col=1)
    for ax in ["xaxis", "xaxis2", "yaxis", "yaxis2"]:
        getattr(fig.layout, ax).update(gridcolor="#E2E8F0", linecolor="#E2E8F0")
    return fig


def plot_signals(result):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=result.index, y=result["Close"], name="Price",
                              line=dict(color="#0F172A", width=1.4)))
    if "fast_ma" in result.columns:
        fig.add_trace(go.Scatter(x=result.index, y=result["fast_ma"], name="Fast MA", line=dict(color=AMBER, width=1.2)))
        fig.add_trace(go.Scatter(x=result.index, y=result["slow_ma"], name="Slow MA", line=dict(color=PURP, width=1.2)))
    if "upper" in result.columns:
        fig.add_trace(go.Scatter(x=result.index, y=result["upper"], name="Upper Band",
                                  line=dict(color=SLATE, width=1, dash="dot")))
        fig.add_trace(go.Scatter(x=result.index, y=result["lower"], name="Lower Band",
                                  line=dict(color=SLATE, width=1, dash="dot"),
                                  fill="tonexty", fillcolor="rgba(148,163,184,0.08)"))
        fig.add_trace(go.Scatter(x=result.index, y=result["mid"], name="Mid Band",
                                  line=dict(color=AMBER, width=1, dash="dash")))
    buys  = result[result["signal"].diff() == 1]
    sells = result[result["signal"].diff() == -1]
    fig.add_trace(go.Scatter(x=buys.index,  y=buys["Close"],  mode="markers", name="Buy",
                              marker=dict(symbol="triangle-up",   color=EMRLD, size=9, line=dict(width=0))))
    fig.add_trace(go.Scatter(x=sells.index, y=sells["Close"], mode="markers", name="Sell",
                              marker=dict(symbol="triangle-down", color=ROSE,  size=9, line=dict(width=0))))
    fig.update_layout(height=440, title="Entry / Exit Signals", **LAYOUT)
    fig.layout.xaxis.update(gridcolor="#E2E8F0", linecolor="#E2E8F0")
    fig.layout.yaxis.update(gridcolor="#E2E8F0", linecolor="#E2E8F0")
    return fig


def plot_return_distribution(strategy_returns, bh_returns):
    """Histogram of daily returns with normal-distribution overlay and fat-tail annotation."""
    r = strategy_returns.dropna()
    r = r[r != 0]
    if len(r) < 20:
        return None

    mu, sigma = r.mean(), r.std()
    skew = float(stats.skew(r))
    kurt = float(stats.kurtosis(r))  # excess kurtosis (normal = 0)

    x_range = np.linspace(r.quantile(0.001), r.quantile(0.999), 300)
    normal_pdf = stats.norm.pdf(x_range, mu, sigma)
    # scale to match histogram density
    hist_counts, hist_bins = np.histogram(r, bins=60, density=True)

    var95, _ = compute_var_cvar(r, 0.95)

    fig = go.Figure()

    # Strategy histogram
    fig.add_trace(go.Histogram(
        x=r * 100, nbinsx=60, histnorm="probability density",
        name="Strategy returns", marker_color=SKY, opacity=0.65,
        xbins=dict(size=0.1),
    ))

    # Normal overlay
    fig.add_trace(go.Scatter(
        x=x_range * 100, y=normal_pdf / 100,
        name="Normal distribution", line=dict(color=AMBER, width=2, dash="dash"),
        mode="lines",
    ))

    # VaR line
    if not np.isnan(var95):
        fig.add_vline(x=var95 * 100, line_color=ROSE, line_dash="dot",
                       annotation_text=f"VaR 95%: {var95*100:.2f}%",
                       annotation_font_color=ROSE, annotation_font_size=10)

    # Fat-tail insight annotation
    tail_note = (
        f"Skewness: {skew:.2f}  |  Excess Kurtosis: {kurt:.2f}"
        + ("  ← fat tails" if kurt > 1 else "")
    )
    fig.update_layout(
        height=420, title=f"Return Distribution  ·  {tail_note}",
        xaxis_title="Daily Return (%)", yaxis_title="Density",
        **LAYOUT,
    )
    fig.layout.xaxis.update(gridcolor="#E2E8F0", linecolor="#E2E8F0")
    fig.layout.yaxis.update(gridcolor="#E2E8F0", linecolor="#E2E8F0")
    return fig


def plot_rolling_sharpe(strategy_returns, window=63):
    """Rolling annualised Sharpe ratio — detects strategy decay over time."""
    r = strategy_returns.dropna()
    roll_mean = r.rolling(window).mean() * 252
    roll_std  = r.rolling(window).std()  * np.sqrt(252)
    roll_sharpe = (roll_mean / roll_std).replace([np.inf, -np.inf], np.nan)

    fig = go.Figure()
    # Shade positive/negative regions
    fig.add_hrect(y0=0, y1=roll_sharpe.max() * 1.1 if roll_sharpe.max() > 0 else 2,
                   fillcolor="rgba(16,185,129,0.04)", line_width=0)
    fig.add_hrect(y0=roll_sharpe.min() * 1.1 if roll_sharpe.min() < 0 else -2, y1=0,
                   fillcolor="rgba(239,68,68,0.04)", line_width=0)

    fig.add_trace(go.Scatter(
        x=roll_sharpe.index, y=roll_sharpe,
        name=f"Rolling Sharpe ({window}d)", line=dict(color=SKY, width=2),
        fill="tozeroy", fillcolor="rgba(14,165,233,0.07)",
    ))
    fig.add_hline(y=1.0, line_color=EMRLD, line_dash="dash",
                   annotation_text="Sharpe = 1", annotation_font_color=EMRLD, annotation_font_size=10)
    fig.add_hline(y=0.0, line_color=SLATE, line_dash="dot")
    fig.update_layout(
        height=380,
        title=f"Rolling Sharpe Ratio  ·  {window}-day window  (annualised)",
        yaxis_title="Sharpe Ratio",
        **LAYOUT,
    )
    fig.layout.xaxis.update(gridcolor="#E2E8F0", linecolor="#E2E8F0")
    fig.layout.yaxis.update(gridcolor="#E2E8F0", linecolor="#E2E8F0")
    return fig


def plot_monte_carlo(strategy_returns, initial_capital, n_sims=300, horizon=252):
    """
    Bootstrap Monte Carlo: resample daily returns (with replacement) to generate
    N simulated equity paths. Shows 5th/95th percentile confidence band.
    """
    r = strategy_returns.dropna().values
    r = r[r != 0]
    if len(r) < 30:
        return None

    rng = np.random.default_rng(42)
    all_paths = np.zeros((n_sims, horizon + 1))
    all_paths[:, 0] = initial_capital
    for s in range(n_sims):
        sampled = rng.choice(r, size=horizon, replace=True)
        all_paths[s, 1:] = initial_capital * np.cumprod(1 + sampled)

    p5   = np.percentile(all_paths, 5,  axis=0)
    p25  = np.percentile(all_paths, 25, axis=0)
    p50  = np.percentile(all_paths, 50, axis=0)
    p75  = np.percentile(all_paths, 75, axis=0)
    p95  = np.percentile(all_paths, 95, axis=0)
    days = np.arange(horizon + 1)

    fig = go.Figure()

    # Draw a handful of individual paths (lightly)
    for i in range(min(60, n_sims)):
        fig.add_trace(go.Scatter(
            x=days, y=all_paths[i], mode="lines",
            line=dict(color="rgba(14,165,233,0.07)", width=1),
            showlegend=False, hoverinfo="skip",
        ))

    # Confidence bands
    fig.add_trace(go.Scatter(x=np.concatenate([days, days[::-1]]),
                              y=np.concatenate([p95, p5[::-1]]),
                              fill="toself", fillcolor="rgba(14,165,233,0.10)",
                              line=dict(color="rgba(0,0,0,0)"), name="5th–95th pct"))
    fig.add_trace(go.Scatter(x=np.concatenate([days, days[::-1]]),
                              y=np.concatenate([p75, p25[::-1]]),
                              fill="toself", fillcolor="rgba(14,165,233,0.18)",
                              line=dict(color="rgba(0,0,0,0)"), name="25th–75th pct"))

    fig.add_trace(go.Scatter(x=days, y=p50, name="Median path",
                              line=dict(color=SKY, width=2.5)))
    fig.add_hline(y=initial_capital, line_color=SLATE, line_dash="dot",
                   annotation_text="Initial capital", annotation_font_color=SLATE, annotation_font_size=10)

    # Summary stats annotation
    prob_profit = (all_paths[:, -1] > initial_capital).mean()
    median_final = p50[-1]
    fig.update_layout(
        height=440,
        title=(f"Monte Carlo Simulation  ·  {n_sims} paths, {horizon}-day horizon  "
               f"·  P(profit) = {prob_profit:.0%}  ·  Median final = ${median_final:,.0f}"),
        yaxis_title="Portfolio Value ($)",
        xaxis_title="Trading Days",
        **LAYOUT,
    )
    fig.layout.xaxis.update(gridcolor="#E2E8F0", linecolor="#E2E8F0")
    fig.layout.yaxis.update(gridcolor="#E2E8F0", linecolor="#E2E8F0")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(CSS, unsafe_allow_html=True)

with st.sidebar:
    MODE_LIST = ["Single", "Multi-Strategy", "Multi-Stock", "Parameter Sensitivity"]
    st.markdown('<p class="sidebar-label">Mode</p>', unsafe_allow_html=True)
    _mode_idx = MODE_LIST.index(st.session_state.get("selected_mode", "Single"))
    mode = st.radio("", MODE_LIST, index=_mode_idx, label_visibility="collapsed")
    st.session_state.selected_mode = mode

    st.markdown('<p class="sidebar-label">Universe</p>', unsafe_allow_html=True)
    if mode in ("Single", "Multi-Strategy", "Parameter Sensitivity"):
        ticker_raw = st.text_input("Ticker", value="AAPL", label_visibility="collapsed", placeholder="e.g. AAPL")
        tickers = [ticker_raw.strip().upper()]
    else:
        tickers_raw = st.text_input("Tickers", value="AAPL, MSFT, GOOGL", label_visibility="collapsed", placeholder="e.g. AAPL, MSFT, TSLA")
        tickers = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]

    st.markdown('<p class="sidebar-label">Strategy</p>', unsafe_allow_html=True)
    if mode in ("Single", "Multi-Stock", "Parameter Sensitivity"):
        strategy_name  = st.selectbox("Strategy", list(STRATEGIES.keys()), label_visibility="collapsed")
        strategy_names = [strategy_name]
    else:
        strategy_names = st.multiselect("Strategies", list(STRATEGIES.keys()),
                                         default=list(STRATEGIES.keys())[:2], label_visibility="collapsed")

    st.markdown('<p class="sidebar-label">Settings</p>', unsafe_allow_html=True)
    period          = st.selectbox("Period", ["1y","2y","5y","10y","max"], index=2, label_visibility="collapsed")
    initial_capital = st.number_input("Capital ($)", min_value=100, value=10000, step=100)
    cost_bps        = st.slider("Transaction cost (bps)", 0, 50, 5)
    oos_split       = st.slider("In-sample split (%)", 50, 95, 70)

    custom_params = {}
    if mode in ("Single", "Multi-Stock"):
        spec = STRATEGIES[strategy_names[0]]
        if spec["params"]:
            st.markdown('<p class="sidebar-label">Parameters</p>', unsafe_allow_html=True)
            for pkey, (label, lo, hi, default) in spec["params"].items():
                if isinstance(default, int):
                    custom_params[pkey] = st.slider(label, lo, hi, default)
                else:
                    custom_params[pkey] = st.slider(label, float(lo), float(hi), float(default), step=0.1)

    st.markdown('<div style="height:0.8rem"></div>', unsafe_allow_html=True)
    run_btn = st.button("▶  Run Backtest", use_container_width=True, type="primary")
    if st.session_state.get("has_run"):
        if st.button("↩  Back to Home", use_container_width=True):
            st.session_state.has_run = False
            st.rerun()
    st.caption("Educational tool. Past performance does not indicate future results. Long-only, adjusted close via yfinance.")


if run_btn:
    st.session_state.has_run = True
if "has_run" not in st.session_state:
    st.session_state.has_run = False

# ─────────────────────────────────────────────────────────────────────────────
#  HERO
# ─────────────────────────────────────────────────────────────────────────────
if not st.session_state.has_run:
    st.markdown("""
    <div class="hero-badge">Portfolio Project · Finance &amp; Analytics</div>
    <h1>Quantitative<br>Trading Backtester</h1>
    <p class="hero-subtitle">
        Research-grade backtesting with risk-adjusted metrics, benchmark comparison,
        transaction cost modelling, out-of-sample validation, Monte Carlo simulation,
        and fat-tail risk analysis.
    </p>""", unsafe_allow_html=True)

    MODE_CARDS = [
        ("🎯", "Single",         "Single",               "One stock, one strategy — equity curve, signals, rolling Sharpe, return distribution, VaR/CVaR, and Monte Carlo."),
        ("📊", "Multi-Strategy", "Multi-Strategy",       "All four strategies on one stock side-by-side. Sharpe-ranked comparison table."),
        ("🌍", "Multi-Stock",    "Multi-Stock",          "Same strategy across multiple tickers — tests whether the edge is broadly robust."),
        ("🔬", "Sensitivity",    "Parameter Sensitivity","Parameter heatmap — flags overfitting when performance only appears at a single exact setting."),
    ]
    cols = st.columns(4, gap="small")
    for col, (icon, label, mode_key, desc) in zip(cols, MODE_CARDS):
        with col:
            st.markdown(
                f'<div class="mode-card">'
                f'<div class="mode-card-icon">{icon}</div>'
                f'<div class="mode-card-title">{label}</div>'
                f'<div class="mode-card-desc">{desc}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            if st.button(f"Open {label} →", key=f"hero_{mode_key}", use_container_width=True):
                st.session_state.selected_mode = mode_key
                st.session_state.has_run = True
                st.rerun()

    st.markdown('<hr class="section-rule">', unsafe_allow_html=True)
    st.markdown("##### Built-in strategies")
    cols2 = st.columns(4, gap="small")
    strats = [("📈","SMA Crossover","Trend Following","Fast/slow MA crossover"),
              ("🔄","RSI","Mean Reversion","Oversold/overbought oscillator"),
              ("📊","Bollinger Bands","Mean Reversion","Statistical price-range entry"),
              ("🚀","Momentum","Momentum","Trailing-return trend continuation")]
    for col, (icon, name, style, desc) in zip(cols2, strats):
        with col:
            st.markdown(f'<div class="mode-card" style="padding:1rem 1.2rem">'
                        f'<div style="font-size:1.2rem;margin-bottom:.4rem">{icon} '
                        f'<strong style="font-family:\'Space Grotesk\',sans-serif;color:#0F172A">{name}</strong></div>'
                        f'<div style="font-size:.68rem;text-transform:uppercase;letter-spacing:.1em;color:{SKY};margin-bottom:.35rem">{style}</div>'
                        f'<div class="mode-card-desc">{desc}</div></div>', unsafe_allow_html=True)
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def get_data(ticker):
    df = fetch_price_data(ticker, period)
    if df.empty:
        st.error(f"No data for **{ticker}**. Check the symbol and try again.")
        return None
    return df

def split_idx(df):
    n = len(df); i = int(n * oos_split / 100)
    return df.index[i] if 0 < i < n else None

def metrics_row(m):
    c = st.columns(6)
    c[0].metric("Total Return",    fp(m["Total Return"]))
    c[1].metric("CAGR",            fp(m["CAGR"]))
    c[2].metric("Sharpe Ratio",    fn(m["Sharpe Ratio"]))
    c[3].metric("Sortino Ratio",   fn(m["Sortino Ratio"]))
    c[4].metric("Max Drawdown",    fp(m["Max Drawdown"]))
    c[5].metric("Win Rate",        fp(m["Win Rate"]))

def comp_table(rows):
    df = pd.DataFrame(rows).T
    d  = df.copy()
    for col in ["Total Return","CAGR","Annual Volatility","Max Drawdown","Win Rate"]: d[col] = d[col].apply(fp)
    for col in ["Sharpe Ratio","Sortino Ratio","Calmar Ratio"]:                       d[col] = d[col].apply(fn)
    return d

COLORS = [SKY, PURP, AMBER, EMRLD, ROSE, "#F472B6"]

# Hint banner — reminds user sidebar exists if it's collapsed
st.markdown(
    '<div style="background:#EFF6FF;border:1px solid #BFDBFE;border-radius:10px;'
    'padding:0.55rem 1rem;margin-bottom:1rem;font-size:0.82rem;color:#1D4ED8;">'
    '⬅️  Use the <strong>sidebar on the left</strong> to change ticker, strategy, or settings, '
    'then click <strong>Run Backtest</strong> again.</div>',
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
#  SINGLE MODE
# ─────────────────────────────────────────────────────────────────────────────
if mode == "Single":
    ticker = tickers[0]
    df = get_data(ticker)
    if df is not None:
        spec   = STRATEGIES[strategy_names[0]]
        sig_df = spec["fn"](df, **custom_params)
        result = run_backtest(sig_df, initial_capital, cost_bps)
        split_dt = split_idx(result)

        strat_m = compute_metrics(result["strategy_returns"], result["strategy_equity"])
        bh_m    = compute_metrics(result["returns"],           result["buyhold_equity"])
        is_m = oos_m = None
        if split_dt is not None:
            is_m  = compute_metrics(result.loc[:split_dt,  "strategy_returns"], result.loc[:split_dt,  "strategy_equity"])
            oos_m = compute_metrics(result.loc[split_dt:,  "strategy_returns"], result.loc[split_dt:,  "strategy_equity"])

        # Header tags
        st.markdown("".join(f'<span class="ticker-tag">{t}</span>'
                            for t in [ticker, strategy_names[0], period]), unsafe_allow_html=True)
        st.markdown('<div style="height:.7rem"></div>', unsafe_allow_html=True)

        # KPI row
        metrics_row(strat_m)
        st.markdown('<div style="height:.5rem"></div>', unsafe_allow_html=True)

        # ── NEW: VaR / CVaR row ──
        var95,  cvar95  = compute_var_cvar(result["strategy_returns"], 0.95)
        var99,  cvar99  = compute_var_cvar(result["strategy_returns"], 0.99)
        vc1, vc2, vc3, vc4 = st.columns(4)
        vc1.metric("VaR 95% (daily)",  fp(var95),  help="Daily loss not exceeded 95% of the time (historical)")
        vc2.metric("CVaR 95% (daily)", fp(cvar95), help="Average loss on the worst 5% of days (Expected Shortfall)")
        vc3.metric("VaR 99% (daily)",  fp(var99))
        vc4.metric("CVaR 99% (daily)", fp(cvar99))

        st.markdown('<hr class="section-rule">', unsafe_allow_html=True)

        # Equity + drawdown
        st.plotly_chart(plot_equity(result, strategy_names[0], split_dt), use_container_width=True)

        # Signals chart
        st.plotly_chart(plot_signals(result), use_container_width=True)

        # ── ADVANCED ANALYSIS (opt-in expander to keep page clean & fast) ──
        with st.expander("🔬  Advanced Analysis — Rolling Sharpe · Return Distribution · Monte Carlo", expanded=False):
            st.caption("These charts run on demand. Monte Carlo (300 paths) takes a few seconds.")

            st.markdown("**Rolling Sharpe Ratio**")
            st.caption("A declining rolling Sharpe suggests the strategy's edge is fading.")
            st.plotly_chart(plot_rolling_sharpe(result["strategy_returns"]), use_container_width=True)

            st.markdown("**Return Distribution**")
            st.caption("Empirical vs normal — reveals fat tails and skewness that standard deviation misses.")
            dist_fig = plot_return_distribution(result["strategy_returns"], result["returns"])
            if dist_fig:
                st.plotly_chart(dist_fig, use_container_width=True)
                r = result["strategy_returns"].dropna()
                sk = float(stats.skew(r[r != 0]))
                ku = float(stats.kurtosis(r[r != 0]))
                tail_warn = ku > 1.5
                insight = (
                    f"**Excess kurtosis = {ku:.2f}** — {'heavier tails than normal, meaning extreme losses occur more often than standard models assume' if tail_warn else 'near-normal tail thickness'}. "
                    f"**Skewness = {sk:.2f}** ({'left-skewed: large losses more likely than large gains' if sk < -0.3 else 'right-skewed: large gains outweigh large losses' if sk > 0.3 else 'roughly symmetric'})."
                )
                st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)

            st.markdown("**Monte Carlo Simulation**")
            st.caption("Bootstrap resampling generates 300 simulated paths. Shaded bands show 50% and 90% confidence intervals.")
            mc_fig = plot_monte_carlo(result["strategy_returns"], initial_capital)
            if mc_fig:
                st.plotly_chart(mc_fig, use_container_width=True)

        st.markdown('<hr class="section-rule">', unsafe_allow_html=True)

        # Comparison table
        st.markdown("### Strategy vs Buy & Hold")
        st.dataframe(comp_table({"Strategy": strat_m, "Buy & Hold": bh_m}), use_container_width=True)

        # OOS table
        if is_m is not None:
            st.markdown("### In-Sample vs Out-of-Sample")
            st.caption(f"In-sample: first {oos_split}% (until {split_dt.date()}). A large Sharpe drop out-of-sample signals overfitting.")
            st.dataframe(comp_table({"In-Sample": is_m, "Out-of-Sample": oos_m}), use_container_width=True)

        with st.expander("How this strategy works"):
            st.write(STRAT_DESC[strategy_names[0]])


# ─────────────────────────────────────────────────────────────────────────────
#  MULTI-STRATEGY
# ─────────────────────────────────────────────────────────────────────────────
elif mode == "Multi-Strategy":
    ticker = tickers[0]
    df = get_data(ticker)
    if df is not None and strategy_names:
        st.markdown("".join(f'<span class="ticker-tag">{t}</span>'
                            for t in [ticker, "All Strategies", period]), unsafe_allow_html=True)
        st.markdown('<div style="height:.7rem"></div>', unsafe_allow_html=True)

        fig = go.Figure()
        bh_eq = initial_capital * (1 + df["Close"].pct_change().fillna(0)).cumprod()
        fig.add_trace(go.Scatter(x=df.index, y=bh_eq, name="Buy & Hold",
                                  line=dict(color=SLATE, width=1.4, dash="dot")))
        rows = {}
        for i, name in enumerate(strategy_names):
            sp = STRATEGIES[name]
            dp = {k: v[3] for k, v in sp["params"].items()}
            res = run_backtest(sp["fn"](df, **dp), initial_capital, cost_bps)
            m = compute_metrics(res["strategy_returns"], res["strategy_equity"])
            rows[name] = m
            fig.add_trace(go.Scatter(x=res.index, y=res["strategy_equity"], name=name,
                                      line=dict(color=COLORS[i % len(COLORS)], width=2)))

        fig.update_layout(height=520, title="Equity Curves", **LAYOUT)
        fig.layout.xaxis.update(gridcolor="#E2E8F0", linecolor="#E2E8F0")
        fig.layout.yaxis.update(gridcolor="#E2E8F0", linecolor="#E2E8F0")
        st.plotly_chart(fig, use_container_width=True)

        best = max(rows, key=lambda k: rows[k].get("Sharpe Ratio", -np.inf))
        st.markdown(f'### Performance Comparison  <span class="best-badge">🏆 {best}</span>', unsafe_allow_html=True)
        st.dataframe(comp_table(rows), use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
#  MULTI-STOCK
# ─────────────────────────────────────────────────────────────────────────────
elif mode == "Multi-Stock":
    spec = STRATEGIES[strategy_names[0]]
    st.markdown("".join(f'<span class="ticker-tag">{t}</span>'
                        for t in [strategy_names[0]] + tickers + [period]), unsafe_allow_html=True)
    st.markdown('<div style="height:.7rem"></div>', unsafe_allow_html=True)

    fig = go.Figure()
    rows = {}
    for i, ticker in enumerate(tickers):
        df = get_data(ticker)
        if df is None: continue
        res = run_backtest(spec["fn"](df, **custom_params), initial_capital, cost_bps)
        m = compute_metrics(res["strategy_returns"], res["strategy_equity"])
        rows[ticker] = m
        norm = 100 * res["strategy_equity"] / res["strategy_equity"].iloc[0]
        fig.add_trace(go.Scatter(x=res.index, y=norm, name=ticker,
                                  line=dict(color=COLORS[i % len(COLORS)], width=2)))

    if rows:
        fig.update_layout(height=520, title="Normalised Equity (Base = 100)", **LAYOUT)
        fig.layout.xaxis.update(gridcolor="#E2E8F0", linecolor="#E2E8F0")
        fig.layout.yaxis.update(gridcolor="#E2E8F0", linecolor="#E2E8F0")
        st.plotly_chart(fig, use_container_width=True)
        best = max(rows, key=lambda k: rows[k].get("Sharpe Ratio", -np.inf))
        st.markdown(f'### Performance by Stock  <span class="best-badge">🏆 {best}</span>', unsafe_allow_html=True)
        st.dataframe(comp_table(rows), use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
#  PARAMETER SENSITIVITY
# ─────────────────────────────────────────────────────────────────────────────
elif mode == "Parameter Sensitivity":
    ticker = tickers[0]
    df = get_data(ticker)
    spec = STRATEGIES[strategy_names[0]]
    pkeys = list(spec["params"].keys())

    st.markdown("".join(f'<span class="ticker-tag">{t}</span>'
                        for t in [ticker, "Sensitivity", strategy_names[0]]), unsafe_allow_html=True)
    st.markdown('<div style="height:.5rem"></div>', unsafe_allow_html=True)
    st.caption("A robust strategy works across a broad range of parameter values, not just one exact combination. "
               "A single bright spot surrounded by poor performance everywhere else is a classic overfitting warning sign.")

    if df is not None and len(pkeys) >= 2:
        p1k, p2k = pkeys[0], pkeys[1]
        p1l, p1lo, p1hi, p1d = spec["params"][p1k]
        p2l, p2lo, p2hi, p2d = spec["params"][p2k]
        n = 8
        p1v = np.linspace(p1lo, p1hi, n); p2v = np.linspace(p2lo, p2hi, n)
        if isinstance(p1d, int): p1v = sorted(set(int(x) for x in p1v))
        if isinstance(p2d, int): p2v = sorted(set(int(x) for x in p2v))
        other = {k: v[3] for k, v in spec["params"].items() if k not in (p1k, p2k)}

        grid = np.full((len(p2v), len(p1v)), np.nan)
        prog = st.progress(0, text="Running parameter grid…")
        total = len(p1v) * len(p2v); cnt = 0
        for r, pv2 in enumerate(p2v):
            for c, pv1 in enumerate(p1v):
                try:
                    s = spec["fn"](df, **{p1k: pv1, p2k: pv2, **other})
                    res = run_backtest(s, initial_capital, cost_bps)
                    m = compute_metrics(res["strategy_returns"], res["strategy_equity"])
                    grid[r, c] = m["Sharpe Ratio"]
                except: pass
                cnt += 1; prog.progress(cnt/total, text="Running parameter grid…")
        prog.empty()

        heat = go.Figure(go.Heatmap(
            z=grid, x=[str(v) for v in p1v], y=[str(v) for v in p2v],
            colorscale="RdYlGn",
            colorbar=dict(title="Sharpe", tickfont=dict(family="JetBrains Mono", color="#64748B")),
            hoverongaps=False,
        ))
        heat.update_layout(height=520, title=f"Sharpe Ratio Grid: {p1l} × {p2l}",
                            xaxis_title=p1l, yaxis_title=p2l, **LAYOUT)
        heat.layout.xaxis.update(gridcolor="#E2E8F0", linecolor="#E2E8F0")
        heat.layout.yaxis.update(gridcolor="#E2E8F0", linecolor="#E2E8F0")
        st.plotly_chart(heat, use_container_width=True)

        flat = grid.flatten(); flat = flat[~np.isnan(flat)]
        if len(flat):
            robust = flat.std() <= 0.5
            icon = "✅" if robust else "⚠️"
            msg = (f"{icon}  Sharpe ranges **{flat.min():.2f} → {flat.max():.2f}** "
                   f"(mean {flat.mean():.2f}, σ {flat.std():.2f}). "
                   + ("Relatively stable across the grid — positive sign of robustness." if robust
                      else "High variance — the strategy may be sensitive to exact parameter choice."))
            st.info(msg)
    else:
        st.warning("This strategy needs at least two tunable parameters for a 2D sensitivity grid.")
