import streamlit as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import numpy as np
import sys
import os
from io import BytesIO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(page_title="Quantitative Trading Backtester", layout="wide")
st.title("📈 Quantitative Trading Strategy Backtester")
st.markdown("Backtest 9 trading strategies on any stock using historical data")

with st.sidebar:
    st.header("⚙️ Settings")
    ticker = st.text_input("Stock Ticker", value="AAPL").upper()
    strategy = st.selectbox("Strategy", [
        "sma_crossover", "ema_crossover", "rsi_mean_reversion",
        "macd", "bb_breakout", "bb_mean_reversion",
        "momentum", "buy_and_hold"
    ])
    period = st.selectbox("Period", ["1y", "2y", "3y", "5y"], index=3)
    capital = st.number_input("Initial Capital ($)", value=10000, step=1000)
    run = st.button("🚀 Run Backtest", type="primary")

def compute_signals(df, strategy):
    df = df.copy()
    df['signal'] = 0
    if strategy == "sma_crossover":
        df['fast'] = df['Close'].rolling(20).mean()
        df['slow'] = df['Close'].rolling(50).mean()
        df['signal'] = np.where(df['fast'] > df['slow'], 1, -1)
    elif strategy == "ema_crossover":
        df['fast'] = df['Close'].ewm(span=12).mean()
        df['slow'] = df['Close'].ewm(span=26).mean()
        df['signal'] = np.where(df['fast'] > df['slow'], 1, -1)
    elif strategy == "rsi_mean_reversion":
        delta = df['Close'].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = -delta.clip(upper=0).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        df['signal'] = np.where(df['rsi'] < 30, 1, np.where(df['rsi'] > 70, -1, 0))
    elif strategy == "macd":
        df['macd'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
        df['signal_line'] = df['macd'].ewm(span=9).mean()
        df['signal'] = np.where(df['macd'] > df['signal_line'], 1, -1)
    elif strategy == "bb_breakout":
        df['mid'] = df['Close'].rolling(20).mean()
        df['std'] = df['Close'].rolling(20).std()
        df['upper'] = df['mid'] + 2 * df['std']
        df['lower'] = df['mid'] - 2 * df['std']
        df['signal'] = np.where(df['Close'] > df['upper'], 1, np.where(df['Close'] < df['lower'], -1, 0))
    elif strategy == "bb_mean_reversion":
        df['mid'] = df['Close'].rolling(20).mean()
        df['std'] = df['Close'].rolling(20).std()
        df['upper'] = df['mid'] + 2 * df['std']
        df['lower'] = df['mid'] - 2 * df['std']
        df['signal'] = np.where(df['Close'] < df['lower'], 1, np.where(df['Close'] > df['upper'], -1, 0))
    elif strategy == "momentum":
        df['signal'] = np.where(df['Close'].pct_change(20) > 0, 1, -1)
    elif strategy == "buy_and_hold":
        df['signal'] = 1
    return df

def compute_metrics(equity):
    returns = equity.pct_change().dropna()
    total_return = (equity.iloc[-1] / equity.iloc[0] - 1) * 100
    n_years = len(equity) / 252
    cagr = ((equity.iloc[-1] / equity.iloc[0]) ** (1 / n_years) - 1) * 100
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
    downside = returns[returns < 0].std()
    sortino = (returns.mean() / downside) * np.sqrt(252) if downside > 0 else 0
    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max * 100
    max_dd = drawdown.min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    return {
        'Total Return (%)': round(total_return, 2),
        'CAGR (%)': round(cagr, 2),
        'Sharpe Ratio': round(sharpe, 3),
        'Sortino Ratio': round(sortino, 3),
        'Max Drawdown (%)': round(max_dd, 2),
        'Calmar Ratio': round(calmar, 3),
    }, drawdown

if run:
    with st.spinner(f"Fetching {ticker} data and running {strategy}..."):
        try:
            df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
            if df.empty:
                st.error(f"No data found for {ticker}")
                st.stop()

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df = compute_signals(df, strategy)
            df['returns'] = df['Close'].pct_change()
            df['strategy_returns'] = df['returns'] * df['signal'].shift(1)
            df['equity'] = capital * (1 + df['strategy_returns']).cumprod()
            df['bh_equity'] = capital * (1 + df['returns']).cumprod()
            df.dropna(inplace=True)

            metrics, drawdown = compute_metrics(df['equity'])

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Return", f"{metrics['Total Return (%)']:.2f}%")
            col2.metric("CAGR", f"{metrics['CAGR (%)']:.2f}%")
            col3.metric("Sharpe Ratio", f"{metrics['Sharpe Ratio']:.3f}")

            col4, col5, col6 = st.columns(3)
            col4.metric("Sortino Ratio", f"{metrics['Sortino Ratio']:.3f}")
            col5.metric("Max Drawdown", f"{metrics['Max Drawdown (%)']:.2f}%")
            col6.metric("Calmar Ratio", f"{metrics['Calmar Ratio']:.3f}")

            fig, axes = plt.subplots(3, 1, figsize=(12, 12), facecolor='#0e1117')
            for ax in axes:
                ax.set_facecolor('#0e1117')
                ax.tick_params(colors='white')
                ax.spines['bottom'].set_color('white')
                ax.spines['left'].set_color('white')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

            axes[0].plot(df.index, df['Close'], color='#00d4ff', linewidth=1.5, label='Price')
            buy = df[df['signal'] == 1]
            sell = df[df['signal'] == -1]
            axes[0].scatter(buy.index, buy['Close'], marker='^', color='#00ff88', s=20, alpha=0.7, label='Buy')
            axes[0].scatter(sell.index, sell['Close'], marker='v', color='#ff4444', s=20, alpha=0.7, label='Sell')
            axes[0].set_title(f'{ticker} — {strategy.replace("_", " ").title()}', color='white', fontsize=14)
            axes[0].legend(facecolor='#0e1117', labelcolor='white', fontsize=8)
            axes[0].set_ylabel('Price ($)', color='white')

            axes[1].plot(df.index, df['equity'], color='#00d4ff', linewidth=1.5, label=strategy)
            axes[1].plot(df.index, df['bh_equity'], color='white', linewidth=1, linestyle='--', alpha=0.5, label='Buy & Hold')
            axes[1].set_title('Equity Curve', color='white', fontsize=12)
            axes[1].legend(facecolor='#0e1117', labelcolor='white', fontsize=8)
            axes[1].set_ylabel('Portfolio Value ($)', color='white')

            axes[2].fill_between(drawdown.index, drawdown.values, 0, color='#ff4444', alpha=0.6)
            axes[2].set_title('Drawdown', color='white', fontsize=12)
            axes[2].set_ylabel('Drawdown (%)', color='white')

            plt.tight_layout()
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='#0e1117')
            buf.seek(0)
            st.image(buf, use_container_width=True)
            plt.close()

            st.subheader("📊 Performance Summary")
            st.dataframe(pd.DataFrame([metrics]), use_container_width=True)

        except Exception as e:
            st.error(f"Error: {e}")
else:
    st.info("👈 Enter a stock ticker and click Run Backtest")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### 📊 Strategies
        - SMA Crossover
        - EMA Crossover
        - RSI Mean Reversion
        - MACD
        - Bollinger Band Breakout
        - Bollinger Band Mean Reversion
        - Momentum (ROC)
        - Buy & Hold (Benchmark)
        """)
    with col2:
        st.markdown("""
        ### 📈 Metrics
        - Total Return & CAGR
        - Sharpe & Sortino Ratio
        - Maximum Drawdown
        - Calmar Ratio
        """)
