import streamlit as st
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backtester.data_fetcher import fetch_stock_data

st.set_page_config(page_title="Quantitative Trading Backtester", layout="wide")
st.title("📈 Quantitative Trading Strategy Backtester")
st.markdown("Backtest trading strategies on any stock using historical data")

with st.sidebar:
    st.header("⚙️ Settings")
    ticker = st.text_input("Stock Ticker", value="AAPL").upper()
    strategy = st.selectbox("Strategy", [
        "sma_crossover", "ema_crossover", "rsi_mean_reversion",
        "macd", "bb_breakout", "bb_mean_reversion",
        "momentum", "dual_thrust", "buy_and_hold"
    ])
    capital = st.number_input("Initial Capital ($)", value=10000, step=1000)
    run = st.button("🚀 Run Backtest", type="primary")

if run:
    with st.spinner(f"Running {strategy} on {ticker}..."):
        chart_path = f"backtest_results/{ticker}_{strategy}.png"
        summary_path = "backtest_results/backtest_summary.csv"

        if not os.path.exists(chart_path):
            st.warning(f"No results found for {ticker} {strategy}. Running backtest now...")
            os.system(f"python3 backtest_runner.py --ticker {ticker} --strategy {strategy} --capital {capital} --no-charts")

        if os.path.exists(summary_path):
            import pandas as pd
            df = pd.read_csv(summary_path)
            row = df[(df['Ticker'] == ticker) & (df['Strategy'].str.lower().str.replace(' ', '_') == strategy)]

            if not row.empty:
                r = row.iloc[0]
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Return", f"{r.get('Total Return (%)', 0):.2f}%")
                col2.metric("CAGR", f"{r.get('CAGR (%)', 0):.2f}%")
                col3.metric("Sharpe Ratio", f"{r.get('Sharpe Ratio', 0):.3f}")
                col4.metric("Max Drawdown", f"{r.get('Max Drawdown (%)', 0):.2f}%")

        if os.path.exists(chart_path):
            st.image(chart_path, caption=f"{ticker} - {strategy}", use_container_width=True)
        else:
            st.info(f"Run this in Terminal first:\npython3 backtest_runner.py --ticker {ticker}")
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
        - Dual Thrust
        - Buy & Hold (Benchmark)
        """)
    with col2:
        st.markdown("""
        ### 📈 Metrics
        - Total Return & CAGR
        - Sharpe & Sortino Ratio
        - Maximum Drawdown
        - Win Rate & Profit Factor
        - VaR & CVaR
        - Calmar Ratio
        """)
