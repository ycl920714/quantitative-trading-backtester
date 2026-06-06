# Quantitative Trading Strategy Backtester

A Python-based backtesting framework implementing 9 trading strategies with 20+ performance metrics and automated chart generation.

## Strategies
| Strategy | Description |
|----------|-------------|
| SMA Crossover | Dual SMA golden/death cross |
| EMA Crossover | Dual EMA crossover |
| RSI Mean Reversion | RSI oversold/overbought signals |
| MACD | MACD histogram direction |
| Bollinger Band Breakout | Momentum breakout |
| Bollinger Band Mean Reversion | Mean reversion |
| Momentum (ROC) | N-day Rate-of-Change |
| Dual Thrust | Volatility range breakout |
| Buy & Hold | Benchmark comparison |

## Performance Metrics
CAGR, Sharpe Ratio, Sortino Ratio, Calmar Ratio, Max Drawdown, VaR, CVaR, Win Rate, Profit Factor

## Usage
```bash
# Single ticker
python3 backtest_runner.py --ticker AAPL

# Multi-ticker comparison
python3 backtest_runner.py --ticker AAPL MSFT NVDA --strategy all

# Custom parameters
python3 backtest_runner.py --ticker SPY --strategy sma_crossover --fast 20 --slow 100
```

## Installation
```bash
pip install yfinance pandas numpy matplotlib backtrader
```

## Tech Stack
Python, yfinance, pandas, numpy, matplotlib, backtrader
