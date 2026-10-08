# Trading Strategy Backtester

A personal project I built in Python to test a simple question: do common technical trading rules beat buying and holding once trading costs are included and the rules are tested on data they were not tuned on?

**Live app:** https://quantitative-trading-backtester-yclin.streamlit.app/

Yun-Chen Lin, BSc Finance and Business, University of Sussex

## What it does

- Backtests nine rule-based strategies on daily US stock and ETF prices and compares each with buy-and-hold
- Charges a transaction cost on every position change
- Executes every signal at the next day's open, so a strategy never trades on information it would not have had
- Splits history into an in-sample and an out-of-sample period, and in *Parameter sensitivity* mode picks parameters on the first period only before testing them on the second
- Reports CAGR, volatility, Sharpe, Sortino, maximum drawdown, Calmar, time in market, and historical VaR and CVaR
- Runs a "strategy study" across a basket of tickers to see whether any result holds beyond one stock
- Backtests three Value at Risk models (historical simulation, normal, EWMA) with the Kupiec, Christoffersen and Basel traffic-light tests

## Strategies

| Strategy | Type | Rule |
|---|---|---|
| SMA Crossover | Trend following | Long when the 50-day average is above the 200-day average |
| EMA Crossover | Trend following | Same idea with 12 and 26-day exponential averages |
| MACD | Trend following | Long when the MACD histogram is positive |
| Momentum (ROC) | Momentum | Long when the 60-day return is positive |
| Bollinger Band Breakout | Momentum | Long after a close above the upper band, out after a close below the lower band |
| Dual Thrust | Volatility breakout | Long after a close breaks open + k × recent range |
| RSI Mean Reversion | Mean reversion | Buy when RSI < 30, hold until RSI > 70 |
| Bollinger Band Mean Reversion | Mean reversion | Buy after a close below the lower band, sell back at the moving average |
| Buy & Hold | Benchmark | Always long |

Strategies are long only by default. Short selling can be switched on.

## Methodology

**Timing.** A signal uses prices up to the close of day *t* and is executed at the open of day *t+1*. In the vectorised engine the daily return is

```
r_t = (1 + p_old × gap_t) × (1 + p_new × intraday_t) − 1 − cost × |p_new − p_old|
```

where `gap_t = Open_t / Close_{t−1} − 1`, `intraday_t = Close_t / Open_t − 1`, `p_new` is yesterday's signal and `p_old` the signal from the day before. The event-driven engine (`backtester/engine.py`) follows the same rule and gives the same results.

**Costs.** A fixed number of basis points per unit of position change (default 10 bps in the app).

**Out-of-sample testing.** History is split by date (default 70/30). The parameter grid search ranks parameter pairs by in-sample Sharpe and reports how the best pair does out of sample, which shows how much of the in-sample result was overfitting.

**Benchmark.** Buy at the first close and hold, no costs.

**VaR backtesting.** Each day's one-day VaR is estimated only from earlier returns and compared with that day's actual return. The Kupiec test checks whether the share of exceptions matches the confidence level, the Christoffersen test checks whether exceptions cluster in time, and the Basel traffic light classifies the number of 99% exceptions in the last 250 days (green 0 to 4, yellow 5 to 9, red 10 or more).

## Findings

See [FINDINGS.md](FINDINGS.md). In short: over the last 10 years on eight large US stocks and ETFs, buy-and-hold beat every strategy on average return and Sharpe ratio. Only 4 of 64 strategy and stock combinations had a higher Sharpe ratio.

## Limitations

- Survivorship bias: the default tickers are large companies that still exist today
- Cash earns 0% when a strategy is out of the market
- No bid-ask spread modelling, market impact, borrow fees or taxes
- Strategies hold cash during the indicator warm-up period while buy-and-hold is already invested
- One market and one period, mostly a rising US market
- Monte Carlo resampling assumes daily returns are independent, which ignores volatility clustering

## Project structure

```
app.py                  Streamlit web app
backtest_runner.py      Command-line runner, saves charts and a CSV summary
backtester/
  strategies.py         The nine strategies
  vectorised.py         Vectorised engine and metrics used by the app
  risk.py               VaR forecasts and VaR backtesting tests
  engine.py             Event-driven engine with trade-by-trade records
  performance.py        Full metric set for the command-line runner
  data_fetcher.py       Price download via yfinance
```

## Running it locally

```bash
pip install -r requirements.txt matplotlib
streamlit run app.py

# command line, all strategies on several tickers
python backtest_runner.py --ticker SPY AAPL MSFT --strategy all
```

## Tech

Python, pandas, NumPy, SciPy, Plotly, Streamlit, yfinance, matplotlib
