## Findings

*Run in October 2026 with the Strategy study mode: 10 years of daily data for SPY, QQQ, AAPL, MSFT, JPM, XOM, JNJ and KO, default parameters, 10 bps per trade, long only. Numbers will move slightly as new data comes in.*

| | Avg CAGR | Avg Sharpe | Avg max drawdown | Time in market | Beat B&H on Sharpe |
|---|---|---|---|---|---|
| Highest Sharpe strategy (EMA Crossover) | 9.1% | 0.61 | -28.4% | 66% | 0 of 8 |
| Lowest Sharpe strategy (BB Mean Reversion) | 4.3% | 0.36 | -33.9% | 19% | 0 of 8 |
| Buy & Hold | 18.2% | 0.81 | -39.2% | 100% | |

**1. Buy-and-hold won.** Across 64 strategy and stock combinations, only 4 had a higher Sharpe ratio than simply holding the stock. No strategy beat buy-and-hold on average return or on average Sharpe.

**2. The main cost was being out of the market.** The strategies were invested between 19% and 71% of the time. Over a decade in which these stocks mostly went up, the days spent in cash missed more gains than the strategies saved by avoiding falls.

**3. They did reduce risk, but not by enough.** Average maximum drawdowns were 28% to 38%, against 39% for buy-and-hold. Lower drawdowns did not make up for the lower returns, so risk-adjusted performance was still worse.

**4. Trading costs hit the busy strategies hardest.** MACD changed position about 20 times a year and lost about 2.2 percentage points of CAGR to costs (8.1% before costs, 5.9% after). The 50/200-day SMA crossover traded about once a year and lost only 0.2 points.

**5. The few wins did not hold up.** Only 4 combinations beat buy-and-hold over the full 10 years. In the last 30% of the period the winners were mostly different strategies, and no strategy beat buy-and-hold on more than 3 of the 8 stocks in either period. That looks more like luck than a real edge.

**What I would test next:** a period with a long bear market (for example 2000 to 2012), a wider universe that includes delisted stocks to remove survivorship bias, and paying interest on idle cash.

## VaR backtest findings

*Run in October 2026 with the VaR backtest mode: SPY, 10 years of daily returns, 99% one-day VaR, 250-day estimation window. About 2,260 days were tested, so a correct model should have about 23 exceptions.*

| Model | Exceptions | Exception rate | Kupiec p-value | Independence p-value | Worst 250 days (Basel) |
|---|---|---|---|---|---|
| Historical simulation | 38 | 1.7% | 0.003 | 0.000 | 11 (red), to Sep 2022 |
| Normal | 66 | 2.9% | 0.000 | 0.001 | 20 (red), to Sep 2022 |
| EWMA (lambda 0.94) | 53 | 2.3% | 0.000 | 0.042 | 13 (red), to Mar 2020 |

**1. All three models underestimated risk.** Every model was breached more often than the 1% it promised, and the Kupiec test rejects all three.

**2. Assuming normal returns was the worst choice.** The normal model had almost three times the expected number of exceptions. Daily stock returns have fatter tails than a normal distribution, so a model built on it puts the 99% loss too close to zero.

**3. The bigger problem is timing.** The exceptions came in clusters around March 2020 and the 2022 sell-off, which is why the independence test rejects the historical and normal models. A model that looks back 250 days reacts slowly when volatility jumps. EWMA, which weights recent days more heavily, reduced the clustering, but it still assumes normal returns, so it was breached too often overall.

**4. A green light today says little about a bad year.** All three models are in the Basel green zone for the last 250 days, yet each one would have been in the red zone during 2020 or 2022. Judging a risk model only on a calm recent period would have missed this.

**What I would test next:** combining the two fixes, for example EWMA volatility with a fat-tailed (Student t) distribution or filtered historical simulation, and checking expected shortfall as well as VaR.
