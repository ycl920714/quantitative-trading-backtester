from .engine import Backtester, BacktestResult, Trade
from .strategies import STRATEGY_REGISTRY, BaseStrategy
from .data_fetcher import fetch_stock_data, fetch_multiple, add_indicators
from .performance import compute_metrics, compare_results, print_report

__all__ = [
    "Backtester", "BacktestResult", "Trade",
    "STRATEGY_REGISTRY", "BaseStrategy",
    "fetch_stock_data", "fetch_multiple", "add_indicators",
    "compute_metrics", "compare_results", "print_report",
]
