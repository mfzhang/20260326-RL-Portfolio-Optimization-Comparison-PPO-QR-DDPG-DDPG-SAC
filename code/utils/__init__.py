"""
utils – shared metric and weight utilities.

Exports
-------
normalize_weights          : project weight vector onto the simplex
calculate_portfolio_metrics: compute Sharpe, Sortino, drawdown, CVaR, etc.
calculate_rolling_sharpe   : rolling-window Sharpe ratio series
save_results_to_excel      : persist results DataFrames to .xlsx
"""

from .utils import (
    calculate_portfolio_metrics,
    calculate_rolling_sharpe,
    normalize_weights,
    save_results_to_excel,
)

__all__ = [
    "normalize_weights",
    "calculate_portfolio_metrics",
    "calculate_rolling_sharpe",
    "save_results_to_excel",
]
