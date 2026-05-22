"""
Utility functions for the project.
"""

from typing import Dict, List

import numpy as np
import pandas as pd


def calculate_portfolio_metrics(
    portfolio_values: List[float],
    initial_value: float = 1000000,
    risk_free_rate: float = 0.045,
) -> Dict:
    """
    Calculate comprehensive portfolio performance metrics.

    Args:
        portfolio_values: List of portfolio values over time
        initial_value: Initial portfolio value
        risk_free_rate: Annual risk-free rate

    Returns:
        Dictionary of performance metrics
    """
    portfolio_values = np.array(portfolio_values)
    returns = np.diff(portfolio_values) / portfolio_values[:-1]

    # Total return
    total_return = (portfolio_values[-1] - initial_value) / initial_value

    # Annualized return (252 trading days)
    days = len(returns)
    annual_return = (1 + total_return) ** (252 / days) - 1 if days > 0 else 0

    # Volatility (annualized)
    volatility = np.std(returns) * np.sqrt(252)

    # Sharpe Ratio
    sharpe_ratio = (
        (annual_return - risk_free_rate) / volatility if volatility > 0 else 0
    )

    # Sortino Ratio
    downside_returns = returns[returns < 0]
    downside_std = (
        np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0
    )
    sortino_ratio = (
        (annual_return - risk_free_rate) / downside_std if downside_std > 0 else 0
    )

    # Maximum Drawdown
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (peak - portfolio_values) / peak
    max_drawdown = np.max(drawdown)

    # Calmar Ratio
    calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0

    # CVaR (Conditional Value at Risk) at 5%
    sorted_returns = np.sort(returns)
    cvar_index = int(0.05 * len(sorted_returns))
    cvar = np.mean(sorted_returns[:cvar_index]) if cvar_index > 0 else 0

    # Win Rate
    win_rate = np.sum(returns > 0) / len(returns) if len(returns) > 0 else 0

    # Average Win/Loss
    winning_returns = returns[returns > 0]
    losing_returns = returns[returns < 0]
    avg_win = np.mean(winning_returns) if len(winning_returns) > 0 else 0
    avg_loss = np.mean(losing_returns) if len(losing_returns) > 0 else 0

    # Profit Factor
    total_wins = np.sum(winning_returns) if len(winning_returns) > 0 else 0
    total_losses = abs(np.sum(losing_returns)) if len(losing_returns) > 0 else 0
    profit_factor = total_wins / total_losses if total_losses > 0 else 0

    metrics = {
        "total_return": total_return,
        "annual_return": annual_return * 100,  # Percentage
        "volatility": volatility * 100,  # Percentage
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "max_drawdown": -max_drawdown * 100,  # Percentage
        "calmar_ratio": calmar_ratio,
        "cvar_5": cvar * 100,  # Percentage
        "win_rate": win_rate * 100,  # Percentage
        "avg_win": avg_win * 100,  # Percentage
        "avg_loss": avg_loss * 100,  # Percentage
        "profit_factor": profit_factor,
    }

    return metrics


def normalize_weights(weights: np.ndarray) -> np.ndarray:
    """
    Normalize portfolio weights to sum to 1.

    Args:
        weights: Array of portfolio weights

    Returns:
        Normalized weights
    """
    weights = np.clip(weights, 0, 1)
    weight_sum = np.sum(weights)

    if weight_sum > 0:
        return weights / weight_sum
    else:
        return np.ones_like(weights) / len(weights)


def calculate_drawdown_series(portfolio_values: np.ndarray) -> np.ndarray:
    """
    Calculate drawdown series from portfolio values.

    Args:
        portfolio_values: Array of portfolio values

    Returns:
        Array of drawdowns at each time step
    """
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (peak - portfolio_values) / peak
    return -drawdown * 100  # Return as negative percentage


def calculate_rolling_sharpe(
    returns: np.ndarray, window: int = 20, risk_free_rate: float = 0.045
) -> np.ndarray:
    """
    Calculate rolling Sharpe ratio.

    Args:
        returns: Array of returns
        window: Rolling window size
        risk_free_rate: Annual risk-free rate

    Returns:
        Array of rolling Sharpe ratios
    """
    daily_rf = risk_free_rate / 252

    rolling_sharpe = []
    for i in range(window, len(returns) + 1):
        window_returns = returns[i - window : i]
        excess_returns = window_returns - daily_rf
        _std = np.std(excess_returns)
        sharpe = np.mean(excess_returns) / _std * np.sqrt(252) if _std > 0 else 0.0
        rolling_sharpe.append(sharpe)

    # Pad with NaN for initial window
    rolling_sharpe = [np.nan] * (window - 1) + rolling_sharpe

    return np.array(rolling_sharpe)


def format_metrics_table(metrics: Dict) -> pd.DataFrame:
    """
    Format metrics dictionary as a nice table.

    Args:
        metrics: Dictionary of metrics

    Returns:
        Formatted DataFrame
    """
    formatted = []

    metric_names = {
        "annual_return": "Annual Return (%)",
        "volatility": "Volatility (%)",
        "sharpe_ratio": "Sharpe Ratio",
        "sortino_ratio": "Sortino Ratio",
        "max_drawdown": "Max Drawdown (%)",
        "calmar_ratio": "Calmar Ratio",
        "cvar_5": "CVaR (5%) (%)",
        "win_rate": "Win Rate (%)",
        "profit_factor": "Profit Factor",
    }

    for key, display_name in metric_names.items():
        if key in metrics:
            formatted.append({"Metric": display_name, "Value": f"{metrics[key]:.2f}"})

    return pd.DataFrame(formatted)


def save_results_to_excel(results: Dict[str, pd.DataFrame], filename: str):
    """
    Save multiple DataFrames to Excel with different sheets.

    Args:
        results: Dictionary mapping sheet names to DataFrames
        filename: Output Excel filename
    """
    with pd.ExcelWriter(filename, engine="openpyxl") as writer:
        for sheet_name, df in results.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"Results saved to: {filename}")


if __name__ == "__main__":
    print("Utility functions module loaded successfully")
