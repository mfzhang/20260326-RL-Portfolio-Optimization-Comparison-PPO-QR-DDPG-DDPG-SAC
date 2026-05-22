"""
code – DRL Portfolio Optimisation package.

Subpackages
-----------
agents      : DDPG, QR-DDPG agent implementations
data        : market data fetching and feature engineering
environment : gymnasium portfolio trading environment
training    : agent training orchestration
evaluation  : evaluation and benchmark strategies
analysis    : figure generation and ablation studies
utils       : shared metrics and weight utilities
production  : FastAPI inference service
"""

from .agents import DDPGAgent, QRDDPGAgent, ReplayBuffer
from .data import DataProcessor
from .environment import PortfolioEnv
from .evaluation import BacktestBenchmark, BenchmarkStrategies
from .utils import calculate_portfolio_metrics, normalize_weights

__all__ = [
    "DDPGAgent",
    "QRDDPGAgent",
    "ReplayBuffer",
    "BacktestBenchmark",
    "BenchmarkStrategies",
    "DataProcessor",
    "PortfolioEnv",
    "calculate_portfolio_metrics",
    "normalize_weights",
]
