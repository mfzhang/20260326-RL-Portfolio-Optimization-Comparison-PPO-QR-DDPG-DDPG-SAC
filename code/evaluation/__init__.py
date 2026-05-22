"""
evaluation – agent evaluation and benchmark comparisons.

Exports
-------
EvaluateStrategies    : loads and evaluates trained DRL agents on test data
BacktestBenchmark     : event-driven backtester for classical strategies
BenchmarkStrategies   : equal-weight, 60/40, risk-parity, all-weather, momentum
"""

from .benchmark_strategies import BacktestBenchmark, BenchmarkStrategies

# EvaluateStrategies is imported separately to avoid a circular import:
# evaluate.py itself imports from evaluation.benchmark_strategies, so
# pulling EvaluateStrategies through __init__.py would create a load loop.
# Import it directly when needed: from evaluation.evaluate import EvaluateStrategies
try:
    from .evaluate import EvaluateStrategies
except ImportError:
    pass  # heavy deps (torch, SB3) may not be installed in all environments

__all__ = ["EvaluateStrategies", "BacktestBenchmark", "BenchmarkStrategies"]
