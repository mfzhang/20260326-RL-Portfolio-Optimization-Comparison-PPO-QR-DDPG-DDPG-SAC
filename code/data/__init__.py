"""
data – market-data fetching and feature engineering.

Exports
-------
DataProcessor : downloads prices, builds features, splits train/test sets
"""

from .data_processor import DataProcessor

__all__ = ["DataProcessor"]
