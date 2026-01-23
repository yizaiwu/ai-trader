"""
AI Trader - Backtesting framework for algorithmic trading strategies.

This package provides tools for backtesting trading strategies using a pandas-based engine.
"""

__version__ = "0.3.3"

# Import main utilities for convenience
from ai_trader.utils.backtest import run_backtest

__all__ = [
    "run_backtest",
]
