import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional, Type, Union, Dict, Any

from ai_trader.core.logging import get_logger
from ai_trader.backtesting.strategies.base import BaseStrategy

logger = get_logger(__name__)

class BacktestResult:
    """Container for backtest results."""
    def __init__(
        self,
        strategy_name: str,
        initial_value: float,
        final_value: float,
        df: pd.DataFrame,
        trades: List[Dict[str, Any]]
    ):
        self.strategy_name = strategy_name
        self.initial_value = initial_value
        self.final_value = final_value
        self.df = df
        self.trades = trades
        self.profit_loss = final_value - initial_value
        self.return_pct = (final_value / initial_value - 1) * 100 if initial_value != 0 else 0

def run_backtest(
    strategy_class: Type[BaseStrategy],
    data_source: Union[pd.DataFrame, str, Path],
    cash: float = 1000000,
    commission: float = 0.001425,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    strategy_params: Optional[Dict[str, Any]] = None,
    **kwargs
) -> BacktestResult:
    """
    Run an iterative backtest using a pandas-based strategy.
    
    Args:
        strategy_class: The strategy class to use.
        data_source: DataFrame or path to CSV.
        cash: Initial cash.
        commission: Commission rate.
        start_date: Start date for backtest.
        end_date: End date for backtest.
        strategy_params: Dictionary of parameters for the strategy.
    """
    # Load data
    if isinstance(data_source, (str, Path)):
        df = pd.read_csv(data_source, parse_dates=True, index_col=0)
    else:
        df = data_source.copy()

    # Filter by date range
    if start_date:
        df = df[df.index >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df.index <= pd.to_datetime(end_date)]

    # Initialize strategy
    strategy = strategy_class(params=strategy_params)
    
    # Generate signals
    df = strategy.generate_signals(df)
    
    # Ensure 'signal' column exists
    if 'signal' not in df.columns:
        raise ValueError("Strategy must generate a 'signal' column (1: Buy, -1: Sell, 0: Hold)")

    # Iterative backtest
    current_cash = cash
    position = 0
    trades = []
    
    portfolio_values = []
    
    for i in range(len(df)):
        price = df['close'].iloc[i]
        date = df.index[i]
        signal = df['signal'].iloc[i]
        
        # Simple Logic: 1 = Buy All, -1 = Sell All
        if signal == 1 and position == 0:
            # Buy
            shares = int(current_cash / (price * (1 + commission)))
            if shares > 0:
                cost = shares * price
                comm_paid = cost * commission
                current_cash -= (cost + comm_paid)
                position = shares
                trades.append({'date': date, 'type': 'BUY', 'price': price, 'shares': shares, 'comm': comm_paid})
                logger.info(f"{date.date()} │ BUY {shares} shares at ${price:.2f}")

        elif signal == -1 and position > 0:
            # Sell
            proceeds = position * price
            comm_paid = proceeds * commission
            current_cash += (proceeds - comm_paid)
            trades.append({'date': date, 'type': 'SELL', 'price': price, 'shares': position, 'comm': comm_paid})
            logger.info(f"{date.date()} │ SELL {position} shares at ${price:.2f}")
            position = 0
            
        portfolio_values.append(current_cash + position * price)

    df['portfolio_value'] = portfolio_values
    final_value = portfolio_values[-1]
    
    result = BacktestResult(
        strategy_name=strategy_class.__name__,
        initial_value=cash,
        final_value=final_value,
        df=df,
        trades=trades
    )
    
    logger.info(f"Backtest Finished: Final Value = ${final_value:,.2f} ({result.return_pct:+.2f}%)")
    return result
