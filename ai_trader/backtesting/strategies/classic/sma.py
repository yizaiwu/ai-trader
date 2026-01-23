import pandas as pd
import pandas_ta as ta
from typing import Dict, Any, Optional

from ai_trader.backtesting.strategies.base import BaseStrategy

class CrossSMAStrategy(BaseStrategy):
    """
    Moving Average Crossover Strategy using pandas-ta.
    """
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        # Default params
        if params is None:
            params = {}
        params.setdefault('fast', 5)
        params.setdefault('slow', 37)
        super().__init__(params)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        # Calculate indicators using pandas-ta
        fast_sma = ta.sma(df['close'], length=self.params['fast'])
        slow_sma = ta.sma(df['close'], length=self.params['slow'])
        
        df['fast_ma'] = fast_sma
        df['slow_ma'] = slow_sma
        
        # Generate signals: 1 when fast crosses above slow, -1 when slow crosses above fast
        df['signal'] = 0
        
        # Buy signal: fast > slow and fast was <= slow on previous bar
        buy_cond = (df['fast_ma'] > df['slow_ma']) & (df['fast_ma'].shift(1) <= df['slow_ma'].shift(1))
        # Sell signal: fast < slow and fast was >= slow on previous bar
        sell_cond = (df['fast_ma'] < df['slow_ma']) & (df['fast_ma'].shift(1) >= df['slow_ma'].shift(1))
        
        df.loc[buy_cond, 'signal'] = 1
        df.loc[sell_cond, 'signal'] = -1
        
        return df

if __name__ == "__main__":
    from ai_trader.utils.backtest import run_backtest
    import yfinance as yf
    
    # Example usage
    data = yf.download("AAPL", start="2023-01-01")
    # Yfinance returns multi-index columns sometimes, flatten if needed
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data.columns = [c.lower() for c in data.columns]
        
    result = run_backtest(
        strategy_class=CrossSMAStrategy,
        data_source=data,
        cash=1000000,
        strategy_params={"fast": 10, "slow": 30}
    )
    print(f"Final Portfolio Value: ${result.final_value:,.2f}")
