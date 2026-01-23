import pandas as pd
import pandas_ta as ta
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

from ai_trader.core.logging import get_logger

logger = get_logger(__name__)

class BaseStrategy(ABC):
    """
    Base class for pandas-based trading strategies.
    
    Strategies should implement `generate_signals` which takes a DataFrame
    and returns a DataFrame with a 'signal' column (1: buy, -1: sell, 0: hold).
    """
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params = params or {}
        if self.params:
            params_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
            logger.info(f"{self.__class__.__name__} initialized with {params_str}")
        else:
            logger.info(f"{self.__class__.__name__} initialized with no parameters")

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals from historical data.
        
        Args:
            df: DataFrame with OHLCV data.
            
        Returns:
            DataFrame with additional columns for indicators and a 'signal' column.
        """
        pass

    def log(self, message: str, dt: Optional[Any] = None):
        """Standardized logging for strategies."""
        time_str = dt.isoformat() if hasattr(dt, 'isoformat') else str(dt)
        logger.info(f"{time_str} │ {message}")
