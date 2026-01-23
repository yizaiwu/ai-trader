"""
Indicators for ai-trader.
Note: backtrader based indicators have been removed and are awaiting refactoring to pandas-ta.
"""
from ai_trader.core.logging import get_logger

logger = get_logger(__name__)

# Stubs for indicators previously used in backtrader
class RSRS: pass
class NormRSRS: pass
class RecentHigh: pass
class DailyCandleVolatility: pass
class AverageVolatility: pass
class DiffHighLow: pass
class TripleRSI:
    def __init__(self, *args, **kwargs):
        pass
class DoubleTop: pass
class VCPPattern: pass
class AlphaRSIPro: pass
class AdaptiveRSI: pass
class HybridAlphaRSI: pass
