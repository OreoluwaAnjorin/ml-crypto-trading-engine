from enum import Enum
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

class RiskLevel(Enum):
    LOW = 'low'
    MEDIUM = 'medium'
    HIGH = 'high'

class TradingAction(Enum):
    BUY = 'buy'
    SELL = 'sell'
    HOLD = 'hold'

class MarketTrend(Enum):
    BULLISH = 'bullish'
    BEARISH = 'bearish'
    NEUTRAL = 'neutral'

@dataclass
class MarketData:
    symbol: str
    price: float
    volume: float
    price_change_24h: float
    volatility: float
    timestamp: datetime

@dataclass
class TradingSignal:
    symbol: str
    action: TradingAction
    confidence: float
    price: float
    timestamp: datetime 
    stop_loss: float
    take_profit: float
    reason: str
    position_size: float
    min_hold_time: int
    max_hold_time: int
    expected_duration: int
    volatility_score: float
    urgency_level: str
    leverage: float
    trend: MarketTrend
    news_sentiment: float = 0.0
    news_impact: str = ""

    def __post_init__(self):
        if not isinstance(self.action, TradingAction):
            self.action = TradingAction(self.action)
        if not isinstance(self.trend, MarketTrend):
            self.trend = MarketTrend(self.trend)

@dataclass
class ActiveTrade:
    trade_id: str
    symbol: str
    action: TradingAction
    entry_price: float
    invested_amount: float
    risk_level: RiskLevel
    signal: TradingSignal
    start_time: datetime
    current_price: float = 0.0
    profit_loss: float = 0.0
    last_updated: Optional[datetime] = None