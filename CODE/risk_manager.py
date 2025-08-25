from models import RiskLevel, TradingAction, MarketTrend
from typing import Dict

class RiskManager:
    def __init__(self):
        self.risk_params = {
            RiskLevel.LOW: {
                'min_position_size': 0.05,
                'max_position_size': 0.10,
                'stop_loss_pct': 0.03,
                'take_profit_pct': 0.06,
                'confidence_threshold': 0.75,
                'min_hold_minutes': 30,
                'max_hold_minutes': 120,
                'volatility_threshold': 5.0,
                'leverage': 1.0,
                'max_leverage': 2.0
            },
            RiskLevel.MEDIUM: {
                'min_position_size': 0.08,
                'max_position_size': 0.15,
                'stop_loss_pct': 0.05,
                'take_profit_pct': 0.12,
                'confidence_threshold': 0.65,
                'min_hold_minutes': 45,
                'max_hold_minutes': 180,
                'volatility_threshold': 3.0,
                'leverage': 2.0,
                'max_leverage': 5.0
            },
            RiskLevel.HIGH: {
                'min_position_size': 0.10,
                'max_position_size': 0.20,
                'stop_loss_pct': 0.08,
                'take_profit_pct': 0.20,
                'confidence_threshold': 0.55,
                'min_hold_minutes': 60,
                'max_hold_minutes': 240,
                'volatility_threshold': 2.0,
                'leverage': 5.0,
                'max_leverage': 10.0
            }
        }
    
    def get_risk_params(self, risk_level: RiskLevel) -> Dict:
        return self.risk_params[risk_level]
    
    def calculate_leverage(self, volatility: float, confidence: float, risk_level: RiskLevel) -> float:
        params = self.risk_params[risk_level]
        base_leverage = params['leverage']
        leverage_multiplier = min(confidence * 1.5, 2.0)
        volatility_factor = min(volatility / 10, 1.5)
        calculated_leverage = base_leverage * leverage_multiplier * volatility_factor
        return min(calculated_leverage, params['max_leverage'])
    
    def determine_trend(self, action: TradingAction, confidence: float) -> MarketTrend:
        if action == TradingAction.BUY and confidence > 0.7:
            return MarketTrend.BULLISH
        elif action == TradingAction.SELL and confidence > 0.7:
            return MarketTrend.BEARISH
        return MarketTrend.NEUTRAL