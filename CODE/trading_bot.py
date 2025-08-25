import numpy as np
import pandas as pd
from datetime import datetime
import requests
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import ta
from ta.volatility import BollingerBands
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator
import warnings
from models import MarketData, TradingSignal, TradingAction, RiskLevel, MarketTrend, ActiveTrade
from risk_manager import RiskManager
from news_analyzer import NewsAnalyzer
import uuid

warnings.filterwarnings('ignore')

class CryptoTradingBot:
    def __init__(self):
        self.coin_list = [
            # Low Risk (Large-cap, stable coins)
            'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT',
            'SOLUSDT', 'DOTUSDT', 'DOGEUSDT', 'AVAXUSDT', 'LTCUSDT',
            
            # Medium Risk (Established projects)
            'LINKUSDT', 'MATICUSDT', 'ATOMUSDT', 'UNIUSDT', 'XLMUSDT',
            'VETUSDT', 'ALGOUSDT', 'FILUSDT', 'XTZUSDT', 'ETCUSDT',
            
            # High Risk (High-volatility coins)
            'SHIBUSDT', 'PEPEUSDT', 'FLOKIUSDT', 'BONKUSDT', 'WIFUSDT',
            'JUPUSDT', 'ARBUSDT', 'OPUSDT', 'RNDRUSDT', 'SEIUSDT'
        ]
        self.market_data: Dict[str, MarketData] = {}
        self.ml_model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.risk_manager = RiskManager()
        self.news_analyzer = NewsAnalyzer()
        self.overall_news_sentiment = 0.0
        self.coin_news_sentiments = {}
        self.news_insights = ""
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        self.active_trades: Dict[str, ActiveTrade] = {}
    
    async def update_news_analysis(self):
        try:
            (self.overall_news_sentiment, 
             self.coin_news_sentiments, 
             relevant_news) = await self.news_analyzer.get_market_sentiment()
            self.news_insights = self.news_analyzer.generate_news_insights(relevant_news)
            self.logger.info(f"Updated news sentiment: {self.overall_news_sentiment:.2f}")
        except Exception as e:
            self.logger.error(f"Error updating news analysis: {e}")
    
    def get_news_insights(self) -> str:
        return self.news_insights
    
    async def fetch_market_data(self, symbol: str) -> Optional[MarketData]:
        try:
            url = f"https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}"
            response = requests.get(url)
            
            if response.status_code == 200:
                data = response.json()
                volatility = abs(float(data['priceChangePercent']))
                
                return MarketData(
                    symbol=symbol,
                    price=float(data['lastPrice']),
                    volume=float(data['volume']),
                    price_change_24h=float(data['priceChangePercent']),
                    volatility=volatility,
                    timestamp=datetime.now()
                )
        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {e}")
        return None
    
    async def get_historical_data(self, symbol: str, interval: str = '1h', limit: int = 100) -> pd.DataFrame:
        try:
            url = "https://api.binance.com/api/v3/klines"
            params = {'symbol': symbol, 'interval': interval, 'limit': limit}
            response = requests.get(url, params=params)
            data = response.json()
            
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            df[numeric_cols] = df[numeric_cols].astype(float)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        except Exception as e:
            self.logger.error(f"Error fetching historical data for {symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
            
        # Convert to float to prevent type errors
        # Calculate indicators
        df['sma_20'] = SMAIndicator(df['close'], window=20).sma_indicator()
        df['sma_50'] = SMAIndicator(df['close'], window=50).sma_indicator()
        df['ema_12'] = EMAIndicator(df['close'], window=12).ema_indicator()
        df['ema_26'] = EMAIndicator(df['close'], window=26).ema_indicator()
        df['rsi'] = RSIIndicator(df['close'], window=14).rsi()
        
        # MACD calculation
        macd = MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_hist'] = macd.macd_diff()
        
        # Bollinger Bands
        bollinger = BollingerBands(df['close'])
        df['bb_upper'] = bollinger.bollinger_hband()
        df['bb_lower'] = bollinger.bollinger_lband()
        df['bb_middle'] = bollinger.bollinger_mavg()
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        
        # Feature engineering
        df['price_change'] = df['close'].pct_change()
        df['volume_change'] = df['volume'].pct_change()
        df['volatility'] = df['close'].rolling(window=10).std()
        
        # Target variable
        df['future_return'] = df['close'].shift(-1) / df['close'] - 1
        df['target'] = (df['future_return'] > 0.02).astype(int)
        
        return df.dropna()
    
    async def train_ml_model(self):
        self.logger.info("Training ML model...")
        all_features = []
        all_targets = []
        
        for symbol in self.coin_list:
            try:
                df = await self.get_historical_data(symbol, interval='1h', limit=500)
                if df.empty:
                    continue
                    
                df = self.calculate_technical_indicators(df)
                if df.empty:
                    continue
                
                feature_columns = [
                    'sma_20', 'sma_50', 'ema_12', 'ema_26', 'rsi', 'macd', 
                    'macd_signal', 'bb_upper', 'bb_lower', 'bb_middle',
                    'volume_sma', 'price_change', 'volume_change', 'volatility'
                ]
                
                # Ensure all features are float
                features = df[feature_columns].astype(float).values
                targets = df['target'].values
                
                # Check for NaNs safely
                valid_idx = ~pd.isna(features).any(axis=1) & ~pd.isna(targets)
                features = features[valid_idx]
                targets = targets[valid_idx]
                
                if len(features) > 0:
                    all_features.append(features)
                    all_targets.append(targets)
            except Exception as e:
                self.logger.error(f"Error processing {symbol} for training: {e}")
                continue
        
        if not all_features:
            self.logger.error("No valid training data found")
            return
            
        X = np.vstack(all_features)
        y = np.hstack(all_targets)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.ml_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        self.ml_model.fit(X_train_scaled, y_train)
        
        train_acc = accuracy_score(y_train, self.ml_model.predict(X_train_scaled))
        test_acc = accuracy_score(y_test, self.ml_model.predict(X_test_scaled))
        
        self.logger.info(f"Model trained - Train accuracy: {train_acc:.3f}, Test accuracy: {test_acc:.3f}")
        self.is_trained = True
        return test_acc
    
    async def analyze_coin(self, symbol: str, risk_level: RiskLevel) -> Optional[TradingSignal]:
        try:
            market_data = await self.fetch_market_data(symbol)
            if not market_data:
                return None
                
            df = await self.get_historical_data(symbol, interval='1h', limit=100)
            if df.empty:
                return None
                
            df = self.calculate_technical_indicators(df)
            if df.empty or len(df) < 2:
                return None
            
            latest = df.iloc[-1]
            
            feature_columns = [
                'sma_20', 'sma_50', 'ema_12', 'ema_26', 'rsi', 'macd', 
                'macd_signal', 'bb_upper', 'bb_lower', 'bb_middle',
                'volume_sma', 'price_change', 'volume_change', 'volatility'
            ]
            
            # Convert features to float safely
            try:
                features = latest[feature_columns].astype(float).to_numpy().reshape(1, -1)
            except Exception as e:
                self.logger.error(f"Error converting features to float for {symbol}: {e}")
                return None
            
            # Skip if features contain NaNs
            if pd.isna(features).any():
                self.logger.warning(f"NaN features found for {symbol}")
                return None
            
            ml_confidence = 0.5
            if self.is_trained and self.ml_model:
                try:
                    features_scaled = self.scaler.transform(features)
                    ml_prediction = self.ml_model.predict_proba(features_scaled)[0]
                    ml_confidence = max(ml_prediction)
                except Exception as e:
                    self.logger.warning(f"ML prediction failed for {symbol}: {e}")
                    ml_confidence = 0.5
            
            signals = []
            rsi = latest['rsi']
            if rsi < 30:
                signals.append(('BUY', 0.7, 'RSI oversold'))
            elif rsi > 70:
                signals.append(('SELL', 0.7, 'RSI overbought'))
            
            if latest['macd'] > latest['macd_signal']:
                signals.append(('BUY', 0.6, 'MACD bullish'))
            else:
                signals.append(('SELL', 0.6, 'MACD bearish'))
            
            if latest['close'] > latest['sma_20'] > latest['sma_50']:
                signals.append(('BUY', 0.8, 'Price above MAs'))
            elif latest['close'] < latest['sma_20'] < latest['sma_50']:
                signals.append(('SELL', 0.8, 'Price below MAs'))
            
            if latest['close'] < latest['bb_lower']:
                signals.append(('BUY', 0.6, 'Price below lower BB'))
            elif latest['close'] > latest['bb_upper']:
                signals.append(('SELL', 0.6, 'Price above upper BB'))
            
            buy_signals = [s for s in signals if s[0] == 'BUY']
            sell_signals = [s for s in signals if s[0] == 'SELL']
            
            if len(buy_signals) > len(sell_signals):
                action = TradingAction.BUY
                confidence = (sum(s[1] for s in buy_signals) / len(buy_signals)) * ml_confidence
                reason = '; '.join(s[2] for s in buy_signals)
            elif len(sell_signals) > len(buy_signals):
                action = TradingAction.SELL
                confidence = (sum(s[1] for s in sell_signals) / len(sell_signals)) * ml_confidence
                reason = '; '.join(s[2] for s in sell_signals)
            else:
                action = TradingAction.HOLD
                confidence = 0.5
                reason = 'Mixed signals'
            
            # Apply news sentiment adjustment
            coin_base = symbol.replace("USDT", "")
            news_sentiment = self.coin_news_sentiments.get(coin_base, 0.0)
            news_impact = ""
            
            if news_sentiment > 0.2:
                confidence *= 1.2  # Boost confidence for positive news
                news_impact = "Positive news sentiment"
                reason += "; Positive news"
            elif news_sentiment < -0.2:
                confidence *= 0.8  # Reduce confidence for negative news
                news_impact = "Negative news sentiment"
                reason += "; Negative news"
            
            # Apply overall market sentiment adjustment
            confidence *= (1 + self.overall_news_sentiment * 0.3)
            
            # Cap confidence at 0.95 to avoid overconfidence
            confidence = min(confidence, 0.95)
            
            risk_params = self.risk_manager.get_risk_params(risk_level)
            
            if (confidence < risk_params['confidence_threshold'] or 
                market_data.volatility < risk_params['volatility_threshold']):
                action = TradingAction.HOLD
                if confidence < risk_params['confidence_threshold']:
                    reason = f'Low confidence ({confidence:.2f})'
                else:
                    reason = f'Insufficient volatility ({market_data.volatility:.1f}%)'
            
            position_size, min_hold, max_hold, expected_duration, urgency = self.calculate_position_size_and_timing(
                symbol, confidence, market_data.volatility, risk_level
            )
            
            leverage = self.risk_manager.calculate_leverage(
                market_data.volatility, 
                confidence,
                risk_level
            )
            trend = self.risk_manager.determine_trend(action, confidence)
            
            current_price = market_data.price
            if action == TradingAction.BUY:
                stop_loss = current_price * (1 - risk_params['stop_loss_pct'])
                take_profit = current_price * (1 + risk_params['take_profit_pct'])
            elif action == TradingAction.SELL:
                stop_loss = current_price * (1 + risk_params['stop_loss_pct'])
                take_profit = current_price * (1 - risk_params['take_profit_pct'])
            else:
                stop_loss = take_profit = current_price
                position_size = min_hold = max_hold = expected_duration = 0
                urgency = "N/A"
            
            return TradingSignal(
                symbol=symbol,
                action=action,
                confidence=confidence,
                price=current_price,
                timestamp=datetime.now(),
                stop_loss=stop_loss,
                take_profit=take_profit,
                reason=reason,
                position_size=position_size,
                min_hold_time=min_hold,
                max_hold_time=max_hold,
                expected_duration=expected_duration,
                volatility_score=market_data.volatility,
                urgency_level=urgency,
                leverage=leverage,
                trend=trend,
                news_sentiment=news_sentiment,
                news_impact=news_impact
            )
        except Exception as e:
            self.logger.error(f"Error analyzing {symbol}: {e}")
            return None
    
    def calculate_position_size_and_timing(self, symbol: str, confidence: float, 
                                         volatility: float, risk_level: RiskLevel) -> Tuple[float, int, int, int, str]:
        risk_params = self.risk_manager.get_risk_params(risk_level)
        base_position = risk_params['min_position_size']
        max_additional = risk_params['max_position_size'] - base_position
        position_size = base_position + (max_additional * confidence)
        position_size = min(position_size, risk_params['max_position_size'])
        
        min_hold = risk_params['min_hold_minutes']
        max_hold = risk_params['max_hold_minutes']
        
        if volatility > 10:
            time_multiplier = 0.6
            urgency = "HIGH - Enter within 5 minutes"
        elif volatility > 7:
            time_multiplier = 0.8
            urgency = "MEDIUM - Enter within 10 minutes"
        elif volatility > 4:
            time_multiplier = 1.0
            urgency = "NORMAL - Enter within 15 minutes"
        else:
            time_multiplier = 1.3
            urgency = "LOW - Can wait up to 30 minutes"
        
        expected_min = int(min_hold * time_multiplier)
        expected_max = int(max_hold * time_multiplier)
        expected_duration = int(expected_max * 0.65)
        
        return position_size, expected_min, expected_max, expected_duration, urgency
    
    async def get_optimal_signals(self, risk_level: RiskLevel, duration_type: str = "all") -> List[TradingSignal]:
        """Get top 30 signals optimized for risk level and duration"""
        self.logger.info(f"Analyzing {len(self.coin_list)} coins for {risk_level.value} risk level...")
        signals = []
        for symbol in self.coin_list:
            signal = await self.analyze_coin(symbol, risk_level)
            if signal and signal.action != TradingAction.HOLD:
                # Duration filtering based on max hold time
                is_short_term = signal.max_hold_time <= 120
                
                if duration_type == "short" and is_short_term:
                    signals.append(signal)
                elif duration_type == "long" and not is_short_term:
                    signals.append(signal)
                elif duration_type == "all":
                    signals.append(signal)
        
        # Sort by confidence descending
        signals.sort(key=lambda x: x.confidence, reverse=True)
        
        # Return top 30 coins
        return signals[:30]
    
    def should_stop_trading(self, initial_value: float, current_portfolio_value: float, 
                           daily_loss_limit: float = 0.05) -> Tuple[bool, str]:
        performance = (current_portfolio_value - initial_value) / initial_value
        if performance < -daily_loss_limit:
            return True, f"Daily loss limit exceeded: {performance:.2%}"
        if datetime.now().weekday() in [5, 6]:
            return True, "Weekend trading pause"
        return False, "Continue trading"
    
    def get_optimal_entry_timing(self) -> Dict:
        current_hour = datetime.now().hour
        peak_hours = [13, 14, 15, 16, 17, 18, 19, 20]
        moderate_hours = [9, 10, 11, 12, 21, 22]
        
        if current_hour in peak_hours:
            return {
                'market_session': "PEAK",
                'advice': "Execute immediately - high volume/volatility"
            }
        elif current_hour in moderate_hours:
            return {
                'market_session': "MODERATE",
                'advice': "Execute within 15 minutes - moderate conditions"
            }
        else:
            return {
                'market_session': "SLOW",
                'advice': "Be cautious - lower volume period"
            }
    
    async def monitor_active_trades(self) -> List[Tuple[str, ActiveTrade, TradingSignal]]:
        """Update active trades and check for signal changes"""
        signal_changes = []
        for trade_id, trade in list(self.active_trades.items()):
            current_data = await self.fetch_market_data(trade.symbol)
            if current_data:
                trade.current_price = current_data.price
                trade.last_updated = datetime.now()
                
                # Calculate P&L
                if trade.action == TradingAction.BUY:
                    trade.profit_loss = (trade.current_price - trade.entry_price) * (trade.invested_amount / trade.entry_price)
                else:  # SELL
                    trade.profit_loss = (trade.entry_price - trade.current_price) * (trade.invested_amount / trade.entry_price)
                
                # Re-analyze for signal changes
                new_signal = await self.analyze_coin(trade.symbol, trade.risk_level)
                if new_signal and new_signal.action != trade.signal.action:
                    signal_changes.append((trade_id, trade, new_signal))
        return signal_changes

    def add_active_trade(self, signal: TradingSignal, amount: float, risk_level: RiskLevel) -> str:
        """Add a new trade to monitor"""
        trade_id = f"{signal.symbol}_{uuid.uuid4().hex[:6]}"
        self.active_trades[trade_id] = ActiveTrade(
            trade_id=trade_id,
            symbol=signal.symbol,
            action=signal.action,
            entry_price=signal.price,
            invested_amount=amount,
            risk_level=risk_level,
            signal=signal,
            start_time=datetime.now(),
            current_price=signal.price
        )
        return trade_id