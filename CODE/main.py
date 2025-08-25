import asyncio
import logging
from datetime import datetime
from trading_bot import CryptoTradingBot
from models import RiskLevel, TradingAction
import uuid

async def run_trading_bot():
    print("🚀 Advanced Crypto Trading Bot")
    print("🔍 Tracking cryptocurrencies with risk analysis")
    print("📰 Analyzing market news...")
    print("🧠 Training machine learning model...")
    
    bot = CryptoTradingBot()
    await bot.update_news_analysis()
    await bot.train_ml_model()
    
    cycle_count = 0
    while True:
        try:
            print("\n" + "="*60)
            print(f"📅 Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("="*60)
            
            # Risk level selection
            print("\n🔍 Select Risk Level:")
            print("1. Low Risk (Conservative)")
            print("2. Medium Risk (Balanced)")
            print("3. High Risk (Aggressive)")
            risk_choice = input("Enter choice (1/2/3): ").strip()
            
            risk_level = RiskLevel.MEDIUM
            if risk_choice == "1":
                risk_level = RiskLevel.LOW
            elif risk_choice == "3":
                risk_level = RiskLevel.HIGH
            
            # Duration preference
            print("\n⏱️ Select Trade Duration:")
            print("1. Short-term trades (quick gains)")
            print("2. Long-term trades (sustained growth)")
            print("3. Best opportunities (any duration)")
            duration_choice = input("Enter choice (1/2/3): ").strip()
            
            duration_type = "all"
            if duration_choice == "1":
                duration_type = "short"
            elif duration_choice == "2":
                duration_type = "long"
            
            # Get optimized signals
            print(f"\n🔍 Finding top opportunities for {risk_level.value} risk...")
            signals = await bot.get_optimal_signals(risk_level, duration_type)
            
            if not signals:
                print("\n⚠️ No strong trading signals found. Market conditions may be uncertain.")
            else:
                print(f"\n💎 Top {len(signals)} Optimized Trading Opportunities:")
                for i, signal in enumerate(signals, 1):
                    # Risk category mapping
                    risk_category = "HIGH RISK"
                    if signal.symbol in ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT',
                                      'SOLUSDT', 'DOTUSDT', 'DOGEUSDT', 'AVAXUSDT', 'LTCUSDT']:
                        risk_category = "LOW RISK"
                    elif signal.symbol in ['LINKUSDT', 'MATICUSDT', 'ATOMUSDT', 'UNIUSDT', 'XLMUSDT',
                                         'VETUSDT', 'ALGOUSDT', 'FILUSDT', 'XTZUSDT', 'ETCUSDT']:
                        risk_category = "MEDIUM RISK"
                    
                    # Duration classification
                    duration = "SHORT-TERM" if signal.max_hold_time <= 120 else "LONG-TERM"
                    
                    # Format output
                    action_str = signal.action.value.upper()
                    trend_str = signal.trend.value.upper()
                    
                    if action_str == "BUY":
                        action_emoji = "📈"
                    elif action_str == "SELL":
                        action_emoji = "📉"
                    else:
                        action_emoji = "⏸️"
                    
                    print(f"\n{i}. {signal.symbol} {action_emoji} {risk_category}")
                    print(f"   Action: {action_str} | Duration: {duration} | Trend: {trend_str}")
                    print(f"   Confidence: {signal.confidence:.1%} | Volatility: {signal.volatility_score:.1f}%")
                    print(f"   Price: ${signal.price:.4f} | Leverage: {signal.leverage:.1f}x")
                    print(f"   Position: {signal.position_size:.0%} | Hold: {signal.min_hold_time}-{signal.max_hold_time} mins")
                    print(f"   TP: ${signal.take_profit:.4f} | SL: ${signal.stop_loss:.4f}")
                    if signal.news_impact:
                        print(f"   News Impact: {signal.news_impact} ({signal.news_sentiment:.2f})")
                    print(f"   Reason: {signal.reason}")
            
            # Add trade monitoring option
            if signals:
                try:
                    monitor_choice = input("\n📊 Monitor a trade? Enter signal number (or Enter to skip): ").strip()
                    if monitor_choice.isdigit():
                        idx = int(monitor_choice) - 1
                        if 0 <= idx < len(signals):
                            try:
                                amount = float(input("💰 Enter USD amount to invest: "))
                                trade_id = bot.add_active_trade(signals[idx], amount, risk_level)
                                print(f"✅ Monitoring trade: {trade_id}")
                            except ValueError:
                                print("⚠️ Invalid amount. Trade not monitored.")
                except ValueError:
                    print("⚠️ Invalid input. Trade not monitored.")
            
            # Display active trades
            if bot.active_trades:
                print("\n🔔 Active Trade Monitoring")
                for trade in bot.active_trades.values():
                    status = "🟢 PROFIT" if trade.profit_loss >= 0 else "🔴 LOSS"
                    print(f"{trade.trade_id}: {trade.symbol} {trade.action.value} | "
                          f"P&L: ${abs(trade.profit_loss):.2f} ({status}) | "
                          f"Current: ${trade.current_price:.4f}")
            
            # Display news insights
            print("\n📰 Market News Insights:")
            print(bot.get_news_insights())
            
            timing = bot.get_optimal_entry_timing()
            print(f"\n⏰ Market Timing: {timing['market_session']} session")
            print(f"   Advice: {timing['advice']}")
            
            stop, reason = bot.should_stop_trading(10000, 9500)
            if stop:
                print(f"\n⚠️ Trading Paused: {reason}")
            
            print("\n" + "="*60)
            wait_time = 30  # 30 seconds
            print(f"⏳ Next analysis in {wait_time//60} minutes...")
            
            # Update news periodically (every 60 cycles = 30 minutes)
            cycle_count += 1
            if cycle_count % 60 == 0:
                print("\n🔄 Updating news analysis...")
                await bot.update_news_analysis()
            
            # Monitor trades every 10 minutes (20 cycles)
            if cycle_count % 20 == 0 and bot.active_trades:
                print("\n🔍 Monitoring active trades...")
                signal_changes = await bot.monitor_active_trades()
                if signal_changes:
                    print("\n🚨 SIGNAL CHANGE ALERTS:")
                    for trade_id, trade, new_signal in signal_changes:
                        print(f"{trade.symbol} signal changed from {trade.signal.action.value} to {new_signal.action.value}")
                        print(f"  Original entry: ${trade.entry_price:.4f} | Current: ${trade.current_price:.4f}")
                        print(f"  P&L: ${trade.profit_loss:.2f}")
                        print(f"  New confidence: {new_signal.confidence:.1%}")
                        print(f"  Recommendation: {'Hold' if new_signal.action == trade.action else 'Consider exit'}")
                        print("-"*40)
            
            await asyncio.sleep(wait_time)
            
        except KeyboardInterrupt:
            print("\n🔴 Bot stopped by user")
            break
        except Exception as e:
            print(f"⚠️ Unexpected error: {e}")
            await asyncio.sleep(60)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_trading_bot())