import feedparser
import requests
from textblob import TextBlob
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import re

class NewsAnalyzer:
    def __init__(self):
        self.sources = {
            'BBC': [
                'http://feeds.bbci.co.uk/news/business/rss.xml',
                'http://feeds.bbci.co.uk/news/politics/rss.xml'
            ],
            'AlJazeera': [
                'https://www.aljazeera.com/xml/rss/all.xml'
            ],
            'CoinDesk': [
                'https://www.coindesk.com/arc/outboundfeeds/rss/?outputType=xml'
            ],
            'Cointelegraph': [
                'https://cointelegraph.com/rss'
            ],
            'CryptoSlate': [
                'https://cryptoslate.com/feed/'
            ]
        }
        self.coin_keywords = {
            'BTC': ['bitcoin', 'btc'],
            'ETH': ['ethereum', 'eth'],
            'BNB': ['binance coin', 'bnb'],
            'XRP': ['ripple', 'xrp'],
            'ADA': ['cardano', 'ada'],
            'SOL': ['solana', 'sol'],
            'DOT': ['polkadot', 'dot'],
            'DOGE': ['dogecoin', 'doge'],
            'AVAX': ['avalanche', 'avax'],
            'LTC': ['litecoin', 'ltc'],
            'LINK': ['chainlink', 'link'],
            'MATIC': ['polygon', 'matic'],
            'ATOM': ['cosmos', 'atom'],
            'UNI': ['uniswap', 'uni'],
            'XLM': ['stellar', 'xlm'],
            'VET': ['vechain', 'vet'],
            'ALGO': ['algorand', 'algo'],
            'FIL': ['filecoin', 'fil'],
            'XTZ': ['tezos', 'xtz'],
            'ETC': ['ethereum classic', 'etc'],
            'SHIB': ['shiba inu', 'shib'],
            'PEPE': ['pepe coin', 'pepe'],
            'FLOKI': ['floki', 'floki inu'],
            'BONK': ['bonk', 'bonk coin'],
            'WIF': ['dogwifhat', 'wif'],
            'JUP': ['jupiter', 'jup'],
            'ARB': ['arbitrum', 'arb'],
            'OP': ['optimism', 'op'],
            'RNDR': ['render', 'rndr'],
            'SEI': ['sei network', 'sei']
        }
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
    
    def extract_coin_symbols(self, text: str) -> List[str]:
        text_lower = text.lower()
        symbols = []
        for symbol, keywords in self.coin_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                symbols.append(symbol)
        return symbols
    
    async def fetch_news(self) -> List[dict]:
        all_news = []
        for source, urls in self.sources.items():
            for url in urls:
                try:
                    feed = feedparser.parse(url)
                    for entry in feed.entries:
                        if hasattr(entry, 'published_parsed') and entry.published_parsed is not None and isinstance(entry.published_parsed, tuple):
                            # published_parsed is a time.struct_time (tuple), safe to unpack
                            published = datetime(
                                entry.published_parsed[0],
                                entry.published_parsed[1],
                                entry.published_parsed[2],
                                entry.published_parsed[3],
                                entry.published_parsed[4],
                                entry.published_parsed[5]
                            )
                        else:
                            published = datetime.now()
                        # Only consider news from last 24 hours
                        if datetime.now() - published < timedelta(hours=24):
                            all_news.append({
                                'title': entry.title,
                                'summary': entry.summary if hasattr(entry, 'summary') else '',
                                'link': entry.link,
                                'source': source,
                                'published': published
                            })
                except Exception as e:
                    self.logger.error(f"Error fetching news from {source}: {e}")
        return all_news
    
    def analyze_sentiment(self, text: str) -> float:
        analysis = TextBlob(text)
        try:
            sentiment = analysis.sentiment
            # If sentiment is a method (cached_property), call it
            if callable(sentiment):
                sentiment = sentiment()
            polarity = getattr(sentiment, 'polarity', sentiment[0] if isinstance(sentiment, (tuple, list)) and len(sentiment) > 0 else 0.0)
        except Exception:
            polarity = 0.0
        return polarity
    
    async def get_market_sentiment(self) -> Tuple[float, Dict[str, float], List[dict]]:
        news_items = await self.fetch_news()
        overall_sentiment = 0
        coin_sentiments = {}
        relevant_news = []
        
        for item in news_items:
            # Combine title and summary for analysis
            content = f"{item['title']}. {item['summary']}"
            sentiment = self.analyze_sentiment(content)
            
            # Identify coins mentioned
            coin_symbols = self.extract_coin_symbols(content)
            
            # Update overall sentiment
            overall_sentiment += sentiment
            
            # Update coin-specific sentiments
            for symbol in coin_symbols:
                coin_sentiments[symbol] = coin_sentiments.get(symbol, 0) + sentiment
            
            # Add to relevant news if mentions any coin
            if coin_symbols:
                relevant_news.append({
                    'title': item['title'],
                    'source': item['source'],
                    'sentiment': sentiment,
                    'coins': coin_symbols,
                    'link': item['link'],
                    'published': item['published']
                })
        
        # Normalize overall sentiment
        if news_items:
            overall_sentiment /= len(news_items)
        
        return overall_sentiment, coin_sentiments, relevant_news
    
    def generate_news_insights(self, relevant_news: List[dict]) -> str:
        if not relevant_news:
            return "No significant news impacting cryptocurrencies"
        
        # Group by sentiment
        positive_news = [n for n in relevant_news if n['sentiment'] > 0.2]
        negative_news = [n for n in relevant_news if n['sentiment'] < -0.2]
        
        insights = []
        
        if positive_news:
            insights.append("Positive News Trends:")
            for news in positive_news[:3]:  # Top 3 positive news
                coins = ", ".join(news['coins'])
                insights.append(f"- {news['source']}: {news['title']} (Affects: {coins})")
        
        if negative_news:
            insights.append("\nNegative News Trends:")
            for news in negative_news[:3]:  # Top 3 negative news
                coins = ", ".join(news['coins'])
                insights.append(f"- {news['source']}: {news['title']} (Affects: {coins})")
        
        return "\n".join(insights)