import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import ccxt
import logging
from typing import Dict, List, Optional, Union

class DataManager:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.data_cache = {}
        self.exchange = None
        self.timeframes = {
            '1m': 60,
            '5m': 300,
            '15m': 900,
            '30m': 1800,
            '1h': 3600,
            '4h': 14400,
            '1d': 86400
        }
    
    def initialize_exchange(self, exchange_id: str, api_key: str = None, secret: str = None):
        """Initialize cryptocurrency exchange connection"""
        try:
            exchange_class = getattr(ccxt, exchange_id)
            self.exchange = exchange_class({
                'apiKey': api_key,
                'secret': secret,
                'enableRateLimit': True
            })
            self.logger.info(f"Successfully initialized {exchange_id} exchange")
        except Exception as e:
            self.logger.error(f"Error initializing exchange: {str(e)}")
            raise
    
    def fetch_crypto_data(self, symbol: str, timeframe: str, 
                         start_time: Optional[datetime] = None,
                         end_time: Optional[datetime] = None) -> pd.DataFrame:
        """Fetch cryptocurrency data from exchange"""
        try:
            if not self.exchange:
                raise ValueError("Exchange not initialized")
            
            # Convert timeframe to exchange format
            tf = self._convert_timeframe(timeframe)
            
            # Fetch OHLCV data
            ohlcv = self.exchange.fetch_ohlcv(
                symbol,
                timeframe=tf,
                since=int(start_time.timestamp() * 1000) if start_time else None,
                limit=1000
            )
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Cache the data
            cache_key = f"{symbol}_{timeframe}"
            self.data_cache[cache_key] = df
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching crypto data: {str(e)}")
            raise
    
    def fetch_stock_data(self, symbol: str, start_date: Optional[str] = None,
                        end_date: Optional[str] = None) -> pd.DataFrame:
        """Fetch stock data using yfinance"""
        try:
            # Download data
            data = yf.download(
                symbol,
                start=start_date,
                end=end_date,
                progress=False
            )
            
            # Cache the data
            self.data_cache[symbol] = data
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error fetching stock data: {str(e)}")
            raise
    
    def get_cached_data(self, symbol: str, timeframe: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Retrieve data from cache"""
        cache_key = f"{symbol}_{timeframe}" if timeframe else symbol
        return self.data_cache.get(cache_key)
    
    def update_data(self, symbol: str, timeframe: Optional[str] = None) -> pd.DataFrame:
        """Update cached data with latest information"""
        try:
            if timeframe:
                return self.fetch_crypto_data(symbol, timeframe)
            else:
                return self.fetch_stock_data(symbol)
        except Exception as e:
            self.logger.error(f"Error updating data: {str(e)}")
            raise
    
    def calculate_technical_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate technical indicators for the data"""
        indicators = {}
        
        try:
            # Moving Averages
            indicators['sma_20'] = data['close'].rolling(window=20).mean()
            indicators['sma_50'] = data['close'].rolling(window=50).mean()
            indicators['ema_20'] = data['close'].ewm(span=20, adjust=False).mean()
            
            # RSI
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            indicators['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = data['close'].ewm(span=12, adjust=False).mean()
            exp2 = data['close'].ewm(span=26, adjust=False).mean()
            indicators['macd'] = exp1 - exp2
            indicators['macd_signal'] = indicators['macd'].ewm(span=9, adjust=False).mean()
            indicators['macd_hist'] = indicators['macd'] - indicators['macd_signal']
            
            # Bollinger Bands
            indicators['bb_middle'] = data['close'].rolling(window=20).mean()
            bb_std = data['close'].rolling(window=20).std()
            indicators['bb_upper'] = indicators['bb_middle'] + (bb_std * 2)
            indicators['bb_lower'] = indicators['bb_middle'] - (bb_std * 2)
            
            # ATR
            high_low = data['high'] - data['low']
            high_close = np.abs(data['high'] - data['close'].shift())
            low_close = np.abs(data['low'] - data['close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            indicators['atr'] = true_range.rolling(14).mean()
            
            # Volume indicators
            indicators['volume_sma'] = data['volume'].rolling(window=20).mean()
            indicators['obv'] = (np.sign(data['close'].diff()) * data['volume']).fillna(0).cumsum()
            
            return indicators
            
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {str(e)}")
            raise
    
    def _convert_timeframe(self, timeframe: str) -> str:
        """Convert timeframe to exchange format"""
        return timeframe
    
    def get_available_symbols(self) -> List[str]:
        """Get list of available trading symbols"""
        try:
            if self.exchange:
                return self.exchange.load_markets().keys()
            else:
                # For stocks, you might want to implement a different method
                return []
        except Exception as e:
            self.logger.error(f"Error getting available symbols: {str(e)}")
            return []
    
    def get_market_info(self, symbol: str) -> Dict:
        """Get market information for a symbol"""
        try:
            if self.exchange:
                return self.exchange.fetch_ticker(symbol)
            else:
                # For stocks, use yfinance
                ticker = yf.Ticker(symbol)
                return ticker.info
        except Exception as e:
            self.logger.error(f"Error getting market info: {str(e)}")
            raise
    
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess market data"""
        try:
            # Remove duplicate indices
            data = data[~data.index.duplicated(keep='first')]
            
            # Handle missing values
            data = data.fillna(method='ffill')
            
            # Remove outliers (values more than 3 standard deviations from mean)
            for column in ['open', 'high', 'low', 'close']:
                mean = data[column].mean()
                std = data[column].std()
                data[column] = data[column].clip(mean - 3*std, mean + 3*std)
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error cleaning data: {str(e)}")
            raise 