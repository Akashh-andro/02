import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
import pandas_ta as ta
from scipy import stats
import json
import time

class MarketAnalyzer:
    def __init__(self):
        # Initialize MT5
        if not mt5.initialize():
            logging.error("Failed to initialize MT5")
            raise Exception("MT5 initialization failed")
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('market_analyzer.log'),
                logging.StreamHandler()
            ]
        )
        
        # Market session times (UTC)
        self.sessions = {
            "Sydney": {"start": "22:00", "end": "07:00"},
            "Tokyo": {"start": "00:00", "end": "09:00"},
            "London": {"start": "08:00", "end": "17:00"},
            "New York": {"start": "13:00", "end": "22:00"}
        }
        
        # Load market profiles if exists
        self._load_market_profiles()

    def _load_market_profiles(self):
        """Load market profiles from file"""
        try:
            with open('market_profiles.json', 'r') as f:
                self.market_profiles = json.load(f)
        except FileNotFoundError:
            self.market_profiles = {}

    def _save_market_profiles(self):
        """Save market profiles to file"""
        with open('market_profiles.json', 'w') as f:
            json.dump(self.market_profiles, f)

    def get_historical_data(self, symbol: str, timeframe: str = "H1", 
                          bars: int = 1000) -> Optional[pd.DataFrame]:
        """Get historical price data"""
        try:
            # Convert timeframe string to MT5 timeframe
            tf_map = {
                "M1": mt5.TIMEFRAME_M1,
                "M5": mt5.TIMEFRAME_M5,
                "M15": mt5.TIMEFRAME_M15,
                "M30": mt5.TIMEFRAME_M30,
                "H1": mt5.TIMEFRAME_H1,
                "H4": mt5.TIMEFRAME_H4,
                "D1": mt5.TIMEFRAME_D1,
                "W1": mt5.TIMEFRAME_W1,
                "MN1": mt5.TIMEFRAME_MN1
            }
            
            if timeframe not in tf_map:
                raise ValueError(f"Invalid timeframe: {timeframe}")
            
            # Get historical data
            rates = mt5.copy_rates_from_pos(symbol, tf_map[timeframe], 0, bars)
            if rates is None:
                raise Exception(f"Failed to get historical data for {symbol}")
            
            # Convert to DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            
            return df
            
        except Exception as e:
            logging.error(f"Error getting historical data: {str(e)}")
            return None

    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate advanced technical indicators"""
        try:
            # Trend indicators
            df['sma_20'] = ta.sma(df['close'], length=20)
            df['sma_50'] = ta.sma(df['close'], length=50)
            df['sma_200'] = ta.sma(df['close'], length=200)
            
            # MACD
            macd = ta.macd(df['close'])
            df['macd'] = macd['MACD_12_26_9']
            df['macd_signal'] = macd['MACDs_12_26_9']
            df['macd_hist'] = macd['MACDh_12_26_9']
            
            # RSI
            df['rsi'] = ta.rsi(df['close'], length=14)
            
            # Bollinger Bands
            try:
                bb = ta.bbands(df['close'], length=20, std=2.0)
                if isinstance(bb, pd.DataFrame):
                    df['bb_upper'] = bb['BBU_20_2.0']
                    df['bb_middle'] = bb['BBM_20_2.0']
                    df['bb_lower'] = bb['BBL_20_2.0']
                else:
                    raise ValueError("Invalid Bollinger Bands result")
            except Exception as e:
                logging.warning(f"Falling back to manual Bollinger Bands calculation: {str(e)}")
                # Manual calculation as fallback
                df['bb_middle'] = df['sma_20']
                std = df['close'].rolling(window=20).std()
                df['bb_upper'] = df['bb_middle'] + (std * 2)
                df['bb_lower'] = df['bb_middle'] - (std * 2)
            
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            
            # ATR
            df['atr'] = ta.atr(df['high'], df['low'], df['close'])
            
            # ADX
            try:
                adx = ta.adx(df['high'], df['low'], df['close'])
                df['adx'] = adx['ADX_14'] if isinstance(adx, pd.DataFrame) else 25.0
            except Exception as e:
                logging.warning(f"Error calculating ADX: {str(e)}")
                df['adx'] = 25.0
            
            # Stochastic
            try:
                stoch = ta.stoch(df['high'], df['low'], df['close'])
                df['stoch_k'] = stoch['STOCHk_14_3_3']
                df['stoch_d'] = stoch['STOCHd_14_3_3']
            except Exception as e:
                logging.warning(f"Error calculating Stochastic: {str(e)}")
                df['stoch_k'] = 50.0
                df['stoch_d'] = 50.0
            
            # Volume indicators
            if 'volume' in df.columns:
                try:
                    df['obv'] = ta.obv(df['close'], df['volume'])
                    df['mfi'] = ta.mfi(df['high'], df['low'], df['close'], df['volume'])
                except Exception as e:
                    logging.warning(f"Error calculating volume indicators: {str(e)}")
            
            # Custom indicators
            df['trend_strength'] = df['adx'].rolling(window=14).mean()
            df['volatility'] = df['atr'].rolling(window=20).mean()
            df['momentum'] = df['close'].pct_change(periods=10)
            
            return df.fillna(0)
        except Exception as e:
            logging.error(f"Error calculating technical indicators: {str(e)}")
            return df  # Return original dataframe on error

    def _calculate_market_profile(self, df: pd.DataFrame) -> Dict:
        """Calculate market profile metrics"""
        try:
            # Calculate price distribution
            price_range = df['close'].max() - df['close'].min()
            price_bins = np.linspace(df['close'].min(), df['close'].max(), 50)
            price_hist, _ = np.histogram(df['close'], bins=price_bins)
            
            # Calculate volume profile
            if 'volume' in df.columns:
                volume_profile = df.groupby(pd.cut(df['close'], price_bins))['volume'].sum()
            else:
                volume_profile = None
            
            # Calculate statistical measures
            returns = df['close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # Annualized volatility
            skewness = stats.skew(returns)
            kurtosis = stats.kurtosis(returns)
            
            # Calculate trend metrics
            trend_direction = 1 if df['close'].iloc[-1] > df['sma_50'].iloc[-1] else -1
            trend_strength = df['adx'].iloc[-1]
            
            # Calculate support and resistance levels
            recent_highs = df['high'].rolling(window=20).max()
            recent_lows = df['low'].rolling(window=20).min()
            
            resistance = recent_highs.iloc[-1] + df['atr'].iloc[-1]
            support = recent_lows.iloc[-1] - df['atr'].iloc[-1]
            
            return {
                "price_distribution": price_hist.tolist(),
                "price_bins": price_bins.tolist(),
                "volume_profile": volume_profile.to_dict() if volume_profile is not None else None,
                "volatility": volatility,
                "skewness": skewness,
                "kurtosis": kurtosis,
                "trend_direction": trend_direction,
                "trend_strength": trend_strength,
                "support": support,
                "resistance": resistance
            }
            
        except Exception as e:
            logging.error(f"Error calculating market profile: {str(e)}")
            return {}

    def _get_active_session(self) -> str:
        """Get current active trading session"""
        current_time = datetime.utcnow().strftime("%H:%M")
        
        for session, times in self.sessions.items():
            if times["start"] <= current_time <= times["end"]:
                return session
        
        return "Closed"

    def analyze_market(self, symbol: str, timeframe: str = "H1") -> Dict:
        """Analyze market conditions"""
        try:
            # Get historical data
            df = self.get_historical_data(symbol, timeframe)
            if df is None:
                return {}
            
            # Calculate technical indicators
            df = self._calculate_technical_indicators(df)
            
            # Calculate market profile
            market_profile = self._calculate_market_profile(df)
            
            # Get current market conditions
            current_price = df['close'].iloc[-1]
            current_volume = df['volume'].iloc[-1] if 'volume' in df.columns else None
            current_volatility = df['volatility'].iloc[-1]
            current_trend = df['trend_strength'].iloc[-1]
            
            # Get active session
            active_session = self._get_active_session()
            
            # Prepare analysis
            analysis = {
                "symbol": symbol,
                "timeframe": timeframe,
                "timestamp": datetime.now().isoformat(),
                "current_price": current_price,
                "current_volume": current_volume,
                "current_volatility": current_volatility,
                "current_trend": current_trend,
                "active_session": active_session,
                "market_profile": market_profile,
                "technical_indicators": {
                    "rsi": df['rsi'].iloc[-1],
                    "macd": df['macd'].iloc[-1],
                    "macd_signal": df['macd_signal'].iloc[-1],
                    "bb_upper": df['bb_upper'].iloc[-1],
                    "bb_middle": df['bb_middle'].iloc[-1],
                    "bb_lower": df['bb_lower'].iloc[-1],
                    "adx": df['adx'].iloc[-1],
                    "stoch_k": df['stoch_k'].iloc[-1],
                    "stoch_d": df['stoch_d'].iloc[-1]
                }
            }
            
            # Update market profiles
            self.market_profiles[symbol] = analysis
            self._save_market_profiles()
            
            return analysis
            
        except Exception as e:
            logging.error(f"Error analyzing market: {str(e)}")
            return {}

    def get_market_summary(self, symbol: str) -> Dict:
        """Get market summary for a symbol"""
        try:
            if symbol not in self.market_profiles:
                return {}
            
            profile = self.market_profiles[symbol]
            
            # Calculate market conditions
            conditions = []
            
            # Trend conditions
            if profile['current_trend'] > 25:
                conditions.append("Strong Trend")
            elif profile['current_trend'] > 15:
                conditions.append("Moderate Trend")
            else:
                conditions.append("Weak Trend")
            
            # Volatility conditions
            if profile['current_volatility'] > profile['market_profile']['volatility'] * 1.5:
                conditions.append("High Volatility")
            elif profile['current_volatility'] < profile['market_profile']['volatility'] * 0.5:
                conditions.append("Low Volatility")
            else:
                conditions.append("Normal Volatility")
            
            # RSI conditions
            rsi = profile['technical_indicators']['rsi']
            if rsi > 70:
                conditions.append("Overbought")
            elif rsi < 30:
                conditions.append("Oversold")
            
            # MACD conditions
            macd = profile['technical_indicators']['macd']
            macd_signal = profile['technical_indicators']['macd_signal']
            if macd > macd_signal:
                conditions.append("MACD Bullish")
            else:
                conditions.append("MACD Bearish")
            
            # Bollinger Bands conditions
            price = profile['current_price']
            bb_upper = profile['technical_indicators']['bb_upper']
            bb_lower = profile['technical_indicators']['bb_lower']
            if price > bb_upper:
                conditions.append("Price Above BB")
            elif price < bb_lower:
                conditions.append("Price Below BB")
            
            return {
                "symbol": symbol,
                "timestamp": profile['timestamp'],
                "current_price": profile['current_price'],
                "active_session": profile['active_session'],
                "market_conditions": conditions,
                "trend_strength": profile['current_trend'],
                "volatility": profile['current_volatility'],
                "support": profile['market_profile']['support'],
                "resistance": profile['market_profile']['resistance']
            }
            
        except Exception as e:
            logging.error(f"Error getting market summary: {str(e)}")
            return {}

    def get_correlation_matrix(self, symbols: List[str], timeframe: str = "H1") -> pd.DataFrame:
        """Calculate correlation matrix for multiple symbols"""
        try:
            # Get historical data for all symbols
            data = {}
            for symbol in symbols:
                df = self.get_historical_data(symbol, timeframe)
                if df is not None:
                    data[symbol] = df['close']
            
            # Create DataFrame with all close prices
            df = pd.DataFrame(data)
            
            # Calculate correlation matrix
            corr_matrix = df.corr()
            
            return corr_matrix
            
        except Exception as e:
            logging.error(f"Error calculating correlation matrix: {str(e)}")
            return pd.DataFrame()

    def __del__(self):
        """Cleanup when object is destroyed"""
        mt5.shutdown()

def main():
    # Example usage
    analyzer = MarketAnalyzer()
    
    # Analyze EURUSD
    analysis = analyzer.analyze_market("EURUSD", "H1")
    print("EURUSD Analysis:")
    print(json.dumps(analysis, indent=2))
    
    # Get market summary
    summary = analyzer.get_market_summary("EURUSD")
    print("\nEURUSD Market Summary:")
    print(json.dumps(summary, indent=2))
    
    # Calculate correlation matrix
    symbols = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]
    corr_matrix = analyzer.get_correlation_matrix(symbols)
    print("\nCorrelation Matrix:")
    print(corr_matrix)

if __name__ == "__main__":
    main() 