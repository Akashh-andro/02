import tensorflow as tf
import numpy as np
import pandas as pd
import MetaTrader5 as mt5
from datetime import datetime
import pandas_ta as ta
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Optional

class UniversalForexPredictor:
    def __init__(self):
        self.pair_profiles = {
            'EURUSD': {'pip_value': 0.0001, 'session': 'NY/London', 'volatility_factor': 1.0},
            'USDJPY': {'pip_value': 0.01, 'session': 'Tokyo', 'volatility_factor': 1.2},
            'GBPNZD': {'pip_value': 0.0001, 'session': 'London', 'volatility_factor': 1.5},
            'GBPUSD': {'pip_value': 0.0001, 'session': 'London/NY', 'volatility_factor': 1.1},
            'USDCHF': {'pip_value': 0.0001, 'session': 'NY/London', 'volatility_factor': 0.9},
            'AUDUSD': {'pip_value': 0.0001, 'session': 'Sydney/NY', 'volatility_factor': 1.3},
            'USDCAD': {'pip_value': 0.0001, 'session': 'NY/London', 'volatility_factor': 1.0},
            'NZDUSD': {'pip_value': 0.0001, 'session': 'Sydney/NY', 'volatility_factor': 1.4},
            'EURGBP': {'pip_value': 0.0001, 'session': 'London', 'volatility_factor': 1.2},
            'EURJPY': {'pip_value': 0.01, 'session': 'Tokyo/London', 'volatility_factor': 1.3}
        }
        self.scaler = StandardScaler()
        self.feature_columns = [
            'open', 'high', 'low', 'close', 'volume',
            'rsi', 'macd', 'macd_signal', 'macd_hist',
            'bb_upper', 'bb_middle', 'bb_lower',
            'atr', 'adx', 'cci', 'mfi',
            'stoch_k', 'stoch_d', 'williams_r',
            'obv', 'vwap', 'supertrend'
        ]
        try:
            self.base_model = tf.keras.models.load_model('models/universal_forex.h5')
        except:
            print("Warning: Model not found. Please train the model first.")
            self.base_model = None

    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate advanced technical indicators"""
        # RSI
        df['rsi'] = ta.rsi(df['close'], length=14)
        
        # MACD
        macd = ta.macd(df['close'])
        df['macd'] = macd['MACD_12_26_9']
        df['macd_signal'] = macd['MACDs_12_26_9']
        df['macd_hist'] = macd['MACDh_12_26_9']
        
        # Bollinger Bands
        bb = ta.bbands(df['close'])
        df['bb_upper'] = bb['BBU_20_2.0']
        df['bb_middle'] = bb['BBM_20_2.0']
        df['bb_lower'] = bb['BBL_20_2.0']
        
        # ATR
        df['atr'] = ta.atr(df['high'], df['low'], df['close'])
        
        # ADX
        df['adx'] = ta.adx(df['high'], df['low'], df['close'])['ADX_14']
        
        # CCI
        df['cci'] = ta.cci(df['high'], df['low'], df['close'])
        
        # MFI
        df['mfi'] = ta.mfi(df['high'], df['low'], df['close'], df['volume'])
        
        # Stochastic
        stoch = ta.stoch(df['high'], df['low'], df['close'])
        df['stoch_k'] = stoch['STOCHk_14_3_3']
        df['stoch_d'] = stoch['STOCHd_14_3_3']
        
        # Williams %R
        df['williams_r'] = ta.willr(df['high'], df['low'], df['close'])
        
        # OBV
        df['obv'] = ta.obv(df['close'], df['volume'])
        
        # VWAP
        df['vwap'] = ta.vwap(df['high'], df['low'], df['close'], df['volume'])
        
        # Supertrend
        supertrend = ta.supertrend(df['high'], df['low'], df['close'])
        df['supertrend'] = supertrend['SUPERT_7_3.0']
        
        return df

    def _normalize(self, pair: str, data: pd.DataFrame) -> pd.DataFrame:
        """Enhanced normalization with pair-specific adjustments"""
        if isinstance(data, pd.DataFrame):
            # Calculate technical indicators
            data = self._calculate_technical_indicators(data)
            
            # Get pair-specific volatility factor
            vol_factor = self.pair_profiles.get(pair, {}).get('volatility_factor', 1.0)
            
            # Scale price data using ATR
            atr = data['atr']
            for col in ['open', 'high', 'low', 'close']:
                if col in data.columns:
                    data[col] = (data[col] - data[col].mean()) / (atr * vol_factor)
            
            # Scale technical indicators
            for col in self.feature_columns:
                if col in data.columns and col not in ['open', 'high', 'low', 'close']:
                    data[col] = (data[col] - data[col].mean()) / data[col].std()
            
            return data.fillna(0)
        return data

    def _denormalize(self, pair: str, prediction: np.ndarray) -> np.ndarray:
        """Enhanced denormalization with pair-specific adjustments"""
        if isinstance(prediction, np.ndarray):
            # Get the last known price and volatility factor
            last_price = mt5.symbol_info_tick(pair).last
            vol_factor = self.pair_profiles.get(pair, {}).get('volatility_factor', 1.0)
            
            # Apply pair-specific scaling
            return prediction * last_price * vol_factor
        return prediction

    def predict(self, pair: str, data: pd.DataFrame) -> float:
        """Enhanced prediction with confidence scoring"""
        if self.base_model is None:
            raise ValueError("Model not initialized. Please train the model first.")
        
        # Normalize data with technical indicators
        normalized_data = self._normalize(pair, data)
        
        # Prepare input features
        features = normalized_data[self.feature_columns].values
        
        # Make prediction
        prediction = self.base_model.predict(features)
        
        # Apply pair-specific adjustments
        adjusted_prediction = self._denormalize(pair, prediction)
        
        # Calculate confidence score
        confidence = self._calculate_confidence(pair, normalized_data, adjusted_prediction)
        
        return adjusted_prediction * confidence

    def _calculate_confidence(self, pair: str, data: pd.DataFrame, prediction: float) -> float:
        """Calculate prediction confidence based on multiple factors"""
        confidence_factors = []
        
        # Trend strength (ADX)
        adx = data['adx'].iloc[-1]
        if adx > 25:  # Strong trend
            confidence_factors.append(1.2)
        elif adx > 20:  # Moderate trend
            confidence_factors.append(1.1)
        else:  # Weak trend
            confidence_factors.append(0.9)
        
        # Volatility regime
        atr = data['atr'].iloc[-1]
        avg_atr = data['atr'].mean()
        if atr > avg_atr * 1.5:  # High volatility
            confidence_factors.append(0.8)
        elif atr < avg_atr * 0.5:  # Low volatility
            confidence_factors.append(1.1)
        else:  # Normal volatility
            confidence_factors.append(1.0)
        
        # Technical indicator alignment
        indicators_aligned = 0
        total_indicators = 0
        
        # RSI
        rsi = data['rsi'].iloc[-1]
        if (prediction > 0 and rsi < 30) or (prediction < 0 and rsi > 70):
            indicators_aligned += 1
        total_indicators += 1
        
        # MACD
        macd = data['macd'].iloc[-1]
        macd_signal = data['macd_signal'].iloc[-1]
        if (prediction > 0 and macd > macd_signal) or (prediction < 0 and macd < macd_signal):
            indicators_aligned += 1
        total_indicators += 1
        
        # Bollinger Bands
        bb_upper = data['bb_upper'].iloc[-1]
        bb_lower = data['bb_lower'].iloc[-1]
        close = data['close'].iloc[-1]
        if (prediction > 0 and close < bb_lower) or (prediction < 0 and close > bb_upper):
            indicators_aligned += 1
        total_indicators += 1
        
        # Add indicator alignment factor
        confidence_factors.append(1 + (indicators_aligned / total_indicators) * 0.2)
        
        # Calculate final confidence score
        confidence = np.prod(confidence_factors)
        
        # Ensure confidence is within reasonable bounds
        return max(0.5, min(1.5, confidence))

    def calculate_pip_risk(self, pair: str, risk_percent: float) -> float:
        """Enhanced risk calculation with pair-specific adjustments"""
        if pair not in self.pair_profiles:
            raise ValueError(f"Pair {pair} not found in profiles")
            
        pip_value = self.pair_profiles[pair]['pip_value']
        vol_factor = self.pair_profiles[pair]['volatility_factor']
        balance = mt5.account_info().balance
        risk_amount = balance * risk_percent / 100
        
        # Adjust risk based on volatility factor
        adjusted_risk = risk_amount / vol_factor
        
        pips = adjusted_risk / (pip_value * 100000)
        return round(pips, 1)

    def get_active_session(self, pair: str) -> bool:
        """Enhanced session detection with overlap periods"""
        if pair not in self.pair_profiles:
            raise ValueError(f"Pair {pair} not found in profiles")
            
        sessions = {
            'Tokyo': (0, 6),    # UTC
            'London': (8, 16),  # UTC
            'NY': (13, 21),     # UTC
            'Sydney': (22, 6)   # UTC
        }
        
        pair_sessions = self.pair_profiles[pair]['session'].split('/')
        current_hour = datetime.utcnow().hour
        
        # Check for session overlap periods
        for session in pair_sessions:
            start, end = sessions[session]
            if start <= current_hour < end:
                # Check for session overlap
                if session == 'London' and 13 <= current_hour < 16:  # London/NY overlap
                    return True
                elif session == 'Tokyo' and 0 <= current_hour < 2:  # Tokyo/Sydney overlap
                    return True
                elif session == 'NY' and 13 <= current_hour < 16:  # NY/London overlap
                    return True
                elif session == 'Sydney' and 22 <= current_hour < 24:  # Sydney/Tokyo overlap
                    return True
                else:
                    return True
        
        return False 