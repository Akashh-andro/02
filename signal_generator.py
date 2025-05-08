import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
import tensorflow as tf
from datetime import datetime
import logging
import joblib
from scipy import stats
import pandas_ta as ta

class SignalGenerator:
    def __init__(self, model_path: str = None, scaler_path: str = None):
        self.model = None
        self.scaler = None
        self.confidence_threshold = 0.7
        self.min_risk_reward = 2.0
        self.max_spread_pips = 3.0
        
        if model_path and scaler_path:
            try:
                self.model = tf.keras.models.load_model(model_path)
                self.scaler = joblib.load(scaler_path)
                logging.info("Successfully loaded model and scaler")
            except Exception as e:
                logging.error(f"Error loading model or scaler: {str(e)}")
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('signal_generator.log'),
                logging.StreamHandler()
            ]
        )

    def _calculate_rsi(self, data: pd.Series, periods: int = 14) -> pd.Series:
        """Calculate RSI using a reliable method"""
        try:
            # Calculate price changes
            delta = data.diff()
            
            # Separate gains and losses
            gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
            
            # Calculate RS and RSI
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
        except Exception as e:
            logging.error(f"Error calculating RSI: {str(e)}")
            return pd.Series(50, index=data.index)  # Return neutral RSI on error

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
            
            # RSI using custom calculation
            df['rsi'] = self._calculate_rsi(df['close'])
            
            # Bollinger Bands
            bb = ta.bbands(df['close'])
            df['bb_upper'] = bb['BBU_20_2.0']
            df['bb_middle'] = bb['BBM_20_2.0']
            df['bb_lower'] = bb['BBL_20_2.0']
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            
            # ATR
            df['atr'] = ta.atr(df['high'], df['low'], df['close'])
            
            # ADX
            adx = ta.adx(df['high'], df['low'], df['close'])
            df['adx'] = adx['ADX_14'] if isinstance(adx, pd.DataFrame) else 25.0
            
            # Stochastic
            stoch = ta.stoch(df['high'], df['low'], df['close'])
            df['stoch_k'] = stoch['STOCHk_14_3_3']
            df['stoch_d'] = stoch['STOCHd_14_3_3']
            
            # Volume indicators
            if 'volume' in df.columns:
                df['obv'] = ta.obv(df['close'], df['volume'])
                df['mfi'] = ta.mfi(df['high'], df['low'], df['close'], df['volume'])
            
            # Custom indicators
            df['trend_strength'] = df['adx'].rolling(window=14).mean()
            df['volatility'] = df['atr'].rolling(window=20).mean()
            df['momentum'] = df['close'].pct_change(periods=10)
            
            return df.fillna(0)
        except Exception as e:
            logging.error(f"Error calculating technical indicators: {str(e)}")
            return df  # Return original dataframe on error

    def _get_ml_prediction(self, df: pd.DataFrame) -> Tuple[float, float]:
        """Get machine learning model prediction"""
        if self.model is None or self.scaler is None:
            return 0.0, 0.0
        
        # Prepare features
        features = df[self.scaler.feature_names_in_].values
        scaled_features = self.scaler.transform(features)
        
        # Reshape for LSTM input
        scaled_features = scaled_features.reshape(1, scaled_features.shape[0], scaled_features.shape[1])
        
        # Get prediction
        prediction = self.model.predict(scaled_features, verbose=0)
        
        # Calculate confidence based on model's output distribution
        confidence = abs(prediction[0][0])
        
        return prediction[0][0], confidence

    def _calculate_support_resistance(self, df: pd.DataFrame) -> Tuple[float, float]:
        """Calculate dynamic support and resistance levels"""
        # Use recent price action to identify levels
        recent_highs = df['high'].rolling(window=20).max()
        recent_lows = df['low'].rolling(window=20).min()
        
        # Calculate levels using statistical methods
        resistance = recent_highs.iloc[-1] + df['atr'].iloc[-1]
        support = recent_lows.iloc[-1] - df['atr'].iloc[-1]
        
        return support, resistance

    def _calculate_risk_levels(self, df: pd.DataFrame, direction: str) -> Tuple[float, float]:
        """Calculate stop loss and take profit levels"""
        atr = df['atr'].iloc[-1]
        current_price = df['close'].iloc[-1]
        
        if direction == 'BUY':
            stop_loss = current_price - (atr * 1.5)
            take_profit = current_price + (atr * 3.0)
        else:  # SELL
            stop_loss = current_price + (atr * 1.5)
            take_profit = current_price - (atr * 3.0)
        
        return stop_loss, take_profit

    def _validate_signal(self, signal: Dict) -> bool:
        """Validate trading signal"""
        # Check confidence threshold
        if signal['confidence'] < self.confidence_threshold:
            return False
        
        # Check risk-reward ratio
        risk = abs(signal['entry_price'] - signal['stop_loss'])
        reward = abs(signal['take_profit'] - signal['entry_price'])
        risk_reward = reward / risk if risk != 0 else 0
        
        if risk_reward < self.min_risk_reward:
            return False
        
        # Check spread
        if signal.get('spread_pips', 0) > self.max_spread_pips:
            return False
        
        return True

    def generate_signal(self, df: pd.DataFrame, pair: str) -> Optional[Dict]:
        """Generate trading signal based on multiple factors"""
        try:
            # Calculate technical indicators
            df = self._calculate_technical_indicators(df)
            
            # Get ML prediction
            prediction, ml_confidence = self._get_ml_prediction(df)
            
            # Calculate support and resistance
            support, resistance = self._calculate_support_resistance(df)
            
            # Get current price and indicators
            current_price = df['close'].iloc[-1]
            rsi = df['rsi'].iloc[-1]
            macd = df['macd'].iloc[-1]
            macd_signal = df['macd_signal'].iloc[-1]
            adx = df['adx'].iloc[-1]
            bb_width = df['bb_width'].iloc[-1]
            
            # Initialize signal components
            signal = {
                'pair': pair,
                'timestamp': datetime.now(),
                'entry_price': current_price,
                'confidence': 0.0,
                'direction': None,
                'stop_loss': None,
                'take_profit': None,
                'spread_pips': 0.0
            }
            
            # Determine direction based on multiple factors
            buy_signals = 0
            sell_signals = 0
            
            # ML prediction
            if prediction > 0:
                buy_signals += 1
            else:
                sell_signals += 1
            
            # RSI
            if rsi < 30:
                buy_signals += 1
            elif rsi > 70:
                sell_signals += 1
            
            # MACD
            if macd > macd_signal:
                buy_signals += 1
            else:
                sell_signals += 1
            
            # Bollinger Bands
            if current_price < df['bb_lower'].iloc[-1]:
                buy_signals += 1
            elif current_price > df['bb_upper'].iloc[-1]:
                sell_signals += 1
            
            # Trend strength
            if adx > 25:  # Strong trend
                if current_price > df['sma_20'].iloc[-1]:
                    buy_signals += 1
                else:
                    sell_signals += 1
            
            # Determine final direction
            if buy_signals > sell_signals:
                signal['direction'] = 'BUY'
                signal['confidence'] = (buy_signals / (buy_signals + sell_signals)) * ml_confidence
            elif sell_signals > buy_signals:
                signal['direction'] = 'SELL'
                signal['confidence'] = (sell_signals / (buy_signals + sell_signals)) * ml_confidence
            else:
                return None
            
            # Calculate risk levels
            signal['stop_loss'], signal['take_profit'] = self._calculate_risk_levels(
                df, signal['direction']
            )
            
            # Validate signal
            if not self._validate_signal(signal):
                return None
            
            return signal
            
        except Exception as e:
            logging.error(f"Error generating signal: {str(e)}")
            return None

    def update_parameters(self, confidence_threshold: float = None,
                         min_risk_reward: float = None,
                         max_spread_pips: float = None):
        """Update signal generator parameters"""
        if confidence_threshold is not None:
            self.confidence_threshold = confidence_threshold
        if min_risk_reward is not None:
            self.min_risk_reward = min_risk_reward
        if max_spread_pips is not None:
            self.max_spread_pips = max_spread_pips

def main():
    # Example usage
    generator = SignalGenerator(
        model_path='models/EURUSD_model.h5',
        scaler_path='models/EURUSD_feature_scaler.joblib'
    )
    
    # Update parameters
    generator.update_parameters(
        confidence_threshold=0.75,
        min_risk_reward=2.5,
        max_spread_pips=2.0
    )
    
    # Generate signal
    # Note: In practice, you would get real market data here
    df = pd.DataFrame()  # Your market data
    signal = generator.generate_signal(df, 'EURUSD')
    
    if signal:
        logging.info("Signal generated:")
        logging.info(f"Direction: {signal['direction']}")
        logging.info(f"Confidence: {signal['confidence']:.2%}")
        logging.info(f"Entry: {signal['entry_price']}")
        logging.info(f"SL: {signal['stop_loss']}")
        logging.info(f"TP: {signal['take_profit']}")
    else:
        logging.info("No valid signal generated")

if __name__ == "__main__":
    main() 