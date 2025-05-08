import tensorflow as tf
import numpy as np
import pandas as pd
import MetaTrader5 as mt5
from datetime import datetime, timedelta
import pandas_ta as ta
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional
import logging
import os
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Bidirectional, Attention, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import joblib

class ForexModelTrainer:
    def __init__(self, lookback_period: int = 60, prediction_horizon: int = 1):
        self.lookback_period = lookback_period
        self.prediction_horizon = prediction_horizon
        self.scaler = StandardScaler()
        self.feature_scaler = StandardScaler()
        self.model = None
        self.feature_columns = [
            'open', 'high', 'low', 'close', 'volume',
            'rsi', 'macd', 'macd_signal', 'macd_hist',
            'bb_upper', 'bb_middle', 'bb_lower',
            'atr', 'adx', 'cci', 'mfi',
            'stoch_k', 'stoch_d', 'williams_r',
            'obv', 'vwap', 'supertrend',
            'returns', 'log_returns', 'volatility',
            'momentum', 'trend_strength'
        ]
        
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('model_training.log'),
                logging.StreamHandler()
            ]
        )

    def _calculate_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate advanced technical indicators and features"""
        try:
            # Basic price features
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            
            # Volatility features
            df['volatility'] = df['returns'].rolling(window=20).std()
            df['realized_vol'] = df['returns'].rolling(window=20).apply(lambda x: np.sqrt(np.sum(x**2)))
            
            # RSI
            try:
                df['rsi'] = ta.rsi(df['close'], length=14)
            except Exception as e:
                logging.warning(f"Error calculating RSI: {str(e)}. Using default value.")
                df['rsi'] = 50.0  # Default neutral value
            
            # MACD
            try:
                macd = ta.macd(df['close'])
                df['macd'] = macd['MACD_12_26_9']
                df['macd_signal'] = macd['MACDs_12_26_9']
                df['macd_hist'] = macd['MACDh_12_26_9']
            except Exception as e:
                logging.warning(f"Error calculating MACD: {str(e)}. Using default values.")
                df['macd'] = 0.0
                df['macd_signal'] = 0.0
                df['macd_hist'] = 0.0
            
            # Bollinger Bands
            try:
                sma = df['close'].rolling(window=20).mean()
                std = df['close'].rolling(window=20).std()
                df['bb_upper'] = sma + (std * 2)
                df['bb_middle'] = sma
                df['bb_lower'] = sma - (std * 2)
                df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            except Exception as e:
                logging.warning(f"Error calculating Bollinger Bands: {str(e)}. Using default values.")
                df['bb_upper'] = df['close']
                df['bb_middle'] = df['close']
                df['bb_lower'] = df['close']
                df['bb_width'] = 0.0
            
            # ATR
            try:
                df['atr'] = ta.atr(df['high'], df['low'], df['close'])
            except Exception as e:
                logging.warning(f"Error calculating ATR: {str(e)}. Using default value.")
                df['atr'] = (df['high'] - df['low']).mean()
            
            # ADX
            try:
                adx = ta.adx(df['high'], df['low'], df['close'])
                df['adx'] = adx['ADX_14']
            except Exception as e:
                logging.warning(f"Error calculating ADX: {str(e)}. Using default value.")
                df['adx'] = 25.0  # Default neutral value
            
            # CCI
            try:
                df['cci'] = ta.cci(df['high'], df['low'], df['close'])
            except Exception as e:
                logging.warning(f"Error calculating CCI: {str(e)}. Using default value.")
                df['cci'] = 0.0
            
            # MFI
            try:
                df['mfi'] = ta.mfi(df['high'], df['low'], df['close'], df['volume'])
            except Exception as e:
                logging.warning(f"Error calculating MFI: {str(e)}. Using default value.")
                df['mfi'] = 50.0  # Default neutral value
            
            # Stochastic
            try:
                stoch = ta.stoch(df['high'], df['low'], df['close'])
                df['stoch_k'] = stoch['STOCHk_14_3_3']
                df['stoch_d'] = stoch['STOCHd_14_3_3']
            except Exception as e:
                logging.warning(f"Error calculating Stochastic: {str(e)}. Using default values.")
                df['stoch_k'] = 50.0
                df['stoch_d'] = 50.0
            
            # Williams %R
            try:
                df['williams_r'] = ta.willr(df['high'], df['low'], df['close'])
            except Exception as e:
                logging.warning(f"Error calculating Williams %R: {str(e)}. Using default value.")
                df['williams_r'] = -50.0  # Default neutral value
            
            # OBV
            try:
                df['obv'] = ta.obv(df['close'], df['volume'])
            except Exception as e:
                logging.warning(f"Error calculating OBV: {str(e)}. Using default value.")
                df['obv'] = df['volume'].cumsum()
            
            # VWAP
            try:
                df['vwap'] = ta.vwap(df['high'], df['low'], df['close'], df['volume'])
            except Exception as e:
                logging.warning(f"Error calculating VWAP: {str(e)}. Using default value.")
                df['vwap'] = df['close']
            
            # Supertrend
            try:
                supertrend = ta.supertrend(df['high'], df['low'], df['close'])
                df['supertrend'] = supertrend['SUPERT_7_3.0']
            except Exception as e:
                logging.warning(f"Error calculating Supertrend: {str(e)}. Using default value.")
                df['supertrend'] = df['close']
            
            # Trend features
            df['trend_strength'] = df['adx'].rolling(window=14).mean()
            df['trend_direction'] = np.where(df['close'] > df['close'].shift(1), 1, -1)
            
            # Momentum features
            df['momentum'] = df['close'] - df['close'].shift(10)
            df['momentum_ma'] = df['momentum'].rolling(window=10).mean()
            
            # Volume features
            df['volume_ma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma']
            
            # Additional custom features
            df['price_momentum'] = df['close'].pct_change(periods=5)
            df['volume_momentum'] = df['volume'].pct_change(periods=5)
            df['volatility_ratio'] = df['atr'] / df['atr'].rolling(window=20).mean()
            
            return df.fillna(0)
            
        except Exception as e:
            logging.error(f"Error in _calculate_advanced_features: {str(e)}")
            raise

    def _create_sequences(self, data: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM input"""
        X, y = [], []
        for i in range(len(data) - self.lookback_period - self.prediction_horizon + 1):
            X.append(data[i:(i + self.lookback_period)])
            y.append(target[i + self.lookback_period + self.prediction_horizon - 1])
        return np.array(X), np.array(y)

    def _build_model(self, input_shape: Tuple[int, int]) -> Model:
        """Build a sophisticated deep learning model"""
        # Input layer
        inputs = Input(shape=input_shape)
        
        # First LSTM layer with return sequences
        x = Bidirectional(LSTM(128, return_sequences=True))(inputs)
        x = Dropout(0.2)(x)
        
        # Second LSTM layer
        x = Bidirectional(LSTM(64, return_sequences=True))(x)
        x = Dropout(0.2)(x)
        
        # Third LSTM layer
        x = Bidirectional(LSTM(32))(x)
        x = Dropout(0.2)(x)
        
        # Dense layers
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.2)(x)
        
        # Output layer
        outputs = Dense(1, activation='tanh')(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model

    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for model training"""
        # Calculate features
        df = self._calculate_advanced_features(df)
        
        # Select features
        feature_data = df[self.feature_columns].values
        
        # Scale features
        scaled_features = self.feature_scaler.fit_transform(feature_data)
        
        # Create target (next period's return)
        target = df['returns'].shift(-self.prediction_horizon).values
        
        # Create sequences
        X, y = self._create_sequences(scaled_features, target)
        
        return X, y

    def train(self, pair: str, start_date: datetime, end_date: datetime) -> None:
        """Train the model on historical data"""
        logging.info(f"Starting model training for {pair}")
        
        try:
            # Get historical data
            rates = mt5.copy_rates_range(
                pair,
                mt5.TIMEFRAME_H1,
                start_date,
                end_date
            )
            
            if rates is None or len(rates) == 0:
                raise ValueError(f"Failed to get historical data for {pair}")
            
            # Convert to DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            
            # Ensure required columns exist
            required_columns = ['open', 'high', 'low', 'close']
            for col in required_columns:
                if col not in df.columns:
                    raise ValueError(f"Missing required column: {col}")
            
            # Add volume if missing
            if 'volume' not in df.columns:
                logging.warning(f"Volume data not available for {pair}, using synthetic volume")
                df['volume'] = ((df['high'] - df['low']) / (df['high'] + df['low']) * 1000).round()
            
            # Handle missing values before calculating indicators
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            # Prepare data with error handling
            try:
                X, y = self.prepare_data(df)
            except Exception as e:
                logging.error(f"Error preparing data: {str(e)}")
                raise
            
            if len(X) == 0 or len(y) == 0:
                raise ValueError("No valid sequences created")
            
            # Split data
            train_size = int(len(X) * 0.8)
            X_train, X_val = X[:train_size], X[train_size:]
            y_train, y_val = y[:train_size], y[train_size:]
            
            # Build model
            self.model = self._build_model(input_shape=(self.lookback_period, len(self.feature_columns)))
            
            # Callbacks
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                ),
                ModelCheckpoint(
                    f'models/{pair}_model.h5',
                    monitor='val_loss',
                    save_best_only=True
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=0.0001
                )
            ]
            
            # Train model
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=100,
                batch_size=32,
                callbacks=callbacks,
                verbose=1
            )
            
            # Save scalers
            joblib.dump(self.feature_scaler, f'models/{pair}_feature_scaler.joblib')
            
            logging.info(f"Model training completed for {pair}")
            
        except Exception as e:
            logging.error(f"Error during model training: {str(e)}")
            raise

    def evaluate(self, pair: str, start_date: datetime, end_date: datetime) -> Dict[str, float]:
        """Evaluate model performance"""
        # Get test data
        rates = mt5.copy_rates_range(
            pair,
            mt5.TIMEFRAME_H1,
            start_date,
            end_date
        )
        
        if rates is None:
            raise ValueError(f"Failed to get test data for {pair}")
        
        # Convert to DataFrame
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        # Prepare data
        X, y = self.prepare_data(df)
        
        # Make predictions
        predictions = self.model.predict(X)
        
        # Calculate metrics
        mse = np.mean((y - predictions.flatten()) ** 2)
        mae = np.mean(np.abs(y - predictions.flatten()))
        
        # Calculate directional accuracy
        direction_accuracy = np.mean(
            (y > 0) == (predictions.flatten() > 0)
        )
        
        return {
            'mse': mse,
            'mae': mae,
            'direction_accuracy': direction_accuracy
        }

def main():
    # Initialize MT5
    if not mt5.initialize():
        raise RuntimeError("Failed to initialize MT5")
    
    # Create trainer
    trainer = ForexModelTrainer()
    
    # Define pairs to train
    pairs = ['EURUSD', 'GBPUSD', 'USDJPY']
    
    # Define date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)  # 1 year of data
    
    # Train models for each pair
    for pair in pairs:
        try:
            logging.info(f"Training model for {pair}")
            history = trainer.train(pair, start_date, end_date)
            
            # Evaluate model
            metrics = trainer.evaluate(
                pair,
                end_date - timedelta(days=30),  # Last 30 days for evaluation
                end_date
            )
            
            logging.info(f"Evaluation metrics for {pair}:")
            logging.info(f"MSE: {metrics['mse']:.6f}")
            logging.info(f"MAE: {metrics['mae']:.6f}")
            logging.info(f"Direction Accuracy: {metrics['direction_accuracy']:.2%}")
            
        except Exception as e:
            logging.error(f"Error training {pair}: {str(e)}")
    
    mt5.shutdown()

if __name__ == "__main__":
    main() 