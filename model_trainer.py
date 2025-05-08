try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow not available. Running in simulation mode.")

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    print("MetaTrader5 not available. Running in simulation mode.")

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple
import json
import os

class ForexModelTrainer:
    def __init__(self, model_path: str = "models/forex_model.h5"):
        """
        Initialize the ForexModelTrainer
        
        Args:
            model_path (str): Path to save/load the trained model
        """
        self.model_path = model_path
        self.model = None
        self.scaler = None
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('model_trainer.log'),
                logging.StreamHandler()
            ]
        )
        
        # Create models directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Initialize model if available
        if TF_AVAILABLE:
            self._initialize_model()
        else:
            self._initialize_simulation()

    def _initialize_model(self):
        """Initialize the TensorFlow model"""
        if not TF_AVAILABLE:
            return
            
        try:
            # Try to load existing model
            if os.path.exists(self.model_path):
                self.model = tf.keras.models.load_model(self.model_path)
                logging.info("Loaded existing model")
            else:
                # Create new model
                self.model = tf.keras.Sequential([
                    tf.keras.layers.LSTM(64, input_shape=(60, 5), return_sequences=True),
                    tf.keras.layers.Dropout(0.2),
                    tf.keras.layers.LSTM(32),
                    tf.keras.layers.Dropout(0.2),
                    tf.keras.layers.Dense(16, activation='relu'),
                    tf.keras.layers.Dense(1, activation='sigmoid')
                ])
                
                self.model.compile(
                    optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy']
                )
                logging.info("Created new model")
                
        except Exception as e:
            logging.error(f"Error initializing model: {str(e)}")
            self.model = None

    def _initialize_simulation(self):
        """Initialize simulation mode"""
        logging.info("Running in simulation mode")
        self.model = None

    def train(self, symbol: str, start_date: datetime, end_date: datetime) -> bool:
        """
        Train the model for a specific symbol
        
        Args:
            symbol (str): Trading symbol
            start_date (datetime): Start date for training data
            end_date (datetime): End date for training data
            
        Returns:
            bool: True if training was successful
        """
        try:
            if not TF_AVAILABLE:
                logging.info("Training skipped - running in simulation mode")
                return True
                
            if self.model is None:
                logging.error("Model not initialized")
                return False
            
            # Get training data
            data = self._get_training_data(symbol, start_date, end_date)
            if data is None or len(data) < 100:
                logging.error("Insufficient training data")
                return False
            
            # Prepare data
            X, y = self._prepare_data(data)
            
            # Train model
            history = self.model.fit(
                X, y,
                epochs=50,
                batch_size=32,
                validation_split=0.2,
                verbose=1
            )
            
            # Save model
            self.model.save(self.model_path)
            logging.info("Model trained and saved successfully")
            
            return True
            
        except Exception as e:
            logging.error(f"Error training model: {str(e)}")
            return False

    def predict(self, symbol: str, data: pd.DataFrame) -> Optional[float]:
        """
        Make a prediction using the trained model
        
        Args:
            symbol (str): Trading symbol
            data (pd.DataFrame): Input data for prediction
            
        Returns:
            Optional[float]: Prediction probability or None if prediction fails
        """
        try:
            if not TF_AVAILABLE or self.model is None:
                # Return simulated prediction
                return np.random.random()
            
            # Prepare data
            X = self._prepare_prediction_data(data)
            if X is None:
                return None
            
            # Make prediction
            prediction = self.model.predict(X, verbose=0)
            return float(prediction[0][0])
            
        except Exception as e:
            logging.error(f"Error making prediction: {str(e)}")
            return None

    def _get_training_data(self, symbol: str, start_date: datetime, 
                          end_date: datetime) -> Optional[pd.DataFrame]:
        """Get training data for a symbol"""
        try:
            if MT5_AVAILABLE:
                # Get historical data
                rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_H1, start_date, end_date)
                if rates is None:
                    return None
                
                # Convert to DataFrame
                df = pd.DataFrame(rates)
                df['time'] = pd.to_datetime(df['time'], unit='s')
            else:
                # Generate simulated data
                dates = pd.date_range(start=start_date, end=end_date, freq='H')
                np.random.seed(42)  # For reproducibility
                prices = np.random.normal(1.0, 0.01, len(dates)).cumsum() + 1.0
                df = pd.DataFrame({
                    'time': dates,
                    'open': prices,
                    'high': prices * 1.001,
                    'low': prices * 0.999,
                    'close': prices,
                    'tick_volume': np.random.randint(100, 1000, len(dates))
                })
            
            return df
            
        except Exception as e:
            logging.error(f"Error getting training data: {str(e)}")
            return None

    def _prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for training"""
        try:
            # Calculate features
            data['returns'] = data['close'].pct_change()
            data['volatility'] = data['returns'].rolling(20).std()
            data['trend'] = data['close'].rolling(20).mean()
            data['momentum'] = data['close'] - data['close'].shift(10)
            
            # Create sequences
            X = []
            y = []
            sequence_length = 60
            
            for i in range(len(data) - sequence_length):
                X.append(data[['open', 'high', 'low', 'close', 'tick_volume']].iloc[i:i+sequence_length].values)
                y.append(1 if data['close'].iloc[i+sequence_length] > data['close'].iloc[i+sequence_length-1] else 0)
            
            return np.array(X), np.array(y)
            
        except Exception as e:
            logging.error(f"Error preparing data: {str(e)}")
            return np.array([]), np.array([])

    def _prepare_prediction_data(self, data: pd.DataFrame) -> Optional[np.ndarray]:
        """Prepare data for prediction"""
        try:
            if len(data) < 60:
                return None
            
            # Calculate features
            data['returns'] = data['close'].pct_change()
            data['volatility'] = data['returns'].rolling(20).std()
            data['trend'] = data['close'].rolling(20).mean()
            data['momentum'] = data['close'] - data['close'].shift(10)
            
            # Create sequence
            X = data[['open', 'high', 'low', 'close', 'tick_volume']].iloc[-60:].values
            return np.array([X])
            
        except Exception as e:
            logging.error(f"Error preparing prediction data: {str(e)}")
            return None

    def __del__(self):
        """Cleanup when object is destroyed"""
        if TF_AVAILABLE and self.model is not None:
            try:
                self.model.save(self.model_path)
            except:
                pass

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