import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse
from typing import List, Dict, Optional, Tuple
import json
import os
from collections import deque
import logging
from concurrent.futures import ThreadPoolExecutor
import time  # Add time module for sleep

from universal_forex_predictor import UniversalForexPredictor
from multi_pair_executor import MultiPairExecutor, Signal
from volatility_scaler import VolatilityScaler

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('quantum_forex.log'),
        logging.StreamHandler()
    ]
)

class QuantumForexSystem:
    def __init__(self, pairs: List[str], risk_percent: float = 1.0):
        self.pairs = pairs
        self.risk_percent = risk_percent
        
        # Initialize MT5 with better error handling
        if not mt5.initialize(
            login=None,  # Add your MT5 login if needed
            server=None,  # Add your MT5 server if needed
            password=None,  # Add your MT5 password if needed
            timeout=30000
        ):
            error = mt5.last_error()
            raise RuntimeError(f"Failed to initialize MT5: {error}")
            
        # Verify MT5 connection
        if not mt5.terminal_info():
            raise RuntimeError("Failed to connect to MT5 terminal")
            
        # Verify account info
        if not mt5.account_info():
            raise RuntimeError("Failed to get account info")
            
        logging.info("Successfully connected to MT5")
        
        self.predictor = UniversalForexPredictor()
        self.executor = MultiPairExecutor()
        self.scaler = VolatilityScaler()
        
        # Performance tracking
        self.performance_metrics = {
            pair: {
                'wins': 0,
                'losses': 0,
                'total_trades': 0,
                'profit_factor': 0.0,
                'max_drawdown': 0.0,
                'avg_trade_duration': timedelta(),
                'recent_trades': deque(maxlen=100)
            } for pair in pairs
        }
        
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Initialize thread pool for parallel processing
        self.executor_pool = ThreadPoolExecutor(max_workers=len(pairs))

    def get_historical_data(self, pair: str, timeframe: str = 'H1', 
                          bars: int = 1000) -> pd.DataFrame:
        """Fetch historical data for a pair with enhanced error handling"""
        timeframe_map = {
            'M1': mt5.TIMEFRAME_M1,
            'M5': mt5.TIMEFRAME_M5,
            'M15': mt5.TIMEFRAME_M15,
            'M30': mt5.TIMEFRAME_M30,
            'H1': mt5.TIMEFRAME_H1,
            'H4': mt5.TIMEFRAME_H4,
            'D1': mt5.TIMEFRAME_D1
        }
        
        try:
            rates = mt5.copy_rates_from_pos(pair, timeframe_map[timeframe], 0, bars)
            if rates is None:
                raise ValueError(f"Failed to get historical data for {pair}")
                
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            
            # Add additional features
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            
            return df
            
        except Exception as e:
            logging.error(f"Error fetching historical data for {pair}: {str(e)}")
            raise

    def analyze_pair(self, pair: str) -> Optional[Signal]:
        """Enhanced pair analysis with multiple confirmation signals"""
        try:
            # Get historical data
            df = self.get_historical_data(pair)
            
            # Scale the data
            scaled_data = self.scaler.scale_data(df)
            
            # Get prediction
            prediction = self.predictor.predict(pair, scaled_data)
            
            # Get current price
            current_price = mt5.symbol_info_tick(pair).ask
            
            # Calculate stop loss and take profit levels
            volatility_metrics = self.scaler.get_volatility_metrics(df)
            atr = volatility_metrics['current_atr']
            
            # Adjust SL/TP based on volatility regime and risk metrics
            if volatility_metrics['volatility_regime'] == 'high':
                sl_pips = atr * 2 * volatility_metrics['risk_multiplier']
                tp_pips = atr * 3 * volatility_metrics['risk_multiplier']
            elif volatility_metrics['volatility_regime'] == 'medium':
                sl_pips = atr * 1.5 * volatility_metrics['risk_multiplier']
                tp_pips = atr * 2 * volatility_metrics['risk_multiplier']
            else:
                sl_pips = atr * volatility_metrics['risk_multiplier']
                tp_pips = atr * 1.5 * volatility_metrics['risk_multiplier']
            
            # Get additional confirmation signals
            confirmations = self._get_signal_confirmations(df, prediction)
            
            # Calculate final confidence
            confidence = self._calculate_signal_confidence(
                prediction, confirmations, volatility_metrics
            )
            
            # Generate signal if confidence is high enough
            if confidence > 0.7:  # Strong signal threshold
                direction = 'BUY' if prediction > 0 else 'SELL'
                return Signal(
                    direction=direction,
                    confidence=confidence * 100,
                    sl_pips=sl_pips,
                    tp_pips=tp_pips,
                    entry_price=current_price
                )
            
            return None
            
        except Exception as e:
            logging.error(f"Error analyzing {pair}: {str(e)}")
            return None

    def _get_signal_confirmations(self, df: pd.DataFrame, prediction: float) -> Dict[str, float]:
        """Get multiple confirmation signals for trade validation"""
        confirmations = {}
        
        # Trend confirmation
        adx = df['adx'].iloc[-1]
        confirmations['trend_strength'] = min(adx / 25, 1.0)  # Normalize to [0, 1]
        
        # Momentum confirmation
        rsi = df['rsi'].iloc[-1]
        if prediction > 0:
            confirmations['momentum'] = (30 - rsi) / 30 if rsi < 30 else 0
        else:
            confirmations['momentum'] = (rsi - 70) / 30 if rsi > 70 else 0
        
        # Volatility confirmation
        atr = df['atr'].iloc[-1]
        avg_atr = df['atr'].mean()
        confirmations['volatility'] = 1.0 if 0.5 <= atr/avg_atr <= 1.5 else 0.5
        
        # Volume confirmation
        if 'volume' in df.columns:
            volume_ma = df['volume'].rolling(20).mean()
            confirmations['volume'] = min(df['volume'].iloc[-1] / volume_ma.iloc[-1], 2.0)
        
        return confirmations

    def _calculate_signal_confidence(self, prediction: float, 
                                   confirmations: Dict[str, float],
                                   volatility_metrics: Dict[str, float]) -> float:
        """Calculate final signal confidence using multiple factors"""
        # Base confidence from prediction
        base_confidence = abs(prediction)
        
        # Weight factors for different confirmation types
        weights = {
            'trend_strength': 0.3,
            'momentum': 0.2,
            'volatility': 0.2,
            'volume': 0.1
        }
        
        # Calculate weighted confirmation score
        confirmation_score = sum(
            confirmations.get(factor, 0) * weight
            for factor, weight in weights.items()
        )
        
        # Adjust for volatility regime
        vol_factor = volatility_metrics.get('risk_multiplier', 1.0)
        
        # Calculate final confidence
        confidence = base_confidence * confirmation_score * vol_factor
        
        # Ensure confidence is within [0, 1]
        return max(0, min(1, confidence))

    def update_performance_metrics(self, pair: str, trade_result: Dict):
        """Update performance metrics for a pair"""
        metrics = self.performance_metrics[pair]
        
        if trade_result['profit'] > 0:
            metrics['wins'] += 1
        else:
            metrics['losses'] += 1
        
        metrics['total_trades'] += 1
        metrics['recent_trades'].append(trade_result)
        
        # Update profit factor
        if metrics['losses'] > 0:
            metrics['profit_factor'] = metrics['wins'] / metrics['losses']
        
        # Update max drawdown
        current_drawdown = self._calculate_drawdown(metrics['recent_trades'])
        metrics['max_drawdown'] = max(metrics['max_drawdown'], current_drawdown)
        
        # Update average trade duration
        if trade_result.get('duration'):
            metrics['avg_trade_duration'] = (
                (metrics['avg_trade_duration'] * (metrics['total_trades'] - 1) +
                 trade_result['duration']) / metrics['total_trades']
            )

    def _calculate_drawdown(self, trades: deque) -> float:
        """Calculate current drawdown from recent trades"""
        if not trades:
            return 0.0
        
        cumulative_returns = np.cumsum([t['profit'] for t in trades])
        max_drawdown = 0.0
        peak = cumulative_returns[0]
        
        for value in cumulative_returns:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak if peak > 0 else 0
            max_drawdown = max(max_drawdown, drawdown)
        
        return max_drawdown

    def run(self):
        """Enhanced main system loop with parallel processing"""
        logging.info(f"Starting Quantum Forex System with pairs: {self.pairs}")
        logging.info(f"Risk per trade: {self.risk_percent}%")
        
        while True:
            try:
                # Process pairs in parallel
                futures = []
                for pair in self.pairs:
                    if self.predictor.get_active_session(pair):
                        futures.append(
                            self.executor_pool.submit(self.analyze_pair, pair)
                        )
                
                # Process results
                for pair, future in zip(self.pairs, futures):
                    try:
                        signal = future.result()
                        if signal:
                            logging.info(f"\nSignal generated for {pair}:")
                            logging.info(f"Direction: {signal.direction}")
                            logging.info(f"Confidence: {signal.confidence:.2f}%")
                            logging.info(f"Entry: {signal.entry_price}")
                            logging.info(f"SL: {signal.sl_pips:.1f} pips")
                            logging.info(f"TP: {signal.tp_pips:.1f} pips")
                            
                            # Execute order
                            order_id = self.executor.execute_order(
                                pair, signal, self.risk_percent
                            )
                            if order_id:
                                logging.info(f"Order executed successfully. Order ID: {order_id}")
                    except Exception as e:
                        logging.error(f"Error processing {pair}: {str(e)}")
                
                # Get active positions
                positions = self.executor.get_active_positions()
                if positions:
                    logging.info("\nActive Positions:")
                    for pair, pos in positions.items():
                        logging.info(
                            f"{pair}: {pos['type']} @ {pos['open_price']} "
                            f"(Current: {pos['current_price']}, P/L: {pos['profit']:.2f})"
                        )
                
                # Wait for next iteration
                logging.info("\nWaiting for next analysis cycle...")
                time.sleep(60)  # Use time.sleep instead of mt5.sleep
                
            except KeyboardInterrupt:
                logging.info("\nShutting down Quantum Forex System...")
                break
            except Exception as e:
                logging.error(f"Error in main loop: {str(e)}")
                time.sleep(60)  # Use time.sleep instead of mt5.sleep

def main():
    parser = argparse.ArgumentParser(description='Quantum Forex Trading System')
    parser.add_argument('--pairs', type=str, required=True,
                      help='Comma-separated list of currency pairs')
    parser.add_argument('--risk', type=float, default=1.0,
                      help='Risk percentage per trade (default: 1.0)')
    parser.add_argument('--discover-pairs', action='store_true',
                      help='Discover available currency pairs')
    
    args = parser.parse_args()
    
    if args.discover_pairs:
        if not mt5.initialize():
            logging.error("Failed to initialize MT5")
            return
            
        symbols = mt5.symbols_get()
        if symbols is None:
            logging.error("Failed to get symbols")
            return
            
        logging.info("\nAvailable Currency Pairs:")
        for symbol in symbols:
            if symbol.visible:
                logging.info(f"{symbol.name}: {symbol.description}")
        return
    
    pairs = [pair.strip() for pair in args.pairs.split(',')]
    
    try:
        system = QuantumForexSystem(pairs, args.risk)
        system.run()
    except Exception as e:
        logging.error(f"Error: {str(e)}")
    finally:
        mt5.shutdown()

if __name__ == "__main__":
    main() 