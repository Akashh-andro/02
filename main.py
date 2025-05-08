import argparse
import logging
import sys
from datetime import datetime, timedelta
import MetaTrader5 as mt5
from typing import List, Dict, Optional
import json
import time
import streamlit as st
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

# Import the Streamlit app
from streamlit_app import main

from model_trainer import ForexModelTrainer
from backtester import ForexBacktester
from signal_generator import SignalGenerator
from position_manager import PositionManager
from market_analyzer import MarketAnalyzer
from risk_manager import RiskManager
from performance_analyzer import PerformanceAnalyzer, Trade

class ForexTradingSystem:
    def __init__(self, config_path: str = "config.json"):
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('trading_system.log'),
                logging.StreamHandler()
            ]
        )
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize MT5
        if not self._initialize_mt5():
            raise Exception("Failed to initialize MetaTrader 5")
        
        # Initialize components
        self.model_trainer = ForexModelTrainer()
        self.backtester = ForexBacktester()
        self.signal_generator = SignalGenerator()
        self.position_manager = PositionManager()
        self.market_analyzer = MarketAnalyzer()
        self.risk_manager = RiskManager()
        self.performance_analyzer = PerformanceAnalyzer()
        
        # Trading state
        self.is_running = False
        self.active_symbols = []

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from file"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logging.warning(f"Config file {config_path} not found. Using default configuration.")
            return {
                "symbols": ["EURUSD", "GBPUSD", "USDJPY"],
                "timeframe": "M15",
                "risk_per_trade": 0.02,
                "max_daily_risk": 0.05,
                "max_correlation": 0.7,
                "max_drawdown": 0.15,
                "model_path": "models/forex_model.h5",
                "scaler_path": "models/feature_scaler.joblib"
            }

    def _initialize_mt5(self) -> bool:
        """Initialize MetaTrader 5"""
        try:
            if not mt5.initialize():
                logging.error("Failed to initialize MT5")
                return False
            
            # Check connection
            if not mt5.terminal_info():
                logging.error("Failed to connect to MT5 terminal")
                return False
            
            # Check account info
            account_info = mt5.account_info()
            if account_info is None:
                logging.error("Failed to get account info")
                return False
            
            logging.info(f"Connected to MT5. Account: {account_info.login}")
            return True
            
        except Exception as e:
            logging.error(f"Error initializing MT5: {str(e)}")
            return False

    def discover_symbols(self) -> List[str]:
        """Discover available trading symbols"""
        try:
            symbols = mt5.symbols_get()
            if symbols is None:
                logging.error("Failed to get symbols")
                return []
            
            # Filter for forex pairs
            forex_symbols = [s.name for s in symbols if s.name.endswith(('USD', 'EUR', 'GBP', 'JPY', 'AUD', 'CAD', 'CHF', 'NZD'))]
            return forex_symbols
            
        except Exception as e:
            logging.error(f"Error discovering symbols: {str(e)}")
            return []

    def train_model(self, symbol: str) -> None:
        """Train the model for a specific symbol."""
        try:
            logging.info(f"Training model for {symbol}")
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)  # Use 1 year of data for training
            self.model_trainer.train(symbol, start_date=start_date, end_date=end_date)
        except Exception as e:
            logging.error(f"Error training model: {str(e)}")

    def backtest_strategy(self, symbol: str) -> None:
        """Backtest the strategy for a specific symbol."""
        try:
            logging.info(f"Backtesting strategy for {symbol}")
            end_date = datetime.now()
            start_date = end_date - timedelta(days=180)  # Use 6 months of data for backtesting
            self.backtester.run_backtest(symbol, start_date=start_date, end_date=end_date)
        except Exception as e:
            logging.error(f"Error backtesting strategy: {str(e)}")

    def start_trading(self):
        """Start the trading system"""
        try:
            if self.is_running:
                logging.warning("Trading system is already running")
                return
            
            self.is_running = True
            logging.info("Starting trading system")
            
            while self.is_running:
                try:
                    # Update market analysis
                    for symbol in self.active_symbols:
                        # Get market analysis
                        analysis = self.market_analyzer.analyze_market(symbol)
                        if not analysis:
                            continue
                        
                        # Generate trading signal
                        signal = self.signal_generator.generate_signal(analysis, symbol)
                        if not signal:
                            continue
                        
                        # Assess risk
                        risk_assessment = self.risk_manager.assess_trade_risk(
                            symbol=symbol,
                            signal_type=signal['type'],
                            entry_price=signal['entry_price'],
                            stop_loss=signal['stop_loss'],
                            take_profit=signal['take_profit']
                        )
                        
                        if not risk_assessment['approved']:
                            logging.info(f"Trade rejected for {symbol}: {risk_assessment['reason']}")
                            continue
                        
                        # Execute trade
                        position = self.position_manager.open_position(
                            symbol=symbol,
                            type=signal['type'],
                            volume=risk_assessment['position_size'],
                            price=signal['entry_price'],
                            stop_loss=signal['stop_loss'],
                            take_profit=signal['take_profit']
                        )
                        
                        if position:
                            # Add trade to performance analyzer
                            trade = Trade(
                                ticket=position['ticket'],
                                symbol=symbol,
                                type=signal['type'],
                                volume=risk_assessment['position_size'],
                                open_price=signal['entry_price'],
                                close_price=0.0,  # Will be updated when closed
                                stop_loss=signal['stop_loss'],
                                take_profit=signal['take_profit'],
                                profit=0.0,  # Will be updated when closed
                                swap=0.0,
                                open_time=datetime.now(),
                                close_time=datetime.now(),  # Will be updated when closed
                                comment=signal['comment']
                            )
                            self.performance_analyzer.add_trade(trade)
                    
                    # Update positions
                    self.position_manager.update_positions()
                    
                    # Update risk metrics with current positions
                    current_positions = self.position_manager.get_positions()
                    for position in current_positions:
                        trade_result = {
                            'symbol': position['symbol'],
                            'type': position['type'],
                            'volume': position['volume'],
                            'open_price': position['open_price'],
                            'current_price': position['current_price'],
                            'profit': position['profit'],
                            'risk': position['risk']
                        }
                        self.risk_manager.update_risk_metrics(trade_result)
                    
                    # Sleep for a short time
                    time.sleep(1)
                    
                except Exception as e:
                    logging.error(f"Error in trading loop: {str(e)}")
                    time.sleep(5)  # Sleep longer on error
                    
        except Exception as e:
            logging.error(f"Error starting trading system: {str(e)}")
            self.is_running = False

    def stop_trading(self):
        """Stop the trading system"""
        try:
            if not self.is_running:
                logging.warning("Trading system is not running")
                return
            
            self.is_running = False
            logging.info("Stopping trading system")
            
            # Close all positions
            self.position_manager.close_all_positions()
            
            # Generate final performance report
            self.performance_analyzer.plot_performance("final_performance.png")
            
        except Exception as e:
            logging.error(f"Error stopping trading system: {str(e)}")

    def __del__(self):
        """Cleanup when object is destroyed"""
        self.stop_trading()
        mt5.shutdown()

def main():
    parser = argparse.ArgumentParser(description="Forex Trading System")
    parser.add_argument("--discover-pairs", action="store_true", help="Discover available currency pairs")
    parser.add_argument("--pairs", type=str, help="Comma-separated list of currency pairs to trade")
    parser.add_argument("--train", action="store_true", help="Train models for specified pairs")
    parser.add_argument("--backtest", action="store_true", help="Backtest strategy for specified pairs")
    parser.add_argument("--start", action="store_true", help="Start trading system")
    parser.add_argument("--config", type=str, default="config.json", help="Path to configuration file")
    
    args = parser.parse_args()
    
    try:
        # Initialize trading system
        system = ForexTradingSystem(args.config)
        
        if args.discover_pairs:
            # Discover available pairs
            symbols = system.discover_symbols()
            print("\nAvailable currency pairs:")
            for symbol in symbols:
                print(f"- {symbol}")
            return
        
        if not args.pairs:
            print("Error: --pairs argument is required")
            return
        
        # Parse pairs
        pairs = [pair.strip() for pair in args.pairs.split(",")]
        
        if args.train:
            # Train models
            for pair in pairs:
                system.train_model(pair)
        
        if args.backtest:
            # Backtest strategy
            for pair in pairs:
                system.backtest_strategy(pair)
        
        if args.start:
            # Start trading
            system.active_symbols = pairs
            system.start_trading()
        
    except Exception as e:
        logging.error(f"Error in main: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 