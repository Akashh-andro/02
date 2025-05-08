import logging
import sys
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .config import Config
from .trading_engine import TradingEngine
from .strategies import (
    MACDStrategy, RSIStrategy, BollingerBandsStrategy,
    MovingAverageCrossoverStrategy, ADXStrategy, CombinedStrategy
)
from .data_manager import DataManager
from .risk_manager import RiskManager
from .backtesting import Backtester

class TradingApp:
    def __init__(self):
        # Initialize logging
        self._setup_logging()
        
        # Initialize components
        self.config = Config()
        self.engine = TradingEngine(self.config.get_trading_config()['initial_capital'])
        self.data_manager = DataManager()
        self.risk_manager = RiskManager(self.config.get_trading_config()['initial_capital'])
        self.backtester = Backtester(self.config.get_trading_config()['initial_capital'])
        
        # Initialize strategies
        self.strategies = {
            'MACD': MACDStrategy(),
            'RSI': RSIStrategy(),
            'BollingerBands': BollingerBandsStrategy(),
            'MovingAverageCrossover': MovingAverageCrossoverStrategy(),
            'ADX': ADXStrategy(),
            'Combined': CombinedStrategy()
        }
    
    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('trading.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def initialize(self):
        """Initialize the trading application"""
        try:
            # Validate configuration
            if not self.config.validate_config():
                raise ValueError("Invalid configuration")
            
            # Initialize exchange connection
            exchange_config = self.config.get_exchange_config()
            if exchange_config['name']:
                self.engine.initialize(
                    exchange_config['name'],
                    exchange_config['api_key'],
                    exchange_config['api_secret']
                )
            
            # Add strategies
            for symbol in self.config.get_symbols():
                strategy_config = self.config.get_strategy_config()
                strategy = self.strategies[strategy_config['type']]
                self.engine.add_strategy(strategy, [symbol])
            
            self.logger.info("Trading application initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing trading application: {str(e)}")
            raise
    
    def start_trading(self):
        """Start the trading engine"""
        try:
            self.engine.start()
        except Exception as e:
            self.logger.error(f"Error starting trading engine: {str(e)}")
            raise
    
    def stop_trading(self):
        """Stop the trading engine"""
        try:
            self.engine.stop()
        except Exception as e:
            self.logger.error(f"Error stopping trading engine: {str(e)}")
            raise
    
    def run_backtest(self, symbol: str, strategy_name: str,
                    start_date: Optional[str] = None,
                    end_date: Optional[str] = None) -> Dict:
        """Run backtest for a strategy"""
        try:
            # Get backtesting configuration
            backtest_config = self.config.get_backtesting_config()
            start_date = start_date or backtest_config['start_date']
            end_date = end_date or backtest_config['end_date']
            
            # Get strategy
            strategy = self.strategies[strategy_name]
            
            # Run backtest
            results = self.engine.run_backtest(symbol, strategy, start_date, end_date)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error running backtest: {str(e)}")
            raise
    
    def plot_backtest_results(self, results: Dict):
        """Plot backtest results"""
        try:
            # Create figure with secondary y-axis
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                              vertical_spacing=0.05,
                              subplot_titles=('Price', 'Equity Curve', 'Drawdown'))
            
            # Add price chart
            fig.add_trace(go.Candlestick(
                x=results['data'].index,
                open=results['data']['open'],
                high=results['data']['high'],
                low=results['data']['low'],
                close=results['data']['close'],
                name='Price'
            ), row=1, col=1)
            
            # Add equity curve
            fig.add_trace(go.Scatter(
                x=results['equity_curve'].index,
                y=results['equity_curve'],
                name='Equity'
            ), row=2, col=1)
            
            # Add drawdown
            fig.add_trace(go.Scatter(
                x=results['drawdown'].index,
                y=results['drawdown'],
                name='Drawdown',
                fill='tozeroy'
            ), row=3, col=1)
            
            # Update layout
            fig.update_layout(
                title='Backtest Results',
                xaxis_title='Date',
                yaxis_title='Price',
                height=900
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error plotting backtest results: {str(e)}")
            raise
    
    def get_performance_metrics(self) -> Dict:
        """Get current performance metrics"""
        try:
            return self.engine.get_performance_metrics()
        except Exception as e:
            self.logger.error(f"Error getting performance metrics: {str(e)}")
            raise
    
    def update_config(self, section: str, config: Dict):
        """Update configuration"""
        try:
            if section == 'trading':
                self.config.update_trading_config(config)
            elif section == 'exchange':
                self.config.update_exchange_config(config)
            elif section == 'strategy':
                self.config.update_strategy_config(config.get('name', 'default'), config)
            else:
                raise ValueError(f"Invalid configuration section: {section}")
            
            self.logger.info(f"Configuration updated for section: {section}")
            
        except Exception as e:
            self.logger.error(f"Error updating configuration: {str(e)}")
            raise
    
    def add_symbol(self, symbol: str):
        """Add trading symbol"""
        try:
            self.config.add_symbol(symbol)
            self.logger.info(f"Added symbol: {symbol}")
        except Exception as e:
            self.logger.error(f"Error adding symbol: {str(e)}")
            raise
    
    def remove_symbol(self, symbol: str):
        """Remove trading symbol"""
        try:
            self.config.remove_symbol(symbol)
            self.logger.info(f"Removed symbol: {symbol}")
        except Exception as e:
            self.logger.error(f"Error removing symbol: {str(e)}")
            raise
    
    def get_available_strategies(self) -> List[str]:
        """Get list of available strategies"""
        return list(self.strategies.keys())
    
    def get_active_symbols(self) -> List[str]:
        """Get list of active trading symbols"""
        return self.config.get_symbols()
    
    def get_risk_report(self) -> Dict:
        """Get current risk report"""
        try:
            return self.risk_manager.get_risk_report()
        except Exception as e:
            self.logger.error(f"Error getting risk report: {str(e)}")
            raise

def main():
    """Main entry point for the trading application"""
    try:
        # Create and initialize the trading application
        app = TradingApp()
        app.initialize()
        
        # Start trading
        app.start_trading()
        
        # Keep the application running
        while True:
            try:
                # Get user input
                command = input("Enter command (help for available commands): ")
                
                if command == "help":
                    print("\nAvailable commands:")
                    print("  help - Show this help message")
                    print("  stop - Stop trading")
                    print("  start - Start trading")
                    print("  backtest <symbol> <strategy> - Run backtest")
                    print("  metrics - Show performance metrics")
                    print("  risk - Show risk report")
                    print("  exit - Exit application")
                
                elif command == "stop":
                    app.stop_trading()
                
                elif command == "start":
                    app.start_trading()
                
                elif command.startswith("backtest"):
                    parts = command.split()
                    if len(parts) != 3:
                        print("Usage: backtest <symbol> <strategy>")
                    else:
                        symbol, strategy = parts[1], parts[2]
                        results = app.run_backtest(symbol, strategy)
                        app.plot_backtest_results(results)
                
                elif command == "metrics":
                    metrics = app.get_performance_metrics()
                    print("\nPerformance Metrics:")
                    print(f"Total Trades: {metrics['trade_history']}")
                    print(f"Win Rate: {metrics['risk_metrics']['win_rate']:.2%}")
                    print(f"Profit Factor: {metrics['risk_metrics']['profit_factor']:.2f}")
                    print(f"Sharpe Ratio: {metrics['risk_metrics']['sharpe_ratio']:.2f}")
                
                elif command == "risk":
                    risk_report = app.get_risk_report()
                    print("\nRisk Report:")
                    print(f"Current Capital: ${risk_report['current_capital']:.2f}")
                    print(f"Daily P&L: ${risk_report['daily_pnl']:.2f}")
                    print(f"Open Positions: {risk_report['open_positions']}")
                    print(f"Max Drawdown: {risk_report['risk_metrics']['max_drawdown']:.2%}")
                
                elif command == "exit":
                    app.stop_trading()
                    break
                
                else:
                    print("Unknown command. Type 'help' for available commands.")
                
            except KeyboardInterrupt:
                print("\nStopping trading...")
                app.stop_trading()
                break
            
            except Exception as e:
                print(f"Error: {str(e)}")
                continue
        
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 