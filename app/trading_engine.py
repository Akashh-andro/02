import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np
from .data_manager import DataManager
from .risk_manager import RiskManager
from .strategies import TradingStrategy, MACDStrategy, RSIStrategy, BollingerBandsStrategy
from .backtesting import Backtester

class TradingEngine:
    def __init__(self, initial_capital: float = 10000):
        self.logger = logging.getLogger(__name__)
        self.data_manager = DataManager()
        self.risk_manager = RiskManager(initial_capital)
        self.backtester = Backtester(initial_capital)
        self.active_strategies: Dict[str, TradingStrategy] = {}
        self.positions: Dict[str, Dict] = {}
        self.trade_history: List[Dict] = []
        self.is_running = False
    
    def initialize(self, exchange_id: str = None, api_key: str = None, secret: str = None):
        """Initialize the trading engine"""
        try:
            if exchange_id:
                self.data_manager.initialize_exchange(exchange_id, api_key, secret)
            self.logger.info("Trading engine initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing trading engine: {str(e)}")
            raise
    
    def add_strategy(self, strategy: TradingStrategy, symbols: List[str]):
        """Add a trading strategy for specific symbols"""
        try:
            for symbol in symbols:
                if symbol not in self.active_strategies:
                    self.active_strategies[symbol] = []
                self.active_strategies[symbol].append(strategy)
            self.logger.info(f"Added strategy {strategy.name} for symbols {symbols}")
        except Exception as e:
            self.logger.error(f"Error adding strategy: {str(e)}")
            raise
    
    def remove_strategy(self, strategy: TradingStrategy, symbols: List[str]):
        """Remove a trading strategy from specific symbols"""
        try:
            for symbol in symbols:
                if symbol in self.active_strategies:
                    self.active_strategies[symbol] = [s for s in self.active_strategies[symbol] 
                                                    if s.name != strategy.name]
            self.logger.info(f"Removed strategy {strategy.name} from symbols {symbols}")
        except Exception as e:
            self.logger.error(f"Error removing strategy: {str(e)}")
            raise
    
    def start(self):
        """Start the trading engine"""
        try:
            self.is_running = True
            self.logger.info("Trading engine started")
            self._main_loop()
        except Exception as e:
            self.logger.error(f"Error starting trading engine: {str(e)}")
            self.is_running = False
            raise
    
    def stop(self):
        """Stop the trading engine"""
        self.is_running = False
        self.logger.info("Trading engine stopped")
    
    def _main_loop(self):
        """Main trading loop"""
        while self.is_running:
            try:
                # Update market data
                self._update_market_data()
                
                # Check for trading signals
                self._check_signals()
                
                # Update positions
                self._update_positions()
                
                # Update risk metrics
                self._update_risk_metrics()
                
                # Sleep to prevent excessive API calls
                time.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Error in main loop: {str(e)}")
                continue
    
    def _update_market_data(self):
        """Update market data for all active symbols"""
        try:
            for symbol in self.active_strategies.keys():
                data = self.data_manager.update_data(symbol)
                if data is not None:
                    # Calculate indicators
                    indicators = self.data_manager.calculate_technical_indicators(data)
                    # Store in cache
                    self.data_manager.data_cache[symbol] = {
                        'data': data,
                        'indicators': indicators
                    }
        except Exception as e:
            self.logger.error(f"Error updating market data: {str(e)}")
            raise
    
    def _check_signals(self):
        """Check for trading signals from all active strategies"""
        try:
            for symbol, strategies in self.active_strategies.items():
                if symbol not in self.data_manager.data_cache:
                    continue
                
                data = self.data_manager.data_cache[symbol]['data']
                indicators = self.data_manager.data_cache[symbol]['indicators']
                
                for strategy in strategies:
                    signal = strategy.generate_signal(data.iloc[-1], indicators)
                    
                    if signal != 0:
                        self._execute_trade(symbol, signal, strategy)
                        
        except Exception as e:
            self.logger.error(f"Error checking signals: {str(e)}")
            raise
    
    def _execute_trade(self, symbol: str, signal: int, strategy: TradingStrategy):
        """Execute a trade based on signal"""
        try:
            # Get current market data
            data = self.data_manager.data_cache[symbol]['data']
            current_price = data['close'].iloc[-1]
            
            # Calculate position size
            stop_loss = self._calculate_stop_loss(symbol, signal, current_price)
            position_size = self.risk_manager.calculate_position_size(
                symbol, current_price, stop_loss
            )
            
            # Check risk limits
            new_position = {
                'symbol': symbol,
                'size': position_size,
                'entry_price': current_price,
                'stop_loss': stop_loss,
                'risk': abs(current_price - stop_loss) * position_size
            }
            
            can_trade, message = self.risk_manager.check_risk_limits(new_position)
            
            if can_trade:
                # Execute trade
                if signal == 1:  # Buy
                    self._open_long_position(symbol, position_size, current_price, stop_loss)
                elif signal == -1:  # Sell
                    self._open_short_position(symbol, position_size, current_price, stop_loss)
                
                self.logger.info(f"Executed {signal} trade for {symbol} using {strategy.name}")
            else:
                self.logger.warning(f"Trade rejected: {message}")
                
        except Exception as e:
            self.logger.error(f"Error executing trade: {str(e)}")
            raise
    
    def _calculate_stop_loss(self, symbol: str, signal: int, current_price: float) -> float:
        """Calculate stop loss level based on ATR"""
        try:
            data = self.data_manager.data_cache[symbol]['data']
            indicators = self.data_manager.data_cache[symbol]['indicators']
            
            atr = indicators['atr'].iloc[-1]
            
            if signal == 1:  # Long position
                return current_price - (2 * atr)
            else:  # Short position
                return current_price + (2 * atr)
                
        except Exception as e:
            self.logger.error(f"Error calculating stop loss: {str(e)}")
            raise
    
    def _open_long_position(self, symbol: str, size: float, price: float, stop_loss: float):
        """Open a long position"""
        try:
            position = {
                'type': 'long',
                'size': size,
                'entry_price': price,
                'stop_loss': stop_loss,
                'entry_time': datetime.now(),
                'strategy': self.active_strategies[symbol][0].name
            }
            
            self.positions[symbol] = position
            self.trade_history.append(position)
            
        except Exception as e:
            self.logger.error(f"Error opening long position: {str(e)}")
            raise
    
    def _open_short_position(self, symbol: str, size: float, price: float, stop_loss: float):
        """Open a short position"""
        try:
            position = {
                'type': 'short',
                'size': size,
                'entry_price': price,
                'stop_loss': stop_loss,
                'entry_time': datetime.now(),
                'strategy': self.active_strategies[symbol][0].name
            }
            
            self.positions[symbol] = position
            self.trade_history.append(position)
            
        except Exception as e:
            self.logger.error(f"Error opening short position: {str(e)}")
            raise
    
    def _update_positions(self):
        """Update and manage open positions"""
        try:
            for symbol, position in list(self.positions.items()):
                if symbol not in self.data_manager.data_cache:
                    continue
                
                current_price = self.data_manager.data_cache[symbol]['data']['close'].iloc[-1]
                
                # Check stop loss
                if self._check_stop_loss(position, current_price):
                    self._close_position(symbol, current_price)
                    continue
                
                # Check take profit (implement your take profit logic here)
                if self._check_take_profit(position, current_price):
                    self._close_position(symbol, current_price)
                    continue
                
        except Exception as e:
            self.logger.error(f"Error updating positions: {str(e)}")
            raise
    
    def _check_stop_loss(self, position: Dict, current_price: float) -> bool:
        """Check if stop loss has been hit"""
        if position['type'] == 'long':
            return current_price <= position['stop_loss']
        else:
            return current_price >= position['stop_loss']
    
    def _check_take_profit(self, position: Dict, current_price: float) -> bool:
        """Check if take profit has been hit"""
        # Implement your take profit logic here
        return False
    
    def _close_position(self, symbol: str, current_price: float):
        """Close a position"""
        try:
            position = self.positions[symbol]
            
            # Calculate profit/loss
            if position['type'] == 'long':
                profit = (current_price - position['entry_price']) * position['size']
            else:
                profit = (position['entry_price'] - current_price) * position['size']
            
            # Update trade history
            position['exit_price'] = current_price
            position['exit_time'] = datetime.now()
            position['profit'] = profit
            
            # Update risk metrics
            self.risk_manager.update_risk_metrics(position)
            
            # Remove position
            del self.positions[symbol]
            
            self.logger.info(f"Closed {position['type']} position for {symbol} with profit: {profit}")
            
        except Exception as e:
            self.logger.error(f"Error closing position: {str(e)}")
            raise
    
    def _update_risk_metrics(self):
        """Update risk metrics"""
        try:
            # Update daily metrics
            self.risk_manager.update_daily_metrics()
            
            # Get risk report
            risk_report = self.risk_manager.get_risk_report()
            
            # Log risk metrics
            self.logger.info(f"Risk metrics updated: {risk_report}")
            
        except Exception as e:
            self.logger.error(f"Error updating risk metrics: {str(e)}")
            raise
    
    def run_backtest(self, symbol: str, strategy: TradingStrategy, 
                    start_date: str, end_date: str) -> Dict:
        """Run backtest for a strategy"""
        try:
            # Get historical data
            data = self.data_manager.fetch_stock_data(symbol, start_date, end_date)
            
            # Run backtest
            results = self.backtester.run_backtest(data, strategy)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error running backtest: {str(e)}")
            raise
    
    def get_performance_metrics(self) -> Dict:
        """Get current performance metrics"""
        try:
            return {
                'positions': self.positions,
                'trade_history': self.trade_history,
                'risk_metrics': self.risk_manager.risk_metrics,
                'active_strategies': {symbol: [s.name for s in strategies] 
                                    for symbol, strategies in self.active_strategies.items()}
            }
        except Exception as e:
            self.logger.error(f"Error getting performance metrics: {str(e)}")
            raise 