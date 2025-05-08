import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging
from dataclasses import dataclass
import json
import time

@dataclass
class Position:
    ticket: int
    symbol: str
    type: str
    volume: float
    open_price: float
    current_price: float
    stop_loss: float
    take_profit: float
    profit: float
    swap: float
    open_time: datetime
    comment: str

class PositionManager:
    def __init__(self, risk_percent: float = 1.0, max_positions: int = 5):
        self.risk_percent = risk_percent
        self.max_positions = max_positions
        self.positions: Dict[int, Position] = {}
        self.pending_orders: Dict[int, Dict] = {}
        self.trade_history: List[Dict] = []
        
        # Initialize MT5
        if not mt5.initialize():
            logging.error("Failed to initialize MT5")
            raise Exception("MT5 initialization failed")
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('position_manager.log'),
                logging.StreamHandler()
            ]
        )
        
        # Load trade history if exists
        self._load_trade_history()

    def _load_trade_history(self):
        """Load trade history from file"""
        try:
            with open('trade_history.json', 'r') as f:
                self.trade_history = json.load(f)
        except FileNotFoundError:
            self.trade_history = []

    def _save_trade_history(self):
        """Save trade history to file"""
        with open('trade_history.json', 'w') as f:
            json.dump(self.trade_history, f)

    def _calculate_position_size(self, symbol: str, stop_loss: float, 
                               entry_price: float) -> float:
        """Calculate position size based on risk management"""
        try:
            # Get account info
            account_info = mt5.account_info()
            if account_info is None:
                raise Exception("Failed to get account info")
            
            # Calculate risk amount
            balance = account_info.balance
            risk_amount = balance * (self.risk_percent / 100)
            
            # Get symbol info
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                raise Exception(f"Failed to get symbol info for {symbol}")
            
            # Calculate pip value
            point = symbol_info.point
            pip_size = point * 10 if symbol_info.digits == 5 else point
            
            # Calculate stop loss in pips
            sl_distance = abs(entry_price - stop_loss)
            sl_pips = sl_distance / pip_size
            
            # Calculate position size
            position_size = risk_amount / (sl_pips * symbol_info.trade_tick_value)
            
            # Round to symbol's volume step
            position_size = round(position_size / symbol_info.volume_step) * symbol_info.volume_step
            
            # Ensure minimum and maximum volume
            position_size = max(symbol_info.volume_min, min(position_size, symbol_info.volume_max))
            
            return position_size
            
        except Exception as e:
            logging.error(f"Error calculating position size: {str(e)}")
            return 0.0

    def _validate_trade(self, symbol: str, type: str, volume: float, 
                       price: float, stop_loss: float, take_profit: float) -> bool:
        """Validate trade parameters"""
        try:
            # Check if we have reached maximum positions
            if len(self.positions) >= self.max_positions:
                logging.warning("Maximum number of positions reached")
                return False
            
            # Check if we already have a position in this symbol
            for pos in self.positions.values():
                if pos.symbol == symbol:
                    logging.warning(f"Already have a position in {symbol}")
                    return False
            
            # Validate price levels
            if stop_loss >= price and type == "BUY":
                logging.warning("Invalid stop loss for BUY order")
                return False
            if stop_loss <= price and type == "SELL":
                logging.warning("Invalid stop loss for SELL order")
                return False
            if take_profit <= price and type == "BUY":
                logging.warning("Invalid take profit for BUY order")
                return False
            if take_profit >= price and type == "SELL":
                logging.warning("Invalid take profit for SELL order")
                return False
            
            return True
            
        except Exception as e:
            logging.error(f"Error validating trade: {str(e)}")
            return False

    def open_position(self, symbol: str, type: str, stop_loss: float, 
                     take_profit: float, comment: str = "") -> Optional[int]:
        """Open a new position"""
        try:
            # Get current price
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                raise Exception(f"Failed to get tick data for {symbol}")
            
            price = tick.ask if type == "BUY" else tick.bid
            
            # Calculate position size
            volume = self._calculate_position_size(symbol, stop_loss, price)
            if volume == 0:
                return None
            
            # Validate trade
            if not self._validate_trade(symbol, type, volume, price, stop_loss, take_profit):
                return None
            
            # Prepare trade request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": mt5.ORDER_TYPE_BUY if type == "BUY" else mt5.ORDER_TYPE_SELL,
                "price": price,
                "sl": stop_loss,
                "tp": take_profit,
                "deviation": 20,
                "magic": 234000,
                "comment": comment,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Send trade request
            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logging.error(f"Failed to open position: {result.comment}")
                return None
            
            # Create position object
            position = Position(
                ticket=result.order,
                symbol=symbol,
                type=type,
                volume=volume,
                open_price=price,
                current_price=price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                profit=0.0,
                swap=0.0,
                open_time=datetime.now(),
                comment=comment
            )
            
            # Add to positions dictionary
            self.positions[result.order] = position
            
            # Log trade
            trade_info = {
                "ticket": result.order,
                "symbol": symbol,
                "type": type,
                "volume": volume,
                "open_price": price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "open_time": datetime.now().isoformat(),
                "comment": comment
            }
            self.trade_history.append(trade_info)
            self._save_trade_history()
            
            logging.info(f"Opened {type} position in {symbol} with ticket {result.order}")
            return result.order
            
        except Exception as e:
            logging.error(f"Error opening position: {str(e)}")
            return None

    def close_position(self, ticket: int) -> bool:
        """Close an existing position"""
        try:
            if ticket not in self.positions:
                logging.warning(f"Position {ticket} not found")
                return False
            
            position = self.positions[ticket]
            
            # Prepare close request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": position.symbol,
                "volume": position.volume,
                "type": mt5.ORDER_TYPE_SELL if position.type == "BUY" else mt5.ORDER_TYPE_BUY,
                "position": ticket,
                "price": mt5.symbol_info_tick(position.symbol).bid if position.type == "BUY" else mt5.symbol_info_tick(position.symbol).ask,
                "deviation": 20,
                "magic": 234000,
                "comment": "Close position",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Send close request
            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logging.error(f"Failed to close position: {result.comment}")
                return False
            
            # Update trade history
            for trade in self.trade_history:
                if trade["ticket"] == ticket:
                    trade["close_time"] = datetime.now().isoformat()
                    trade["close_price"] = request["price"]
                    trade["profit"] = position.profit
                    break
            
            self._save_trade_history()
            
            # Remove from positions dictionary
            del self.positions[ticket]
            
            logging.info(f"Closed position {ticket}")
            return True
            
        except Exception as e:
            logging.error(f"Error closing position: {str(e)}")
            return False

    def update_positions(self):
        """Update all open positions"""
        try:
            for ticket, position in list(self.positions.items()):
                # Get current position info from MT5
                mt5_position = mt5.positions_get(ticket=ticket)
                if mt5_position is None:
                    logging.warning(f"Position {ticket} not found in MT5")
                    continue
                
                mt5_position = mt5_position[0]
                
                # Update position object
                position.current_price = mt5_position.price_current
                position.profit = mt5_position.profit
                position.swap = mt5_position.swap
                
                # Check if position was closed
                if mt5_position.volume == 0:
                    del self.positions[ticket]
                    logging.info(f"Position {ticket} was closed externally")
                    
                    # Update trade history
                    for trade in self.trade_history:
                        if trade["ticket"] == ticket:
                            trade["close_time"] = datetime.now().isoformat()
                            trade["close_price"] = position.current_price
                            trade["profit"] = position.profit
                            break
                    
                    self._save_trade_history()
            
        except Exception as e:
            logging.error(f"Error updating positions: {str(e)}")

    def get_position_info(self, ticket: int) -> Optional[Position]:
        """Get information about a specific position"""
        return self.positions.get(ticket)

    def get_all_positions(self) -> List[Position]:
        """Get all open positions"""
        return list(self.positions.values())

    def get_trade_history(self) -> List[Dict]:
        """Get trade history"""
        return self.trade_history

    def calculate_performance_metrics(self) -> Dict:
        """Calculate trading performance metrics"""
        try:
            if not self.trade_history:
                return {}
            
            total_trades = len(self.trade_history)
            winning_trades = sum(1 for trade in self.trade_history if trade.get("profit", 0) > 0)
            losing_trades = sum(1 for trade in self.trade_history if trade.get("profit", 0) < 0)
            
            total_profit = sum(trade.get("profit", 0) for trade in self.trade_history)
            total_loss = sum(trade.get("profit", 0) for trade in self.trade_history if trade.get("profit", 0) < 0)
            
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            profit_factor = abs(total_profit / total_loss) if total_loss != 0 else float('inf')
            
            return {
                "total_trades": total_trades,
                "winning_trades": winning_trades,
                "losing_trades": losing_trades,
                "win_rate": win_rate,
                "total_profit": total_profit,
                "total_loss": total_loss,
                "profit_factor": profit_factor
            }
            
        except Exception as e:
            logging.error(f"Error calculating performance metrics: {str(e)}")
            return {}

    def __del__(self):
        """Cleanup when object is destroyed"""
        mt5.shutdown()

def main():
    # Example usage
    manager = PositionManager(risk_percent=1.0, max_positions=5)
    
    # Open a position
    ticket = manager.open_position(
        symbol="EURUSD",
        type="BUY",
        stop_loss=1.0500,
        take_profit=1.0700,
        comment="Test trade"
    )
    
    if ticket:
        # Get position info
        position = manager.get_position_info(ticket)
        print(f"Opened position: {position}")
        
        # Update positions
        manager.update_positions()
        
        # Get performance metrics
        metrics = manager.calculate_performance_metrics()
        print(f"Performance metrics: {metrics}")
        
        # Close position
        manager.close_position(ticket)

if __name__ == "__main__":
    main() 