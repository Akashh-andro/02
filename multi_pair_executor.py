import MetaTrader5 as mt5
from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np

@dataclass
class Signal:
    direction: str  # 'BUY' or 'SELL'
    confidence: float  # 0-100
    sl_pips: float
    tp_pips: float
    entry_price: float

class MultiPairExecutor:
    def __init__(self):
        self.active_orders: Dict[str, int] = {}
        self.max_spreads = {
            'EURUSD': 2.0,
            'GBPUSD': 3.0,
            'USDJPY': 3.0,
            'AUDUSD': 3.0,
            'USDCAD': 3.0,
            'NZDUSD': 3.0,
            'EURGBP': 3.0,
            'EURJPY': 4.0,
            'GBPJPY': 5.0,
            'USDCHF': 3.0
        }

    def validate_signal(self, pair: str, signal: Signal) -> bool:
        """Validate if a signal meets our criteria for execution"""
        # Check spread
        spread = mt5.symbol_info(pair).spread
        if spread > self.max_spreads.get(pair, 5.0):
            print(f"Spread too high for {pair}: {spread}")
            return False

        # Check if we already have an active order
        if pair in self.active_orders:
            print(f"Already have active order for {pair}")
            return False

        # Validate signal parameters
        if signal.confidence < 70:  # Minimum confidence threshold
            print(f"Signal confidence too low for {pair}: {signal.confidence}")
            return False

        if signal.sl_pips < 10:  # Minimum stop loss distance
            print(f"Stop loss too close for {pair}: {signal.sl_pips}")
            return False

        return True

    def calculate_lots(self, pair: str, signal: Signal, risk_percent: float = 1.0) -> float:
        """Calculate position size based on risk parameters"""
        account_info = mt5.account_info()
        if account_info is None:
            raise ValueError("Failed to get account info")

        symbol_info = mt5.symbol_info(pair)
        if symbol_info is None:
            raise ValueError(f"Failed to get symbol info for {pair}")

        # Calculate risk amount in account currency
        risk_amount = account_info.balance * risk_percent / 100

        # Calculate pip value
        pip_value = symbol_info.trade_tick_value * (signal.sl_pips / symbol_info.trade_tick_size)

        # Calculate position size
        lots = risk_amount / (pip_value * 100000)
        
        # Round to valid lot size
        lots = round(lots / symbol_info.volume_step) * symbol_info.volume_step
        
        # Ensure within min/max lot size
        lots = max(symbol_info.volume_min, min(symbol_info.volume_max, lots))
        
        return lots

    def execute_order(self, pair: str, signal: Signal, risk_percent: float = 1.0) -> Optional[int]:
        """Execute a trading order for a specific pair"""
        if not self.validate_signal(pair, signal):
            return None

        lots = self.calculate_lots(pair, signal, risk_percent)
        
        # Prepare order request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": pair,
            "volume": lots,
            "type": mt5.ORDER_TYPE_BUY if signal.direction == "BUY" else mt5.ORDER_TYPE_SELL,
            "price": signal.entry_price,
            "sl": signal.entry_price - signal.sl_pips * 0.0001 if signal.direction == "BUY" 
                  else signal.entry_price + signal.sl_pips * 0.0001,
            "tp": signal.entry_price + signal.tp_pips * 0.0001 if signal.direction == "BUY"
                  else signal.entry_price - signal.tp_pips * 0.0001,
            "deviation": 20,
            "magic": 234000,
            "comment": "python script",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        # Send the order
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"Order failed for {pair}: {result.comment}")
            return None

        self.active_orders[pair] = result.order
        return result.order

    def close_order(self, pair: str) -> bool:
        """Close an active order for a specific pair"""
        if pair not in self.active_orders:
            print(f"No active order found for {pair}")
            return False

        order_id = self.active_orders[pair]
        position = mt5.positions_get(ticket=order_id)
        
        if position is None:
            print(f"Position not found for order {order_id}")
            return False

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": pair,
            "volume": position[0].volume,
            "type": mt5.ORDER_TYPE_SELL if position[0].type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
            "position": order_id,
            "price": mt5.symbol_info_tick(pair).bid if position[0].type == mt5.ORDER_TYPE_BUY 
                    else mt5.symbol_info_tick(pair).ask,
            "deviation": 20,
            "magic": 234000,
            "comment": "python script close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"Close order failed for {pair}: {result.comment}")
            return False

        del self.active_orders[pair]
        return True

    def get_active_positions(self) -> Dict[str, Dict]:
        """Get information about all active positions"""
        positions = {}
        for pair, order_id in self.active_orders.items():
            position = mt5.positions_get(ticket=order_id)
            if position is not None:
                positions[pair] = {
                    'ticket': order_id,
                    'type': 'BUY' if position[0].type == mt5.ORDER_TYPE_BUY else 'SELL',
                    'volume': position[0].volume,
                    'open_price': position[0].price_open,
                    'current_price': position[0].price_current,
                    'profit': position[0].profit
                }
        return positions 