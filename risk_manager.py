try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    print("MetaTrader5 not available. Running in simulation mode.")

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
import json
from scipy import stats
import pandas_ta as ta

class RiskManager:
    def __init__(self, max_risk_per_trade: float = 0.02, max_daily_risk: float = 0.05,
                 max_correlation: float = 0.7, max_drawdown: float = 0.15):
        """
        Initialize the RiskManager with risk parameters
        
        Args:
            max_risk_per_trade (float): Maximum risk per trade as a percentage of account balance
            max_daily_risk (float): Maximum daily risk as a percentage of account balance
            max_correlation (float): Maximum allowed correlation between positions
            max_drawdown (float): Maximum allowed drawdown as a percentage
        """
        self.max_risk_per_trade = max_risk_per_trade
        self.max_daily_risk = max_daily_risk
        self.max_correlation = max_correlation
        self.max_drawdown = max_drawdown
        
        # Initialize risk metrics
        self.daily_pnl = 0.0
        self.trades_today = 0
        self.current_drawdown = 0.0
        self.max_drawdown_seen = 0.0
        self.peak_balance = 0.0
        self.current_balance = 0.0
        
        # Initialize MT5 if available
        if MT5_AVAILABLE:
            self._initialize_mt5()
        else:
            self._initialize_simulation()
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('risk_manager.log'),
                logging.StreamHandler()
            ]
        )
        
        # Load risk metrics if exists
        self._load_risk_metrics()

    def _initialize_mt5(self):
        """Initialize MetaTrader 5 connection"""
        if not MT5_AVAILABLE:
            return
            
        try:
            if not mt5.initialize():
                logging.error("Failed to initialize MT5")
                return
            
            # Get initial account info
            account_info = mt5.account_info()
            if account_info is None:
                logging.error("Failed to get account info")
                return
            
            self.current_balance = account_info.balance
            self.peak_balance = account_info.balance
            logging.info(f"MT5 initialized. Account balance: ${self.current_balance:,.2f}")
            
        except Exception as e:
            logging.error(f"Error initializing MT5: {str(e)}")
    
    def _initialize_simulation(self):
        """Initialize simulation mode with default values"""
        self.current_balance = 10000.0  # Default starting balance
        self.peak_balance = self.current_balance
        logging.info(f"Running in simulation mode. Starting balance: ${self.current_balance:,.2f}")
    
    def _load_risk_metrics(self):
        """Load risk metrics from file"""
        try:
            with open('risk_metrics.json', 'r') as f:
                self.risk_metrics = json.load(f)
        except FileNotFoundError:
            self.risk_metrics = {
                "daily_risk": 0.0,
                "current_drawdown": 0.0,
                "max_drawdown": 0.0,
                "trades_today": 0,
                "last_reset": datetime.now().isoformat()
            }

    def _save_risk_metrics(self):
        """Save risk metrics to file"""
        with open('risk_metrics.json', 'w') as f:
            json.dump(self.risk_metrics, f)

    def _reset_daily_metrics(self):
        """Reset daily risk metrics"""
        current_date = datetime.now().date()
        last_reset = datetime.fromisoformat(self.risk_metrics["last_reset"]).date()
        
        if current_date > last_reset:
            self.risk_metrics["daily_risk"] = 0.0
            self.risk_metrics["trades_today"] = 0
            self.risk_metrics["last_reset"] = datetime.now().isoformat()
            self._save_risk_metrics()

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
            risk_amount = balance * (self.max_risk_per_trade / 100)
            
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

    def _calculate_correlation(self, symbol: str, other_symbols: List[str]) -> Dict[str, float]:
        """Calculate correlation with other symbols"""
        try:
            # Get historical data
            df1 = self._get_historical_data(symbol)
            if df1 is None:
                return {}
            
            correlations = {}
            for other_symbol in other_symbols:
                df2 = self._get_historical_data(other_symbol)
                if df2 is not None:
                    corr = df1['close'].corr(df2['close'])
                    correlations[other_symbol] = corr
            
            return correlations
            
        except Exception as e:
            logging.error(f"Error calculating correlation: {str(e)}")
            return {}

    def _get_historical_data(self, symbol: str, timeframe: str = "H1", 
                           bars: int = 1000) -> Optional[pd.DataFrame]:
        """Get historical price data"""
        try:
            # Convert timeframe string to MT5 timeframe
            tf_map = {
                "M1": mt5.TIMEFRAME_M1,
                "M5": mt5.TIMEFRAME_M5,
                "M15": mt5.TIMEFRAME_M15,
                "M30": mt5.TIMEFRAME_M30,
                "H1": mt5.TIMEFRAME_H1,
                "H4": mt5.TIMEFRAME_H4,
                "D1": mt5.TIMEFRAME_D1,
                "W1": mt5.TIMEFRAME_W1,
                "MN1": mt5.TIMEFRAME_MN1
            }
            
            if timeframe not in tf_map:
                raise ValueError(f"Invalid timeframe: {timeframe}")
            
            # Get historical data
            rates = mt5.copy_rates_from_pos(symbol, tf_map[timeframe], 0, bars)
            if rates is None:
                raise Exception(f"Failed to get historical data for {symbol}")
            
            # Convert to DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            
            return df
            
        except Exception as e:
            logging.error(f"Error getting historical data: {str(e)}")
            return None

    def _calculate_drawdown(self, equity_curve: List[float]) -> float:
        """Calculate current drawdown"""
        try:
            if not equity_curve:
                return 0.0
            
            peak = max(equity_curve)
            current = equity_curve[-1]
            drawdown = (peak - current) / peak * 100
            
            return drawdown
            
        except Exception as e:
            logging.error(f"Error calculating drawdown: {str(e)}")
            return 0.0

    def _calculate_volatility(self, symbol: str) -> float:
        """Calculate current market volatility"""
        try:
            df = self._get_historical_data(symbol)
            if df is None:
                return 0.0
            
            # Calculate returns
            returns = df['close'].pct_change().dropna()
            
            # Calculate annualized volatility
            volatility = returns.std() * np.sqrt(252)
            
            return volatility
            
        except Exception as e:
            logging.error(f"Error calculating volatility: {str(e)}")
            return 0.0

    def assess_trade_risk(self, symbol: str, type: str, stop_loss: float, 
                         take_profit: float, other_positions: List[Dict]) -> Dict:
        """
        Assess the risk of a potential trade
        
        Args:
            symbol (str): Trading symbol
            type (str): Trade type ('BUY' or 'SELL')
            stop_loss (float): Stop loss price
            take_profit (float): Take profit price
            other_positions (List[Dict]): List of other open positions
            
        Returns:
            Dict: Risk assessment results
        """
        try:
            self._reset_daily_metrics()
            
            # Get current price
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                raise Exception(f"Failed to get tick data for {symbol}")
            
            entry_price = tick.ask if type == "BUY" else tick.bid
            
            # Calculate position size
            position_size = self._calculate_position_size(symbol, stop_loss, entry_price)
            if position_size == 0:
                return {"allowed": False, "reason": "Invalid position size"}
            
            # Calculate risk amount
            account_info = mt5.account_info()
            if account_info is None:
                raise Exception("Failed to get account info")
            
            risk_amount = abs(entry_price - stop_loss) * position_size * account_info.trade_tick_value
            risk_percent = (risk_amount / account_info.balance) * 100
            
            # Check daily risk limit
            if self.risk_metrics["daily_risk"] + risk_percent > self.max_daily_risk:
                return {
                    "allowed": False,
                    "reason": f"Daily risk limit exceeded: {self.risk_metrics['daily_risk'] + risk_percent:.2f}%"
                }
            
            # Check correlation with other positions
            other_symbols = [pos["symbol"] for pos in other_positions]
            correlations = self._calculate_correlation(symbol, other_symbols)
            
            for other_symbol, corr in correlations.items():
                if abs(corr) > self.max_correlation:
                    return {
                        "allowed": False,
                        "reason": f"High correlation with {other_symbol}: {corr:.2f}"
                    }
            
            # Check drawdown
            if self.risk_metrics["current_drawdown"] > self.max_drawdown:
                return {
                    "allowed": False,
                    "reason": f"Maximum drawdown exceeded: {self.risk_metrics['current_drawdown']:.2f}%"
                }
            
            # Calculate volatility
            volatility = self._calculate_volatility(symbol)
            
            # Calculate risk-reward ratio
            risk = abs(entry_price - stop_loss)
            reward = abs(take_profit - entry_price)
            risk_reward = reward / risk if risk != 0 else 0
            
            return {
                "allowed": True,
                "position_size": position_size,
                "risk_amount": risk_amount,
                "risk_percent": risk_percent,
                "volatility": volatility,
                "risk_reward": risk_reward,
                "correlations": correlations
            }
            
        except Exception as e:
            logging.error(f"Error assessing trade risk: {str(e)}")
            return {"allowed": False, "reason": str(e)}

    def update_risk_metrics(self, trade_result: Dict = None):
        """Update risk metrics based on trade result"""
        try:
            # Reset daily metrics if needed
            self._reset_daily_metrics()
            
            if trade_result:
                # Update metrics based on trade result
                self.risk_metrics["daily_risk"] += trade_result.get("risk", 0.0)
                self.risk_metrics["trades_today"] += 1
                
                # Update drawdown if trade resulted in a loss
                if trade_result.get("profit", 0.0) < 0:
                    self.risk_metrics["current_drawdown"] = self._calculate_drawdown(
                        self.risk_metrics.get("equity_curve", [])
                    )
                    self.risk_metrics["max_drawdown"] = max(
                        self.risk_metrics["max_drawdown"],
                        self.risk_metrics["current_drawdown"]
                    )
            
            # Save updated metrics
            self._save_risk_metrics()
            
        except Exception as e:
            logging.error(f"Error updating risk metrics: {str(e)}")

    def get_risk_metrics(self) -> Dict:
        """Get current risk metrics"""
        return self.risk_metrics

    def __del__(self):
        """Cleanup when object is destroyed"""
        if MT5_AVAILABLE:
            mt5.shutdown() 