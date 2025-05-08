import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging
import json
from scipy import stats
import pandas_ta as ta
import streamlit as st
import plotly.graph_objects as go

class RiskManager:
    def __init__(self, max_risk_per_trade: float = 1.0, max_daily_risk: float = 5.0,
                 max_correlation: float = 0.7, max_drawdown: float = 20.0):
        self.max_risk_per_trade = max_risk_per_trade
        self.max_daily_risk = max_daily_risk
        self.max_correlation = max_correlation
        self.max_drawdown = max_drawdown
        
        # Initialize MT5
        if not mt5.initialize():
            logging.error("Failed to initialize MT5")
            raise Exception("MT5 initialization failed")
        
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
        """Assess risk for a potential trade"""
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
        mt5.shutdown()

def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if 'risk_manager' not in st.session_state:
        st.session_state.risk_manager = RiskManager(
            max_risk_per_trade=1.0,
            max_daily_risk=5.0,
            max_correlation=0.7,
            max_drawdown=20.0
        )
    if 'is_trading' not in st.session_state:
        st.session_state.is_trading = False

def main():
    """Main Streamlit application"""
    # Initialize session state
    initialize_session_state()
    
    # Page config
    st.set_page_config(
        page_title="Quantum Forex Trading System",
        page_icon="📈",
        layout="wide"
    )
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Dashboard", "Risk Management", "Settings"])
    
    # Main content
    st.title("Quantum Forex Trading System")
    
    if page == "Dashboard":
        show_dashboard()
    elif page == "Risk Management":
        show_risk_management()
    elif page == "Settings":
        show_settings()

def show_dashboard():
    """Display the trading dashboard"""
    st.header("Trading Dashboard")
    
    # Risk Overview
    col1, col2, col3 = st.columns(3)
    risk_report = st.session_state.risk_manager.get_risk_metrics()
    
    with col1:
        st.metric("Current Capital", f"${risk_report['balance']:,.2f}")
    with col2:
        st.metric("Daily P/L", f"${risk_report['daily_pnl']:,.2f}")
    with col3:
        st.metric("Open Positions", str(risk_report['trades_today']))
    
    # Risk Metrics
    st.subheader("Risk Metrics")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Current Drawdown", f"{risk_report['current_drawdown']:.2%}")
    with col2:
        st.metric("Max Drawdown", f"{risk_report['max_drawdown']:.2%}")
    with col3:
        st.metric("Daily Risk", f"{risk_report['daily_risk']:.2%}")

def show_risk_management():
    """Display risk management controls"""
    st.header("Risk Management")
    
    # Risk Parameters
    st.subheader("Risk Parameters")
    col1, col2 = st.columns(2)
    
    with col1:
        symbol = st.selectbox("Trading Pair", ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"])
        entry_price = st.number_input("Entry Price", min_value=0.0, value=1.0, step=0.0001)
    with col2:
        stop_loss = st.number_input("Stop Loss", min_value=0.0, value=0.99, step=0.0001)
        take_profit = st.number_input("Take Profit", min_value=0.0, value=1.01, step=0.0001)
    
    if st.button("Calculate Risk", type="primary"):
        try:
            assessment = st.session_state.risk_manager.assess_trade_risk(
                symbol=symbol,
                type="BUY",
                stop_loss=stop_loss,
                take_profit=take_profit,
                other_positions=[]
            )
            
            st.subheader("Risk Assessment")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Position Size", f"{assessment['position_size']:.4f}")
            with col2:
                st.metric("Risk-Reward Ratio", f"{assessment['risk_reward']:.2f}")
            with col3:
                st.metric("Position Correlation", f"{assessment['correlations'].get(symbol, 0.0):.2f}")
            
            if assessment['allowed']:
                st.success("Trade meets risk management criteria")
            else:
                st.error(f"Trade rejected: {assessment['reason']}")
                
        except Exception as e:
            st.error(f"Error calculating risk: {str(e)}")

def show_settings():
    """Display risk management settings"""
    st.header("Risk Management Settings")
    
    # Risk Limits
    st.subheader("Risk Limits")
    col1, col2 = st.columns(2)
    
    with col1:
        max_risk = st.number_input(
            "Max Risk per Trade (%)", 
            min_value=0.1, 
            max_value=5.0, 
            value=float(st.session_state.risk_manager.max_risk_per_trade * 100),
            step=0.1
        )
        max_correlation = st.number_input(
            "Max Position Correlation", 
            min_value=0.0, 
            max_value=1.0, 
            value=st.session_state.risk_manager.max_correlation,
            step=0.1
        )
    
    with col2:
        max_portfolio_risk = st.number_input(
            "Max Portfolio Risk (%)", 
            min_value=1.0, 
            max_value=20.0, 
            value=float(st.session_state.risk_manager.max_daily_risk * 100),
            step=1.0
        )
        max_drawdown = st.number_input(
            "Max Drawdown (%)", 
            min_value=5.0, 
            max_value=50.0, 
            value=float(st.session_state.risk_manager.max_drawdown * 100),
            step=5.0
        )
    
    if st.button("Save Settings", type="primary"):
        try:
            # Update risk manager settings
            st.session_state.risk_manager.max_risk_per_trade = max_risk / 100
            st.session_state.risk_manager.max_daily_risk = max_portfolio_risk / 100
            st.session_state.risk_manager.max_correlation = max_correlation
            st.session_state.risk_manager.max_drawdown = max_drawdown / 100
            
            st.success("Settings saved successfully!")
        except Exception as e:
            st.error(f"Error saving settings: {str(e)}")

if __name__ == "__main__":
    main() 