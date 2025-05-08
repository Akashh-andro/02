import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_system.log'),
        logging.StreamHandler()
    ]
)

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Import trading system components
from risk_manager import RiskManager
from model_trainer import ForexModelTrainer
from backtester import ForexBacktester
from signal_generator import SignalGenerator
from position_manager import PositionManager
from market_analyzer import MarketAnalyzer
from performance_analyzer import PerformanceAnalyzer

def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if 'model_trainer' not in st.session_state:
        st.session_state.model_trainer = ForexModelTrainer()
    if 'backtester' not in st.session_state:
        st.session_state.backtester = ForexBacktester()
    if 'signal_generator' not in st.session_state:
        st.session_state.signal_generator = SignalGenerator()
    if 'position_manager' not in st.session_state:
        st.session_state.position_manager = PositionManager()
    if 'market_analyzer' not in st.session_state:
        st.session_state.market_analyzer = MarketAnalyzer()
    if 'risk_manager' not in st.session_state:
        st.session_state.risk_manager = RiskManager(
            max_risk_per_trade=1.0,
            max_daily_risk=5.0,
            max_correlation=0.7,
            max_drawdown=20.0
        )
    if 'performance_analyzer' not in st.session_state:
        st.session_state.performance_analyzer = PerformanceAnalyzer()
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
        st.metric("Current Capital", f"${risk_report.get('balance', 0):,.2f}")
    with col2:
        st.metric("Daily P/L", f"${risk_report.get('daily_pnl', 0):,.2f}")
    with col3:
        st.metric("Open Positions", str(risk_report.get('trades_today', 0)))
    
    # Risk Metrics
    st.subheader("Risk Metrics")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Current Drawdown", f"{risk_report.get('current_drawdown', 0):.2%}")
    with col2:
        st.metric("Max Drawdown", f"{risk_report.get('max_drawdown', 0):.2%}")
    with col3:
        st.metric("Daily Risk", f"{risk_report.get('daily_risk', 0):.2%}")

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
                st.metric("Position Size", f"{assessment.get('position_size', 0):.4f}")
            with col2:
                st.metric("Risk-Reward Ratio", f"{assessment.get('risk_reward', 0):.2f}")
            with col3:
                st.metric("Position Correlation", f"{assessment.get('correlations', {}).get(symbol, 0):.2f}")
            
            if assessment.get('allowed', False):
                st.success("Trade meets risk management criteria")
            else:
                st.error(f"Trade rejected: {assessment.get('reason', 'Unknown error')}")
                
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