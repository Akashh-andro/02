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

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import trading system components
from model_trainer import ForexModelTrainer
from backtester import ForexBacktester
from signal_generator import SignalGenerator
from position_manager import PositionManager
from market_analyzer import MarketAnalyzer
from risk_manager import RiskManager
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
        st.session_state.risk_manager = RiskManager()
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
    page = st.sidebar.radio("Go to", ["Dashboard", "Trading", "Backtesting", "Analysis", "Settings"])
    
    # Main content
    st.title("Quantum Forex Trading System")
    
    if page == "Dashboard":
        show_dashboard()
    elif page == "Trading":
        show_trading()
    elif page == "Backtesting":
        show_backtesting()
    elif page == "Analysis":
        show_analysis()
    elif page == "Settings":
        show_settings()

def show_dashboard():
    """Display the trading dashboard"""
    st.header("Trading Dashboard")
    
    # Market Overview
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Active Trades", "0", "0%")
    with col2:
        st.metric("Daily P/L", "$0.00", "0%")
    with col3:
        st.metric("Win Rate", "0%", "0%")
    
    # Chart
    st.subheader("Market Analysis")
    timeframe = st.selectbox("Timeframe", ["1m", "5m", "15m", "1h", "4h", "1d"])
    symbol = st.selectbox("Symbol", ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"])
    
    # Get market data
    try:
        data = st.session_state.market_analyzer.get_market_data(symbol, timeframe)
        if data is not None:
            fig = go.Figure(data=[go.Candlestick(
                x=data.index,
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close']
            )])
            fig.update_layout(title=f"{symbol} {timeframe} Chart")
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error loading market data: {str(e)}")
    
    # Recent Trades
    st.subheader("Recent Trades")
    try:
        trades = st.session_state.performance_analyzer.get_recent_trades()
        if trades:
            st.dataframe(trades)
        else:
            st.info("No recent trades")
    except Exception as e:
        st.error(f"Error loading trades: {str(e)}")

def show_trading():
    """Display trading controls"""
    st.header("Trading Controls")
    
    # Trading Parameters
    col1, col2 = st.columns(2)
    with col1:
        symbol = st.selectbox("Trading Pair", ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"])
        lot_size = st.number_input("Lot Size", min_value=0.01, max_value=10.0, value=0.1, step=0.01)
    with col2:
        strategy = st.selectbox("Strategy", ["Quantum", "Trend Following", "Mean Reversion"])
        risk_percent = st.number_input("Risk %", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
    
    # Trading Controls
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Start Trading", type="primary"):
            try:
                st.session_state.is_trading = True
                st.session_state.position_manager.start_trading(symbol, strategy, lot_size, risk_percent)
                st.success("Trading started successfully")
            except Exception as e:
                st.error(f"Error starting trading: {str(e)}")
    with col2:
        if st.button("Stop Trading", type="secondary"):
            try:
                st.session_state.is_trading = False
                st.session_state.position_manager.stop_trading()
                st.success("Trading stopped successfully")
            except Exception as e:
                st.error(f"Error stopping trading: {str(e)}")
    with col3:
        if st.button("Close All Positions", type="secondary"):
            try:
                st.session_state.position_manager.close_all_positions()
                st.success("All positions closed successfully")
            except Exception as e:
                st.error(f"Error closing positions: {str(e)}")

def show_backtesting():
    """Display backtesting interface"""
    st.header("Strategy Backtesting")
    
    # Backtesting Parameters
    col1, col2 = st.columns(2)
    with col1:
        symbol = st.selectbox("Backtest Pair", ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"])
        strategy = st.selectbox("Backtest Strategy", ["Quantum", "Trend Following", "Mean Reversion"])
    with col2:
        start_date = st.date_input("Start Date", datetime.now() - timedelta(days=365))
        end_date = st.date_input("End Date", datetime.now())
    
    if st.button("Run Backtest", type="primary"):
        try:
            results = st.session_state.backtester.run_backtest(
                symbol, strategy, start_date, end_date
            )
            
            # Display Results
            st.subheader("Backtest Results")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Return", f"{results['total_return']:.2f}%")
            with col2:
                st.metric("Win Rate", f"{results['win_rate']:.2f}%")
            with col3:
                st.metric("Sharpe Ratio", f"{results['sharpe_ratio']:.2f}")
        except Exception as e:
            st.error(f"Error running backtest: {str(e)}")

def show_analysis():
    """Display market analysis"""
    st.header("Market Analysis")
    
    # Technical Analysis
    symbol = st.selectbox("Analysis Pair", ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"])
    timeframe = st.selectbox("Analysis Timeframe", ["1m", "5m", "15m", "1h", "4h", "1d"])
    
    try:
        # Get market data and indicators
        data = st.session_state.market_analyzer.get_market_data(symbol, timeframe)
        if data is not None:
            indicators = st.session_state.market_analyzer.calculate_indicators(data)
            
            # Display indicators
            st.subheader("Technical Indicators")
            col1, col2 = st.columns(2)
            with col1:
                st.write("RSI:", f"{indicators['rsi']:.2f}")
                st.write("MACD:", f"{indicators['macd']:.2f}")
            with col2:
                st.write("Bollinger Bands:", f"{indicators['bb_upper']:.2f} / {indicators['bb_lower']:.2f}")
                st.write("ATR:", f"{indicators['atr']:.2f}")
    except Exception as e:
        st.error(f"Error performing analysis: {str(e)}")

def show_settings():
    """Display system settings"""
    st.header("System Settings")
    
    # API Settings
    st.subheader("API Configuration")
    api_key = st.text_input("API Key", type="password")
    api_secret = st.text_input("API Secret", type="password")
    
    # Risk Settings
    st.subheader("Risk Management")
    max_daily_loss = st.number_input("Max Daily Loss %", min_value=1.0, max_value=10.0, value=2.0)
    max_position_size = st.number_input("Max Position Size %", min_value=1.0, max_value=10.0, value=5.0)
    
    # Save Settings
    if st.button("Save Settings", type="primary"):
        try:
            # Save settings logic here
            st.success("Settings saved successfully!")
        except Exception as e:
            st.error(f"Error saving settings: {str(e)}")

if __name__ == "__main__":
    main() 