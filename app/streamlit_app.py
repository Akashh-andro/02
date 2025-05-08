import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path to import trading system modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.trading_engine import TradingEngine
from app.data_manager import DataManager
from app.risk_manager import RiskManager
from app.strategies import StrategyManager
from app.backtesting import Backtester
from app.indicators import TechnicalIndicators

# Initialize session state
if 'trading_engine' not in st.session_state:
    st.session_state.trading_engine = TradingEngine()
if 'data_manager' not in st.session_state:
    st.session_state.data_manager = DataManager()
if 'risk_manager' not in st.session_state:
    st.session_state.risk_manager = RiskManager()
if 'strategy_manager' not in st.session_state:
    st.session_state.strategy_manager = StrategyManager()
if 'backtester' not in st.session_state:
    st.session_state.backtester = Backtester()
if 'indicators' not in st.session_state:
    st.session_state.indicators = TechnicalIndicators()

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
    data = st.session_state.data_manager.get_market_data(symbol, timeframe)
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
    
    # Recent Trades
    st.subheader("Recent Trades")
    trades_df = pd.DataFrame({
        'Time': [],
        'Symbol': [],
        'Type': [],
        'Entry': [],
        'Exit': [],
        'P/L': []
    })
    st.dataframe(trades_df)

elif page == "Trading":
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
            st.session_state.trading_engine.start_trading(symbol, strategy, lot_size, risk_percent)
    with col2:
        if st.button("Stop Trading", type="secondary"):
            st.session_state.trading_engine.stop_trading()
    with col3:
        if st.button("Close All Positions", type="secondary"):
            st.session_state.trading_engine.close_all_positions()

elif page == "Backtesting":
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

elif page == "Analysis":
    st.header("Market Analysis")
    
    # Technical Analysis
    symbol = st.selectbox("Analysis Pair", ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"])
    timeframe = st.selectbox("Analysis Timeframe", ["1m", "5m", "15m", "1h", "4h", "1d"])
    
    # Get indicators
    data = st.session_state.data_manager.get_market_data(symbol, timeframe)
    if data is not None:
        indicators = st.session_state.indicators.calculate_indicators(data)
        
        # Display indicators
        st.subheader("Technical Indicators")
        col1, col2 = st.columns(2)
        with col1:
            st.write("RSI:", f"{indicators['rsi']:.2f}")
            st.write("MACD:", f"{indicators['macd']:.2f}")
        with col2:
            st.write("Bollinger Bands:", f"{indicators['bb_upper']:.2f} / {indicators['bb_lower']:.2f}")
            st.write("ATR:", f"{indicators['atr']:.2f}")

elif page == "Settings":
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
        st.success("Settings saved successfully!")

# Footer
st.markdown("---")
st.markdown("Quantum Forex Trading System v1.0") 