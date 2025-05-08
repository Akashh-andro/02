import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from main import ForexTradingSystem
import json
import os

# Set page config
st.set_page_config(
    page_title="Forex Trading System",
    page_icon="📈",
    layout="wide"
)

# Load configuration
def load_config():
    with open('config.json', 'r') as f:
        return json.load(f)

# Initialize trading system
@st.cache_resource
def init_trading_system():
    return ForexTradingSystem()

# Main app
def main():
    st.title("Forex Trading System Dashboard")
    
    # Sidebar
    st.sidebar.header("Configuration")
    
    # Load config
    config = load_config()
    
    # Pair selection
    selected_pairs = st.sidebar.multiselect(
        "Select Currency Pairs",
        config['symbols'],
        default=config['symbols'][:5]
    )
    
    # Timeframe selection
    timeframe = st.sidebar.selectbox(
        "Select Timeframe",
        ["M1", "M5", "M15", "M30", "H1", "H4", "D1"],
        index=2  # Default to M15
    )
    
    # Risk management
    st.sidebar.subheader("Risk Management")
    risk_per_trade = st.sidebar.slider(
        "Risk per Trade (%)",
        min_value=0.1,
        max_value=5.0,
        value=config['risk_per_trade'] * 100,
        step=0.1
    )
    
    max_daily_risk = st.sidebar.slider(
        "Max Daily Risk (%)",
        min_value=0.5,
        max_value=10.0,
        value=config['max_daily_risk'] * 100,
        step=0.5
    )
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Market Analysis")
        
        # Trading system instance
        trading_system = init_trading_system()
        
        # Analysis for each pair
        for pair in selected_pairs:
            with st.expander(f"{pair} Analysis"):
                try:
                    # Get current market data
                    analysis = trading_system.signal_generator.analyze_market(pair)
                    
                    # Display technical indicators
                    st.write("Technical Indicators:")
                    indicators_df = pd.DataFrame({
                        'Indicator': ['RSI', 'MACD', 'SMA20', 'SMA50', 'SMA200'],
                        'Value': [
                            analysis['rsi'],
                            analysis['macd'],
                            analysis['sma_short'],
                            analysis['sma_medium'],
                            analysis['sma_long']
                        ]
                    })
                    st.dataframe(indicators_df)
                    
                    # Generate and display signal
                    signal = trading_system.signal_generator.generate_signal(analysis, pair)
                    if signal:
                        st.success(f"Signal: {signal['action']} at {signal['price']}")
                        st.write(f"Confidence: {signal['confidence']:.2%}")
                    else:
                        st.info("No trading signal at the moment")
                        
                except Exception as e:
                    st.error(f"Error analyzing {pair}: {str(e)}")
    
    with col2:
        st.subheader("Trading Status")
        
        # Display current positions
        st.write("Current Positions:")
        positions = trading_system.get_open_positions()
        if positions:
            positions_df = pd.DataFrame(positions)
            st.dataframe(positions_df)
        else:
            st.info("No open positions")
        
        # Display account summary
        st.write("Account Summary:")
        account_info = trading_system.get_account_info()
        st.write(f"Balance: ${account_info['balance']:,.2f}")
        st.write(f"Equity: ${account_info['equity']:,.2f}")
        st.write(f"Margin: ${account_info['margin']:,.2f}")
        st.write(f"Free Margin: ${account_info['free_margin']:,.2f}")
        
        # Trading controls
        st.subheader("Trading Controls")
        if st.button("Start Trading"):
            try:
                trading_system.start_trading(selected_pairs)
                st.success("Trading started successfully!")
            except Exception as e:
                st.error(f"Error starting trading: {str(e)}")
        
        if st.button("Stop Trading"):
            try:
                trading_system.stop_trading()
                st.success("Trading stopped successfully!")
            except Exception as e:
                st.error(f"Error stopping trading: {str(e)}")

if __name__ == "__main__":
    main() 