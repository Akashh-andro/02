# Quantum Forex Trading System

A comprehensive Streamlit-based forex trading system with advanced features including real-time market data, technical analysis, backtesting, and automated trading strategies.

## Features

- 📊 Real-time market data visualization
- 📈 Multiple trading strategies (Quantum, Trend Following, Mean Reversion)
- 🤖 Automated trading with risk management
- 📉 Advanced backtesting engine
- 📱 User-friendly Streamlit interface
- 🔍 Technical analysis tools
- 🎯 Position sizing and risk management
- 📊 Performance analytics
- 🔐 Secure API integration

## Live Demo

Visit our live demo at: [Quantum Forex Trading System](https://quantum-forex.streamlit.app)

## Quick Start

1. Clone the repository:
```bash
git clone https://github.com/Akashh-andro/02.git
cd 02
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file in the root directory:
```env
BINANCE_API_KEY=your_binance_api_key
BINANCE_API_SECRET=your_binance_api_secret
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
ALPACA_API_KEY=your_alpaca_api_key
ALPACA_API_SECRET=your_alpaca_api_secret
```

5. Run the app:
```bash
streamlit run main.py
```

## System Components

### 1. Market Data Integration
- Real-time data from multiple sources (Binance, Alpaca, YFinance)
- WebSocket connections for live price updates
- Historical data for backtesting

### 2. Trading Strategies
- Quantum Trading Algorithm
- Trend Following Strategy
- Mean Reversion Strategy
- Custom Strategy Framework

### 3. Risk Management
- Position Sizing
- Stop Loss Management
- Take Profit Optimization
- Portfolio Correlation Analysis
- Maximum Drawdown Control

### 4. Technical Analysis
- Multiple Technical Indicators
- Custom Indicator Framework
- Signal Generation
- Pattern Recognition

### 5. Backtesting Engine
- Historical Performance Analysis
- Strategy Optimization
- Risk Metrics Calculation
- Performance Visualization

### 6. User Interface
- Interactive Charts
- Trading Controls
- Performance Metrics
- System Settings
- Real-time Notifications

## Deployment

### Local Deployment
```bash
streamlit run main.py
```

### Streamlit Cloud Deployment
1. Fork this repository
2. Connect your GitHub account to Streamlit Cloud
3. Deploy the app from your forked repository
4. Configure environment variables in Streamlit Cloud

## API Integration

The system supports multiple trading APIs:
- Binance (Crypto)
- Alpaca (Stocks/Forex)
- YFinance (Market Data)
- Telegram (Notifications)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Security

- API keys are stored securely using environment variables
- All sensitive data is encrypted
- Regular security audits
- Rate limiting implementation

## Support

For support, please:
1. Check the [Issues](https://github.com/Akashh-andro/02/issues) page
2. Join our [Telegram Group](https://t.me/quantumforex)
3. Email us at support@quantumforex.com

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

Trading forex carries a high level of risk and may not be suitable for all investors. Before deciding to trade forex, you should carefully consider your investment objectives, level of experience, and risk appetite. 