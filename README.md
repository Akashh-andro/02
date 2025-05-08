# Advanced Trading System

A comprehensive trading system that supports both cryptocurrency and stock trading with advanced features including backtesting, risk management, and multiple trading strategies. The system includes real-time TradingView charts and is deployed using Streamlit.

## Features

- Multiple trading strategies (MACD, RSI, Bollinger Bands, Moving Average Crossover, ADX, and Combined)
- Real-time market data integration with support for both crypto and stock markets
- Advanced risk management system with position sizing and correlation analysis
- Comprehensive backtesting engine with performance metrics
- Configurable trading parameters
- Real-time performance monitoring and reporting
- Support for multiple timeframes
- Technical indicator calculations
- Logging and error handling
- Interactive TradingView charts
- Streamlit web interface

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Akashh-andro/01.git
cd 01
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

4. Install TA-Lib:
- Windows: Download and install from [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib)
- Linux: `sudo apt-get install ta-lib`
- macOS: `brew install ta-lib`

## Configuration

1. Create a `config.json` file in the root directory:
```json
{
    "trading": {
        "initial_capital": 10000,
        "max_positions": 5,
        "risk_per_trade": 0.02,
        "max_daily_risk": 0.05,
        "max_drawdown": 0.15,
        "max_correlation": 0.7
    },
    "exchange": {
        "name": "binance",
        "api_key": "your_api_key",
        "api_secret": "your_api_secret",
        "testnet": true
    },
    "strategies": {
        "default": {
            "type": "MACD",
            "parameters": {
                "fast_period": 12,
                "slow_period": 26,
                "signal_period": 9
            }
        }
    },
    "symbols": ["BTC/USDT", "ETH/USDT"],
    "timeframes": ["1h", "4h", "1d"],
    "logging": {
        "level": "INFO",
        "file": "trading.log"
    },
    "backtesting": {
        "start_date": "2023-01-01",
        "end_date": "2023-12-31",
        "initial_capital": 10000
    }
}
```

## Running Locally

1. Start the Streamlit app:
```bash
streamlit run app/streamlit_app.py
```

2. Open your browser and navigate to `http://localhost:8501`

## Deployment

### GitHub Deployment

1. Create a new repository on GitHub
2. Push your code:
```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/Akashh-andro/01.git
git push -u origin main
```

### Streamlit Cloud Deployment

1. Go to [Streamlit Cloud](https://streamlit.io/cloud)
2. Sign in with your GitHub account
3. Click "New app"
4. Select your repository
5. Set the main file path to `app/streamlit_app.py`
6. Click "Deploy"

## Usage

### Web Interface

The Streamlit web interface provides:

1. TradingView Charts
   - Real-time price charts
   - Multiple timeframes
   - Technical indicators
   - Drawing tools

2. Technical Analysis
   - TradingView analysis
   - Multiple indicators
   - Buy/Sell signals

3. Backtesting
   - Strategy performance
   - Equity curves
   - Risk metrics
   - Trade history

4. Risk Management
   - Position sizing
   - Risk metrics
   - Portfolio overview
   - Performance tracking

5. Trading Controls
   - Start/Stop trading
   - Strategy selection
   - Symbol selection
   - Timeframe selection

### Command Line Interface

The system also provides a command-line interface:

```bash
python -m app.main
```

Available commands:
- `help`: Show available commands
- `stop`: Stop trading
- `start`: Start trading
- `backtest <symbol> <strategy>`: Run backtest
- `metrics`: Show performance metrics
- `risk`: Show risk report
- `exit`: Exit application

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This software is for educational purposes only. Do not risk money which you are afraid to lose. USE THE SOFTWARE AT YOUR OWN RISK. THE AUTHORS AND ALL AFFILIATES ASSUME NO RESPONSIBILITY FOR YOUR TRADING RESULTS. 