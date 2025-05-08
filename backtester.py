import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import MetaTrader5 as mt5
from typing import Dict, List, Tuple, Optional
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json
import os
import ta

class ForexBacktester:
    def __init__(self, initial_balance: float = 10000.0):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.equity_curve = []
        self.trades = []
        self.current_positions = {}
        
        # Performance metrics
        self.metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'average_win': 0.0,
            'average_loss': 0.0,
            'largest_win': 0.0,
            'largest_loss': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'calmar_ratio': 0.0,
            'recovery_factor': 0.0,
            'expectancy': 0.0,
            'risk_reward_ratio': 0.0
        }
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('backtest.log'),
                logging.StreamHandler()
            ]
        )

    def _calculate_position_size(self, pair: str, risk_percent: float, 
                               stop_loss_pips: float) -> float:
        """Calculate position size based on risk parameters"""
        account_info = mt5.account_info()
        if account_info is None:
            raise ValueError("Failed to get account info")
            
        balance = account_info.balance
        risk_amount = balance * risk_percent / 100
        
        # Get pip value
        symbol_info = mt5.symbol_info(pair)
        if symbol_info is None:
            raise ValueError(f"Failed to get symbol info for {pair}")
            
        pip_value = symbol_info.trade_tick_value
        
        # Calculate position size
        position_size = risk_amount / (stop_loss_pips * pip_value)
        
        # Round to standard lot sizes
        position_size = round(position_size / 0.01) * 0.01
        
        return position_size

    def _calculate_trade_metrics(self) -> None:
        """Calculate comprehensive trade metrics"""
        if not self.trades:
            return
            
        # Basic metrics
        self.metrics['total_trades'] = len(self.trades)
        self.metrics['winning_trades'] = len([t for t in self.trades if t['profit'] > 0])
        self.metrics['losing_trades'] = len([t for t in self.trades if t['profit'] <= 0])
        
        # Win rate
        self.metrics['win_rate'] = (
            self.metrics['winning_trades'] / self.metrics['total_trades']
            if self.metrics['total_trades'] > 0 else 0
        )
        
        # Profit metrics
        winning_trades = [t['profit'] for t in self.trades if t['profit'] > 0]
        losing_trades = [t['profit'] for t in self.trades if t['profit'] <= 0]
        
        self.metrics['average_win'] = np.mean(winning_trades) if winning_trades else 0
        self.metrics['average_loss'] = np.mean(losing_trades) if losing_trades else 0
        self.metrics['largest_win'] = max(winning_trades) if winning_trades else 0
        self.metrics['largest_loss'] = min(losing_trades) if losing_trades else 0
        
        # Profit factor
        gross_profit = sum(winning_trades) if winning_trades else 0
        gross_loss = abs(sum(losing_trades)) if losing_trades else 0
        self.metrics['profit_factor'] = (
            gross_profit / gross_loss if gross_loss != 0 else float('inf')
        )
        
        # Risk metrics
        returns = pd.Series([t['profit'] for t in self.trades])
        self.metrics['sharpe_ratio'] = (
            np.sqrt(252) * returns.mean() / returns.std()
            if len(returns) > 1 else 0
        )
        
        # Sortino ratio (using negative returns only)
        negative_returns = returns[returns < 0]
        self.metrics['sortino_ratio'] = (
            np.sqrt(252) * returns.mean() / negative_returns.std()
            if len(negative_returns) > 1 else 0
        )
        
        # Maximum drawdown
        cumulative_returns = returns.cumsum()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = cumulative_returns - rolling_max
        self.metrics['max_drawdown'] = abs(drawdowns.min())
        
        # Calmar ratio
        self.metrics['calmar_ratio'] = (
            returns.mean() * 252 / self.metrics['max_drawdown']
            if self.metrics['max_drawdown'] != 0 else 0
        )
        
        # Recovery factor
        total_return = cumulative_returns.iloc[-1]
        self.metrics['recovery_factor'] = (
            total_return / self.metrics['max_drawdown']
            if self.metrics['max_drawdown'] != 0 else 0
        )
        
        # Expectancy
        self.metrics['expectancy'] = (
            (self.metrics['win_rate'] * self.metrics['average_win']) +
            ((1 - self.metrics['win_rate']) * self.metrics['average_loss'])
        )
        
        # Risk-reward ratio
        self.metrics['risk_reward_ratio'] = (
            abs(self.metrics['average_win'] / self.metrics['average_loss'])
            if self.metrics['average_loss'] != 0 else 0
        )

    def run_backtest(self, pair: str, start_date: datetime, end_date: datetime,
                    risk_percent: float = 1.0) -> Dict[str, float]:
        """Run backtest on historical data"""
        logging.info(f"Starting backtest for {pair}")
        
        # Get historical data
        rates = mt5.copy_rates_range(
            pair,
            mt5.TIMEFRAME_H1,
            start_date,
            end_date
        )
        
        if rates is None:
            raise ValueError(f"Failed to get historical data for {pair}")
        
        # Convert to DataFrame
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        # Initialize equity curve
        self.equity_curve = [self.initial_balance]
        
        # Process each bar
        for i in range(len(df)):
            current_bar = df.iloc[i]
            
            # Update existing positions
            self._update_positions(current_bar)
            
            # Generate trading signal (implement your strategy here)
            signal = self._generate_signal(df.iloc[:i+1])
            
            if signal:
                # Calculate position size
                position_size = self._calculate_position_size(
                    pair, risk_percent, signal['stop_loss_pips']
                )
                
                # Execute trade
                self._execute_trade(
                    pair=pair,
                    direction=signal['direction'],
                    entry_price=current_bar['close'],
                    stop_loss=signal['stop_loss'],
                    take_profit=signal['take_profit'],
                    position_size=position_size,
                    timestamp=current_bar['time']
                )
            
            # Update equity curve
            self.equity_curve.append(self.balance)
        
        # Calculate final metrics
        self._calculate_trade_metrics()
        
        # Generate performance report
        self._generate_report(pair)
        
        return self.metrics

    def _update_positions(self, current_bar: pd.Series) -> None:
        """Update existing positions and check for exits"""
        for pair, position in list(self.current_positions.items()):
            # Check for stop loss
            if position['direction'] == 'BUY':
                if current_bar['low'] <= position['stop_loss']:
                    self._close_position(pair, current_bar['time'], position['stop_loss'])
                elif current_bar['high'] >= position['take_profit']:
                    self._close_position(pair, current_bar['time'], position['take_profit'])
            else:  # SELL
                if current_bar['high'] >= position['stop_loss']:
                    self._close_position(pair, current_bar['time'], position['stop_loss'])
                elif current_bar['low'] <= position['take_profit']:
                    self._close_position(pair, current_bar['time'], position['take_profit'])

    def _execute_trade(self, pair: str, direction: str, entry_price: float,
                      stop_loss: float, take_profit: float, position_size: float,
                      timestamp: datetime) -> None:
        """Execute a new trade"""
        # Calculate pip value
        pip_value = mt5.symbol_info(pair).trade_tick_value
        
        # Calculate potential profit/loss
        if direction == 'BUY':
            profit_pips = (take_profit - entry_price) / pip_value
            loss_pips = (entry_price - stop_loss) / pip_value
        else:  # SELL
            profit_pips = (entry_price - take_profit) / pip_value
            loss_pips = (stop_loss - entry_price) / pip_value
        
        # Record trade
        trade = {
            'pair': pair,
            'direction': direction,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'position_size': position_size,
            'entry_time': timestamp,
            'exit_time': None,  # Will be set when position is closed
            'profit_pips': profit_pips,
            'loss_pips': loss_pips,
            'profit': 0.0,  # Will be updated when closed
            'status': 'OPEN'
        }
        
        # Add to current positions
        self.current_positions[pair] = trade

    def _close_position(self, pair: str, timestamp: datetime, exit_price: float) -> None:
        """Close an existing position"""
        if pair not in self.current_positions:
            return
            
        position = self.current_positions[pair]
        
        # Calculate profit/loss
        pip_value = mt5.symbol_info(pair).trade_tick_value
        if position['direction'] == 'BUY':
            profit = (exit_price - position['entry_price']) * position['position_size'] * pip_value
        else:  # SELL
            profit = (position['entry_price'] - exit_price) * position['position_size'] * pip_value
        
        # Update balance
        self.balance += profit
        
        # Record trade
        trade = {
            'pair': pair,
            'direction': position['direction'],
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'entry_time': position['entry_time'],
            'exit_time': timestamp,
            'position_size': position['position_size'],
            'stop_loss': position['stop_loss'],
            'take_profit': position['take_profit'],
            'profit': profit,
            'status': 'CLOSED'
        }
        
        self.trades.append(trade)
        del self.current_positions[pair]

    def _generate_signal(self, data: pd.DataFrame) -> Optional[Dict]:
        """Generate trading signal using the trained model"""
        try:
            if len(data) < 60:  # Need enough data for features
                return None
                
            # Calculate features
            df = data.copy()
            
            # Basic price features
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            
            # Volatility features
            df['volatility'] = df['returns'].rolling(window=20).std()
            df['realized_vol'] = df['returns'].rolling(window=20).apply(lambda x: np.sqrt(np.sum(x**2)))
            
            # RSI
            df['rsi'] = ta.rsi(df['close'], length=14)
            
            # MACD
            macd = ta.macd(df['close'])
            df['macd'] = macd['MACD_12_26_9']
            df['macd_signal'] = macd['MACDs_12_26_9']
            df['macd_hist'] = macd['MACDh_12_26_9']
            
            # Bollinger Bands
            sma = df['close'].rolling(window=20).mean()
            std = df['close'].rolling(window=20).std()
            df['bb_upper'] = sma + (std * 2)
            df['bb_middle'] = sma
            df['bb_lower'] = sma - (std * 2)
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            
            # ATR
            df['atr'] = ta.atr(df['high'], df['low'], df['close'])
            
            # ADX
            df['adx'] = 25.0  # Default value
            try:
                adx = ta.adx(df['high'], df['low'], df['close'])
                df['adx'] = adx['ADX_14']
            except:
                pass
            
            # Fill missing values
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            # Get the latest data point
            current_price = df['close'].iloc[-1]
            atr = df['atr'].iloc[-1]
            
            # Generate signal based on technical indicators
            rsi = df['rsi'].iloc[-1]
            macd_hist = df['macd_hist'].iloc[-1]
            bb_width = df['bb_width'].iloc[-1]
            
            # Signal conditions
            if rsi < 30 and macd_hist > 0 and bb_width > 0.01:  # Oversold and MACD turning positive
                return {
                    'direction': 'BUY',
                    'stop_loss': current_price - (2 * atr),
                    'take_profit': current_price + (3 * atr),
                    'stop_loss_pips': 2 * atr,
                    'confidence': 0.8
                }
            elif rsi > 70 and macd_hist < 0 and bb_width > 0.01:  # Overbought and MACD turning negative
                return {
                    'direction': 'SELL',
                    'stop_loss': current_price + (2 * atr),
                    'take_profit': current_price - (3 * atr),
                    'stop_loss_pips': 2 * atr,
                    'confidence': 0.8
                }
            
            return None
            
        except Exception as e:
            logging.error(f"Error generating signal: {str(e)}")
            return None

    def _generate_report(self, pair: str) -> None:
        """Generate comprehensive backtest report"""
        # Create reports directory
        os.makedirs('reports', exist_ok=True)
        
        # Save metrics
        with open(f'reports/{pair}_backtest_metrics.json', 'w') as f:
            json.dump(self.metrics, f, indent=4)
        
        # Generate equity curve plot
        plt.figure(figsize=(12, 6))
        plt.plot(self.equity_curve)
        plt.title(f'Equity Curve - {pair}')
        plt.xlabel('Trades')
        plt.ylabel('Equity')
        plt.grid(True)
        plt.savefig(f'reports/{pair}_equity_curve.png')
        plt.close()
        
        # Generate trade distribution plot
        profits = [t['profit'] for t in self.trades]
        plt.figure(figsize=(10, 6))
        sns.histplot(profits, bins=50)
        plt.title(f'Trade Profit Distribution - {pair}')
        plt.xlabel('Profit')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.savefig(f'reports/{pair}_profit_distribution.png')
        plt.close()
        
        # Generate monthly returns heatmap
        if self.trades:
            monthly_returns = pd.DataFrame(self.trades)
            monthly_returns['month'] = pd.to_datetime(monthly_returns['entry_time']).dt.to_period('M')
            monthly_returns = monthly_returns.groupby('month')['profit'].sum()
            monthly_returns = monthly_returns.to_frame()
            monthly_returns['year'] = monthly_returns.index.astype(str).str[:4]
            monthly_returns['month'] = monthly_returns.index.astype(str).str[5:7]
            monthly_returns = monthly_returns.pivot(index='year', columns='month', values='profit')
            
            plt.figure(figsize=(12, 8))
            sns.heatmap(monthly_returns, annot=True, fmt='.0f', cmap='RdYlGn')
            plt.title(f'Monthly Returns Heatmap - {pair}')
            plt.savefig(f'reports/{pair}_monthly_returns.png')
            plt.close()

def main():
    # Initialize MT5
    if not mt5.initialize():
        raise RuntimeError("Failed to initialize MT5")
    
    # Create backtester
    backtester = ForexBacktester(initial_balance=10000.0)
    
    # Define pairs to backtest
    pairs = ['EURUSD', 'GBPUSD', 'USDJPY']
    
    # Define date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)  # 1 year of data
    
    # Run backtests
    for pair in pairs:
        try:
            logging.info(f"Running backtest for {pair}")
            metrics = backtester.run_backtest(
                pair,
                start_date,
                end_date,
                risk_percent=1.0
            )
            
            logging.info(f"Backtest results for {pair}:")
            logging.info(f"Total Trades: {metrics['total_trades']}")
            logging.info(f"Win Rate: {metrics['win_rate']:.2%}")
            logging.info(f"Profit Factor: {metrics['profit_factor']:.2f}")
            logging.info(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
            logging.info(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
            
        except Exception as e:
            logging.error(f"Error backtesting {pair}: {str(e)}")
    
    mt5.shutdown()

if __name__ == "__main__":
    main() 