import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from .indicators import TechnicalIndicators
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class Backtester:
    def __init__(self, initial_capital=10000):
        self.initial_capital = initial_capital
        self.indicators = TechnicalIndicators()
        self.results = {}
        self.trades = []
        self.equity_curve = []
    
    def run_backtest(self, data, strategy, params=None):
        """Run backtest with given strategy and parameters"""
        self.data = data.copy()
        self.strategy = strategy
        self.params = params or {}
        
        # Calculate indicators
        self.indicators.calculate_all(self.data)
        
        # Initialize results
        self.results = {
            'initial_capital': self.initial_capital,
            'final_capital': self.initial_capital,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0,
            'returns': [],
            'equity_curve': []
        }
        
        # Run strategy
        self._run_strategy()
        
        # Calculate performance metrics
        self._calculate_metrics()
        
        return self.results
    
    def _run_strategy(self):
        """Execute trading strategy"""
        position = 0
        entry_price = 0
        entry_time = None
        
        for i in range(len(self.data)):
            current_data = self.data.iloc[i]
            current_indicators = {k: v.iloc[i] for k, v in self.indicators.indicators.items()}
            
            # Get strategy signal
            signal = self.strategy.generate_signal(current_data, current_indicators, self.params)
            
            if signal == 1 and position <= 0:  # Buy signal
                if position < 0:  # Close short position
                    self._record_trade(entry_time, current_data.name, 'short', entry_price, current_data['close'])
                position = 1
                entry_price = current_data['close']
                entry_time = current_data.name
                
            elif signal == -1 and position >= 0:  # Sell signal
                if position > 0:  # Close long position
                    self._record_trade(entry_time, current_data.name, 'long', entry_price, current_data['close'])
                position = -1
                entry_price = current_data['close']
                entry_time = current_data.name
            
            # Update equity curve
            self._update_equity(current_data['close'], position, entry_price)
    
    def _record_trade(self, entry_time, exit_time, position_type, entry_price, exit_price):
        """Record trade details"""
        trade = {
            'entry_time': entry_time,
            'exit_time': exit_time,
            'position_type': position_type,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'profit': (exit_price - entry_price) if position_type == 'long' else (entry_price - exit_price),
            'profit_pct': ((exit_price - entry_price) / entry_price) if position_type == 'long' else ((entry_price - exit_price) / entry_price)
        }
        self.trades.append(trade)
    
    def _update_equity(self, current_price, position, entry_price):
        """Update equity curve"""
        if position != 0:
            unrealized_pnl = (current_price - entry_price) * position
            current_equity = self.results['final_capital'] + unrealized_pnl
        else:
            current_equity = self.results['final_capital']
        
        self.equity_curve.append(current_equity)
    
    def _calculate_metrics(self):
        """Calculate performance metrics"""
        if not self.trades:
            return
        
        # Basic metrics
        self.results['total_trades'] = len(self.trades)
        self.results['winning_trades'] = len([t for t in self.trades if t['profit'] > 0])
        self.results['losing_trades'] = len([t for t in self.trades if t['profit'] <= 0])
        self.results['win_rate'] = self.results['winning_trades'] / self.results['total_trades']
        
        # Profit metrics
        total_profit = sum([t['profit'] for t in self.trades if t['profit'] > 0])
        total_loss = abs(sum([t['profit'] for t in self.trades if t['profit'] <= 0]))
        self.results['profit_factor'] = total_profit / total_loss if total_loss != 0 else float('inf')
        
        # Drawdown
        self.results['max_drawdown'] = self._calculate_max_drawdown()
        
        # Risk-adjusted returns
        self.results['sharpe_ratio'] = self._calculate_sharpe_ratio()
        
        # Final capital
        self.results['final_capital'] = self.initial_capital + sum([t['profit'] for t in self.trades])
    
    def _calculate_max_drawdown(self):
        """Calculate maximum drawdown"""
        peak = self.equity_curve[0]
        max_dd = 0
        
        for value in self.equity_curve:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            max_dd = max(max_dd, dd)
        
        return max_dd
    
    def _calculate_sharpe_ratio(self):
        """Calculate Sharpe ratio"""
        returns = pd.Series([t['profit_pct'] for t in self.trades])
        if len(returns) < 2:
            return 0
        return np.sqrt(252) * returns.mean() / returns.std()
    
    def plot_results(self):
        """Plot backtest results"""
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                           vertical_spacing=0.05,
                           subplot_titles=('Price', 'Equity Curve', 'Drawdown'))
        
        # Price chart
        fig.add_trace(go.Candlestick(
            x=self.data.index,
            open=self.data['open'],
            high=self.data['high'],
            low=self.data['low'],
            close=self.data['close'],
            name='Price'
        ), row=1, col=1)
        
        # Equity curve
        fig.add_trace(go.Scatter(
            x=self.data.index,
            y=self.equity_curve,
            name='Equity'
        ), row=2, col=1)
        
        # Drawdown
        peak = pd.Series(self.equity_curve).expanding(min_periods=1).max()
        drawdown = (peak - self.equity_curve) / peak
        fig.add_trace(go.Scatter(
            x=self.data.index,
            y=drawdown,
            name='Drawdown',
            fill='tozeroy'
        ), row=3, col=1)
        
        # Update layout
        fig.update_layout(
            title='Backtest Results',
            xaxis_title='Date',
            yaxis_title='Price',
            height=900
        )
        
        return fig 