import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class RiskManager:
    def __init__(self, initial_capital=10000):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.risk_per_trade = 0.02  # 2% risk per trade
        self.max_daily_risk = 0.05  # 5% max daily risk
        self.max_drawdown = 0.15    # 15% max drawdown
        self.max_correlation = 0.7   # Maximum correlation between positions
        self.max_positions = 5       # Maximum number of concurrent positions
        self.position_sizes = {}     # Current position sizes
        self.daily_pnl = 0          # Daily profit/loss
        self.trade_history = []      # History of trades
        self.risk_metrics = {
            'sharpe_ratio': 0,
            'sortino_ratio': 0,
            'max_drawdown': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'average_win': 0,
            'average_loss': 0,
            'largest_win': 0,
            'largest_loss': 0,
            'recovery_factor': 0
        }
    
    def calculate_position_size(self, symbol, entry_price, stop_loss):
        """Calculate position size based on risk parameters"""
        try:
            # Calculate risk amount
            risk_amount = self.current_capital * self.risk_per_trade
            
            # Calculate stop loss distance
            stop_distance = abs(entry_price - stop_loss)
            if stop_distance == 0:
                return 0
            
            # Calculate position size
            position_size = risk_amount / stop_distance
            
            # Check if position size exceeds daily risk limit
            if self.daily_pnl + (position_size * stop_distance) > self.current_capital * self.max_daily_risk:
                position_size = (self.current_capital * self.max_daily_risk - self.daily_pnl) / stop_distance
            
            return position_size
        except Exception as e:
            print(f"Error calculating position size: {str(e)}")
            return 0
    
    def check_correlation(self, new_symbol, current_positions):
        """Check correlation between new position and existing positions"""
        if not current_positions:
            return True
        
        # Calculate correlation matrix
        prices = pd.DataFrame()
        for symbol in current_positions + [new_symbol]:
            # Get historical prices (implement your data fetching logic here)
            prices[symbol] = self._get_historical_prices(symbol)
        
        correlation_matrix = prices.corr()
        
        # Check if any correlation exceeds the limit
        for symbol in current_positions:
            if abs(correlation_matrix.loc[new_symbol, symbol]) > self.max_correlation:
                return False
        
        return True
    
    def update_risk_metrics(self, trade_result):
        """Update risk metrics based on trade result"""
        self.trade_history.append(trade_result)
        
        # Calculate basic metrics
        winning_trades = [t for t in self.trade_history if t['profit'] > 0]
        losing_trades = [t for t in self.trade_history if t['profit'] <= 0]
        
        self.risk_metrics['win_rate'] = len(winning_trades) / len(self.trade_history) if self.trade_history else 0
        self.risk_metrics['average_win'] = np.mean([t['profit'] for t in winning_trades]) if winning_trades else 0
        self.risk_metrics['average_loss'] = np.mean([t['profit'] for t in losing_trades]) if losing_trades else 0
        self.risk_metrics['largest_win'] = max([t['profit'] for t in winning_trades]) if winning_trades else 0
        self.risk_metrics['largest_loss'] = min([t['profit'] for t in losing_trades]) if losing_trades else 0
        
        # Calculate returns
        returns = pd.Series([t['profit'] for t in self.trade_history])
        
        # Calculate Sharpe ratio
        if len(returns) > 1:
            self.risk_metrics['sharpe_ratio'] = np.sqrt(252) * returns.mean() / returns.std()
        
        # Calculate Sortino ratio
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 1:
            self.risk_metrics['sortino_ratio'] = np.sqrt(252) * returns.mean() / downside_returns.std()
        
        # Calculate maximum drawdown
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - rolling_max) / rolling_max
        self.risk_metrics['max_drawdown'] = drawdowns.min()
        
        # Calculate profit factor
        gross_profit = sum([t['profit'] for t in winning_trades])
        gross_loss = abs(sum([t['profit'] for t in losing_trades]))
        self.risk_metrics['profit_factor'] = gross_profit / gross_loss if gross_loss != 0 else float('inf')
        
        # Calculate recovery factor
        if self.risk_metrics['max_drawdown'] != 0:
            self.risk_metrics['recovery_factor'] = gross_profit / abs(self.risk_metrics['max_drawdown'])
    
    def check_risk_limits(self, new_position):
        """Check if new position violates any risk limits"""
        # Check maximum positions
        if len(self.position_sizes) >= self.max_positions:
            return False, "Maximum number of positions reached"
        
        # Check daily risk limit
        if self.daily_pnl + new_position['risk'] > self.current_capital * self.max_daily_risk:
            return False, "Daily risk limit exceeded"
        
        # Check maximum drawdown
        if self.risk_metrics['max_drawdown'] < -self.max_drawdown:
            return False, "Maximum drawdown exceeded"
        
        # Check correlation
        if not self.check_correlation(new_position['symbol'], list(self.position_sizes.keys())):
            return False, "Correlation limit exceeded"
        
        return True, "Risk limits satisfied"
    
    def update_daily_metrics(self):
        """Update daily risk metrics"""
        self.daily_pnl = 0
        # Reset daily metrics at the start of each day
        # Implement your daily reset logic here
    
    def _get_historical_prices(self, symbol):
        """Get historical prices for correlation calculation"""
        # Implement your data fetching logic here
        # This is a placeholder
        return pd.Series(np.random.randn(100))
    
    def get_risk_report(self):
        """Generate risk management report"""
        return {
            'current_capital': self.current_capital,
            'daily_pnl': self.daily_pnl,
            'open_positions': len(self.position_sizes),
            'risk_metrics': self.risk_metrics,
            'position_sizes': self.position_sizes,
            'risk_limits': {
                'risk_per_trade': self.risk_per_trade,
                'max_daily_risk': self.max_daily_risk,
                'max_drawdown': self.max_drawdown,
                'max_correlation': self.max_correlation,
                'max_positions': self.max_positions
            }
        } 