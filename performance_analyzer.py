import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
import json
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import MetaTrader5 as mt5

@dataclass
class Trade:
    ticket: int
    symbol: str
    type: str
    volume: float
    open_price: float
    close_price: float
    stop_loss: float
    take_profit: float
    profit: float
    swap: float
    open_time: datetime
    close_time: datetime
    comment: str

class PerformanceAnalyzer:
    def __init__(self):
        # Initialize MT5
        if not mt5.initialize():
            logging.error("Failed to initialize MT5")
            raise Exception("MT5 initialization failed")
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('performance_analyzer.log'),
                logging.StreamHandler()
            ]
        )
        
        # Load trade history if exists
        self._load_trade_history()

    def _load_trade_history(self):
        """Load trade history from file"""
        try:
            with open('trade_history.json', 'r') as f:
                self.trade_history = json.load(f)
        except FileNotFoundError:
            self.trade_history = []

    def _save_trade_history(self):
        """Save trade history to file"""
        with open('trade_history.json', 'w') as f:
            json.dump(self.trade_history, f)

    def _calculate_equity_curve(self) -> List[float]:
        """Calculate equity curve"""
        try:
            if not self.trade_history:
                return []
            
            # Sort trades by open time
            trades = sorted(self.trade_history, key=lambda x: x['open_time'])
            
            # Calculate cumulative equity
            equity = [0.0]
            for trade in trades:
                equity.append(equity[-1] + trade['profit'])
            
            return equity[1:]  # Remove initial 0
            
        except Exception as e:
            logging.error(f"Error calculating equity curve: {str(e)}")
            return []

    def _calculate_drawdown(self, equity_curve: List[float]) -> Tuple[float, float]:
        """Calculate maximum drawdown and current drawdown"""
        try:
            if not equity_curve:
                return 0.0, 0.0
            
            # Calculate drawdowns
            peak = equity_curve[0]
            max_drawdown = 0.0
            current_drawdown = 0.0
            
            for equity in equity_curve:
                if equity > peak:
                    peak = equity
                else:
                    drawdown = (peak - equity) / peak * 100
                    max_drawdown = max(max_drawdown, drawdown)
                    current_drawdown = drawdown
            
            return max_drawdown, current_drawdown
            
        except Exception as e:
            logging.error(f"Error calculating drawdown: {str(e)}")
            return 0.0, 0.0

    def _calculate_win_rate(self) -> float:
        """Calculate win rate"""
        try:
            if not self.trade_history:
                return 0.0
            
            winning_trades = sum(1 for trade in self.trade_history if trade['profit'] > 0)
            total_trades = len(self.trade_history)
            
            return winning_trades / total_trades if total_trades > 0 else 0.0
            
        except Exception as e:
            logging.error(f"Error calculating win rate: {str(e)}")
            return 0.0

    def _calculate_profit_factor(self) -> float:
        """Calculate profit factor"""
        try:
            if not self.trade_history:
                return 0.0
            
            gross_profit = sum(trade['profit'] for trade in self.trade_history if trade['profit'] > 0)
            gross_loss = abs(sum(trade['profit'] for trade in self.trade_history if trade['profit'] < 0))
            
            return gross_profit / gross_loss if gross_loss != 0 else float('inf')
            
        except Exception as e:
            logging.error(f"Error calculating profit factor: {str(e)}")
            return 0.0

    def _calculate_sharpe_ratio(self, equity_curve: List[float]) -> float:
        """Calculate Sharpe ratio"""
        try:
            if not equity_curve:
                return 0.0
            
            # Calculate returns
            returns = np.diff(equity_curve) / equity_curve[:-1]
            
            # Calculate annualized Sharpe ratio
            if len(returns) > 0:
                sharpe = np.sqrt(252) * (np.mean(returns) / np.std(returns))
                return sharpe
            return 0.0
            
        except Exception as e:
            logging.error(f"Error calculating Sharpe ratio: {str(e)}")
            return 0.0

    def _calculate_sortino_ratio(self, equity_curve: List[float]) -> float:
        """Calculate Sortino ratio"""
        try:
            if not equity_curve:
                return 0.0
            
            # Calculate returns
            returns = np.diff(equity_curve) / equity_curve[:-1]
            
            # Calculate downside returns
            downside_returns = returns[returns < 0]
            
            # Calculate annualized Sortino ratio
            if len(returns) > 0 and len(downside_returns) > 0:
                sortino = np.sqrt(252) * (np.mean(returns) / np.std(downside_returns))
                return sortino
            return 0.0
            
        except Exception as e:
            logging.error(f"Error calculating Sortino ratio: {str(e)}")
            return 0.0

    def _calculate_trade_statistics(self) -> Dict:
        """Calculate trade statistics"""
        try:
            if not self.trade_history:
                return {}
            
            # Calculate basic statistics
            total_trades = len(self.trade_history)
            winning_trades = sum(1 for trade in self.trade_history if trade['profit'] > 0)
            losing_trades = sum(1 for trade in self.trade_history if trade['profit'] < 0)
            
            # Calculate profit statistics
            profits = [trade['profit'] for trade in self.trade_history if trade['profit'] > 0]
            losses = [trade['profit'] for trade in self.trade_history if trade['profit'] < 0]
            
            avg_profit = np.mean(profits) if profits else 0.0
            avg_loss = np.mean(losses) if losses else 0.0
            max_profit = max(profits) if profits else 0.0
            max_loss = min(losses) if losses else 0.0
            
            # Calculate trade duration statistics
            durations = []
            for trade in self.trade_history:
                open_time = datetime.fromisoformat(trade['open_time'])
                close_time = datetime.fromisoformat(trade['close_time'])
                duration = (close_time - open_time).total_seconds() / 3600  # in hours
                durations.append(duration)
            
            avg_duration = np.mean(durations) if durations else 0.0
            
            return {
                "total_trades": total_trades,
                "winning_trades": winning_trades,
                "losing_trades": losing_trades,
                "avg_profit": avg_profit,
                "avg_loss": avg_loss,
                "max_profit": max_profit,
                "max_loss": max_loss,
                "avg_duration": avg_duration
            }
            
        except Exception as e:
            logging.error(f"Error calculating trade statistics: {str(e)}")
            return {}

    def _calculate_symbol_performance(self) -> Dict[str, Dict]:
        """Calculate performance metrics by symbol"""
        try:
            if not self.trade_history:
                return {}
            
            symbol_performance = {}
            
            for trade in self.trade_history:
                symbol = trade['symbol']
                if symbol not in symbol_performance:
                    symbol_performance[symbol] = {
                        "total_trades": 0,
                        "winning_trades": 0,
                        "losing_trades": 0,
                        "total_profit": 0.0,
                        "total_loss": 0.0
                    }
                
                perf = symbol_performance[symbol]
                perf["total_trades"] += 1
                
                if trade['profit'] > 0:
                    perf["winning_trades"] += 1
                    perf["total_profit"] += trade['profit']
                else:
                    perf["losing_trades"] += 1
                    perf["total_loss"] += trade['profit']
            
            # Calculate additional metrics
            for symbol, perf in symbol_performance.items():
                perf["win_rate"] = perf["winning_trades"] / perf["total_trades"]
                perf["profit_factor"] = abs(perf["total_profit"] / perf["total_loss"]) if perf["total_loss"] != 0 else float('inf')
                perf["net_profit"] = perf["total_profit"] + perf["total_loss"]
            
            return symbol_performance
            
        except Exception as e:
            logging.error(f"Error calculating symbol performance: {str(e)}")
            return {}

    def analyze_performance(self) -> Dict:
        """Analyze trading performance"""
        try:
            # Calculate equity curve
            equity_curve = self._calculate_equity_curve()
            
            # Calculate drawdowns
            max_drawdown, current_drawdown = self._calculate_drawdown(equity_curve)
            
            # Calculate performance metrics
            win_rate = self._calculate_win_rate()
            profit_factor = self._calculate_profit_factor()
            sharpe_ratio = self._calculate_sharpe_ratio(equity_curve)
            sortino_ratio = self._calculate_sortino_ratio(equity_curve)
            
            # Calculate trade statistics
            trade_stats = self._calculate_trade_statistics()
            
            # Calculate symbol performance
            symbol_performance = self._calculate_symbol_performance()
            
            return {
                "timestamp": datetime.now().isoformat(),
                "equity_curve": equity_curve,
                "max_drawdown": max_drawdown,
                "current_drawdown": current_drawdown,
                "win_rate": win_rate,
                "profit_factor": profit_factor,
                "sharpe_ratio": sharpe_ratio,
                "sortino_ratio": sortino_ratio,
                "trade_statistics": trade_stats,
                "symbol_performance": symbol_performance
            }
            
        except Exception as e:
            logging.error(f"Error analyzing performance: {str(e)}")
            return {}

    def plot_performance(self, save_path: str = "performance_analysis.png"):
        """Plot performance analysis"""
        try:
            # Get performance analysis
            analysis = self.analyze_performance()
            if not analysis:
                return
            
            # Create figure with subplots
            fig = plt.figure(figsize=(15, 10))
            
            # Plot equity curve
            ax1 = plt.subplot(2, 2, 1)
            equity_curve = analysis['equity_curve']
            ax1.plot(equity_curve)
            ax1.set_title('Equity Curve')
            ax1.set_xlabel('Trade Number')
            ax1.set_ylabel('Equity')
            ax1.grid(True)
            
            # Plot drawdown
            ax2 = plt.subplot(2, 2, 2)
            drawdown = [(max(equity_curve[:i+1]) - equity) / max(equity_curve[:i+1]) * 100 
                       for i, equity in enumerate(equity_curve)]
            ax2.fill_between(range(len(drawdown)), drawdown, 0, color='red', alpha=0.3)
            ax2.set_title('Drawdown')
            ax2.set_xlabel('Trade Number')
            ax2.set_ylabel('Drawdown %')
            ax2.grid(True)
            
            # Plot symbol performance
            ax3 = plt.subplot(2, 2, 3)
            symbol_perf = analysis['symbol_performance']
            symbols = list(symbol_perf.keys())
            profits = [perf['net_profit'] for perf in symbol_perf.values()]
            ax3.bar(symbols, profits)
            ax3.set_title('Profit by Symbol')
            ax3.set_xlabel('Symbol')
            ax3.set_ylabel('Net Profit')
            plt.xticks(rotation=45)
            ax3.grid(True)
            
            # Plot win rate by symbol
            ax4 = plt.subplot(2, 2, 4)
            win_rates = [perf['win_rate'] * 100 for perf in symbol_perf.values()]
            ax4.bar(symbols, win_rates)
            ax4.set_title('Win Rate by Symbol')
            ax4.set_xlabel('Symbol')
            ax4.set_ylabel('Win Rate %')
            plt.xticks(rotation=45)
            ax4.grid(True)
            
            # Adjust layout and save
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
            
        except Exception as e:
            logging.error(f"Error plotting performance: {str(e)}")

    def add_trade(self, trade: Trade):
        """Add a new trade to history"""
        try:
            trade_dict = {
                "ticket": trade.ticket,
                "symbol": trade.symbol,
                "type": trade.type,
                "volume": trade.volume,
                "open_price": trade.open_price,
                "close_price": trade.close_price,
                "stop_loss": trade.stop_loss,
                "take_profit": trade.take_profit,
                "profit": trade.profit,
                "swap": trade.swap,
                "open_time": trade.open_time.isoformat(),
                "close_time": trade.close_time.isoformat(),
                "comment": trade.comment
            }
            
            self.trade_history.append(trade_dict)
            self._save_trade_history()
            
        except Exception as e:
            logging.error(f"Error adding trade: {str(e)}")

    def get_trade_history(self) -> List[Dict]:
        """Get trade history"""
        return self.trade_history

    def __del__(self):
        """Cleanup when object is destroyed"""
        mt5.shutdown()

def main():
    # Example usage
    analyzer = PerformanceAnalyzer()
    
    # Analyze performance
    analysis = analyzer.analyze_performance()
    print("Performance Analysis:")
    print(json.dumps(analysis, indent=2))
    
    # Plot performance
    analyzer.plot_performance()

if __name__ == "__main__":
    main() 