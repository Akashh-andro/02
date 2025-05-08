class TradingStrategy:
    def __init__(self):
        self.name = "Base Strategy"
    
    def generate_signal(self, data, indicators, params):
        """Generate trading signal (1 for buy, -1 for sell, 0 for hold)"""
        raise NotImplementedError

class MACDStrategy(TradingStrategy):
    def __init__(self):
        super().__init__()
        self.name = "MACD Strategy"
    
    def generate_signal(self, data, indicators, params):
        if indicators['macd'] > indicators['macd_signal'] and indicators['macd_hist'] > 0:
            return 1  # Buy signal
        elif indicators['macd'] < indicators['macd_signal'] and indicators['macd_hist'] < 0:
            return -1  # Sell signal
        return 0

class RSIStrategy(TradingStrategy):
    def __init__(self):
        super().__init__()
        self.name = "RSI Strategy"
    
    def generate_signal(self, data, indicators, params):
        rsi = indicators['rsi']
        if rsi < 30:  # Oversold
            return 1  # Buy signal
        elif rsi > 70:  # Overbought
            return -1  # Sell signal
        return 0

class BollingerBandsStrategy(TradingStrategy):
    def __init__(self):
        super().__init__()
        self.name = "Bollinger Bands Strategy"
    
    def generate_signal(self, data, indicators, params):
        price = data['close']
        if price < indicators['bb_lower']:
            return 1  # Buy signal
        elif price > indicators['bb_upper']:
            return -1  # Sell signal
        return 0

class MovingAverageCrossoverStrategy(TradingStrategy):
    def __init__(self):
        super().__init__()
        self.name = "Moving Average Crossover Strategy"
    
    def generate_signal(self, data, indicators, params):
        if indicators['sma_20'] > indicators['sma_50']:
            return 1  # Buy signal
        elif indicators['sma_20'] < indicators['sma_50']:
            return -1  # Sell signal
        return 0

class ADXStrategy(TradingStrategy):
    def __init__(self):
        super().__init__()
        self.name = "ADX Strategy"
    
    def generate_signal(self, data, indicators, params):
        if indicators['adx'] > 25:  # Strong trend
            if indicators['adx_pos'] > indicators['adx_neg']:
                return 1  # Buy signal
            elif indicators['adx_pos'] < indicators['adx_neg']:
                return -1  # Sell signal
        return 0

class CombinedStrategy(TradingStrategy):
    def __init__(self):
        super().__init__()
        self.name = "Combined Strategy"
        self.strategies = [
            MACDStrategy(),
            RSIStrategy(),
            BollingerBandsStrategy(),
            MovingAverageCrossoverStrategy(),
            ADXStrategy()
        ]
    
    def generate_signal(self, data, indicators, params):
        signals = [strategy.generate_signal(data, indicators, params) 
                  for strategy in self.strategies]
        
        # Count buy and sell signals
        buy_signals = signals.count(1)
        sell_signals = signals.count(-1)
        
        # Generate final signal based on majority
        if buy_signals > sell_signals and buy_signals >= 3:
            return 1  # Buy signal
        elif sell_signals > buy_signals and sell_signals >= 3:
            return -1  # Sell signal
        return 0

class CustomStrategy(TradingStrategy):
    def __init__(self):
        super().__init__()
        self.name = "Custom Strategy"
    
    def generate_signal(self, data, indicators, params):
        # Combine multiple indicators with custom logic
        signal = 0
        
        # Trend following component
        if (indicators['sma_20'] > indicators['sma_50'] and 
            indicators['macd'] > indicators['macd_signal']):
            signal += 1
        elif (indicators['sma_20'] < indicators['sma_50'] and 
              indicators['macd'] < indicators['macd_signal']):
            signal -= 1
        
        # Mean reversion component
        if (indicators['rsi'] < 30 and 
            data['close'] < indicators['bb_lower']):
            signal += 1
        elif (indicators['rsi'] > 70 and 
              data['close'] > indicators['bb_upper']):
            signal -= 1
        
        # Volatility component
        if indicators['adx'] > 25:  # Strong trend
            if indicators['adx_pos'] > indicators['adx_neg']:
                signal += 1
            elif indicators['adx_pos'] < indicators['adx_neg']:
                signal -= 1
        
        # Volume confirmation
        if indicators['obv'] > indicators['volume_sma']:
            signal += 0.5
        elif indicators['obv'] < indicators['volume_sma']:
            signal -= 0.5
        
        # Generate final signal
        if signal >= 2:
            return 1  # Strong buy signal
        elif signal <= -2:
            return -1  # Strong sell signal
        return 0 