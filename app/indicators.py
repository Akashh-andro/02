import pandas as pd
import numpy as np
import ta
from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, VolumeWeightedAveragePrice

class TechnicalIndicators:
    def __init__(self):
        self.indicators = {}
    
    def calculate_all(self, df):
        """Calculate all technical indicators"""
        # Trend Indicators
        self._calculate_moving_averages(df)
        self._calculate_macd(df)
        self._calculate_adx(df)
        
        # Momentum Indicators
        self._calculate_rsi(df)
        self._calculate_stochastic(df)
        self._calculate_williams_r(df)
        
        # Volatility Indicators
        self._calculate_bollinger_bands(df)
        self._calculate_atr(df)
        
        # Volume Indicators
        self._calculate_volume_indicators(df)
        
        # Custom Indicators
        self._calculate_custom_indicators(df)
        
        return self.indicators
    
    def _calculate_moving_averages(self, df):
        """Calculate various moving averages"""
        # Simple Moving Averages
        self.indicators['sma_20'] = SMAIndicator(close=df['close'], window=20).sma_indicator()
        self.indicators['sma_50'] = SMAIndicator(close=df['close'], window=50).sma_indicator()
        self.indicators['sma_200'] = SMAIndicator(close=df['close'], window=200).sma_indicator()
        
        # Exponential Moving Averages
        self.indicators['ema_12'] = EMAIndicator(close=df['close'], window=12).ema_indicator()
        self.indicators['ema_26'] = EMAIndicator(close=df['close'], window=26).ema_indicator()
        self.indicators['ema_50'] = EMAIndicator(close=df['close'], window=50).ema_indicator()
        
        # VWAP
        self.indicators['vwap'] = VolumeWeightedAveragePrice(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            volume=df['volume']
        ).volume_weighted_average_price()
    
    def _calculate_macd(self, df):
        """Calculate MACD"""
        macd = MACD(
            close=df['close'],
            window_slow=26,
            window_fast=12,
            window_sign=9
        )
        self.indicators['macd'] = macd.macd()
        self.indicators['macd_signal'] = macd.macd_signal()
        self.indicators['macd_hist'] = macd.macd_diff()
    
    def _calculate_adx(self, df):
        """Calculate ADX"""
        adx = ADXIndicator(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            window=14
        )
        self.indicators['adx'] = adx.adx()
        self.indicators['adx_pos'] = adx.adx_pos()
        self.indicators['adx_neg'] = adx.adx_neg()
    
    def _calculate_rsi(self, df):
        """Calculate RSI"""
        rsi = RSIIndicator(close=df['close'], window=14)
        self.indicators['rsi'] = rsi.rsi()
    
    def _calculate_stochastic(self, df):
        """Calculate Stochastic Oscillator"""
        stoch = StochasticOscillator(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            window=14,
            smooth_window=3
        )
        self.indicators['stoch_k'] = stoch.stoch()
        self.indicators['stoch_d'] = stoch.stoch_signal()
    
    def _calculate_williams_r(self, df):
        """Calculate Williams %R"""
        williams = WilliamsRIndicator(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            lbp=14
        )
        self.indicators['williams_r'] = williams.williams_r()
    
    def _calculate_bollinger_bands(self, df):
        """Calculate Bollinger Bands"""
        bb = BollingerBands(
            close=df['close'],
            window=20,
            window_dev=2
        )
        self.indicators['bb_upper'] = bb.bollinger_hband()
        self.indicators['bb_middle'] = bb.bollinger_mavg()
        self.indicators['bb_lower'] = bb.bollinger_lband()
        self.indicators['bb_width'] = (self.indicators['bb_upper'] - self.indicators['bb_lower']) / self.indicators['bb_middle']
    
    def _calculate_atr(self, df):
        """Calculate Average True Range"""
        atr = AverageTrueRange(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            window=14
        )
        self.indicators['atr'] = atr.average_true_range()
    
    def _calculate_volume_indicators(self, df):
        """Calculate Volume-based Indicators"""
        # On Balance Volume
        obv = OnBalanceVolumeIndicator(close=df['close'], volume=df['volume'])
        self.indicators['obv'] = obv.on_balance_volume()
        
        # Volume SMA
        self.indicators['volume_sma'] = df['volume'].rolling(window=20).mean()
    
    def _calculate_custom_indicators(self, df):
        """Calculate custom indicators"""
        # Price Action
        self.indicators['price_range'] = df['high'] - df['low']
        self.indicators['price_range_ma'] = self.indicators['price_range'].rolling(window=20).mean()
        
        # Trend Strength
        self.indicators['trend_strength'] = abs(self.indicators['sma_20'] - self.indicators['sma_50']) / self.indicators['atr']
        
        # Volatility
        self.indicators['volatility'] = self.indicators['atr'] / df['close']
        
        # Momentum
        self.indicators['momentum'] = df['close'].pct_change(periods=10)
        
        # Support/Resistance Levels
        self.indicators['support'] = df['low'].rolling(window=20).min()
        self.indicators['resistance'] = df['high'].rolling(window=20).max() 