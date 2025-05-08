import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Union, Dict, List, Tuple
from scipy import stats
from sklearn.preprocessing import RobustScaler

class VolatilityScaler:
    def __init__(self, lookback_period: int = 14):
        self.lookback_period = lookback_period
        self.volatility_thresholds = {
            'low': 0.5,    # Below this is considered low volatility
            'medium': 1.0, # Between low and high
            'high': 2.0    # Above this is considered high volatility
        }
        self.robust_scaler = RobustScaler()
        self.volatility_regimes = {
            'low': {'scale_factor': 1.2, 'risk_multiplier': 0.8},
            'medium': {'scale_factor': 1.0, 'risk_multiplier': 1.0},
            'high': {'scale_factor': 0.7, 'risk_multiplier': 1.2}
        }

    def calculate_atr(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Average True Range with dynamic period"""
        # Calculate standard ATR
        atr = ta.atr(df['high'], df['low'], df['close'], length=self.lookback_period)
        
        # Calculate volatility of volatility
        atr_std = atr.rolling(window=self.lookback_period).std()
        
        # Adjust ATR based on volatility of volatility
        adjusted_atr = atr * (1 + atr_std / atr.mean())
        
        return adjusted_atr

    def get_volatility_regime(self, atr: pd.Series) -> Tuple[str, Dict[str, float]]:
        """Enhanced volatility regime detection with statistical analysis"""
        current_atr = atr.iloc[-1]
        avg_atr = atr.mean()
        std_atr = atr.std()
        
        # Calculate volatility ratio
        ratio = current_atr / avg_atr
        
        # Calculate statistical measures
        skewness = stats.skew(atr.dropna())
        kurtosis = stats.kurtosis(atr.dropna())
        
        # Determine regime based on multiple factors
        if ratio < self.volatility_thresholds['low']:
            regime = 'low'
        elif ratio < self.volatility_thresholds['medium']:
            regime = 'medium'
        else:
            regime = 'high'
        
        # Adjust regime based on statistical measures
        if abs(skewness) > 1.0:  # Significant skewness
            if skewness > 0:  # Right-skewed (more high volatility)
                regime = 'high'
            else:  # Left-skewed (more low volatility)
                regime = 'low'
        
        if kurtosis > 3.0:  # Heavy tails
            regime = 'high'
        
        # Get regime parameters
        regime_params = self.volatility_regimes[regime].copy()
        
        # Add statistical measures
        regime_params.update({
            'volatility_ratio': ratio,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'std_atr': std_atr
        })
        
        return regime, regime_params

    def scale_data(self, df: pd.DataFrame, pair: str = None) -> pd.DataFrame:
        """Enhanced data scaling with regime-specific adjustments"""
        # Calculate ATR
        atr = self.calculate_atr(df)
        
        # Get volatility regime and parameters
        regime, regime_params = self.get_volatility_regime(atr)
        
        # Scale the data
        scaled_df = df.copy()
        
        # Scale price data
        for col in ['open', 'high', 'low', 'close']:
            if col in scaled_df.columns:
                # Use robust scaling for price data
                scaled_df[col] = self.robust_scaler.fit_transform(
                    scaled_df[col].values.reshape(-1, 1)
                ).flatten()
                
                # Apply regime-specific scaling
                scaled_df[col] = scaled_df[col] * regime_params['scale_factor']
        
        # Scale volume
        if 'volume' in scaled_df.columns:
            scaled_df['volume'] = scaled_df['volume'] / scaled_df['volume'].mean()
        
        # Calculate and scale technical indicators
        scaled_df = self._scale_technical_indicators(scaled_df, regime_params)
        
        return scaled_df.fillna(0)

    def _scale_technical_indicators(self, df: pd.DataFrame, regime_params: Dict[str, float]) -> pd.DataFrame:
        """Scale technical indicators based on volatility regime"""
        # RSI
        if 'rsi' in df.columns:
            df['rsi'] = (df['rsi'] - 50) / 50  # Normalize to [-1, 1]
        
        # MACD
        for col in ['macd', 'macd_signal', 'macd_hist']:
            if col in df.columns:
                df[col] = df[col] / df[col].std()
        
        # Bollinger Bands
        for col in ['bb_upper', 'bb_middle', 'bb_lower']:
            if col in df.columns:
                df[col] = (df[col] - df['close']) / df['close'].std()
        
        # ATR
        if 'atr' in df.columns:
            df['atr'] = df['atr'] / df['atr'].mean()
        
        # ADX
        if 'adx' in df.columns:
            df['adx'] = df['adx'] / 100  # Normalize to [0, 1]
        
        # CCI
        if 'cci' in df.columns:
            df['cci'] = df['cci'] / 100  # Normalize to reasonable range
        
        # Stochastic
        for col in ['stoch_k', 'stoch_d']:
            if col in df.columns:
                df[col] = df[col] / 100  # Normalize to [0, 1]
        
        # Apply regime-specific scaling
        for col in df.columns:
            if col not in ['time', 'open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col] * regime_params['scale_factor']
        
        return df

    def scale_features(self, features: Union[pd.DataFrame, np.ndarray], 
                      pair: str = None) -> Union[pd.DataFrame, np.ndarray]:
        """Scale feature data for model input with enhanced preprocessing"""
        if isinstance(features, pd.DataFrame):
            return self.scale_data(features, pair)
        elif isinstance(features, np.ndarray):
            # Convert to DataFrame, scale, and convert back
            df = pd.DataFrame(features)
            scaled_df = self.scale_data(df, pair)
            return scaled_df.values
        else:
            raise TypeError("Features must be either pandas DataFrame or numpy array")

    def inverse_scale(self, scaled_data: Union[pd.DataFrame, np.ndarray],
                     original_data: pd.DataFrame) -> Union[pd.DataFrame, np.ndarray]:
        """Convert scaled data back to original scale with regime awareness"""
        atr = self.calculate_atr(original_data)
        regime, regime_params = self.get_volatility_regime(atr)
        
        if isinstance(scaled_data, pd.DataFrame):
            unscaled = scaled_data.copy()
            for col in ['open', 'high', 'low', 'close']:
                if col in unscaled.columns:
                    # Inverse robust scaling
                    unscaled[col] = self.robust_scaler.inverse_transform(
                        (scaled_data[col] / regime_params['scale_factor']).values.reshape(-1, 1)
                    ).flatten()
            return unscaled
        elif isinstance(scaled_data, np.ndarray):
            return self.robust_scaler.inverse_transform(
                (scaled_data / regime_params['scale_factor']).reshape(-1, 1)
            ).flatten()
        else:
            raise TypeError("Scaled data must be either pandas DataFrame or numpy array")

    def get_volatility_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Get detailed volatility metrics with statistical analysis"""
        atr = self.calculate_atr(df)
        regime, regime_params = self.get_volatility_regime(atr)
        
        # Calculate additional metrics
        returns = df['close'].pct_change().dropna()
        realized_vol = returns.std() * np.sqrt(252)  # Annualized volatility
        
        return {
            'current_atr': atr.iloc[-1],
            'average_atr': atr.mean(),
            'volatility_regime': regime,
            'volatility_ratio': regime_params['volatility_ratio'],
            'skewness': regime_params['skewness'],
            'kurtosis': regime_params['kurtosis'],
            'realized_volatility': realized_vol,
            'max_atr': atr.max(),
            'min_atr': atr.min(),
            'risk_multiplier': regime_params['risk_multiplier']
        } 