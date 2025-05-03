import numpy as np
import pandas as pd
from typing import Union, List, Optional, Dict, Any

class TechnicalIndicators:
    """
    Class for calculating technical indicators for financial time series data.
    """
    
    @staticmethod
    def add_all_indicators(df: pd.DataFrame, price_col: str = 'close', volume_col: Optional[str] = 'volume') -> pd.DataFrame:
        """
        Add all available technical indicators to a DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
            price_col: Column name for price data
            volume_col: Column name for volume data (if available)
            
        Returns:
            DataFrame with additional technical indicator columns
        """
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Add moving averages
        for period in [5, 10, 20, 50, 200]:
            df = TechnicalIndicators.add_sma(df, price_col, period)
            
        # Add EMA
        for period in [5, 12, 26]:
            df = TechnicalIndicators.add_ema(df, price_col, period)
            
        # Add RSI
        df = TechnicalIndicators.add_rsi(df, price_col, 14)
        
        # Add MACD
        df = TechnicalIndicators.add_macd(df, price_col)
        
        # Add Bollinger Bands
        df = TechnicalIndicators.add_bollinger_bands(df, price_col)
        
        # Add ATR
        df = TechnicalIndicators.add_atr(df, 14)
        
        # Add Stochastic Oscillator
        df = TechnicalIndicators.add_stochastic_oscillator(df)
        
        # Add momentum indicators
        for period in [5, 10, 20]:
            df = TechnicalIndicators.add_momentum(df, price_col, period)
            
        # Add rate of change
        for period in [5, 10, 20]:
            df = TechnicalIndicators.add_roc(df, price_col, period)
        
        # Add volume-based indicators if volume data is available
        if volume_col and volume_col in df.columns:
            # Add OBV
            df = TechnicalIndicators.add_obv(df, price_col, volume_col)
            
            # Add Volume MA
            for period in [5, 10, 20]:
                df = TechnicalIndicators.add_volume_ma(df, volume_col, period)
        
        # Add volatility measures
        df = TechnicalIndicators.add_historical_volatility(df, price_col, 20)
        
        # Add price channels
        df = TechnicalIndicators.add_price_channels(df, 20)
        
        # Add Z-Score
        df = TechnicalIndicators.add_zscore(df, price_col, 20)
        
        # Drop rows with NaN values (due to indicators needing historical data)
        # df = df.dropna().reset_index(drop=True)
        
        return df
    
    @staticmethod
    def add_sma(df: pd.DataFrame, price_col: str = 'close', period: int = 20) -> pd.DataFrame:
        """Add Simple Moving Average to DataFrame"""
        df[f'sma_{period}'] = df[price_col].rolling(window=period).mean()
        return df
    
    @staticmethod
    def add_ema(df: pd.DataFrame, price_col: str = 'close', period: int = 20) -> pd.DataFrame:
        """Add Exponential Moving Average to DataFrame"""
        df[f'ema_{period}'] = df[price_col].ewm(span=period, adjust=False).mean()
        return df
    
    @staticmethod
    def add_rsi(df: pd.DataFrame, price_col: str = 'close', period: int = 14) -> pd.DataFrame:
        """Add Relative Strength Index to DataFrame"""
        delta = df[price_col].diff()
        gain = delta.mask(delta < 0, 0)
        loss = -delta.mask(delta > 0, 0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        # Handle division by zero
        rs = pd.Series(np.where(avg_loss == 0, 100, avg_gain / avg_loss), index=df.index)
        df['rsi'] = 100 - (100 / (1 + rs))
        return df
    
    @staticmethod
    def add_macd(df: pd.DataFrame, price_col: str = 'close', 
                 fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> pd.DataFrame:
        """Add MACD to DataFrame"""
        # Calculate MACD components
        fast_ema = df[price_col].ewm(span=fast_period, adjust=False).mean()
        slow_ema = df[price_col].ewm(span=slow_period, adjust=False).mean()
        
        # MACD line
        df['macd_line'] = fast_ema - slow_ema
        
        # Signal line
        df['macd_signal'] = df['macd_line'].ewm(span=signal_period, adjust=False).mean()
        
        # Histogram
        df['macd_histogram'] = df['macd_line'] - df['macd_signal']
        
        return df
    
    @staticmethod
    def add_bollinger_bands(df: pd.DataFrame, price_col: str = 'close', 
                           period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
        """Add Bollinger Bands to DataFrame"""
        # Calculate SMA and standard deviation
        df[f'sma_{period}'] = df[price_col].rolling(window=period).mean()
        df[f'std_{period}'] = df[price_col].rolling(window=period).std()
        
        # Calculate upper and lower bands
        df['bb_upper'] = df[f'sma_{period}'] + (df[f'std_{period}'] * std_dev)
        df['bb_lower'] = df[f'sma_{period}'] - (df[f'std_{period}'] * std_dev)
        
        # Calculate %B (position within bands)
        df['bb_pct_b'] = (df[price_col] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Calculate bandwidth
        df['bb_bandwidth'] = (df['bb_upper'] - df['bb_lower']) / df[f'sma_{period}']
        
        return df
    
    @staticmethod
    def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add Average True Range to DataFrame"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Calculate true range
        tr1 = abs(high - low)
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        df['atr'] = tr.rolling(window=period).mean()
        
        return df
    
    @staticmethod
    def add_stochastic_oscillator(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
        """Add Stochastic Oscillator to DataFrame"""
        # Get highest high and lowest low
        high_roll = df['high'].rolling(window=k_period).max()
        low_roll = df['low'].rolling(window=k_period).min()
        
        # Fast %K
        df['stoch_k'] = 100 * ((df['close'] - low_roll) / (high_roll - low_roll))
        
        # Slow %D (moving average of %K)
        df['stoch_d'] = df['stoch_k'].rolling(window=d_period).mean()
        
        return df
    
    @staticmethod
    def add_momentum(df: pd.DataFrame, price_col: str = 'close', period: int = 10) -> pd.DataFrame:
        """Add Momentum indicator to DataFrame"""
        df[f'momentum_{period}'] = df[price_col] - df[price_col].shift(period)
        return df
    
    @staticmethod
    def add_roc(df: pd.DataFrame, price_col: str = 'close', period: int = 10) -> pd.DataFrame:
        """Add Rate of Change indicator to DataFrame"""
        df[f'roc_{period}'] = ((df[price_col] - df[price_col].shift(period)) / 
                              df[price_col].shift(period)) * 100
        return df
    
    @staticmethod
    def add_obv(df: pd.DataFrame, price_col: str = 'close', volume_col: str = 'volume') -> pd.DataFrame:
        """Add On-Balance Volume to DataFrame"""
        # Calculate price change direction
        price_change = df[price_col].diff()
        
        # Initialize OBV
        obv = [0]
        
        # Calculate OBV
        for i in range(1, len(df)):
            if price_change.iloc[i] > 0:
                obv.append(obv[-1] + df[volume_col].iloc[i])
            elif price_change.iloc[i] < 0:
                obv.append(obv[-1] - df[volume_col].iloc[i])
            else:
                obv.append(obv[-1])
        
        df['obv'] = obv
        return df
    
    @staticmethod
    def add_volume_ma(df: pd.DataFrame, volume_col: str = 'volume', period: int = 20) -> pd.DataFrame:
        """Add Volume Moving Average to DataFrame"""
        df[f'volume_ma_{period}'] = df[volume_col].rolling(window=period).mean()
        return df
    
    @staticmethod
    def add_historical_volatility(df: pd.DataFrame, price_col: str = 'close', period: int = 20) -> pd.DataFrame:
        """Add Historical Volatility to DataFrame"""
        # Calculate log returns
        log_returns = np.log(df[price_col] / df[price_col].shift(1))
        
        # Calculate volatility (standard deviation of log returns)
        df[f'volatility_{period}'] = log_returns.rolling(window=period).std() * np.sqrt(252)  # Annualized
        
        return df
    
    @staticmethod
    def add_price_channels(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Add Price Channels to DataFrame"""
        df[f'channel_high_{period}'] = df['high'].rolling(window=period).max()
        df[f'channel_low_{period}'] = df['low'].rolling(window=period).min()
        df[f'channel_mid_{period}'] = (df[f'channel_high_{period}'] + df[f'channel_low_{period}']) / 2
        
        return df
    
    @staticmethod
    def add_zscore(df: pd.DataFrame, price_col: str = 'close', period: int = 20) -> pd.DataFrame:
        """Add Z-Score to DataFrame"""
        mean = df[price_col].rolling(window=period).mean()
        std = df[price_col].rolling(window=period).std()
        df[f'zscore_{period}'] = (df[price_col] - mean) / std
        
        return df

# Example usage
if __name__ == "__main__":
    # Create sample data
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    np.random.seed(42)
    
    # Generate random price data
    data = {
        'open': np.random.normal(100, 5, size=100),
        'high': np.random.normal(102, 5, size=100),
        'low': np.random.normal(98, 5, size=100),
        'close': np.random.normal(101, 5, size=100),
        'volume': np.random.normal(1000000, 200000, size=100)
    }
    
    # Ensure high is highest, low is lowest
    for i in range(len(data['high'])):
        values = [data['open'][i], data['close'][i], data['high'][i], data['low'][i]]
        data['high'][i] = max(values)
        data['low'][i] = min(values)
    
    df = pd.DataFrame(data, index=dates)
    
    # Add all technical indicators
    df_with_indicators = TechnicalIndicators.add_all_indicators(df)
    
    # Print the first few rows
    print(df_with_indicators.head())
    
    # Print the list of columns
    print("\nList of all indicators added:")
    original_cols = ['open', 'high', 'low', 'close', 'volume']
    indicator_cols = [col for col in df_with_indicators.columns if col not in original_cols]
    print(indicator_cols) 