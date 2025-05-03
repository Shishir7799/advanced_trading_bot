import pandas as pd
import numpy as np
import joblib
import os
import logging
from typing import Tuple, Dict, Optional, List, Any, Union
from technical_indicators import TechnicalIndicators
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class XGBoostPredictor:
    """
    Class for making price predictions using a trained XGBoost model
    with technical indicators.
    """
    
    def __init__(self, symbol: str = "BTCUSDT", window_size: int = 5, 
                 model_path: str = "models", threshold: Optional[float] = None):
        """
        Initialize the predictor with a trained model
        
        Args:
            symbol: Trading pair symbol
            window_size: Size of price window used for prediction
            model_path: Path to saved model files
            threshold: Confidence threshold for predictions (None: use model metadata)
        """
        self.symbol = symbol
        self.window_size = window_size
        self.model_path = model_path
        self.custom_threshold = threshold
        
        # Define file paths
        self.model_file = os.path.join(model_path, f"{symbol.lower()}_xgboost.joblib")
        self.scaler_file = os.path.join(model_path, f"{symbol.lower()}_xgboost_scaler.joblib")
        self.model_metadata_file = os.path.join(model_path, f"{symbol.lower()}_xgboost_metadata.json")
        
        # Load model, scaler and metadata
        self.model = self._load_model()
        self.scaler = self._load_scaler()
        self.metadata = self._load_metadata()
        
        # Set confidence threshold based on metadata or custom value
        if self.custom_threshold is not None:
            self.threshold = self.custom_threshold
        elif self.metadata and 'suggested_threshold' in self.metadata.get('metrics', {}):
            self.threshold = self.metadata['metrics']['suggested_threshold']
        else:
            self.threshold = 0.5
            
        self.feature_names = self.metadata.get('feature_names', None) if self.metadata else None
        logger.info(f"Initialized XGBoost predictor for {symbol} with threshold {self.threshold}")
    
    def _load_model(self):
        """Load the trained XGBoost model"""
        try:
            if not os.path.exists(self.model_file):
                logger.error(f"Model file not found: {self.model_file}")
                return None
            
            model = joblib.load(self.model_file)
            logger.info(f"Loaded model from {self.model_file}")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None
    
    def _load_scaler(self):
        """Load the feature scaler"""
        try:
            if not os.path.exists(self.scaler_file):
                logger.error(f"Scaler file not found: {self.scaler_file}")
                return None
            
            scaler = joblib.load(self.scaler_file)
            logger.info(f"Loaded scaler from {self.scaler_file}")
            return scaler
        except Exception as e:
            logger.error(f"Error loading scaler: {e}")
            return None
    
    def _load_metadata(self):
        """Load model metadata"""
        try:
            if not os.path.exists(self.model_metadata_file):
                logger.warning(f"Metadata file not found: {self.model_metadata_file}")
                return None
            
            import json
            with open(self.model_metadata_file, 'r') as f:
                metadata = json.load(f)
            logger.info(f"Loaded metadata from {self.model_metadata_file}")
            return metadata
        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
            return None
    
    def is_ready(self) -> bool:
        """Check if model and scaler are loaded and ready to use"""
        return self.model is not None and self.scaler is not None
    
    def is_model_ready(self) -> bool:
        """Alias for is_ready() for compatibility with advanced_trading_bot.py"""
        return self.is_ready()
    
    def create_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Create features for prediction using the same process as during training
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Feature array and list of feature names
        """
        # Add all technical indicators
        enriched_df = TechnicalIndicators.add_all_indicators(df)
        
        # Define lookback periods for feature creation
        lookback_periods = [self.window_size]
        
        # Initialize feature names (will be populated if needed)
        feature_names = []
        
        # Get index of last row
        idx = len(enriched_df) - 1
        
        if idx < self.window_size:
            raise ValueError(f"Not enough data for window size {self.window_size}")
        
        # Basic price features from window
        price_window = df['close'].iloc[idx-self.window_size:idx].values
        feature_vector = price_window.copy()
        feature_names.append([f'price_{j}' for j in range(self.window_size)])
        
        # Add momentum features (price differences)
        price_momentum = np.diff(price_window)
        feature_vector = np.append(feature_vector, price_momentum)
        feature_names.append([f'momentum_{j}' for j in range(len(price_momentum))])
        
        # Add rolling statistics
        rolling_mean = np.mean(price_window)
        rolling_std = np.std(price_window)
        feature_vector = np.append(feature_vector, [rolling_mean, rolling_std])
        feature_names.append(['rolling_mean', 'rolling_std'])
        
        # Add Z-score of latest price
        if rolling_std != 0:
            z_score = (price_window[-1] - rolling_mean) / rolling_std
        else:
            z_score = 0
        feature_vector = np.append(feature_vector, z_score)
        feature_names.append(['z_score'])
        
        # Add relative position within price range
        price_range = np.max(price_window) - np.min(price_window)
        if price_range != 0:
            rel_position = (price_window[-1] - np.min(price_window)) / price_range
        else:
            rel_position = 0.5
        feature_vector = np.append(feature_vector, rel_position)
        feature_names.append(['rel_position'])
        
        # Add technical indicators at the current point
        current_row = enriched_df.iloc[idx]
        
        # RSI
        if not np.isnan(current_row['rsi']):
            feature_vector = np.append(feature_vector, current_row['rsi'])
            feature_names.append(['rsi'])
        
        # MACD components
        for macd_feature in ['macd_line', 'macd_signal', 'macd_histogram']:
            if macd_feature in current_row and not np.isnan(current_row[macd_feature]):
                feature_vector = np.append(feature_vector, current_row[macd_feature])
                feature_names.append([macd_feature])
        
        # Bollinger Bands metrics
        for bb_feature in ['bb_pct_b', 'bb_bandwidth']:
            if bb_feature in current_row and not np.isnan(current_row[bb_feature]):
                feature_vector = np.append(feature_vector, current_row[bb_feature])
                feature_names.append([bb_feature])
        
        # Stochastic oscillator
        for stoch_feature in ['stoch_k', 'stoch_d']:
            if stoch_feature in current_row and not np.isnan(current_row[stoch_feature]):
                feature_vector = np.append(feature_vector, current_row[stoch_feature])
                feature_names.append([stoch_feature])
        
        # ATR for volatility
        if 'atr' in current_row and not np.isnan(current_row['atr']):
            feature_vector = np.append(feature_vector, current_row['atr'])
            feature_names.append(['atr'])
        
        # Moving averages - use 5, 10, 20 periods
        for period in [5, 10, 20]:
            sma_col = f'sma_{period}'
            if sma_col in current_row and not np.isnan(current_row[sma_col]):
                # Calculate relative position to current price
                sma_value = current_row[sma_col]
                rel_to_sma = (price_window[-1] - sma_value) / sma_value
                feature_vector = np.append(feature_vector, rel_to_sma)
                feature_names.append([f'rel_to_sma_{period}'])
        
        # Add features for two important moving average crossovers
        if 'sma_5' in current_row and 'sma_10' in current_row:
            if not np.isnan(current_row['sma_5']) and not np.isnan(current_row['sma_10']):
                cross_5_10 = current_row['sma_5'] - current_row['sma_10']
                feature_vector = np.append(feature_vector, cross_5_10)
                feature_names.append(['cross_5_10'])
        
        if 'sma_10' in current_row and 'sma_20' in current_row:
            if not np.isnan(current_row['sma_10']) and not np.isnan(current_row['sma_20']):
                cross_10_20 = current_row['sma_10'] - current_row['sma_20']
                feature_vector = np.append(feature_vector, cross_10_20)
                feature_names.append(['cross_10_20'])
        
        # Volume-related features if available
        if 'volume' in df.columns:
            # Use recent volume changes
            recent_volumes = df['volume'].iloc[idx-self.window_size:idx].values
            vol_change = recent_volumes[-1] / np.mean(recent_volumes) - 1
            feature_vector = np.append(feature_vector, vol_change)
            feature_names.append(['vol_change'])
            
            # Add OBV if available
            if 'obv' in current_row and not np.isnan(current_row['obv']):
                feature_vector = np.append(feature_vector, current_row['obv'])
                feature_names.append(['obv'])
        
        # Flatten feature names list
        flat_feature_names = [item for sublist in feature_names for item in sublist]
        
        return feature_vector.reshape(1, -1), flat_feature_names
    
    def predict(self, df: pd.DataFrame) -> Tuple[int, float, Dict[str, Any]]:
        """
        Make a prediction for the given price data
        
        Args:
            df: DataFrame with price data from fetch_crypto_prices
            
        Returns:
            Tuple with:
            - Prediction (1 for UP, 0 for DOWN)
            - Confidence (probability of prediction)
            - Dictionary with additional prediction details
        """
        if not self.is_ready():
            logger.error("Model is not ready for prediction")
            return -1, 0.0, {"error": "Model not ready"}
        
        try:
            # Make sure we have sufficient data
            if len(df) < self.window_size:
                logger.error(f"Not enough price data, need at least {self.window_size} points")
                return -1, 0.0, {"error": "Insufficient price data"}
            
            # Create features for prediction
            X, feature_names = self.create_features(df)
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Get probability prediction
            pred_proba = self.model.predict_proba(X_scaled)[0]
            
            # Get raw prediction (highest probability class)
            raw_prediction = np.argmax(pred_proba)
            
            # Get confidence of prediction
            confidence = pred_proba[raw_prediction]
            
            # Apply threshold to determine final prediction
            if confidence >= self.threshold:
                prediction = raw_prediction
            else:
                # Not confident enough, don't make a prediction
                prediction = -1
                
            # Create details dictionary
            details = {
                "raw_prediction": int(raw_prediction),
                "confidence": float(confidence),
                "threshold": float(self.threshold),
                "sufficient_confidence": bool(confidence >= self.threshold),
                "class_probabilities": {
                    "DOWN": float(pred_proba[0]),
                    "UP": float(pred_proba[1])
                },
                "feature_count": len(feature_names)
            }
            
            logger.info(f"XGBoost prediction: {'UP' if raw_prediction == 1 else 'DOWN'} with {confidence:.4f} confidence")
            if confidence < self.threshold:
                logger.info(f"Confidence below threshold ({self.threshold}), no action recommended")
                
            return prediction, confidence, details
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return -1, 0.0, {"error": str(e)}

# Example usage
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Test XGBoost prediction')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='Trading pair symbol')
    parser.add_argument('--threshold', type=float, default=None, help='Custom confidence threshold')
    args = parser.parse_args()
    
    from fetch_crypto_prices import PriceFetcher
    
    # Create price fetcher and predictor
    fetcher = PriceFetcher(symbol=args.symbol)
    predictor = XGBoostPredictor(symbol=args.symbol, threshold=args.threshold)
    
    if predictor.is_ready():
        # Fetch latest prices
        df = fetcher.fetch_historical_klines(interval="1h", limit=100)
        
        if not df.empty:
            # Make prediction
            prediction, confidence, details = predictor.predict(df)
            
            # Print detailed results
            print("\n=== XGBoost Prediction Results ===")
            print(f"Prediction: {'UP' if prediction == 1 else 'DOWN' if prediction == 0 else 'NEUTRAL'}")
            print(f"Confidence: {confidence:.4f}")
            print(f"Threshold: {predictor.threshold}")
            print(f"\nClass Probabilities:")
            print(f"- UP: {details['class_probabilities']['UP']:.4f}")
            print(f"- DOWN: {details['class_probabilities']['DOWN']:.4f}")
            
            # Get current and previous price for context
            current_price = df['close'].iloc[-1]
            previous_price = df['close'].iloc[-2]
            price_change = (current_price - previous_price) / previous_price * 100
            
            print(f"\nCurrent Price: {current_price:.2f}")
            print(f"Previous Price: {previous_price:.2f}")
            print(f"Recent Change: {price_change:.2f}%")
            
            # Technical indicators
            enriched_df = TechnicalIndicators.add_all_indicators(df)
            current_row = enriched_df.iloc[-1]
            
            print("\nTechnical Indicators:")
            print(f"RSI: {current_row['rsi']:.2f}")
            print(f"MACD Line: {current_row['macd_line']:.4f}")
            print(f"MACD Signal: {current_row['macd_signal']:.4f}")
            print(f"MACD Histogram: {current_row['macd_histogram']:.4f}")
            print(f"Bollinger %B: {current_row['bb_pct_b']:.4f}")
            
            # SMA crossovers
            sma5 = current_row['sma_5']
            sma10 = current_row['sma_10']
            sma20 = current_row['sma_20']
            
            print("\nMoving Averages:")
            print(f"SMA 5: {sma5:.2f}")
            print(f"SMA 10: {sma10:.2f}")
            print(f"SMA 20: {sma20:.2f}")
            print(f"5/10 Crossover: {'Bullish' if sma5 > sma10 else 'Bearish'} ({sma5-sma10:.2f})")
            print(f"10/20 Crossover: {'Bullish' if sma10 > sma20 else 'Bearish'} ({sma10-sma20:.2f})")
        else:
            print(f"Failed to fetch price data for {args.symbol}")
    else:
        print(f"XGBoost model for {args.symbol} is not ready. Make sure it's trained first.") 