import logging
import numpy as np
from typing import Tuple, Dict, Any, List, Optional
import pandas as pd
from datetime import datetime

# Import custom modules
from price_prediction_model import PricePredictor
from xgboost_price_predictor import XGBoostPredictor
from allora_integration import fetch_allora_signal
from secure_predictions import run_secure_model_prediction

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ensemble_predictor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EnsemblePredictor:
    """
    Ensemble predictor that combines predictions from multiple models
    including RandomForest, XGBoost, and the Allora predictive oracle
    """
    
    def __init__(self, symbol: str = "BTCUSDT", 
                 use_allora: bool = True, 
                 allora_api_key: str = None,
                 allora_weight: float = 0.3,
                 rf_threshold: float = 0.5,
                 xgb_threshold: float = 0.5,
                 ensemble_threshold: float = 0.6,
                 require_allora_agreement: bool = True,
                 use_secure_enclave: bool = False,
                 secure_model_id: str = "btc_price_predictor_v1",
                 secure_model_weight: float = 0.3):
        """
        Initialize the ensemble predictor
        
        Args:
            symbol: Trading pair symbol
            use_allora: Whether to include Allora predictions
            allora_api_key: API key for Allora (optional for mock mode)
            allora_weight: Weight to give Allora predictions (0.0 to 1.0)
            rf_threshold: Confidence threshold for RandomForest model
            xgb_threshold: Confidence threshold for XGBoost model
            ensemble_threshold: Confidence threshold for ensemble prediction
            require_allora_agreement: Only predict if Allora agrees with models
            use_secure_enclave: Whether to use the Marlin TEE secure enclave
            secure_model_id: ID of the model to use in secure enclave
            secure_model_weight: Weight to give secure model predictions
        """
        self.symbol = symbol
        self.use_allora = use_allora
        self.allora_api_key = allora_api_key
        self.allora_weight = allora_weight
        self.rf_threshold = rf_threshold
        self.xgb_threshold = xgb_threshold
        self.ensemble_threshold = ensemble_threshold
        self.require_allora_agreement = require_allora_agreement
        self.use_secure_enclave = use_secure_enclave
        self.secure_model_id = secure_model_id
        self.secure_model_weight = secure_model_weight
        
        # Initialize models
        self.rf_predictor = PricePredictor(symbol=symbol)
        self.rf_predictor.threshold = rf_threshold
        
        self.xgb_predictor = XGBoostPredictor(symbol=symbol)
        self.xgb_predictor.threshold = xgb_threshold
        
        logger.info(f"Ensemble predictor initialized for {symbol}")
        logger.info(f"Allora integration: {'Enabled' if use_allora else 'Disabled'}")
        if use_allora:
            logger.info(f"Allora weight: {allora_weight}")
            logger.info(f"Require Allora agreement: {require_allora_agreement}")
            
        logger.info(f"Secure enclave: {'Enabled' if use_secure_enclave else 'Disabled'}")
        if use_secure_enclave:
            logger.info(f"Secure model ID: {secure_model_id}")
            logger.info(f"Secure model weight: {secure_model_weight}")
    
    def predict(self, df: pd.DataFrame) -> Tuple[int, float, Dict[str, Any]]:
        """
        Make a prediction based on ensemble of models
        
        Args:
            df: DataFrame with price data and indicators
            
        Returns:
            Tuple of (prediction, confidence, details)
            prediction: 1 for up, 0 for down, -1 for no prediction
            confidence: Float between 0 and 1
            details: Dictionary with detailed prediction info
        """
        # Get predictions from individual models
        rf_pred, rf_conf, rf_probs = self.rf_predictor.predict(df)
        xgb_pred, xgb_conf, xgb_probs = self.xgb_predictor.predict(df)
        
        # Initialize Allora prediction
        allora_pred = -1
        allora_conf = 0.0
        allora_data = {}
        
        # Get Allora prediction if enabled
        if self.use_allora:
            try:
                # Determine appropriate timeframe based on dataframe interval
                timeframe = self._determine_timeframe(df)
                allora_pred, allora_conf, allora_data = fetch_allora_signal(
                    timeframe=timeframe, 
                    use_mock=True, 
                    api_key=self.allora_api_key
                )
            except Exception as e:
                logger.error(f"Error fetching Allora prediction: {e}")
        
        # Initialize secure model prediction
        secure_pred = -1
        secure_conf = 0.0
        secure_metadata = {}
        
        # Get secure model prediction if enabled
        if self.use_secure_enclave:
            try:
                logger.info(f"Requesting prediction from secure enclave")
                secure_pred, secure_conf, secure_metadata = run_secure_model_prediction(
                    price_window=df,
                    model_id=self.secure_model_id
                )
                logger.info(f"Secure model prediction: {'UP' if secure_pred == 1 else 'DOWN' if secure_pred == 0 else 'NONE'} with {secure_conf:.2f} confidence")
            except Exception as e:
                logger.error(f"Error running secure model prediction: {e}")
        
        # Prepare prediction details
        details = {
            "timestamp": datetime.now().isoformat(),
            "models": {
                "random_forest": {
                    "prediction": rf_pred,
                    "confidence": rf_conf,
                    "probabilities": rf_probs
                },
                "xgboost": {
                    "prediction": xgb_pred,
                    "confidence": xgb_conf,
                    "probabilities": xgb_probs
                }
            }
        }
        
        # Add Allora details if available
        if self.use_allora and allora_pred != -1:
            details["models"]["allora"] = {
                "prediction": allora_pred,
                "confidence": allora_conf,
                "data": allora_data
            }
        
        # Add secure model details if available
        if self.use_secure_enclave and secure_pred != -1:
            details["models"]["secure_model"] = {
                "prediction": secure_pred,
                "confidence": secure_conf,
                "metadata": secure_metadata
            }
        
        # Determine ensemble prediction
        prediction, confidence = self._combine_predictions(
            rf_pred, rf_conf, 
            xgb_pred, xgb_conf, 
            allora_pred, allora_conf,
            secure_pred, secure_conf
        )
        
        # Add ensemble results to details
        details["ensemble"] = {
            "prediction": prediction,
            "confidence": confidence,
            "threshold": self.ensemble_threshold
        }
        
        # Log prediction
        self._log_prediction(prediction, confidence, rf_pred, rf_conf, 
                            xgb_pred, xgb_conf, allora_pred, allora_conf,
                            secure_pred, secure_conf)
        
        return prediction, confidence, details
    
    def _combine_predictions(self, 
                            rf_pred: int, rf_conf: float,
                            xgb_pred: int, xgb_conf: float,
                            allora_pred: int, allora_conf: float,
                            secure_pred: int = -1, secure_conf: float = 0.0) -> Tuple[int, float]:
        """
        Combine predictions from multiple models
        
        Args:
            rf_pred: RandomForest prediction
            rf_conf: RandomForest confidence
            xgb_pred: XGBoost prediction
            xgb_conf: XGBoost confidence
            allora_pred: Allora prediction
            allora_conf: Allora confidence
            secure_pred: Secure model prediction
            secure_conf: Secure model confidence
            
        Returns:
            Tuple of (prediction, confidence)
        """
        # Compute weighted confidence for each direction (up/down)
        up_confidence = 0.0
        down_confidence = 0.0
        model_weight_sum = 0.0
        
        # RandomForest contribution
        if rf_pred != -1:
            model_weight = 0.35  # 35% weight to RandomForest
            model_weight_sum += model_weight
            if rf_pred == 1:
                up_confidence += rf_conf * model_weight
            else:
                down_confidence += rf_conf * model_weight
        
        # XGBoost contribution
        if xgb_pred != -1:
            model_weight = 0.35  # 35% weight to XGBoost
            model_weight_sum += model_weight
            if xgb_pred == 1:
                up_confidence += xgb_conf * model_weight
            else:
                down_confidence += xgb_conf * model_weight
        
        # Allora contribution
        if self.use_allora and allora_pred != -1:
            model_weight = self.allora_weight  # Configurable weight to Allora
            model_weight_sum += model_weight
            if allora_pred == 1:
                up_confidence += allora_conf * model_weight
            else:
                down_confidence += allora_conf * model_weight
                
        # Secure model contribution
        if self.use_secure_enclave and secure_pred != -1:
            model_weight = self.secure_model_weight  # Configurable weight to secure model
            model_weight_sum += model_weight
            if secure_pred == 1:
                up_confidence += secure_conf * model_weight
            else:
                down_confidence += secure_conf * model_weight
        
        # If no valid predictions, return no prediction
        if model_weight_sum == 0:
            return -1, 0.0
        
        # Normalize confidences
        up_confidence /= model_weight_sum
        down_confidence /= model_weight_sum
        
        # Determine direction with highest confidence
        if up_confidence > down_confidence:
            prediction = 1
            confidence = up_confidence
        else:
            prediction = 0
            confidence = down_confidence
        
        # Apply Allora agreement requirement if enabled
        if self.require_allora_agreement and self.use_allora and allora_pred != -1:
            if prediction != allora_pred:
                logger.info("Allora disagrees with model ensemble, suppressing prediction")
                return -1, 0.0
        
        # Apply ensemble threshold
        if confidence < self.ensemble_threshold:
            logger.info(f"Ensemble confidence {confidence:.2f} below threshold {self.ensemble_threshold}")
            return -1, confidence
        
        return prediction, confidence
    
    def _determine_timeframe(self, df: pd.DataFrame) -> str:
        """
        Determine appropriate Allora timeframe based on dataframe
        
        Args:
            df: DataFrame with price data
            
        Returns:
            Timeframe string for Allora API
        """
        # Try to determine the interval from the dataframe
        if len(df) >= 2:
            try:
                # Calculate the time difference between rows
                time_diff = (df.index[-1] - df.index[-2]).total_seconds()
                
                # Map to common timeframes
                if time_diff <= 60*5:  # 5 minutes or less
                    return "1h"  # Use 1-hour predictions for short timeframes
                elif time_diff <= 60*60:  # 1 hour or less
                    return "1h"
                elif time_diff <= 60*60*4:  # 4 hours or less
                    return "4h"
                else:
                    return "1d"
            except:
                pass
        
        # Default to 1-hour timeframe
        return "1h"
    
    def _log_prediction(self, prediction: int, confidence: float, 
                        rf_pred: int, rf_conf: float, 
                        xgb_pred: int, xgb_conf: float, 
                        allora_pred: int, allora_conf: float,
                        secure_pred: int = -1, secure_conf: float = 0.0):
        """Log prediction details"""
        pred_str = "UP" if prediction == 1 else "DOWN" if prediction == 0 else "NONE"
        rf_str = "UP" if rf_pred == 1 else "DOWN" if rf_pred == 0 else "NONE"
        xgb_str = "UP" if xgb_pred == 1 else "DOWN" if xgb_pred == 0 else "NONE"
        allora_str = "UP" if allora_pred == 1 else "DOWN" if allora_pred == 0 else "NONE"
        secure_str = "UP" if secure_pred == 1 else "DOWN" if secure_pred == 0 else "NONE"
        
        log_msg = f"ENSEMBLE: {pred_str} ({confidence:.2f}) | " \
                 f"RF: {rf_str} ({rf_conf:.2f}) | " \
                 f"XGB: {xgb_str} ({xgb_conf:.2f})"
                 
        if self.use_allora:
            log_msg += f" | ALLORA: {allora_str} ({allora_conf:.2f})"
            
        if self.use_secure_enclave:
            log_msg += f" | SECURE: {secure_str} ({secure_conf:.2f})"
            
        logger.info(log_msg)


if __name__ == "__main__":
    import pandas as pd
    from fetch_crypto_prices import PriceFetcher
    from technical_indicators import TechnicalIndicators
    
    # Test the ensemble predictor
    fetcher = PriceFetcher(symbol="BTCUSDT")
    df = fetcher.fetch_recent_prices(limit=100)
    
    if df is not None and not df.empty:
        # Add technical indicators
        df = TechnicalIndicators.add_all_indicators(df)
        
        # Create predictor with Allora integration and secure enclave
        predictor = EnsemblePredictor(
            symbol="BTCUSDT",
            use_allora=True,
            require_allora_agreement=True,
            use_secure_enclave=True,
            secure_model_weight=0.3
        )
        
        # Make prediction
        prediction, confidence, details = predictor.predict(df)
        
        print(f"Ensemble Prediction: {'UP' if prediction == 1 else 'DOWN' if prediction == 0 else 'NONE'}")
        print(f"Confidence: {confidence:.2f}")
        
        # Individual model predictions
        rf_pred = details["models"]["random_forest"]["prediction"]
        rf_conf = details["models"]["random_forest"]["confidence"]
        print(f"RandomForest: {'UP' if rf_pred == 1 else 'DOWN' if rf_pred == 0 else 'NONE'} ({rf_conf:.2f})")
        
        xgb_pred = details["models"]["xgboost"]["prediction"]
        xgb_conf = details["models"]["xgboost"]["confidence"]
        print(f"XGBoost: {'UP' if xgb_pred == 1 else 'DOWN' if xgb_pred == 0 else 'NONE'} ({xgb_conf:.2f})")
        
        if "allora" in details["models"]:
            allora_pred = details["models"]["allora"]["prediction"]
            allora_conf = details["models"]["allora"]["confidence"]
            print(f"Allora: {'UP' if allora_pred == 1 else 'DOWN' if allora_pred == 0 else 'NONE'} ({allora_conf:.2f})")
            
        if "secure_model" in details["models"]:
            secure_pred = details["models"]["secure_model"]["prediction"]
            secure_conf = details["models"]["secure_model"]["confidence"]
            print(f"Secure Model: {'UP' if secure_pred == 1 else 'DOWN' if secure_pred == 0 else 'NONE'} ({secure_conf:.2f})")
            
            # Print secure model metadata
            secure_metadata = details["models"]["secure_model"]["metadata"]
            print(f"Secure Prediction Signature: {secure_metadata.get('signature', 'N/A')}")
            print(f"Secure Enclave ID: {secure_metadata.get('enclave_id', 'N/A')}")
    else:
        print("Failed to fetch price data") 