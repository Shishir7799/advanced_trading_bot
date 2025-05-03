import numpy as np
import joblib
import os
import logging
from typing import Optional, List, Tuple
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PricePredictor:
    def __init__(self, symbol: str = "BTCUSDT", window_size: int = 5, model_path: str = "models"):
        self.symbol = symbol
        self.window_size = window_size
        self.model_path = model_path
        self.model_file = os.path.join(model_path, f"{symbol.lower()}_predictor.joblib")
        self.scaler_file = os.path.join(model_path, f"{symbol.lower()}_scaler.joblib")
        
        # Ensure model directory exists
        os.makedirs(model_path, exist_ok=True)
        
        # Load model and scaler
        self.model = self._load_model()
        self.scaler = self._load_scaler()
        
        if not self.is_model_ready():
            logger.warning(f"Model for {symbol} is not ready. Please ensure it has been trained.")
        
    def _load_model(self):
        """Load the trained model from disk"""
        try:
            if not os.path.exists(self.model_file):
                logger.warning(f"Model file {self.model_file} not found.")
                return None
                
            model = joblib.load(self.model_file)
            logger.info(f"Successfully loaded model from {self.model_file}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            logger.debug(f"Model load error details: {str(e)}")
            return None
            
    def _load_scaler(self):
        """Load the scaler from disk"""
        try:
            if not os.path.exists(self.scaler_file):
                logger.warning(f"Scaler file {self.scaler_file} not found.")
                return None
                
            scaler = joblib.load(self.scaler_file)
            logger.info(f"Successfully loaded scaler from {self.scaler_file}")
            return scaler
        except Exception as e:
            logger.error(f"Failed to load scaler: {e}")
            logger.debug(f"Scaler load error details: {str(e)}")
            return None
    
    def predict(self, prices: List[float]) -> Tuple[Optional[int], float]:
        """
        Predict whether the price will go up or down
        
        Args:
            prices: List of recent prices (should be of length window_size)
            
        Returns:
            Tuple of (prediction, confidence)
            - prediction: 1 for price up, 0 for price down, None for error
            - confidence: Probability of the predicted class
        """
        if self.model is None or self.scaler is None:
            logger.error("Model or scaler not loaded - cannot make prediction")
            return None, 0.0
            
        if len(prices) != self.window_size:
            logger.error(f"Expected {self.window_size} prices, got {len(prices)}")
            return None, 0.0
            
        try:
            # Convert prices to numpy array and reshape for model
            X = np.array(prices).reshape(1, -1)
            
            # Scale the features
            X_scaled = self.scaler.transform(X)
            
            # Make prediction and get probabilities
            prediction = self.model.predict(X_scaled)[0]
            probabilities = self.model.predict_proba(X_scaled)[0]
            
            # Get confidence (probability of the predicted class)
            confidence = probabilities[prediction]
            
            logger.info(f"Prediction: {'UP' if prediction == 1 else 'DOWN'} with {confidence:.2f} confidence")
            
            return prediction, confidence
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return None, 0.0
    
    def is_model_ready(self) -> bool:
        """Check if the model and scaler are loaded and ready"""
        return self.model is not None and self.scaler is not None
    
    def evaluate_prediction(self, prices: List[float], actual_next_price: float) -> dict:
        """
        Evaluate a prediction against the actual result
        
        Args:
            prices: The window of prices used for prediction
            actual_next_price: The actual next price that occurred
            
        Returns:
            Dictionary with evaluation metrics
        """
        if len(prices) != self.window_size:
            logger.error(f"Expected {self.window_size} prices, got {len(prices)}")
            return {"error": "Invalid price window size"}
            
        # Make prediction
        prediction, confidence = self.predict(prices)
        
        if prediction is None:
            return {"error": "Prediction failed"}
            
        # Determine actual outcome
        last_price = prices[-1]
        actual_direction = 1 if actual_next_price > last_price else 0
        
        # Calculate metrics
        was_correct = prediction == actual_direction
        price_change = actual_next_price - last_price
        percent_change = (price_change / last_price) * 100
        
        return {
            "prediction": "UP" if prediction == 1 else "DOWN",
            "confidence": confidence,
            "actual_direction": "UP" if actual_direction == 1 else "DOWN",
            "correct": was_correct,
            "price_change": price_change,
            "percent_change": percent_change
        }

# Test the predictor
if __name__ == "__main__":
    # Example prices (5 most recent BTC prices)
    example_prices = [19500.50, 19600.75, 19550.25, 19575.00, 19625.50]
    
    predictor = PricePredictor()
    
    if predictor.is_model_ready():
        prediction, confidence = predictor.predict(example_prices)
        print(f"Prediction: {'UP' if prediction == 1 else 'DOWN'} with {confidence:.2f} confidence")
    else:
        print("Model not ready. Please train the model first.") 