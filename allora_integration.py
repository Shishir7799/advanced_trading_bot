import requests
import json
import logging
import random
import time
from typing import Tuple, Optional, Dict, Any
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("allora_integration.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AlloraIntegration:
    """Class for integrating with the Allora predictive oracle"""
    
    # Base API URL (mock URL for demonstration)
    BASE_URL = "https://api.allora.network/v1"
    
    def __init__(self, api_key: str = None, use_mock: bool = True):
        """
        Initialize the Allora integration
        
        Args:
            api_key: API key for Allora (optional for mock mode)
            use_mock: Whether to use mock data instead of real API
        """
        self.api_key = api_key
        self.use_mock = use_mock
        logger.info(f"Allora integration initialized in {'mock' if use_mock else 'live'} mode")
        
        # Cache to store recent predictions and avoid excessive API calls
        self.cache = {}
        self.cache_expiry = 300  # 5 minutes
    
    def fetch_btc_prediction(self, timeframe: str = "1h") -> Tuple[int, float, Dict[str, Any]]:
        """
        Fetch BTC/USDT price direction prediction from Allora
        
        Args:
            timeframe: Prediction timeframe (1h, 4h, 1d)
            
        Returns:
            Tuple of (prediction, confidence, raw_data)
            prediction: 1 for up, 0 for down, -1 for no prediction
            confidence: Float between 0 and 1
            raw_data: Dictionary with raw response data
        """
        # Check cache first
        cache_key = f"BTC_USDT_{timeframe}"
        if cache_key in self.cache:
            cache_entry = self.cache[cache_key]
            if datetime.now().timestamp() - cache_entry['timestamp'] < self.cache_expiry:
                logger.info(f"Using cached Allora prediction for BTC/USDT ({timeframe})")
                return cache_entry['prediction'], cache_entry['confidence'], cache_entry['data']
        
        if self.use_mock:
            return self._generate_mock_prediction(timeframe)
        else:
            return self._fetch_real_prediction(timeframe)
    
    def _generate_mock_prediction(self, timeframe: str) -> Tuple[int, float, Dict[str, Any]]:
        """
        Generate a mock prediction when not using the real API
        
        Args:
            timeframe: Prediction timeframe
            
        Returns:
            Tuple of (prediction, confidence, raw_data)
        """
        # Add some randomness but with bias toward realistic patterns
        # In real markets, there's often momentum and trend persistence
        
        # Get current hour to create some time-based patterns
        current_hour = datetime.now().hour
        
        # Bias prediction based on time of day (just for mock data variation)
        if current_hour < 8:  # Early hours - slightly bearish bias
            prediction_bias = 0.45  
        elif current_hour < 16:  # Middle of day - bullish bias
            prediction_bias = 0.60
        else:  # Evening - neutral bias
            prediction_bias = 0.50
            
        # Random prediction with bias
        prediction = 1 if random.random() < prediction_bias else 0
        
        # Confidence tends to be higher for shorter timeframes and when continuing trends
        base_confidence = 0.70 if timeframe == "1h" else 0.65 if timeframe == "4h" else 0.60
        
        # Add some variability to confidence
        confidence = min(0.95, max(0.55, base_confidence + random.uniform(-0.15, 0.15)))
        
        # Prepare mock response data
        mock_data = {
            "symbol": "BTCUSDT",
            "timeframe": timeframe,
            "prediction": "up" if prediction == 1 else "down",
            "confidence": confidence,
            "timestamp": datetime.now().isoformat(),
            "prediction_id": f"mock_{int(time.time())}_{random.randint(1000, 9999)}",
            "source": "allora_mock",
            "factors": {
                "technical": random.uniform(0.3, 0.7),
                "sentiment": random.uniform(0.3, 0.7),
                "on_chain": random.uniform(0.3, 0.7),
                "market_structure": random.uniform(0.3, 0.7)
            }
        }
        
        # Cache the prediction
        self.cache[f"BTC_USDT_{timeframe}"] = {
            'prediction': prediction,
            'confidence': confidence,
            'data': mock_data,
            'timestamp': datetime.now().timestamp()
        }
        
        logger.info(f"Generated mock Allora prediction: {'UP' if prediction == 1 else 'DOWN'} with {confidence:.2f} confidence")
        return prediction, confidence, mock_data
    
    def _fetch_real_prediction(self, timeframe: str) -> Tuple[int, float, Dict[str, Any]]:
        """
        Fetch a real prediction from the Allora API
        
        Args:
            timeframe: Prediction timeframe
            
        Returns:
            Tuple of (prediction, confidence, raw_data)
        """
        try:
            # Prepare API request
            headers = {
                "Content-Type": "application/json",
                "X-API-Key": self.api_key
            }
            
            # Request parameters
            params = {
                "symbol": "BTCUSDT",
                "timeframe": timeframe
            }
            
            # Make API request
            response = requests.get(
                f"{self.BASE_URL}/predictions/crypto",
                headers=headers,
                params=params
            )
            
            # Handle response
            if response.status_code == 200:
                data = response.json()
                
                # Extract prediction and confidence
                pred_str = data.get("prediction", "").lower()
                prediction = 1 if pred_str == "up" else 0 if pred_str == "down" else -1
                confidence = float(data.get("confidence", 0.0))
                
                # Cache the prediction
                self.cache[f"BTC_USDT_{timeframe}"] = {
                    'prediction': prediction,
                    'confidence': confidence,
                    'data': data,
                    'timestamp': datetime.now().timestamp()
                }
                
                logger.info(f"Fetched Allora prediction: {'UP' if prediction == 1 else 'DOWN'} with {confidence:.2f} confidence")
                return prediction, confidence, data
            else:
                logger.error(f"Error fetching Allora prediction: {response.status_code} - {response.text}")
                return -1, 0.0, {"error": response.text}
        
        except Exception as e:
            logger.error(f"Exception fetching Allora prediction: {e}")
            return -1, 0.0, {"error": str(e)}
    
    def get_historical_predictions(self, symbol: str = "BTCUSDT", days: int = 7) -> Dict[str, Any]:
        """
        Get historical predictions for performance analysis
        
        Args:
            symbol: Trading pair symbol
            days: Number of days of historical data
            
        Returns:
            Dictionary with historical predictions
        """
        if self.use_mock:
            return self._generate_mock_historical_predictions(symbol, days)
        else:
            return self._fetch_real_historical_predictions(symbol, days)
    
    def _generate_mock_historical_predictions(self, symbol: str, days: int) -> Dict[str, Any]:
        """Generate mock historical predictions"""
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        # Generate hourly predictions
        timestamps = []
        current = start_time
        predictions = []
        confidences = []
        accuracies = []
        
        # Create a synthetic accuracy pattern that's better than random
        # but not too perfect (around 65-75% accurate)
        accuracy_base = 0.70
        
        while current < end_time:
            timestamps.append(current.isoformat())
            
            # Generate prediction with a slight up bias (55%)
            pred = 1 if random.random() < 0.55 else 0
            predictions.append(pred)
            
            # Generate confidence
            conf = random.uniform(0.6, 0.85)
            confidences.append(conf)
            
            # Determine if prediction was correct
            # Higher confidence should correlate with higher accuracy
            accuracy_threshold = accuracy_base * (conf / 0.7)
            was_correct = random.random() < accuracy_threshold
            accuracies.append(1 if was_correct else 0)
            
            # Move to next hour
            current += timedelta(hours=1)
        
        # Calculate overall accuracy
        overall_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0
        
        return {
            "symbol": symbol,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "count": len(predictions),
            "overall_accuracy": overall_accuracy,
            "predictions": [
                {
                    "timestamp": timestamps[i],
                    "prediction": "up" if predictions[i] == 1 else "down",
                    "confidence": confidences[i],
                    "was_correct": bool(accuracies[i])
                }
                for i in range(len(predictions))
            ]
        }
    
    def _fetch_real_historical_predictions(self, symbol: str, days: int) -> Dict[str, Any]:
        """Fetch real historical predictions from Allora API"""
        try:
            # Prepare API request
            headers = {
                "Content-Type": "application/json",
                "X-API-Key": self.api_key
            }
            
            # Request parameters
            params = {
                "symbol": symbol,
                "days": days
            }
            
            # Make API request
            response = requests.get(
                f"{self.BASE_URL}/predictions/history",
                headers=headers,
                params=params
            )
            
            # Handle response
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Error fetching historical predictions: {response.status_code} - {response.text}")
                return {"error": response.text}
        
        except Exception as e:
            logger.error(f"Exception fetching historical predictions: {e}")
            return {"error": str(e)}


# Create a simple function to directly fetch predictions
def fetch_allora_signal(timeframe: str = "1h", use_mock: bool = True, api_key: str = None) -> Tuple[int, float, Dict[str, Any]]:
    """
    Fetch BTC/USDT price direction prediction from Allora
    
    Args:
        timeframe: Prediction timeframe (1h, 4h, 1d)
        use_mock: Whether to use mock data
        api_key: API key for Allora (optional for mock mode)
        
    Returns:
        Tuple of (prediction, confidence, raw_data)
        prediction: 1 for up, 0 for down, -1 for no prediction
        confidence: Float between 0 and 1
        raw_data: Dictionary with raw response data
    """
    allora = AlloraIntegration(api_key=api_key, use_mock=use_mock)
    return allora.fetch_btc_prediction(timeframe=timeframe)


if __name__ == "__main__":
    # Test the Allora integration
    allora = AlloraIntegration(use_mock=True)
    
    # Test prediction
    prediction, confidence, data = allora.fetch_btc_prediction()
    
    print(f"Allora BTC/USDT Prediction: {'UP' if prediction == 1 else 'DOWN' if prediction == 0 else 'NONE'}")
    print(f"Confidence: {confidence:.2f}")
    print(f"Raw data: {json.dumps(data, indent=2)}")
    
    # Test historical predictions
    history = allora.get_historical_predictions(days=3)
    print(f"\nHistorical Prediction Accuracy: {history.get('overall_accuracy', 0):.2%}")
    print(f"Total predictions: {history.get('count', 0)}") 