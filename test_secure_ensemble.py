import pandas as pd
import numpy as np
import logging
import json
from datetime import datetime, timedelta

# Import custom modules
from secure_predictions import run_secure_model_prediction
from ensemble_predictor import EnsemblePredictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("test_secure_ensemble.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def generate_mock_price_data(days=5, freq='1h'):
    """Generate mock price data for testing"""
    # Create date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq=freq)
    
    # Generate base price (starting at around $50,000)
    base_price = 50000
    
    # Add trend and noise
    trend = np.linspace(0, 2000, len(dates))  # Upward trend
    noise = np.random.normal(0, 500, len(dates))  # Random noise
    
    # Calculate prices with some volatility
    closes = base_price + trend + noise
    opens = closes - np.random.normal(0, 100, len(dates))
    highs = np.maximum(opens, closes) + np.random.normal(100, 50, len(dates))
    lows = np.minimum(opens, closes) - np.random.normal(100, 50, len(dates))
    volumes = np.random.normal(1000, 200, len(dates))
    
    # Create DataFrame
    df = pd.DataFrame({
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes
    }, index=dates)
    
    # Add some technical indicators (simplified for mock data)
    df['ma_7'] = df['close'].rolling(7).mean()
    df['ma_25'] = df['close'].rolling(25).mean()
    df['rsi'] = 50 + np.random.normal(0, 15, len(df))  # Fake RSI
    df['volume_ma'] = df['volume'].rolling(14).mean()
    
    return df

def test_secure_enclave_prediction():
    """Test the secure enclave prediction function directly"""
    print("\n=== Testing Secure Enclave Prediction ===")
    
    # Generate sample price data
    df = generate_mock_price_data(days=3, freq='1h')
    print(f"Generated {len(df)} periods of mock price data")
    
    # Call secure prediction function
    prediction, confidence, metadata = run_secure_model_prediction(
        price_window=df,
        model_id="btc_price_predictor_v1",
        attestation_enabled=True
    )
    
    # Display results
    print(f"\nSecure Prediction: {'UP' if prediction == 1 else 'DOWN' if prediction == 0 else 'NONE'}")
    print(f"Confidence: {confidence:.2f}")
    print(f"Secure Enclave ID: {metadata.get('enclave_id', 'unknown')}")
    print(f"Request ID: {metadata.get('request_id', 'unknown')}")
    print(f"Computation time: {metadata.get('computation_time_ms', 0)} ms")
    print(f"Signature: {metadata.get('signature', 'none')}")

def test_ensemble_with_secure_enclave():
    """Test the ensemble predictor with secure enclave integration"""
    print("\n=== Testing Ensemble Predictor with Secure Enclave ===")
    
    # Generate sample price data with indicators
    df = generate_mock_price_data(days=5, freq='1h')
    print(f"Generated {len(df)} periods of mock price data for ensemble testing")
    
    # Create predictor with Allora and secure enclave enabled
    predictor = EnsemblePredictor(
        symbol="BTCUSDT",
        use_allora=True,
        use_secure_enclave=True,
        secure_model_weight=0.3,
        require_allora_agreement=False  # Don't require agreement for this test
    )
    
    # Monkey patch the predict methods of the base models to return mock predictions
    def mock_rf_predict(self, df):
        return 1, 0.75, {"class_0": 0.25, "class_1": 0.75}
        
    def mock_xgb_predict(self, df):
        return 1, 0.82, {"class_0": 0.18, "class_1": 0.82}
    
    # Apply monkey patches
    import types
    predictor.rf_predictor.predict = types.MethodType(mock_rf_predict, predictor.rf_predictor)
    predictor.xgb_predictor.predict = types.MethodType(mock_xgb_predict, predictor.xgb_predictor)
    
    # Make prediction with ensemble
    prediction, confidence, details = predictor.predict(df)
    
    # Display results
    print(f"\nEnsemble Prediction: {'UP' if prediction == 1 else 'DOWN' if prediction == 0 else 'NONE'}")
    print(f"Ensemble Confidence: {confidence:.2f}")
    
    # Display individual model predictions
    for model_name, model_data in details.get("models", {}).items():
        if model_name == "secure_model":
            pred = model_data.get("prediction", -1)
            conf = model_data.get("confidence", 0.0)
            metadata = model_data.get("metadata", {})
            print(f"\nSecure Model:")
            print(f"  Prediction: {'UP' if pred == 1 else 'DOWN' if pred == 0 else 'NONE'}")
            print(f"  Confidence: {conf:.2f}")
            print(f"  Enclave ID: {metadata.get('enclave_id', 'unknown')}")
            print(f"  Computation time: {metadata.get('computation_time_ms', 0)} ms")
            print(f"  Signature: {metadata.get('signature', 'none')[:16]}...")
        elif model_name == "allora":
            pred = model_data.get("prediction", -1)
            conf = model_data.get("confidence", 0.0)
            print(f"\nAllora Oracle:")
            print(f"  Prediction: {'UP' if pred == 1 else 'DOWN' if pred == 0 else 'NONE'}")
            print(f"  Confidence: {conf:.2f}")
        elif model_name == "random_forest":
            pred = model_data.get("prediction", -1)
            conf = model_data.get("confidence", 0.0)
            print(f"\nRandomForest Model:")
            print(f"  Prediction: {'UP' if pred == 1 else 'DOWN' if pred == 0 else 'NONE'}")
            print(f"  Confidence: {conf:.2f}")
        elif model_name == "xgboost":
            pred = model_data.get("prediction", -1)
            conf = model_data.get("confidence", 0.0)
            print(f"\nXGBoost Model:")
            print(f"  Prediction: {'UP' if pred == 1 else 'DOWN' if pred == 0 else 'NONE'}")
            print(f"  Confidence: {conf:.2f}")

if __name__ == "__main__":
    print("=== Marlin TEE Secure Prediction Testing ===")
    
    # Test standalone secure enclave
    test_secure_enclave_prediction()
    
    # Test ensemble predictor with secure enclave
    test_ensemble_with_secure_enclave() 