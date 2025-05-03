import pandas as pd
import numpy as np
import argparse
import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple, Optional

# Import secure model prediction
from secure_predictions import run_secure_model_prediction

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("secure_model_runner.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_price_window(file_path: str) -> Optional[pd.DataFrame]:
    """
    Load a price window from a CSV or JSON file
    
    Args:
        file_path: Path to the file
        
    Returns:
        DataFrame with price data or None if loading failed
    """
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
            # If there's a 'timestamp' or 'date' column, set it as index
            for col in ['timestamp', 'date', 'time', 'datetime']:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])
                    df.set_index(col, inplace=True)
                    break
            
            return df
        
        elif file_path.endswith('.json'):
            df = pd.read_json(file_path)
            return df
        
        else:
            logger.error(f"Unsupported file format: {file_path}")
            return None
            
    except Exception as e:
        logger.error(f"Error loading price window from {file_path}: {e}")
        return None

def generate_mock_price_window(periods: int = 48) -> pd.DataFrame:
    """
    Generate a mock price window for testing
    
    Args:
        periods: Number of periods to generate
        
    Returns:
        DataFrame with mock price data
    """
    # Create date range
    end_date = datetime.now()
    start_date = end_date - timedelta(hours=periods)
    dates = pd.date_range(start=start_date, end=end_date, freq='1h')
    
    # Generate base price around $50,000
    base_price = 50000
    
    # Add trend and noise
    trend = np.linspace(0, 1000, len(dates))  # Upward trend
    noise = np.random.normal(0, 200, len(dates))  # Random noise
    
    # Calculate prices with some volatility
    closes = base_price + trend + noise
    opens = closes - np.random.normal(0, 50, len(dates))
    highs = np.maximum(opens, closes) + np.random.normal(50, 20, len(dates))
    lows = np.minimum(opens, closes) - np.random.normal(50, 20, len(dates))
    volumes = np.random.normal(1000, 200, len(dates))
    
    # Create DataFrame
    df = pd.DataFrame({
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes
    }, index=dates)
    
    # Add some basic technical indicators
    df['ma_7'] = df['close'].rolling(7).mean()
    df['ma_25'] = df['close'].rolling(25).mean()
    df['rsi'] = 50 + np.random.normal(0, 10, len(df))  # Fake RSI
    
    return df

def secure_model_prediction(price_window: pd.DataFrame, 
                           model_id: str = "btc_price_predictor_v1",
                           verbose: bool = False) -> Tuple[int, float, Dict[str, Any]]:
    """
    Run a prediction in the Marlin TEE secure enclave
    
    Args:
        price_window: DataFrame with price data
        model_id: Model ID to use in the secure enclave
        verbose: Whether to print verbose output
        
    Returns:
        Tuple of (prediction, confidence, metadata)
    """
    start_time = time.time()
    logger.info(f"Running secure model prediction using model {model_id}")
    
    # Log input data summary
    logger.info(f"Input data: {len(price_window)} periods from "
               f"{price_window.index[0]} to {price_window.index[-1]}")
    
    # Run the prediction in the secure enclave
    prediction, confidence, metadata = run_secure_model_prediction(
        price_window=price_window,
        model_id=model_id,
        attestation_enabled=True
    )
    
    # Calculate total time including secure enclave overhead
    total_time_ms = int((time.time() - start_time) * 1000)
    secure_compute_time_ms = metadata.get("computation_time_ms", 0)
    overhead_time_ms = total_time_ms - secure_compute_time_ms
    
    # Log prediction result
    logger.info(f"Secure prediction complete: {'UP' if prediction == 1 else 'DOWN' if prediction == 0 else 'NONE'} "
               f"with {confidence:.2f} confidence")
    logger.info(f"Enclave ID: {metadata.get('enclave_id', 'unknown')}")
    logger.info(f"Request ID: {metadata.get('request_id', 'unknown')}")
    logger.info(f"Total time: {total_time_ms}ms (Secure compute: {secure_compute_time_ms}ms, "
               f"Overhead: {overhead_time_ms}ms)")
    logger.info(f"Result signature: {metadata.get('signature', 'none')}")
    
    if verbose:
        print(f"\nSecure Model Prediction: {'UP' if prediction == 1 else 'DOWN' if prediction == 0 else 'NONE'}")
        print(f"Confidence: {confidence:.2f}")
        print(f"Enclave ID: {metadata.get('enclave_id', 'unknown')}")
        print(f"Computation time: {secure_compute_time_ms} ms")
        print(f"Total time: {total_time_ms} ms")
        print(f"Signature: {metadata.get('signature', 'none')}")
    
    return prediction, confidence, metadata

def save_prediction_result(prediction: int, confidence: float, 
                          metadata: Dict[str, Any], output_file: str):
    """
    Save prediction result to a JSON file
    
    Args:
        prediction: The prediction (1 for up, 0 for down, -1 for no prediction)
        confidence: Confidence value
        metadata: Prediction metadata
        output_file: Output file path
    """
    result = {
        "prediction": "up" if prediction == 1 else "down" if prediction == 0 else "none",
        "confidence": confidence,
        "timestamp": datetime.now().isoformat(),
        "metadata": metadata
    }
    
    try:
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        logger.info(f"Prediction result saved to {output_file}")
    except Exception as e:
        logger.error(f"Error saving prediction result: {e}")

def main():
    parser = argparse.ArgumentParser(description='Run secure model prediction on price data')
    parser.add_argument('--input', '-i', type=str, help='Input price window file (CSV or JSON)')
    parser.add_argument('--output', '-o', type=str, default='secure_prediction_result.json', 
                       help='Output file for prediction result')
    parser.add_argument('--model-id', '-m', type=str, default='btc_price_predictor_v1',
                       help='Model ID to use in secure enclave')
    parser.add_argument('--periods', '-p', type=int, default=48,
                       help='Number of periods for mock data (if no input file)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Print verbose output')
    
    args = parser.parse_args()
    
    # Load or generate price window
    if args.input:
        price_window = load_price_window(args.input)
        if price_window is None:
            print(f"Error loading price data from {args.input}. Using mock data instead.")
            price_window = generate_mock_price_window(args.periods)
    else:
        print("No input file specified. Using mock price data.")
        price_window = generate_mock_price_window(args.periods)
    
    # Run prediction
    prediction, confidence, metadata = secure_model_prediction(
        price_window=price_window,
        model_id=args.model_id,
        verbose=args.verbose
    )
    
    # Save result
    if args.output:
        save_prediction_result(prediction, confidence, metadata, args.output)

if __name__ == "__main__":
    main() 