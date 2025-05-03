import os
import time
import logging
import pandas as pd
import argparse
from datetime import datetime, timedelta

from fetch_crypto_prices import PriceFetcher
from price_prediction_model import PricePredictor
from train_model import ModelTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_model_exists(symbol="BTCUSDT", window_size=5):
    """Check if model files exist for the given parameters"""
    model_path = "models"
    model_file = os.path.join(model_path, f"{symbol.lower()}_predictor.joblib")
    scaler_file = os.path.join(model_path, f"{symbol.lower()}_scaler.joblib")
    
    model_exists = os.path.exists(model_file)
    scaler_exists = os.path.exists(scaler_file)
    
    logger.info(f"Model file: {model_file} exists: {model_exists}")
    logger.info(f"Scaler file: {scaler_file} exists: {scaler_exists}")
    
    return model_exists and scaler_exists

def train_or_load_model(symbol="BTCUSDT", window_size=5, force_train=False):
    """Train a new model or load an existing one"""
    # Create trainer
    trainer = ModelTrainer(symbol=symbol, window_size=window_size)
    
    # Check if model exists
    if not trainer.is_model_trained() or force_train:
        logger.info(f"{'Training new model' if not trainer.is_model_trained() else 'Force retraining model'} for {symbol}...")
        success = trainer.train_model(force_retrain=force_train)
        
        if not success:
            logger.error("Model training failed.")
            return None
        
        logger.info("Model training completed successfully.")
    else:
        logger.info(f"Model for {symbol} already exists. Skipping training.")
    
    # Create predictor to load the model
    predictor = PricePredictor(symbol=symbol, window_size=window_size)
    
    if predictor.is_model_ready():
        logger.info("Model loaded successfully.")
        return predictor
    else:
        logger.error("Failed to load model.")
        return None

def make_predictions_with_loaded_model(predictor, fetcher, num_predictions=5, interval_seconds=5):
    """Make a series of predictions using the loaded model"""
    if not predictor.is_model_ready():
        logger.error("Model not ready.")
        return
    
    for i in range(num_predictions):
        logger.info(f"Making prediction {i+1}/{num_predictions}")
        
        # Get current price
        current_price = fetcher.get_current_price()
        
        # Get price window
        price_df = fetcher.get_last_n_prices(predictor.window_size)
        
        if len(price_df) < predictor.window_size:
            logger.warning(f"Not enough price data. Have {len(price_df)}/{predictor.window_size}")
            continue
        
        # Extract prices
        price_window = price_df['price'].tolist()
        
        # Make prediction
        prediction, confidence = predictor.predict(price_window)
        
        if prediction is not None:
            logger.info(f"Prediction {i+1}: {'UP' if prediction == 1 else 'DOWN'} with {confidence:.2f} confidence")
            logger.info(f"Price window: {price_window}")
        else:
            logger.error("Prediction failed")
        
        # Wait between predictions
        if i < num_predictions - 1:
            time.sleep(interval_seconds)

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Model Persistence Demo")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Trading pair symbol")
    parser.add_argument("--window", type=int, default=5, help="Price window size")
    parser.add_argument("--force-train", action="store_true", help="Force retraining the model")
    parser.add_argument("--predictions", type=int, default=5, help="Number of predictions to make")
    args = parser.parse_args()
    
    # Check if model exists
    logger.info("Checking if model exists...")
    model_exists = check_model_exists(args.symbol, args.window)
    
    # Train or load model
    logger.info("Training or loading model...")
    predictor = train_or_load_model(args.symbol, args.window, args.force_train)
    
    if predictor is None:
        logger.error("Failed to get a working model. Exiting.")
        return
    
    # Initialize price fetcher
    fetcher = PriceFetcher(symbol=args.symbol)
    
    # Make predictions
    logger.info(f"Making {args.predictions} predictions...")
    make_predictions_with_loaded_model(predictor, fetcher, args.predictions)
    
    logger.info("Demo completed successfully.")

if __name__ == "__main__":
    main() 