import pandas as pd
import numpy as np
import joblib
import os
import logging
import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from fetch_crypto_prices import PriceFetcher

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, symbol: str = "BTCUSDT", window_size: int = 5, model_path: str = "models"):
        self.symbol = symbol
        self.window_size = window_size
        self.model_path = model_path
        self.model_file = os.path.join(model_path, f"{symbol.lower()}_predictor.joblib")
        self.scaler_file = os.path.join(model_path, f"{symbol.lower()}_scaler.joblib")
        self.model_metadata_file = os.path.join(model_path, f"{symbol.lower()}_metadata.json")
        
        # Create model directory if it doesn't exist
        os.makedirs(model_path, exist_ok=True)
        
        # Initialize price fetcher
        self.price_fetcher = PriceFetcher(symbol=symbol)
    
    def is_model_trained(self) -> bool:
        """Check if model and scaler files already exist"""
        return os.path.exists(self.model_file) and os.path.exists(self.scaler_file)
    
    def save_model(self, model, scaler, metrics=None) -> bool:
        """
        Save model, scaler and metadata to disk
        
        Args:
            model: Trained model to save
            scaler: Fitted scaler to save
            metrics: Dictionary of model performance metrics
            
        Returns:
            bool: Whether saving was successful
        """
        try:
            # Save model
            joblib.dump(model, self.model_file)
            logger.info(f"Saved model to {self.model_file}")
            
            # Save scaler
            joblib.dump(scaler, self.scaler_file)
            logger.info(f"Saved scaler to {self.scaler_file}")
            
            # Save metadata
            metadata = {
                "symbol": self.symbol,
                "window_size": self.window_size,
                "created_at": datetime.datetime.now().isoformat(),
                "model_type": type(model).__name__,
            }
            
            # Add metrics if provided
            if metrics:
                metadata["metrics"] = metrics
            
            # Save metadata as JSON
            import json
            with open(self.model_metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Saved model metadata to {self.model_metadata_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
        
    def prepare_data(self, df: pd.DataFrame) -> tuple:
        """
        Prepare data for model training by:
        1. Creating features from price data
        2. Creating target variable (price goes up or down)
        3. Splitting into training and testing sets
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        # Make sure we have enough data
        if len(df) < self.window_size + 1:
            raise ValueError(f"Not enough data for window size {self.window_size}")
        
        # Create features
        features = []
        targets = []
        
        # Use closing prices
        prices = df['close'].values
        
        # For each window of data
        for i in range(len(prices) - self.window_size):
            window = prices[i:i+self.window_size]
            next_price = prices[i+self.window_size]
            
            # Features: current window of prices
            features.append(window)
            
            # Target: 1 if price goes up, 0 if it goes down
            targets.append(1 if next_price > window[-1] else 0)
            
        X = np.array(features)
        y = np.array(targets)
        
        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        return X_train, X_test, y_train, y_test, scaler
    
    def train_model(self, force_retrain=False) -> bool:
        """
        Train a machine learning model and save it to disk
        
        Args:
            force_retrain: Whether to force retraining even if model exists
            
        Returns:
            bool: Whether training was successful
        """
        # Check if model already exists and we're not forcing a retrain
        if self.is_model_trained() and not force_retrain:
            logger.info(f"Model for {self.symbol} already exists. Use force_retrain=True to retrain.")
            return True
            
        logger.info(f"Fetching historical data for {self.symbol}...")
        df = self.price_fetcher.fetch_historical_klines(interval="1h", limit=1000)
        
        if df.empty:
            logger.error("Failed to fetch historical data")
            return False
            
        logger.info(f"Preparing data with window size {self.window_size}...")
        try:
            X_train, X_test, y_train, y_test, scaler = self.prepare_data(df)
        except ValueError as e:
            logger.error(f"Data preparation failed: {e}")
            return False
        
        logger.info("Training model...")
        # Define the model
        model = RandomForestClassifier(random_state=42)
        
        # Define the parameter grid for GridSearchCV
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        }
        
        # Perform grid search to find best parameters
        grid_search = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)
        
        # Get the best model
        best_model = grid_search.best_estimator_
        
        # Evaluate on test set
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Test accuracy: {accuracy:.4f}")
        
        # Get classification report as string
        report = classification_report(y_test, y_pred)
        logger.info("\nClassification Report:\n" + report)
        
        # Prepare metrics for metadata
        metrics = {
            "accuracy": float(accuracy),
            "best_parameters": grid_search.best_params_,
            "training_samples": len(X_train),
            "test_samples": len(X_test)
        }
        
        # Save model, scaler and metadata
        save_success = self.save_model(best_model, scaler, metrics)
        
        return save_success

if __name__ == "__main__":
    # Accept command-line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Train crypto price prediction model')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='Trading pair symbol')
    parser.add_argument('--window', type=int, default=5, help='Price window size')
    parser.add_argument('--force', action='store_true', help='Force retraining of model')
    args = parser.parse_args()
    
    # Create trainer
    trainer = ModelTrainer(symbol=args.symbol, window_size=args.window)
    
    # Train model
    success = trainer.train_model(force_retrain=args.force)
    
    if success:
        print(f"Successfully trained model for {args.symbol}")
    else:
        print(f"Failed to train model for {args.symbol}") 