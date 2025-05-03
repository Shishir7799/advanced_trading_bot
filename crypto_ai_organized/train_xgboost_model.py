import pandas as pd
import numpy as np
import joblib
import os
import logging
import datetime
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, precision_recall_curve, f1_score
from typing import Dict, Any, Tuple, List
import matplotlib.pyplot as plt
from fetch_crypto_prices import PriceFetcher
from technical_indicators import TechnicalIndicators

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class XGBoostModelTrainer:
    def __init__(self, symbol: str = "BTCUSDT", window_size: int = 5, model_path: str = "models"):
        self.symbol = symbol
        self.window_size = window_size
        self.model_path = model_path
        
        # Define file paths
        self.model_file = os.path.join(model_path, f"{symbol.lower()}_xgboost.joblib")
        self.scaler_file = os.path.join(model_path, f"{symbol.lower()}_xgboost_scaler.joblib")
        self.model_metadata_file = os.path.join(model_path, f"{symbol.lower()}_xgboost_metadata.json")
        self.feature_importance_file = os.path.join(model_path, f"{symbol.lower()}_xgboost_feature_importance.png")
        self.performance_metrics_file = os.path.join(model_path, f"{symbol.lower()}_xgboost_performance.png")
        
        # Create model directory if it doesn't exist
        os.makedirs(model_path, exist_ok=True)
        
        # Initialize price fetcher
        self.price_fetcher = PriceFetcher(symbol=symbol)
    
    def is_model_trained(self) -> bool:
        """Check if model and scaler files already exist"""
        return os.path.exists(self.model_file) and os.path.exists(self.scaler_file)
    
    def save_model(self, model, scaler, metrics=None, feature_names=None) -> bool:
        """
        Save model, scaler and metadata to disk
        
        Args:
            model: Trained model to save
            scaler: Fitted scaler to save
            metrics: Dictionary of model performance metrics
            feature_names: List of feature names for importance plotting
            
        Returns:
            bool: Whether saving was successful
        """
        try:
            # Save model
            joblib.dump(model, self.model_file)
            logger.info(f"Saved XGBoost model to {self.model_file}")
            
            # Save scaler
            joblib.dump(scaler, self.scaler_file)
            logger.info(f"Saved scaler to {self.scaler_file}")
            
            # Save metadata
            metadata = {
                "model_type": "XGBoost",
                "symbol": self.symbol,
                "window_size": self.window_size,
                "created_at": datetime.datetime.now().isoformat(),
                "xgboost_version": xgb.__version__,
                "feature_names": feature_names
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
    
    def create_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Create advanced features from price data
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Feature matrix and list of feature names
        """
        # First add all technical indicators
        enriched_df = TechnicalIndicators.add_all_indicators(df)
        
        # Define lookback periods for feature creation
        lookback_periods = [self.window_size]
        
        # Initialize feature matrix and names
        features = []
        feature_names = []
        
        # Get the price data minus the lookback window
        for i in range(len(enriched_df) - max(lookback_periods)):
            idx = i + max(lookback_periods)
            
            # Basic price features from window
            price_window = df['close'].iloc[idx-self.window_size:idx].values
            feature_vector = price_window.copy()
            feature_names_i = [f'price_{j}' for j in range(self.window_size)]
            
            # Add momentum features (price differences)
            price_momentum = np.diff(price_window)
            feature_vector = np.append(feature_vector, price_momentum)
            feature_names_i.extend([f'momentum_{j}' for j in range(len(price_momentum))])
            
            # Add rolling statistics
            rolling_mean = np.mean(price_window)
            rolling_std = np.std(price_window)
            feature_vector = np.append(feature_vector, [rolling_mean, rolling_std])
            feature_names_i.extend(['rolling_mean', 'rolling_std'])
            
            # Add Z-score of latest price
            if rolling_std != 0:
                z_score = (price_window[-1] - rolling_mean) / rolling_std
            else:
                z_score = 0
            feature_vector = np.append(feature_vector, z_score)
            feature_names_i.append('z_score')
            
            # Add relative position within price range
            price_range = np.max(price_window) - np.min(price_window)
            if price_range != 0:
                rel_position = (price_window[-1] - np.min(price_window)) / price_range
            else:
                rel_position = 0.5
            feature_vector = np.append(feature_vector, rel_position)
            feature_names_i.append('rel_position')
            
            # Add technical indicators at the current point
            current_row = enriched_df.iloc[idx]
            
            # RSI
            if not np.isnan(current_row['rsi']):
                feature_vector = np.append(feature_vector, current_row['rsi'])
                feature_names_i.append('rsi')
            
            # MACD components
            for macd_feature in ['macd_line', 'macd_signal', 'macd_histogram']:
                if macd_feature in current_row and not np.isnan(current_row[macd_feature]):
                    feature_vector = np.append(feature_vector, current_row[macd_feature])
                    feature_names_i.append(macd_feature)
            
            # Bollinger Bands metrics
            for bb_feature in ['bb_pct_b', 'bb_bandwidth']:
                if bb_feature in current_row and not np.isnan(current_row[bb_feature]):
                    feature_vector = np.append(feature_vector, current_row[bb_feature])
                    feature_names_i.append(bb_feature)
            
            # Stochastic oscillator
            for stoch_feature in ['stoch_k', 'stoch_d']:
                if stoch_feature in current_row and not np.isnan(current_row[stoch_feature]):
                    feature_vector = np.append(feature_vector, current_row[stoch_feature])
                    feature_names_i.append(stoch_feature)
            
            # ATR for volatility
            if 'atr' in current_row and not np.isnan(current_row['atr']):
                feature_vector = np.append(feature_vector, current_row['atr'])
                feature_names_i.append('atr')
            
            # Moving averages - use 5, 10, 20 periods
            for period in [5, 10, 20]:
                sma_col = f'sma_{period}'
                if sma_col in current_row and not np.isnan(current_row[sma_col]):
                    # Calculate relative position to current price
                    sma_value = current_row[sma_col]
                    rel_to_sma = (price_window[-1] - sma_value) / sma_value
                    feature_vector = np.append(feature_vector, rel_to_sma)
                    feature_names_i.append(f'rel_to_sma_{period}')
            
            # Add features for two important moving average crossovers
            if 'sma_5' in current_row and 'sma_10' in current_row:
                if not np.isnan(current_row['sma_5']) and not np.isnan(current_row['sma_10']):
                    cross_5_10 = current_row['sma_5'] - current_row['sma_10']
                    feature_vector = np.append(feature_vector, cross_5_10)
                    feature_names_i.append('cross_5_10')
            
            if 'sma_10' in current_row and 'sma_20' in current_row:
                if not np.isnan(current_row['sma_10']) and not np.isnan(current_row['sma_20']):
                    cross_10_20 = current_row['sma_10'] - current_row['sma_20']
                    feature_vector = np.append(feature_vector, cross_10_20)
                    feature_names_i.append('cross_10_20')
            
            # Volume-related features if available
            if 'volume' in df.columns:
                # Use recent volume changes
                recent_volumes = df['volume'].iloc[idx-self.window_size:idx].values
                vol_change = recent_volumes[-1] / np.mean(recent_volumes) - 1
                feature_vector = np.append(feature_vector, vol_change)
                feature_names_i.append('vol_change')
                
                # Add OBV if available
                if 'obv' in current_row and not np.isnan(current_row['obv']):
                    feature_vector = np.append(feature_vector, current_row['obv'])
                    feature_names_i.append('obv')
            
            # Store the feature vector
            features.append(feature_vector)
            
            # Only store feature names once (they're the same for all instances)
            if len(feature_names) == 0:
                feature_names = feature_names_i
        
        return np.array(features), feature_names
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, List[str]]:
        """
        Prepare data for model training by:
        1. Creating features from price data with technical indicators
        2. Creating target variable (price goes up or down)
        3. Splitting into training and testing sets
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            X_train, X_test, y_train, y_test, scaler, feature_names
        """
        # Make sure we have enough data
        if len(df) < self.window_size + 1:
            raise ValueError(f"Not enough data for window size {self.window_size}")
        
        # Create enhanced features
        X, feature_names = self.create_features(df)
        
        # Create target variable (1 if price goes up, 0 if it goes down)
        y = []
        
        # Calculate target based on future price movement
        lookback = max(self.window_size)
        for i in range(len(df) - lookback - 1):
            idx = i + lookback
            current_price = df['close'].iloc[idx]
            next_price = df['close'].iloc[idx + 1]
            y.append(1 if next_price > current_price else 0)
        
        y = np.array(y)
        
        # Ensure X and y have the same number of samples
        if len(X) > len(y):
            X = X[:len(y)]
        elif len(y) > len(X):
            y = y[:len(X)]
        
        # Print feature information
        logger.info(f"Created {X.shape[1]} features per sample")
        logger.info(f"Dataset shape: {X.shape}, Target distribution: {np.bincount(y)}")
        
        # Use TimeSeriesSplit for more appropriate validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Get the train/test split by using the last fold
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
        
        # Scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        logger.info(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test, scaler, feature_names
    
    def plot_feature_importance(self, model, feature_names):
        """Plot and save feature importance"""
        try:
            plt.figure(figsize=(12, 8))
            
            # Get feature importance
            importance = model.feature_importances_
            
            # Sort features by importance
            indices = np.argsort(importance)[::-1]
            
            # Limit to top 30 features for readability
            top_indices = indices[:30]
            top_features = [feature_names[i] for i in top_indices]
            top_importance = importance[top_indices]
            
            # Plot feature importance
            plt.bar(range(len(top_importance)), top_importance)
            plt.xticks(range(len(top_importance)), top_features, rotation=90)
            plt.title('XGBoost Feature Importance (Top 30 Features)')
            plt.xlabel('Features')
            plt.ylabel('Importance')
            plt.tight_layout()
            plt.savefig(self.feature_importance_file)
            logger.info(f"Saved feature importance plot to {self.feature_importance_file}")
        except Exception as e:
            logger.error(f"Failed to plot feature importance: {e}")
    
    def plot_performance_metrics(self, model, X_test, y_test):
        """Plot and save model performance metrics"""
        try:
            # Get predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Create a figure with multiple subplots
            fig, axs = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('XGBoost Model Performance Metrics', fontsize=16)
            
            # Plot ROC curve
            from sklearn.metrics import roc_curve, auc
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            
            axs[0, 0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            axs[0, 0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            axs[0, 0].set_xlim([0.0, 1.0])
            axs[0, 0].set_ylim([0.0, 1.05])
            axs[0, 0].set_xlabel('False Positive Rate')
            axs[0, 0].set_ylabel('True Positive Rate')
            axs[0, 0].set_title('Receiver Operating Characteristic')
            axs[0, 0].legend(loc="lower right")
            
            # Plot precision-recall curve
            from sklearn.metrics import precision_recall_curve, average_precision_score
            precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
            avg_precision = average_precision_score(y_test, y_pred_proba)
            
            axs[0, 1].plot(recall, precision, color='blue', lw=2, label=f'Precision-Recall curve (AP = {avg_precision:.2f})')
            axs[0, 1].set_xlim([0.0, 1.0])
            axs[0, 1].set_ylim([0.0, 1.05])
            axs[0, 1].set_xlabel('Recall')
            axs[0, 1].set_ylabel('Precision')
            axs[0, 1].set_title('Precision-Recall Curve')
            axs[0, 1].legend(loc="lower left")
            
            # Plot confusion matrix
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_test, y_pred)
            
            axs[1, 0].matshow(cm, cmap=plt.cm.Blues)
            axs[1, 0].set_title('Confusion Matrix')
            axs[1, 0].set_xlabel('Predicted')
            axs[1, 0].set_ylabel('Actual')
            
            # Add text annotations
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    axs[1, 0].text(j, i, str(cm[i, j]), ha='center', va='center', color='red')
            
            # Plot F1 score vs. threshold
            f1_scores = []
            for threshold in thresholds:
                y_pred_th = (y_pred_proba >= threshold).astype(int)
                f1 = f1_score(y_test, y_pred_th)
                f1_scores.append(f1)
            
            axs[1, 1].plot(thresholds, f1_scores[:-1], color='green', lw=2)  # Last element of precision/recall has no threshold
            axs[1, 1].set_xlabel('Threshold')
            axs[1, 1].set_ylabel('F1 Score')
            axs[1, 1].set_title('F1 Score vs. Threshold')
            
            # Add best threshold
            best_idx = np.argmax(f1_scores[:-1])
            best_threshold = thresholds[best_idx]
            best_f1 = f1_scores[best_idx]
            
            axs[1, 1].axvline(x=best_threshold, color='r', linestyle='--')
            axs[1, 1].text(best_threshold, 0.2, f'Best threshold: {best_threshold:.2f}\nF1: {best_f1:.2f}', 
                           color='black', bbox=dict(facecolor='white', alpha=0.8))
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout for title
            plt.savefig(self.performance_metrics_file)
            logger.info(f"Saved performance metrics plot to {self.performance_metrics_file}")
            
        except Exception as e:
            logger.error(f"Failed to plot performance metrics: {e}")
    
    def train_model(self, force_retrain=False, use_large_dataset=True, days=30) -> bool:
        """
        Train an XGBoost model and save it to disk
        
        Args:
            force_retrain: Whether to force retraining even if model exists
            use_large_dataset: Whether to use a larger dataset for training
            days: Number of days of historical data to fetch (if use_large_dataset=True)
            
        Returns:
            bool: Whether training was successful
        """
        # Check if model already exists and we're not forcing a retrain
        if self.is_model_trained() and not force_retrain:
            logger.info(f"XGBoost model for {self.symbol} already exists. Use force_retrain=True to retrain.")
            return True
        
        # Fetch historical data
        logger.info(f"Fetching historical data for {self.symbol}...")
        
        if use_large_dataset:
            # Use larger dataset for better training
            end_time = datetime.datetime.now()
            start_time = end_time - datetime.timedelta(days=days)
            
            logger.info(f"Fetching {days} days of historical data from {start_time} to {end_time}")
            df = self.price_fetcher.fetch_large_historical_dataset(
                start_time=start_time,
                end_time=end_time,
                interval="1h"
            )
        else:
            # Use standard API call for less data
            df = self.price_fetcher.fetch_historical_klines(interval="1h", limit=1000)
        
        if df.empty:
            logger.error("Failed to fetch historical data")
            return False
        
        logger.info(f"Successfully fetched {len(df)} candles of historical data")
        
        # Prepare data
        logger.info(f"Preparing data with window size {self.window_size}...")
        try:
            X_train, X_test, y_train, y_test, scaler, feature_names = self.prepare_data(df)
        except ValueError as e:
            logger.error(f"Data preparation failed: {e}")
            return False
        
        # Train XGBoost model
        logger.info("Training XGBoost model...")
        
        # Define model parameters
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'use_label_encoder': False,
            'random_state': 42
        }
        
        # Define parameter grid for GridSearchCV
        from sklearn.model_selection import GridSearchCV
        
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9],
            'min_child_weight': [1, 3, 5],
            'gamma': [0, 0.1, 0.2]
        }
        
        # Create base model
        base_model = xgb.XGBClassifier(**params)
        
        # Use TimeSeriesSplit for cross-validation
        tscv = TimeSeriesSplit(n_splits=3)
        
        # Perform grid search to find best parameters
        logger.info("Performing grid search to find optimal hyperparameters...")
        grid_search = GridSearchCV(
            base_model, 
            param_grid, 
            cv=tscv, 
            n_jobs=-1, 
            verbose=1,
            scoring='roc_auc'
        )
        
        grid_search.fit(X_train, y_train)
        
        # Get the best model
        best_model = grid_search.best_estimator_
        
        # Evaluate on test set
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        # Log results
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Test accuracy: {accuracy:.4f}, AUC: {auc_score:.4f}")
        
        # Classification report
        report = classification_report(y_test, y_pred)
        logger.info(f"\nClassification Report:\n{report}")
        
        # Save feature importance plot
        self.plot_feature_importance(best_model, feature_names)
        
        # Save performance metrics plot
        self.plot_performance_metrics(best_model, X_test, y_test)
        
        # Calculate precision-recall curve for different thresholds
        precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
        
        # Find good threshold options using F1 score
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        best_threshold_idx = np.argmax(f1_scores[:-1])  # Last element of precision doesn't have a threshold
        if best_threshold_idx < len(thresholds):
            best_threshold = thresholds[best_threshold_idx]
        else:
            best_threshold = 0.5
        
        # Prepare metrics for metadata
        metrics = {
            "accuracy": float(accuracy),
            "auc": float(auc_score),
            "best_parameters": grid_search.best_params_,
            "training_samples": len(X_train),
            "test_samples": len(X_test),
            "suggested_threshold": float(best_threshold),
            "class_distribution": {
                "up": int(np.sum(y_train)),
                "down": int(len(y_train) - np.sum(y_train))
            }
        }
        
        # Save model, scaler and metadata
        save_success = self.save_model(best_model, scaler, metrics, feature_names)
        
        return save_success

if __name__ == "__main__":
    # Accept command-line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Train XGBoost crypto price prediction model')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='Trading pair symbol')
    parser.add_argument('--window', type=int, default=5, help='Price window size')
    parser.add_argument('--force', action='store_true', help='Force retraining of model')
    parser.add_argument('--days', type=int, default=30, help='Days of historical data to use')
    args = parser.parse_args()
    
    # Create trainer
    trainer = XGBoostModelTrainer(symbol=args.symbol, window_size=args.window)
    
    # Train model
    success = trainer.train_model(force_retrain=args.force, days=args.days)
    
    if success:
        print(f"Successfully trained XGBoost model for {args.symbol}")
    else:
        print(f"Failed to train XGBoost model for {args.symbol}") 