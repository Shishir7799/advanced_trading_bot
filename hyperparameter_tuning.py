import pandas as pd
import numpy as np
import os
import logging
import joblib
import json
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Any, Optional
from fetch_crypto_prices import PriceFetcher
from technical_indicators import TechnicalIndicators

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("hyperparameter_tuning.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ModelHyperparameterTuner:
    """Class for advanced hyperparameter tuning of crypto trading models."""
    
    def __init__(self, symbol: str = "BTCUSDT", window_size: int = 5, 
                 model_path: str = "models", results_path: str = "tuning_results"):
        """
        Initialize the tuner
        
        Args:
            symbol: Trading pair symbol
            window_size: Size of price window for feature creation
            model_path: Path to save models
            results_path: Path to save tuning results
        """
        self.symbol = symbol
        self.window_size = window_size
        self.model_path = model_path
        self.results_path = results_path
        
        # Create directories if they don't exist
        os.makedirs(model_path, exist_ok=True)
        os.makedirs(results_path, exist_ok=True)
        
        # Result file paths
        self.xgboost_results_file = os.path.join(results_path, f"{symbol.lower()}_xgboost_tuning.json")
        self.xgboost_plot_file = os.path.join(results_path, f"{symbol.lower()}_xgboost_tuning.png")
        self.rf_results_file = os.path.join(results_path, f"{symbol.lower()}_rf_tuning.json")
        self.rf_plot_file = os.path.join(results_path, f"{symbol.lower()}_rf_tuning.png")
        
        # Initialize price fetcher
        self.price_fetcher = PriceFetcher(symbol=symbol)
    
    def fetch_data(self, days: int = 60, interval: str = "1h") -> pd.DataFrame:
        """
        Fetch historical price data for tuning
        
        Args:
            days: Number of days of historical data to fetch
            interval: Time interval for candles
            
        Returns:
            DataFrame with price data
        """
        logger.info(f"Fetching {days} days of historical {self.symbol} data...")
        
        # Fetch larger dataset for better tuning
        from datetime import datetime, timedelta
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        df = self.price_fetcher.fetch_large_historical_dataset(
            start_time=start_time,
            end_time=end_time,
            interval=interval
        )
        
        if df.empty:
            logger.error("Failed to fetch historical data")
            raise ValueError("Empty dataset received from price fetcher")
        
        logger.info(f"Successfully fetched {len(df)} candles of historical data")
        return df
    
    def create_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Create features and target for model tuning
        
        Args:
            df: DataFrame with price data
            
        Returns:
            Features array, target array, and feature names
        """
        logger.info("Creating features with technical indicators...")
        
        # Add technical indicators
        enriched_df = TechnicalIndicators.add_all_indicators(df)
        
        # Initialize containers
        features = []
        feature_names = []
        
        # Create feature matrix
        for i in range(len(enriched_df) - self.window_size - 1):
            idx = i + self.window_size
            
            # Basic price features from window
            price_window = df['close'].iloc[idx-self.window_size:idx].values
            feature_vector = price_window.copy()
            feature_names_i = [f'price_{j}' for j in range(self.window_size)]
            
            # Add momentum features
            price_momentum = np.diff(price_window)
            feature_vector = np.append(feature_vector, price_momentum)
            feature_names_i.extend([f'momentum_{j}' for j in range(len(price_momentum))])
            
            # Add rolling statistics
            rolling_mean = np.mean(price_window)
            rolling_std = np.std(price_window)
            feature_vector = np.append(feature_vector, [rolling_mean, rolling_std])
            feature_names_i.extend(['rolling_mean', 'rolling_std'])
            
            # Z-score
            if rolling_std != 0:
                z_score = (price_window[-1] - rolling_mean) / rolling_std
            else:
                z_score = 0
            feature_vector = np.append(feature_vector, z_score)
            feature_names_i.append('z_score')
            
            # Relative position
            price_range = np.max(price_window) - np.min(price_window)
            if price_range != 0:
                rel_position = (price_window[-1] - np.min(price_window)) / price_range
            else:
                rel_position = 0.5
            feature_vector = np.append(feature_vector, rel_position)
            feature_names_i.append('rel_position')
            
            # Add technical indicators
            current_row = enriched_df.iloc[idx]
            
            # Technical indicators to include
            tech_indicators = [
                'rsi', 'macd_line', 'macd_signal', 'macd_histogram', 
                'bb_pct_b', 'bb_bandwidth', 'stoch_k', 'stoch_d', 'atr'
            ]
            
            for indicator in tech_indicators:
                if indicator in current_row and not np.isnan(current_row[indicator]):
                    feature_vector = np.append(feature_vector, current_row[indicator])
                    feature_names_i.append(indicator)
            
            # Moving averages
            for period in [5, 10, 20]:
                sma_col = f'sma_{period}'
                if sma_col in current_row and not np.isnan(current_row[sma_col]):
                    sma_value = current_row[sma_col]
                    rel_to_sma = (price_window[-1] - sma_value) / sma_value
                    feature_vector = np.append(feature_vector, rel_to_sma)
                    feature_names_i.append(f'rel_to_sma_{period}')
            
            # Moving average crossovers
            for pair in [('sma_5', 'sma_10'), ('sma_10', 'sma_20')]:
                if pair[0] in current_row and pair[1] in current_row:
                    if not np.isnan(current_row[pair[0]]) and not np.isnan(current_row[pair[1]]):
                        cross = current_row[pair[0]] - current_row[pair[1]]
                        feature_vector = np.append(feature_vector, cross)
                        feature_names_i.append(f'cross_{pair[0]}_{pair[1]}')
            
            # Volume features
            if 'volume' in df.columns:
                recent_volumes = df['volume'].iloc[idx-self.window_size:idx].values
                vol_change = recent_volumes[-1] / np.mean(recent_volumes) - 1
                feature_vector = np.append(feature_vector, vol_change)
                feature_names_i.append('vol_change')
                
                if 'obv' in current_row and not np.isnan(current_row['obv']):
                    feature_vector = np.append(feature_vector, current_row['obv'])
                    feature_names_i.append('obv')
            
            features.append(feature_vector)
            
            # Only store feature names once
            if len(feature_names) == 0:
                feature_names = feature_names_i
        
        # Create target variable (1 if price goes up, 0 if down)
        y = []
        for i in range(len(df) - self.window_size - 1):
            idx = i + self.window_size
            current_price = df['close'].iloc[idx]
            next_price = df['close'].iloc[idx + 1]
            y.append(1 if next_price > current_price else 0)
        
        X = np.array(features)
        y = np.array(y)
        
        # Ensure X and y have the same length
        min_len = min(len(X), len(y))
        X = X[:min_len]
        y = y[:min_len]
        
        logger.info(f"Created dataset with {X.shape[1]} features and {len(X)} samples")
        logger.info(f"Target distribution: {np.bincount(y)}")
        
        return X, y, feature_names
    
    def tune_xgboost(self, X: np.ndarray, y: np.ndarray, n_iter: int = 50) -> Dict[str, Any]:
        """
        Tune XGBoost hyperparameters
        
        Args:
            X: Feature matrix
            y: Target variable
            n_iter: Number of parameter settings to try
            
        Returns:
            Dictionary with tuning results
        """
        logger.info("Starting XGBoost hyperparameter tuning...")
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [100, 200, 300, 500, 1000],
            'max_depth': [3, 4, 5, 6, 7, 8, 10],
            'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2],
            'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'min_child_weight': [1, 3, 5, 7, 9],
            'gamma': [0, 0.1, 0.2, 0.3, 0.4, 0.5],
            'reg_alpha': [0, 0.1, 0.5, 1.0, 5.0, 10.0],
            'reg_lambda': [0, 0.1, 0.5, 1.0, 5.0, 10.0],
            'scale_pos_weight': [1, 2, 3, 5]  # Handle imbalanced classes
        }
        
        # Base model
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            use_label_encoder=False,
            random_state=42
        )
        
        # Time series-based cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Randomized search
        search = RandomizedSearchCV(
            model,
            param_distributions=param_grid,
            n_iter=n_iter,
            scoring='roc_auc',
            cv=tscv,
            verbose=2,
            random_state=42,
            n_jobs=-1
        )
        
        search.fit(X_scaled, y)
        
        # Get results
        best_params = search.best_params_
        best_score = search.best_score_
        
        # Save results
        results = {
            'model_type': 'XGBoost',
            'best_params': best_params,
            'best_score': float(best_score),
            'all_results': [
                {
                    'params': params,
                    'score': float(score),
                    'rank': rank
                }
                for params, score, rank in zip(
                    search.cv_results_['params'],
                    search.cv_results_['mean_test_score'],
                    search.cv_results_['rank_test_score']
                )
            ]
        }
        
        # Save results to file
        with open(self.xgboost_results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create visualization
        self._plot_tuning_results(
            search.cv_results_,
            'n_estimators',
            'max_depth',
            title='XGBoost Hyperparameter Tuning',
            save_path=self.xgboost_plot_file
        )
        
        logger.info(f"Best XGBoost parameters: {best_params}")
        logger.info(f"Best ROC AUC score: {best_score:.4f}")
        
        return results
    
    def tune_random_forest(self, X: np.ndarray, y: np.ndarray, n_iter: int = 50) -> Dict[str, Any]:
        """
        Tune RandomForest hyperparameters
        
        Args:
            X: Feature matrix
            y: Target variable
            n_iter: Number of parameter settings to try
            
        Returns:
            Dictionary with tuning results
        """
        logger.info("Starting RandomForest hyperparameter tuning...")
        
        from sklearn.ensemble import RandomForestClassifier
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [100, 200, 300, 500, 1000],
            'max_depth': [None, 5, 10, 15, 20, 30],
            'min_samples_split': [2, 5, 10, 15, 20],
            'min_samples_leaf': [1, 2, 4, 8],
            'max_features': ['sqrt', 'log2', None, 0.7, 0.8],
            'bootstrap': [True, False],
            'class_weight': [None, 'balanced', 'balanced_subsample']
        }
        
        # Base model
        model = RandomForestClassifier(random_state=42)
        
        # Time series-based cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Randomized search
        search = RandomizedSearchCV(
            model,
            param_distributions=param_grid,
            n_iter=n_iter,
            scoring='roc_auc',
            cv=tscv,
            verbose=2,
            random_state=42,
            n_jobs=-1
        )
        
        search.fit(X_scaled, y)
        
        # Get results
        best_params = search.best_params_
        best_score = search.best_score_
        
        # Save results
        results = {
            'model_type': 'RandomForest',
            'best_params': best_params,
            'best_score': float(best_score),
            'all_results': [
                {
                    'params': {k: (str(v) if k == 'max_features' and v is None else v) for k, v in params.items()},
                    'score': float(score),
                    'rank': rank
                }
                for params, score, rank in zip(
                    search.cv_results_['params'],
                    search.cv_results_['mean_test_score'],
                    search.cv_results_['rank_test_score']
                )
            ]
        }
        
        # Save results to file
        with open(self.rf_results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create visualization
        self._plot_tuning_results(
            search.cv_results_,
            'n_estimators',
            'min_samples_split',
            title='RandomForest Hyperparameter Tuning',
            save_path=self.rf_plot_file
        )
        
        logger.info(f"Best RandomForest parameters: {best_params}")
        logger.info(f"Best ROC AUC score: {best_score:.4f}")
        
        return results
    
    def _plot_tuning_results(self, cv_results: Dict[str, Any], x_param: str, y_param: str, 
                           title: str, save_path: str) -> None:
        """
        Plot hyperparameter tuning results
        
        Args:
            cv_results: Cross-validation results
            x_param: Parameter to plot on x-axis
            y_param: Parameter to plot on y-axis
            title: Plot title
            save_path: Path to save the plot
        """
        try:
            plt.figure(figsize=(12, 8))
            
            # Get parameter values
            x_values = cv_results['param_' + x_param].data
            y_values = cv_results['param_' + y_param].data
            scores = cv_results['mean_test_score']
            
            # Convert to numeric if possible
            try:
                x_numeric = [float(x) if x is not None else 0 for x in x_values]
                y_numeric = [float(y) if y is not None else 0 for y in y_values]
                
                # Create scatter plot
                sc = plt.scatter(x_numeric, y_numeric, c=scores, cmap='viridis', 
                                s=100, alpha=0.8, edgecolors='black')
                
                plt.colorbar(sc, label='ROC AUC Score')
                plt.xlabel(x_param)
                plt.ylabel(y_param)
                plt.title(title)
                
                # Add best parameter combination
                best_idx = np.argmax(scores)
                plt.scatter([x_numeric[best_idx]], [y_numeric[best_idx]], 
                           c='red', s=200, alpha=0.5, marker='*', 
                           edgecolors='black', label='Best Parameters')
                
                plt.legend()
                plt.tight_layout()
                plt.savefig(save_path)
                logger.info(f"Saved tuning plot to {save_path}")
                
            except (ValueError, TypeError):
                logger.warning(f"Could not convert parameters to numeric values for plotting")
                # Just save the scores distribution instead
                plt.figure(figsize=(10, 6))
                plt.hist(scores, bins=20, alpha=0.7)
                plt.xlabel('ROC AUC Score')
                plt.ylabel('Frequency')
                plt.title(f'{title} - Score Distribution')
                plt.savefig(save_path)
                logger.info(f"Saved score distribution to {save_path}")
                
        except Exception as e:
            logger.error(f"Error plotting tuning results: {e}")
    
    def run_hyperparameter_tuning(self, days: int = 60, n_iter: int = 50) -> Dict[str, Any]:
        """
        Run the complete hyperparameter tuning process
        
        Args:
            days: Number of days of historical data to use
            n_iter: Number of parameter settings to try
            
        Returns:
            Dictionary with tuning results
        """
        # Fetch data
        df = self.fetch_data(days=days)
        
        # Create features
        X, y, feature_names = self.create_features(df)
        
        # Tune XGBoost
        xgb_results = self.tune_xgboost(X, y, n_iter=n_iter)
        
        # Tune RandomForest
        rf_results = self.tune_random_forest(X, y, n_iter=n_iter)
        
        # Return combined results
        return {
            'xgboost': xgb_results,
            'random_forest': rf_results,
            'data_info': {
                'symbol': self.symbol,
                'days': days,
                'samples': len(X),
                'features': len(feature_names),
                'feature_names': feature_names
            }
        }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Hyperparameter tuning for trading models')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='Trading pair symbol')
    parser.add_argument('--days', type=int, default=60, help='Days of historical data to use')
    parser.add_argument('--window', type=int, default=5, help='Price window size for features')
    parser.add_argument('--iterations', type=int, default=50, help='Number of parameter combinations to try')
    args = parser.parse_args()
    
    tuner = ModelHyperparameterTuner(
        symbol=args.symbol,
        window_size=args.window
    )
    
    results = tuner.run_hyperparameter_tuning(
        days=args.days,
        n_iter=args.iterations
    )
    
    print("\n--- Hyperparameter Tuning Results ---")
    print(f"XGBoost best score: {results['xgboost']['best_score']:.4f}")
    print(f"RandomForest best score: {results['random_forest']['best_score']:.4f}")
    print("\nBest parameters saved to:")
    print(f"- {tuner.xgboost_results_file}")
    print(f"- {tuner.rf_results_file}")
    print("\nVisualization plots saved to:")
    print(f"- {tuner.xgboost_plot_file}")
    print(f"- {tuner.rf_plot_file}")
    
    # Print winner
    if results['xgboost']['best_score'] > results['random_forest']['best_score']:
        print("\nXGBoost outperformed RandomForest")
    else:
        print("\nRandomForest outperformed XGBoost") 