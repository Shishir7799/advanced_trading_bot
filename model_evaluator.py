import pandas as pd
import numpy as np
import joblib
import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix, 
                            classification_report)
from fetch_crypto_prices import PriceFetcher
from price_prediction_model import PricePredictor
from xgboost_price_predictor import XGBoostPredictor
from technical_indicators import TechnicalIndicators

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("model_evaluation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """
    Class for evaluating and comparing crypto price prediction models.
    """
    
    def __init__(self, symbol: str = "BTCUSDT", 
                 evaluation_path: str = "evaluation_results",
                 rf_threshold: float = 0.5,
                 xgb_threshold: float = 0.5):
        """
        Initialize the model evaluator
        
        Args:
            symbol: Trading pair symbol
            evaluation_path: Path to save evaluation results
            rf_threshold: Confidence threshold for RandomForest model
            xgb_threshold: Confidence threshold for XGBoost model
        """
        self.symbol = symbol
        self.evaluation_path = evaluation_path
        
        # Create evaluation directory if it doesn't exist
        os.makedirs(evaluation_path, exist_ok=True)
        
        # Initialize price fetcher and predictors
        self.price_fetcher = PriceFetcher(symbol=symbol)
        self.rf_predictor = PricePredictor(symbol=symbol, threshold=rf_threshold)
        self.xgb_predictor = XGBoostPredictor(symbol=symbol, threshold=xgb_threshold)
        
        # Initialize results storage
        self.results = {
            'rf': {
                'predictions': [],
                'confidences': [],
                'actual': []
            },
            'xgb': {
                'predictions': [],
                'confidences': [],
                'actual': []
            },
            'ensemble': {
                'predictions': [],
                'method_used': []
            },
            'prices': [],
            'timestamps': []
        }
    
    def fetch_evaluation_data(self, hours: int = 24, interval: str = "1h") -> pd.DataFrame:
        """
        Fetch historical data for model evaluation
        
        Args:
            hours: Number of hours of historical data to fetch
            interval: Candle interval
            
        Returns:
            DataFrame with price data
        """
        logger.info(f"Fetching {hours} hours of {self.symbol} data for evaluation...")
        
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        if hours > 24:
            # For larger datasets, use the larger fetcher
            df = self.price_fetcher.fetch_large_historical_dataset(
                start_time=start_time,
                end_time=end_time,
                interval=interval
            )
        else:
            # For smaller datasets, use standard API
            df = self.price_fetcher.fetch_historical_klines(
                interval=interval,
                limit=hours
            )
        
        if df.empty:
            logger.error("Failed to fetch evaluation data")
            raise ValueError("Empty dataset received from price fetcher")
        
        logger.info(f"Successfully fetched {len(df)} candles for evaluation")
        return df
    
    def evaluate_models(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluate models on historical data
        
        Args:
            df: DataFrame with price data
            
        Returns:
            Dictionary with evaluation results
        """
        logger.info("Starting model evaluation...")
        
        # Reset results
        self.results = {
            'rf': {
                'predictions': [],
                'confidences': [],
                'actual': []
            },
            'xgb': {
                'predictions': [],
                'confidences': [],
                'actual': []
            },
            'ensemble': {
                'predictions': [],
                'method_used': []
            },
            'prices': [],
            'timestamps': []
        }
        
        # Check if models are ready
        if not self.rf_predictor.is_ready():
            logger.error("RandomForest model is not ready")
            return {}
            
        if not self.xgb_predictor.is_ready():
            logger.error("XGBoost model is not ready")
            return {}
        
        # Add technical indicators for richer evaluation context
        enriched_df = TechnicalIndicators.add_all_indicators(df)
        
        # Get the actual price movements (up/down)
        actual_movements = []
        for i in range(len(df) - 1):
            current_price = df['close'].iloc[i]
            next_price = df['close'].iloc[i + 1]
            actual_movements.append(1 if next_price > current_price else 0)
        
        # Store prices and timestamps for plotting
        self.results['prices'] = df['close'].values[:-1]  # Exclude the last price (no next price to compare)
        self.results['timestamps'] = [pd.to_datetime(ts) for ts in df.index.values[:-1]]
        
        # Process each time point
        for i in range(len(df) - 1):  # -1 because we need the next price to know actual movement
            # Create a dataframe with data up to this point
            current_df = df.iloc[:i+1].copy()
            
            # RandomForest prediction
            if len(current_df) >= self.rf_predictor.window_size:
                rf_pred, rf_conf, _ = self.rf_predictor.predict(current_df)
                # Store results
                self.results['rf']['predictions'].append(rf_pred)
                self.results['rf']['confidences'].append(rf_conf)
                self.results['rf']['actual'].append(actual_movements[i])
                
                # XGBoost prediction
                xgb_pred, xgb_conf, _ = self.xgb_predictor.predict(current_df)
                # Store results
                self.results['xgb']['predictions'].append(xgb_pred)
                self.results['xgb']['confidences'].append(xgb_conf)
                self.results['xgb']['actual'].append(actual_movements[i])
                
                # Ensemble prediction
                ensemble_pred = -1  # Default to no prediction
                method_used = 'none'
                
                # If both models agree
                if rf_pred == xgb_pred and rf_pred != -1 and xgb_pred != -1:
                    ensemble_pred = rf_pred
                    method_used = 'agreement'
                # If models disagree, but XGBoost is more confident
                elif rf_pred != xgb_pred and xgb_pred != -1 and xgb_conf > 0.7:
                    ensemble_pred = xgb_pred
                    method_used = 'xgb_confidence'
                
                self.results['ensemble']['predictions'].append(ensemble_pred)
                self.results['ensemble']['method_used'].append(method_used)
        
        # Calculate metrics
        rf_metrics = self._calculate_metrics(
            [p for p, a in zip(self.results['rf']['predictions'], self.results['rf']['actual']) if p != -1],
            [a for p, a in zip(self.results['rf']['predictions'], self.results['rf']['actual']) if p != -1],
            [c for p, c in zip(self.results['rf']['predictions'], self.results['rf']['confidences']) if p != -1],
            'RandomForest'
        )
        
        xgb_metrics = self._calculate_metrics(
            [p for p, a in zip(self.results['xgb']['predictions'], self.results['xgb']['actual']) if p != -1],
            [a for p, a in zip(self.results['xgb']['predictions'], self.results['xgb']['actual']) if p != -1],
            [c for p, c in zip(self.results['xgb']['predictions'], self.results['xgb']['confidences']) if p != -1],
            'XGBoost'
        )
        
        # Ensemble metrics - only include points where ensemble made a prediction
        ensemble_predictions = []
        ensemble_actuals = []
        
        for i, pred in enumerate(self.results['ensemble']['predictions']):
            if pred != -1:
                ensemble_predictions.append(pred)
                ensemble_actuals.append(self.results['rf']['actual'][i])  # Both rf and xgb have same actuals
        
        ensemble_metrics = self._calculate_metrics(
            ensemble_predictions,
            ensemble_actuals,
            None,  # No confidences for ensemble
            'Ensemble'
        )
        
        # Additional ensemble statistics
        ensemble_stats = {
            'total_predictions': len(self.results['ensemble']['predictions']),
            'predictions_made': len(ensemble_predictions),
            'prediction_rate': len(ensemble_predictions) / len(self.results['ensemble']['predictions']) if len(self.results['ensemble']['predictions']) > 0 else 0,
            'method_counts': {
                'agreement': self.results['ensemble']['method_used'].count('agreement'),
                'xgb_confidence': self.results['ensemble']['method_used'].count('xgb_confidence'),
                'none': self.results['ensemble']['method_used'].count('none')
            }
        }
        
        # Combine all results
        evaluation_results = {
            'rf_metrics': rf_metrics,
            'xgb_metrics': xgb_metrics,
            'ensemble_metrics': ensemble_metrics,
            'ensemble_stats': ensemble_stats,
            'symbol': self.symbol,
            'evaluation_time': datetime.now().isoformat(),
            'data_points': len(df) - 1,
            'rf_threshold': self.rf_predictor.threshold,
            'xgb_threshold': self.xgb_predictor.threshold
        }
        
        logger.info("Model evaluation completed")
        logger.info(f"RandomForest accuracy: {rf_metrics['accuracy']:.2f}")
        logger.info(f"XGBoost accuracy: {xgb_metrics['accuracy']:.2f}")
        logger.info(f"Ensemble accuracy: {ensemble_metrics['accuracy']:.2f}")
        
        # Generate plots
        self._plot_results(evaluation_results)
        
        return evaluation_results
    
    def _calculate_metrics(self, predictions: List[int], actual: List[int], 
                         confidences: Optional[List[float]] = None,
                         model_name: str = 'Model') -> Dict[str, Any]:
        """
        Calculate performance metrics
        
        Args:
            predictions: List of model predictions (0, 1)
            actual: List of actual values (0, 1)
            confidences: List of prediction confidences
            model_name: Name of the model for logging
            
        Returns:
            Dictionary with metrics
        """
        if not predictions or len(predictions) == 0:
            logger.warning(f"{model_name}: No predictions to calculate metrics")
            return {
                'accuracy': 0,
                'precision': 0,
                'recall': 0,
                'f1': 0,
                'predictions_made': 0
            }
        
        try:
            # Basic metrics
            accuracy = accuracy_score(actual, predictions)
            precision = precision_score(actual, predictions, zero_division=0)
            recall = recall_score(actual, predictions, zero_division=0)
            f1 = f1_score(actual, predictions, zero_division=0)
            
            # Detailed metrics by class
            cm = confusion_matrix(actual, predictions)
            tn, fp, fn, tp = cm.ravel()
            
            # Calculate class-specific metrics
            metrics = {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'confusion_matrix': {
                    'tn': int(tn),
                    'fp': int(fp),
                    'fn': int(fn),
                    'tp': int(tp)
                },
                'predictions_made': len(predictions),
                'class_metrics': {
                    'up': {
                        'precision': float(tp / (tp + fp)) if (tp + fp) > 0 else 0,
                        'recall': float(tp / (tp + fn)) if (tp + fn) > 0 else 0,
                        'count': int(tp + fn)
                    },
                    'down': {
                        'precision': float(tn / (tn + fn)) if (tn + fn) > 0 else 0,
                        'recall': float(tn / (tn + fp)) if (tn + fp) > 0 else 0,
                        'count': int(tn + fp)
                    }
                }
            }
            
            # Add confidence-based metrics if available
            if confidences and len(confidences) == len(predictions):
                # Average confidence
                metrics['avg_confidence'] = float(np.mean(confidences))
                
                # Average confidence for correct & incorrect predictions
                correct_indices = [i for i, (p, a) in enumerate(zip(predictions, actual)) if p == a]
                incorrect_indices = [i for i, (p, a) in enumerate(zip(predictions, actual)) if p != a]
                
                correct_confidences = [confidences[i] for i in correct_indices] if correct_indices else []
                incorrect_confidences = [confidences[i] for i in incorrect_indices] if incorrect_indices else []
                
                metrics['correct_predictions'] = {
                    'count': len(correct_indices),
                    'avg_confidence': float(np.mean(correct_confidences)) if correct_confidences else 0
                }
                
                metrics['incorrect_predictions'] = {
                    'count': len(incorrect_indices),
                    'avg_confidence': float(np.mean(incorrect_confidences)) if incorrect_confidences else 0
                }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics for {model_name}: {e}")
            return {
                'error': str(e),
                'predictions_made': len(predictions)
            }
    
    def _plot_results(self, results: Dict[str, Any]) -> None:
        """
        Generate evaluation plots
        
        Args:
            results: Dictionary with evaluation results
        """
        try:
            # Create directory for plots
            plots_dir = os.path.join(self.evaluation_path, 'plots')
            os.makedirs(plots_dir, exist_ok=True)
            
            # Plot 1: Accuracy Comparison
            plt.figure(figsize=(10, 6))
            models = ['RandomForest', 'XGBoost', 'Ensemble']
            accuracies = [
                results['rf_metrics']['accuracy'],
                results['xgb_metrics']['accuracy'],
                results['ensemble_metrics']['accuracy']
            ]
            
            # Color bars based on performance
            colors = ['#ff9999', '#66b3ff', '#99ff99']
            if accuracies[2] > max(accuracies[0], accuracies[1]):
                # Ensemble is best
                colors = ['#ff9999', '#66b3ff', '#50C878']
            elif accuracies[1] > accuracies[0]:
                # XGBoost is best
                colors = ['#ff9999', '#4682B4', '#99ff99']
            else:
                # RandomForest is best
                colors = ['#B22222', '#66b3ff', '#99ff99']
            
            bars = plt.bar(models, accuracies, color=colors)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.2f}', ha='center', va='bottom')
            
            plt.ylim(0, 1.0)
            plt.title(f'Model Accuracy Comparison - {self.symbol}')
            plt.ylabel('Accuracy')
            plt.savefig(os.path.join(plots_dir, f'{self.symbol}_accuracy_comparison.png'))
            
            # Plot 2: Confusion Matrices
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            
            # RandomForest confusion matrix
            cm_rf = np.array([
                [results['rf_metrics']['confusion_matrix']['tn'], results['rf_metrics']['confusion_matrix']['fp']],
                [results['rf_metrics']['confusion_matrix']['fn'], results['rf_metrics']['confusion_matrix']['tp']]
            ])
            
            sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                       xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'])
            axes[0].set_title('RandomForest Confusion Matrix')
            axes[0].set_ylabel('Actual')
            axes[0].set_xlabel('Predicted')
            
            # XGBoost confusion matrix
            cm_xgb = np.array([
                [results['xgb_metrics']['confusion_matrix']['tn'], results['xgb_metrics']['confusion_matrix']['fp']],
                [results['xgb_metrics']['confusion_matrix']['fn'], results['xgb_metrics']['confusion_matrix']['tp']]
            ])
            
            sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Blues', ax=axes[1],
                       xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'])
            axes[1].set_title('XGBoost Confusion Matrix')
            axes[1].set_ylabel('Actual')
            axes[1].set_xlabel('Predicted')
            
            # Ensemble confusion matrix
            cm_ens = np.array([
                [results['ensemble_metrics']['confusion_matrix']['tn'], results['ensemble_metrics']['confusion_matrix']['fp']],
                [results['ensemble_metrics']['confusion_matrix']['fn'], results['ensemble_metrics']['confusion_matrix']['tp']]
            ])
            
            sns.heatmap(cm_ens, annot=True, fmt='d', cmap='Blues', ax=axes[2],
                       xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'])
            axes[2].set_title('Ensemble Confusion Matrix')
            axes[2].set_ylabel('Actual')
            axes[2].set_xlabel('Predicted')
            
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f'{self.symbol}_confusion_matrices.png'))
            
            # Plot 3: Prediction timeline
            plt.figure(figsize=(14, 8))
            
            # Price movement
            plt.subplot(211)
            plt.plot(self.results['timestamps'], self.results['prices'], 'k-', label='Price')
            plt.title(f'{self.symbol} Price Movement')
            plt.ylabel('Price')
            plt.legend()
            
            # Prediction results
            plt.subplot(212)
            
            # Prepare data for visualization
            correct_rf_times = []
            correct_rf_preds = []
            incorrect_rf_times = []
            incorrect_rf_preds = []
            
            correct_xgb_times = []
            correct_xgb_preds = []
            incorrect_xgb_times = []
            incorrect_xgb_preds = []
            
            correct_ens_times = []
            correct_ens_preds = []
            incorrect_ens_times = []
            incorrect_ens_preds = []
            
            for i in range(len(self.results['rf']['predictions'])):
                # RandomForest
                if self.results['rf']['predictions'][i] != -1:
                    if self.results['rf']['predictions'][i] == self.results['rf']['actual'][i]:
                        correct_rf_times.append(self.results['timestamps'][i])
                        correct_rf_preds.append(self.results['rf']['predictions'][i])
                    else:
                        incorrect_rf_times.append(self.results['timestamps'][i])
                        incorrect_rf_preds.append(self.results['rf']['predictions'][i])
                
                # XGBoost
                if self.results['xgb']['predictions'][i] != -1:
                    if self.results['xgb']['predictions'][i] == self.results['xgb']['actual'][i]:
                        correct_xgb_times.append(self.results['timestamps'][i])
                        correct_xgb_preds.append(self.results['xgb']['predictions'][i])
                    else:
                        incorrect_xgb_times.append(self.results['timestamps'][i])
                        incorrect_xgb_preds.append(self.results['xgb']['predictions'][i])
                
                # Ensemble
                if i < len(self.results['ensemble']['predictions']) and self.results['ensemble']['predictions'][i] != -1:
                    if self.results['ensemble']['predictions'][i] == self.results['rf']['actual'][i]:
                        correct_ens_times.append(self.results['timestamps'][i])
                        correct_ens_preds.append(self.results['ensemble']['predictions'][i])
                    else:
                        incorrect_ens_times.append(self.results['timestamps'][i])
                        incorrect_ens_preds.append(self.results['ensemble']['predictions'][i])
            
            # Plot correct predictions as green, incorrect as red
            plt.scatter(correct_rf_times, [0.2 + p*0.1 for p in correct_rf_preds], color='green', marker='o', s=30, label='RF Correct')
            plt.scatter(incorrect_rf_times, [0.2 + p*0.1 for p in incorrect_rf_preds], color='red', marker='o', s=30, label='RF Incorrect')
            
            plt.scatter(correct_xgb_times, [0.4 + p*0.1 for p in correct_xgb_preds], color='green', marker='s', s=30, label='XGB Correct')
            plt.scatter(incorrect_xgb_times, [0.4 + p*0.1 for p in incorrect_xgb_preds], color='red', marker='s', s=30, label='XGB Incorrect')
            
            plt.scatter(correct_ens_times, [0.6 + p*0.1 for p in correct_ens_preds], color='green', marker='^', s=45, label='Ensemble Correct')
            plt.scatter(incorrect_ens_times, [0.6 + p*0.1 for p in incorrect_ens_preds], color='red', marker='^', s=45, label='Ensemble Incorrect')
            
            plt.yticks([0.2, 0.3, 0.4, 0.5, 0.6, 0.7], ['RF Down', 'RF Up', 'XGB Down', 'XGB Up', 'Ensemble Down', 'Ensemble Up'])
            plt.title('Model Predictions')
            plt.xlabel('Time')
            plt.legend(loc='upper right')
            
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f'{self.symbol}_prediction_timeline.png'))
            
            # Plot 4: Ensemble method distribution
            plt.figure(figsize=(10, 6))
            methods = ['Agreement', 'XGBoost Confidence', 'No Prediction']
            counts = [
                results['ensemble_stats']['method_counts']['agreement'],
                results['ensemble_stats']['method_counts']['xgb_confidence'],
                results['ensemble_stats']['method_counts']['none']
            ]
            
            plt.pie(counts, labels=methods, autopct='%1.1f%%', colors=['#99ff99', '#66b3ff', '#ff9999'])
            plt.title('Ensemble Prediction Methods')
            plt.savefig(os.path.join(plots_dir, f'{self.symbol}_ensemble_methods.png'))
            
            logger.info(f"Saved evaluation plots to {plots_dir}")
            
        except Exception as e:
            logger.error(f"Error generating plots: {e}")
    
    def run_evaluation(self, hours: int = 24, save_results: bool = True) -> Dict[str, Any]:
        """
        Run a complete model evaluation
        
        Args:
            hours: Number of hours of historical data to evaluate
            save_results: Whether to save results to disk
            
        Returns:
            Dictionary with evaluation results
        """
        # Fetch data
        df = self.fetch_evaluation_data(hours=hours)
        
        # Evaluate models
        results = self.evaluate_models(df)
        
        # Save results if requested
        if save_results and results:
            import json
            
            # Ensure timestamps are serializable
            if 'timestamps' in self.results:
                self.results['timestamps'] = [ts.isoformat() for ts in self.results['timestamps']]
            
            # Create file path
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = os.path.join(
                self.evaluation_path, 
                f"{self.symbol}_evaluation_{timestamp}.json"
            )
            
            # Save results
            with open(results_file, 'w') as f:
                json.dump({
                    'metrics': results,
                    'raw_data': self.results
                }, f, indent=2)
            
            logger.info(f"Saved evaluation results to {results_file}")
        
        return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate crypto price prediction models')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='Trading pair symbol')
    parser.add_argument('--hours', type=int, default=24, help='Hours of historical data to evaluate')
    parser.add_argument('--rf-threshold', type=float, default=0.5, help='Confidence threshold for RandomForest')
    parser.add_argument('--xgb-threshold', type=float, default=0.5, help='Confidence threshold for XGBoost')
    args = parser.parse_args()
    
    evaluator = ModelEvaluator(
        symbol=args.symbol,
        rf_threshold=args.rf_threshold,
        xgb_threshold=args.xgb_threshold
    )
    
    results = evaluator.run_evaluation(hours=args.hours)
    
    if results:
        print("\n--- Model Evaluation Results ---")
        print(f"RandomForest accuracy: {results['rf_metrics']['accuracy']:.4f}")
        print(f"XGBoost accuracy: {results['xgb_metrics']['accuracy']:.4f}")
        print(f"Ensemble accuracy: {results['ensemble_metrics']['accuracy']:.4f}")
        
        print("\nEnsemble statistics:")
        print(f"Total predictions possible: {results['ensemble_stats']['total_predictions']}")
        print(f"Predictions made: {results['ensemble_stats']['predictions_made']}")
        print(f"Prediction rate: {results['ensemble_stats']['prediction_rate']:.2f}")
        
        print("\nEnsemble methods:")
        print(f"Agreement: {results['ensemble_stats']['method_counts']['agreement']}")
        print(f"XGBoost confidence: {results['ensemble_stats']['method_counts']['xgb_confidence']}")
        print(f"No prediction: {results['ensemble_stats']['method_counts']['none']}")
        
        print("\nDetailed metrics available in the evaluation_results directory.")
    else:
        print("Evaluation failed. Check the logs for details.") 