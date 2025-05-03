import logging
import pandas as pd
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import argparse
from typing import List, Dict, Any, Optional

from fetch_crypto_prices import PriceFetcher
from price_prediction_model import PricePredictor
from xgboost_price_predictor import XGBoostPredictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelComparison:
    def __init__(self, symbol: str = "BTCUSDT", window_size: int = 5):
        self.symbol = symbol
        self.window_size = window_size
        
        # Create output directories
        os.makedirs("results", exist_ok=True)
        os.makedirs("plots", exist_ok=True)
        
        # Initialize components
        self.price_fetcher = PriceFetcher(symbol=symbol)
        self.rf_predictor = PricePredictor(symbol=symbol, window_size=window_size)
        self.xgb_predictor = XGBoostPredictor(symbol=symbol, window_size=window_size)
        
        # Check which models are available
        self.rf_available = self.rf_predictor.is_model_ready()
        self.xgb_available = self.xgb_predictor.is_model_ready()
        
        # Store results
        self.results = []
        
        logger.info(f"Model comparison initialized for {symbol}")
        logger.info(f"RandomForest model available: {self.rf_available}")
        logger.info(f"XGBoost model available: {self.xgb_available}")
    
    def get_test_dataset(self, hours: int = 48) -> pd.DataFrame:
        """
        Get a test dataset of recent prices
        
        Args:
            hours: Number of hours of data to fetch
            
        Returns:
            DataFrame with price data
        """
        logger.info(f"Fetching {hours} hours of recent price data...")
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        df = self.price_fetcher.fetch_large_historical_dataset(
            start_time=start_time,
            end_time=end_time,
            interval="15m"  # Use 15-minute intervals for more granular testing
        )
        
        if df.empty:
            logger.error("Failed to fetch test data")
            return pd.DataFrame()
        
        logger.info(f"Successfully fetched {len(df)} candles of test data")
        return df
    
    def evaluate_models_on_data(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Evaluate models on historical data
        
        Args:
            df: DataFrame with price data
            
        Returns:
            List of evaluation results
        """
        if df.empty:
            logger.error("Empty dataframe provided for evaluation")
            return []
        
        results = []
        prices = df['close'].values
        
        # Need at least window_size + 1 prices for evaluation
        if len(prices) <= self.window_size:
            logger.error(f"Not enough prices for window size {self.window_size}")
            return []
        
        logger.info(f"Evaluating models on {len(prices) - self.window_size} test points...")
        
        # For each possible window in the data
        for i in range(len(prices) - self.window_size):
            # Create price window
            price_window = prices[i:i+self.window_size].tolist()
            actual_next_price = prices[i+self.window_size]
            
            timestamp = df['open_time'].iloc[i+self.window_size]
            
            # Create result dict with time and price info
            result = {
                "timestamp": timestamp,
                "window_start_price": price_window[0],
                "window_end_price": price_window[-1],
                "next_price": actual_next_price,
                "actual_direction": "UP" if actual_next_price > price_window[-1] else "DOWN",
                "price_change": actual_next_price - price_window[-1],
                "percent_change": ((actual_next_price - price_window[-1]) / price_window[-1]) * 100
            }
            
            # RandomForest prediction
            if self.rf_available:
                rf_eval = self.rf_predictor.evaluate_prediction(price_window, actual_next_price)
                result["rf_prediction"] = rf_eval["prediction"]
                result["rf_confidence"] = rf_eval["confidence"]
                result["rf_correct"] = rf_eval["correct"]
            
            # XGBoost prediction
            if self.xgb_available:
                xgb_eval = self.xgb_predictor.evaluate_prediction(price_window, actual_next_price)
                result["xgb_prediction"] = xgb_eval["prediction"]
                result["xgb_confidence"] = xgb_eval["confidence"]
                result["xgb_correct"] = xgb_eval["correct"]
            
            # Ensemble: take the prediction with higher confidence
            if self.rf_available and self.xgb_available:
                # If both models agree, take the higher confidence one
                if result["rf_prediction"] == result["xgb_prediction"]:
                    ensemble_prediction = result["rf_prediction"]
                    ensemble_confidence = max(result["rf_confidence"], result["xgb_confidence"])
                else:
                    # If they disagree, take the one with higher confidence
                    if result["rf_confidence"] > result["xgb_confidence"]:
                        ensemble_prediction = result["rf_prediction"]
                        ensemble_confidence = result["rf_confidence"]
                    else:
                        ensemble_prediction = result["xgb_prediction"]
                        ensemble_confidence = result["xgb_confidence"]
                
                result["ensemble_prediction"] = ensemble_prediction
                result["ensemble_confidence"] = ensemble_confidence
                result["ensemble_correct"] = ensemble_prediction == result["actual_direction"]
            
            results.append(result)
        
        return results
    
    def run_comparison(self, hours: int = 48) -> pd.DataFrame:
        """
        Run a complete model comparison
        
        Args:
            hours: Number of hours of data to use for testing
            
        Returns:
            DataFrame with comparison results
        """
        # Get test data
        df = self.get_test_dataset(hours=hours)
        if df.empty:
            return pd.DataFrame()
        
        # Evaluate models
        results = self.evaluate_models_on_data(df)
        if not results:
            return pd.DataFrame()
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Calculate summary statistics
        self._calculate_stats(results_df)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = f"results/{self.symbol}_model_comparison_{timestamp}.csv"
        results_df.to_csv(results_path, index=False)
        logger.info(f"Saved results to {results_path}")
        
        # Generate plots
        self._generate_plots(results_df, timestamp)
        
        return results_df
    
    def _calculate_stats(self, df: pd.DataFrame) -> None:
        """Calculate and log summary statistics"""
        logger.info("\n=== Model Comparison Summary ===")
        
        # Number of test points
        total_tests = len(df)
        logger.info(f"Total test points: {total_tests}")
        
        # Actual price movements
        up_count = (df['actual_direction'] == 'UP').sum()
        down_count = total_tests - up_count
        logger.info(f"Actual price movements: UP: {up_count} ({up_count/total_tests:.1%}), DOWN: {down_count} ({down_count/total_tests:.1%})")
        
        # RandomForest stats
        if 'rf_correct' in df.columns:
            rf_correct = df['rf_correct'].sum()
            rf_accuracy = rf_correct / total_tests
            logger.info(f"RandomForest accuracy: {rf_correct}/{total_tests} ({rf_accuracy:.2%})")
            
            # Split by direction
            rf_up_correct = ((df['rf_prediction'] == 'UP') & df['rf_correct']).sum()
            rf_down_correct = ((df['rf_prediction'] == 'DOWN') & df['rf_correct']).sum()
            
            rf_up_total = (df['rf_prediction'] == 'UP').sum()
            rf_down_total = (df['rf_prediction'] == 'DOWN').sum()
            
            logger.info(f"  UP predictions: {rf_up_correct}/{rf_up_total} correct ({rf_up_correct/max(1,rf_up_total):.2%})")
            logger.info(f"  DOWN predictions: {rf_down_correct}/{rf_down_total} correct ({rf_down_correct/max(1,rf_down_total):.2%})")
        
        # XGBoost stats
        if 'xgb_correct' in df.columns:
            xgb_correct = df['xgb_correct'].sum()
            xgb_accuracy = xgb_correct / total_tests
            logger.info(f"XGBoost accuracy: {xgb_correct}/{total_tests} ({xgb_accuracy:.2%})")
            
            # Split by direction
            xgb_up_correct = ((df['xgb_prediction'] == 'UP') & df['xgb_correct']).sum()
            xgb_down_correct = ((df['xgb_prediction'] == 'DOWN') & df['xgb_correct']).sum()
            
            xgb_up_total = (df['xgb_prediction'] == 'UP').sum()
            xgb_down_total = (df['xgb_prediction'] == 'DOWN').sum()
            
            logger.info(f"  UP predictions: {xgb_up_correct}/{xgb_up_total} correct ({xgb_up_correct/max(1,xgb_up_total):.2%})")
            logger.info(f"  DOWN predictions: {xgb_down_correct}/{xgb_down_total} correct ({xgb_down_correct/max(1,xgb_down_total):.2%})")
        
        # Ensemble stats
        if 'ensemble_correct' in df.columns:
            ensemble_correct = df['ensemble_correct'].sum()
            ensemble_accuracy = ensemble_correct / total_tests
            logger.info(f"Ensemble accuracy: {ensemble_correct}/{total_tests} ({ensemble_accuracy:.2%})")
            
            # Split by direction
            ensemble_up_correct = ((df['ensemble_prediction'] == 'UP') & df['ensemble_correct']).sum()
            ensemble_down_correct = ((df['ensemble_prediction'] == 'DOWN') & df['ensemble_correct']).sum()
            
            ensemble_up_total = (df['ensemble_prediction'] == 'UP').sum()
            ensemble_down_total = (df['ensemble_prediction'] == 'DOWN').sum()
            
            logger.info(f"  UP predictions: {ensemble_up_correct}/{ensemble_up_total} correct ({ensemble_up_correct/max(1,ensemble_up_total):.2%})")
            logger.info(f"  DOWN predictions: {ensemble_down_correct}/{ensemble_down_total} correct ({ensemble_down_correct/max(1,ensemble_down_total):.2%})")
    
    def _generate_plots(self, df: pd.DataFrame, timestamp: str) -> None:
        """
        Generate comparison plots
        
        Args:
            df: DataFrame with comparison results
            timestamp: Timestamp for file names
        """
        try:
            # Plot 1: Accuracy Comparison
            plt.figure(figsize=(10, 6))
            model_names = []
            accuracies = []
            
            if 'rf_correct' in df.columns:
                model_names.append('RandomForest')
                accuracies.append(df['rf_correct'].mean())
            
            if 'xgb_correct' in df.columns:
                model_names.append('XGBoost')
                accuracies.append(df['xgb_correct'].mean())
            
            if 'ensemble_correct' in df.columns:
                model_names.append('Ensemble')
                accuracies.append(df['ensemble_correct'].mean())
            
            plt.bar(model_names, accuracies, color=['#ff9999', '#66b3ff', '#99ff99'])
            plt.ylabel('Accuracy')
            plt.title(f'{self.symbol} Model Accuracy Comparison')
            
            # Add values on bars
            for i, v in enumerate(accuracies):
                plt.text(i, v + 0.01, f'{v:.2%}', ha='center')
            
            plt.ylim(0, max(accuracies) * 1.2)
            plt.savefig(f"plots/{self.symbol}_accuracy_comparison_{timestamp}.png")
            logger.info(f"Saved accuracy comparison plot")
            
            # Plot 2: Confidence Distribution
            plt.figure(figsize=(12, 8))
            
            if 'rf_confidence' in df.columns and 'rf_correct' in df.columns:
                plt.subplot(2, 2, 1)
                correct_conf = df[df['rf_correct']]['rf_confidence']
                incorrect_conf = df[~df['rf_correct']]['rf_confidence']
                
                plt.hist([correct_conf, incorrect_conf], bins=10, alpha=0.7, 
                         label=['Correct', 'Incorrect'])
                plt.title('RandomForest Confidence Distribution')
                plt.xlabel('Confidence')
                plt.ylabel('Count')
                plt.legend()
            
            if 'xgb_confidence' in df.columns and 'xgb_correct' in df.columns:
                plt.subplot(2, 2, 2)
                correct_conf = df[df['xgb_correct']]['xgb_confidence']
                incorrect_conf = df[~df['xgb_correct']]['xgb_confidence']
                
                plt.hist([correct_conf, incorrect_conf], bins=10, alpha=0.7,
                         label=['Correct', 'Incorrect'])
                plt.title('XGBoost Confidence Distribution')
                plt.xlabel('Confidence')
                plt.ylabel('Count')
                plt.legend()
            
            if 'ensemble_confidence' in df.columns and 'ensemble_correct' in df.columns:
                plt.subplot(2, 2, 3)
                correct_conf = df[df['ensemble_correct']]['ensemble_confidence']
                incorrect_conf = df[~df['ensemble_correct']]['ensemble_confidence']
                
                plt.hist([correct_conf, incorrect_conf], bins=10, alpha=0.7,
                         label=['Correct', 'Incorrect'])
                plt.title('Ensemble Confidence Distribution')
                plt.xlabel('Confidence')
                plt.ylabel('Count')
                plt.legend()
            
            plt.tight_layout()
            plt.savefig(f"plots/{self.symbol}_confidence_distribution_{timestamp}.png")
            logger.info(f"Saved confidence distribution plot")
            
            # Plot 3: Model Agreement Analysis
            if 'rf_prediction' in df.columns and 'xgb_prediction' in df.columns:
                plt.figure(figsize=(10, 6))
                
                # Calculate agreement and correctness
                agreement = (df['rf_prediction'] == df['xgb_prediction']).astype(int)
                correct_when_agree = ((df['rf_prediction'] == df['xgb_prediction']) & 
                                     (df['rf_prediction'] == df['actual_direction'])).astype(int)
                
                correct_rf_only = ((df['rf_prediction'] != df['xgb_prediction']) & 
                                  (df['rf_prediction'] == df['actual_direction'])).astype(int)
                
                correct_xgb_only = ((df['rf_prediction'] != df['xgb_prediction']) & 
                                   (df['xgb_prediction'] == df['actual_direction'])).astype(int)
                
                # Count statistics
                total_tests = len(df)
                agreement_count = agreement.sum()
                disagreement_count = total_tests - agreement_count
                
                correct_when_agree_count = correct_when_agree.sum()
                correct_rf_only_count = correct_rf_only.sum()
                correct_xgb_only_count = correct_xgb_only.sum()
                
                # Create pie chart for agreement
                plt.subplot(1, 2, 1)
                plt.pie([agreement_count, disagreement_count], 
                       labels=['Agreement', 'Disagreement'],
                       autopct='%1.1f%%',
                       colors=['#66b3ff', '#ff9999'])
                plt.title('Model Agreement')
                
                # Create bar chart for correctness
                plt.subplot(1, 2, 2)
                agreement_accuracy = correct_when_agree_count / agreement_count if agreement_count > 0 else 0
                rf_only_accuracy = correct_rf_only_count / disagreement_count if disagreement_count > 0 else 0
                xgb_only_accuracy = correct_xgb_only_count / disagreement_count if disagreement_count > 0 else 0
                
                plt.bar(['When Agree', 'RF Only Correct', 'XGB Only Correct'],
                       [agreement_accuracy, rf_only_accuracy, xgb_only_accuracy],
                       color=['#66b3ff', '#ff9999', '#99ff99'])
                plt.ylabel('Accuracy')
                plt.title('Accuracy Analysis')
                
                # Add values on bars
                for i, v in enumerate([agreement_accuracy, rf_only_accuracy, xgb_only_accuracy]):
                    plt.text(i, v + 0.01, f'{v:.2%}', ha='center')
                
                plt.tight_layout()
                plt.savefig(f"plots/{self.symbol}_agreement_analysis_{timestamp}.png")
                logger.info(f"Saved model agreement analysis plot")
        
        except Exception as e:
            logger.error(f"Error generating plots: {e}")

def main():
    parser = argparse.ArgumentParser(description='Compare cryptocurrency price prediction models')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='Trading pair symbol')
    parser.add_argument('--hours', type=int, default=48, help='Hours of data to use for testing')
    parser.add_argument('--window', type=int, default=5, help='Window size for price history')
    args = parser.parse_args()
    
    # Run comparison
    comparison = ModelComparison(symbol=args.symbol, window_size=args.window)
    results_df = comparison.run_comparison(hours=args.hours)
    
    if not results_df.empty:
        print(f"\nComparison results saved to results/ directory")
        print(f"Plots saved to plots/ directory")
    else:
        print(f"\nComparison failed. Check logs for details.")

if __name__ == "__main__":
    main() 