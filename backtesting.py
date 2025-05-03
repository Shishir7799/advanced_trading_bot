import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
import seaborn as sns

from fetch_crypto_prices import PriceFetcher
from price_prediction_model import PricePredictor
from xgboost_price_predictor import XGBoostPredictor
from technical_indicators import TechnicalIndicators

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("backtesting.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Backtester:
    """Enhanced class for backtesting trading strategies on historical data"""
    
    def __init__(self, symbol: str = "BTCUSDT", 
                 initial_balance: float = 1000.0,
                 trade_amount_percent: float = 10.0, 
                 fee_percent: float = 0.1):
        """
        Initialize the backtester
        
        Args:
            symbol: Trading pair symbol
            initial_balance: Initial balance in USDT
            trade_amount_percent: Percentage of balance to use per trade
            fee_percent: Trading fee percentage
        """
        self.symbol = symbol
        self.initial_balance = initial_balance
        self.trade_amount_percent = trade_amount_percent
        self.fee_percent = fee_percent
        
        # Create output directories
        os.makedirs("backtest_results", exist_ok=True)
        
        # Initialize price fetcher and indicators
        self.price_fetcher = PriceFetcher(symbol=symbol)
        
        # Initialize models
        self.rf_predictor = PricePredictor(symbol=symbol)
        self.xgb_predictor = XGBoostPredictor(symbol=symbol)
        
        logger.info(f"Backtester initialized for {symbol}")
        logger.info(f"Initial balance: {initial_balance} USDT")
        logger.info(f"Trade amount: {trade_amount_percent}% of balance")
        logger.info(f"Fee: {fee_percent}%")
    
    def fetch_historical_data(self, days: int = 30, interval: str = "1h") -> pd.DataFrame:
        """
        Fetch historical data for backtesting
        
        Args:
            days: Number of days of historical data
            interval: Candlestick interval
            
        Returns:
            DataFrame with historical price data
        """
        logger.info(f"Fetching {days} days of historical {self.symbol} data...")
        
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        df = self.price_fetcher.fetch_large_historical_dataset(
            start_time=start_time,
            end_time=end_time,
            interval=interval
        )
        
        if df.empty:
            logger.error("Failed to fetch historical data")
            raise ValueError("Empty dataset received")
        
        logger.info(f"Successfully fetched {len(df)} candles of historical data")
        
        # Add technical indicators
        enriched_df = TechnicalIndicators.add_all_indicators(df)
        return enriched_df
    
    def run_backtest(self, strategy: str = "ensemble", 
                     days: int = 30, 
                     interval: str = "1h",
                     rf_threshold: float = 0.5,
                     xgb_threshold: float = 0.5) -> Dict[str, Any]:
        """
        Run backtest with the specified strategy
        
        Args:
            strategy: Strategy to test ("rf", "xgb", or "ensemble")
            days: Number of days of historical data
            interval: Candlestick interval
            rf_threshold: Confidence threshold for RandomForest model
            xgb_threshold: Confidence threshold for XGBoost model
            
        Returns:
            Dictionary with backtest results
        """
        logger.info(f"Starting backtest with {strategy} strategy")
        
        # Set model thresholds
        self.rf_predictor.threshold = rf_threshold
        self.xgb_predictor.threshold = xgb_threshold
        
        # Fetch historical data
        df = self.fetch_historical_data(days=days, interval=interval)
        
        # Initialize backtest variables
        balance = self.initial_balance
        crypto_holdings = 0.0
        trades = []
        portfolio_values = []
        timestamps = []
        actual_directions = []
        predictions = []
        confidences = []
        correct_predictions = []
        
        # Loop through data points
        for i in range(self.rf_predictor.window_size, len(df) - 1):
            # Current price and time
            current_price = df['close'].iloc[i]
            next_price = df['close'].iloc[i + 1]
            current_time = df.index[i]
            
            # Calculate portfolio value
            portfolio_value = balance + (crypto_holdings * current_price)
            portfolio_values.append(portfolio_value)
            timestamps.append(current_time)
            
            # Record actual price direction
            actual_direction = 1 if next_price > current_price else 0
            actual_directions.append(actual_direction)
            
            # Get slice of data up to current point
            df_slice = df.iloc[:i+1]
            
            # Make predictions based on strategy
            prediction = -1
            confidence = 0.0
            
            if strategy == "rf":
                prediction, confidence, _ = self.rf_predictor.predict(df_slice)
            elif strategy == "xgb":
                prediction, confidence, _ = self.xgb_predictor.predict(df_slice)
            else:  # ensemble
                rf_prediction, rf_confidence, _ = self.rf_predictor.predict(df_slice)
                xgb_prediction, xgb_confidence, _ = self.xgb_predictor.predict(df_slice)
                
                # Logic for ensemble prediction
                prediction = -1  # Default to no action
                
                # If both models agree, use the prediction
                if rf_prediction == xgb_prediction and rf_prediction != -1:
                    prediction = rf_prediction
                    confidence = max(rf_confidence, xgb_confidence)
                # If models disagree but XGBoost is very confident
                elif rf_prediction != xgb_prediction and xgb_prediction != -1 and xgb_confidence > 0.7:
                    prediction = xgb_prediction
                    confidence = xgb_confidence
            
            # Store prediction results
            predictions.append(prediction)
            confidences.append(confidence)
            
            # Check if prediction was correct (only for actual predictions, not -1)
            if prediction != -1:
                correct_predictions.append(prediction == actual_direction)
            
            # Execute trade based on prediction
            if prediction == 1 and balance > 0:  # Buy signal
                trade_amount = balance * (self.trade_amount_percent / 100)
                fee = trade_amount * (self.fee_percent / 100)
                crypto_bought = (trade_amount - fee) / current_price
                
                balance -= trade_amount
                crypto_holdings += crypto_bought
                
                trades.append({
                    'time': current_time,
                    'type': 'BUY',
                    'price': current_price,
                    'amount_usdt': trade_amount,
                    'amount_crypto': crypto_bought,
                    'fee': fee,
                    'balance_after': balance,
                    'holdings_after': crypto_holdings,
                    'portfolio_value': balance + (crypto_holdings * current_price),
                    'prediction': prediction,
                    'confidence': confidence,
                    'actual_direction': actual_direction,
                    'correct': prediction == actual_direction
                })
                
                logger.info(f"BUY: {crypto_bought:.6f} {self.symbol} at {current_price:.2f}")
                
            elif prediction == 0 and crypto_holdings > 0:  # Sell signal
                trade_amount = crypto_holdings * current_price
                fee = trade_amount * (self.fee_percent / 100)
                usdt_received = trade_amount - fee
                
                balance += usdt_received
                crypto_holdings = 0
                
                trades.append({
                    'time': current_time,
                    'type': 'SELL',
                    'price': current_price,
                    'amount_usdt': usdt_received,
                    'amount_crypto': crypto_holdings,
                    'fee': fee,
                    'balance_after': balance,
                    'holdings_after': crypto_holdings,
                    'portfolio_value': balance,
                    'prediction': prediction,
                    'confidence': confidence,
                    'actual_direction': actual_direction,
                    'correct': prediction == actual_direction
                })
                
                logger.info(f"SELL: {crypto_holdings:.6f} {self.symbol} at {current_price:.2f}")
        
        # Final evaluation
        final_price = df['close'].iloc[-1]
        final_portfolio_value = balance + (crypto_holdings * final_price)
        total_return = (final_portfolio_value / self.initial_balance) - 1
        
        # Calculate accuracy metrics
        filtered_predictions = [p for p in predictions if p != -1]
        filtered_actual = [actual_directions[i] for i, p in enumerate(predictions) if p != -1]
        
        accuracy = sum(correct_predictions) / len(correct_predictions) if correct_predictions else 0
        
        # If we still have crypto, sell it for final evaluation
        if crypto_holdings > 0:
            logger.info(f"Note: Still holding {crypto_holdings:.6f} {self.symbol} at end of test")
        
        # Calculate performance metrics
        win_trades = len([t for t in trades if t['correct']])
        total_trades = len(trades)
        win_rate = win_trades / total_trades if total_trades > 0 else 0
        
        # Prepare results
        results = {
            'initial_balance': self.initial_balance,
            'final_balance': balance,
            'final_crypto_holdings': crypto_holdings,
            'final_portfolio_value': final_portfolio_value,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'buy_and_hold_return': (df['close'].iloc[-1] / df['close'].iloc[self.rf_predictor.window_size]) - 1,
            'buy_and_hold_return_pct': ((df['close'].iloc[-1] / df['close'].iloc[self.rf_predictor.window_size]) - 1) * 100,
            'total_trades': len(trades),
            'win_rate': win_rate,
            'accuracy': accuracy,
            'trades': trades,
            'portfolio_values': portfolio_values,
            'timestamps': timestamps,
            'actual_directions': actual_directions,
            'predictions': predictions,
            'confidences': confidences,
            'strategy': strategy,
            'test_period': f"{days} days",
            'interval': interval,
            'rf_threshold': rf_threshold,
            'xgb_threshold': xgb_threshold,
            'prices': df['close'].values[self.rf_predictor.window_size:-1].tolist(),
            'start_price': df['close'].iloc[self.rf_predictor.window_size],
            'end_price': df['close'].iloc[-1]
        }
        
        # Log results
        logger.info(f"Backtest completed for {strategy} strategy")
        logger.info(f"Initial balance: {self.initial_balance:.2f} USDT")
        logger.info(f"Final balance: {balance:.2f} USDT")
        logger.info(f"Final crypto holdings: {crypto_holdings:.6f} {self.symbol}")
        logger.info(f"Final portfolio value: {final_portfolio_value:.2f} USDT")
        logger.info(f"Total return: {total_return:.2%}")
        logger.info(f"Buy and hold return: {results['buy_and_hold_return']:.2%}")
        logger.info(f"Total trades: {len(trades)}")
        logger.info(f"Win rate: {win_rate:.2%}")
        logger.info(f"Prediction accuracy: {accuracy:.2%}")
        
        # Generate performance charts
        chart_file = self._generate_performance_charts(results)
        results['chart_file'] = chart_file
        
        # Save results to JSON
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"backtest_results/{self.symbol}_{strategy}_{timestamp}.json"
        
        # Convert datetime objects to strings for JSON serialization
        results_json = results.copy()
        results_json['timestamps'] = [t.strftime('%Y-%m-%d %H:%M:%S') for t in results_json['timestamps']]
        results_json['trades'] = [{**t, 'time': t['time'].strftime('%Y-%m-%d %H:%M:%S')} for t in results_json['trades']]
        
        with open(results_file, 'w') as f:
            json.dump(results_json, f, indent=2)
        
        logger.info(f"Saved backtest results to {results_file}")
        
        return results
    
    def _generate_performance_charts(self, results: Dict[str, Any]) -> str:
        """
        Generate comprehensive performance charts for backtest results
        
        Args:
            results: Dictionary with backtest results
            
        Returns:
            Path to saved chart file
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"backtest_results/{self.symbol}_{results['strategy']}_{timestamp}.png"
            
            # Create a multi-panel figure
            fig = plt.figure(figsize=(15, 18))
            
            # 1. Price chart with buy/sell markers
            ax1 = plt.subplot2grid((4, 2), (0, 0), colspan=2)
            ax1.plot(results['timestamps'], results['prices'], label='Price', color='black')
            
            # Add buy/sell markers
            for trade in results['trades']:
                if trade['type'] == 'BUY':
                    ax1.scatter(trade['time'], trade['price'], color='green', marker='^', s=100)
                else:  # SELL
                    ax1.scatter(trade['time'], trade['price'], color='red', marker='v', s=100)
            
            ax1.set_title(f"{self.symbol} Price with Trading Signals")
            ax1.set_ylabel('Price (USDT)')
            ax1.grid(True)
            ax1.legend()
            
            # 2. Portfolio value
            ax2 = plt.subplot2grid((4, 2), (1, 0), colspan=2)
            ax2.plot(results['timestamps'], results['portfolio_values'], label='Portfolio Value', color='blue')
            
            # Add reference line for buy & hold
            buy_hold_values = [results['start_price'] * (1 + results['buy_and_hold_return']) * 
                              (results['initial_balance'] / results['start_price']) 
                              for _ in range(len(results['timestamps']))]
            ax2.plot(results['timestamps'], buy_hold_values, label='Buy & Hold', color='gray', linestyle='--')
            
            ax2.set_title('Portfolio Value vs Buy & Hold')
            ax2.set_ylabel('Value (USDT)')
            ax2.grid(True)
            ax2.legend()
            
            # 3. Prediction vs Actual (filtered for valid predictions)
            ax3 = plt.subplot2grid((4, 2), (2, 0))
            
            # Create cleaner visualization showing prediction correctness
            valid_indices = [i for i, p in enumerate(results['predictions']) if p != -1]
            valid_times = [results['timestamps'][i] for i in valid_indices]
            valid_preds = [results['predictions'][i] for i in valid_indices]
            valid_actual = [results['actual_directions'][i] for i in valid_indices]
            valid_correct = [valid_preds[i] == valid_actual[i] for i in range(len(valid_preds))]
            
            # Plot correct/incorrect predictions
            correct_times = [valid_times[i] for i in range(len(valid_times)) if valid_correct[i]]
            incorrect_times = [valid_times[i] for i in range(len(valid_times)) if not valid_correct[i]]
            
            correct_preds = [valid_preds[i] for i in range(len(valid_preds)) if valid_correct[i]]
            incorrect_preds = [valid_preds[i] for i in range(len(valid_preds)) if not valid_correct[i]]
            
            # Show as up/down arrows
            for i, t in enumerate(correct_times):
                if correct_preds[i] == 1:  # Up prediction
                    ax3.scatter(t, 1, color='green', marker='^', s=50, alpha=0.7)
                else:  # Down prediction
                    ax3.scatter(t, 0, color='red', marker='v', s=50, alpha=0.7)
                    
            for i, t in enumerate(incorrect_times):
                if incorrect_preds[i] == 1:  # Incorrect up prediction
                    ax3.scatter(t, 1, color='green', marker='^', s=50, alpha=0.3, edgecolors='black')
                else:  # Incorrect down prediction
                    ax3.scatter(t, 0, color='red', marker='v', s=50, alpha=0.3, edgecolors='black')
            
            ax3.set_yticks([0, 1])
            ax3.set_yticklabels(['Down', 'Up'])
            ax3.set_title('Predictions (Bright = Correct, Faded = Incorrect)')
            ax3.grid(True)
            
            # 4. Accuracy over time (rolling window)
            ax4 = plt.subplot2grid((4, 2), (2, 1))
            
            # Calculate rolling accuracy
            window_size = min(30, len(valid_correct))
            if len(valid_correct) > window_size:
                rolling_accuracy = [
                    sum(valid_correct[max(0, i-window_size):i]) / min(window_size, i) 
                    for i in range(1, len(valid_correct)+1)
                ]
                ax4.plot(valid_times, rolling_accuracy, color='purple')
                ax4.set_title(f'Rolling Accuracy ({window_size}-period window)')
                ax4.set_ylabel('Accuracy')
                ax4.set_ylim(0, 1)
                ax4.grid(True)
            else:
                ax4.text(0.5, 0.5, "Not enough data for rolling accuracy", 
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax4.transAxes)
            
            # 5. Confidence distribution
            ax5 = plt.subplot2grid((4, 2), (3, 0))
            
            # Filter for valid predictions with confidence
            valid_confidences = [results['confidences'][i] for i in valid_indices]
            correct_conf = [valid_confidences[i] for i in range(len(valid_confidences)) if valid_correct[i]]
            incorrect_conf = [valid_confidences[i] for i in range(len(valid_confidences)) if not valid_correct[i]]
            
            sns.histplot(correct_conf, color='green', alpha=0.5, bins=10, label='Correct', ax=ax5)
            sns.histplot(incorrect_conf, color='red', alpha=0.5, bins=10, label='Incorrect', ax=ax5)
            
            ax5.set_title('Prediction Confidence Distribution')
            ax5.set_xlabel('Confidence')
            ax5.set_ylabel('Count')
            ax5.legend()
            
            # 6. Summary metrics
            ax6 = plt.subplot2grid((4, 2), (3, 1))
            ax6.axis('off')
            
            metrics_text = (
                f"Strategy: {results['strategy']}\n"
                f"Total Return: {results['total_return_pct']:.2f}%\n"
                f"Buy & Hold Return: {results['buy_and_hold_return_pct']:.2f}%\n"
                f"Alpha: {results['total_return_pct'] - results['buy_and_hold_return_pct']:.2f}%\n\n"
                f"Total Trades: {results['total_trades']}\n"
                f"Win Rate: {results['win_rate']*100:.2f}%\n"
                f"Prediction Accuracy: {results['accuracy']*100:.2f}%\n\n"
                f"Initial Balance: {results['initial_balance']:.2f} USDT\n"
                f"Final Value: {results['final_portfolio_value']:.2f} USDT\n"
                f"Test Period: {results['test_period']} ({results['interval']} candles)"
            )
            
            ax6.text(0.05, 0.95, metrics_text, verticalalignment='top', 
                    fontsize=12, fontweight='bold', family='monospace')
            
            plt.tight_layout()
            plt.savefig(filename)
            plt.close()
            
            logger.info(f"Performance charts saved to {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Error generating performance charts: {e}")
            return ""

    def compare_strategies(self, days: int = 30, interval: str = "1h") -> Dict[str, Any]:
        """
        Compare different trading strategies on the same dataset
        
        Args:
            days: Number of days of historical data
            interval: Candlestick interval
            
        Returns:
            Dictionary with comparison results
        """
        logger.info(f"Comparing strategies on {days} days of {interval} data")
        
        # Run backtests for each strategy
        strategies = ["rf", "xgb", "ensemble"]
        results = {}
        
        for strategy in strategies:
            logger.info(f"Testing {strategy} strategy...")
            result = self.run_backtest(
                strategy=strategy,
                days=days,
                interval=interval
            )
            results[strategy] = result
        
        # Generate comparison chart
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"backtest_results/strategy_comparison_{self.symbol}_{timestamp}.png"
            
            plt.figure(figsize=(12, 8))
            
            # 1. Compare portfolio values
            plt.subplot(2, 1, 1)
            for strategy in strategies:
                plt.plot(results[strategy]['timestamps'], 
                        results[strategy]['portfolio_values'], 
                        label=f"{strategy.upper()}")
            
            # Add buy & hold comparison
            buy_hold_values = [results['rf']['start_price'] * (1 + results['rf']['buy_and_hold_return']) * 
                              (results['rf']['initial_balance'] / results['rf']['start_price']) 
                              for _ in range(len(results['rf']['timestamps']))]
            plt.plot(results['rf']['timestamps'], buy_hold_values, 
                    label='Buy & Hold', color='black', linestyle='--')
            
            plt.title(f"{self.symbol} Strategy Comparison")
            plt.ylabel("Portfolio Value (USDT)")
            plt.grid(True)
            plt.legend()
            
            # 2. Performance metrics comparison
            plt.subplot(2, 1, 2)
            metrics = {
                'Return (%)': [results[s]['total_return_pct'] for s in strategies],
                'Win Rate (%)': [results[s]['win_rate'] * 100 for s in strategies],
                'Accuracy (%)': [results[s]['accuracy'] * 100 for s in strategies],
                'Trades': [results[s]['total_trades'] for s in strategies]
            }
            
            # Add Buy & Hold for return comparison
            strategies_with_bh = strategies + ['Buy & Hold']
            metrics['Return (%)'].append(results['rf']['buy_and_hold_return_pct'])
            
            # Use separate subplots for metrics with different scales
            x = range(len(strategies_with_bh))
            width = 0.25
            
            plt.bar([i - width for i in x[:len(strategies)]], metrics['Accuracy (%)'], 
                   width=width, label='Accuracy (%)', color='lightblue')
            plt.bar([i for i in x[:len(strategies)]], metrics['Win Rate (%)'], 
                   width=width, label='Win Rate (%)', color='lightgreen')
            plt.bar([i + width for i in x], metrics['Return (%)'], 
                   width=width, label='Return (%)', color='salmon')
            
            plt.xticks(x, [s.upper() for s in strategies_with_bh])
            plt.ylabel('Percentage')
            plt.title('Performance Metrics Comparison')
            plt.grid(True, axis='y')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(filename)
            plt.close()
            
            logger.info(f"Strategy comparison chart saved to {filename}")
            
            # Add file path to results
            results['comparison_chart'] = filename
            
        except Exception as e:
            logger.error(f"Error generating comparison chart: {e}")
        
        return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Backtest crypto trading strategies')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='Trading pair symbol')
    parser.add_argument('--days', type=int, default=30, help='Days of historical data to use')
    parser.add_argument('--interval', type=str, default='1h', help='Candlestick interval')
    parser.add_argument('--strategy', type=str, default='ensemble', choices=['rf', 'xgb', 'ensemble', 'compare'], 
                       help='Trading strategy to test (use "compare" for all)')
    parser.add_argument('--initial', type=float, default=1000.0, help='Initial balance in USDT')
    parser.add_argument('--trade-pct', type=float, default=10.0, help='Percentage of balance to trade')
    parser.add_argument('--rf-threshold', type=float, default=0.5, help='RandomForest confidence threshold')
    parser.add_argument('--xgb-threshold', type=float, default=0.5, help='XGBoost confidence threshold')
    
    args = parser.parse_args()
    
    # Create backtester
    backtester = Backtester(
        symbol=args.symbol,
        initial_balance=args.initial,
        trade_amount_percent=args.trade_pct
    )
    
    # Run backtest
    if args.strategy == 'compare':
        results = backtester.compare_strategies(
            days=args.days,
            interval=args.interval
        )
        print("\n=== Strategy Comparison Results ===")
        for strategy in ['rf', 'xgb', 'ensemble']:
            print(f"\n{strategy.upper()} Strategy:")
            print(f"  Return: {results[strategy]['total_return_pct']:.2f}% (Buy & Hold: {results[strategy]['buy_and_hold_return_pct']:.2f}%)")
            print(f"  Accuracy: {results[strategy]['accuracy']*100:.2f}%")
            print(f"  Win Rate: {results[strategy]['win_rate']*100:.2f}%")
            print(f"  Trades: {results[strategy]['total_trades']}")
            
        print(f"\nComparison chart saved to {results.get('comparison_chart', 'N/A')}")
    else:
        results = backtester.run_backtest(
            strategy=args.strategy,
            days=args.days,
            interval=args.interval,
            rf_threshold=args.rf_threshold,
            xgb_threshold=args.xgb_threshold
        )
        
        print(f"\n=== Backtest Results ({args.strategy} strategy) ===")
        print(f"Initial Balance: {args.initial:.2f} USDT")
        print(f"Final Portfolio Value: {results['final_portfolio_value']:.2f} USDT")
        print(f"Total Return: {results['total_return_pct']:.2f}%")
        print(f"Buy & Hold Return: {results['buy_and_hold_return_pct']:.2f}%")
        print(f"Alpha: {results['total_return_pct'] - results['buy_and_hold_return_pct']:.2f}%")
        print(f"Total Trades: {results['total_trades']}")
        print(f"Win Rate: {results['win_rate']*100:.2f}%")
        print(f"Prediction Accuracy: {results['accuracy']*100:.2f}%")
        print(f"Performance charts saved to {results['chart_file']}") 