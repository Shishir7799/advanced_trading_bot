import time
import logging
import os
import pandas as pd
import argparse
import signal
import sys
import traceback
from typing import List, Optional, Tuple, Dict
from datetime import datetime, timedelta

from fetch_crypto_prices import PriceFetcher
from price_prediction_model import PricePredictor
from xgboost_price_predictor import XGBoostPredictor
from aptos_transaction import AptosTransactionManager

# Configure enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ensemble_trading_bot.log"),
        logging.StreamHandler()
    ]
)

# Create a separate DEBUG level log file for detailed debugging
debug_handler = logging.FileHandler("ensemble_trading_bot_debug.log")
debug_handler.setLevel(logging.DEBUG)
debug_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

# Get logger
logger = logging.getLogger(__name__)
logger.addHandler(debug_handler)

class EnsembleTradingBot:
    def __init__(self, 
                 symbol: str = "BTCUSDT", 
                 window_size: int = 5, 
                 check_interval: int = 300,  # 5 minutes in seconds
                 confidence_threshold: float = 0.6,
                 xgboost_threshold: float = 0.7,
                 trade_amount: float = 0.01,
                 mock_mode: bool = True,
                 train_if_missing: bool = True):
        """
        Initialize the ensemble trading bot
        
        Args:
            symbol: Trading pair symbol
            window_size: Size of price window for predictions
            check_interval: Time between price checks in seconds (default: 300s = 5 minutes)
            confidence_threshold: Minimum confidence to execute a trade
            xgboost_threshold: Minimum confidence for XGBoost when models disagree
            trade_amount: Amount of crypto to buy/sell per trade
            mock_mode: Whether to use mock transactions
            train_if_missing: Whether to train models if none exists
        """
        self.symbol = symbol
        self.window_size = window_size
        self.check_interval = check_interval
        self.confidence_threshold = confidence_threshold
        self.xgboost_threshold = xgboost_threshold
        self.trade_amount = trade_amount
        self.mock_mode = mock_mode
        self.train_if_missing = train_if_missing
        
        # Flag to control the main loop
        self.running = False
        
        # Create required directories
        os.makedirs("models", exist_ok=True)
        os.makedirs("data", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        os.makedirs("transaction_logs", exist_ok=True)
        
        logger.info(f"Initializing ensemble trading bot with: symbol={symbol}, window_size={window_size}, "
                   f"check_interval={check_interval}s, confidence_threshold={confidence_threshold}, "
                   f"xgboost_threshold={xgboost_threshold}, trade_amount={trade_amount}, mock_mode={mock_mode}")
        
        # Initialize components
        self.price_fetcher = PriceFetcher(symbol=symbol)
        self.rf_predictor = PricePredictor(symbol=symbol, window_size=window_size)
        self.xgb_predictor = XGBoostPredictor(symbol=symbol, window_size=window_size)
        self.tx_manager = AptosTransactionManager(mock_mode=mock_mode)
        
        # Check if models exist and train if needed
        self._ensure_models_available()
        
        # Agreement statistics
        self.agreements = 0
        self.disagreements = 0
        self.total_predictions = 0
        
        # Performance tracking
        self.predictions = []
        self.last_prediction_time = None
        self.last_trade_time = None
        self.prediction_cooldown = timedelta(seconds=check_interval)
        self.trade_cooldown = timedelta(seconds=check_interval * 2)  # Less frequent trading
        
        # Loop statistics
        self.loop_count = 0
        self.start_time = datetime.now()
        self.total_prediction_time = 0
        self.total_trade_time = 0
        self.error_count = 0
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.debug("Ensemble trading bot initialization complete")
        
    def _signal_handler(self, signum, frame):
        """Handle termination signals"""
        logger.info(f"Received signal {signum}. Shutting down gracefully...")
        self.running = False
        
    def _save_performance_data(self):
        """Save performance data to CSV file"""
        if not self.predictions:
            return
            
        try:
            df = pd.DataFrame(self.predictions)
            filename = f"data/{self.symbol}_ensemble_performance_{datetime.now().strftime('%Y%m%d')}.csv"
            df.to_csv(filename, index=False)
            logger.info(f"Performance data saved to {filename}")
            
            # Log statistics summary
            total_runtime = (datetime.now() - self.start_time).total_seconds()
            avg_prediction_time = self.total_prediction_time / max(1, len(self.predictions))
            avg_trade_time = self.total_trade_time / max(1, sum(1 for p in self.predictions if p.get('trade_executed', False)))
            
            # Calculate agreement percentage
            agreement_pct = (self.agreements / max(1, self.total_predictions)) * 100
            
            logger.info(f"Bot statistics: loops={self.loop_count}, runtime={total_runtime:.1f}s, "
                       f"predictions={len(self.predictions)}, errors={self.error_count}, "
                       f"avg_prediction_time={avg_prediction_time:.3f}s, avg_trade_time={avg_trade_time:.3f}s")
            logger.info(f"Model agreement: {self.agreements}/{self.total_predictions} ({agreement_pct:.1f}%)")
            
        except Exception as e:
            logger.error(f"Failed to save performance data: {e}")
            logger.debug(f"Performance data save error details: {traceback.format_exc()}")
            
    def get_price_window(self) -> Optional[List[float]]:
        """Get the latest price window for prediction"""
        logger.debug("Fetching current price...")
        start_time = time.time()
        
        # Get current price and add to history
        current_price = self.price_fetcher.get_current_price()
        
        fetch_time = time.time() - start_time
        logger.debug(f"Price fetch completed in {fetch_time:.3f}s")
        
        if current_price is None:
            logger.error("Failed to fetch current price")
            return None
            
        # Log the price fetched
        logger.debug(f"Price fetched: ${current_price:.2f} for {self.symbol}")
            
        # Get the last n prices
        price_df = self.price_fetcher.get_last_n_prices(self.window_size)
        
        # Check if we have enough prices
        if len(price_df) < self.window_size:
            logger.warning(f"Not enough price data yet. Have {len(price_df)}/{self.window_size}")
            return None
            
        # Extract prices from DataFrame
        price_window = price_df['price'].tolist()
        logger.debug(f"Price window: {price_window}")
        
        return price_window
        
    def make_ensemble_prediction(self) -> Optional[dict]:
        """Make ensemble predictions using both RandomForest and XGBoost models"""
        prediction_start_time = time.time()
        
        # Check if models are ready
        if not self.rf_predictor.is_model_ready():
            logger.error("RandomForest model not ready. Please train the model first.")
            return None
            
        if not self.xgb_predictor.is_model_ready():
            logger.error("XGBoost model not ready. Please train the model first.")
            return None
            
        # Check if we have a cooldown period for predictions
        now = datetime.now()
        if self.last_prediction_time and now - self.last_prediction_time < self.prediction_cooldown:
            logger.debug(f"Prediction cooldown active. Last prediction at {self.last_prediction_time}")
            return None
            
        # Get price window
        price_window = self.get_price_window()
        if not price_window:
            return None
            
        # Make RandomForest prediction
        logger.debug(f"Making RandomForest prediction with window: {price_window}")
        rf_prediction, rf_confidence = self.rf_predictor.predict(price_window)
        
        # Make XGBoost prediction
        logger.debug(f"Making XGBoost prediction with window: {price_window}")
        xgb_prediction, xgb_confidence = self.xgb_predictor.predict(price_window)
        
        if rf_prediction is None or xgb_prediction is None:
            logger.error("One or both predictions failed")
            return None
            
        # Convert numeric predictions to string format
        rf_prediction_str = "UP" if rf_prediction == 1 else "DOWN"
        xgb_prediction_str = "UP" if xgb_prediction == 1 else "DOWN"
        
        # Check if models agree
        models_agree = rf_prediction == xgb_prediction
        
        # Update agreement statistics
        self.total_predictions += 1
        if models_agree:
            self.agreements += 1
            decision_model = "ENSEMBLE (AGREEMENT)"
            final_prediction = rf_prediction
            final_confidence = max(rf_confidence, xgb_confidence)
            should_trade = final_confidence >= self.confidence_threshold
        else:
            self.disagreements += 1
            # Only use XGBoost when models disagree and XGBoost confidence is high
            if xgb_confidence >= self.xgboost_threshold:
                decision_model = "XGBOOST (HIGH CONFIDENCE)"
                final_prediction = xgb_prediction
                final_confidence = xgb_confidence
                should_trade = True
            else:
                decision_model = "NONE (DISAGREEMENT)"
                final_prediction = None
                final_confidence = 0
                should_trade = False
        
        # Prepare prediction data
        prediction_data = {
            "timestamp": int(time.time()),
            "datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
            "price_window": price_window,
            "current_price": price_window[-1],
            "rf_prediction": rf_prediction_str,
            "rf_confidence": rf_confidence,
            "xgb_prediction": xgb_prediction_str,
            "xgb_confidence": xgb_confidence,
            "models_agree": models_agree,
            "decision_model": decision_model,
            "prediction": "UP" if final_prediction == 1 else "DOWN" if final_prediction == 0 else "NONE",
            "confidence": final_confidence,
            "threshold_met": should_trade,
            "prediction_time": time.time() - prediction_start_time
        }
        
        # Update last prediction time
        self.last_prediction_time = now
        
        # Track prediction time
        prediction_time = time.time() - prediction_start_time
        self.total_prediction_time += prediction_time
        
        # Add to predictions list
        self.predictions.append(prediction_data)
        
        # Print clearer prediction information
        print(f"\n{'='*70}")
        print(f"CURRENT PRICE: ${prediction_data['current_price']:.2f}")
        print(f"RF PREDICTION: {rf_prediction_str} with {rf_confidence:.2f} confidence")
        print(f"XGB PREDICTION: {xgb_prediction_str} with {xgb_confidence:.2f} confidence")
        print(f"MODELS AGREE: {'Yes' if models_agree else 'No'}")
        print(f"DECISION MODEL: {decision_model}")
        print(f"FINAL PREDICTION: {prediction_data['prediction']}")
        print(f"CONFIDENCE: {final_confidence:.2f}")
        print(f"SHOULD TRADE: {'Yes' if should_trade else 'No'}")
        print(f"PREDICTION TIME: {prediction_time:.3f}s")
        print(f"{'='*70}\n")
        
        # Detailed logging of prediction
        logger.info(f"Ensemble prediction: {prediction_data['prediction']} using {decision_model} for {self.symbol} at ${prediction_data['current_price']:.2f}")
        logger.info(f"RF: {rf_prediction_str}({rf_confidence:.2f}) | XGB: {xgb_prediction_str}({xgb_confidence:.2f}) | Agree: {models_agree}")
        logger.debug(f"Full prediction data: {prediction_data}")
        logger.debug(f"Prediction completed in {prediction_time:.3f}s")
        
        return prediction_data
        
    def execute_trade(self, prediction_data: dict) -> bool:
        """Execute a trade based on prediction"""
        trade_start_time = time.time()
        logger.debug(f"Evaluating trade for prediction: {prediction_data['prediction']} with model {prediction_data['decision_model']}")
        
        # Check if we have a cooldown period for trades
        now = datetime.now()
        if self.last_trade_time and now - self.last_trade_time < self.trade_cooldown:
            logger.info(f"Trade cooldown period active. Last trade at {self.last_trade_time}. Skipping trade.")
            return False
            
        # Check if prediction meets criteria for trading
        if not prediction_data["threshold_met"] or prediction_data["prediction"] == "NONE":
            logger.info(f"Trade criteria not met. Decision model: {prediction_data['decision_model']}")
            return False
            
        # Execute buy transaction if prediction is UP
        if prediction_data["prediction"] == "UP":
            logger.debug(f"Attempting to execute BUY transaction for {self.symbol} at ${prediction_data['current_price']:.2f}")
            result = self.tx_manager.execute_buy_transaction(
                symbol=self.symbol,
                price=prediction_data["current_price"],
                amount=self.trade_amount,
                confidence=prediction_data["confidence"]
            )
            
            if result:
                trade_time = time.time() - trade_start_time
                self.total_trade_time += trade_time
                
                print(f"\n{'*'*70}")
                print(f"BUY TRANSACTION EXECUTED: {result['transaction_id']}")
                print(f"AMOUNT: {self.trade_amount} BTC at ${prediction_data['current_price']:.2f}")
                print(f"MODEL: {prediction_data['decision_model']}")
                print(f"EXECUTION TIME: {trade_time:.3f}s")
                print(f"{'*'*70}\n")
                
                logger.info(f"Buy transaction executed: {result['transaction_id']} for {self.trade_amount} {self.symbol} at ${prediction_data['current_price']:.2f}")
                logger.info(f"Based on model: {prediction_data['decision_model']}")
                logger.debug(f"Transaction details: {result}")
                logger.debug(f"Trade execution completed in {trade_time:.3f}s")
                
                self.last_trade_time = now
                prediction_data["trade_executed"] = True
                prediction_data["trade_id"] = result["transaction_id"]
                prediction_data["trade_type"] = "BUY"
                prediction_data["trade_time"] = trade_time
                return True
            else:
                logger.error(f"Transaction execution failed for {self.symbol}")
                
        return False
        
    def run(self):
        """Run the trading bot main loop"""
        self.start_time = datetime.now()
        logger.info(f"Ensemble trading bot started at {self.start_time}")
        
        print(f"\n{'#'*70}")
        print(f"STARTING ENSEMBLE AI CRYPTO TRADING BOT")
        print(f"Symbol: {self.symbol}")
        print(f"Check interval: {self.check_interval} seconds (5 minutes)")
        print(f"Confidence threshold: {self.confidence_threshold}")
        print(f"XGBoost disagreement threshold: {self.xgboost_threshold}")
        print(f"Trade amount: {self.trade_amount} BTC")
        print(f"Mock mode: {self.mock_mode}")
        print(f"Start time: {self.start_time}")
        print(f"{'#'*70}\n")
        
        logger.info(f"Starting ensemble trading bot for {self.symbol} with {self.window_size} window size")
        logger.info(f"Check interval: {self.check_interval} seconds (5 minutes)")
        logger.info(f"Confidence threshold: {self.confidence_threshold}")
        logger.info(f"XGBoost disagreement threshold: {self.xgboost_threshold}")
        logger.info(f"Trade amount: {self.trade_amount} BTC")
        logger.info(f"Mock mode: {self.mock_mode}")
        
        self.running = True
        
        # Initialize price data by getting a few readings
        print("Initializing price data...")
        logger.info("Initializing price window with initial readings")
        for i in range(self.window_size):
            price = self.price_fetcher.get_current_price()
            if price:
                print(f"Fetched initial price {i+1}/{self.window_size}: ${price:.2f}")
                logger.debug(f"Initial price {i+1}/{self.window_size}: ${price:.2f}")
            time.sleep(1)
            
        print("\nStarting main trading loop...\n")
        logger.info("Starting main trading loop")
        
        # Main loop
        while self.running:
            loop_start_time = time.time()
            self.loop_count += 1
            
            logger.debug(f"Starting loop {self.loop_count} at {datetime.now()}")
            
            try:
                # Make ensemble prediction
                prediction_data = self.make_ensemble_prediction()
                
                # Execute trade if prediction is valid
                if prediction_data and prediction_data["prediction"] != "NONE":
                    self.execute_trade(prediction_data)
                    
                # Save performance data periodically
                if len(self.predictions) % 10 == 0 and len(self.predictions) > 0:
                    self._save_performance_data()
                    
            except Exception as e:
                self.error_count += 1
                error_trace = traceback.format_exc()
                logger.error(f"Error in main loop (count: {self.error_count}): {e}")
                logger.debug(f"Detailed error traceback: {error_trace}")
                print(f"\nERROR: {e}\n")
                
            finally:
                # Log loop completion time
                loop_time = time.time() - loop_start_time
                logger.debug(f"Loop {self.loop_count} completed in {loop_time:.3f}s")
                
                # Sleep until next check
                sleep_time = max(0.1, self.check_interval - loop_time)
                logger.debug(f"Sleeping for {sleep_time:.1f}s")
                time.sleep(sleep_time)
            
        # Bot shutdown
        end_time = datetime.now()
        runtime = (end_time - self.start_time).total_seconds()
        
        # Save performance data on exit
        self._save_performance_data()
        
        # Calculate final agreement percentage
        agreement_pct = (self.agreements / max(1, self.total_predictions)) * 100
        
        logger.info(f"Trading bot stopped at {end_time}. Total runtime: {runtime:.1f}s")
        logger.info(f"Total loops: {self.loop_count}, Predictions: {len(self.predictions)}, Errors: {self.error_count}")
        logger.info(f"Model agreement: {self.agreements}/{self.total_predictions} ({agreement_pct:.1f}%)")
        
        print(f"\nTrading bot stopped at {end_time}. Runtime: {runtime:.1f}s")
        print(f"Total loops: {self.loop_count}, Predictions: {len(self.predictions)}, Errors: {self.error_count}")
        print(f"Model agreement: {self.agreements}/{self.total_predictions} ({agreement_pct:.1f}%)")
        print("Performance data saved.")

    def _ensure_models_available(self):
        """Check if both models exist and train them if not available"""
        rf_ready = self.rf_predictor.is_model_ready()
        xgb_ready = self.xgb_predictor.is_model_ready()
        
        logger.info(f"RandomForest model ready: {rf_ready}")
        logger.info(f"XGBoost model ready: {xgb_ready}")
        
        if not rf_ready:
            logger.warning(f"RandomForest model for {self.symbol} not ready.")
            
            # Train RandomForest model if requested
            if self.train_if_missing:
                self._train_rf_model()
            
        if not xgb_ready:
            logger.warning(f"XGBoost model for {self.symbol} not ready.")
            
            # Train XGBoost model if requested
            if self.train_if_missing:
                self._train_xgb_model()
                
        # Recheck model status
        rf_ready = self.rf_predictor.is_model_ready()
        xgb_ready = self.xgb_predictor.is_model_ready()
        
        if not rf_ready or not xgb_ready:
            logger.warning(f"Some required models are still not available. RandomForest: {rf_ready}, XGBoost: {xgb_ready}")
            logger.warning("Bot will continue but may not function correctly without both models.")
        else:
            logger.info("Both RandomForest and XGBoost models are ready.")
    
    def _train_rf_model(self):
        """Train the RandomForest model"""
        logger.info("Training RandomForest model...")
        try:
            from train_model import ModelTrainer
            trainer = ModelTrainer(symbol=self.symbol, window_size=self.window_size)
            success = trainer.train_model()
            
            if success:
                logger.info("RandomForest model training successful. Reloading model...")
                # Reinitialize predictor to load the newly trained model
                self.rf_predictor = PricePredictor(symbol=self.symbol, window_size=self.window_size)
                if self.rf_predictor.is_model_ready():
                    logger.info("RandomForest model loaded successfully.")
                else:
                    logger.error("Failed to load the newly trained RandomForest model.")
            else:
                logger.error("RandomForest model training failed.")
        except Exception as e:
            logger.error(f"Error training RandomForest model: {e}")
    
    def _train_xgb_model(self):
        """Train the XGBoost model"""
        logger.info("Training XGBoost model...")
        try:
            from train_xgboost_model import XGBoostModelTrainer
            trainer = XGBoostModelTrainer(symbol=self.symbol, window_size=self.window_size)
            success = trainer.train_model()
            
            if success:
                logger.info("XGBoost model training successful. Reloading model...")
                # Reinitialize predictor to load the newly trained model
                self.xgb_predictor = XGBoostPredictor(symbol=self.symbol, window_size=self.window_size)
                if self.xgb_predictor.is_model_ready():
                    logger.info("XGBoost model loaded successfully.")
                else:
                    logger.error("Failed to load the newly trained XGBoost model.")
            else:
                logger.error("XGBoost model training failed.")
        except Exception as e:
            logger.error(f"Error training XGBoost model: {e}")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Ensemble Crypto Trading Bot')
    
    parser.add_argument('--symbol', type=str, default='BTCUSDT',
                        help='Trading pair symbol (default: BTCUSDT)')
    
    parser.add_argument('--window', type=int, default=5,
                        help='Price window size for prediction (default: 5)')
    
    parser.add_argument('--interval', type=int, default=300,  # 5 minutes in seconds
                        help='Time between price checks in seconds (default: 300 = 5 minutes)')
    
    parser.add_argument('--threshold', type=float, default=0.6,
                        help='Confidence threshold for trading (default: 0.6)')
    
    parser.add_argument('--xgb-threshold', type=float, default=0.7,
                        help='XGBoost confidence threshold when models disagree (default: 0.7)')
    
    parser.add_argument('--amount', type=float, default=0.01,
                        help='Amount to trade per transaction (default: 0.01)')
    
    parser.add_argument('--real', action='store_false', dest='mock_mode',
                        help='Use real transactions instead of mock mode')
    
    parser.add_argument('--no-train', action='store_false', dest='train_if_missing',
                        help='Do not train model if missing, just exit')
    
    parser.add_argument('--force-train-rf', action='store_true',
                        help='Force training RandomForest model before starting')
    
    parser.add_argument('--force-train-xgb', action='store_true',
                        help='Force training XGBoost model before starting')
    
    parser.set_defaults(mock_mode=True, train_if_missing=True)
    
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    
    # Train models if requested
    if args.force_train_rf:
        from train_model import ModelTrainer
        logger.info("Force training RandomForest model before starting...")
        trainer = ModelTrainer(symbol=args.symbol, window_size=args.window)
        success = trainer.train_model(force_retrain=True)
        
        if not success:
            logger.error("Failed to train RandomForest model. Exiting.")
            sys.exit(1)
        
        logger.info("RandomForest model training completed successfully.")
    
    if args.force_train_xgb:
        from train_xgboost_model import XGBoostModelTrainer
        logger.info("Force training XGBoost model before starting...")
        trainer = XGBoostModelTrainer(symbol=args.symbol, window_size=args.window)
        success = trainer.train_model(force_retrain=True)
        
        if not success:
            logger.error("Failed to train XGBoost model. Exiting.")
            sys.exit(1)
        
        logger.info("XGBoost model training completed successfully.")
    
    # Create and run the trading bot
    bot = EnsembleTradingBot(
        symbol=args.symbol,
        window_size=args.window,
        check_interval=args.interval,
        confidence_threshold=args.threshold,
        xgboost_threshold=args.xgb_threshold,
        trade_amount=args.amount,
        mock_mode=args.mock_mode,
        train_if_missing=args.train_if_missing
    )
    
    # Run the bot
    try:
        bot.run()
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1) 