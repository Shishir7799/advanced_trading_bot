import time
import logging
import os
import pandas as pd
import argparse
import signal
import sys
import traceback
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
import json

from fetch_crypto_prices import PriceFetcher
from price_prediction_model import PricePredictor
from xgboost_price_predictor import XGBoostPredictor
from aptos_transaction import AptosTransactionManager

# Import modules for Allora and secure predictions
try:
    from allora_integration import fetch_allora_signal
    from secure_predictions import run_secure_model_prediction
    ALLORA_AVAILABLE = True
    TEE_AVAILABLE = True
except ImportError:
    ALLORA_AVAILABLE = False
    TEE_AVAILABLE = False
    logging.warning("Allora and/or TEE modules not available. Some features will be disabled.")

# Configure enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("advanced_trading_bot.log"),
        logging.StreamHandler()
    ]
)

# Create a separate DEBUG level log file for detailed debugging
debug_handler = logging.FileHandler("advanced_trading_bot_debug.log")
debug_handler.setLevel(logging.DEBUG)
debug_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

# Get logger
logger = logging.getLogger(__name__)
logger.addHandler(debug_handler)

class AdvancedTradingBot:
    def __init__(self, 
                 symbol: str = "BTCUSDT", 
                 window_size: int = 5, 
                 check_interval: int = 10,
                 confidence_threshold: float = 0.6,
                 trade_amount: float = 0.01,
                 mock_mode: bool = True,
                 model_type: str = "ensemble",
                 train_if_missing: bool = True,
                 use_allora: bool = False,
                 use_tee: bool = False,
                 voting_only: bool = False,
                 once: bool = False):
        """
        Initialize the advanced trading bot
        
        Args:
            symbol: Trading pair symbol
            window_size: Size of price window for predictions
            check_interval: Time between price checks in seconds
            confidence_threshold: Minimum confidence for executing trades
            trade_amount: Amount to trade per transaction
            mock_mode: If True, simulate transactions
            model_type: Type of model to use (rf, xgboost, ensemble)
            train_if_missing: If True, train model if missing
            use_allora: If True, include Allora signals in ensemble voting
            use_tee: If True, include TEE secure predictions in ensemble voting
            voting_only: If True, only perform voting, no trades
            once: If True, run only one iteration then exit
        """
        self.symbol = symbol
        self.window_size = window_size
        self.check_interval = check_interval
        self.confidence_threshold = confidence_threshold
        self.trade_amount = trade_amount
        self.mock_mode = mock_mode
        self.model_type = model_type
        self.train_if_missing = train_if_missing
        self.use_allora = use_allora and ALLORA_AVAILABLE
        self.use_tee = use_tee and TEE_AVAILABLE
        self.voting_only = voting_only
        self.once = once
        
        # If modules aren't available, log warnings
        if use_allora and not ALLORA_AVAILABLE:
            logger.warning("Allora module not available, disabling Allora integration")
        
        if use_tee and not TEE_AVAILABLE:
            logger.warning("TEE module not available, disabling secure prediction")
        
        # Initialize components
        self.price_fetcher = PriceFetcher(symbol=symbol)
        self.aptos_manager = AptosTransactionManager(mock_mode=mock_mode)
        
        # Ensure necessary directories exist
        os.makedirs("logs", exist_ok=True)
        os.makedirs("logs/decision_logs", exist_ok=True)
        os.makedirs("models", exist_ok=True)
        
        # Initialize prediction models
        self.rf_predictor = PricePredictor(symbol=symbol, window_size=window_size)
        self.xgb_predictor = XGBoostPredictor(symbol=symbol, window_size=window_size)
        
        # Ensure models are available (train if needed and allowed)
        if train_if_missing:
            self._ensure_models_available()
        
        # Statistics & tracking
        self.running = False
        self.start_time = None
        self.last_prediction_time = None
        self.last_trade_time = None
        self.prediction_cooldown = timedelta(seconds=3)
        self.trade_cooldown = timedelta(seconds=10)
        self.loop_count = 0
        self.error_count = 0
        self.total_prediction_time = 0
        self.predictions = []
        self.trades = []
        
        # Register SIGINT handler for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        
        logger.info(f"Initializing advanced trading bot with: symbol={symbol}, window_size={window_size}, "
                   f"check_interval={check_interval}s, confidence_threshold={confidence_threshold}, "
                   f"trade_amount={trade_amount}, mock_mode={mock_mode}, model_type={model_type}")
    
    def _ensure_models_available(self):
        """Check if the required models exist and train them if needed"""
        logger.info(f"Checking availability of requested model(s): {self.model_type}")
        
        rf_ready = self.rf_predictor.is_model_ready()
        xgb_ready = self.xgb_predictor.is_model_ready()
        
        logger.info(f"RandomForest model ready: {rf_ready}")
        logger.info(f"XGBoost model ready: {xgb_ready}")
        
        # Check if required models are available
        if self.model_type == 'rf' and not rf_ready:
            logger.warning("RandomForest model requested but not available")
            if self.train_if_missing:
                self._train_rf_model()
        elif self.model_type == 'xgboost' and not xgb_ready:
            logger.warning("XGBoost model requested but not available")
            if self.train_if_missing:
                self._train_xgb_model()
        elif self.model_type == 'ensemble' and (not rf_ready or not xgb_ready):
            logger.warning("Ensemble requested but not all models available")
            if self.train_if_missing:
                if not rf_ready:
                    self._train_rf_model()
                if not xgb_ready:
                    self._train_xgb_model()
    
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
            filename = f"data/{self.symbol}_{self.model_type}_performance_{datetime.now().strftime('%Y%m%d')}.csv"
            df.to_csv(filename, index=False)
            logger.info(f"Performance data saved to {filename}")
            
            # Log statistics summary
            total_runtime = (datetime.now() - self.start_time).total_seconds()
            avg_prediction_time = self.total_prediction_time / max(1, len(self.predictions))
            avg_trade_time = self.total_trade_time / max(1, sum(1 for p in self.predictions if p.get('trade_executed', False)))
            
            logger.info(f"Bot statistics: loops={self.loop_count}, runtime={total_runtime:.1f}s, "
                       f"predictions={len(self.predictions)}, errors={self.error_count}, "
                       f"avg_prediction_time={avg_prediction_time:.3f}s, avg_trade_time={avg_trade_time:.3f}s")
            
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
    
    def ensemble_voting(self, 
                       ml_prediction: str, 
                       ml_confidence: float,
                       allora_signal: Optional[str] = None, 
                       allora_confidence: Optional[float] = None,
                       tee_output: Optional[str] = None, 
                       tee_confidence: Optional[float] = None,
                       price_window: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Implement ensemble voting with majority rule between ML, Allora, and TEE signals
        
        Args:
            ml_prediction: ML model prediction ("UP" or "DOWN")
            ml_confidence: ML model confidence
            allora_signal: Allora prediction signal ("UP", "DOWN", or None)
            allora_confidence: Allora prediction confidence
            tee_output: TEE secure prediction ("UP", "DOWN", or None)
            tee_confidence: TEE prediction confidence
            price_window: Current price window for context
            
        Returns:
            Dict with voting results and final decision
        """
        logger.info("Performing ensemble voting with available signals")
        
        # Initialize vote tracking
        votes = {"UP": 0, "DOWN": 0, "HOLD": 0}
        vote_details = {}
        voter_confidences = {}
        
        # Count ML vote
        votes[ml_prediction] += 1
        vote_details["ML"] = ml_prediction
        voter_confidences["ML"] = ml_confidence
        
        # Count Allora vote if available
        if allora_signal and allora_confidence:
            votes[allora_signal] += 1
            vote_details["Allora"] = allora_signal
            voter_confidences["Allora"] = allora_confidence
        
        # Count TEE vote if available
        if tee_output and tee_confidence:
            votes[tee_output] += 1
            vote_details["TEE"] = tee_output
            voter_confidences["TEE"] = tee_confidence
        
        # Determine the total number of voters
        total_voters = len(vote_details)
        
        # Calculate average confidence for each direction
        direction_confidences = {}
        for direction in ["UP", "DOWN"]:
            confidences = [
                voter_confidences[voter] for voter, vote in vote_details.items() 
                if vote == direction
            ]
            if confidences:
                direction_confidences[direction] = sum(confidences) / len(confidences)
            else:
                direction_confidences[direction] = 0.0
        
        # Determine the vote winner
        if votes["UP"] > votes["DOWN"]:
            decision = "BUY"
            confidence = direction_confidences["UP"]
        elif votes["DOWN"] > votes["UP"]:
            decision = "HOLD"  # or SELL, but requirements specified HOLD
            confidence = direction_confidences["DOWN"]
        elif votes["UP"] == votes["DOWN"]:
            # Tie-breaking based on confidence
            if direction_confidences.get("UP", 0) > direction_confidences.get("DOWN", 0):
                decision = "BUY"
                confidence = direction_confidences["UP"]
            else:
                decision = "HOLD"
                confidence = direction_confidences["DOWN"]
        else:
            decision = "HOLD"
            confidence = 0.5
        
        # Create the result object
        result = {
            "decision": decision,
            "confidence": confidence,
            "votes": votes,
            "vote_details": vote_details,
            "voter_confidences": voter_confidences,
            "total_voters": total_voters,
            "direction_confidences": direction_confidences,
            "timestamp": datetime.now().isoformat()
        }
        
        # Log the voting results
        log_lines = [
            f"Ensemble Voting Results:",
            f"- ML Vote: {vote_details.get('ML', 'N/A')} ({voter_confidences.get('ML', 0):.2f})",
            f"- Allora Vote: {vote_details.get('Allora', 'N/A')} ({voter_confidences.get('Allora', 0):.2f})",
            f"- TEE Vote: {vote_details.get('TEE', 'N/A')} ({voter_confidences.get('TEE', 0):.2f})",
            f"- UP Votes: {votes['UP']}, DOWN Votes: {votes['DOWN']}",
            f"- Decision: {decision} with confidence {confidence:.2f}"
        ]
        
        for line in log_lines:
            logger.info(line)
        
        # Pretty print for console
        print("\n" + "="*50)
        print("ENSEMBLE VOTING RESULTS")
        print("-" * 30)
        print(f"ML Vote:     {vote_details.get('ML', 'N/A'):4} ({voter_confidences.get('ML', 0):.2f})")
        print(f"Allora Vote: {vote_details.get('Allora', 'N/A'):4} ({voter_confidences.get('Allora', 0):.2f})")
        print(f"TEE Vote:    {vote_details.get('TEE', 'N/A'):4} ({voter_confidences.get('TEE', 0):.2f})")
        print("-" * 30)
        print(f"UP Votes:    {votes['UP']}")
        print(f"DOWN Votes:  {votes['DOWN']}")
        print("-" * 30)
        print(f"FINAL DECISION: {decision} with confidence {confidence:.2f}")
        print("="*50 + "\n")
        
        return result

    def make_prediction(self) -> Optional[dict]:
        """Make a price prediction based on current window using the specified model"""
        prediction_start_time = time.time()
        
        # Check if we have a cooldown period for predictions
        now = datetime.now()
        if self.last_prediction_time and now - self.last_prediction_time < self.prediction_cooldown:
            logger.debug(f"Prediction cooldown active. Last prediction at {self.last_prediction_time}")
            return None
            
        # Get price window
        price_window = self.get_price_window()
        if not price_window:
            return None
        
        # Make predictions based on model type
        rf_prediction = None
        rf_confidence = 0.0
        xgb_prediction = None
        xgb_confidence = 0.0
        
        prediction_data = {
            "timestamp": int(time.time()),
            "datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
            "price_window": price_window,
            "current_price": price_window[-1],
            "model_type": self.model_type
        }
        
        # Get RandomForest prediction if needed
        if self.model_type in ['rf', 'ensemble']:
            if not self.rf_predictor.is_model_ready():
                logger.error("RandomForest model not ready")
                if self.model_type == 'rf':
                    return None
            else:
                logger.debug(f"Making RandomForest prediction with window: {price_window}")
                rf_prediction, rf_confidence = self.rf_predictor.predict(price_window)
                prediction_data["rf_prediction"] = "UP" if rf_prediction == 1 else "DOWN"
                prediction_data["rf_confidence"] = rf_confidence
        
        # Get XGBoost prediction if needed
        if self.model_type in ['xgboost', 'ensemble']:
            if not self.xgb_predictor.is_model_ready():
                logger.error("XGBoost model not ready")
                if self.model_type == 'xgboost':
                    return None
            else:
                logger.debug(f"Making XGBoost prediction with window: {price_window}")
                xgb_prediction, xgb_confidence = self.xgb_predictor.predict(price_window)
                prediction_data["xgb_prediction"] = "UP" if xgb_prediction == 1 else "DOWN"
                prediction_data["xgb_confidence"] = xgb_confidence
        
        # Determine combined ML prediction
        if self.model_type == 'rf':
            ml_prediction = "UP" if rf_prediction == 1 else "DOWN"
            ml_confidence = rf_confidence
        elif self.model_type == 'xgboost':
            ml_prediction = "UP" if xgb_prediction == 1 else "DOWN"
            ml_confidence = xgb_confidence
        elif self.model_type == 'ensemble':
            # For ML ensemble, use the model with higher confidence
            if rf_confidence >= xgb_confidence:
                ml_prediction = "UP" if rf_prediction == 1 else "DOWN"
                ml_confidence = rf_confidence
                prediction_data["used_model"] = "RandomForest"
            else:
                ml_prediction = "UP" if xgb_prediction == 1 else "DOWN"
                ml_confidence = xgb_confidence
                prediction_data["used_model"] = "XGBoost"
        else:
            logger.error(f"Unknown model type: {self.model_type}")
            return None
        
        # Try to get Allora prediction if available
        allora_prediction = None
        allora_confidence = None
        if self.use_allora:
            try:
                # Use 1h timeframe for Allora
                allora_signal_data = fetch_allora_signal(timeframe="1h", use_mock=True)
                if allora_signal_data and len(allora_signal_data) == 3:
                    allora_pred_value, allora_confidence, allora_metadata = allora_signal_data
                    allora_prediction = "UP" if allora_pred_value == 1 else "DOWN"
                    prediction_data["allora_prediction"] = allora_prediction
                    prediction_data["allora_confidence"] = allora_confidence
                    prediction_data["allora_metadata"] = allora_metadata
                    logger.info(f"Allora prediction: {allora_prediction} with {allora_confidence:.2f} confidence")
                else:
                    logger.warning("Invalid Allora prediction data format")
            except Exception as e:
                logger.error(f"Error fetching Allora prediction: {e}")
        
        # Try to get TEE secure prediction if available
        tee_prediction = None
        tee_confidence = None
        tee_metadata = None
        if self.use_tee:
            try:
                # Convert price window to appropriate format for secure prediction
                # Create mock DataFrame if TEE requires it (this depends on your TEE implementation)
                tee_df = pd.DataFrame({
                    'close': price_window,
                    'high': [p * 1.01 for p in price_window],
                    'low': [p * 0.99 for p in price_window],
                    'open': [p * 0.995 for p in price_window],
                    'volume': [1000 for _ in price_window]
                })
                
                logger.info(f"Requesting secure prediction from TEE...")
                # Get prediction from TEE
                tee_pred_value, tee_confidence, tee_metadata = run_secure_model_prediction(
                    price_window=tee_df,
                    model_id="btc_price_predictor_v1"
                )
                
                tee_prediction = "UP" if tee_pred_value == 1 else "DOWN"
                prediction_data["tee_prediction"] = tee_prediction
                prediction_data["tee_confidence"] = tee_confidence
                prediction_data["tee_metadata"] = tee_metadata
                
                logger.info(f"TEE prediction: {tee_prediction} with {tee_confidence:.2f} confidence")
                if tee_metadata and 'enclave_id' in tee_metadata:
                    logger.info(f"TEE enclave ID: {tee_metadata['enclave_id']}")
            except Exception as e:
                logger.error(f"Error running secure prediction: {e}")
                logger.debug(f"TEE error details: {traceback.format_exc()}")
        
        # Run ensemble voting with all available signals
        voting_result = self.ensemble_voting(
            ml_prediction=ml_prediction,
            ml_confidence=ml_confidence,
            allora_signal=allora_prediction,
            allora_confidence=allora_confidence,
            tee_output=tee_prediction,
            tee_confidence=tee_confidence,
            price_window=price_window
        )
        
        # Update prediction data with voting results
        prediction_data["voting_result"] = voting_result
        prediction_data["final_prediction"] = voting_result["decision"]
        prediction_data["final_confidence"] = voting_result["confidence"]
        prediction_data["threshold_met"] = voting_result["confidence"] >= self.confidence_threshold
        prediction_data["prediction_time"] = time.time() - prediction_start_time
        
        # Update last prediction time
        self.last_prediction_time = now
        
        # Track prediction time
        prediction_time = time.time() - prediction_start_time
        self.total_prediction_time += prediction_time
        
        # Add to predictions list
        self.predictions.append(prediction_data)
        
        # Log detailed prediction info
        logger.info(f"Final decision from ensemble voting: {prediction_data['final_prediction']} with {prediction_data['final_confidence']:.2f} confidence")
        logger.debug(f"Full prediction data: {prediction_data}")
        logger.debug(f"Prediction completed in {prediction_time:.3f}s")
        
        return prediction_data
    
    def execute_trade(self, prediction_data: dict) -> bool:
        """Execute a trade based on prediction"""
        trade_start_time = time.time()
        logger.debug(f"Evaluating trade for prediction: {prediction_data['final_prediction']} with confidence {prediction_data['final_confidence']:.3f}")
        
        # Save trade decision logs to logs/decision_logs
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        decision_log_file = f"logs/decision_logs/trade_decision_{timestamp}.json"
        
        # Create detailed metadata for logging
        decision_metadata = {
            "timestamp": timestamp,
            "unix_timestamp": int(time.time()),
            "symbol": self.symbol,
            "current_price": prediction_data['current_price'],
            "model_type": self.model_type,
            "prediction": prediction_data['final_prediction'],
            "confidence": prediction_data['final_confidence'],
            "threshold_met": prediction_data.get('threshold_met', False),
            "voting_data": prediction_data.get('voting_result', {})
        }
        
        # Add ML model predictions if available
        if 'rf_prediction' in prediction_data:
            decision_metadata["ml_signals"] = decision_metadata.get("ml_signals", {})
            decision_metadata["ml_signals"]["random_forest"] = {
                "prediction": prediction_data['rf_prediction'],
                "confidence": prediction_data['rf_confidence']
            }
        
        if 'xgb_prediction' in prediction_data:
            decision_metadata["ml_signals"] = decision_metadata.get("ml_signals", {})
            decision_metadata["ml_signals"]["xgboost"] = {
                "prediction": prediction_data['xgb_prediction'],
                "confidence": prediction_data['xgb_confidence']
            }
        
        # Add Allora data if available
        if 'allora_prediction' in prediction_data:
            decision_metadata["allora_signal"] = {
                "prediction": prediction_data['allora_prediction'],
                "confidence": prediction_data['allora_confidence'],
                "metadata": prediction_data.get('allora_metadata', {})
            }
        
        # Add TEE data if available
        if 'tee_prediction' in prediction_data:
            decision_metadata["tee_signal"] = {
                "prediction": prediction_data['tee_prediction'],
                "confidence": prediction_data['tee_confidence'],
                "metadata": prediction_data.get('tee_metadata', {})
            }
        
        # Save decision log
        try:
            with open(decision_log_file, 'w') as f:
                json.dump(decision_metadata, f, indent=2)
            logger.info(f"Trade decision logged to {decision_log_file}")
        except Exception as e:
            logger.error(f"Failed to save trade decision log: {e}")
        
        # Check if we're in voting-only mode
        if self.voting_only:
            logger.info(f"Voting-only mode: Would recommend {prediction_data['final_prediction']} for {self.symbol} at ${prediction_data['current_price']:.2f} with {prediction_data['final_confidence']:.2f} confidence")
            print(f"\n{'*'*50}")
            print(f"TRADING RECOMMENDATION (VOTING-ONLY MODE)")
            print(f"ACTION: {prediction_data['final_prediction']} {self.symbol} at ${prediction_data['current_price']:.2f}")
            print(f"CONFIDENCE: {prediction_data['final_confidence']:.2f}")
            print(f"{'*'*50}\n")
            return False
        
        # Check if we have a cooldown period for trades
        now = datetime.now()
        if self.last_trade_time and now - self.last_trade_time < self.trade_cooldown:
            logger.info(f"Trade cooldown period active. Last trade at {self.last_trade_time}. Skipping trade.")
            return False
            
        # Check if prediction meets confidence threshold
        if not prediction_data["threshold_met"]:
            logger.info(f"Prediction confidence {prediction_data['final_confidence']:.2f} below threshold "
                       f"{self.confidence_threshold}. No trade executed.")
            return False
            
        # Execute buy transaction if prediction is BUY
        if prediction_data["final_prediction"] == "BUY":
            logger.debug(f"Attempting to execute BUY transaction for {self.symbol} at ${prediction_data['current_price']:.2f}")
            result = self.aptos_manager.execute_buy_transaction(
                symbol=self.symbol,
                price=prediction_data["current_price"],
                amount=self.trade_amount,
                confidence=prediction_data["final_confidence"]
            )
            
            if result:
                trade_time = time.time() - trade_start_time
                self.total_trade_time += trade_time
                
                print(f"\n{'*'*50}")
                print(f"BUY TRANSACTION EXECUTED: {result['transaction_id']}")
                print(f"AMOUNT: {self.trade_amount} BTC at ${prediction_data['current_price']:.2f}")
                print(f"BASED ON: Ensemble voting with {prediction_data['voting_result']['total_voters']} signals")
                print(f"CONFIDENCE: {prediction_data['final_confidence']:.2f}")
                print(f"EXECUTION TIME: {trade_time:.3f}s")
                print(f"{'*'*50}\n")
                
                logger.info(f"Buy transaction executed: {result['transaction_id']} for {self.trade_amount} {self.symbol} at ${prediction_data['current_price']:.2f}")
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
        logger.info(f"Advanced trading bot started at {self.start_time}")
        
        print(f"\n{'#'*70}")
        print(f"STARTING ADVANCED AI CRYPTO TRADING BOT")
        print(f"Symbol: {self.symbol}")
        print(f"Model: {self.model_type.upper()}")
        print(f"Check interval: {self.check_interval} seconds")
        print(f"Confidence threshold: {self.confidence_threshold}")
        print(f"Trade amount: {self.trade_amount} BTC")
        print(f"Mock mode: {self.mock_mode}")
        print(f"Voting-only mode: {self.voting_only}")
        
        # Display enabled voting signals
        print(f"\nENSEMBLE VOTING SIGNALS:")
        print(f"- ML Models: Enabled")
        print(f"- Allora Oracle: {'Enabled' if self.use_allora else 'Disabled'}")
        print(f"- TEE Secure Prediction: {'Enabled' if self.use_tee else 'Disabled'}")
        
        print(f"\nStart time: {self.start_time}")
        print(f"{'#'*70}\n")
        
        # Log enabled voting signals
        logger.info(f"Ensemble voting configuration - ML: Enabled, Allora: {'Enabled' if self.use_allora else 'Disabled'}, TEE: {'Enabled' if self.use_tee else 'Disabled'}")
        
        # Check if models are ready
        if (self.model_type == 'rf' and not self.rf_predictor.is_model_ready()) or \
           (self.model_type == 'xgboost' and not self.xgb_predictor.is_model_ready()) or \
           (self.model_type == 'ensemble' and (not self.rf_predictor.is_model_ready() or not self.xgb_predictor.is_model_ready())):
            logger.error(f"Required models for {self.model_type} are not ready. Exiting.")
            return
        
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
                # Make prediction
                prediction_data = self.make_prediction()
                
                # Execute trade if prediction is valid
                if prediction_data:
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
            
            # Check if we should exit after one iteration
            if self.once:
                self.running = False
        
        # Bot shutdown
        end_time = datetime.now()
        runtime = (end_time - self.start_time).total_seconds()
        
        # Save performance data on exit
        self._save_performance_data()
        
        logger.info(f"Trading bot stopped at {end_time}. Total runtime: {runtime:.1f}s")
        logger.info(f"Total loops: {self.loop_count}, Predictions: {len(self.predictions)}, Errors: {self.error_count}")
        print(f"\nTrading bot stopped at {end_time}. Runtime: {runtime:.1f}s")
        print(f"Total loops: {self.loop_count}, Predictions: {len(self.predictions)}, Errors: {self.error_count}")
        print("Performance data saved.")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Advanced Crypto Trading Bot')
    
    parser.add_argument('--symbol', type=str, default='BTCUSDT',
                        help='Trading pair symbol (default: BTCUSDT)')
    
    parser.add_argument('--window', type=int, default=5,
                        help='Price window size for prediction (default: 5)')
    
    parser.add_argument('--interval', type=int, default=10,
                        help='Time between price checks in seconds (default: 10)')
    
    parser.add_argument('--threshold', type=float, default=0.6,
                        help='Confidence threshold for trading (default: 0.6)')
    
    parser.add_argument('--amount', type=float, default=0.01,
                        help='Amount to trade per transaction (default: 0.01)')
    
    parser.add_argument('--real', action='store_false', dest='mock_mode',
                        help='Use real transactions instead of mock mode')
    
    parser.add_argument('--model', type=str, choices=['rf', 'xgboost', 'ensemble'], default='ensemble',
                        help='Model to use for predictions (default: ensemble)')
    
    parser.add_argument('--no-train', action='store_false', dest='train_if_missing',
                        help='Do not train model if missing, just exit')
    
    parser.add_argument('--force-train-rf', action='store_true',
                        help='Force training RandomForest model before starting')
    
    parser.add_argument('--force-train-xgb', action='store_true',
                        help='Force training XGBoost model before starting')
    
    # New arguments for ensemble voting
    parser.add_argument('--use-allora', action='store_true',
                        help='Include Allora signals in ensemble voting')
    
    parser.add_argument('--use-tee', action='store_true',
                        help='Include TEE secure predictions in ensemble voting')
    
    parser.add_argument('--voting-only', action='store_true',
                        help='Only perform voting, no trades')
    
    parser.add_argument('--once', action='store_true',
                        help='Run only one iteration then exit (useful for testing)')
    
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
    
    # Check if Allora and TEE modules are available if requested
    if args.use_allora and not ALLORA_AVAILABLE:
        logger.warning("--use-allora was specified but Allora module is not available")
        
    if args.use_tee and not TEE_AVAILABLE:
        logger.warning("--use-tee was specified but TEE module is not available")
    
    # Create and run the trading bot
    bot = AdvancedTradingBot(
        symbol=args.symbol,
        window_size=args.window,
        check_interval=args.interval,
        confidence_threshold=args.threshold,
        trade_amount=args.amount,
        mock_mode=args.mock_mode,
        model_type=args.model,
        train_if_missing=args.train_if_missing,
        use_allora=args.use_allora,
        use_tee=args.use_tee,
        voting_only=args.voting_only,
        once=args.once
    )
    
    # Run the bot
    try:
        bot.run()
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1) 