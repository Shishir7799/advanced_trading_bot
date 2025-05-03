import os
import time
import logging
import json
import argparse
from datetime import datetime
from typing import Dict, Any, Tuple

# Import custom modules
from fetch_crypto_prices import PriceFetcher
from technical_indicators import TechnicalIndicators
from ensemble_predictor import EnsemblePredictor
from aptos_integration import AptosIntegration
from agent_metadata import AgentMetadata

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TradingBot:
    """Advanced crypto trading bot with ML, blockchain integration and Allora oracle"""
    
    def __init__(self, 
                 symbol: str = "BTCUSDT", 
                 check_interval: int = 300, 
                 trade_amount_pct: float = 10.0,
                 ensemble_threshold: float = 0.65,
                 use_allora: bool = True,
                 require_allora_agreement: bool = True,
                 use_blockchain: bool = False,
                 use_secure_enclave: bool = False,
                 secure_model_id: str = "btc_price_predictor_v1",
                 wallet_address: str = None,
                 bot_name: str = "CryptoTradingBot",
                 version: str = "1.0.0",
                 enable_ipfs: bool = False):
        """
        Initialize the trading bot
        
        Args:
            symbol: Trading pair symbol
            check_interval: Seconds between price checks
            trade_amount_pct: Percentage of balance to trade
            ensemble_threshold: Confidence threshold for trading
            use_allora: Whether to include Allora predictions
            require_allora_agreement: Only trade if Allora agrees
            use_blockchain: Whether to use blockchain for transactions
            use_secure_enclave: Whether to use secure model predictions
            secure_model_id: Model ID to use in secure enclave
            wallet_address: Wallet address for the trading bot
            bot_name: Name of the trading bot
            version: Version of the trading bot
            enable_ipfs: Whether to enable IPFS logging
        """
        self.symbol = symbol
        self.check_interval = check_interval
        self.trade_amount_pct = trade_amount_pct
        self.use_blockchain = use_blockchain
        
        # Determine strategy ID based on configuration
        strategy_id = self._determine_strategy_id(use_allora, use_secure_enclave, require_allora_agreement)
        
        # Initialize agent metadata
        self.agent_metadata = AgentMetadata(
            wallet_address=wallet_address,
            bot_name=bot_name,
            version=version,
            strategy_id=strategy_id
        )
        
        # Enable IPFS if requested
        if enable_ipfs:
            self.agent_metadata.enable_ipfs_logging(True)
        
        # Initialize components
        self.price_fetcher = PriceFetcher(symbol=symbol)
        
        # Initialize ensemble predictor
        self.predictor = EnsemblePredictor(
            symbol=symbol,
            use_allora=use_allora,
            require_allora_agreement=require_allora_agreement,
            ensemble_threshold=ensemble_threshold,
            use_secure_enclave=use_secure_enclave,
            secure_model_id=secure_model_id
        )
        
        # Initialize blockchain integration if enabled
        self.blockchain = None
        if use_blockchain:
            try:
                self.blockchain = AptosIntegration()
                logger.info(f"Blockchain integration initialized")
            except Exception as e:
                logger.error(f"Failed to initialize blockchain integration: {e}")
                self.use_blockchain = False
        
        # Create transaction log directory
        os.makedirs("transaction_logs", exist_ok=True)
        
        logger.info(f"Trading bot initialized for {symbol}")
        logger.info(f"Check interval: {check_interval} seconds")
        logger.info(f"Trade amount: {trade_amount_pct}% of balance")
        logger.info(f"Using Allora predictions: {use_allora}")
        logger.info(f"Require Allora agreement: {require_allora_agreement}")
        logger.info(f"Using secure enclave: {use_secure_enclave}")
        if use_secure_enclave:
            logger.info(f"Secure model ID: {secure_model_id}")
        logger.info(f"Using blockchain: {self.use_blockchain}")
        logger.info(f"Agent ID: {self.agent_metadata.agent_id}")
        logger.info(f"Wallet address: {self.agent_metadata.wallet_address}")
    
    def _determine_strategy_id(self, use_allora: bool, use_secure_enclave: bool, require_allora_agreement: bool) -> str:
        """Determine strategy ID based on configuration"""
        if use_secure_enclave and use_allora:
            return "secure_ensemble" if require_allora_agreement else "hybrid_ensemble"
        elif use_secure_enclave:
            return "secure_ensemble"
        elif use_allora:
            return "allora_hybrid" if require_allora_agreement else "adaptive_ensemble"
        else:
            return "adaptive_ensemble"
    
    def start(self):
        """Start the trading bot"""
        logger.info("Starting trading bot...")
        
        while True:
            try:
                # Fetch the latest price data
                df = self.price_fetcher.fetch_recent_prices(limit=100)
                
                if df is not None and not df.empty:
                    # Add technical indicators
                    df = TechnicalIndicators.add_all_indicators(df)
                    
                    # Get the current price
                    current_price = df['close'].iloc[-1]
                    logger.info(f"Current {self.symbol} price: {current_price}")
                    
                    # Make prediction
                    prediction, confidence, details = self.predictor.predict(df)
                    
                    # Process prediction
                    self._process_prediction(prediction, confidence, details, current_price)
                    
                else:
                    logger.warning("Failed to fetch price data")
                
                # Wait for next check
                logger.info(f"Waiting {self.check_interval} seconds until next check...")
                time.sleep(self.check_interval)
                
            except KeyboardInterrupt:
                logger.info("Trading bot stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                time.sleep(30)  # Wait before retrying
    
    def _process_prediction(self, prediction: int, confidence: float, 
                           details: Dict[str, Any], current_price: float):
        """
        Process a prediction and execute a trade if appropriate
        
        Args:
            prediction: 1 for buy, 0 for sell, -1 for no action
            confidence: Prediction confidence
            details: Prediction details
            current_price: Current price
        """
        # Format prediction for logging
        pred_str = "BUY" if prediction == 1 else "SELL" if prediction == 0 else "HOLD"
        
        # Log the prediction
        logger.info(f"Prediction: {pred_str} with confidence {confidence:.2f}")
        
        # Log secure enclave details if available
        secure_data = None
        if "models" in details and "secure_model" in details["models"]:
            secure_data = details["models"]["secure_model"].get("metadata", {})
            
            if secure_data:
                logger.info(f"Secure prediction from enclave {secure_data.get('enclave_id', 'unknown')}")
                logger.info(f"Secure prediction signature: {secure_data.get('signature', 'none')}")
        
        # Prepare models and signals information
        models_used = []
        signals = {}
        
        # Extract models and signals
        if "models" in details:
            for model_name, model_data in details["models"].items():
                model_pred = model_data.get("prediction", -1)
                model_conf = model_data.get("confidence", 0.0)
                
                if model_pred != -1:  # Only include models that made a prediction
                    if model_name == "random_forest":
                        models_used.append("RandomForest")
                        pred_str = "buy" if model_pred == 1 else "sell" if model_pred == 0 else "hold"
                        signals["RandomForest"] = {"prediction": pred_str, "confidence": model_conf}
                    
                    elif model_name == "xgboost":
                        models_used.append("XGBoost")
                        pred_str = "buy" if model_pred == 1 else "sell" if model_pred == 0 else "hold"
                        signals["XGBoost"] = {"prediction": pred_str, "confidence": model_conf}
                    
                    elif model_name == "allora":
                        models_used.append("AlloraOracle")
                        pred_str = "buy" if model_pred == 1 else "sell" if model_pred == 0 else "hold"
                        signals["Allora"] = {"prediction": pred_str, "confidence": model_conf}
                    
                    elif model_name == "secure_model":
                        models_used.append("SecureModel")
                        pred_str = "buy" if model_pred == 1 else "sell" if model_pred == 0 else "hold"
                        signals["SecureModel"] = {"prediction": pred_str, "confidence": model_conf}
        
        # Log to agent metadata if we have a clear prediction
        if prediction != -1:
            # Determine direction string
            direction = "buy" if prediction == 1 else "sell"
            
            # Log the analysis decision
            self.agent_metadata.log_trade_decision(
                decision_type="analysis",
                symbol=self.symbol,
                direction=direction,
                confidence=confidence,
                price=current_price,
                models_used=models_used,
                signals=signals,
                secure_data=secure_data
            )
        
        # Check if prediction requires action
        if prediction == -1:
            logger.info("No trading action required")
            return
        
        # Execute trade
        if prediction == 1:
            # Buy signal
            self._execute_buy(confidence, details, current_price, models_used, signals, secure_data)
        else:
            # Sell signal
            self._execute_sell(confidence, details, current_price, models_used, signals, secure_data)
    
    def _execute_buy(self, confidence: float, details: Dict[str, Any], price: float,
                    models_used: list, signals: dict, secure_data: dict = None):
        """Execute a buy order"""
        logger.info(f"Executing BUY at price {price} with confidence {confidence:.2f}")
        
        # Prepare transaction details
        transaction = {
            "type": "BUY",
            "symbol": self.symbol,
            "price": price,
            "amount_pct": self.trade_amount_pct,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat(),
            "prediction_details": details
        }
        
        # Calculate the trade amount (for demonstration)
        trade_amount = 0.01  # Default small amount for demo
        
        # Execute on blockchain if enabled
        tx_hash = None
        if self.use_blockchain and self.blockchain is not None:
            try:
                # Create a small test transaction for demo purposes
                # In a real system, this would execute the actual trade
                tx_hash = self.blockchain.create_test_transaction(0.01)
                transaction["blockchain"] = {
                    "tx_hash": tx_hash,
                    "status": "submitted" if tx_hash else "failed"
                }
                logger.info(f"Blockchain transaction submitted: {tx_hash}")
            except Exception as e:
                logger.error(f"Blockchain transaction failed: {e}")
        
        # Log the transaction
        self._log_transaction(transaction)
        
        # Log the trade in agent metadata
        trade_decision = self.agent_metadata.log_trade_decision(
            decision_type="trade",
            symbol=self.symbol,
            direction="buy",
            confidence=confidence,
            price=price,
            amount=trade_amount,
            models_used=models_used,
            signals=signals,
            secure_data=secure_data,
            extra_data={
                "tx_hash": tx_hash,
                "amount_pct": self.trade_amount_pct
            }
        )
        
        # Optionally upload to IPFS
        if self.agent_metadata.ipfs_enabled:
            single_trade_data = {
                "agent_info": self.agent_metadata.get_agent_info(),
                "trade": trade_decision
            }
            ipfs_hash = self.agent_metadata.upload_to_ipfs(single_trade_data)
            logger.info(f"Trade decision uploaded to IPFS (mock): {ipfs_hash}")
    
    def _execute_sell(self, confidence: float, details: Dict[str, Any], price: float,
                     models_used: list, signals: dict, secure_data: dict = None):
        """Execute a sell order"""
        logger.info(f"Executing SELL at price {price} with confidence {confidence:.2f}")
        
        # Prepare transaction details
        transaction = {
            "type": "SELL",
            "symbol": self.symbol,
            "price": price,
            "amount_pct": self.trade_amount_pct,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat(),
            "prediction_details": details
        }
        
        # Calculate the trade amount (for demonstration)
        trade_amount = 0.01  # Default small amount for demo
        
        # Execute on blockchain if enabled
        tx_hash = None
        if self.use_blockchain and self.blockchain is not None:
            try:
                # Create a small test transaction for demo purposes
                # In a real system, this would execute the actual trade
                tx_hash = self.blockchain.create_test_transaction(0.01)
                transaction["blockchain"] = {
                    "tx_hash": tx_hash,
                    "status": "submitted" if tx_hash else "failed"
                }
                logger.info(f"Blockchain transaction submitted: {tx_hash}")
            except Exception as e:
                logger.error(f"Blockchain transaction failed: {e}")
        
        # Log the transaction
        self._log_transaction(transaction)
        
        # Log the trade in agent metadata
        trade_decision = self.agent_metadata.log_trade_decision(
            decision_type="trade",
            symbol=self.symbol,
            direction="sell",
            confidence=confidence,
            price=price,
            amount=trade_amount,
            models_used=models_used,
            signals=signals,
            secure_data=secure_data,
            extra_data={
                "tx_hash": tx_hash,
                "amount_pct": self.trade_amount_pct
            }
        )
        
        # Optionally upload to IPFS
        if self.agent_metadata.ipfs_enabled:
            single_trade_data = {
                "agent_info": self.agent_metadata.get_agent_info(),
                "trade": trade_decision
            }
            ipfs_hash = self.agent_metadata.upload_to_ipfs(single_trade_data)
            logger.info(f"Trade decision uploaded to IPFS (mock): {ipfs_hash}")
    
    def _log_transaction(self, transaction: Dict[str, Any]):
        """Log a transaction to a file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"transaction_logs/{self.symbol}_{transaction['type']}_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(transaction, f, indent=2)
        
        logger.info(f"Transaction logged to {filename}")
    
    def get_agent_stats(self) -> Dict[str, Any]:
        """Get agent statistics and metadata"""
        return self.agent_metadata.get_stats()
    
    def get_recent_decisions(self, limit: int = 10) -> list:
        """Get recent trading decisions"""
        return self.agent_metadata.get_recent_decisions(limit)


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='AI Crypto Trading Bot with Blockchain Integration')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='Trading pair symbol')
    parser.add_argument('--interval', type=int, default=300, help='Check interval in seconds')
    parser.add_argument('--threshold', type=float, default=0.65, help='Ensemble confidence threshold')
    parser.add_argument('--trade-pct', type=float, default=10.0, help='Percentage of balance to trade')
    parser.add_argument('--use-blockchain', action='store_true', help='Use blockchain for transactions')
    parser.add_argument('--use-allora', action='store_true', help='Use Allora predictive oracle')
    parser.add_argument('--require-allora', action='store_true', help='Require Allora agreement for trades')
    parser.add_argument('--use-secure-enclave', action='store_true', help='Use secure model enclave for predictions')
    parser.add_argument('--secure-model-id', type=str, default='btc_price_predictor_v1', help='Secure model ID to use')
    parser.add_argument('--wallet', type=str, help='Wallet address for the trading bot')
    parser.add_argument('--bot-name', type=str, default='CryptoTradingBot', help='Name of the trading bot')
    parser.add_argument('--version', type=str, default='1.0.0', help='Version of the trading bot')
    parser.add_argument('--enable-ipfs', action='store_true', help='Enable IPFS logging (mock)')
    
    args = parser.parse_args()
    
    # Create and start trading bot
    bot = TradingBot(
        symbol=args.symbol,
        check_interval=args.interval,
        trade_amount_pct=args.trade_pct,
        ensemble_threshold=args.threshold,
        use_allora=args.use_allora,
        require_allora_agreement=args.require_allora,
        use_blockchain=args.use_blockchain,
        use_secure_enclave=args.use_secure_enclave,
        secure_model_id=args.secure_model_id,
        wallet_address=args.wallet,
        bot_name=args.bot_name,
        version=args.version,
        enable_ipfs=args.enable_ipfs
    )
    
    bot.start() 