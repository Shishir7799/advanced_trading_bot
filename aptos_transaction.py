import logging
import time
import uuid
import os
from typing import Dict, Any, Optional
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AptosTransactionManager:
    def __init__(self, 
                 mock_mode: bool = True,
                 transaction_log_dir: str = "transaction_logs"):
        self.mock_mode = mock_mode
        self.transaction_log_dir = transaction_log_dir
        
        # Create log directory if it doesn't exist
        os.makedirs(transaction_log_dir, exist_ok=True)
        
        # Simple mock wallet data
        self.mock_wallet = {
            "address": "0x1a2b3c4d5e6f",
            "private_key": "MOCK_PRIVATE_KEY_DO_NOT_USE_IN_PRODUCTION",
            "public_key": "MOCK_PUBLIC_KEY",
            "initial_balance": 1000.0  # Mock USDT balance
        }
        
        # Transaction history
        self.transaction_history = []
        
    def execute_buy_transaction(self, 
                               symbol: str, 
                               price: float, 
                               amount: float = 0.01,
                               confidence: float = 0.0) -> Optional[Dict[str, Any]]:
        """
        Execute a buy transaction on Aptos blockchain
        
        Args:
            symbol: Trading pair symbol (e.g. BTCUSDT)
            price: Current price
            amount: Amount to buy (in BTC)
            confidence: Model prediction confidence
            
        Returns:
            Transaction details if successful, None otherwise
        """
        transaction_id = str(uuid.uuid4())
        timestamp = int(time.time())
        
        # Calculate total cost
        total_cost = price * amount
        
        logger.info(f"Executing {'MOCK ' if self.mock_mode else ''}BUY transaction: "
                   f"{amount} {symbol} at {price}")
        
        # Create transaction details
        transaction = {
            "transaction_id": transaction_id,
            "timestamp": timestamp,
            "datetime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp)),
            "type": "BUY",
            "symbol": symbol,
            "price": price,
            "amount": amount,
            "total_cost": total_cost,
            "prediction_confidence": confidence,
            "status": "PENDING"
        }
        
        try:
            if not self.mock_mode:
                # In a real implementation, this would connect to Aptos blockchain
                # and submit a real transaction
                # For now, we'll just simulate success
                pass
                
            # Simulate transaction processing time
            time.sleep(1)
            
            # Update transaction status
            transaction["status"] = "COMPLETED"
            
            # Add to transaction history
            self.transaction_history.append(transaction)
            
            # Log the transaction
            self._log_transaction(transaction)
            
            logger.info(f"Transaction {transaction_id} completed successfully")
            
            return transaction
            
        except Exception as e:
            logger.error(f"Transaction failed: {e}")
            transaction["status"] = "FAILED"
            transaction["error"] = str(e)
            self._log_transaction(transaction)
            return None
    
    def execute_sell_transaction(self, 
                                symbol: str, 
                                price: float, 
                                amount: float = 0.01,
                                confidence: float = 0.0) -> Optional[Dict[str, Any]]:
        """
        Execute a sell transaction on Aptos blockchain (mocked for testing)
        
        Args:
            symbol: Trading pair symbol (e.g. BTCUSDT)
            price: Current price
            amount: Amount to sell (in BTC)
            confidence: Model prediction confidence
            
        Returns:
            Transaction details if successful, None otherwise
        """
        # Implementation similar to buy but with "SELL" type
        transaction_id = str(uuid.uuid4())
        timestamp = int(time.time())
        
        # Calculate total value
        total_value = price * amount
        
        logger.info(f"Executing {'MOCK ' if self.mock_mode else ''}SELL transaction: "
                   f"{amount} {symbol} at {price}")
        
        # Create transaction details
        transaction = {
            "transaction_id": transaction_id,
            "timestamp": timestamp,
            "datetime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp)),
            "type": "SELL",
            "symbol": symbol,
            "price": price,
            "amount": amount,
            "total_value": total_value,
            "prediction_confidence": confidence,
            "status": "PENDING"
        }
        
        try:
            if not self.mock_mode:
                # In a real implementation, this would connect to Aptos blockchain
                # and submit a real transaction
                pass
                
            # Simulate transaction processing time
            time.sleep(1)
            
            # Update transaction status
            transaction["status"] = "COMPLETED"
            
            # Add to transaction history
            self.transaction_history.append(transaction)
            
            # Log the transaction
            self._log_transaction(transaction)
            
            logger.info(f"Transaction {transaction_id} completed successfully")
            
            return transaction
            
        except Exception as e:
            logger.error(f"Transaction failed: {e}")
            transaction["status"] = "FAILED"
            transaction["error"] = str(e)
            self._log_transaction(transaction)
            return None
    
    def _log_transaction(self, transaction: Dict[str, Any]) -> None:
        """Log a transaction to a file"""
        try:
            filename = os.path.join(
                self.transaction_log_dir, 
                f"tx_{transaction['transaction_id']}.json"
            )
            
            with open(filename, 'w') as f:
                json.dump(transaction, f, indent=2)
                
            logger.debug(f"Transaction logged to {filename}")
        except Exception as e:
            logger.error(f"Failed to log transaction: {e}")
    
    def get_transaction_history(self) -> list:
        """Get the transaction history"""
        return self.transaction_history
    
    def get_transaction_by_id(self, transaction_id: str) -> Optional[Dict[str, Any]]:
        """Get a transaction by ID"""
        for tx in self.transaction_history:
            if tx["transaction_id"] == transaction_id:
                return tx
        return None

# Test the Aptos transaction manager
if __name__ == "__main__":
    tx_manager = AptosTransactionManager()
    
    # Test a buy transaction
    buy_tx = tx_manager.execute_buy_transaction(
        symbol="BTCUSDT",
        price=19500.75,
        amount=0.01,
        confidence=0.85
    )
    
    # Test a sell transaction
    sell_tx = tx_manager.execute_sell_transaction(
        symbol="BTCUSDT",
        price=19600.50,
        amount=0.01,
        confidence=0.78
    )
    
    # Print transaction history
    for tx in tx_manager.get_transaction_history():
        print(f"{tx['type']} {tx['amount']} {tx['symbol']} at {tx['price']} " +
              f"(ID: {tx['transaction_id'][:8]})") 