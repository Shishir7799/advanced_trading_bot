import os
import json
import logging
import requests
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("aptos_integration.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AptosIntegration:
    """Class for integrating with the Aptos blockchain"""
    
    # Aptos API endpoints
    TESTNET_URL = "https://fullnode.testnet.aptoslabs.com/v1"
    FAUCET_URL = "https://faucet.testnet.aptoslabs.com"
    
    def __init__(self, wallet_file: str = "aptos_wallet.json"):
        """
        Initialize the Aptos integration
        
        Args:
            wallet_file: Path to the wallet JSON file
        """
        self.wallet_file = wallet_file
        self.wallet = self._load_or_create_wallet()
        logger.info(f"Aptos integration initialized with account: {self.wallet['address']}")
    
    def _load_or_create_wallet(self) -> Dict[str, str]:
        """
        Load an existing wallet or create a new one
        
        Returns:
            Wallet dict with address, public_key, and private_key
        """
        if os.path.exists(self.wallet_file):
            try:
                with open(self.wallet_file, 'r') as f:
                    wallet = json.load(f)
                logger.info(f"Loaded existing wallet from {self.wallet_file}")
                return wallet
            except Exception as e:
                logger.error(f"Error loading wallet: {e}")
        
        # Create a new wallet
        return self._create_new_wallet()
    
    def _create_new_wallet(self) -> Dict[str, str]:
        """
        Create a new Aptos wallet
        
        Returns:
            Wallet dict with address, public_key, and private_key
        """
        try:
            # Generate a new private key
            response = requests.post(f"{self.TESTNET_URL}/accounts")
            if response.status_code != 200:
                logger.error(f"Failed to create account: {response.text}")
                raise Exception(f"Failed to create account: {response.text}")
            
            account_data = response.json()
            wallet = {
                "address": account_data["address"],
                "public_key": account_data["authentication_key"],
                "private_key": account_data["private_key"]
            }
            
            # Save the wallet to file
            with open(self.wallet_file, 'w') as f:
                json.dump(wallet, f, indent=2)
            
            logger.info(f"Created new wallet and saved to {self.wallet_file}")
            
            # Fund the wallet from the faucet
            self._fund_wallet(wallet["address"])
            
            return wallet
        except Exception as e:
            logger.error(f"Error creating wallet: {e}")
            raise
    
    def _fund_wallet(self, address: str, amount: int = 100000000) -> bool:
        """
        Fund a wallet from the testnet faucet
        
        Args:
            address: The account address to fund
            amount: Amount of tokens to fund (in octas)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Fund the account using the faucet
            response = requests.post(
                f"{self.FAUCET_URL}/mint",
                params={"address": address, "amount": amount}
            )
            
            if response.status_code != 200:
                logger.error(f"Failed to fund account: {response.text}")
                return False
            
            logger.info(f"Successfully funded account {address} with {amount} test tokens")
            return True
        except Exception as e:
            logger.error(f"Error funding wallet: {e}")
            return False
    
    def get_account_balance(self) -> Tuple[float, str]:
        """
        Get the account balance
        
        Returns:
            Tuple of (balance, currency_code)
        """
        try:
            # Get account resources
            response = requests.get(
                f"{self.TESTNET_URL}/accounts/{self.wallet['address']}/resources"
            )
            
            if response.status_code != 200:
                logger.error(f"Failed to get account resources: {response.text}")
                return (0.0, "APT")
            
            resources = response.json()
            
            # Find the coin resource
            for resource in resources:
                if resource["type"].endswith("::coin::CoinStore<0x1::aptos_coin::AptosCoin>"):
                    balance = int(resource["data"]["coin"]["value"]) / 100000000  # Convert from octas to APT
                    logger.info(f"Account balance: {balance} APT")
                    return (balance, "APT")
            
            logger.warning("No coin resource found")
            return (0.0, "APT")
        except Exception as e:
            logger.error(f"Error getting account balance: {e}")
            return (0.0, "APT")
    
    def send_transaction(self, recipient: str, amount: float) -> Optional[str]:
        """
        Send a transaction on the Aptos blockchain
        
        Args:
            recipient: Recipient address
            amount: Amount to send (in APT)
            
        Returns:
            Transaction hash if successful, None otherwise
        """
        try:
            # Convert APT to octas (smallest unit)
            amount_octas = int(amount * 100000000)
            
            # Check if we have enough balance
            balance, _ = self.get_account_balance()
            if balance < amount:
                logger.error(f"Insufficient balance: {balance} APT, trying to send {amount} APT")
                return None
            
            # Prepare transaction payload
            payload = {
                "function": "0x1::coin::transfer",
                "type_arguments": ["0x1::aptos_coin::AptosCoin"],
                "arguments": [recipient, str(amount_octas)]
            }
            
            # Get account info for sequence number
            account_response = requests.get(f"{self.TESTNET_URL}/accounts/{self.wallet['address']}")
            if account_response.status_code != 200:
                logger.error(f"Failed to get account info: {account_response.text}")
                return None
            
            account_data = account_response.json()
            sequence_number = int(account_data.get("sequence_number", 0))
            
            # Create transaction
            transaction_payload = {
                "sender": self.wallet["address"],
                "sequence_number": str(sequence_number),
                "max_gas_amount": "2000",
                "gas_unit_price": "100",
                "expiration_timestamp_secs": str(int(time.time()) + 600),  # 10 minutes from now
                "payload": payload
            }
            
            # Submit transaction
            transaction_response = requests.post(
                f"{self.TESTNET_URL}/transactions",
                json=transaction_payload
            )
            
            if transaction_response.status_code != 200:
                logger.error(f"Failed to submit transaction: {transaction_response.text}")
                return None
            
            tx_data = transaction_response.json()
            tx_hash = tx_data.get("hash")
            
            logger.info(f"Transaction submitted: {tx_hash}")
            logger.info(f"Sent {amount} APT to {recipient}")
            
            # Wait for transaction to be confirmed
            for _ in range(10):  # Try up to 10 times
                time.sleep(1)
                tx_status = self._check_transaction_status(tx_hash)
                if tx_status:
                    logger.info(f"Transaction confirmed: {tx_hash}")
                    return tx_hash
            
            logger.warning(f"Transaction not confirmed yet: {tx_hash}")
            return tx_hash
        
        except Exception as e:
            logger.error(f"Error sending transaction: {e}")
            return None
    
    def _check_transaction_status(self, tx_hash: str) -> bool:
        """
        Check the status of a transaction
        
        Args:
            tx_hash: Transaction hash
            
        Returns:
            True if transaction is confirmed, False otherwise
        """
        try:
            response = requests.get(f"{self.TESTNET_URL}/transactions/by_hash/{tx_hash}")
            if response.status_code != 200:
                return False
            
            tx_data = response.json()
            return tx_data.get("success", False)
        
        except Exception as e:
            logger.error(f"Error checking transaction status: {e}")
            return False
    
    def get_transaction_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get transaction history for the account
        
        Args:
            limit: Maximum number of transactions to return
            
        Returns:
            List of transaction dicts
        """
        try:
            # Get transactions for account
            response = requests.get(
                f"{self.TESTNET_URL}/accounts/{self.wallet['address']}/transactions",
                params={"limit": limit}
            )
            
            if response.status_code != 200:
                logger.error(f"Failed to get transaction history: {response.text}")
                return []
            
            transactions = response.json()
            
            # Format transactions for easier use
            formatted_txs = []
            for tx in transactions:
                tx_type = "Unknown"
                amount = 0.0
                recipient = ""
                
                # Try to determine transaction type and details
                if "payload" in tx and "function" in tx["payload"]:
                    function = tx["payload"]["function"]
                    if "0x1::coin::transfer" in function:
                        tx_type = "Transfer"
                        
                        if "arguments" in tx["payload"] and len(tx["payload"]["arguments"]) >= 2:
                            recipient = tx["payload"]["arguments"][0]
                            amount = int(tx["payload"]["arguments"][1]) / 100000000  # Convert from octas to APT
                
                formatted_tx = {
                    "hash": tx["hash"],
                    "type": tx_type,
                    "timestamp": datetime.fromtimestamp(int(tx["timestamp"]) / 1000000).strftime('%Y-%m-%d %H:%M:%S'),
                    "status": "Successful" if tx.get("success", False) else "Failed",
                    "amount": amount,
                    "recipient": recipient,
                    "gas_used": int(tx.get("gas_used", 0)) / 100000000,
                    "version": tx.get("version", "")
                }
                
                formatted_txs.append(formatted_tx)
            
            return formatted_txs
        
        except Exception as e:
            logger.error(f"Error getting transaction history: {e}")
            return []
    
    def create_test_transaction(self, amount: float = 0.01) -> Optional[str]:
        """
        Create a test transaction sending a small amount back to the same account
        Just for demonstration purposes
        
        Args:
            amount: Amount to send (in APT)
            
        Returns:
            Transaction hash if successful, None otherwise
        """
        return self.send_transaction(self.wallet["address"], amount)


if __name__ == "__main__":
    # Test the Aptos integration
    aptos = AptosIntegration()
    
    # Get account balance
    balance, currency = aptos.get_account_balance()
    print(f"Account balance: {balance} {currency}")
    
    # Create a test transaction
    tx_hash = aptos.create_test_transaction(0.01)
    if tx_hash:
        print(f"Test transaction successful! Hash: {tx_hash}")
    else:
        print("Test transaction failed")
    
    # Get transaction history
    tx_history = aptos.get_transaction_history()
    print("\nTransaction History:")
    for tx in tx_history:
        print(f"{tx['timestamp']} - {tx['type']} - {tx['amount']} APT - {tx['status']} - {tx['hash'][:10]}...") 