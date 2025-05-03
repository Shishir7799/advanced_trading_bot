import os
import json
import logging
import hashlib
import time
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("agent_metadata.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AgentMetadata:
    """Class to manage trading bot agent metadata and decision logging"""
    
    def __init__(self, 
                 wallet_address: str = None,
                 bot_name: str = "CryptoTradingBot",
                 version: str = "1.0.0",
                 strategy_id: str = "adaptive_ensemble",
                 metadata_dir: str = "agent_metadata"):
        """
        Initialize agent metadata
        
        Args:
            wallet_address: Wallet address associated with the bot
            bot_name: Name of the trading bot
            version: Version of the trading bot
            strategy_id: Identifier for the trading strategy
            metadata_dir: Directory to store metadata logs
        """
        # Core identity
        self.wallet_address = wallet_address or self._generate_demo_address()
        self.bot_name = bot_name
        self.version = version
        self.strategy_id = strategy_id
        self.agent_id = self._generate_agent_id()
        
        # Strategy details
        self.strategy_details = {
            "id": strategy_id,
            "name": self._get_strategy_name(strategy_id),
            "description": self._get_strategy_description(strategy_id),
            "risk_level": self._get_strategy_risk_level(strategy_id),
            "models": self._get_strategy_models(strategy_id)
        }
        
        # Storage
        self.metadata_dir = metadata_dir
        os.makedirs(metadata_dir, exist_ok=True)
        
        # Decision log
        self.decisions_log_file = os.path.join(metadata_dir, f"decisions_{self.agent_id}.json")
        self.decisions = self._load_existing_decisions()
        
        # IPFS mock
        self.ipfs_enabled = False
        self.ipfs_hashes = []
        
        logger.info(f"Initialized agent metadata for {bot_name} v{version} (ID: {self.agent_id})")
        logger.info(f"Wallet address: {self.wallet_address}")
        logger.info(f"Strategy: {self.strategy_details['name']}")
    
    def _generate_demo_address(self) -> str:
        """Generate a demo wallet address"""
        # Use a hash of machine ID and current time for a consistent but unique address
        machine_id = uuid.getnode()
        addr_hash = hashlib.sha256(f"{machine_id}:{time.time()}".encode()).hexdigest()
        
        # Format as an Ethereum address
        return f"0x{addr_hash[-40:].lower()}"
    
    def _generate_agent_id(self) -> str:
        """Generate a unique agent ID based on name, version, and wallet"""
        agent_data = f"{self.bot_name}:{self.version}:{self.wallet_address}:{self.strategy_id}"
        return hashlib.sha256(agent_data.encode()).hexdigest()[:12]
    
    def _get_strategy_name(self, strategy_id: str) -> str:
        """Get strategy name from ID"""
        strategy_names = {
            "adaptive_ensemble": "Adaptive Ensemble ML",
            "secure_ensemble": "Secure TEE-Enhanced Ensemble",
            "allora_hybrid": "Allora-Enhanced Hybrid",
            "momentum_ml": "Momentum ML"
        }
        return strategy_names.get(strategy_id, "Custom Strategy")
    
    def _get_strategy_description(self, strategy_id: str) -> str:
        """Get strategy description from ID"""
        strategy_descriptions = {
            "adaptive_ensemble": "Ensemble of ML models with adaptive weighting based on recent performance",
            "secure_ensemble": "Ensemble strategy enhanced with secure model execution in TEE",
            "allora_hybrid": "Hybrid strategy combining ML models with Allora oracle signals",
            "momentum_ml": "ML-enhanced momentum trading with multiple timeframes"
        }
        return strategy_descriptions.get(strategy_id, "Custom trading strategy")
    
    def _get_strategy_risk_level(self, strategy_id: str) -> str:
        """Get strategy risk level from ID"""
        strategy_risk_levels = {
            "adaptive_ensemble": "Medium",
            "secure_ensemble": "Medium-Low",
            "allora_hybrid": "Medium",
            "momentum_ml": "Medium-High"
        }
        return strategy_risk_levels.get(strategy_id, "Medium")
    
    def _get_strategy_models(self, strategy_id: str) -> List[str]:
        """Get list of models used in the strategy"""
        strategy_models = {
            "adaptive_ensemble": ["RandomForest", "XGBoost", "LSTM"],
            "secure_ensemble": ["RandomForest", "XGBoost", "SecureModel"],
            "allora_hybrid": ["RandomForest", "XGBoost", "AlloraOracle"],
            "momentum_ml": ["RandomForest", "XGBoost", "MomentumIndicators"]
        }
        return strategy_models.get(strategy_id, ["Unknown"])
    
    def _load_existing_decisions(self) -> List[Dict[str, Any]]:
        """Load existing decisions if file exists"""
        if os.path.exists(self.decisions_log_file):
            try:
                with open(self.decisions_log_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading existing decisions: {e}")
                return []
        return []
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get complete agent information"""
        return {
            "agent_id": self.agent_id,
            "wallet_address": self.wallet_address,
            "bot_name": self.bot_name,
            "version": self.version,
            "strategy": self.strategy_details,
            "decision_count": len(self.decisions),
            "ipfs_enabled": self.ipfs_enabled,
            "ipfs_logs": self.ipfs_hashes
        }
    
    def log_trade_decision(self, 
                          decision_type: str,
                          symbol: str,
                          direction: str,
                          confidence: float,
                          price: float,
                          amount: Optional[float] = None,
                          models_used: Optional[List[str]] = None,
                          signals: Optional[Dict[str, Any]] = None,
                          secure_data: Optional[Dict[str, Any]] = None,
                          extra_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Log a trade decision with full metadata
        
        Args:
            decision_type: Type of decision (e.g., "trade", "analysis", "alert")
            symbol: Trading pair symbol
            direction: Trade direction ("buy", "sell", "hold")
            confidence: Confidence score (0.0 to 1.0)
            price: Current price at decision time
            amount: Optional trade amount
            models_used: List of models that generated the decision
            signals: Dictionary of signals from different models/sources
            secure_data: Data from secure model execution (if applicable)
            extra_data: Any additional data to include
            
        Returns:
            The complete decision record
        """
        # Create timestamp
        timestamp = datetime.now()
        
        # Generate decision ID
        decision_id = hashlib.sha256(
            f"{self.agent_id}:{symbol}:{direction}:{timestamp.isoformat()}".encode()
        ).hexdigest()[:16]
        
        # Create base decision record
        decision = {
            "decision_id": decision_id,
            "agent_id": self.agent_id,
            "wallet_address": self.wallet_address,
            "timestamp": timestamp.isoformat(),
            "type": decision_type,
            "symbol": symbol,
            "direction": direction,
            "confidence": confidence,
            "price": price,
            "strategy_id": self.strategy_id,
            "strategy_name": self.strategy_details["name"],
            "bot_version": self.version
        }
        
        # Add optional fields if provided
        if amount is not None:
            decision["amount"] = amount
        
        if models_used:
            decision["models_used"] = models_used
        
        if signals:
            decision["signals"] = signals
        
        if secure_data:
            decision["secure_data"] = secure_data
        
        if extra_data:
            decision["extra_data"] = extra_data
        
        # Add to decisions list
        self.decisions.append(decision)
        
        # Save to file
        self._save_decisions()
        
        # Log this decision
        logger.info(f"Logged {decision_type} decision {decision_id}: {direction} {symbol} at {price} with {confidence:.2f} confidence")
        
        return decision
    
    def _save_decisions(self):
        """Save decisions to JSON file"""
        try:
            with open(self.decisions_log_file, 'w') as f:
                json.dump(self.decisions, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving decisions: {e}")
    
    def enable_ipfs_logging(self, enabled: bool = True):
        """Enable or disable IPFS logging (mock)"""
        self.ipfs_enabled = enabled
        logger.info(f"IPFS logging {'enabled' if enabled else 'disabled'}")
    
    def upload_to_ipfs(self, data: Optional[Dict[str, Any]] = None) -> str:
        """
        Mock function to simulate uploading data to IPFS
        
        Args:
            data: Data to upload (defaults to all decisions if None)
            
        Returns:
            Mock IPFS hash
        """
        if not self.ipfs_enabled:
            logger.warning("IPFS logging is disabled")
            return ""
        
        # Use provided data or all decisions
        upload_data = data if data is not None else {
            "agent_info": self.get_agent_info(),
            "decisions": self.decisions
        }
        
        # Generate mock IPFS hash
        timestamp = datetime.now().isoformat()
        data_str = json.dumps(upload_data)
        mock_hash = f"Qm{hashlib.sha256((data_str + timestamp).encode()).hexdigest()[:44]}"
        
        # Save mock upload
        mock_upload_dir = os.path.join(self.metadata_dir, "ipfs_mock")
        os.makedirs(mock_upload_dir, exist_ok=True)
        
        mock_file_path = os.path.join(mock_upload_dir, f"{mock_hash}.json")
        
        try:
            with open(mock_file_path, 'w') as f:
                json.dump(upload_data, f, indent=2)
                
            # Add to hashes list
            self.ipfs_hashes.append({
                "hash": mock_hash,
                "timestamp": timestamp,
                "file": mock_file_path,
                "content_type": "decisions"
            })
            
            logger.info(f"Mock IPFS upload complete: {mock_hash}")
            return mock_hash
            
        except Exception as e:
            logger.error(f"Error in mock IPFS upload: {e}")
            return ""
    
    def get_recent_decisions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get the most recent decisions"""
        return self.decisions[-limit:] if self.decisions else []
    
    def get_decisions_by_symbol(self, symbol: str) -> List[Dict[str, Any]]:
        """Get all decisions for a specific trading pair"""
        return [d for d in self.decisions if d.get("symbol") == symbol]
    
    def get_decision_by_id(self, decision_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific decision by ID"""
        for decision in self.decisions:
            if decision.get("decision_id") == decision_id:
                return decision
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about decisions"""
        if not self.decisions:
            return {"decision_count": 0}
        
        # Count decisions by type and direction
        types = {}
        directions = {}
        symbols = {}
        
        for decision in self.decisions:
            # Count by type
            d_type = decision.get("type", "unknown")
            types[d_type] = types.get(d_type, 0) + 1
            
            # Count by direction
            d_direction = decision.get("direction", "unknown")
            directions[d_direction] = directions.get(d_direction, 0) + 1
            
            # Count by symbol
            d_symbol = decision.get("symbol", "unknown")
            symbols[d_symbol] = symbols.get(d_symbol, 0) + 1
        
        # Get timestamp of first and last decision
        first_timestamp = self.decisions[0].get("timestamp", "unknown")
        last_timestamp = self.decisions[-1].get("timestamp", "unknown")
        
        return {
            "decision_count": len(self.decisions),
            "types": types,
            "directions": directions,
            "symbols": symbols,
            "first_decision": first_timestamp,
            "last_decision": last_timestamp
        }


# Example usage
if __name__ == "__main__":
    # Create agent metadata instance
    agent = AgentMetadata(
        bot_name="SecureTrader",
        version="1.1.0",
        strategy_id="secure_ensemble"
    )
    
    # Print agent info
    print("Agent Info:")
    print(json.dumps(agent.get_agent_info(), indent=2))
    
    # Log a trade decision
    decision = agent.log_trade_decision(
        decision_type="trade",
        symbol="BTCUSDT",
        direction="buy",
        confidence=0.78,
        price=51240.50,
        amount=0.05,
        models_used=["RandomForest", "XGBoost", "SecureModel"],
        signals={
            "RandomForest": {"prediction": "buy", "confidence": 0.75},
            "XGBoost": {"prediction": "buy", "confidence": 0.82},
            "SecureModel": {"prediction": "buy", "confidence": 0.95}
        },
        secure_data={
            "enclave_id": "marlin-tee-159ad989",
            "request_id": "1746271608-2936",
            "signature": "fc2f66612cea573ade9e9ccdc8294d7a9a697a34f9fbf4fa2eae57bf94625458"
        }
    )
    
    # Enable IPFS logging and upload
    agent.enable_ipfs_logging(True)
    ipfs_hash = agent.upload_to_ipfs()
    
    print(f"\nUploaded to IPFS (mock): {ipfs_hash}")
    
    # Print stats
    print("\nDecision Stats:")
    print(json.dumps(agent.get_stats(), indent=2)) 