#!/usr/bin/env python
"""
Generate sample trade decision logs for verification testing.
This helps ensure the verify_crypto_bot.sh script can properly validate log creation.
"""

import os
import json
import time
from datetime import datetime

def generate_sample_logs(count=3):
    """Generate sample trade decision logs in logs/decision_logs/"""
    print(f"🔍 Generating {count} sample trade decision logs...")
    
    # Ensure logs directory exists
    os.makedirs("logs/decision_logs", exist_ok=True)
    
    # Sample model signals with varying confidence levels
    models = ["ensemble", "xgboost", "randomforest"]
    predictions = ["BUY", "HOLD"]
    
    for i in range(count):
        # Create timestamp with microseconds for uniqueness
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        
        # Create log file path
        log_file = f"logs/decision_logs/trade_decision_{timestamp}.json"
        
        # Create sample decision data
        decision_data = {
            "timestamp": timestamp,
            "unix_timestamp": int(time.time()),
            "symbol": "BTCUSDT",
            "current_price": 95000 + (i * 10),
            "model_type": models[i % len(models)],
            "prediction": predictions[i % len(predictions)],
            "confidence": 0.65 + (i * 0.05),
            "threshold_met": True,
            "ml_signals": {
                "random_forest": {
                    "prediction": "BUY",
                    "confidence": 0.72
                },
                "xgboost": {
                    "prediction": "HOLD" if i % 2 else "BUY",
                    "confidence": 0.68
                }
            },
            "voting_data": {
                "total_voters": 2,
                "up_votes": 2 if i % 2 == 0 else 1,
                "down_votes": 0 if i % 2 == 0 else 1,
                "decision": "BUY" if i % 2 == 0 else "HOLD",
                "confidence": 0.7
            }
        }
        
        # Add TEE and Allora signals for one of the logs
        if i == 1:
            decision_data["tee_signal"] = {
                "prediction": "BUY",
                "confidence": 0.81,
                "metadata": {
                    "enclave_id": "marlin-enclave-btc-123456",
                    "attestation_id": "att-7890abcdef",
                    "secure_execution": True
                }
            }
            
            decision_data["allora_signal"] = {
                "prediction": "BUY",
                "confidence": 0.78,
                "metadata": {
                    "oracle_id": "allora-btc-oracle-1",
                    "block_height": 1234567,
                    "signature": "0x7a8b9c..."
                }
            }
        
        # Save to file
        with open(log_file, 'w') as f:
            json.dump(decision_data, f, indent=2)
        
        print(f"✅ Generated: {log_file}")
        time.sleep(0.1)  # Small delay to ensure unique timestamps
    
    print(f"✨ Successfully generated {count} sample trade decision logs")

if __name__ == "__main__":
    generate_sample_logs(3) 