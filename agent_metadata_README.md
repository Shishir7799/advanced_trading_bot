# Agent Metadata for Crypto Trading Bot

This document describes the agent metadata implementation for the cryptocurrency trading bot, which provides comprehensive tracking and logging of trading decisions, strategies, and model signals.

## Overview

The `AgentMetadata` class maintains the identity and decision history of a trading bot, storing detailed information about every trade decision, including:

1. Bot identity (name, version, wallet address)
2. Trading strategy information
3. Models used for each trade decision
4. Signal sources and confidence levels
5. Secure enclave attestations when available
6. Complete decision history with timestamps

## Key Features

### Agent Identity Management

- Unique agent ID generation based on bot name, version, and wallet address
- Demo wallet address generation for testing
- Strategy categorization with detailed metadata

### Decision Logging

- Detailed logging of all trade decisions (buy, sell, hold)
- Storage of model signals and confidence levels
- Inclusion of secure enclave signatures and attestations
- Historical decision tracking

### IPFS Integration (Mock)

- Optional logging of decisions to simulated IPFS storage
- Generation of mock IPFS hashes for traceability
- Preservation of full decision context for auditability

## Usage

### Basic Initialization

```python
from agent_metadata import AgentMetadata

# Initialize agent metadata
agent = AgentMetadata(
    wallet_address="0x1234567890abcdef1234567890abcdef12345678",
    bot_name="SecureTrader",
    version="1.1.0",
    strategy_id="secure_ensemble"
)
```

### Logging Trade Decisions

```python
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
        "signature": "fc2f66612cea573ade9e9ccdc8294d7a9a697a34f9fbf4fa2eae57bf94625458"
    }
)
```

### Enabling IPFS Logging

```python
# Enable IPFS logging
agent.enable_ipfs_logging(True)

# Upload current decisions to IPFS
ipfs_hash = agent.upload_to_ipfs()
```

### Retrieving Decision History

```python
# Get recent decisions
recent_decisions = agent.get_recent_decisions(limit=10)

# Get decisions for a specific symbol
btc_decisions = agent.get_decisions_by_symbol("BTCUSDT")

# Get statistics about decisions
stats = agent.get_stats()
```

## Integration with Trading Bot

The agent metadata is integrated with the trading bot to automatically log all trade decisions, including:

1. Analysis decisions (when a prediction is made but no trade is executed)
2. Trade decisions (when an actual buy or sell is initiated)
3. Strategy details and model signals

Command-line arguments allow configuration of the agent metadata:

```
python trading_bot.py --wallet 0x1234... --bot-name "MyTrader" --version "1.0.0" --enable-ipfs
```

## Data Storage

All decisions are stored in JSON format in the following locations:

- Regular decisions: `agent_metadata/decisions_{agent_id}.json`
- IPFS mock: `agent_metadata/ipfs_mock/{ipfs_hash}.json`

## Example Decision JSON

```json
{
  "decision_id": "579cf2f7a7adc071",
  "agent_id": "2757f3947884",
  "wallet_address": "0x001a521f48c81edba42d94b2c8b267365c089be8",
  "timestamp": "2025-05-03T17:00:07.880592",
  "type": "trade",
  "symbol": "BTCUSDT",
  "direction": "buy",
  "confidence": 0.78,
  "price": 51240.5,
  "strategy_id": "secure_ensemble",
  "strategy_name": "Secure TEE-Enhanced Ensemble",
  "bot_version": "1.1.0",
  "amount": 0.05,
  "models_used": ["RandomForest", "XGBoost", "SecureModel"],
  "signals": {
    "RandomForest": {"prediction": "buy", "confidence": 0.75},
    "XGBoost": {"prediction": "buy", "confidence": 0.82},
    "SecureModel": {"prediction": "buy", "confidence": 0.95}
  },
  "secure_data": {
    "enclave_id": "marlin-tee-159ad989",
    "request_id": "1746271608-2936",
    "signature": "fc2f66612cea573ade9e9ccdc8294d7a9a697a34f9fbf4fa2eae57bf94625458"
  }
}
```

## Future Enhancements

1. Real IPFS integration using a proper IPFS client
2. Blockchain logging of decision hashes for immutability
3. Performance metrics tracking based on decision outcomes
4. Decision visualization and analytics dashboard
5. Secure backup and encryption of decision history 