# Marlin TEE Secure Enclave Integration for Crypto Trading Bot

This document describes the implementation of a Trusted Execution Environment (TEE) simulation for secure model predictions in our cryptocurrency trading bot.

## Overview

The Marlin TEE integration provides a simulated secure enclave for running price prediction models, ensuring:

1. Model execution happens in a "secure environment"
2. Predictions are signed for verification
3. Model weights are protected from inspection
4. Attestation guarantees model integrity

This implementation simulates the behavior of a real TEE hardware enclave, such as those provided by Marlin Network, by adding appropriate security logs, signatures, and computation delays.

## Files Implemented

1. **secure_predictions.py**: Core implementation of the simulated secure enclave
   - `SecureEnclave` class that handles attestation and secure computation
   - `run_secure_model_prediction()` function for easy access

2. **secure_model_runner.py**: Standalone script for running secure predictions
   - Accepts price data from files or generates mock data
   - Outputs signed predictions with attestation details

3. **ensemble_predictor.py** (updated): Enhanced to use secure enclave predictions
   - Integrates secure model predictions with other models
   - Configurable weighting of secure model in ensemble

4. **trading_bot.py** (updated): Modified to support secure enclave
   - Command-line arguments for enabling secure enclave
   - Logging of secure prediction signatures

5. **dashboard.py** (updated): Enhanced to display secure model information
   - Shows secure predictions alongside other models
   - Displays enclave attestation information

## Features

### Secure Enclave Simulation

- Realistic enclave startup sequence with attestation
- Protection for model execution with "secure memory"
- Computation of input data hashes for reproducibility
- Generation of unique session and request identifiers
- Signing of prediction results for verification

### Performance Metrics

- Tracking of computation time inside the secure enclave
- Measurement of overall prediction latency including TEE overhead
- Detailed logging of execution steps

### Integration with Ensemble Predictor

- Configurable weighting of secure model in the ensemble
- Ability to use secure model as a confirmation signal
- Detailed prediction information from all sources

## Usage

### Standalone Secure Prediction

```bash
python secure_model_runner.py --input price_data.csv --output prediction.json --verbose
```

### Within Trading Bot

```bash
python trading_bot.py --use-secure-enclave --secure-model-id btc_price_predictor_v1
```

### Running Tests

```bash
python test_secure_ensemble.py
```

## Example Output

```json
{
  "prediction": "up",
  "confidence": 0.95,
  "timestamp": "2025-05-03T16:56:50.278713",
  "metadata": {
    "request_id": "1746271608-2936",
    "enclave_id": "marlin-tee-159ad989",
    "model_id": "btc_price_predictor_v1",
    "timestamp": "2025-05-03T16:56:50.267632",
    "input_hash": "832e5c266ca4637fc6c20c9920113b3867d3944502741628464d10934eb0d643",
    "input_rows": 49,
    "computation_time_ms": 1004,
    "prediction": "up",
    "confidence": 0.95,
    "session_nonce": 22583808,
    "signature": "fc2f66612cea573ade9e9ccdc8294d7a9a697a34f9fbf4fa2eae57bf94625458"
  }
}
```

## Production Considerations

For a production implementation:

1. Replace simulation with actual TEE hardware integration
2. Implement proper cryptographic key management
3. Use hardware-based random number generation
4. Add remote attestation verification
5. Implement secure input/output channels

## Future Enhancements

1. Support for multiple model versions in the enclave
2. Integration with model training pipeline
3. Enhanced verification of prediction signatures 
4. Performance optimization for lower latency 