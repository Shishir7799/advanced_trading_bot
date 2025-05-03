# Verification Instructions

This document explains how to run the verification scripts to ensure your crypto trading bot with secure predictions and metadata logging is working correctly.

## Prerequisites

- Python 3.8 or higher
- Git
- Pip package manager

## Running Verification

### On Linux/macOS

1. Make the script executable:
   ```bash
   chmod +x verify_crypto_bot.sh
   ```

2. Run the verification script:
   ```bash
   ./verify_crypto_bot.sh
   ```

### On Windows

1. Open Command Prompt or PowerShell

2. Run the Windows verification script:
   ```
   verify_crypto_bot.bat
   ```

## What the Verification Script Does

The verification script performs the following checks:

1. **Environment Setup**: Creates a virtual environment and installs dependencies
2. **Model Training**: Trains both XGBoost and RandomForest models
3. **Model Comparison**: Runs a comparison of model performance
4. **Unit Tests**: Tests the secure ensemble and agent metadata functionality
5. **Bot Execution**: Runs one iteration of the trading bot to verify functionality
6. **Log Verification**: Confirms that metadata logs are being generated properly

If any step fails, the script will exit with an error message.

## Expected Output

When everything works correctly, you should see:

```
🎉 All checks passed! Your bot is ready for judging ✅
```

## Troubleshooting

If a verification step fails:

1. Check the error message for details
2. Look at the console output and log files
3. Resolve the issue and run the verification script again

For specific issues:

- **Model training failures**: Ensure you have sufficient training data
- **Test failures**: Make sure all module dependencies are installed
- **Bot execution failures**: Check configuration parameters
- **Missing logs**: Verify proper permissions for log directories 