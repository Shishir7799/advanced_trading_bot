#!/bin/bash

echo "🧠 Verifying: AI-powered Crypto Trading Bot with Secure Predictions & Metadata Logging"

# 1. Set up virtual environment
echo "🔧 Setting up virtual environment..."
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
echo "📦 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# 3. Train models
echo "🎯 Training XGBoost model..."
python train_xgboost_model.py --days 7 --force || { echo "❌ XGBoost training failed"; exit 1; }

echo "🎯 Training RandomForest model..."
python train_model.py --force || { echo "❌ RandomForest training failed"; exit 1; }

# 4. Compare model performance
echo "📊 Comparing models..."
python compare_models.py --hours 12 || { echo "❌ Model comparison failed"; exit 1; }

# 5. Run tests for secure predictions & metadata
echo "🧪 Running unit tests..."
python test_secure_ensemble.py || { echo "❌ Secure ensemble test failed"; exit 1; }
python test_agent_metadata.py || { echo "❌ Agent metadata test failed"; exit 1; }

# 6. Run the main bot (dry run)
echo "🚀 Running the trading bot (1 iteration)..."
python advanced_trading_bot.py --model ensemble --interval 1 --threshold 0.6 --once || { echo "❌ Bot execution failed"; exit 1; }

# 7. Confirm metadata logs
echo "📂 Checking metadata logs..."
if ls logs/decision_logs/*.json 1> /dev/null 2>&1; then
    echo "✅ Metadata logs created successfully."
else
    echo "❌ No metadata logs found."
    exit 1
fi

echo "🎉 All checks passed! Your bot is ready for judging ✅" 