@echo off
echo 🧠 Verifying: AI-powered Crypto Trading Bot with Secure Predictions ^& Metadata Logging

rem 1. Set up virtual environment
echo 🔧 Setting up virtual environment...
python -m venv venv
call venv\Scripts\activate.bat

rem 2. Install dependencies
echo 📦 Installing dependencies...
pip install --upgrade pip
pip install -r requirements.txt

rem 3. Train models
echo 🎯 Training XGBoost model...
python train_xgboost_model.py --days 7 --force
if %ERRORLEVEL% NEQ 0 (
    echo ❌ XGBoost training failed
    exit /b 1
)

echo 🎯 Training RandomForest model...
python train_model.py --force
if %ERRORLEVEL% NEQ 0 (
    echo ❌ RandomForest training failed
    exit /b 1
)

rem 4. Compare model performance
echo 📊 Comparing models...
python compare_models.py --hours 12
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Model comparison failed
    exit /b 1
)

rem 5. Run tests for secure predictions & metadata
echo 🧪 Running unit tests...
python test_secure_ensemble.py
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Secure ensemble test failed
    exit /b 1
)

python test_agent_metadata.py
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Agent metadata test failed
    exit /b 1
)

rem 6. Run the main bot (dry run)
echo 🚀 Running the trading bot (1 iteration)...
python advanced_trading_bot.py --model ensemble --interval 1 --threshold 0.6 --once
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Bot execution failed
    exit /b 1
)

rem 7. Confirm metadata logs
echo 📂 Checking metadata logs...
dir logs\decision_logs\*.json >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ❌ No metadata logs found.
    exit /b 1
) else (
    echo ✅ Metadata logs created successfully.
)

echo 🎉 All checks passed! Your bot is ready for judging ✅ 