#!/bin/bash

# AI Crypto Trading Bot Runner Script
# This script simplifies running the trading bot with common configurations

# Default values
SYMBOL="BTCUSDT"
INTERVAL=300
THRESHOLD=0.7
XGB_THRESHOLD=0.7
TRADE_AMOUNT=0.01
MOCK_MODE=true
FORCE_TRAIN=""

# Display banner
echo "========================================================"
echo "    AI CRYPTO TRADING BOT"
echo "========================================================"
echo ""

# Help function
function show_help {
    echo "Usage: ./run_trading_bot.sh [options]"
    echo ""
    echo "Options:"
    echo "  -h, --help            Show this help message"
    echo "  -s, --symbol SYMBOL   Trading pair symbol (default: BTCUSDT)"
    echo "  -i, --interval SEC    Check interval in seconds (default: 300)"
    echo "  -t, --threshold VAL   Confidence threshold (default: 0.7)"
    echo "  -x, --xgb-threshold VAL XGBoost confidence when models disagree (default: 0.7)"
    echo "  -a, --amount VAL      Trade amount (default: 0.01)"
    echo "  -r, --real            Use real transactions (default: mock mode)"
    echo "  -f, --force-train     Force training models before starting"
    echo "  -e, --evaluate HOURS  Evaluate models on recent data before starting"
    echo ""
    echo "Examples:"
    echo "  ./run_trading_bot.sh                   # Run with default settings"
    echo "  ./run_trading_bot.sh -s ETHUSDT -i 600 # Run for ETH with 10min interval"
    echo "  ./run_trading_bot.sh -f -r             # Train models and use real transactions"
    echo ""
    exit 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -h|--help)
            show_help
            ;;
        -s|--symbol)
            SYMBOL="$2"
            shift 2
            ;;
        -i|--interval)
            INTERVAL="$2"
            shift 2
            ;;
        -t|--threshold)
            THRESHOLD="$2"
            shift 2
            ;;
        -x|--xgb-threshold)
            XGB_THRESHOLD="$2"
            shift 2
            ;;
        -a|--amount)
            TRADE_AMOUNT="$2"
            shift 2
            ;;
        -r|--real)
            MOCK_MODE=false
            shift
            ;;
        -f|--force-train)
            FORCE_TRAIN="--force-train-rf --force-train-xgb"
            shift
            ;;
        -e|--evaluate)
            EVALUATE_HOURS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            ;;
    esac
done

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "Error: Python not found. Please install Python."
    exit 1
fi

# Check if required files exist
if [ ! -f "main_agent.py" ]; then
    echo "Error: main_agent.py not found. Please run this script from the project directory."
    exit 1
fi

# Create directories if they don't exist
mkdir -p models data logs evaluation_results transaction_logs

# Run model evaluation if requested
if [ ! -z "$EVALUATE_HOURS" ]; then
    echo "Evaluating models on the last $EVALUATE_HOURS hours of data..."
    python model_evaluator.py --hours "$EVALUATE_HOURS" --symbol "$SYMBOL"
    
    echo ""
    echo "Continue with trading bot? (y/n)"
    read -r continue_choice
    if [[ ! "$continue_choice" =~ ^[Yy]$ ]]; then
        echo "Exiting."
        exit 0
    fi
fi

# Build command based on parameters
CMD="python main_agent.py --symbol $SYMBOL --interval $INTERVAL --threshold $THRESHOLD --xgb-threshold $XGB_THRESHOLD --amount $TRADE_AMOUNT"

# Add mock/real mode
if [ "$MOCK_MODE" = false ]; then
    CMD="$CMD --real"
    echo "⚠️  WARNING: Running in REAL transaction mode ⚠️"
    echo "This will use real funds for trading."
    echo "Are you sure you want to continue? (yes/no)"
    read -r confirmation
    if [[ ! "$confirmation" =~ ^[Yy][Ee][Ss]$ ]]; then
        echo "Aborted."
        exit 0
    fi
else
    echo "Running in mock transaction mode (no real funds will be used)"
fi

# Add force train if specified
if [ ! -z "$FORCE_TRAIN" ]; then
    CMD="$CMD $FORCE_TRAIN"
    echo "Models will be trained before starting"
fi

# Display configuration
echo ""
echo "Configuration:"
echo "- Symbol: $SYMBOL"
echo "- Check interval: $INTERVAL seconds"
echo "- Confidence threshold: $THRESHOLD"
echo "- XGBoost disagreement threshold: $XGB_THRESHOLD"
echo "- Trade amount: $TRADE_AMOUNT"
echo "- Transaction mode: $(if [ "$MOCK_MODE" = true ]; then echo "MOCK"; else echo "REAL"; fi)"
echo ""

# Ask for confirmation
echo "Start trading bot with these settings? (y/n)"
read -r start_choice
if [[ ! "$start_choice" =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

# Run the bot
echo ""
echo "Starting trading bot..."
echo "Press Ctrl+C to stop"
echo ""
echo "$CMD"
echo ""

# Execute the command
eval "$CMD" 