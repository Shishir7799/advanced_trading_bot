#!/usr/bin/env python3
import os
import subprocess
import sys
import platform
import time
import shutil
from pathlib import Path

def print_header(message):
    """Print a formatted header message"""
    print("\n" + "="*80)
    print(f"  {message}")
    print("="*80)

def print_step(message, indent=2):
    """Print a step message with indentation"""
    print(f"{' ' * indent}→ {message}")

def print_success(message, indent=2):
    """Print a success message with indentation"""
    print(f"{' ' * indent}✓ {message}")

def print_error(message, indent=2):
    """Print an error message with indentation"""
    print(f"{' ' * indent}✗ {message}")

def run_command(command, desc=None, suppress_output=False):
    """Run a shell command and return success/failure"""
    if desc:
        print_step(desc)
    
    try:
        if suppress_output:
            result = subprocess.run(
                command, 
                shell=True, 
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
        else:
            result = subprocess.run(
                command, 
                shell=True, 
                check=True
            )
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Command failed with exit code {e.returncode}")
        if suppress_output and e.stderr:
            print(f"Error details: {e.stderr.decode('utf-8')}")
        return False

def install_dependencies():
    """Install required Python packages"""
    print_header("Installing Dependencies")
    
    # Check if requirements.txt exists
    if not os.path.exists("requirements.txt"):
        print_error("requirements.txt not found in current directory")
        return False
    
    # Install packages from requirements.txt
    success = run_command(
        f"{sys.executable} -m pip install -r requirements.txt",
        "Installing packages from requirements.txt"
    )
    
    if success:
        print_success("All dependencies installed successfully")
    else:
        print_error("Failed to install dependencies")
        
    return success

def create_directories():
    """Create necessary directories"""
    print_header("Creating Directories")
    
    dirs = [
        "models",
        "data",
        "logs",
        "evaluation_results",
        "tuning_results",
        "transaction_logs"
    ]
    
    for dir_name in dirs:
        dir_path = Path(dir_name)
        if not dir_path.exists():
            try:
                dir_path.mkdir(parents=True)
                print_success(f"Created directory: {dir_name}")
            except Exception as e:
                print_error(f"Failed to create directory {dir_name}: {e}")
                return False
        else:
            print_step(f"Directory exists: {dir_name}")
    
    return True

def check_environment():
    """Check environment dependencies"""
    print_header("Checking Environment")
    
    # Check Python version
    python_version = platform.python_version()
    print_step(f"Python version: {python_version}")
    
    major, minor, _ = map(int, python_version.split('.'))
    if major < 3 or (major == 3 and minor < 7):
        print_error("Python 3.7 or higher is required")
        return False
    
    # Check pip
    pip_check = run_command(
        f"{sys.executable} -m pip --version",
        "Checking pip installation",
        suppress_output=True
    )
    
    if not pip_check:
        print_error("pip is not installed properly")
        return False
    
    print_success("Environment check passed")
    return True

def setup_config():
    """Setup configuration files"""
    print_header("Setting Up Configuration")
    
    # Create .env file template if it doesn't exist
    env_path = Path(".env")
    if not env_path.exists():
        try:
            with open(env_path, "w") as env_file:
                env_file.write("""# API Keys Configuration
# Binance API credentials (required for live price data)
BINANCE_API_KEY=your_binance_api_key_here
BINANCE_API_SECRET=your_binance_api_secret_here

# Aptos blockchain credentials (required for blockchain transactions)
APTOS_NODE_URL=https://fullnode.mainnet.aptoslabs.com
APTOS_PRIVATE_KEY=your_aptos_private_key_here

# Trading bot configuration
DEFAULT_SYMBOL=BTCUSDT
DEFAULT_TRADE_AMOUNT=0.01
DEFAULT_CONFIDENCE_THRESHOLD=0.6
DEFAULT_CHECK_INTERVAL=300
""")
            print_success("Created .env configuration file template")
            print_step("Please edit the .env file to add your API keys and configure the bot")
        except Exception as e:
            print_error(f"Failed to create .env file: {e}")
            return False
    else:
        print_step(".env file already exists")
    
    return True

def test_imports():
    """Test importing key modules to ensure they're installed correctly"""
    print_header("Testing Module Imports")
    
    modules = [
        "pandas",
        "numpy",
        "joblib",
        "sklearn",
        "xgboost",
        "matplotlib",
        "seaborn",
        "requests"
    ]
    
    all_success = True
    for module in modules:
        try:
            __import__(module)
            print_success(f"Successfully imported {module}")
        except ImportError as e:
            print_error(f"Failed to import {module}: {e}")
            all_success = False
    
    return all_success

def install_sample_models():
    """Install sample pre-trained models for quick testing"""
    print_header("Installing Sample Models")
    
    # Create models directory if it doesn't exist
    models_dir = Path("models")
    if not models_dir.exists():
        models_dir.mkdir(parents=True)
    
    # Check if we should download sample models
    print_step("Do you want to download sample pre-trained models? (y/n)")
    choice = input().strip().lower()
    
    if choice != 'y':
        print_step("Skipping sample model installation")
        return True
    
    # This would typically download models from a repository
    # For this example, we'll simulate creating sample model files
    
    try:
        # RandomForest model placeholder
        with open(models_dir / "btcusdt_model.joblib", "w") as f:
            f.write("# This is a placeholder for a pre-trained RandomForest model\n")
        
        # XGBoost model placeholder
        with open(models_dir / "btcusdt_xgboost.joblib", "w") as f:
            f.write("# This is a placeholder for a pre-trained XGBoost model\n")
            
        print_success("Sample model files created")
        print_step("Note: These are placeholder files. To use real models, you'll need to train them first.")
        
        return True
    except Exception as e:
        print_error(f"Failed to create sample model files: {e}")
        return False

def setup_complete():
    """Show final instructions after setup is complete"""
    print_header("Setup Complete")
    
    print("""
    Your AI crypto trading bot system is now set up!
    
    To get started:
    
    1. Edit the .env file to add your API keys and configuration
    
    2. Train your models:
       - For RandomForest: python train_model.py
       - For XGBoost:      python train_xgboost_model.py --days 30
    
    3. Run the hyperparameter tuning:
       - python hyperparameter_tuning.py
    
    4. Evaluate model performance:
       - python model_evaluator.py --hours 24
    
    5. Run the trading bot:
       - python main_agent.py
    
    For help with any command, use the --help flag, e.g.:
       - python main_agent.py --help
    
    Enjoy trading!
    """)

def main():
    print_header("AI Crypto Trading Bot Setup")
    print("This script will set up your environment for the AI crypto trading bot.")
    
    # Ask for confirmation to proceed
    print("\nWould you like to proceed with setup? (y/n)")
    response = input().strip().lower()
    
    if response != 'y':
        print("Setup cancelled.")
        return
    
    # Run setup steps
    steps = [
        (check_environment, "environment check"),
        (install_dependencies, "dependency installation"),
        (create_directories, "directory creation"),
        (setup_config, "configuration setup"),
        (test_imports, "module import testing"),
        (install_sample_models, "sample model installation")
    ]
    
    all_success = True
    for step_func, step_name in steps:
        success = step_func()
        if not success:
            print_error(f"Setup failed during {step_name}")
            all_success = False
            break
        time.sleep(0.5)  # Brief pause between steps
    
    if all_success:
        setup_complete()

if __name__ == "__main__":
    main() 