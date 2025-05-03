import pandas as pd
import numpy as np
import os
import logging
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple

# Configure logging
def setup_logger(name: str, log_file: str, level=logging.INFO) -> logging.Logger:
    """
    Set up a logger with file and console handlers
    
    Args:
        name: Logger name
        log_file: Path to log file
        level: Logging level
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create handlers
    file_handler = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler()
    
    # Create formatter and add to handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def ensure_directories_exist(dir_list: List[str]) -> None:
    """
    Create directories if they don't exist
    
    Args:
        dir_list: List of directory paths to create
    """
    for dir_path in dir_list:
        os.makedirs(dir_path, exist_ok=True)

def load_config(config_file: str = ".env") -> Dict[str, str]:
    """
    Load configuration from .env file
    
    Args:
        config_file: Path to configuration file
        
    Returns:
        Dictionary with configuration values
    """
    config = {}
    
    try:
        with open(config_file, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                    
                # Parse key-value pairs
                key, value = line.split('=', 1)
                config[key.strip()] = value.strip()
                
        return config
    except FileNotFoundError:
        print(f"Config file {config_file} not found")
        return {}
    except Exception as e:
        print(f"Error loading config: {e}")
        return {}

def save_to_json(data: Any, filename: str) -> bool:
    """
    Save data to JSON file
    
    Args:
        data: Data to save
        filename: Path to save file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        return True
    except Exception as e:
        print(f"Error saving to JSON: {e}")
        return False

def load_from_json(filename: str) -> Any:
    """
    Load data from JSON file
    
    Args:
        filename: Path to JSON file
        
    Returns:
        Loaded data or None if error
    """
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"JSON file {filename} not found")
        return None
    except Exception as e:
        print(f"Error loading from JSON: {e}")
        return None

def format_timestamp(timestamp: Optional[Union[datetime, str, int]] = None, 
                   format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Format timestamp to string
    
    Args:
        timestamp: Timestamp to format (None for current time)
        format_str: Format string
        
    Returns:
        Formatted timestamp string
    """
    if timestamp is None:
        dt = datetime.now()
    elif isinstance(timestamp, int):
        dt = datetime.fromtimestamp(timestamp / 1000)  # Assuming milliseconds
    elif isinstance(timestamp, str):
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        except ValueError:
            try:
                dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                return timestamp  # Return as is if can't parse
    else:
        dt = timestamp
        
    return dt.strftime(format_str)

def calculate_moving_average(prices: List[float], window: int) -> List[float]:
    """
    Calculate moving average of prices
    
    Args:
        prices: List of prices
        window: Moving average window
        
    Returns:
        List of moving averages
    """
    if len(prices) < window:
        return []
        
    return list(pd.Series(prices).rolling(window=window).mean().dropna())

def calculate_percentage_change(start_value: float, end_value: float) -> float:
    """
    Calculate percentage change between two values
    
    Args:
        start_value: Starting value
        end_value: Ending value
        
    Returns:
        Percentage change
    """
    if start_value == 0:
        return 0
    return ((end_value - start_value) / start_value) * 100

def rate_limited(max_per_second: float):
    """
    Decorator for rate limiting function calls
    
    Args:
        max_per_second: Maximum calls per second
        
    Returns:
        Decorated function
    """
    min_interval = 1.0 / max_per_second
    last_time_called = [0.0]
    
    def decorator(func):
        def rate_limited_function(*args, **kwargs):
            elapsed = time.time() - last_time_called[0]
            left_to_wait = min_interval - elapsed
            
            if left_to_wait > 0:
                time.sleep(left_to_wait)
                
            last_time_called[0] = time.time()
            return func(*args, **kwargs)
        return rate_limited_function
    return decorator

def get_timestamp_range(days: int = 30) -> Tuple[int, int]:
    """
    Get start and end timestamps for a range of days
    
    Args:
        days: Number of days in the past
        
    Returns:
        Tuple of (start_timestamp_ms, end_timestamp_ms)
    """
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)
    
    start_timestamp = int(start_time.timestamp() * 1000)
    end_timestamp = int(end_time.timestamp() * 1000)
    
    return start_timestamp, end_timestamp

def parse_timeframe(timeframe: str) -> Tuple[int, str]:
    """
    Parse timeframe string (e.g., '1h', '15m', '1d')
    
    Args:
        timeframe: Timeframe string
        
    Returns:
        Tuple of (value, unit)
    """
    # Extract number and unit
    for i, char in enumerate(timeframe):
        if not char.isdigit():
            value = int(timeframe[:i])
            unit = timeframe[i:]
            return value, unit
    
    # Default if no unit found
    return int(timeframe), ''

def milliseconds_to_timeframe(ms: int, timeframe: str) -> int:
    """
    Convert milliseconds to number of timeframe units
    
    Args:
        ms: Milliseconds
        timeframe: Timeframe string (e.g., '1h', '15m', '1d')
        
    Returns:
        Number of timeframe units
    """
    value, unit = parse_timeframe(timeframe)
    
    # Convert to milliseconds
    if unit == 'm':
        timeframe_ms = value * 60 * 1000
    elif unit == 'h':
        timeframe_ms = value * 60 * 60 * 1000
    elif unit == 'd':
        timeframe_ms = value * 24 * 60 * 60 * 1000
    else:
        timeframe_ms = value * 1000  # Default to seconds
    
    return ms // timeframe_ms

def calculate_trading_metrics(trades: List[Dict[str, Any]], 
                             initial_balance: float) -> Dict[str, Any]:
    """
    Calculate trading performance metrics
    
    Args:
        trades: List of trade dictionaries
        initial_balance: Initial account balance
        
    Returns:
        Dictionary with trading metrics
    """
    if not trades:
        return {
            "total_trades": 0,
            "profit_trades": 0,
            "loss_trades": 0,
            "win_rate": 0.0,
            "average_profit": 0.0,
            "average_loss": 0.0,
            "profit_factor": 0.0,
            "total_profit": 0.0,
            "total_loss": 0.0,
            "net_profit": 0.0,
            "return_percentage": 0.0
        }
    
    # Calculate metrics
    profits = []
    losses = []
    
    for trade in trades:
        if trade.get('profit', 0) > 0:
            profits.append(trade['profit'])
        else:
            losses.append(abs(trade.get('profit', 0)))
    
    total_trades = len(trades)
    profit_trades = len(profits)
    loss_trades = len(losses)
    
    win_rate = profit_trades / total_trades if total_trades > 0 else 0
    average_profit = sum(profits) / profit_trades if profit_trades > 0 else 0
    average_loss = sum(losses) / loss_trades if loss_trades > 0 else 0
    
    total_profit = sum(profits)
    total_loss = sum(losses)
    net_profit = total_profit - total_loss
    
    profit_factor = total_profit / total_loss if total_loss > 0 else float('inf') if total_profit > 0 else 0
    
    # Calculate return percentage
    final_balance = initial_balance + net_profit
    return_percentage = ((final_balance / initial_balance) - 1) * 100
    
    return {
        "total_trades": total_trades,
        "profit_trades": profit_trades,
        "loss_trades": loss_trades,
        "win_rate": win_rate,
        "average_profit": average_profit,
        "average_loss": average_loss,
        "profit_factor": profit_factor,
        "total_profit": total_profit,
        "total_loss": total_loss,
        "net_profit": net_profit,
        "return_percentage": return_percentage
    } 