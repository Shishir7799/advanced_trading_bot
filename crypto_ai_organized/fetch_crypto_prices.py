import requests
import pandas as pd
import time
import logging
import traceback
import datetime
import os
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PriceFetcher:
    def __init__(self, symbol: str = "BTCUSDT"):
        self.symbol = symbol
        self.base_url = "https://api.binance.com/api/v3"
        self.price_history: List[Dict[str, Any]] = []
        self.fetch_count = 0
        self.error_count = 0
        logger.info(f"PriceFetcher initialized for {symbol}")
        
    def get_current_price(self) -> Optional[float]:
        """Fetch the current price of the symbol from Binance"""
        self.fetch_count += 1
        fetch_id = self.fetch_count
        start_time = time.time()
        
        try:
            logger.debug(f"Fetching price #{fetch_id} for {self.symbol}...")
            endpoint = f"{self.base_url}/ticker/price"
            params = {"symbol": self.symbol}
            
            # Log the API request
            logger.debug(f"API Request #{fetch_id}: GET {endpoint} with params {params}")
            
            # Make the request with a timeout
            response = requests.get(endpoint, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            price = float(data["price"])
            
            # Add to history with timestamp
            timestamp = int(time.time())
            self.price_history.append({
                "timestamp": timestamp,
                "datetime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp)),
                "price": price,
                "fetch_id": fetch_id
            })
            
            # Calculate elapsed time
            elapsed = time.time() - start_time
            
            logger.info(f"Current {self.symbol} price: {price} (fetch #{fetch_id}, {elapsed:.3f}s)")
            logger.debug(f"Price history length: {len(self.price_history)}")
            
            return price
            
        except requests.exceptions.RequestException as e:
            self.error_count += 1
            logger.error(f"Error fetching price #{fetch_id}: {e}")
            logger.debug(f"Detailed error: {traceback.format_exc()}")
            
            # Add error to history for tracking
            if hasattr(e, 'response') and e.response is not None:
                logger.debug(f"API error response: {e.response.status_code} - {e.response.text}")
                
            return None
    
    def get_last_n_prices(self, n: int = 5) -> pd.DataFrame:
        """Return the last n prices from history as a DataFrame"""
        if len(self.price_history) < n:
            logger.warning(f"Requested {n} prices but only {len(self.price_history)} available")
            
        # Get the last n items from price history
        last_n = self.price_history[-n:] if self.price_history else []
        
        # Convert to DataFrame
        df = pd.DataFrame(last_n)
        if not df.empty:
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
            logger.debug(f"Retrieved last {len(df)} prices from history")
            logger.debug(f"Price range: {df['price'].min():.2f} - {df['price'].max():.2f}")
        else:
            logger.warning("Empty price history DataFrame")
        
        return df
    
    def fetch_historical_klines(self, interval: str = "1h", limit: int = 500) -> pd.DataFrame:
        """
        Fetch historical price data from Binance
        
        Args:
            interval: Time interval (e.g., "1m", "5m", "1h", "1d")
            limit: Number of candles to fetch (max 1000)
        
        Returns:
            DataFrame with OHLCV data
        """
        start_time = time.time()
        request_id = f"{self.symbol}_{interval}_{int(start_time)}"
        
        try:
            logger.info(f"Fetching historical data: {self.symbol} at {interval} interval, limit={limit}")
            endpoint = f"{self.base_url}/klines"
            params = {
                "symbol": self.symbol,
                "interval": interval,
                "limit": limit
            }
            
            logger.debug(f"Historical API request {request_id}: GET {endpoint} with params {params}")
            
            response = requests.get(endpoint, params=params, timeout=30)
            response.raise_for_status()
            
            # Parse response
            data = response.json()
            logger.debug(f"Received {len(data)} historical candles")
            
            df = pd.DataFrame(data, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Convert types
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
            df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric)
            
            elapsed = time.time() - start_time
            logger.info(f"Fetched {len(df)} historical candles for {self.symbol} in {elapsed:.3f}s")
            
            if not df.empty:
                start_date = df['open_time'].min()
                end_date = df['close_time'].max()
                price_range = f"${float(df['low'].min()):.2f} - ${float(df['high'].max()):.2f}"
                
                logger.info(f"Historical data range: {start_date} to {end_date}, price range: {price_range}")
            
            return df
            
        except requests.exceptions.RequestException as e:
            self.error_count += 1
            logger.error(f"Error fetching historical data (request {request_id}): {e}")
            logger.debug(f"Historical data error details: {traceback.format_exc()}")
            
            if hasattr(e, 'response') and e.response is not None:
                logger.debug(f"Historical API error response: {e.response.status_code} - {e.response.text}")
                
            return pd.DataFrame()
    
    def fetch_historical_klines_with_time_range(self, 
                                               start_time: datetime.datetime,
                                               end_time: Optional[datetime.datetime] = None,
                                               interval: str = "1h") -> pd.DataFrame:
        """
        Fetch historical price data for a specific time range
        
        Args:
            start_time: Start time for historical data
            end_time: End time for historical data (defaults to current time)
            interval: Time interval (e.g., "1m", "5m", "1h", "1d")
            
        Returns:
            DataFrame with OHLCV data
        """
        if end_time is None:
            end_time = datetime.datetime.now()
            
        # Convert to milliseconds timestamp for Binance API
        start_ms = int(start_time.timestamp() * 1000)
        end_ms = int(end_time.timestamp() * 1000)
        
        logger.info(f"Fetching {interval} klines from {start_time} to {end_time}")
        
        try:
            endpoint = f"{self.base_url}/klines"
            params = {
                "symbol": self.symbol,
                "interval": interval,
                "startTime": start_ms,
                "endTime": end_ms,
                "limit": 1000  # Max limit per request
            }
            
            logger.debug(f"API Request: GET {endpoint} with params {params}")
            
            response = requests.get(endpoint, params=params, timeout=30)
            response.raise_for_status()
            
            # Parse response
            data = response.json()
            
            if not data:
                logger.warning(f"No data returned for time range")
                return pd.DataFrame()
                
            logger.info(f"Received {len(data)} klines")
            
            df = pd.DataFrame(data, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Convert types
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
            df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric)
            
            return df
            
        except requests.exceptions.RequestException as e:
            self.error_count += 1
            logger.error(f"Error fetching historical data with time range: {e}")
            logger.debug(f"Error details: {traceback.format_exc()}")
            return pd.DataFrame()
    
    def fetch_large_historical_dataset(self, 
                                      start_time: datetime.datetime,
                                      end_time: Optional[datetime.datetime] = None,
                                      interval: str = "1h") -> pd.DataFrame:
        """
        Fetch a large historical dataset by breaking it into smaller chunks
        to avoid API limitations
        
        Args:
            start_time: Start time for historical data
            end_time: End time for historical data (defaults to current time)
            interval: Time interval (e.g., "1m", "5m", "1h", "1d")
            
        Returns:
            DataFrame with OHLCV data
        """
        if end_time is None:
            end_time = datetime.datetime.now()
            
        # Define mapping of interval to milliseconds
        interval_ms = {
            "1m": 60 * 1000,
            "3m": 3 * 60 * 1000,
            "5m": 5 * 60 * 1000,
            "15m": 15 * 60 * 1000,
            "30m": 30 * 60 * 1000,
            "1h": 60 * 60 * 1000,
            "2h": 2 * 60 * 60 * 1000,
            "4h": 4 * 60 * 60 * 1000,
            "6h": 6 * 60 * 60 * 1000,
            "8h": 8 * 60 * 60 * 1000,
            "12h": 12 * 60 * 60 * 1000,
            "1d": 24 * 60 * 60 * 1000,
            "3d": 3 * 24 * 60 * 60 * 1000,
            "1w": 7 * 24 * 60 * 60 * 1000,
        }
        
        # Calculate chunk size based on interval
        # Each chunk will request 1000 candles (max API limit)
        chunk_size_ms = 1000 * interval_ms.get(interval, 60 * 60 * 1000)
        
        # Convert to milliseconds timestamp for Binance API
        current_start_ms = int(start_time.timestamp() * 1000)
        end_ms = int(end_time.timestamp() * 1000)
        
        all_df = []
        chunk_count = 0
        total_records = 0
        
        logger.info(f"Fetching large historical dataset from {start_time} to {end_time} with {interval} interval")
        
        while current_start_ms < end_ms:
            chunk_count += 1
            
            # Calculate end time for this chunk
            chunk_end_ms = min(current_start_ms + chunk_size_ms, end_ms)
            
            # Convert ms timestamps back to datetime for logging
            chunk_start_dt = datetime.datetime.fromtimestamp(current_start_ms / 1000)
            chunk_end_dt = datetime.datetime.fromtimestamp(chunk_end_ms / 1000)
            
            logger.info(f"Fetching chunk {chunk_count}: {chunk_start_dt} to {chunk_end_dt}")
            
            try:
                endpoint = f"{self.base_url}/klines"
                params = {
                    "symbol": self.symbol,
                    "interval": interval,
                    "startTime": current_start_ms,
                    "endTime": chunk_end_ms,
                    "limit": 1000  # Max limit per request
                }
                
                response = requests.get(endpoint, params=params, timeout=30)
                response.raise_for_status()
                
                # Parse response
                data = response.json()
                
                if data:
                    df_chunk = pd.DataFrame(data, columns=[
                        'open_time', 'open', 'high', 'low', 'close', 'volume',
                        'close_time', 'quote_asset_volume', 'number_of_trades',
                        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
                    ])
                    
                    # Convert types
                    df_chunk['open_time'] = pd.to_datetime(df_chunk['open_time'], unit='ms')
                    df_chunk['close_time'] = pd.to_datetime(df_chunk['close_time'], unit='ms')
                    numeric_columns = ['open', 'high', 'low', 'close', 'volume']
                    df_chunk[numeric_columns] = df_chunk[numeric_columns].apply(pd.to_numeric)
                    
                    all_df.append(df_chunk)
                    total_records += len(df_chunk)
                    logger.info(f"Chunk {chunk_count}: Retrieved {len(df_chunk)} records")
                    
                    # Update start time for next chunk - use the last close_time + 1ms
                    if not df_chunk.empty:
                        # Get timestamp in ms from the last row's close_time
                        last_close_time = df_chunk['close_time'].iloc[-1].timestamp() * 1000
                        current_start_ms = int(last_close_time) + 1
                    else:
                        # If no data, move to next chunk
                        current_start_ms = chunk_end_ms + 1
                else:
                    logger.warning(f"No data for chunk {chunk_count}")
                    current_start_ms = chunk_end_ms + 1
                
                # Add a small delay to avoid rate limits
                time.sleep(0.5)
                
            except requests.exceptions.RequestException as e:
                self.error_count += 1
                logger.error(f"Error fetching chunk {chunk_count}: {e}")
                
                # If we get an error, try to continue with the next chunk
                current_start_ms = chunk_end_ms + 1
                
                # Add a longer delay after an error
                time.sleep(2)
        
        # Combine all chunks into a single DataFrame
        if all_df:
            result_df = pd.concat(all_df, ignore_index=True)
            
            # Remove any duplicate records (can happen at chunk boundaries)
            result_df = result_df.drop_duplicates(subset=['open_time'])
            
            # Sort by time
            result_df = result_df.sort_values('open_time')
            
            logger.info(f"Successfully fetched {len(result_df)} total records across {chunk_count} chunks")
            
            # Calculate time range and price range
            if not result_df.empty:
                start_date = result_df['open_time'].min()
                end_date = result_df['close_time'].max()
                price_range = f"${float(result_df['low'].min()):.2f} - ${float(result_df['high'].max()):.2f}"
                
                logger.info(f"Data range: {start_date} to {end_date}, price range: {price_range}")
            
            return result_df
        else:
            logger.error("Failed to retrieve any data")
            return pd.DataFrame()
    
    def save_dataset_to_csv(self, df: pd.DataFrame, filename: str) -> bool:
        """Save a DataFrame to CSV file"""
        try:
            # Create directory if needed
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            df.to_csv(filename, index=False)
            logger.info(f"Saved {len(df)} records to {filename}")
            return True
        except Exception as e:
            logger.error(f"Error saving to CSV: {e}")
            return False
    
    def load_dataset_from_csv(self, filename: str) -> pd.DataFrame:
        """Load a DataFrame from CSV file"""
        try:
            df = pd.read_csv(filename)
            
            # Convert string dates back to datetime
            if 'open_time' in df.columns:
                df['open_time'] = pd.to_datetime(df['open_time'])
            if 'close_time' in df.columns:
                df['close_time'] = pd.to_datetime(df['close_time'])
                
            logger.info(f"Loaded {len(df)} records from {filename}")
            return df
        except Exception as e:
            logger.error(f"Error loading from CSV: {e}")
            return pd.DataFrame()
    
    def get_stats(self) -> Dict[str, Any]:
        """Return statistics about price fetching"""
        return {
            "symbol": self.symbol,
            "fetch_count": self.fetch_count,
            "error_count": self.error_count,
            "success_rate": (self.fetch_count - self.error_count) / max(1, self.fetch_count) * 100,
            "price_history_length": len(self.price_history),
            "price_range": {
                "min": min([p["price"] for p in self.price_history]) if self.price_history else None,
                "max": max([p["price"] for p in self.price_history]) if self.price_history else None,
                "latest": self.price_history[-1]["price"] if self.price_history else None
            },
            "time_range": {
                "first": time.strftime("%Y-%m-%d %H:%M:%S", 
                                       time.localtime(self.price_history[0]["timestamp"])) if self.price_history else None,
                "last": time.strftime("%Y-%m-%d %H:%M:%S", 
                                      time.localtime(self.price_history[-1]["timestamp"])) if self.price_history else None
            }
        }

# Simple test
if __name__ == "__main__":
    # Configure logging for the test
    logging.basicConfig(level=logging.DEBUG)
    
    fetcher = PriceFetcher()
    price = fetcher.get_current_price()
    print(f"Current BTC price: {price}")
    
    # Example of fetching a large historical dataset
    # Data directory
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    
    # CSV file path
    csv_file = os.path.join(data_dir, "btcusdt_1h_3months.csv")
    
    # Check if we already have the data
    if os.path.exists(csv_file):
        print(f"Loading data from {csv_file}")
        df = fetcher.load_dataset_from_csv(csv_file)
    else:
        # Fetch 3 months of hourly data
        end_time = datetime.datetime.now()
        start_time = end_time - datetime.timedelta(days=90)
        
        print(f"Fetching 3 months of hourly data from {start_time} to {end_time}")
        df = fetcher.fetch_large_historical_dataset(start_time, end_time, interval="1h")
        
        # Save to CSV for future use
        if not df.empty:
            fetcher.save_dataset_to_csv(df, csv_file)
    
    # Display information about the dataset
    if not df.empty:
        print(f"\nDataset Information:")
        print(f"Shape: {df.shape}")
        print(f"Date Range: {df['open_time'].min()} to {df['open_time'].max()}")
        print(f"Price Range: ${df['low'].min():.2f} to ${df['high'].max():.2f}")
        print(f"\nSample data:")
        print(df.head()) 