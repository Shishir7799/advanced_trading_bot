from flask import Flask, render_template, jsonify
import pandas as pd
import numpy as np
import threading
import time
import logging
import os
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
import plotly
import plotly.graph_objs as go

# Import custom modules
from fetch_crypto_prices import PriceFetcher
from ensemble_predictor import EnsemblePredictor
from technical_indicators import TechnicalIndicators
from allora_integration import AlloraIntegration

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("dashboard.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global state
global_state = {
    'current_price': 0.0,
    'last_update_time': None,
    'rf_prediction': None,
    'rf_confidence': 0.0,
    'xgb_prediction': None,
    'xgb_confidence': 0.0,
    'ensemble_prediction': None,
    'ensemble_confidence': 0.0,
    'allora_prediction': None,
    'allora_confidence': 0.0,
    'secure_prediction': None,
    'secure_confidence': 0.0,
    'secure_signature': None,
    'secure_enclave_id': None,
    'recent_trades': [],
    'price_history': [],
    'prediction_history': [],
    'portfolio_value': 1000.0,  # Initial portfolio value
    'crypto_holdings': 0.0,
    'usdt_balance': 1000.0,
    'aptos_tx_history': []  # For Aptos blockchain transactions
}

# Initialize data fetcher and predictors
fetcher = PriceFetcher(symbol="BTCUSDT")
ensemble_predictor = EnsemblePredictor(
    symbol="BTCUSDT", 
    use_allora=True,
    use_secure_enclave=True
)
allora = AlloraIntegration(use_mock=True)

def get_latest_data():
    """Fetch latest data and update predictions"""
    try:
        # Get the latest price data
        df = fetcher.fetch_recent_prices(limit=100)
        
        if df is not None and not df.empty:
            # Add technical indicators
            df = TechnicalIndicators.add_all_indicators(df)
            
            # Get current price
            current_price = df['close'].iloc[-1]
            timestamp = df.index[-1]
            
            # Update global state
            global_state['current_price'] = current_price
            global_state['last_update_time'] = timestamp
            
            # Append to price history (keep last 100 points)
            global_state['price_history'].append({
                'timestamp': timestamp,
                'price': current_price
            })
            if len(global_state['price_history']) > 100:
                global_state['price_history'] = global_state['price_history'][-100:]
            
            # Get predictions from ensemble
            prediction, confidence, details = ensemble_predictor.predict(df)
            
            # Extract individual model predictions
            if 'models' in details:
                if 'random_forest' in details['models']:
                    rf_data = details['models']['random_forest']
                    global_state['rf_prediction'] = "BUY" if rf_data['prediction'] == 1 else "SELL" if rf_data['prediction'] == 0 else "HOLD"
                    global_state['rf_confidence'] = rf_data['confidence']
                
                if 'xgboost' in details['models']:
                    xgb_data = details['models']['xgboost']
                    global_state['xgb_prediction'] = "BUY" if xgb_data['prediction'] == 1 else "SELL" if xgb_data['prediction'] == 0 else "HOLD"
                    global_state['xgb_confidence'] = xgb_data['confidence']
                
                if 'allora' in details['models']:
                    allora_data = details['models']['allora']
                    global_state['allora_prediction'] = "BUY" if allora_data['prediction'] == 1 else "SELL" if allora_data['prediction'] == 0 else "HOLD"
                    global_state['allora_confidence'] = allora_data['confidence']
                
                if 'secure_model' in details['models']:
                    secure_data = details['models']['secure_model']
                    global_state['secure_prediction'] = "BUY" if secure_data['prediction'] == 1 else "SELL" if secure_data['prediction'] == 0 else "HOLD"
                    global_state['secure_confidence'] = secure_data['confidence']
                    
                    # Extract secure model metadata
                    if 'metadata' in secure_data:
                        metadata = secure_data['metadata']
                        global_state['secure_signature'] = metadata.get('signature', None)
                        global_state['secure_enclave_id'] = metadata.get('enclave_id', None)
            
            # Update ensemble prediction
            global_state['ensemble_prediction'] = "BUY" if prediction == 1 else "SELL" if prediction == 0 else "HOLD"
            global_state['ensemble_confidence'] = confidence
            
            # Add prediction to history
            prediction_entry = {
                'timestamp': timestamp,
                'rf_prediction': global_state['rf_prediction'],
                'rf_confidence': global_state['rf_confidence'],
                'xgb_prediction': global_state['xgb_prediction'],
                'xgb_confidence': global_state['xgb_confidence'],
                'ensemble_prediction': global_state['ensemble_prediction'],
                'ensemble_confidence': global_state['ensemble_confidence'],
                'price': current_price
            }
            
            # Add Allora data if available
            if global_state['allora_prediction'] is not None:
                prediction_entry['allora_prediction'] = global_state['allora_prediction']
                prediction_entry['allora_confidence'] = global_state['allora_confidence']
                
            # Add secure model data if available
            if global_state['secure_prediction'] is not None:
                prediction_entry['secure_prediction'] = global_state['secure_prediction']
                prediction_entry['secure_confidence'] = global_state['secure_confidence']
            
            global_state['prediction_history'].append(prediction_entry)
            
            if len(global_state['prediction_history']) > 100:
                global_state['prediction_history'] = global_state['prediction_history'][-100:]
            
            # Simulate a trade based on ensemble prediction (for demonstration)
            if len(global_state['prediction_history']) > 1:
                simulate_trade(prediction, confidence, current_price, timestamp)
            
            logger.info(f"Updated data: BTC price ${current_price:.2f}, "
                       f"Ensemble: {global_state['ensemble_prediction']} ({confidence:.2f})")
        else:
            logger.warning("Failed to fetch recent price data")
    
    except Exception as e:
        logger.error(f"Error updating data: {e}")

def simulate_trade(prediction, confidence, price, timestamp):
    """Simulate a trade based on prediction"""
    # Only trade if we have a clear signal with good confidence
    if prediction != -1 and confidence > 0.6:
        trade = {
            'timestamp': timestamp,
            'price': price,
            'type': 'BUY' if prediction == 1 else 'SELL',
            'confidence': confidence
        }
        
        # Simple simulation logic
        if prediction == 1 and global_state['usdt_balance'] > 0:  # Buy signal
            # Use 25% of available balance
            trade_amount_usdt = global_state['usdt_balance'] * 0.25
            fee = trade_amount_usdt * 0.001  # 0.1% fee
            crypto_bought = (trade_amount_usdt - fee) / price
            
            global_state['usdt_balance'] -= trade_amount_usdt
            global_state['crypto_holdings'] += crypto_bought
            
            trade['amount_usdt'] = trade_amount_usdt
            trade['amount_crypto'] = crypto_bought
            trade['fee'] = fee
            
            logger.info(f"Simulated BUY: {crypto_bought:.6f} BTC at ${price:.2f}")
            
        elif prediction == 0 and global_state['crypto_holdings'] > 0:  # Sell signal
            # Sell all holdings
            trade_amount_crypto = global_state['crypto_holdings']
            trade_amount_usdt = trade_amount_crypto * price
            fee = trade_amount_usdt * 0.001  # 0.1% fee
            usdt_received = trade_amount_usdt - fee
            
            global_state['usdt_balance'] += usdt_received
            global_state['crypto_holdings'] = 0
            
            trade['amount_usdt'] = usdt_received
            trade['amount_crypto'] = trade_amount_crypto
            trade['fee'] = fee
            
            logger.info(f"Simulated SELL: {trade_amount_crypto:.6f} BTC at ${price:.2f}")
        
        # Update portfolio value
        global_state['portfolio_value'] = global_state['usdt_balance'] + (global_state['crypto_holdings'] * price)
        
        # Update trade list
        trade['portfolio_value'] = global_state['portfolio_value']
        global_state['recent_trades'].append(trade)
        
        # Keep only the most recent 50 trades
        if len(global_state['recent_trades']) > 50:
            global_state['recent_trades'] = global_state['recent_trades'][-50:]

def update_data_thread():
    """Background thread to periodically update data"""
    while True:
        try:
            get_latest_data()
            time.sleep(60)  # Update every minute
        except Exception as e:
            logger.error(f"Error in update thread: {e}")
            time.sleep(60)  # Wait before retrying

def create_price_chart():
    """Create a price chart with predictions"""
    if not global_state['price_history']:
        return None
    
    # Extract data
    timestamps = [entry['timestamp'] for entry in global_state['price_history']]
    prices = [entry['price'] for entry in global_state['price_history']]
    
    # Create figure
    fig = go.Figure()
    
    # Add price line
    fig.add_trace(go.Scatter(
        x=timestamps, 
        y=prices,
        mode='lines',
        name='BTC/USDT',
        line=dict(color='blue', width=2)
    ))
    
    # Add buy/sell markers from trades
    buy_times = [trade['timestamp'] for trade in global_state['recent_trades'] if trade['type'] == 'BUY']
    buy_prices = [trade['price'] for trade in global_state['recent_trades'] if trade['type'] == 'BUY']
    
    sell_times = [trade['timestamp'] for trade in global_state['recent_trades'] if trade['type'] == 'SELL']
    sell_prices = [trade['price'] for trade in global_state['recent_trades'] if trade['type'] == 'SELL']
    
    if buy_times:
        fig.add_trace(go.Scatter(
            x=buy_times, 
            y=buy_prices,
            mode='markers',
            name='Buy',
            marker=dict(color='green', size=10, symbol='triangle-up')
        ))
    
    if sell_times:
        fig.add_trace(go.Scatter(
            x=sell_times, 
            y=sell_prices,
            mode='markers',
            name='Sell',
            marker=dict(color='red', size=10, symbol='triangle-down')
        ))
    
    # Update layout
    fig.update_layout(
        title='BTC/USDT Price with Trading Signals',
        xaxis_title='Time',
        yaxis_title='Price (USDT)',
        template='plotly_white',
        height=500,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    return fig

def create_portfolio_chart():
    """Create a portfolio value chart"""
    if not global_state['recent_trades']:
        return None
    
    # Extract portfolio values from trades
    timestamps = [trade['timestamp'] for trade in global_state['recent_trades']]
    values = [trade['portfolio_value'] for trade in global_state['recent_trades']]
    
    # Create figure
    fig = go.Figure()
    
    # Add portfolio value line
    fig.add_trace(go.Scatter(
        x=timestamps, 
        y=values,
        mode='lines+markers',
        name='Portfolio Value',
        line=dict(color='purple', width=2),
        marker=dict(size=5)
    ))
    
    # Update layout
    fig.update_layout(
        title='Portfolio Value Over Time',
        xaxis_title='Time',
        yaxis_title='Value (USDT)',
        template='plotly_white',
        height=400,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    return fig

def create_prediction_comparison_chart():
    """Create a chart comparing different prediction sources"""
    if not global_state['prediction_history'] or len(global_state['prediction_history']) < 2:
        return None
    
    # Extract data
    history = global_state['prediction_history']
    timestamps = [entry['timestamp'] for entry in history]
    
    # Create confidence data for each model, mapping predictions to values
    # 1 for Buy, 0 for Hold, -1 for Sell (for visual separation)
    rf_values = []
    xgb_values = []
    allora_values = []
    secure_values = []
    ensemble_values = []
    
    for entry in history:
        # RandomForest
        if entry['rf_prediction'] == 'BUY':
            rf_values.append(1 * entry['rf_confidence'])
        elif entry['rf_prediction'] == 'SELL':
            rf_values.append(-1 * entry['rf_confidence'])
        else:
            rf_values.append(0)
            
        # XGBoost
        if entry['xgb_prediction'] == 'BUY':
            xgb_values.append(1 * entry['xgb_confidence'])
        elif entry['xgb_prediction'] == 'SELL':
            xgb_values.append(-1 * entry['xgb_confidence'])
        else:
            xgb_values.append(0)
            
        # Allora (if available)
        if 'allora_prediction' in entry and entry['allora_prediction'] is not None:
            if entry['allora_prediction'] == 'BUY':
                allora_values.append(1 * entry['allora_confidence'])
            elif entry['allora_prediction'] == 'SELL':
                allora_values.append(-1 * entry['allora_confidence'])
            else:
                allora_values.append(0)
        else:
            allora_values.append(None)
            
        # Secure model (if available)
        if 'secure_prediction' in entry and entry['secure_prediction'] is not None:
            if entry['secure_prediction'] == 'BUY':
                secure_values.append(1 * entry['secure_confidence'])
            elif entry['secure_prediction'] == 'SELL':
                secure_values.append(-1 * entry['secure_confidence'])
            else:
                secure_values.append(0)
        else:
            secure_values.append(None)
            
        # Ensemble
        if entry['ensemble_prediction'] == 'BUY':
            ensemble_values.append(1 * entry['ensemble_confidence'])
        elif entry['ensemble_prediction'] == 'SELL':
            ensemble_values.append(-1 * entry['ensemble_confidence'])
        else:
            ensemble_values.append(0)
    
    # Create figure
    fig = go.Figure()
    
    # Add traces for each model
    fig.add_trace(go.Scatter(
        x=timestamps, 
        y=rf_values,
        mode='lines',
        name='RandomForest',
        line=dict(color='blue', width=1.5)
    ))
    
    fig.add_trace(go.Scatter(
        x=timestamps, 
        y=xgb_values,
        mode='lines',
        name='XGBoost',
        line=dict(color='green', width=1.5)
    ))
    
    # Add Allora predictions if available
    if any(v is not None for v in allora_values):
        fig.add_trace(go.Scatter(
            x=timestamps, 
            y=allora_values,
            mode='lines',
            name='Allora Oracle',
            line=dict(color='orange', width=1.5)
        ))
    
    # Add Secure model predictions if available
    if any(v is not None for v in secure_values):
        fig.add_trace(go.Scatter(
            x=timestamps, 
            y=secure_values,
            mode='lines',
            name='Secure Model (TEE)',
            line=dict(color='red', width=1.5, dash='dot')
        ))
    
    fig.add_trace(go.Scatter(
        x=timestamps, 
        y=ensemble_values,
        mode='lines',
        name='Ensemble',
        line=dict(color='purple', width=2.5)
    ))
    
    # Add horizontal lines for reference
    fig.add_shape(
        type="line", line=dict(dash="dash", width=1, color="green"),
        y0=0.6, y1=0.6, x0=timestamps[0], x1=timestamps[-1],
        layer="below"
    )
    
    fig.add_shape(
        type="line", line=dict(dash="dash", width=1, color="red"),
        y0=-0.6, y1=-0.6, x0=timestamps[0], x1=timestamps[-1],
        layer="below"
    )
    
    # Update layout
    fig.update_layout(
        title='Model Predictions Comparison',
        xaxis_title='Time',
        yaxis_title='Signal Strength (+ Buy / - Sell)',
        template='plotly_white',
        height=400,
        margin=dict(l=0, r=0, t=40, b=0),
        yaxis=dict(
            range=[-1.1, 1.1],
            tickvals=[-1, -0.5, 0, 0.5, 1],
            ticktext=['Strong Sell', 'Sell', 'Hold', 'Buy', 'Strong Buy']
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        )
    )
    
    return fig

# Flask routes
@app.route('/')
def index():
    return render_template('index.html', 
                          state=global_state,
                          portfolio_value=global_state['portfolio_value'],
                          usdt_balance=global_state['usdt_balance'],
                          crypto_holdings=global_state['crypto_holdings'])

@app.route('/api/data')
def get_data():
    """API endpoint to get the latest data"""
    return jsonify({
        'current_price': global_state['current_price'],
        'last_update': global_state['last_update_time'].strftime('%Y-%m-%d %H:%M:%S') if global_state['last_update_time'] else None,
        'rf_prediction': global_state['rf_prediction'],
        'rf_confidence': global_state['rf_confidence'],
        'xgb_prediction': global_state['xgb_prediction'],
        'xgb_confidence': global_state['xgb_confidence'],
        'allora_prediction': global_state['allora_prediction'],
        'allora_confidence': global_state['allora_confidence'],
        'secure_prediction': global_state['secure_prediction'],
        'secure_confidence': global_state['secure_confidence'],
        'secure_enclave_id': global_state['secure_enclave_id'],
        'ensemble_prediction': global_state['ensemble_prediction'],
        'ensemble_confidence': global_state['ensemble_confidence'],
        'portfolio_value': global_state['portfolio_value'],
        'usdt_balance': global_state['usdt_balance'],
        'crypto_holdings': global_state['crypto_holdings']
    })

@app.route('/api/trades')
def get_trades():
    """API endpoint to get recent trades"""
    # Convert datetime objects to strings for JSON serialization
    trades = []
    for trade in global_state['recent_trades']:
        trade_copy = trade.copy()
        trade_copy['timestamp'] = trade['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        trades.append(trade_copy)
    
    return jsonify(trades)

@app.route('/api/price_chart')
def get_price_chart():
    """API endpoint to get price chart"""
    fig = create_price_chart()
    if fig:
        chart_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        return chart_json
    return jsonify({'error': 'No data available'})

@app.route('/api/portfolio_chart')
def get_portfolio_chart():
    """API endpoint to get portfolio chart"""
    fig = create_portfolio_chart()
    if fig:
        chart_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        return chart_json
    return jsonify({'error': 'No data available'})

@app.route('/api/prediction_chart')
def get_prediction_chart():
    """API endpoint to get prediction comparison chart"""
    fig = create_prediction_comparison_chart()
    if fig:
        chart_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        return chart_json
    return jsonify({'error': 'No data available'})

@app.route('/api/secure_model_info')
def get_secure_model_info():
    """API endpoint to get secure model details"""
    return jsonify({
        'secure_prediction': global_state['secure_prediction'],
        'secure_confidence': global_state['secure_confidence'],
        'secure_enclave_id': global_state['secure_enclave_id'],
        'secure_signature': global_state['secure_signature'],
        'timestamp': global_state['last_update_time'].strftime('%Y-%m-%d %H:%M:%S') if global_state['last_update_time'] else None
    })

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Start background update thread
    update_thread = threading.Thread(target=update_data_thread, daemon=True)
    update_thread.start()
    
    # Run Flask app
    app.run(debug=True, use_reloader=False) 