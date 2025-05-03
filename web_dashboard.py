import dash
from dash import dcc, html, Input, Output, State, dash_table
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta
import argparse
import threading
import time
from typing import Dict, List, Any, Optional

from utils import load_from_json, ensure_directories_exist, format_timestamp
from fetch_crypto_prices import PriceFetcher

# Initialize the Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

# Global state
global_state = {
    "symbol": "BTCUSDT",
    "price_data": pd.DataFrame(),
    "trade_history": [],
    "model_predictions": [],
    "last_update": None,
    "running": False,
    "refresh_interval": 60  # seconds
}

# Create required directories
ensure_directories_exist(["logs", "data", "results"])

# Define app layout
app.layout = html.Div([
    html.H1("AI Crypto Trading Bot Dashboard", style={"textAlign": "center"}),
    
    html.Div([
        html.Div([
            html.H3("Trading Pair"),
            dcc.Dropdown(
                id="symbol-dropdown",
                options=[
                    {"label": "BTC/USDT", "value": "BTCUSDT"},
                    {"label": "ETH/USDT", "value": "ETHUSDT"},
                    {"label": "BNB/USDT", "value": "BNBUSDT"},
                    {"label": "ADA/USDT", "value": "ADAUSDT"},
                    {"label": "SOL/USDT", "value": "SOLUSDT"}
                ],
                value="BTCUSDT",
                clearable=False
            ),
            
            html.H3("Auto Refresh"),
            dcc.Checklist(
                id="auto-refresh",
                options=[{"label": "Enable", "value": "enabled"}],
                value=[]
            ),
            
            html.Button("Refresh Data", id="refresh-button", n_clicks=0),
            
            html.Div(id="last-update-info", style={"marginTop": "20px"})
        ], style={"width": "25%", "display": "inline-block", "verticalAlign": "top", "padding": "20px"}),
        
        html.Div([
            html.H3("Current Status"),
            dcc.Loading(id="loading-status", children=[
                html.Div(id="status-panel")
            ]),
        ], style={"width": "70%", "display": "inline-block", "verticalAlign": "top", "padding": "20px"})
    ]),
    
    html.Div([
        dcc.Tabs([
            dcc.Tab(label="Price Chart", children=[
                dcc.Graph(id="price-chart")
            ]),
            
            dcc.Tab(label="Model Predictions", children=[
                dcc.Graph(id="prediction-chart")
            ]),
            
            dcc.Tab(label="Trade History", children=[
                dash_table.DataTable(
                    id="trade-table",
                    columns=[
                        {"name": "Time", "id": "time"},
                        {"name": "Type", "id": "type"},
                        {"name": "Price", "id": "price"},
                        {"name": "Amount", "id": "amount"},
                        {"name": "Total", "id": "total"},
                        {"name": "Model", "id": "model"}
                    ],
                    style_table={"overflowX": "auto"},
                    style_cell={"textAlign": "left"},
                    style_header={"backgroundColor": "rgb(230, 230, 230)", "fontWeight": "bold"}
                )
            ]),
            
            dcc.Tab(label="Performance Metrics", children=[
                html.Div(id="performance-metrics")
            ])
        ])
    ], style={"padding": "20px"}),
    
    # Hidden div for storing data
    html.Div(id="price-data-store", style={"display": "none"}),
    html.Div(id="trade-history-store", style={"display": "none"}),
    html.Div(id="predictions-store", style={"display": "none"}),
    
    # Interval for auto refresh
    dcc.Interval(
        id="auto-refresh-interval",
        interval=60 * 1000,  # 60 seconds in milliseconds
        n_intervals=0,
        disabled=True
    )
])

@app.callback(
    Output("auto-refresh-interval", "disabled"),
    Input("auto-refresh", "value")
)
def toggle_auto_refresh(value):
    return "enabled" not in value

@app.callback(
    [
        Output("price-data-store", "children"),
        Output("trade-history-store", "children"),
        Output("predictions-store", "children"),
        Output("last-update-info", "children")
    ],
    [
        Input("refresh-button", "n_clicks"),
        Input("symbol-dropdown", "value"),
        Input("auto-refresh-interval", "n_intervals")
    ]
)
def fetch_data(n_clicks, symbol, n_intervals):
    global global_state
    
    # Update the current symbol
    global_state["symbol"] = symbol
    
    # Fetch price data
    try:
        price_fetcher = PriceFetcher(symbol=symbol)
        
        # Get last 7 days of hourly data
        end_time = datetime.now()
        start_time = end_time - timedelta(days=7)
        
        df = price_fetcher.fetch_large_historical_dataset(
            start_time=start_time,
            end_time=end_time,
            interval="1h"
        )
        
        global_state["price_data"] = df
        
        # Convert to JSON for storage
        price_data_json = df.reset_index().to_json(date_format="iso", orient="split")
    except Exception as e:
        price_data_json = "{}"
        print(f"Error fetching price data: {e}")
    
    # Load trade history
    try:
        trades_file = f"transaction_logs/{symbol.lower()}_trades.json"
        if os.path.exists(trades_file):
            trades = load_from_json(trades_file)
            if trades:
                global_state["trade_history"] = trades
                trades_json = json.dumps(trades)
            else:
                trades_json = "[]"
        else:
            trades_json = "[]"
    except Exception as e:
        trades_json = "[]"
        print(f"Error loading trade history: {e}")
    
    # Load model predictions
    try:
        predictions_file = f"results/{symbol.lower()}_predictions.json"
        if os.path.exists(predictions_file):
            predictions = load_from_json(predictions_file)
            if predictions:
                global_state["model_predictions"] = predictions
                predictions_json = json.dumps(predictions)
            else:
                predictions_json = "[]"
        else:
            predictions_json = "[]"
    except Exception as e:
        predictions_json = "[]"
        print(f"Error loading predictions: {e}")
    
    # Update last refresh time
    global_state["last_update"] = datetime.now()
    last_update_info = html.Div([
        html.P(f"Last updated: {global_state['last_update'].strftime('%Y-%m-%d %H:%M:%S')}"),
        html.P(f"Trading pair: {symbol}")
    ])
    
    return price_data_json, trades_json, predictions_json, last_update_info

@app.callback(
    Output("price-chart", "figure"),
    Input("price-data-store", "children"),
    Input("trade-history-store", "children")
)
def update_price_chart(price_data_json, trades_json):
    # Initialize empty figure
    fig = go.Figure()
    
    # Parse price data
    try:
        df = pd.read_json(price_data_json, orient="split")
        
        # Create candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=df["index"],
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
                name="Price"
            )
        )
        
        # Parse trade data
        if trades_json != "[]":
            trades = json.loads(trades_json)
            
            # Add buy markers
            buy_times = []
            buy_prices = []
            
            # Add sell markers
            sell_times = []
            sell_prices = []
            
            for trade in trades:
                if trade["type"] == "BUY":
                    buy_times.append(trade["time"])
                    buy_prices.append(trade["price"])
                elif trade["type"] == "SELL":
                    sell_times.append(trade["time"])
                    sell_prices.append(trade["price"])
            
            # Add buy markers
            fig.add_trace(
                go.Scatter(
                    x=buy_times,
                    y=buy_prices,
                    mode="markers",
                    marker=dict(symbol="triangle-up", size=15, color="green"),
                    name="Buy"
                )
            )
            
            # Add sell markers
            fig.add_trace(
                go.Scatter(
                    x=sell_times,
                    y=sell_prices,
                    mode="markers",
                    marker=dict(symbol="triangle-down", size=15, color="red"),
                    name="Sell"
                )
            )
    except Exception as e:
        print(f"Error updating price chart: {e}")
    
    # Update layout
    fig.update_layout(
        title=f"{global_state['symbol']} Price Chart",
        xaxis_title="Time",
        yaxis_title="Price (USDT)",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

@app.callback(
    Output("prediction-chart", "figure"),
    Input("price-data-store", "children"),
    Input("predictions-store", "children")
)
def update_prediction_chart(price_data_json, predictions_json):
    # Initialize empty figure
    fig = go.Figure()
    
    # Parse price data
    try:
        df = pd.read_json(price_data_json, orient="split")
        
        # Add price line
        fig.add_trace(
            go.Scatter(
                x=df["index"],
                y=df["close"],
                mode="lines",
                name="Price",
                line=dict(color="black", width=1)
            )
        )
        
        # Parse predictions data
        if predictions_json != "[]":
            predictions = json.loads(predictions_json)
            
            # Prepare data for visualization
            times = []
            rf_predictions = []
            rf_confidences = []
            xgb_predictions = []
            xgb_confidences = []
            ensemble_predictions = []
            
            for pred in predictions:
                times.append(pred["timestamp"])
                
                # RandomForest
                if "rf_prediction" in pred:
                    rf_dir = 1 if pred["rf_prediction"] == "UP" else 0
                    rf_predictions.append(rf_dir)
                    rf_confidences.append(pred.get("rf_confidence", 0.5))
                
                # XGBoost
                if "xgb_prediction" in pred:
                    xgb_dir = 1 if pred["xgb_prediction"] == "UP" else 0
                    xgb_predictions.append(xgb_dir)
                    xgb_confidences.append(pred.get("xgb_confidence", 0.5))
                
                # Ensemble
                if "ensemble_prediction" in pred:
                    ens_dir = 1 if pred["ensemble_prediction"] == "UP" else 0
                    ensemble_predictions.append(ens_dir)
            
            # Add predictions
            if rf_predictions:
                fig.add_trace(
                    go.Scatter(
                        x=times,
                        y=[0.5 if pred == 1 else -0.5 for pred in rf_predictions],
                        mode="markers",
                        marker=dict(
                            size=[conf * 20 for conf in rf_confidences],
                            color=["green" if pred == 1 else "red" for pred in rf_predictions],
                            symbol="circle"
                        ),
                        name="RandomForest"
                    )
                )
            
            if xgb_predictions:
                fig.add_trace(
                    go.Scatter(
                        x=times,
                        y=[0.7 if pred == 1 else -0.7 for pred in xgb_predictions],
                        mode="markers",
                        marker=dict(
                            size=[conf * 20 for conf in xgb_confidences],
                            color=["green" if pred == 1 else "red" for pred in xgb_predictions],
                            symbol="square"
                        ),
                        name="XGBoost"
                    )
                )
            
            if ensemble_predictions:
                fig.add_trace(
                    go.Scatter(
                        x=times,
                        y=[0.9 if pred == 1 else -0.9 for pred in ensemble_predictions],
                        mode="markers",
                        marker=dict(
                            size=15,
                            color=["green" if pred == 1 else "red" for pred in ensemble_predictions],
                            symbol="star"
                        ),
                        name="Ensemble"
                    )
                )
    except Exception as e:
        print(f"Error updating prediction chart: {e}")
    
    # Update layout with dual y-axis
    fig.update_layout(
        title=f"{global_state['symbol']} Price and Predictions",
        xaxis_title="Time",
        yaxis=dict(
            title="Price (USDT)",
            side="left"
        ),
        yaxis2=dict(
            title="Prediction",
            side="right",
            range=[-1, 1],
            showgrid=False,
            zeroline=True,
            showticklabels=False
        ),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

@app.callback(
    Output("trade-table", "data"),
    Input("trade-history-store", "children")
)
def update_trade_table(trades_json):
    try:
        if trades_json != "[]":
            trades = json.loads(trades_json)
            
            # Format trade data for table
            table_data = []
            for trade in trades:
                table_data.append({
                    "time": format_timestamp(trade["time"]) if "time" in trade else "",
                    "type": trade.get("type", ""),
                    "price": f"{float(trade.get('price', 0)):.2f}",
                    "amount": f"{float(trade.get('amount_crypto', 0)):.6f}",
                    "total": f"{float(trade.get('amount_usdt', 0)):.2f}",
                    "model": trade.get("model", "ensemble")
                })
            
            return table_data
        return []
    except Exception as e:
        print(f"Error updating trade table: {e}")
        return []

@app.callback(
    Output("performance-metrics", "children"),
    [Input("trade-history-store", "children")]
)
def update_performance_metrics(trades_json):
    try:
        if trades_json == "[]":
            return html.Div([
                html.H3("No trading data available")
            ])
        
        trades = json.loads(trades_json)
        
        # Calculate basic metrics
        total_trades = len(trades)
        buy_trades = len([t for t in trades if t.get("type") == "BUY"])
        sell_trades = len([t for t in trades if t.get("type") == "SELL"])
        
        # Calculate profit/loss if available
        total_spent = sum([float(t.get("amount_usdt", 0)) for t in trades if t.get("type") == "BUY"])
        total_received = sum([float(t.get("amount_usdt", 0)) for t in trades if t.get("type") == "SELL"])
        
        net_profit = total_received - total_spent
        roi_pct = (net_profit / total_spent * 100) if total_spent > 0 else 0
        
        # Count by model type
        rf_trades = len([t for t in trades if t.get("model") == "RandomForest"])
        xgb_trades = len([t for t in trades if t.get("model") == "XGBoost"])
        ensemble_trades = len([t for t in trades if t.get("model") == "ensemble"])
        
        return html.Div([
            html.Div([
                html.H3("Trading Summary"),
                html.P(f"Total Trades: {total_trades}"),
                html.P(f"Buy Trades: {buy_trades}"),
                html.P(f"Sell Trades: {sell_trades}"),
            ], style={"width": "30%", "display": "inline-block", "verticalAlign": "top"}),
            
            html.Div([
                html.H3("Profit/Loss Summary"),
                html.P(f"Total Spent: {total_spent:.2f} USDT"),
                html.P(f"Total Received: {total_received:.2f} USDT"),
                html.P(f"Net Profit: {net_profit:.2f} USDT"),
                html.P(f"ROI: {roi_pct:.2f}%"),
            ], style={"width": "30%", "display": "inline-block", "verticalAlign": "top"}),
            
            html.Div([
                html.H3("Model Distribution"),
                html.P(f"RandomForest Trades: {rf_trades}"),
                html.P(f"XGBoost Trades: {xgb_trades}"),
                html.P(f"Ensemble Trades: {ensemble_trades}"),
            ], style={"width": "30%", "display": "inline-block", "verticalAlign": "top"}),
        ])
    except Exception as e:
        print(f"Error updating performance metrics: {e}")
        return html.Div([
            html.H3("Error loading performance metrics")
        ])

@app.callback(
    Output("status-panel", "children"),
    [Input("price-data-store", "children")]
)
def update_status_panel(price_data_json):
    try:
        if price_data_json == "{}":
            return html.Div([
                html.H4("No data available")
            ])
        
        df = pd.read_json(price_data_json, orient="split")
        
        if df.empty:
            return html.Div([
                html.H4("No price data available")
            ])
        
        # Get current price
        current_price = df["close"].iloc[-1]
        
        # Calculate 24h change
        if len(df) >= 24:
            price_24h_ago = df["close"].iloc[-24]
            change_24h = (current_price - price_24h_ago) / price_24h_ago * 100
            change_color = "green" if change_24h >= 0 else "red"
        else:
            change_24h = 0
            change_color = "black"
        
        # Calculate 7d change
        if len(df) >= 7*24:
            price_7d_ago = df["close"].iloc[-7*24]
            change_7d = (current_price - price_7d_ago) / price_7d_ago * 100
            change_7d_color = "green" if change_7d >= 0 else "red"
        else:
            change_7d = 0
            change_7d_color = "black"
        
        return html.Div([
            html.Div([
                html.H4(f"Current Price: {current_price:.2f} USDT"),
                html.H5(f"24h Change: ", style={"display": "inline"}),
                html.H5(f"{change_24h:+.2f}%", style={"color": change_color, "display": "inline"}),
                html.H5(f" | 7d Change: ", style={"display": "inline"}),
                html.H5(f"{change_7d:+.2f}%", style={"color": change_7d_color, "display": "inline"})
            ]),
            
            html.Div([
                html.H4("Recent Prices"),
                dash_table.DataTable(
                    data=[
                        {"Time": format_timestamp(df["index"].iloc[i]), 
                         "Open": f"{df['open'].iloc[i]:.2f}",
                         "High": f"{df['high'].iloc[i]:.2f}",
                         "Low": f"{df['low'].iloc[i]:.2f}",
                         "Close": f"{df['close'].iloc[i]:.2f}",
                         "Volume": f"{df['volume'].iloc[i]:.2f}" if 'volume' in df.columns else "N/A"}
                        for i in range(max(0, len(df) - 5), len(df))
                    ],
                    columns=[
                        {"name": "Time", "id": "Time"},
                        {"name": "Open", "id": "Open"},
                        {"name": "High", "id": "High"},
                        {"name": "Low", "id": "Low"},
                        {"name": "Close", "id": "Close"},
                        {"name": "Volume", "id": "Volume"}
                    ],
                    style_table={"overflowX": "auto"},
                    style_cell={"textAlign": "left"},
                    style_header={"backgroundColor": "rgb(230, 230, 230)", "fontWeight": "bold"}
                )
            ])
        ])
    except Exception as e:
        print(f"Error updating status panel: {e}")
        return html.Div([
            html.H4("Error loading status information")
        ])

def run_dashboard(host="0.0.0.0", port=8050, debug=False):
    app.run_server(host=host, port=port, debug=debug)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the AI Crypto Trading Bot Dashboard")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host IP")
    parser.add_argument("--port", type=int, default=8050, help="Port number")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    
    args = parser.parse_args()
    
    print(f"Starting dashboard on http://{args.host}:{args.port}")
    run_dashboard(host=args.host, port=args.port, debug=args.debug) 