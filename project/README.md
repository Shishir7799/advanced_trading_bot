# AI-Powered Cryptocurrency Trading Bot

![Project Banner](https://img.shields.io/badge/AI%20Trading%20Bot-Machine%20Learning%20%2B%20Blockchain-blue)

A sophisticated cryptocurrency trading bot that combines advanced machine learning algorithms with blockchain technology. This bot analyzes BTC/USDT price patterns using ensemble learning models, makes price movement predictions, and executes trades automatically. Transactions are recorded on the Aptos blockchain, enabling secure and transparent operation.

## 🔍 Project Overview

This bot uses a combination of Random Forest and XGBoost machine learning models to predict cryptocurrency price movements. It works by:

1. **Data Collection**: Fetches real-time BTC/USDT price data from Binance API
2. **Feature Engineering**: Generates technical indicators (RSI, MACD, Bollinger Bands, etc.)
3. **Prediction**: Applies trained ML models to predict price direction
4. **Trading**: Simulates or executes trades based on high-confidence predictions
5. **Blockchain Integration**: Records transactions on the Aptos blockchain
6. **Performance Analysis**: Provides extensive backtesting and visualization tools

The system uses an ensemble approach that combines predictions from multiple models to increase accuracy and reduce false signals.

## 🛠️ Tech Stack

### Machine Learning
- **Models**: Random Forest, XGBoost
- **Technical Indicators**: RSI, MACD, Bollinger Bands, Stochastic Oscillator, etc.
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn, Plotly

### Backend
- **Framework**: Python (core logic)
- **API Integration**: Binance API (crypto data)
- **Database**: JSON/CSV files (model data, transactions)

### Web Interface
- **Framework**: Flask
- **Frontend**: HTML/CSS/JavaScript, Bootstrap
- **Visualization**: Plotly.js

### Blockchain & DeFi
- **Network**: Aptos Testnet
- **Oracles**: Allora Predictive Oracle
- **Secure Compute**: Marlin TEE
- **Integration**: Aptos REST API
- **Operations**: Wallet management, transaction signing, agent metadata

## 🚀 Features

- **ML-Powered Price Prediction**: Uses ensemble learning to forecast price movements
- **Comprehensive Backtesting**: Test strategies on historical data with detailed performance metrics
- **Live Trading Dashboard**: Real-time visualization of predictions, trades and portfolio value
- **Risk Management**: Configurable trade size and confidence thresholds
- **Aptos Blockchain Integration**: Secure transaction recording on the Aptos testnet
- **Strategy Comparison**: Compare different ML models and trading strategies
- **Performance Visualization**: Detailed charts for analysis and optimization
- **Secure Predictions**: TEE-protected model execution through Marlin
- **Oracle Validation**: Allora predictive oracle confirms AI signals
- **Agent Metadata**: Comprehensive tracking of all decisions with auditability

## 📊 Dashboard

The trading bot includes a web-based dashboard built with Flask and Plotly that provides:

- Real-time BTC/USDT price data
- Current model predictions with confidence levels
- Portfolio value tracking
- Recent trades history
- Performance visualization
- Integration with Aptos blockchain transactions
- Secure prediction verification status
- Oracle signal agreement indicators

## 🔄 Workflow

1. The bot fetches current BTC/USDT price data from Binance
2. Technical indicators are calculated to create feature sets
3. ML models (Random Forest & XGBoost) predict price direction
4. An ensemble logic combines these predictions
5. If confidence exceeds threshold, a trade is simulated/executed
6. Trade details are recorded on the Aptos blockchain
7. Performance is tracked and visualized on the dashboard

## 🔧 Setup and Installation

### Prerequisites
- Python 3.8+
- pip (Python package manager)
- Access to Binance API (or historical data CSV)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/crypto-ai-bot.git
cd crypto-ai-bot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
# Create a .env file with your API keys if using Binance
BINANCE_API_KEY=your_api_key
BINANCE_API_SECRET=your_api_secret
```

### Usage

#### Training Models
```bash
python train_models.py --days 90 --symbol BTCUSDT
```

#### Running Backtests
```bash
python backtesting.py --strategy ensemble --days 30 --interval 1h
```

#### Starting the Bot
```bash
python trading_bot.py
```

#### Running the Dashboard
```bash
python dashboard.py
```
Then open http://localhost:5000 in your browser.

#### Testing Aptos Integration
```bash
python aptos_integration.py
```

## 🧪 Backtesting

The backtesting module allows you to:

- Test trading strategies on historical data
- Compare different ML models and ensemble approaches
- Calculate key performance metrics (accuracy, win rate, P&L)
- Generate detailed visualizations of strategy performance
- Compare against "buy and hold" benchmark
- Optimize model parameters and trading thresholds

Example backtesting command:
```bash
python backtesting.py --strategy compare --days 60 --interval 1h
```

## 🔗 Aptos Blockchain Integration

The bot integrates with the Aptos blockchain for:

- Secure wallet creation and management
- Recording trade transactions
- Tracking portfolio value
- Creating an immutable audit trail of activities

The system first simulates trades and then can optionally execute them on the blockchain, providing transparency and security.

## 🌐 AI + DeFi Integration

This trading bot represents a new generation of decentralized autonomous DeFi agents that combine multiple AI and blockchain technologies to create a secure, transparent, and efficient trading system:

### Decentralized Autonomous Agent Architecture

The bot functions as a decentralized autonomous agent with three key components working in harmony:

1. **AI Prediction Core**: Local ML models make base predictions
2. **Allora Oracle Validation**: External oracle provides consensus verification
3. **Marlin TEE Secure Compute**: Secure computation environment protects sensitive models
4. **Aptos Blockchain Settlement**: Immutable transaction recording and settlement

### Modular AI and Web3 Components

#### 🧠 ML Prediction Module
- Ensemble of RandomForest and XGBoost models
- Analyzes price patterns and technical indicators
- Provides initial trading signals with confidence levels
- Completely transparent model scoring and weighting

#### 🔮 Allora Predictive Oracle
- External decentralized price prediction oracle
- Provides independent confirmation of trading signals
- Aggregates signals from multiple sources for consensus
- Acts as a safeguard against model overfitting or bias

#### 🛡️ Marlin TEE Secure Compute
- Trusted Execution Environment for sensitive models
- Enables running proprietary models securely
- Verifiable attestation of model integrity and execution
- Prevents tampering or front-running by securing predictions
- Signs prediction outputs cryptographically for verification

#### ⚖️ Aptos Transaction Settlement
- Records all trades with complete metadata
- Creates immutable audit trail of decisions
- Enables fully transparent operations
- Provides cryptographic proof of trading activity

#### 📝 Agent Metadata System
- Tracks complete decision history with signal sources
- Records model confidence scores and voting results
- Stores secure attestations and verification data
- Creates comprehensive audit trail for accountability

### Integrated Decision Flow

```
┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│                 │      │                 │      │                 │      │                 │
│       ML        │─────▶│     Allora      │─────▶│     Marlin      │─────▶│     Aptos       │
│    Prediction   │Price │   Predictive    │Signal│    Trusted      │Final │   Blockchain    │
│     Models      │Data  │     Oracle      │Valid │   Execution     │Trade │  Transaction    │
│                 │      │                 │      │  Environment    │      │                 │
└─────────────────┘      └─────────────────┘      └─────────────────┘      └─────────────────┘
       Step 1                  Step 2                   Step 3                   Step 4
 
 Local ML ensemble      Oracle confirms or       Secure model runs      Transaction with all
 generates prediction   provides independent     in TEE with secure     metadata recorded on
 with confidence score  price movement signal    attestation & signing  Aptos blockchain
```

### Data Flow Process

1. **ML Predictions** → The local ML ensemble generates initial trade signals
   - Multiple models (RandomForest, XGBoost) vote on price direction
   - Confidence scores are calculated for each prediction
   - Feature importance is analyzed for explainable decisions
   - Initial BUY/HOLD recommendation is made based on ML consensus

2. **Oracle Validation** → Allora predictive oracle confirms or challenges ML signal
   - Independent prediction from decentralized oracle network
   - Provides market perspective from external sources
   - Helps validate ML model predictions
   - Acts as consensus mechanism for enhanced decision quality

3. **Secure Computation** → Marlin TEE executes secure model and signs results
   - Proprietary models run in tamper-proof environment
   - Hardware attestation verifies model hasn't been compromised
   - Prediction results are cryptographically signed
   - Provides verifiable proof that computation occurred correctly
   - Prevents front-running or manipulation of signals

4. **Blockchain Settlement** → Final decision recorded on Aptos blockchain
   - Complete decision metadata stored immutably
   - Includes all model signals, confidence levels, and attestations
   - Creates permanent, auditable record of agent behavior
   - Enables verification of the entire decision process
   - Records transaction details with agent identity

### Benefits of This Architecture

- **Enhanced Security**: Critical predictions protected by TEE
- **Increased Reliability**: Multiple independent signals reduce false trades
- **Full Auditability**: Complete decision record on blockchain
- **Trustless Operation**: Cryptographic verification of decisions
- **Modular Design**: Components can be upgraded individually
- **Oracle Safeguard**: External validation prevents model bias
- **Regulatory Compliance**: Complete audit trail of agent decisions

This integration of AI + DeFi creates a transparent, secure, and auditable autonomous trading agent that leverages the best capabilities of both artificial intelligence and blockchain technology.

## 📝 Future Roadmap

- [ ] Add more technical indicators and price prediction features
- [ ] Implement deep learning models (LSTM, Transformer)
- [ ] Optimize hyperparameters using Bayesian optimization
- [ ] Add support for multiple trading pairs
- [ ] Enhance the web dashboard with more analytics
- [ ] Implement real trading functionality with risk management
- [ ] Expand blockchain integration to include smart contracts
- [ ] Add mobile notifications for trade alerts
- [ ] Implement sentiment analysis from social media
- [ ] Integrate with additional oracle networks
- [ ] Support cross-chain settlement and operations
- [ ] Create multi-agent trading strategies

## 📌 AI + DeFi Innovation

This project demonstrates the powerful synergy between AI and blockchain technology:

- **AI for Trading Decisions**: Machine learning models analyze complex market patterns to make informed trading decisions
- **Blockchain for Transaction Security**: Aptos blockchain provides a secure, immutable record of all trades
- **Transparency & Trust**: All trading activities are recorded on the blockchain, providing transparency
- **DeFi Integration**: The system could be extended to interact with DeFi protocols on Aptos
- **Cross-Chain Potential**: Future versions could extend to cross-chain operations

The combination of AI prediction capabilities with blockchain's transparency creates a new paradigm for algorithmic trading that is both intelligent and trustworthy.

## 📸 Screenshots

![Dashboard Preview](dashboard_preview.png)
![Backtesting Results](backtesting_results.png)
![Strategy Comparison](strategy_comparison.png)

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

## 📬 Contact

Your Name - [@yourtwitter](https://twitter.com/yourtwitter) - email@example.com

Project Link: [https://github.com/yourusername/crypto-ai-bot](https://github.com/yourusername/crypto-ai-bot) 