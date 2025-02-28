# Crypto Wallet & Trend Analysis

A comprehensive Streamlit application for cryptocurrency wallet management, AI-driven trend analysis, and ETF investment tools.

## Features

- **Multi-Currency Wallet Management**: Track and manage multiple cryptocurrency wallets and exchange accounts.
- **AI-Powered Analysis**: Get insights on stocks, cryptocurrencies, and market trends using OpenAI and Ollama.
- **ETF Investment Tools**: Explore and invest in ETFs with AI-driven recommendations.
- **Social Media Integration**: Connect social accounts, schedule posts, and analyze social sentiment.
- **Secure Backend**: Encrypted API key storage and robust authentication.

## Installation

### Prerequisites

- Python 3.9+
- Virtual environment (recommended)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/crypto-wallet-trend-analysis.git
cd crypto-wallet-trend-analysis
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

## Running the Application

1. Initialize the database:
```bash
python -m src.utils.db_init
```

2. Start the Streamlit app:
```bash
streamlit run app.py
```

3. Open your browser and navigate to http://localhost:8501

## Usage

### Authentication

- Register a new account or use the demo account:
  - Username: demo
  - Password: password123

### Wallet Management

- Connect exchange accounts using API keys
- Add on-chain wallets by address
- View balances and transaction history
- Send and receive cryptocurrency

### AI Analysis

- Analyze stock trends with historical data
- Get cryptocurrency market insights
- Receive ETF recommendations based on risk profile
- Ask custom questions about market trends

### Social Media

- Connect social accounts (Twitter, Facebook, LinkedIn)
- Create and schedule posts
- Analyze social sentiment for assets
- Auto-generate posts about your portfolio or market trends

### Settings

- Manage account information
- Configure API keys
- Set application preferences
- Export data for record keeping

## Project Structure

```
crypto-wallet-trend-analysis/
├── app.py               # Main Streamlit application
├── requirements.txt     # Python dependencies
├── .env.example         # Example environment variables
├── data/                # Database and data files
├── src/
│   ├── api/             # External API integrations
│   ├── components/      # Reusable UI components
│   ├── models/          # Database models
│   ├── pages/           # Application pages
│   ├── tests/           # Test suite
│   └── utils/           # Utility functions
```

## Security

- API keys are encrypted before storage
- Passwords are hashed using bcrypt
- JWT tokens for authentication
- Secure HTTP-only cookies
- Principle of least privilege for API permissions

## Development

### Running Tests

```bash
pytest
```

### Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit your changes: `git commit -am 'Add feature'`
4. Push to the branch: `git push origin feature-name`
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Streamlit](https://streamlit.io/) for the web framework
- [OpenAI](https://openai.com/) and [Ollama](https://ollama.ai/) for AI capabilities
- [CCXT](https://github.com/ccxt/ccxt) for cryptocurrency exchange integration
- [yfinance](https://github.com/ranaroussi/yfinance) for financial data