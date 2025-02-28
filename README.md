# Monies - Crypto Wallet & Trend Analysis

A comprehensive Streamlit application for cryptocurrency wallet management, AI-driven trend analysis, and ETF investment tools.

[![Code Quality](https://img.shields.io/badge/code%20quality-10-green)]()
[![GitHub Actions](https://img.shields.io/badge/CI-passing-green)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

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
git clone https://github.com/yourusername/monies.git
cd monies
```

2. Run the setup script:
```bash
# Standard setup
./scripts/setup.sh

# Development setup (includes dev tools)
./scripts/setup.sh dev
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

4. OAuth Configuration (Optional):
```bash
cp oauth.env.example .env
# Or add OAuth variables to your existing .env file
# Add your OAuth credentials for any of the supported providers:
# - Google
# - Facebook
# - Twitter
# - GitHub
# - Microsoft
# - Coinbase
```

5. Run the OAuth database migration:
```bash
./scripts/migrate_oauth.py
```

## Running the Application

1. Initialize the database (if not already done by the setup script):
```bash
./scripts/run.sh db-init
# or
python init_db.py
```

2. Start the Streamlit app:
```bash
./scripts/run.sh
# or
streamlit run app.py
```

3. Open your browser and navigate to http://localhost:8501

## Usage

### Authentication

- Register a new account or use the demo account:
  - Username: demo
  - Password: password123
- Log in with multiple OAuth providers:
  - Google, Facebook, Twitter, GitHub, Microsoft, and Coinbase
- Connect your Coinbase account for seamless wallet integration

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
- OAuth 2.0 for secure social login
- Encrypted OAuth tokens for third-party API access
- Secure HTTP-only cookies
- Principle of least privilege for API permissions

## Development

### Quick Start

We provide several ways to run common tasks:

#### Using Scripts

```bash
# Run application
./scripts/run.sh

# Run tests
./scripts/test.sh

# Lint code
./scripts/lint.sh

# Format code
./scripts/format.sh
```

#### Using Just

If you have [Just](https://github.com/casey/just) installed:

```bash
# List available commands
just

# Setup development environment
just setup

# Run the application
just run

# Run tests
just test

# Run linters
just lint

# Format code
just format

# Clean cache files
just clean

# Install pre-commit hooks
just hooks
```

#### Using Make

```bash
# List available commands
make help

# Setup development environment
make setup

# Run the application
make run

# Run tests
make test

# Run linters
make lint

# Format code
make format
```

### Code Quality Tools

This project uses several tools to ensure code quality:

- **Black**: Code formatting
- **isort**: Import sorting
- **Flake8**: Linting
- **MyPy**: Type checking
- **Bandit**: Security checks
- **Pytest**: Testing
- **Pre-commit**: Git hooks

### Pre-commit Hooks

Pre-commit hooks are installed automatically when you run `setup.sh dev`. They ensure code quality before each commit.

To manually install or update pre-commit hooks:

```bash
pre-commit install
```

To run pre-commit checks on all files:

```bash
pre-commit run --all-files
```

### VS Code Integration

If you're using VS Code, we provide settings and extension recommendations for an optimal development experience.

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
