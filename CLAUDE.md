# Monies Project Guidelines

## Overview
Monies is a comprehensive cryptocurrency portfolio management and market analysis app built with Streamlit, offering real-time tracking of crypto wallets, AI-powered market insights, and seamless exchange integrations.

## Feature Summary
- **Wallet Management**: Track crypto balances across exchanges and on-chain wallets
- **Exchange Integration**: Connect to Binance, Coinbase, and other exchanges via API keys or OAuth
- **Market Analysis**: Real-time price data, charts, and historical trends
- **AI Insights**: AI-powered analysis of market trends and portfolio recommendations
- **Portfolio Optimization**: Tools for portfolio optimization based on risk profiles
- **Social Media Integration**: Connect social accounts and analyze social sentiment
- **OAuth Support**: Simplified authentication for supported platforms
- **Security**: Encrypted API keys, password hashing, and JWT tokens

## Build/Test/Lint Commands
- Run application: `./scripts/run.sh` or `streamlit run app.py`
- Run tests: `./scripts/test.sh` or `pytest`
- Run specific test: `pytest tests/path/to/test_file.py::test_function_name`
- Run with coverage: `pytest --cov=src --cov-report=term --cov-report=html`
- Lint code: `./scripts/lint.sh` or `flake8 src/ tests/`
- Type checking: `mypy src/ tests/`
- Format code: `./scripts/format.sh` or `black src/ tests/ && isort src/ tests/`
- Install dev dependencies: `pip install -r requirements-dev.txt`
- Cache market data: `./scripts/cache_data.sh` (optional flags: `--symbols SPY,QQQ,AAPL` `--periods 1mo,3mo,1y` `--cache-hours 48`)
- Database migrations: `python scripts/migrate_api_keys.py` and `python scripts/migrate_oauth_scopes.py`

## Alternative Command Runners
- Using Makefile: `make <command>` (run `make help` to see available commands)
- Using Justfile: `just <command>` (run `just` to see available commands)

## Developer Experience Tools
- Pre-commit hooks: Install with `pre-commit install` or set up automatically with `./scripts/setup.sh dev`
- Run pre-commit checks manually: `pre-commit run --all-files`
- Clean project: `find . -type d -name "__pycache__" -exec rm -rf {} +` or `just clean` or `make clean`
- VS Code settings are provided in `.vscode/settings.json`
- Editor configuration in `.editorconfig`
- Linting and formatting configuration in `pyproject.toml` and `.flake8`

## Code Style Guidelines
- **Imports**: Standard library first, third-party next, local modules last
- **Typing**: Use type hints for function parameters and return values
- **Naming**: snake_case for variables/functions, CamelCase for classes
- **Documentation**: Docstrings for all functions, classes, and modules
- **Error Handling**: Use try/except blocks with specific exceptions
- **Security**: Never hardcode secrets, use environment variables via python-dotenv
- **Testing**: Write tests for all new functionality, use pytest fixtures when appropriate
- **SQLAlchemy**: Use ORM models, proper relationship definitions, session management
- **Streamlit**: Use st.cache for expensive operations, organize pages with proper hierarchy
- **API Clients**: Use proper error handling and rate limiting for external API calls

## Data Source and Caching Guidelines
- **ALWAYS** prefer live data from APIs over cached data
- **ALWAYS** attempt to fetch live data first before falling back to cached data
- Update cached data with fresh data whenever possible
- Cache data with appropriate expiration time to reduce API calls when necessary
- **NEVER USE SYNTHETIC DATA** - always show error if cannot connect to actual data source
- For chart displays, if data from multiple sources is needed, try to get all from a single consistent source
- Run `./scripts/cache_data.sh` periodically to pre-populate the cache with historical data
- Data sources priority: Yahoo Finance → Alpha Vantage → Financial Modeling Prep → Fallback to cached data

## Security Guidelines
- **API Keys**: Always encrypt API keys and secrets before storing in database
- **Passwords**: Use bcrypt for password hashing
- **OAuth Tokens**: Encrypt OAuth tokens and refresh tokens
- **JWT**: Use JWT tokens for authentication with proper expiration
- **Environment Variables**: Store sensitive configuration in environment variables
- **Rate Limiting**: Implement rate limiting for API endpoints
- **Input Validation**: Validate all user inputs to prevent injection attacks

## OAuth Integration
- **Supported Providers**: Google, Coinbase, Facebook, Twitter, GitHub, Microsoft
- **API Integration**: Coinbase OAuth tokens can be used directly for API calls
- **Multiple Keys**: Support for multiple API keys for the same service
- **Default Selection**: Users can select which API key to use as default
- **Token Refresh**: OAuth tokens are automatically refreshed when they expire

## Exchange Integration
- **Binance/Binance.US**: Connect via API keys with appropriate permissions
- **Coinbase**: Connect via API keys or OAuth for simplified authentication
- **Other Exchanges**: Use CCXT library for standardized access to multiple exchanges
- **Wallet Balance**: View balances across exchanges with automatic updates
- **Transaction History**: View transaction history from connected exchanges
- **Trading**: Execute trades directly from the application

Remember to keep database interactions secure and properly handle encryption/decryption for sensitive data.
