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
- Run all quality checks at once: `./scripts/lint.sh && mypy src/ tests/ && black --check src/ tests/ && isort --check src/ tests/`
- Check for security issues: `bandit -r src/`
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

## Code Quality Configuration
- **flake8**: Configuration in `.flake8`
  - Line length: 88 characters (matching black)
  - Ignored rules: E203, E501, W293, W291, F541, E128, E722
  - Special exclusions for `__init__.py` and test files
- **black**: Configuration in `pyproject.toml`
  - Line length: 88 characters
  - Target Python version: 3.9+
- **isort**: Configuration in `pyproject.toml`
  - Profile: black (for compatibility)
  - Line length: 88 characters
- **mypy**: Configuration in `pyproject.toml` and `mypy.ini`
  - Python version: 3.9
  - Strict mode with multiple validations enabled
  - Special exemptions for test files
- **pytest**: Configuration in `pyproject.toml` and `pytest.ini`
  - Includes coverage reporting
- **bandit**: Configuration in `pyproject.toml`
  - Excludes test directories
- **pylint**: Basic configuration in `pyproject.toml`
  - Disabled rules: C0111, R0903, C0103
  - Max line length: 88 characters

## Code Style Guidelines
- **Code Quality Tools**: ALWAYS run all quality checks before committing code:
  - **flake8**: For linting (PEP8 compliance, docstring validation)
  - **black**: For consistent code formatting (88 character line length)
  - **isort**: For import sorting (standard library → third-party → local)
  - **mypy**: For static type checking (use strict typing)
  - **bandit**: For security vulnerability scanning
  - **pre-commit**: Run all checks automatically before commit
- **Imports**:
  - Standard library first, third-party next, local modules last
  - Always use isort profile=black configuration
  - Avoid wildcard imports (`from module import *`)
- **Typing**:
  - Use type hints for ALL function parameters and return values
  - Follow mypy strict typing guidelines
  - Use Optional[] for parameters that can be None
  - Use Union[] for parameters that can be multiple types
- **Naming**:
  - snake_case for variables/functions/modules
  - CamelCase for classes
  - UPPER_CASE for constants
  - Avoid single letter variable names except in list comprehensions
- **Documentation**:
  - Docstrings for all functions, classes, and modules (follow Google docstring format)
  - First line must end with a period (D400 rule)
  - Include type hints in docstrings that match the function signature
  - Document all parameters, return values, and exceptions raised
- **Error Handling**:
  - Use try/except blocks with specific exceptions
  - Never use bare except: clauses
  - Include helpful error messages that guide the user
- **Security**:
  - Never hardcode secrets, use environment variables via python-dotenv
  - Run bandit scans regularly to detect security issues
  - Validate all user inputs before processing
- **Testing**:
  - Write tests for all new functionality
  - Use pytest fixtures when appropriate
  - Maintain high code coverage (aim for >90%)
  - Include both unit and integration tests
- **SQLAlchemy**:
  - Use ORM models with proper type annotations
  - Define proper relationship definitions
  - Follow session management best practices
- **Streamlit**:
  - Use st.cache_data or st.cache_resource for expensive operations
  - Organize pages with proper hierarchy
- **API Clients**:
  - Use proper error handling and rate limiting for external API calls
  - Implement timeout handling for all network requests
- **Code Reviews**:
  - Always run linting and type checking before submitting for review
  - Address all flake8, mypy, and black issues before requesting review

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
- **Dependency Scanning**: Regularly scan and update dependencies for security vulnerabilities
- **Secrets Management**: Never log, print, or expose secrets in debug output or error messages
- **Access Control**: Implement proper role-based access controls for all API endpoints

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

## AI Integration Guidelines
- **API Usage**: Always use environment variables for API keys to AI services
- **Rate Limiting**: Implement proper rate limiting for AI API calls
- **Error Handling**: Gracefully handle AI service unavailability
- **Prompt Management**: Store and version AI prompts in dedicated files
- **Content Filtering**: Implement appropriate content filtering for AI-generated outputs
- **User Feedback**: Provide mechanisms for users to report inappropriate AI responses
- **Transparency**: Clearly indicate to users when content is AI-generated

Remember to keep database interactions secure and properly handle encryption/decryption for sensitive data.
