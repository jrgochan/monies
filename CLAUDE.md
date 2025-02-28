# Monies Project Guidelines

## Build/Test/Lint Commands
- Run application: `./scripts/run.sh` or `streamlit run app.py`
- Run tests: `./scripts/test.sh` or `pytest`
- Run specific test: `pytest tests/path/to/test_file.py::test_function_name`
- Run with coverage: `pytest --cov=src --cov-report=term --cov-report=html`
- Lint code: `./scripts/lint.sh` or `flake8 src/ tests/`
- Type checking: `mypy src/ tests/`
- Format code: `./scripts/format.sh` or `black src/ tests/ && isort src/ tests/`
- Install dev dependencies: `pip install -r requirements-dev.txt`

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
- **Data Sources**: NEVER use simulated data; show error if cannot connect to actual data source

Remember to keep database interactions secure and properly handle encryption/decryption for sensitive data.