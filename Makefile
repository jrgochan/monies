.PHONY: help setup run test coverage lint format clean hooks check-all security shell db-init

# Default target
help:
	@echo "Makefile for Monies project"
	@echo "Usage:"
	@echo "  make setup     - Setup the development environment"
	@echo "  make run       - Run the application"
	@echo "  make test      - Run tests"
	@echo "  make coverage  - Run tests with coverage"
	@echo "  make lint      - Run linting checks"
	@echo "  make format    - Format code"
	@echo "  make clean     - Clean up cache files"
	@echo "  make hooks     - Install pre-commit hooks"
	@echo "  make check-all - Run pre-commit hooks on all files"
	@echo "  make security  - Check for security vulnerabilities"
	@echo "  make shell     - Start interactive Python shell with project context"
	@echo "  make db-init   - Run database initialization"

# Setup the development environment
setup:
	./scripts/setup.sh dev

# Run the application
run:
	./scripts/run.sh

# Run tests
test:
	./scripts/test.sh

# Run tests with coverage
coverage:
	pytest --cov=src --cov-report=term --cov-report=html

# Run linting checks
lint:
	./scripts/lint.sh

# Format code
format:
	./scripts/format.sh

# Clean up cache files
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".coverage" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	rm -f .coverage

# Install pre-commit hooks
hooks:
	pre-commit install

# Run pre-commit hooks on all files
check-all:
	pre-commit run --all-files

# Check for security vulnerabilities
security:
	bandit -r src

# Start interactive Python shell with project context
shell:
	python -i -c "import sys; sys.path.insert(0, '.')"

# Run database initialization
db-init:
	python init_db.py