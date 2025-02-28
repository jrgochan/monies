# Justfile for Monies project
# To use: install 'just' command and run 'just <command>'
# See: https://github.com/casey/just

# Set the default shell to bash
set shell := ["bash", "-c"]

# Default recipe to run when just is called without arguments
default:
    @just --list

# Setup the development environment
setup:
    ./scripts/setup.sh dev

# Run the application
run:
    ./scripts/run.sh

# Run tests
test *args:
    ./scripts/test.sh {{args}}

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