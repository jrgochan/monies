#!/bin/bash
set -e

# Activate virtual environment
if [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
else
  echo "Virtual environment not found. Run setup.sh first."
  exit 1
fi

# Run linters
echo "Running flake8..."
flake8 src app.py
echo "Running mypy..."
mypy src app.py
echo "Running black (check only)..."
black --check src app.py
echo "Running isort (check only)..."
isort --check src app.py
echo "Running bandit (security check)..."
bandit -r src app.py
echo "Running pylint..."
pylint src app.py || true  # Don't fail on pylint warnings
