#!/bin/bash
set -e

# Activate virtual environment
if [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
else
  echo "Virtual environment not found. Run setup.sh first."
  exit 1
fi

# Default test options
PYTEST_ARGS="--cov=src --cov-report=term --cov-report=html"

# If arguments are provided, use them instead
if [ $# -gt 0 ]; then
  # Run tests with provided arguments
  pytest "$@"
else
  # Run tests with default arguments
  echo "Running tests with coverage..."
  pytest $PYTEST_ARGS
  
  # Run with parallelization if no specific args were provided
  if command -v pytest-xdist >/dev/null 2>&1 || pip show pytest-xdist >/dev/null 2>&1; then
    echo "Running tests in parallel mode..."
    pytest -xvs $PYTEST_ARGS -n auto
  fi
fi
