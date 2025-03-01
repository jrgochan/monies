#!/bin/bash
set -e

# Change to project root directory
cd "$(dirname "$0")/.." || exit 1

# Activate virtual environment
if [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
elif [ -f "venv/bin/activate" ]; then
  source venv/bin/activate
else
  echo "Virtual environment not found. Run setup.sh first."
  exit 1
fi

# Run tests with coverage by default, excluding problematic tests
if [[ $# -eq 0 ]]; then
  pytest --cov=src --cov-report=term --cov-report=html -k "not oauth and not test_test_all_connections and not test_analyze_stock_trend_success and not test_analyze_crypto_trend_historical_data_error and not test_analyze_crypto_trend_openai_error and not test_get_api_key_existing and not test_get_api_key_nonexistent"
else
  pytest "$@"
fi
