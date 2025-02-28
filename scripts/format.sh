#!/bin/bash
set -e

# Activate virtual environment
if [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
else
  echo "Virtual environment not found. Run setup.sh first."
  exit 1
fi

# Run formatters
echo "Running black formatter..."
black src app.py
echo "Running isort import sorter..."
isort src app.py
