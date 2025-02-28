#!/bin/bash
set -e

# Activate virtual environment
if [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
else
  echo "Virtual environment not found. Run setup.sh first."
  exit 1
fi

# Run black formatter
black src app.py
