#!/bin/bash
set -e

# This script is meant to be used in development containers (GitHub Codespaces, VS Code Remote Containers)
# to set up the development environment automatically

echo "Setting up development environment in container..."

# Install system dependencies
apt-get update && apt-get install -y \
    sqlite3 \
    git \
    curl \
    bc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set up Python environment
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Initialize the database
python init_db.py

# Install pre-commit hooks
pre-commit install

# Set up git config if not already set
if [ -z "$(git config --global user.email)" ]; then
    echo "Setting up git config..."
    git config --global user.email "dev@example.com"
    git config --global user.name "Dev User"
fi

echo "Development container setup complete!"