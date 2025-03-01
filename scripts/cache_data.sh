#!/bin/bash
# Script to cache market data for all configured symbols
# This helps prepare the database with historical data to speed up the application

# Navigate to the project root directory
cd "$(dirname "$0")/.." || exit

# Set up Python environment (use virtual environment if it exists)
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
fi

echo "Starting market data caching process..."

# Run the cache_market_data.py script
python scripts/cache_market_data.py "$@"

# Check if the script executed successfully
if [ $? -eq 0 ]; then
    echo "Market data caching completed successfully."
else
    echo "Error occurred during market data caching."
    exit 1
fi

echo "Done!"