#!/usr/bin/env python3
"""
Script to pre-cache market data from various sources.
This script fetches historical data for specified symbols and time periods
and stores it in the database for faster access.

Usage:
    python cache_market_data.py [--symbols SYMBOLS] [--periods PERIODS] [--cache-hours HOURS]

Options:
    --symbols SYMBOLS      Comma-separated list of symbols to fetch (default: common ETFs and stocks)
    --periods PERIODS      Comma-separated list of time periods to fetch (default: 1mo,3mo,6mo,1y,2y,3y,5y)
    --cache-hours HOURS    How long to cache the data in hours (default: 24)
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv
from sqlalchemy.orm import Session

# Add the parent directory to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.database import MarketData, SessionLocal
from src.utils.data_cache import MarketDataCache

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('market_data_cache.log')
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# API keys from environment variables
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY", "")
FMP_API_KEY = os.getenv("FMP_API_KEY", "")
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "")

# Default symbols and time periods
DEFAULT_SYMBOLS = [
    # Major ETFs
    "SPY", "QQQ", "IWM", "DIA", "VTI", "EFA", "EEM", "AGG", "LQD", "TLT", "GLD", "SLV", "VNQ", "XLE", "XLF", "XLK", "XLV",
    # Major stocks
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "BRK-B", "JNJ", "V", "PG", "JPM", "UNH", "HD", "BAC", "MA", "DIS",
    # Leveraged ETFs
    "SPXL", "SPXS", "TQQQ", "SQQQ", "TNA", "TZA", "UDOW", "SDOW", "SSO", "SDS", "QLD", "QID", "UWM", "TWM",
    # Inverse ETFs
    "SH", "PSQ", "RWM",
    # Crypto ETFs
    "GBTC", "ETHE"
]

DEFAULT_PERIODS = ["1mo", "3mo", "6mo", "1y", "2y", "3y", "5y"]


def fetch_and_cache_from_yahoo(
    db: Session, 
    symbol: str, 
    time_period: str,
    cache_duration_hours: int
) -> bool:
    """
    Fetch data from Yahoo Finance and cache it.
    """
    try:
        logger.info(f"Fetching data for {symbol} from Yahoo Finance ({time_period})")
        df = yf.download(
            symbol, period=time_period, progress=False, interval="1d"
        )
        
        if df.empty or len(df) < 5:
            logger.warning(f"Not enough data returned from Yahoo Finance for {symbol}")
            return False
        
        # Create a serializable DataFrame
        serializable_df = df.reset_index()
        serializable_df["Date"] = serializable_df["Date"].dt.strftime("%Y-%m-%d")
        serializable_df = serializable_df.set_index("Date")
            
        # Cache the data if it was successfully fetched
        success = MarketDataCache.cache_data(
            db=db,
            symbol=symbol,
            data_type="price_history",
            time_period=time_period,
            data=serializable_df,
            data_source="yahoo_finance",
            cache_duration_hours=cache_duration_hours,
        )
        
        if success:
            logger.info(f"Successfully cached Yahoo Finance data for {symbol} ({time_period})")
            return True
        else:
            logger.warning(f"Failed to cache Yahoo Finance data for {symbol} ({time_period})")
            return False
            
    except Exception as e:
        logger.error(f"Error fetching {symbol} from Yahoo Finance: {str(e)}")
        return False


def fetch_and_cache_from_alpha_vantage(
    db: Session, 
    symbol: str, 
    time_period: str,
    cache_duration_hours: int
) -> bool:
    """
    Fetch data from Alpha Vantage API and cache it.
    """
    if not ALPHA_VANTAGE_KEY:
        logger.warning("Alpha Vantage API key not found in environment variables")
        return False
        
    try:
        logger.info(f"Fetching data for {symbol} from Alpha Vantage ({time_period})")
        
        # Convert time period to Alpha Vantage format
        outputsize = 'full'  # Default to full (up to 20 years of data)
        if time_period in ['1d', '5d', '1mo']:
            outputsize = 'compact'  # last 100 data points
            
        # Build Alpha Vantage API request
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol}&outputsize={outputsize}&apikey={ALPHA_VANTAGE_KEY}"
        response = requests.get(url)
        
        if response.status_code != 200:
            logger.warning(f"Alpha Vantage API returned status code {response.status_code}")
            return False
        
        data = response.json()
        
        # Check for error messages
        if "Error Message" in data:
            logger.warning(f"Alpha Vantage API error: {data['Error Message']}")
            return False
            
        if "Time Series (Daily)" not in data:
            logger.warning("No time series data found in Alpha Vantage response")
            return False
        
        # Convert to DataFrame
        time_series = data["Time Series (Daily)"]
        alpha_df = pd.DataFrame(time_series).T
        alpha_df.index = pd.to_datetime(alpha_df.index)
        alpha_df = alpha_df.sort_index()
        
        # Rename columns to match yfinance format
        alpha_df = alpha_df.rename(
            columns={
                "1. open": "Open",
                "2. high": "High",
                "3. low": "Low",
                "4. close": "Close",
                "5. adjusted close": "Adj Close",
                "6. volume": "Volume",
            }
        )
        
        # Convert columns to numeric
        for col in alpha_df.columns:
            alpha_df[col] = pd.to_numeric(alpha_df[col])
        
        # Filter based on period if needed
        days = convert_period_to_days(time_period)
        start_date = datetime.now() - timedelta(days=days)
        alpha_df = alpha_df[alpha_df.index >= start_date]
        
        if alpha_df.empty or len(alpha_df) < 5:
            logger.warning("Not enough data returned from Alpha Vantage after filtering")
            return False
        
        # Create a serializable DataFrame
        serializable_df = alpha_df.reset_index()
        # Convert datetime index to strings for serialization
        serializable_df["index"] = serializable_df["index"].dt.strftime("%Y-%m-%d")
        serializable_df = serializable_df.set_index("index")
        
        # Cache the data
        success = MarketDataCache.cache_data(
            db=db,
            symbol=symbol,
            data_type="price_history",
            time_period=time_period,
            data=serializable_df,
            data_source="alpha_vantage",
            cache_duration_hours=cache_duration_hours
        )
        
        if success:
            logger.info(f"Successfully cached Alpha Vantage data for {symbol} ({time_period})")
            return True
        else:
            logger.warning(f"Failed to cache Alpha Vantage data for {symbol} ({time_period})")
            return False
            
    except Exception as e:
        logger.error(f"Error fetching {symbol} from Alpha Vantage: {str(e)}")
        return False


def fetch_and_cache_from_fmp(
    db: Session, 
    symbol: str, 
    time_period: str,
    cache_duration_hours: int
) -> bool:
    """
    Fetch data from Financial Modeling Prep API and cache it.
    """
    if not FMP_API_KEY:
        logger.warning("Financial Modeling Prep API key not found in environment variables")
        return False
        
    try:
        logger.info(f"Fetching data for {symbol} from FMP ({time_period})")
        
        # Convert time period to days for FMP
        days = convert_period_to_days(time_period)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Format dates for FMP API
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")
        
        # FMP API endpoint
        url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}?from={start_str}&to={end_str}&apikey={FMP_API_KEY}"
        
        response = requests.get(url)
        if response.status_code != 200:
            logger.warning(f"FMP API returned status code {response.status_code}")
            return False
            
        data = response.json()
        
        if "historical" not in data:
            logger.warning("No historical data found in FMP response")
            return False
            
        # Convert to DataFrame
        fmp_df = pd.DataFrame(data["historical"])
        
        # Format dates and set index
        fmp_df["date"] = pd.to_datetime(fmp_df["date"])
        
        # Convert index to date strings (YYYY-MM-DD) for serialization 
        date_column = fmp_df["date"].copy()
        
        # Set the index and sort
        fmp_df = fmp_df.set_index("date")
        fmp_df = fmp_df.sort_index()
        
        # Rename columns to match yfinance format
        fmp_df = fmp_df.rename(
            columns={
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume"
            }
        )
        
        if fmp_df.empty or len(fmp_df) < 5:
            logger.warning("Not enough data returned from FMP API")
            return False
            
        # Create a new DataFrame with string dates to avoid serialization issues
        serializable_df = fmp_df.reset_index()
        serializable_df["date"] = serializable_df["date"].dt.strftime("%Y-%m-%d")
        serializable_df = serializable_df.set_index("date")
        
        # Cache the data
        success = MarketDataCache.cache_data(
            db=db,
            symbol=symbol,
            data_type="price_history",
            time_period=time_period,
            data=serializable_df,
            data_source="fmp",
            cache_duration_hours=cache_duration_hours
        )
        
        if success:
            logger.info(f"Successfully cached FMP data for {symbol} ({time_period})")
            return True
        else:
            logger.warning(f"Failed to cache FMP data for {symbol} ({time_period})")
            return False
            
    except Exception as e:
        logger.error(f"Error fetching {symbol} from FMP: {str(e)}")
        return False


def convert_period_to_days(period: str) -> int:
    """Convert lookback period string to days"""
    if period.endswith("y"):
        return int(period[:-1]) * 365
    elif period.endswith("mo"):
        return int(period[:-2]) * 30
    elif period.endswith("d"):
        return int(period[:-1])
    elif period == "max":
        return 1825  # Cap at 5 years
    else:
        # Default to 1 year if unknown format
        return 365


def cache_symbol_data(
    db: Session,
    symbol: str,
    time_period: str,
    cache_duration_hours: int
) -> bool:
    """
    Attempt to cache data for a symbol from multiple sources in priority order.
    """
    logger.info(f"Caching data for {symbol} ({time_period})")
    
    # Check if we already have recent enough data
    existing_data = MarketDataCache.get_cached_price_history(
        db=db, symbol=symbol, time_period=time_period
    )
    
    if existing_data is not None and not existing_data.empty:
        logger.info(f"Already have cached data for {symbol} ({time_period})")
        return True
    
    # Try sources in priority order
    # 1. Yahoo Finance (most reliable for US stocks and ETFs)
    if fetch_and_cache_from_yahoo(db, symbol, time_period, cache_duration_hours):
        return True
        
    # Sleep to avoid rate limits
    time.sleep(1)
    
    # 2. Alpha Vantage (good backup source)
    if fetch_and_cache_from_alpha_vantage(db, symbol, time_period, cache_duration_hours):
        return True
        
    # Sleep to avoid rate limits
    time.sleep(1)
    
    # 3. Financial Modeling Prep (another backup)
    if fetch_and_cache_from_fmp(db, symbol, time_period, cache_duration_hours):
        return True
    
    logger.error(f"Failed to cache data for {symbol} ({time_period}) from any source")
    return False


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description='Cache market data from various sources')
    parser.add_argument('--symbols', type=str, help='Comma-separated list of symbols to fetch')
    parser.add_argument('--periods', type=str, help='Comma-separated list of time periods to fetch')
    parser.add_argument('--cache-hours', type=int, default=24, help='How long to cache the data in hours')
    
    args = parser.parse_args()
    
    # Parse symbols and periods from arguments or use defaults
    symbols = args.symbols.split(',') if args.symbols else DEFAULT_SYMBOLS
    periods = args.periods.split(',') if args.periods else DEFAULT_PERIODS
    cache_duration_hours = args.cache_hours
    
    logger.info(f"Starting to cache data for {len(symbols)} symbols and {len(periods)} time periods")
    logger.info(f"Cache duration: {cache_duration_hours} hours")
    
    # Create database session
    db = SessionLocal()
    
    try:
        # Clear expired cache entries first
        cleared_count = MarketDataCache.clear_expired_cache(db)
        logger.info(f"Cleared {cleared_count} expired cache entries")
        
        # Use a counter to track progress
        total_requests = len(symbols) * len(periods)
        success_count = 0
        failure_count = 0
        
        # Process each symbol and period
        for i, symbol in enumerate(symbols, 1):
            for period in periods:
                logger.info(f"Processing {symbol} ({period}) - {i}/{len(symbols)}")
                
                success = cache_symbol_data(
                    db=db,
                    symbol=symbol,
                    time_period=period,
                    cache_duration_hours=cache_duration_hours
                )
                
                if success:
                    success_count += 1
                else:
                    failure_count += 1
                
                # Sleep between requests to avoid rate limits
                time.sleep(1)
                
                # Print progress
                completed = success_count + failure_count
                logger.info(f"Progress: {completed}/{total_requests} ({completed/total_requests*100:.1f}%)")
                
        logger.info(f"Completed caching data for {len(symbols)} symbols and {len(periods)} time periods")
        logger.info(f"Success: {success_count}, Failure: {failure_count}")
        
    except Exception as e:
        logger.error(f"Error in main caching function: {str(e)}")
    finally:
        db.close()
        logger.info("Database session closed")


if __name__ == "__main__":
    main()