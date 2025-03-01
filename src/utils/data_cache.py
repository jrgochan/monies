import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from sqlalchemy.orm import Session

from src.models.database import MarketData

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MarketDataCache:
    """
    A utility class for caching market data to reduce API calls and improve performance.
    This is used to cache stock/ETF/crypto data from external sources.
    """

    @staticmethod
    def get_cached_data(
        db: Session,
        symbol: str,
        data_type: str,
        time_period: str,
        data_source: str = "any",
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached data for a given symbol, type and time period if it exists and is not expired.

        Args:
            db: Database session
            symbol: Stock/ETF/Crypto symbol
            data_type: Type of data (e.g., 'price_history', 'fundamentals')
            time_period: Time period (e.g., '1d', '1mo', '1y')
            data_source: Specific data source, or "any" for any available source

        Returns:
            Cached data as a dictionary if found and valid, None otherwise
        """
        try:
            # Build the query
            query = db.query(MarketData).filter(
                MarketData.symbol == symbol,
                MarketData.data_type == data_type,
                MarketData.time_period == time_period,
            )

            # Filter by specific data source if provided
            if data_source != "any":
                query = query.filter(MarketData.data_source == data_source)

            # Order by last updated to get the most recent data
            query = query.order_by(MarketData.last_updated.desc())

            # Get the first result
            cache_entry = query.first()

            # Check if the entry exists and is not expired
            if cache_entry:
                # If there's an expiration date, check if it's expired
                if cache_entry.expires_at is not None:
                    if cache_entry.expires_at < datetime.utcnow():
                        logger.info(
                            f"Cached data for {symbol} ({data_type}, {time_period}) is expired"
                        )
                        return None

                # Parse and return the cached data
                try:
                    return json.loads(cache_entry.data)
                except json.JSONDecodeError:
                    logger.error(
                        f"Failed to parse cached data for {symbol} ({data_type}, {time_period})"
                    )
                    return None

            return None

        except Exception as e:
            logger.error(f"Error retrieving cached data: {str(e)}")
            return None

    @staticmethod
    def get_cached_price_history(
        db: Session, symbol: str, time_period: str, data_source: str = "any"
    ) -> Optional[pd.DataFrame]:
        """
        Retrieve cached price history data as a pandas DataFrame.

        Args:
            db: Database session
            symbol: Stock/ETF/Crypto symbol
            time_period: Time period (e.g., '1mo', '1y')
            data_source: Specific data source, or "any" for any available source

        Returns:
            Pandas DataFrame with price history if found and valid, None otherwise
        """
        data = MarketDataCache.get_cached_data(
            db=db,
            symbol=symbol,
            data_type="price_history",
            time_period=time_period,
            data_source=data_source,
        )

        if data:
            try:
                # Convert the data to a DataFrame
                df = pd.DataFrame(data["prices"])

                # Convert date strings to datetime
                if "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"])
                    df.set_index("date", inplace=True)

                return df
            except Exception as e:
                logger.error(f"Error converting cached data to DataFrame: {str(e)}")
                return None

        return None

    @staticmethod
    def cache_data(
        db: Session,
        symbol: str,
        data_type: str,
        time_period: str,
        data: Union[Dict[str, Any], pd.DataFrame],
        data_source: str,
        cache_duration_hours: int = 24,
    ) -> bool:
        """
        Cache market data to the database.

        Args:
            db: Database session
            symbol: Stock/ETF/Crypto symbol
            data_type: Type of data (e.g., 'price_history', 'fundamentals')
            time_period: Time period (e.g., '1d', '1mo', '1y')
            data: The data to cache (dictionary or DataFrame)
            data_source: The source of the data (e.g., 'yahoo_finance')
            cache_duration_hours: How long to cache the data in hours

        Returns:
            True if successfully cached, False otherwise
        """
        try:
            # Convert DataFrame to dict if needed
            if isinstance(data, pd.DataFrame):
                # Handle DataFrame with datetime index
                if isinstance(data.index, pd.DatetimeIndex):
                    try:
                        # Reset index to include the dates as a column
                        data_reset = data.reset_index()
                        # Get the name of the index column (usually 'date' or 'Date' or 'index')
                        date_col = data_reset.columns[0]
                        # Convert datetime objects to strings
                        data_reset[date_col] = data_reset[date_col].dt.strftime("%Y-%m-%d")
                        # Convert to dict
                        data_dict = {"prices": data_reset.to_dict(orient="records")}
                    except Exception as e:
                        logger.error(f"Error processing DataFrame with DatetimeIndex: {str(e)}")
                        # Fallback: convert to a simpler format with string dates
                        try:
                            simple_df = data.copy()
                            simple_df.index = simple_df.index.strftime("%Y-%m-%d")
                            data_dict = {"prices": simple_df.to_dict()}
                        except Exception as e2:
                            logger.error(f"Fallback serialization also failed: {str(e2)}")
                            # Last resort: convert to a very simple format
                            data_dict = {"prices": data.reset_index().to_dict()}
                else:
                    # Regular DataFrame without datetime index
                    data_dict = {"prices": data.to_dict(orient="records")}
            else:
                # Already a dict
                data_dict = data

            # Serialize to JSON
            json_data = json.dumps(data_dict)

            # Calculate expiration time
            expires_at = datetime.utcnow() + timedelta(hours=cache_duration_hours)

            # Check if an entry already exists
            existing_entry = (
                db.query(MarketData)
                .filter(
                    MarketData.symbol == symbol,
                    MarketData.data_type == data_type,
                    MarketData.time_period == time_period,
                    MarketData.data_source == data_source,
                )
                .first()
            )

            if existing_entry:
                # Update existing entry
                existing_entry.data = json_data
                existing_entry.last_updated = datetime.utcnow()
                existing_entry.expires_at = expires_at
            else:
                # Create new entry
                new_entry = MarketData(
                    symbol=symbol,
                    data_type=data_type,
                    time_period=time_period,
                    data_source=data_source,
                    data=json_data,
                    last_updated=datetime.utcnow(),
                    expires_at=expires_at,
                )
                db.add(new_entry)

            # Commit changes
            db.commit()
            logger.info(
                f"Successfully cached {data_type} data for {symbol} ({time_period})"
            )
            return True

        except Exception as e:
            logger.error(f"Error caching data: {str(e)}")
            db.rollback()
            return False

    @staticmethod
    def clear_expired_cache(db: Session) -> int:
        """
        Clear all expired cache entries from the database.

        Args:
            db: Database session

        Returns:
            Number of entries cleared
        """
        try:
            # Find all expired entries
            expired = (
                db.query(MarketData)
                .filter(MarketData.expires_at < datetime.utcnow())
                .all()
            )

            count = len(expired)

            # Delete the expired entries
            for entry in expired:
                db.delete(entry)

            # Commit changes
            db.commit()
            logger.info(f"Cleared {count} expired cache entries")
            return count

        except Exception as e:
            logger.error(f"Error clearing expired cache: {str(e)}")
            db.rollback()
            return 0
