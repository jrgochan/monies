"""Utility for caching market data to reduce API calls and improve performance."""

import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Union

import pandas as pd
from sqlalchemy.orm import Session

from src.models.database import MarketData

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MarketDataCache:
    """A utility class for caching market data to reduce API calls and improve performance.

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
                # Check the format of the cached data
                if "has_datetime_index" in data and data["has_datetime_index"]:
                    # Handle the updated datetime-preserving format
                    if "index_column" in data:
                        # Use the stored index column
                        df = pd.DataFrame(data["prices"])
                        index_col = data["index_column"]
                        if index_col in df.columns:
                            # Convert the index column back to datetime if it's a date string
                            try:
                                df[index_col] = pd.to_datetime(df[index_col])
                                df.set_index(index_col, inplace=True)
                                return df
                            except Exception as e:
                                # If conversion fails, still use the column as index
                                logger.warning(
                                    f"Could not convert index column to datetime: {str(e)}"
                                )
                                df.set_index(index_col, inplace=True)
                                return df

                    # Backward compatibility with old formats
                    elif "index_values" in data:
                        # Reconstruct with timestamp values (old format)
                        try:
                            df = pd.DataFrame(data["prices"])
                            # Convert int64 timestamps back to datetime index
                            datetime_index = pd.to_datetime(
                                data["index_values"], unit="ns"
                            )
                            df.index = datetime_index
                            return df
                        except Exception as e:
                            # Fall back to default handling if conversion fails
                            logger.warning(
                                f"Failed to convert old index_values format: {str(e)}"
                            )
                            # Continue to other methods
                    elif "date_column" in data:
                        # Use the stored date column (old format)
                        df = pd.DataFrame(data["prices"])
                        date_col = data["date_column"]
                        if date_col in df.columns:
                            try:
                                df[date_col] = pd.to_datetime(df[date_col])
                                df.set_index(date_col, inplace=True)
                                return df
                            except Exception as e:
                                # If conversion fails, still use the column as index
                                logger.warning(
                                    f"Could not convert date column to datetime: {str(e)}"
                                )
                                df.set_index(date_col, inplace=True)
                                return df

                # Fall back to the old format
                df = pd.DataFrame(data["prices"])

                # Convert date strings to datetime
                date_columns = ["date", "Date"]
                for col in date_columns:
                    if col in df.columns:
                        try:
                            df[col] = pd.to_datetime(df[col], errors="coerce")
                            # Check if we have valid dates after conversion
                            if df[col].notna().any():
                                # Skip if invalid dates detected
                                valid_dates = df[col][df[col].notna()]
                                if not valid_dates.empty:
                                    min_year = valid_dates.dt.year.min()
                                    if min_year <= 1970:
                                        logger.warning(
                                            f"Invalid dates detected in cached data (epoch)"
                                        )
                                        return None
                                df.set_index(col, inplace=True)
                                return df
                        except Exception as e:
                            logger.warning(
                                f"Error converting {col} to datetime: {str(e)}"
                            )

                # Try the index column as a fallback
                if "index" in df.columns:
                    try:
                        df["index"] = pd.to_datetime(df["index"], errors="coerce")
                        if df["index"].notna().any():
                            valid_dates = df["index"][df["index"].notna()]
                            if not valid_dates.empty:
                                min_year = valid_dates.dt.year.min()
                                if min_year <= 1970:
                                    logger.warning(
                                        f"Invalid dates detected in cached data (epoch)"
                                    )
                                    return None
                            df.set_index("index", inplace=True)
                            return df
                    except Exception as e:
                        logger.warning(f"Error converting index to datetime: {str(e)}")

                # If we got here, no valid date column was found
                logger.warning("No valid date column found in cached data")
                return None

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
                        # Create a copy to avoid modifying the original
                        data_copy = data.copy()
                        # Handle DatetimeIndex by resetting index and converting to string format
                        # This avoids serialization issues with tuple keys
                        data_reset = data_copy.reset_index()

                        # Convert all datetime columns to strings
                        for col in data_reset.columns:
                            if pd.api.types.is_datetime64_any_dtype(data_reset[col]):
                                data_reset[col] = data_reset[col].dt.strftime(
                                    "%Y-%m-%d"
                                )

                        # Store with metadata to track the original index column
                        index_col_name = data_reset.columns[
                            0
                        ]  # The former index is now the first column
                        data_dict = {
                            "prices": data_reset.to_dict(orient="records"),
                            "index_column": index_col_name,
                            "has_datetime_index": True,
                        }
                    except Exception as e:
                        logger.error(
                            f"Error processing DataFrame with DatetimeIndex: {str(e)}"
                        )
                        # Fallback: convert to a format that preserves date strings
                        try:
                            data_reset = data.reset_index()
                            date_col = data_reset.columns[0]

                            # Convert any datetime columns to strings
                            for col in data_reset.columns:
                                if pd.api.types.is_datetime64_any_dtype(
                                    data_reset[col]
                                ):
                                    data_reset[col] = data_reset[col].dt.strftime(
                                        "%Y-%m-%d"
                                    )

                            data_dict = {
                                "prices": data_reset.to_dict(orient="records"),
                                "index_column": date_col,
                                "has_datetime_index": True,
                            }
                        except Exception as e2:
                            logger.error(
                                f"Fallback serialization also failed: {str(e2)}"
                            )
                            # Last resort: convert to a very simple format
                            # Convert to string format that will be serializable
                            data_reset = data.reset_index()

                            # Convert any datetime columns to strings
                            for col in data_reset.columns:
                                if pd.api.types.is_datetime64_any_dtype(
                                    data_reset[col]
                                ):
                                    data_reset[col] = data_reset[col].dt.strftime(
                                        "%Y-%m-%d"
                                    )

                            # Use records oriented format to avoid tuple keys
                            data_dict = {"prices": data_reset.to_dict(orient="records")}
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
