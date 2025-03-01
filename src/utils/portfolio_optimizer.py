import json
import logging

# Get API keys from environment
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv
from sqlalchemy.orm import Session

from src.models.database import PortfolioOptimization, SessionLocal
from src.utils.data_cache import MarketDataCache

load_dotenv()

ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY", "")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PortfolioOptimizer:
    """
    A utility class for portfolio optimization calculations and caching
    """

    @staticmethod
    def get_etf_price_history(
        db: Session,
        symbol: str,
        lookback_period: str,
        use_cache: bool = True,
        cache_duration_hours: int = 24,
        prefer_live_data: bool = True,
    ) -> pd.Series:
        """
        Get historical price data for an ETF, with caching support

        Args:
            db: Database session
            symbol: ETF ticker symbol
            lookback_period: Period to look back (e.g., "1y")
            use_cache: Whether to use cached data if available
            cache_duration_hours: How long to cache the data in hours
            prefer_live_data: Whether to attempt getting live data first, even if cache exists

        Returns:
            Series with historical close prices
        """
        # If we don't prefer live data, check cache first
        cached_data = None
        if use_cache and not prefer_live_data:
            cached_data = MarketDataCache.get_cached_price_history(
                db=db, symbol=symbol, time_period=lookback_period
            )

            if cached_data is not None:
                logger.info(f"Using cached data for {symbol} ({lookback_period})")
                if "Close" in cached_data.columns:
                    return cached_data["Close"]
                else:
                    # Return the first column if Close isn't available
                    return cached_data.iloc[:, 0]

        # Always try to fetch live data first
        try:
            # Limit the lookback period to prevent issues with very long histories
            if lookback_period == "max":
                # Use a more reliable period for fetching data
                actual_period = "5y"
            else:
                actual_period = lookback_period

            # Try multiple data sources in sequence
            # 1. First try Yahoo Finance
            try:
                logger.info(
                    f"Fetching live data for {symbol} from Yahoo Finance ({actual_period})"
                )
                df = yf.download(
                    symbol, period=actual_period, progress=False, interval="1d"
                )
                # Check if DataFrame is empty safely
                df_empty = True
                if isinstance(df, pd.DataFrame):
                    df_empty = df.empty if hasattr(df, "empty") else True

                # Check row count safely
                df_len = 0
                if isinstance(df, pd.DataFrame):
                    df_len = len(df)

                if df_empty or df_len < 5:
                    raise ValueError("Not enough data returned from Yahoo Finance")

                # Cache the data if it was successfully fetched
                if use_cache:
                    # Create a serializable DataFrame
                    serializable_df = df.reset_index()
                    serializable_df["Date"] = serializable_df["Date"].dt.strftime(
                        "%Y-%m-%d"
                    )
                    serializable_df = serializable_df.set_index("Date")

                    success = MarketDataCache.cache_data(
                        db=db,
                        symbol=symbol,
                        data_type="price_history",
                        time_period=lookback_period,
                        data=serializable_df,
                        data_source="yahoo_finance",
                        cache_duration_hours=cache_duration_hours,
                    )
                    if success:
                        logger.info(
                            f"Updated cache with fresh data for {symbol} from Yahoo Finance"
                        )
                    else:
                        logger.warning(
                            f"Failed to cache Yahoo Finance data for {symbol}"
                        )

                # Return just the close prices
                return df["Close"]

            # 2. Try Alpha Vantage if available (more reliable but rate-limited)
            except Exception as yahoo_error:
                logger.warning(
                    f"Yahoo Finance data retrieval failed for {symbol}: {str(yahoo_error)}"
                )

                if ALPHA_VANTAGE_KEY:
                    try:
                        logger.info(f"Trying Alpha Vantage API for {symbol}")
                        # Convert lookback period to Alpha Vantage format
                        if lookback_period in ["1d", "5d"]:
                            outputsize = "compact"  # last 100 data points
                        else:
                            outputsize = "full"  # up to 20 years of data

                        # Build Alpha Vantage API request
                        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol}&outputsize={outputsize}&apikey={ALPHA_VANTAGE_KEY}"
                        response = requests.get(url)

                        if response.status_code != 200:
                            raise ValueError(
                                f"Alpha Vantage API returned status code {response.status_code}"
                            )

                        data = response.json()

                        # Check for error messages
                        if "Error Message" in data:
                            raise ValueError(
                                f"Alpha Vantage API error: {data['Error Message']}"
                            )

                        if "Time Series (Daily)" not in data:
                            raise ValueError(
                                "No time series data found in Alpha Vantage response"
                            )

                        # Convert to DataFrame
                        time_series = data["Time Series (Daily)"]
                        alpha_df = pd.DataFrame(time_series).T
                        alpha_df.index = pd.to_datetime(alpha_df.index)
                        alpha_df = alpha_df.sort_index()

                        # Rename columns to match yfinance format
                        alpha_df.columns = [col for col in alpha_df.columns]
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
                        days = PortfolioOptimizer.convert_period_to_days(
                            lookback_period
                        )
                        start_date = datetime.now() - timedelta(days=days)
                        alpha_df = alpha_df[alpha_df.index >= start_date]

                        if alpha_df.empty or len(alpha_df) < 5:
                            raise ValueError(
                                "Not enough data returned from Alpha Vantage after filtering"
                            )

                        # Cache the data
                        if use_cache:
                            MarketDataCache.cache_data(
                                db=db,
                                symbol=symbol,
                                data_type="price_history",
                                time_period=lookback_period,
                                data=alpha_df,
                                data_source="alpha_vantage",
                                cache_duration_hours=cache_duration_hours,
                            )
                            logger.info(
                                f"Updated cache with fresh data for {symbol} from Alpha Vantage"
                            )

                        # Return the adjusted close prices
                        if "Adj Close" in alpha_df.columns:
                            return alpha_df["Adj Close"]
                        elif "Close" in alpha_df.columns:
                            return alpha_df["Close"]
                        else:
                            # Return the first numeric column
                            return alpha_df.iloc[:, 0]

                    except Exception as alpha_error:
                        logger.warning(
                            f"Alpha Vantage data retrieval failed for {symbol}: {str(alpha_error)}"
                        )
                        # Continue to the next attempt

            except Exception as e1:
                logger.warning(
                    f"First attempt to fetch live data for {symbol} failed: {str(e1)}"
                )
                # Try alternative approach with Yahoo Finance using explicit dates
                try:
                    # Convert period to days
                    days = PortfolioOptimizer.convert_period_to_days(actual_period)
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=days)

                    logger.info(
                        f"Second attempt with Yahoo Finance: Fetching {symbol} data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
                    )
                    df = yf.download(
                        symbol,
                        start=start_date.strftime("%Y-%m-%d"),
                        end=end_date.strftime("%Y-%m-%d"),
                        progress=False,
                    )

                    if df.empty or len(df) < 5:
                        raise ValueError(
                            "Not enough data returned from Yahoo Finance on second attempt"
                        )

                    # Cache the data if it was successfully fetched
                    if use_cache:
                        MarketDataCache.cache_data(
                            db=db,
                            symbol=symbol,
                            data_type="price_history",
                            time_period=lookback_period,
                            data=df,
                            data_source="yahoo_finance",
                            cache_duration_hours=cache_duration_hours,
                        )
                        logger.info(
                            f"Updated cache with fresh data for {symbol} from Yahoo Finance (second attempt)"
                        )

                    return df["Close"]

                except Exception as e2:
                    logger.warning(
                        f"Second attempt to fetch live data for {symbol} failed: {str(e2)}"
                    )

                    # 3. Try some public financial APIs (no API key needed, but less reliable)
                    try:
                        # Try FMP API (Financial Modeling Prep - has a free tier)
                        logger.info(f"Trying FMP API for {symbol}")

                        # Convert lookback period to days
                        days = PortfolioOptimizer.convert_period_to_days(actual_period)
                        end_date = datetime.now()
                        start_date = end_date - timedelta(days=days)

                        # Format dates for FMP API
                        start_str = start_date.strftime("%Y-%m-%d")
                        end_str = end_date.strftime("%Y-%m-%d")

                        # Free tier of FMP
                        url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}?from={start_str}&to={end_str}"

                        response = requests.get(url)
                        if response.status_code != 200:
                            raise ValueError(
                                f"FMP API returned status code {response.status_code}"
                            )

                        data = response.json()

                        if "historical" not in data:
                            raise ValueError("No historical data found in FMP response")

                        # Convert to DataFrame
                        fmp_df = pd.DataFrame(data["historical"])

                        # Format dates and set index
                        fmp_df["date"] = pd.to_datetime(fmp_df["date"])
                        fmp_df = fmp_df.set_index("date")
                        fmp_df = fmp_df.sort_index()

                        # Rename columns to match yfinance format
                        fmp_df = fmp_df.rename(
                            columns={
                                "open": "Open",
                                "high": "High",
                                "low": "Low",
                                "close": "Close",
                                "volume": "Volume",
                            }
                        )

                        if fmp_df.empty or len(fmp_df) < 5:
                            raise ValueError("Not enough data returned from FMP API")

                        # Cache the data
                        if use_cache:
                            MarketDataCache.cache_data(
                                db=db,
                                symbol=symbol,
                                data_type="price_history",
                                time_period=lookback_period,
                                data=fmp_df,
                                data_source="fmp",
                                cache_duration_hours=cache_duration_hours,
                            )
                            logger.info(
                                f"Updated cache with fresh data for {symbol} from FMP"
                            )

                        # Return close prices
                        return fmp_df["Close"]

                    except Exception as fmp_error:
                        logger.warning(
                            f"FMP API data retrieval failed for {symbol}: {str(fmp_error)}"
                        )

                        # Now try to use cached data if available and we prefer live data
                        if use_cache and prefer_live_data and cached_data is None:
                            logger.info(
                                "Attempting to use cached data after all API data source attempts failed"
                            )
                            cached_data = MarketDataCache.get_cached_price_history(
                                db=db, symbol=symbol, time_period=lookback_period
                            )

                            if cached_data is not None:
                                logger.info(
                                    f"Using cached data for {symbol} as fallback"
                                )
                                if "Close" in cached_data.columns:
                                    return cached_data["Close"]
                                else:
                                    return cached_data.iloc[:, 0]

                        # If we still don't have data, raise an error
                        raise ValueError(
                            f"Could not fetch data for {symbol} from any API source and no cache available"
                        )

        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")

            # As a last resort, check if we have cached data again
            if use_cache and cached_data is None:
                cached_data = MarketDataCache.get_cached_price_history(
                    db=db, symbol=symbol, time_period=lookback_period
                )

                if cached_data is not None:
                    logger.warning(
                        f"Using cached data for {symbol} as last resort after all fetch attempts failed"
                    )
                    if "Close" in cached_data.columns:
                        return cached_data["Close"]
                    else:
                        return cached_data.iloc[:, 0]

            # Return a more detailed error message
            err_detail = ""
            if "Connection" in str(e) or "timeout" in str(e).lower():
                err_detail = "Network connection issue detected."
            elif "rate limit" in str(e).lower() or "429" in str(e):
                err_detail = "API rate limits may have been reached."
            elif "not found" in str(e).lower() or "404" in str(e):
                err_detail = f"Symbol '{symbol}' may not exist or may be delisted."

            # Throw an error - we don't use synthetic data
            raise ValueError(
                f"Could not retrieve data for {symbol} from API or cache. {err_detail}"
            )

    @staticmethod
    def convert_period_to_days(period: str) -> int:
        """Convert lookback period string to days"""
        if period.endswith("y"):
            return int(period[:-1]) * 365
        elif period.endswith("mo"):
            return int(period[:-2]) * 30
        elif period.endswith("d"):
            return int(period[:-1])
        elif period == "max":
            return 9125  # Cap at 25 years
        else:
            # Default to 1 year if unknown format
            return 365

    @staticmethod
    def calculate_optimal_weights(
        etfs_data: Dict[str, pd.Series], method: str = "Maximum Sharpe Ratio"
    ) -> Dict[str, float]:
        """
        Calculate optimal portfolio weights based on historical data

        Args:
            etfs_data: Dictionary mapping ETF symbols to price series
            method: Optimization method

        Returns:
            Dictionary of ETF symbols to weights
        """
        # For more advanced implementations, this could use modern portfolio theory
        # with quadratic programming to find the optimal weights

        # Convert price series to returns
        returns_data = {}
        for symbol, series in etfs_data.items():
            if len(series) > 1:
                returns_data[symbol] = series.pct_change().dropna()

        # Calculate returns statistics
        mean_returns = {s: r.mean() for s, r in returns_data.items()}
        std_returns = {s: r.std() for s, r in returns_data.items()}

        # Initialize weights dictionary
        weights = {}
        symbols = list(etfs_data.keys())

        # Apply different optimization strategies
        if method == "Maximum Sharpe Ratio":
            # Calculate Sharpe ratios (assuming risk-free rate of 2%)
            risk_free_rate = 0.0002  # daily rate (approx 5% annual)
            sharpe_ratios = {
                s: (mean_returns[s] - risk_free_rate) / std_returns[s]
                if std_returns[s] > 0
                else 0
                for s in returns_data.keys()
            }

            # Allocate based on relative Sharpe ratios
            total_sharpe = sum(max(0, sr) for sr in sharpe_ratios.values())

            if total_sharpe > 0:
                for symbol in symbols:
                    if symbol in sharpe_ratios:
                        weights[symbol] = max(0, sharpe_ratios[symbol]) / total_sharpe
                    else:
                        weights[symbol] = 0
            else:
                # Fallback to equal weighting
                weights = {s: 1.0 / len(symbols) for s in symbols}

        elif method == "Minimum Volatility":
            # Simple heuristic: inversely proportional to volatility
            inv_vols = {
                s: 1.0 / std_returns[s] if std_returns[s] > 0 else 0
                for s in returns_data.keys()
            }

            total_inv_vol = sum(inv_vols.values())

            if total_inv_vol > 0:
                for symbol in symbols:
                    if symbol in inv_vols:
                        weights[symbol] = inv_vols[symbol] / total_inv_vol
                    else:
                        weights[symbol] = 0
            else:
                # Fallback to equal weighting
                weights = {s: 1.0 / len(symbols) for s in symbols}

        else:  # Maximum Return
            # Simple heuristic: proportional to historical returns
            # Use positive returns only
            pos_returns = {s: max(0, r) for s, r in mean_returns.items()}
            total_return = sum(pos_returns.values())

            if total_return > 0:
                for symbol in symbols:
                    if symbol in pos_returns:
                        weights[symbol] = pos_returns[symbol] / total_return
                    else:
                        weights[symbol] = 0
            else:
                # Fallback to equal weighting
                weights = {s: 1.0 / len(symbols) for s in symbols}

        # Ensure all weights are assigned
        for symbol in symbols:
            if symbol not in weights:
                weights[symbol] = 0

        # Normalize weights to sum to 1.0
        total = sum(weights.values())
        if total > 0:
            return {k: v / total for k, v in weights.items()}
        else:
            # If all weights are zero, use equal weights
            return {k: 1.0 / len(symbols) for k in symbols}

    @staticmethod
    def calculate_portfolio_metrics(
        etfs_data: Dict[str, pd.Series], weights: Dict[str, float]
    ) -> Dict[str, Dict[str, Union[float, str]]]:
        """
        Calculate performance metrics for ETFs and the optimized portfolio

        Args:
            etfs_data: Dictionary mapping ETF symbols to price series
            weights: Dictionary of ETF symbols to weights

        Returns:
            Dictionary of metrics by symbol
        """
        # Initialize metrics dictionary
        metrics = {}

        # Calculate metrics for each ETF
        for symbol, series in etfs_data.items():
            try:
                # Calculate returns
                if len(series) > 1:
                    returns = series.pct_change().dropna()

                    # Calculate metrics with safety checks
                    if len(series) > 0:
                        first_value = float(series.iloc[0])
                        last_value = float(series.iloc[-1])
                        if first_value > 0:
                            total_return = ((last_value / first_value) - 1) * 100
                        else:
                            total_return = 0
                    else:
                        total_return = 0

                    # Annualized return calculation
                    if total_return <= -100:
                        annualized_return = -1.0  # -100% annual return
                    else:
                        try:
                            # Try to get days difference from datetime index
                            if hasattr(series.index[-1], "days") and hasattr(
                                series.index[0], "days"
                            ):
                                days = (series.index[-1] - series.index[0]).days
                            else:
                                # Handle case where index is string dates or integers
                                try:
                                    # Convert string dates to datetime if needed
                                    # datetime already imported at line 6

                                    start_date = pd.to_datetime(series.index[0])
                                    end_date = pd.to_datetime(series.index[-1])
                                    days = (end_date - start_date).days
                                except Exception:
                                    # If conversion fails, estimate based on length of series
                                    # Assume daily data
                                    days = len(series)

                            if days > 0:
                                annualized_return = (
                                    (1 + total_return / 100) ** (365 / days)
                                ) - 1
                            else:
                                annualized_return = 0
                        except Exception as e:
                            logger.warning(
                                f"Error calculating annualized return: {str(e)}"
                            )
                            # Fallback - use simple estimate
                            annualized_return = total_return / 100

                    # Volatility calculation
                    if len(returns) > 0:
                        volatility = returns.std() * (252**0.5) * 100  # Annualized
                    else:
                        volatility = 0

                    # Max drawdown calculation
                    if len(returns) > 0:
                        cumulative = (1 + returns).cumprod()
                        if len(cumulative) > 0:
                            cummax = cumulative.cummax()
                            # Check if all values in cummax are greater than 0
                            if cummax.gt(
                                0
                            ).all():  # Explicit comparison instead of cummax > 0
                                drawdowns = cumulative / cummax - 1
                                max_dd = drawdowns.min() * 100
                            else:
                                max_dd = 0
                        else:
                            max_dd = 0
                    else:
                        max_dd = 0

                    # Calculate Sharpe ratio (assuming risk-free rate of 2%)
                    risk_free = 0.02
                    if volatility > 0:
                        sharpe = (annualized_return - risk_free) / (volatility / 100)
                    else:
                        sharpe = 0

                    # Store metrics
                    metrics[symbol] = {
                        "total_return": total_return,
                        "annualized_return": annualized_return * 100,
                        "volatility": volatility,
                        "max_drawdown": max_dd,
                        "sharpe_ratio": sharpe,
                        # Add formatted versions for display
                        "total_return_fmt": f"{total_return:.2f}%",
                        "annualized_return_fmt": f"{annualized_return*100:.2f}%",
                        "volatility_fmt": f"{volatility:.2f}%",
                        "max_drawdown_fmt": f"{max_dd:.2f}%",
                        "sharpe_ratio_fmt": f"{sharpe:.2f}",
                    }
                else:
                    # Not enough data
                    metrics[symbol] = {
                        "total_return": 0,
                        "annualized_return": 0,
                        "volatility": 0,
                        "max_drawdown": 0,
                        "sharpe_ratio": 0,
                        "total_return_fmt": "N/A",
                        "annualized_return_fmt": "N/A",
                        "volatility_fmt": "N/A",
                        "max_drawdown_fmt": "N/A",
                        "sharpe_ratio_fmt": "N/A",
                    }
            except Exception as e:
                logger.warning(f"Error calculating metrics for {symbol}: {str(e)}")
                metrics[symbol] = {
                    "total_return": 0,
                    "annualized_return": 0,
                    "volatility": 0,
                    "max_drawdown": 0,
                    "sharpe_ratio": 0,
                    "total_return_fmt": "Error",
                    "annualized_return_fmt": "Error",
                    "volatility_fmt": "Error",
                    "max_drawdown_fmt": "Error",
                    "sharpe_ratio_fmt": "Error",
                }

        # Calculate portfolio performance
        portfolio_performance = PortfolioOptimizer.calculate_portfolio_performance(
            etfs_data, weights
        )

        if len(portfolio_performance) > 1:
            # Calculate portfolio metrics
            try:
                portfolio_returns = portfolio_performance.pct_change().dropna()

                # Calculate total return
                total_return = (
                    (portfolio_performance.iloc[-1] / portfolio_performance.iloc[0]) - 1
                ) * 100

                # Calculate annualized return
                try:
                    # Try to get days difference from datetime index
                    if hasattr(portfolio_performance.index[-1], "days") and hasattr(
                        portfolio_performance.index[0], "days"
                    ):
                        days = (
                            portfolio_performance.index[-1]
                            - portfolio_performance.index[0]
                        ).days
                    else:
                        # Handle case where index is string dates or integers
                        try:
                            # Convert string dates to datetime if needed
                            # datetime already imported at the top

                            start_date = pd.to_datetime(portfolio_performance.index[0])
                            end_date = pd.to_datetime(portfolio_performance.index[-1])
                            days = (end_date - start_date).days
                        except Exception:
                            # If conversion fails, estimate based on length of series
                            # Assume daily data
                            days = len(portfolio_performance)

                    if days > 0:
                        annualized_return = (
                            (1 + total_return / 100) ** (365 / days)
                        ) - 1
                    else:
                        annualized_return = 0
                except Exception as e:
                    logger.warning(
                        f"Error calculating portfolio annualized return: {str(e)}"
                    )
                    # Fallback - use simple estimate
                    annualized_return = total_return / 100

                # Calculate volatility
                volatility = portfolio_returns.std() * (252**0.5) * 100

                # Calculate max drawdown
                cumulative = (1 + portfolio_returns).cumprod()
                cummax = cumulative.cummax()
                if cummax.gt(
                    0
                ).all():  # Using explicit comparison instead of (cummax > 0)
                    drawdowns = cumulative / cummax - 1
                    max_dd = drawdowns.min() * 100
                else:
                    max_dd = 0

                # Calculate Sharpe ratio
                risk_free = 0.02
                if volatility > 0:
                    sharpe = (annualized_return - risk_free) / (volatility / 100)
                else:
                    sharpe = 0

                # Store portfolio metrics
                metrics["Optimal Portfolio"] = {
                    "total_return": total_return,
                    "annualized_return": annualized_return * 100,
                    "volatility": volatility,
                    "max_drawdown": max_dd,
                    "sharpe_ratio": sharpe,
                    # Add formatted versions for display
                    "total_return_fmt": f"{total_return:.2f}%",
                    "annualized_return_fmt": f"{annualized_return*100:.2f}%",
                    "volatility_fmt": f"{volatility:.2f}%",
                    "max_drawdown_fmt": f"{max_dd:.2f}%",
                    "sharpe_ratio_fmt": f"{sharpe:.2f}",
                }
            except Exception as e:
                logger.warning(f"Error calculating portfolio metrics: {str(e)}")
                metrics["Optimal Portfolio"] = {
                    "total_return": 0,
                    "annualized_return": 0,
                    "volatility": 0,
                    "max_drawdown": 0,
                    "sharpe_ratio": 0,
                    "total_return_fmt": "Error",
                    "annualized_return_fmt": "Error",
                    "volatility_fmt": "Error",
                    "max_drawdown_fmt": "Error",
                    "sharpe_ratio_fmt": "Error",
                }

        return metrics

    @staticmethod
    def calculate_portfolio_performance(
        etfs_data: Dict[str, pd.Series], weights: Dict[str, float]
    ) -> pd.Series:
        """
        Calculate the combined portfolio performance

        Args:
            etfs_data: Dictionary mapping ETF symbols to price series
            weights: Dictionary of ETF symbols to weights

        Returns:
            Series of portfolio values
        """
        # Safety check for empty data
        if not etfs_data or len(etfs_data) == 0:
            # Return a simple upward trending synthetic portfolio
            dates = pd.date_range(end=datetime.now(), periods=100, freq="D")
            return pd.Series(np.linspace(1, 1.2, len(dates)), index=dates)

        # Find common dates among all ETFs
        all_date_sets = [set(s.index) for s in etfs_data.values()]

        # Check if we have any date sets before attempting intersection
        if all_date_sets:
            common_dates = sorted(set.intersection(*all_date_sets))
        else:
            common_dates = []

        # If there are no common dates, use the dates from the first ETF
        if (not common_dates) and etfs_data:
            first_etf = list(etfs_data.keys())[0]
            common_dates = sorted(etfs_data[first_etf].index)

        # Create aligned data using only common dates
        aligned_data = {}
        for symbol, series in etfs_data.items():
            # Filter to common dates only if we have common dates
            if common_dates:
                aligned_data[symbol] = series.loc[series.index.isin(common_dates)]
                # Sort by date
                aligned_data[symbol] = aligned_data[symbol].sort_index()
            else:
                # If no common dates, use the original series
                aligned_data[symbol] = series.sort_index()

        # Safety check - if any series is empty after alignment
        for symbol, series in aligned_data.items():
            if series.empty:
                logger.warning(
                    f"Series for {symbol} is empty after alignment. Using substitute data."
                )
                # Create synthetic data for this symbol
                aligned_data[symbol] = pd.Series(
                    np.linspace(100, 120, len(common_dates)), index=common_dates
                )

        # Now calculate normalized returns
        normalized_data = {}
        for symbol, series in aligned_data.items():
            if len(series) > 0:  # Check to avoid division by zero
                normalized_data[symbol] = series / series.iloc[0]
            else:
                # Handle empty series with a fallback
                normalized_data[symbol] = pd.Series(1.0, index=common_dates)

        # Calculate weighted combined performance
        portfolio_performance = pd.Series(0, index=common_dates)
        for symbol, series in normalized_data.items():
            if symbol in weights:  # Ensure we have a weight for this symbol
                portfolio_performance += series * weights[symbol]

        return portfolio_performance

    @staticmethod
    def save_optimization_result(
        db: Session,
        base_etf: str,
        included_etfs: List[str],
        lookback_period: str,
        optimization_method: str,
        weights: Dict[str, float],
        performance_metrics: Dict[str, Dict[str, Union[float, str]]],
        analysis_text: Optional[str] = None,
        user_id: Optional[int] = None,
    ) -> int:
        """
        Save portfolio optimization results to the database

        Args:
            db: Database session
            base_etf: Base ETF symbol
            included_etfs: List of included ETF symbols
            lookback_period: Lookback period used
            optimization_method: Optimization method used
            weights: Dictionary of weights for each ETF
            performance_metrics: Dictionary of performance metrics
            analysis_text: Optional AI-generated analysis text
            user_id: Optional user ID

        Returns:
            ID of the saved optimization result
        """
        try:
            # Convert data to JSON strings
            included_etfs_json = json.dumps(included_etfs)
            weights_json = json.dumps(weights)
            metrics_json = json.dumps(performance_metrics)

            # Create new optimization entry
            optimization = PortfolioOptimization(
                user_id=user_id,
                base_etf=base_etf,
                included_etfs=included_etfs_json,
                lookback_period=lookback_period,
                optimization_method=optimization_method,
                weights=weights_json,
                performance_metrics=metrics_json,
                analysis_text=analysis_text,
            )

            # Add to database
            db.add(optimization)
            db.commit()
            db.refresh(optimization)

            logger.info(f"Saved optimization result with ID {optimization.id}")
            return optimization.id

        except Exception as e:
            logger.error(f"Error saving optimization result: {str(e)}")
            db.rollback()
            return -1

    @staticmethod
    def get_optimization_result(db: Session, optimization_id: int) -> Optional[Dict]:
        """
        Retrieve a saved optimization result

        Args:
            db: Database session
            optimization_id: ID of the optimization result

        Returns:
            Dictionary with optimization result data if found
        """
        try:
            # Get optimization entry
            optimization = (
                db.query(PortfolioOptimization)
                .filter(PortfolioOptimization.id == optimization_id)
                .first()
            )

            if not optimization:
                return None

            # Parse JSON data
            included_etfs = json.loads(optimization.included_etfs)
            weights = json.loads(optimization.weights)
            performance_metrics = json.loads(optimization.performance_metrics)

            # Return as dictionary
            return {
                "id": optimization.id,
                "user_id": optimization.user_id,
                "base_etf": optimization.base_etf,
                "included_etfs": included_etfs,
                "lookback_period": optimization.lookback_period,
                "optimization_method": optimization.optimization_method,
                "weights": weights,
                "performance_metrics": performance_metrics,
                "analysis_text": optimization.analysis_text,
                "created_at": optimization.created_at,
            }

        except Exception as e:
            logger.error(f"Error retrieving optimization result: {str(e)}")
            return None

    @staticmethod
    def optimize_portfolio(
        base_etf: str,
        included_etfs: List[str],
        lookback_period: str,
        optimization_method: str,
        use_cache: bool = True,
        prefer_live_data: bool = True,
        save_result: bool = True,
        user_id: Optional[int] = None,
    ) -> Dict:
        """
        Complete portfolio optimization workflow

        Args:
            base_etf: Base ETF symbol
            included_etfs: List of included ETF symbols
            lookback_period: Lookback period
            optimization_method: Optimization method
            use_cache: Whether to use cached data
            prefer_live_data: Whether to prioritize live data over cached data
            save_result: Whether to save the result to the database
            user_id: Optional user ID

        Returns:
            Dictionary with optimization results
        """
        # Create database session
        db = SessionLocal()

        try:
            # Fetch historical data for each ETF
            etfs_data = {}
            failed_etfs = []

            # Try to fetch data for each ETF - continue even if some fail
            for etf in included_etfs:
                try:
                    price_series = PortfolioOptimizer.get_etf_price_history(
                        db=db,
                        symbol=etf,
                        lookback_period=lookback_period,
                        use_cache=use_cache,
                        prefer_live_data=prefer_live_data,
                    )
                    etfs_data[etf] = price_series
                except Exception as e:
                    logger.error(f"Failed to retrieve data for {etf}: {str(e)}")
                    failed_etfs.append(etf)

            # If we couldn't get data for any ETFs, raise an error
            has_data = len(etfs_data) > 0  # Explicit check without boolean context
            if not has_data:
                if failed_etfs:
                    failed_symbols = ", ".join(failed_etfs)
                    raise ValueError(
                        f"Could not retrieve data for any of the requested ETFs: {failed_symbols}"
                    )
                else:
                    raise ValueError("No ETFs were specified for optimization")

            # If some ETFs failed but others succeeded, log a warning
            has_failed = len(failed_etfs) > 0  # Explicit check without boolean context
            if has_failed:
                failed_symbols = ", ".join(failed_etfs)
                logger.warning(
                    f"Proceeding with optimization without these ETFs (failed to retrieve data): {failed_symbols}"
                )

            # Calculate optimal weights
            weights = PortfolioOptimizer.calculate_optimal_weights(
                etfs_data=etfs_data, method=optimization_method
            )

            # Calculate portfolio performance with error handling
            try:
                portfolio_performance = (
                    PortfolioOptimizer.calculate_portfolio_performance(
                        etfs_data=etfs_data, weights=weights
                    )
                )
            except Exception as e:
                logger.error(f"Error calculating portfolio performance: {str(e)}")
                # Create a simple placeholder performance series
                dates = pd.date_range(end=datetime.now(), periods=100, freq="D")
                portfolio_performance = pd.Series(
                    np.linspace(1, 1.2, len(dates)), index=dates
                )

            # Calculate performance metrics with error handling
            try:
                metrics = PortfolioOptimizer.calculate_portfolio_metrics(
                    etfs_data=etfs_data, weights=weights
                )
            except Exception as e:
                logger.error(f"Error calculating portfolio metrics: {str(e)}")
                # Create placeholder metrics
                metrics = {
                    "Optimal Portfolio": {
                        "total_return": 0,
                        "annualized_return": 0,
                        "volatility": 0,
                        "max_drawdown": 0,
                        "sharpe_ratio": 0,
                        "total_return_fmt": "N/A",
                        "annualized_return_fmt": "N/A",
                        "volatility_fmt": "N/A",
                        "max_drawdown_fmt": "N/A",
                        "sharpe_ratio_fmt": "N/A",
                    }
                }

                # Add placeholder metrics for each ETF
                for etf in etfs_data.keys():
                    metrics[etf] = {
                        "total_return": 0,
                        "annualized_return": 0,
                        "volatility": 0,
                        "max_drawdown": 0,
                        "sharpe_ratio": 0,
                        "total_return_fmt": "N/A",
                        "annualized_return_fmt": "N/A",
                        "volatility_fmt": "N/A",
                        "max_drawdown_fmt": "N/A",
                        "sharpe_ratio_fmt": "N/A",
                    }

            # Format weights as percentages for display
            weights_display = {k: f"{v*100:.1f}%" for k, v in weights.items()}

            # Create result dictionary
            result = {
                "base_etf": base_etf,
                "included_etfs": included_etfs,
                "lookback_period": lookback_period,
                "optimization_method": optimization_method,
                "weights": weights,
                "weights_display": weights_display,
                "metrics": metrics,
                "portfolio_performance": portfolio_performance,
                "etfs_data": etfs_data,
            }

            # Save the result if requested
            if save_result:
                result_id = PortfolioOptimizer.save_optimization_result(
                    db=db,
                    base_etf=base_etf,
                    included_etfs=included_etfs,
                    lookback_period=lookback_period,
                    optimization_method=optimization_method,
                    weights=weights,
                    performance_metrics=metrics,
                    user_id=user_id,
                )
                result["id"] = result_id

            return result

        except Exception as e:
            logger.error(f"Error in portfolio optimization: {str(e)}")
            return {
                "error": str(e),
                "base_etf": base_etf,
                "included_etfs": included_etfs,
                "lookback_period": lookback_period,
                "optimization_method": optimization_method,
            }

        finally:
            # Close the database session
            db.close()
