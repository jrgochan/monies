import logging
import os
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union, cast

import openai
import pandas as pd
import requests
import yfinance as yf
from openai import OpenAI

from src.api import API_KEYS, logger, make_api_request
from src.utils.api_config import APIConfigManager

# Initialize OpenAI client
client = None
try:
    client = OpenAI(api_key=API_KEYS["openai"])
except Exception as e:
    logger.warning(f"Could not initialize OpenAI client: {str(e)}")
    # Try without proxies parameter that might be causing the error
    try:
        openai.api_key = API_KEYS["openai"]
        client = openai.OpenAI()
    except Exception as e2:
        logger.warning(f"Second attempt to initialize OpenAI client failed: {str(e2)}")


class OllamaManager:
    """Manager for Ollama models and settings."""

    # Class variables for caching
    _base_url: Optional[str] = None
    _default_model: Optional[str] = None
    _available_models: Optional[List[str]] = None
    _model_cache: Dict[str, Any] = {}

    @classmethod
    def get_settings(cls) -> Tuple[str, str]:
        """
        Get Ollama settings from the API config manager

        Returns:
            Tuple of (base_url, default_model)
        """
        # Use cached values if available
        if cls._base_url is not None and cls._default_model is not None:
            return cls._base_url, cls._default_model

        # Get the base URL from the environment variable through API config
        base_url = APIConfigManager.get_api_value_from_env("OLLAMA_BASE_URL")
        # Use default if not configured
        if not base_url:
            base_url = "http://localhost:11434"

        # Ensure base URL ends with a slash
        if not base_url.endswith("/"):
            base_url += "/"

        # Get default model from environment variable or API config
        default_model = APIConfigManager.get_api_value_from_env("OLLAMA_MODEL")
        if not default_model:
            default_model = "llama2"

        # Cache the values
        cls._base_url = base_url
        cls._default_model = default_model

        return base_url, default_model

    @classmethod
    def get_available_models(cls, force_refresh: bool = False) -> List[str]:
        """
        Get available models from Ollama instance

        Args:
            force_refresh: Force refresh of model list instead of using cache

        Returns:
            List of available model names
        """
        # Use cached models if available and not forcing refresh
        if cls._available_models is not None and not force_refresh:
            return cls._available_models

        base_url, _ = cls.get_settings()

        try:
            success, data = make_api_request(
                url=f"{base_url}api/tags",
                timeout=10,
                error_msg="Error fetching Ollama models",
            )

            if success and isinstance(data, dict) and "models" in data:
                models = [model["name"] for model in data["models"]]
                # Cache the result
                cls._available_models = models
                return models
            else:
                logger.warning("No models found in Ollama response")
                return []
        except Exception as e:
            logger.error(f"Error fetching Ollama models: {str(e)}")
            return []

    @classmethod
    def select_best_model(cls, task_type: str = "general") -> str:
        """
        Select the best Ollama model for a given task type

        Args:
            task_type: Type of task ("finance", "general", "coding", etc.)

        Returns:
            Best model name for the task
        """
        # Check cache first
        if task_type in cls._model_cache:
            return cls._model_cache[task_type]

        # Get available models
        available_models = cls.get_available_models()
        _, default_model = cls.get_settings()

        if not available_models:
            return default_model

        # Model preferences by task type
        model_preferences = {
            "finance": [
                "mistral-medium",
                "mixtral",
                "llama3",
                "llama3:70b",
                "llama3:8b",
                "mistral",
                "codellama",
                "llama2:70b",
                "llama2",
            ],
            "general": [
                "llama3",
                "llama3:70b",
                "llama3:8b",
                "mistral",
                "mistral-medium",
                "mixtral",
                "llama2:70b",
                "llama2",
            ],
            "coding": [
                "codellama",
                "llama3",
                "llama3:70b",
                "mixtral",
                "mistral-medium",
                "mistral",
                "llama2:70b",
                "llama2",
            ],
        }

        # Use general preferences if task type not found
        preferences = model_preferences.get(task_type, model_preferences["general"])

        # Find the first available preferred model
        for model in preferences:
            for available_model in available_models:
                # Check for exact match or if the model name starts with the preference
                # (handles cases like llama3 matching llama3:8b, etc.)
                if available_model == model or available_model.startswith(f"{model}:"):
                    logger.info(
                        f"Selected Ollama model {available_model} for {task_type} task"
                    )
                    # Cache the result
                    cls._model_cache[task_type] = available_model
                    return available_model

        # If no preferred models are available, use the first available model
        if available_models:
            selected_model = available_models[0]
            logger.info(
                f"No preferred model available for {task_type}. Using {selected_model}"
            )
            cls._model_cache[task_type] = selected_model
            return selected_model

        # If no models available, return default
        return default_model


# Legacy function to maintain backward compatibility
def get_ollama_settings() -> Tuple[str, str]:
    """Legacy function that calls OllamaManager.get_settings()"""
    return OllamaManager.get_settings()


# Legacy function to maintain backward compatibility
def get_available_ollama_models() -> List[str]:
    """Legacy function that calls OllamaManager.get_available_models()"""
    return OllamaManager.get_available_models()


# Legacy function to maintain backward compatibility
def select_best_ollama_model(task_type: str = "general") -> str:
    """Legacy function that calls OllamaManager.select_best_model()"""
    return OllamaManager.select_best_model(task_type)


def analyze_with_openai(
    prompt: str, model: str = "gpt-3.5-turbo", task_type: str = "general"
) -> str:
    """
    Analyze data using OpenAI API.

    Args:
        prompt: The prompt to send to OpenAI
        model: The model to use
        task_type: Type of task for system prompt customization

    Returns:
        Analysis result as a string
    """
    try:
        if not client:
            raise ValueError("OpenAI client not initialized")

        # Customize system prompt based on task type
        system_prompt = "You are a helpful AI assistant."
        if task_type == "finance":
            system_prompt = (
                "You are a financial analyst providing insights on market data."
            )
        elif task_type == "coding":
            system_prompt = (
                "You are a programming expert helping with code"
                " analysis and development."
            )

        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            max_tokens=1000,
        )

        return completion.choices[0].message.content
    except Exception as e:
        logger.error(f"Error with OpenAI analysis: {str(e)}")
        raise


def analyze_with_ollama(
    prompt: str, model: Optional[str] = None, task_type: str = "general"
) -> str:
    """
    Analyze data using Ollama locally hosted model.

    Args:
        prompt: The prompt to send to Ollama
        model: The model to use (if None, selects best model for task)
        task_type: Type of task ("finance", "general", "coding", etc.)

    Returns:
        Analysis result as a string
    """
    base_url, default_model = OllamaManager.get_settings()

    # If no model specified, select the best model for the task
    if model is None:
        try:
            model = OllamaManager.select_best_model(task_type)
            logger.debug(f"Selected model based on task: {model}")
        except Exception as e:
            logger.warning(f"Failed to select best model: {str(e)}. Using default.")
            model = default_model

    # Customize system prompt based on task type
    system_prompt = ""
    if task_type == "finance":
        system_prompt = (
            "You are a financial analyst providing insights on market data. "
            "Be concise and factual."
        )
    elif task_type == "coding":
        system_prompt = (
            "You are a programming expert helping with code analysis "
            "and development."
        )

    # Create request body with appropriate parameters
    request_body = {
        "model": model,
        "prompt": prompt,
        "system": system_prompt,  # Some models support system parameter
        "stream": False,
    }

    try:
        # Make API call to Ollama using our common request function
        success, response_data = make_api_request(
            url=f"{base_url}api/generate",
            method="POST",
            json_data=request_body,
            timeout=60,  # Higher timeout for LLM inference
            error_msg=f"Ollama API error for model {model}",
        )

        if success and isinstance(response_data, dict) and "response" in response_data:
            return response_data["response"]
        elif not success:
            # Handle specific error cases
            error_msg = str(response_data)
            if "404" in error_msg:
                return (
                    f"Error: Model '{model}' not found or Ollama server not running. "
                    f"Please check your Ollama installation."
                )
            return f"Error: {error_msg}"
        else:
            logger.error(f"Unexpected Ollama API response format: {response_data}")
            return "Error: Unexpected API response format"

    except Exception as e:
        logger.error(f"Error with Ollama analysis: {str(e)}")
        return f"Error: {str(e)}"


def _filter_dataframe_by_period(df: pd.DataFrame, period: str) -> pd.DataFrame:
    """
    Filter a DataFrame based on the specified time period.

    Args:
        df: DataFrame with datetime index
        period: Time period (e.g., "30d", "6mo", "1y")

    Returns:
        Filtered DataFrame
    """
    # Filter based on period
    if period.endswith("d"):
        days = int(period[:-1])
        start_date = datetime.now() - timedelta(days=days)
        df = df[df.index >= start_date]
    elif period.endswith("mo"):
        months = int(period[:-2])
        start_date = datetime.now() - timedelta(days=months * 30)
        df = df[df.index >= start_date]
    elif period.endswith("y"):
        years = int(period[:-1])
        start_date = datetime.now() - timedelta(days=years * 365)
        df = df[df.index >= start_date]

    return df


def _calculate_period_start_date(period: str) -> str:
    """
    Calculate start date string based on period.

    Args:
        period: Time period (e.g., "30d", "6mo", "1y")

    Returns:
        Start date string in YYYY-MM-DD format
    """
    if period.endswith("d"):
        days = int(period[:-1])
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    elif period.endswith("mo"):
        months = int(period[:-2])
        start_date = (datetime.now() - timedelta(days=months * 30)).strftime("%Y-%m-%d")
    elif period.endswith("y"):
        years = int(period[:-1])
        start_date = (datetime.now() - timedelta(days=years * 365)).strftime("%Y-%m-%d")
    else:
        # Default to 6 months if period format is not recognized
        start_date = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")

    return start_date


def get_alpha_vantage_data(
    ticker: str, period: str = "6mo"
) -> Tuple[bool, pd.DataFrame, Dict[str, Any]]:
    """
    Get stock data from Alpha Vantage API.

    Args:
        ticker: Stock ticker symbol
        period: Time period for analysis

    Returns:
        Tuple of (success, DataFrame of historical data, company info)
    """
    if not API_KEYS["alpha_vantage"]:
        return False, pd.DataFrame(), {}

    # Get daily time series data
    url = f"https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": ticker,
        "outputsize": "full",
        "apikey": API_KEYS["alpha_vantage"],
    }

    success, data = make_api_request(
        url=url,
        params=params,
        error_msg=f"Error getting data from Alpha Vantage for {ticker}",
    )

    if not success or not isinstance(data, dict):
        return False, pd.DataFrame(), {}

    if "Error Message" in data:
        logger.error(f"Alpha Vantage API error: {data['Error Message']}")
        return False, pd.DataFrame(), {}

    if "Time Series (Daily)" not in data:
        logger.error("No time series data found in Alpha Vantage response")
        return False, pd.DataFrame(), {}

    try:
        # Convert to DataFrame
        time_series = data["Time Series (Daily)"]
        df = pd.DataFrame(time_series).T
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        # Rename columns to match yfinance format
        df.columns = ["1. open", "2. high", "3. low", "4. close", "5. volume"]
        df = df.rename(
            columns={
                "1. open": "Open",
                "2. high": "High",
                "3. low": "Low",
                "4. close": "Close",
                "5. volume": "Volume",
            }
        )

        # Convert columns to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col])

        # Filter based on period
        df = _filter_dataframe_by_period(df, period)

        # Get company overview
        overview_url = "https://www.alphavantage.co/query"
        overview_params = {
            "function": "OVERVIEW",
            "symbol": ticker,
            "apikey": API_KEYS["alpha_vantage"],
        }

        overview_success, company_info = make_api_request(
            url=overview_url,
            params=overview_params,
            error_msg=f"Error getting company overview from Alpha Vantage for {ticker}",
        )

        if not overview_success or not isinstance(company_info, dict):
            company_info = {}

        return True, df, company_info
    except Exception as e:
        logger.error(f"Error processing Alpha Vantage data: {str(e)}")
        return False, pd.DataFrame(), {}


def get_financial_modeling_prep_data(
    ticker: str, period: str = "6mo"
) -> Tuple[bool, pd.DataFrame, Dict[str, Any]]:
    """
    Get stock data from Financial Modeling Prep API.

    Args:
        ticker: Stock ticker symbol
        period: Time period for analysis

    Returns:
        Tuple of (success, DataFrame of historical data, company info)
    """
    if not API_KEYS["fmp"]:
        return False, pd.DataFrame(), {}

    try:
        # Calculate the start date based on period
        start_date = _calculate_period_start_date(period)

        # Get historical price data
        url = "https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}"
        params = {"from": start_date, "apikey": API_KEYS["fmp"]}

        success, data = make_api_request(
            url=url.format(ticker=ticker),
            params=params,
            error_msg=f"Error getting data from Financial Modeling Prep for {ticker}",
        )

        if not success or not isinstance(data, dict):
            return False, pd.DataFrame(), {}

        if "historical" not in data:
            logger.error("No historical data found in FMP response")
            return False, pd.DataFrame(), {}

        # Convert to DataFrame
        historical = data["historical"]
        df = pd.DataFrame(historical)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        df = df.sort_index()

        # Rename columns to match yfinance format
        df = df.rename(
            columns={
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            }
        )

        # Get company profile
        profile_url = "https://financialmodelingprep.com/api/v3/profile/{ticker}"
        profile_params = {"apikey": API_KEYS["fmp"]}

        profile_success, profile_data = make_api_request(
            url=profile_url.format(ticker=ticker),
            params=profile_params,
            error_msg=f"Error getting company profile from FMP for {ticker}",
        )

        company_info: Dict[str, Any] = {}
        if profile_success and isinstance(profile_data, list) and profile_data:
            company_info = profile_data[0]

        return True, df, company_info
    except Exception as e:
        logger.error(f"Error processing Financial Modeling Prep data: {str(e)}")
        return False, pd.DataFrame(), {}


def get_yahoo_finance_data(
    ticker: str, period: str = "6mo"
) -> Tuple[bool, pd.DataFrame, Dict[str, Any]]:
    """
    Get stock data from Yahoo Finance.

    Args:
        ticker: Stock ticker symbol
        period: Time period for analysis

    Returns:
        Tuple of (success, DataFrame of historical data, company info)
    """
    try:
        # Get historical data from Yahoo Finance with multiple attempts
        max_attempts = 3
        hist = None
        info = {"shortName": ticker}

        for attempt in range(max_attempts):
            try:
                # Try the download method
                hist = yf.download(ticker, period=period, progress=False)
                if not hist.empty:
                    # Get company info if possible
                    try:
                        stock = yf.Ticker(ticker)
                        info = stock.info
                    except Exception as e:
                        logger.debug(f"Non-critical info retrieval error: {str(e)}")
                    break

                # If we got an empty DataFrame, try using Ticker method
                stock = yf.Ticker(ticker)
                hist = stock.history(period=period)
                info = stock.info
                if not hist.empty:
                    break

                time.sleep(1)
            except Exception as e:
                logging.error(f"Yahoo Finance attempt {attempt+1} failed: {str(e)}")
                if attempt < max_attempts - 1:
                    time.sleep(2)

        if hist is None or hist.empty:
            return False, pd.DataFrame(), {}

        return True, hist, info
    except Exception as e:
        logger.error(f"Error getting data from Yahoo Finance: {str(e)}")
        return False, pd.DataFrame(), {}


def analyze_stock_trend(ticker: str, period: str = "6mo", user_id: int = None) -> Dict:
    """
    Analyze a stock's trend using historical data and AI.

    Args:
        ticker: Stock ticker symbol
        period: Time period for analysis
        user_id: Optional user ID to get data source preferences

    Returns:
        Dictionary with analysis results
    """
    # Access API keys
    alpha_vantage_key = API_KEYS["alpha_vantage"]
    fmp_api_key = API_KEYS["fmp"]

    result = {
        "ticker": ticker,
        "period": period,
        "data": {},
        "analysis": "",
        "success": False,
        "data_source": "",
        "data_sources": [],
    }

    # Get user preferences for data sources if user_id is provided
    user_preferences = None
    db = None
    data_sources = []

    try:
        if user_id:
            from src.models.database import SessionLocal
            from src.utils.data_aggregator import DataAggregator

            db = SessionLocal()
            # Get user's data source preferences
            from src.models.database import User

            user = db.query(User).filter(User.id == user_id).first()

            if user and user.data_source_preferences:
                try:
                    import json

                    preferences = json.loads(user.data_source_preferences)
                    if "stocks" in preferences:
                        user_preferences = preferences["stocks"]
                except Exception:
                    user_preferences = None

            # Get enabled data sources in priority order
            data_sources = DataAggregator.get_user_data_sources(db, user_id, "stocks")
            data_sources = [ds for ds in data_sources if ds["enabled"]]
    except Exception as e:
        logging.error(f"Error getting user preferences: {str(e)}")

    # List to collect results from all data sources
    all_results = []

    # If we have user-specific data sources, use those in priority order
    if data_sources:
        # Helper function to process data for each source
        def get_stock_data_from_source(source_name):
            source_result = {
                "ticker": ticker,
                "period": period,
                "data": {},
                "analysis": "",
                "success": False,
                "data_source": source_name,
            }

            if source_name == "Financial Modeling Prep" and fmp_api_key:
                success, hist, company_info = get_financial_modeling_prep_data(
                    ticker, period
                )
                if success and not hist.empty:
                    # Basic statistics
                    start_price = hist["Close"].iloc[0]
                    end_price = hist["Close"].iloc[-1]
                    percent_change = ((end_price - start_price) / start_price) * 100
                    high = hist["High"].max()
                    low = hist["Low"].min()

                    # Calculate some indicators
                    hist["MA50"] = (
                        hist["Close"].rolling(window=min(50, len(hist))).mean()
                    )
                    hist["MA200"] = (
                        hist["Close"].rolling(window=min(200, len(hist))).mean()
                    )

                    # Create data object with FMP specific mappings
                    source_result["data"] = {
                        "start_date": hist.index[0].strftime("%Y-%m-%d"),
                        "end_date": hist.index[-1].strftime("%Y-%m-%d"),
                        "start_price": f"{start_price:.2f}",
                        "end_price": f"{end_price:.2f}",
                        "percent_change": f"{percent_change:.2f}",
                        "high": f"{high:.2f}",
                        "low": f"{low:.2f}",
                        "volume_avg": int(hist["Volume"].mean())
                        if "Volume" in hist.columns
                        else 0,
                        "company_name": company_info.get("companyName", ticker),
                        "sector": company_info.get("sector", "Unknown"),
                        "industry": company_info.get("industry", "Unknown"),
                        "market_cap": company_info.get("mktCap", None),
                        "pe_ratio": company_info.get("pe", None),
                    }
                    source_result["success"] = True
                    return source_result

            elif source_name == "Alpha Vantage" and alpha_vantage_key:
                success, hist, company_info = get_alpha_vantage_data(ticker, period)
                if success and not hist.empty:
                    # Basic statistics
                    start_price = hist["Close"].iloc[0]
                    end_price = hist["Close"].iloc[-1]
                    percent_change = ((end_price - start_price) / start_price) * 100
                    high = hist["High"].max()
                    low = hist["Low"].min()

                    # Calculate some indicators
                    hist["MA50"] = (
                        hist["Close"].rolling(window=min(50, len(hist))).mean()
                    )
                    hist["MA200"] = (
                        hist["Close"].rolling(window=min(200, len(hist))).mean()
                    )

                    # Create data object with Alpha Vantage specific mappings
                    source_result["data"] = {
                        "start_date": hist.index[0].strftime("%Y-%m-%d"),
                        "end_date": hist.index[-1].strftime("%Y-%m-%d"),
                        "start_price": f"{start_price:.2f}",
                        "end_price": f"{end_price:.2f}",
                        "percent_change": f"{percent_change:.2f}",
                        "high": f"{high:.2f}",
                        "low": f"{low:.2f}",
                        "volume_avg": int(hist["Volume"].mean())
                        if "Volume" in hist.columns
                        else 0,
                        "company_name": company_info.get("Name", ticker),
                        "sector": company_info.get("Sector", "Unknown"),
                        "industry": company_info.get("Industry", "Unknown"),
                        "market_cap": company_info.get("MarketCapitalization", None),
                        "pe_ratio": company_info.get("PERatio", None),
                    }
                    source_result["success"] = True
                    return source_result

            elif source_name == "Yahoo Finance":
                success, hist, info = get_yahoo_finance_data(ticker, period)
                if success and not hist.empty:
                    # Basic statistics
                    start_price = hist["Close"].iloc[0]
                    end_price = hist["Close"].iloc[-1]
                    percent_change = ((end_price - start_price) / start_price) * 100
                    high = hist["High"].max()
                    low = hist["Low"].min()

                    # Calculate some indicators
                    hist["MA50"] = (
                        hist["Close"].rolling(window=min(50, len(hist))).mean()
                    )
                    hist["MA200"] = (
                        hist["Close"].rolling(window=min(200, len(hist))).mean()
                    )

                    # Create data object
                    source_result["data"] = {
                        "start_date": hist.index[0].strftime("%Y-%m-%d"),
                        "end_date": hist.index[-1].strftime("%Y-%m-%d"),
                        "start_price": f"{start_price:.2f}",
                        "end_price": f"{end_price:.2f}",
                        "percent_change": f"{percent_change:.2f}",
                        "high": f"{high:.2f}",
                        "low": f"{low:.2f}",
                        "volume_avg": int(hist["Volume"].mean())
                        if "Volume" in hist.columns
                        else 0,
                        "company_name": info.get("shortName", ticker),
                        "sector": info.get("sector", "Unknown"),
                        "industry": info.get("industry", "Unknown"),
                        "market_cap": info.get("marketCap", None),
                        "pe_ratio": info.get("trailingPE", None),
                    }

                    # Try to get recent news
                    try:
                        stock = yf.Ticker(ticker)
                        news = stock.news
                        if news:
                            source_result["data"]["news"] = [
                                {
                                    "title": item.get("title", ""),
                                    "link": item.get("link", ""),
                                    "publisher": item.get("publisher", ""),
                                    "date": datetime.fromtimestamp(
                                        item.get("providerPublishTime", 0)
                                    ).strftime("%Y-%m-%d"),
                                }
                                for item in news[:3]  # Just take the top 3 news
                            ]
                    except Exception as e:
                        logger.debug(f"Non-critical news retrieval error: {str(e)}")

                    source_result["success"] = True
                    return source_result

            return None

        # Map data source names from database to display names
        source_name_map = {
            "financial_modeling_prep": "Financial Modeling Prep",
            "alpha_vantage": "Alpha Vantage",
            "yahoo_finance": "Yahoo Finance",
        }

        # Process each data source
        for ds in data_sources:
            source_display_name = source_name_map.get(ds["name"], ds["name"])
            source_result = get_stock_data_from_source(source_display_name)

            # Add to results if successful
            if source_result and source_result.get("success"):
                all_results.append(source_result)

        # Check if we have enough results for aggregation
        if len(all_results) > 0:
            # Check if we should aggregate data
            should_aggregate = False
            min_sources = 2  # Default minimum sources for aggregation

            if user_preferences and "aggregate" in user_preferences:
                should_aggregate = user_preferences["aggregate"]
                if "min_sources" in user_preferences:
                    min_sources = user_preferences["min_sources"]

            if should_aggregate and len(all_results) >= min_sources:
                # Import data aggregator
                from src.utils.data_aggregator import DataAggregator

                # Aggregate data from all sources
                result = DataAggregator.aggregate_stock_data(all_results)
                result["success"] = True

                # Close the database connection if it was opened
                if db:
                    db.close()

                # Generate AI analysis if data is available
                if result.get("data"):
                    try:
                        # Prepare prompt for AI
                        prompt = f"""
                        Analyze the following stock data for {ticker} ({result['data']['company_name']}):

                        Period: {period}
                        Date range: {result['data']['start_date']} to {result['data']['end_date']}
                        Starting price: ${result['data']['start_price']}
                        Current price: ${result['data']['end_price']}
                        Change: {result['data']['percent_change']}%
                        Highest price in period: ${result['data']['high']}
                        Lowest price in period: ${result['data']['low']}
                        Average daily volume: {result['data']['volume_avg']}
                        Sector: {result['data']['sector']}
                        Industry: {result['data']['industry']}
                        Data source: Aggregated from multiple sources

                        Provide a concise analysis (maximum 200 words) of this stock's
                        trend over the given period. Mention whether it's bullish,
                        bearish, or neutral. Note any significant events or patterns.
                        Conclude with a brief outlook on what investors might expect.
                        """

                        # Try OpenAI first, fall back to Ollama
                        try:
                            result["analysis"] = analyze_with_openai(prompt)
                            result["model_used"] = "OpenAI"
                        except Exception as e:
                            logger.error(
                                f"OpenAI analysis failed, falling back to Ollama: {str(e)}"
                            )
                            # Get user's preferred model from settings
                            _, configured_model = get_ollama_settings()
                            result["analysis"] = analyze_with_ollama(
                                prompt, model=configured_model
                            )
                            result["model_used"] = f"Ollama ({configured_model})"
                    except Exception as e:
                        logger.error(f"Error generating analysis: {str(e)}")
                        result["analysis"] = f"Error generating analysis: {str(e)}"

                return result
            elif len(all_results) > 0:
                # Just use the first (highest priority) result
                result = all_results[0]
                result["success"] = True

                # Close the database connection if it was opened
                if db:
                    db.close()

                # Generate AI analysis if data is available
                if result.get("data"):
                    try:
                        # Prepare prompt for AI
                        prompt = f"""
                        Analyze the following stock data for {ticker} ({result['data']['company_name']}):

                        Period: {period}
                        Date range: {result['data']['start_date']} to {result['data']['end_date']}
                        Starting price: ${result['data']['start_price']}
                        Current price: ${result['data']['end_price']}
                        Change: {result['data']['percent_change']}%
                        Highest price in period: ${result['data']['high']}
                        Lowest price in period: ${result['data']['low']}
                        Average daily volume: {result['data']['volume_avg']}
                        Sector: {result['data']['sector']}
                        Industry: {result['data']['industry']}
                        Data source: {result['data_source']}

                        Provide a concise analysis (maximum 200 words) of this stock's
                        trend over the given period. Mention whether it's bullish,
                        bearish, or neutral. Note any significant events or patterns.
                        Conclude with a brief outlook on what investors might expect.
                        """

                        # Try OpenAI first, fall back to Ollama
                        try:
                            result["analysis"] = analyze_with_openai(prompt)
                            result["model_used"] = "OpenAI"
                        except Exception as e:
                            logger.error(
                                f"OpenAI analysis failed, falling back to Ollama: {str(e)}"
                            )
                            # Get user's preferred model from settings
                            _, configured_model = get_ollama_settings()
                            result["analysis"] = analyze_with_ollama(
                                prompt, model=configured_model
                            )
                            result["model_used"] = f"Ollama ({configured_model})"
                    except Exception as e:
                        logger.error(f"Error generating analysis: {str(e)}")
                        result["analysis"] = f"Error generating analysis: {str(e)}"

                return result

    # Use default fallback approach if no user preferences
    # or if all user-preferred sources failed
    # Get a list of all available data sources
    available_sources = []

    # Check which data sources we can use based on API keys
    if fmp_api_key:
        available_sources.append(
            {
                "name": "Financial Modeling Prep",
                "priority": 1,
                "fetcher": get_financial_modeling_prep_data,
            }
        )

    if alpha_vantage_key:
        available_sources.append(
            {
                "name": "Alpha Vantage",
                "priority": 2,
                "fetcher": get_alpha_vantage_data,
            }
        )

    # Yahoo Finance doesn't need an API key
    available_sources.append(
        {
            "name": "Yahoo Finance",
            "priority": 3,
            "fetcher": get_yahoo_finance_data,
        }
    )

    # Sort sources by priority
    available_sources.sort(key=lambda x: x["priority"])

    # Try each data source until we get data
    for source in available_sources:
        if result["data"]:
            # We already have data, no need to continue
            break

        source_name = source["name"]
        fetcher_func = source["fetcher"]

        # Log which source we're trying
        logger.info(f"Trying to fetch data for {ticker} from {source_name}")

        # Try to get data from this source
        success, hist, company_info = fetcher_func(ticker, period)

        if success and not hist.empty:
            result["data_source"] = source_name
            logger.info(f"Successfully using {source_name} data for {ticker}")

            # Process the data
            try:
                # Basic statistics
                start_price = hist["Close"].iloc[0]
                end_price = hist["Close"].iloc[-1]
                percent_change = ((end_price - start_price) / start_price) * 100
                high = hist["High"].max()
                low = hist["Low"].min()

                # Calculate indicators
                hist["MA50"] = hist["Close"].rolling(window=min(50, len(hist))).mean()
                hist["MA200"] = hist["Close"].rolling(window=min(200, len(hist))).mean()

                # Create data object with common mappings
                common_data = {
                    "start_date": hist.index[0].strftime("%Y-%m-%d"),
                    "end_date": hist.index[-1].strftime("%Y-%m-%d"),
                    "start_price": f"{start_price:.2f}",
                    "end_price": f"{end_price:.2f}",
                    "percent_change": f"{percent_change:.2f}",
                    "high": f"{high:.2f}",
                    "low": f"{low:.2f}",
                    "volume_avg": int(hist["Volume"].mean())
                    if "Volume" in hist.columns
                    else 0,
                }

                # Initialize result["data"] as a dictionary
                result["data"] = common_data

                # Add source-specific mappings
                additional_data = {}

                if source_name == "Financial Modeling Prep" and isinstance(
                    company_info, dict
                ):
                    additional_data = {
                        "company_name": company_info.get("companyName", ticker),
                        "sector": company_info.get("sector", "Unknown"),
                        "industry": company_info.get("industry", "Unknown"),
                        "market_cap": company_info.get("mktCap", None),
                        "pe_ratio": company_info.get("pe", None),
                    }

                elif source_name == "Alpha Vantage" and isinstance(company_info, dict):
                    additional_data = {
                        "company_name": company_info.get("Name", ticker),
                        "sector": company_info.get("Sector", "Unknown"),
                        "industry": company_info.get("Industry", "Unknown"),
                        "market_cap": company_info.get("MarketCapitalization", None),
                        "pe_ratio": company_info.get("PERatio", None),
                    }

                elif source_name == "Yahoo Finance" and isinstance(company_info, dict):
                    additional_data = {
                        "company_name": company_info.get("shortName", ticker),
                        "sector": company_info.get("sector", "Unknown"),
                        "industry": company_info.get("industry", "Unknown"),
                        "market_cap": company_info.get("marketCap", None),
                        "pe_ratio": company_info.get("trailingPE", None),
                    }

                    # Add news if available
                    if "news" in company_info and company_info["news"]:
                        try:
                            news_data = [
                                {
                                    "title": item.get("title", ""),
                                    "link": item.get("link", ""),
                                    "publisher": item.get("publisher", ""),
                                    "date": datetime.fromtimestamp(
                                        item.get("providerPublishTime", 0)
                                    ).strftime("%Y-%m-%d"),
                                }
                                for item in company_info["news"][
                                    :3
                                ]  # Just take the top 3 news
                            ]
                            additional_data["news"] = news_data
                        except Exception as e:
                            logger.warning(f"Error processing news data: {str(e)}")

                # Merge additional_data into result["data"]
                if additional_data and isinstance(result["data"], dict):
                    for key, value in additional_data.items():
                        result["data"][key] = value

            except Exception as e:
                logger.error(f"Error processing {source_name} data: {str(e)}")
                # Continue to the next data source

    # If we still don't have data after trying all sources, log an error
    if not result["data"]:
        logger.error(f"Could not retrieve data for {ticker} from any available source")
        result["analysis"] = (
            f"No data found for stock {ticker}. We tried multiple data sources "
            f"but couldn't retrieve the data."
        )
        return result
    # If we have data from any source, generate AI analysis
    if result["data"]:
        try:
            # Prepare prompt for AI
            prompt = f"""
            Analyze the following stock data for {ticker} ({result['data']['company_name']}):

            Period: {period}
            Date range: {result['data']['start_date']} to {result['data']['end_date']}
            Starting price: ${result['data']['start_price']}
            Current price: ${result['data']['end_price']}
            Change: {result['data']['percent_change']}%
            Highest price in period: ${result['data']['high']}
            Lowest price in period: ${result['data']['low']}
            Average daily volume: {result['data']['volume_avg']}
            Sector: {result['data']['sector']}
            Industry: {result['data']['industry']}
            Data source: {result['data_source']}

            Provide a concise analysis (maximum 200 words) of this stock's
            trend over the given period. Mention whether it's bullish,
            bearish, or neutral. Note any significant events or patterns.
            Conclude with a brief outlook on what investors might expect.
            """

            # Try OpenAI first, fall back to Ollama
            try:
                result["analysis"] = analyze_with_openai(prompt)
                result["model_used"] = "OpenAI"
            except Exception as e:
                logger.error(
                    f"OpenAI analysis failed, falling back to Ollama: {str(e)}"
                )
                # Get user's preferred model from settings
                _, configured_model = get_ollama_settings()
                result["analysis"] = analyze_with_ollama(prompt, model=configured_model)
                result["model_used"] = f"Ollama ({configured_model})"

            result["success"] = True
        except Exception as e:
            logger.error(f"Error generating analysis: {str(e)}")
            result["analysis"] = f"Error generating analysis: {str(e)}"

    # Close database connection if it was opened
    if db:
        db.close()

    return result


def analyze_crypto_trend(symbol: str, days: int = 180) -> Dict:
    """
    Analyze a cryptocurrency's trend using historical data and AI.
    """
    result = {
        "symbol": symbol,
        "days": days,
        "data": {},
        "analysis": "",
        "success": False,
    }

    try:
        # Try multiple ways to get crypto price data
        hist = pd.DataFrame()  # Initialize empty dataframe
        formats_tried = []

        # Dictionary of possible formats to try
        ticker_formats = [
            f"{symbol}-USD",  # Standard Yahoo format (BTC-USD)
            f"{symbol}USD=X",  # Currency pair format (BTCUSD=X)
            f"{symbol}-USD.CC",  # Crypto specific format
            f"{symbol.upper()}USDT",  # Some exchanges use USDT pairs
            f"{symbol.lower()}usdt",  # Lowercase format for some exchanges
        ]

        # Try all formats until we get data
        for ticker_format in ticker_formats:
            try:
                formats_tried.append(ticker_format)
                logger.info(f"Trying ticker format: {ticker_format}")

                # Use a custom User-Agent to avoid rate limiting
                headers = {
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                    )
                }
                # Create a custom session
                session = requests.Session()
                session.headers.update(headers)

                # Create ticker with custom session
                crypto = yf.Ticker(ticker_format)

                # Add an increasing delay between requests to avoid rate limiting
                time.sleep(0.5)  # Add a small delay before each request

                # Get history with error handling
                try:
                    current_hist = crypto.history(period=f"{days}d")

                    if not current_hist.empty and "Close" in current_hist.columns:
                        hist = current_hist
                        logger.info(
                            f"Successfully retrieved data using format: {ticker_format}"
                        )
                        break
                except Exception as inner_e:
                    logger.warning(
                        f"History fetch failed for {ticker_format}: {str(inner_e)}"
                    )
            except Exception as e:
                logger.warning(f"Failed to get data for {ticker_format}: {str(e)}")
                continue

        # If all formats failed, use API price directly or fallback to synthetic data
        if hist.empty or "Close" not in hist.columns:
            logger.warning(
                (
                    f"Could not retrieve historical data for {symbol} "
                    f"using formats: {formats_tried}"
                )
            )

            # Return error instead of synthetic data
            logger.error(
                (
                    f"Could not retrieve historical data for {symbol} "
                    f"using formats: {formats_tried}"
                )
            )
            # Try to get current price from CoinGecko's free API as fallback
            try:
                api_url = (
                    f"https://api.coingecko.com/api/v3/simple/price?ids={symbol.lower()}"
                    f"&vs_currencies=usd"
                )
                response = requests.get(api_url, timeout=10)  # 10 second timeout
                if response.status_code == 200:
                    data = response.json()
                    if symbol.lower() in data and "usd" in data[symbol.lower()]:
                        current_price = data[symbol.lower()]["usd"]
                        logger.info(
                            f"Got current price from CoinGecko: ${current_price}"
                        )
                        error_msg = (
                            f"Could not retrieve historical data for {symbol}, "
                            f"but current price is ${current_price}."
                        )
                    else:
                        error_msg = (
                            f"Could not retrieve historical data for {symbol} "
                            f"and price data is not available."
                        )
                else:
                    error_msg = (
                        f"Could not retrieve historical data for {symbol} "
                        f"and CoinGecko API returned status {response.status_code}."
                    )
            except Exception as e:
                logger.warning(f"Failed to get price from CoinGecko: {str(e)}")
                error_msg = (
                    f"Could not retrieve historical data for {symbol} "
                    f"from any data source."
                )
            # Return error
            return {
                "symbol": symbol,
                "days": days,
                "data": {},
                "analysis": error_msg,
                "success": False,
                "error": error_msg,
            }

            # This code will never be reached due to the early return above

        # Basic statistics
        start_price = hist["Close"].iloc[0]
        end_price = hist["Close"].iloc[-1]
        percent_change = ((end_price - start_price) / start_price) * 100
        high = hist["High"].max()
        low = hist["Low"].min()

        # Calculate some indicators (with error handling)
        try:
            hist["MA20"] = hist["Close"].rolling(window=min(20, len(hist))).mean()
            hist["MA50"] = hist["Close"].rolling(window=min(50, len(hist))).mean()

            ma20 = hist["MA20"].iloc[-1] if not pd.isna(hist["MA20"].iloc[-1]) else None
            ma50 = hist["MA50"].iloc[-1] if not pd.isna(hist["MA50"].iloc[-1]) else None
        except Exception as e:
            logger.warning(f"Error calculating MA indicators: {str(e)}")
            ma20 = None
            ma50 = None

        # Calculate volume average with error handling
        try:
            volume_avg = int(hist["Volume"].mean()) if "Volume" in hist.columns else 0
        except Exception:
            volume_avg = 0

        # Create data object
        result["data"] = {
            "start_date": hist.index[0].strftime("%Y-%m-%d"),
            "end_date": hist.index[-1].strftime("%Y-%m-%d"),
            "start_price": round(start_price, 2),
            "end_price": round(end_price, 2),
            "percent_change": round(percent_change, 2),
            "high": round(high, 2),
            "low": round(low, 2),
            "volume_avg": volume_avg,
            "current_ma20": round(ma20, 2) if ma20 is not None else None,
            "current_ma50": round(ma50, 2) if ma50 is not None else None,
        }

        # Generate summary using AI
        try:
            # Prepare prompt for AI
            prompt = f"""
            Analyze the following cryptocurrency data for {symbol}:

            Period: {days} days
            Date range: {result['data']['start_date']} to {result['data']['end_date']}
            Starting price: ${result['data']['start_price']}
            Current price: ${result['data']['end_price']}
            Change: {result['data']['percent_change']}%
            Highest price in period: ${result['data']['high']}
            Lowest price in period: ${result['data']['low']}
            Average daily volume: {result['data']['volume_avg']}
            Current 20-day moving average: ${result['data']['current_ma20']}
            Current 50-day moving average: ${result['data']['current_ma50']}

            Provide a concise analysis (maximum 200 words) of this cryptocurrency's
            trend over the given period. Mention whether it's bullish,
            bearish, or neutral. Note any significant events or patterns.
            Conclude with a brief outlook on what investors might expect.
            """

            # Try to use Ollama first, fall back to OpenAI if that fails
            try:
                # Get the configured model from environment instead of auto-selecting
                _, configured_model = get_ollama_settings()
                result["analysis"] = analyze_with_ollama(
                    prompt, model=configured_model, task_type="finance"
                )
                result["model_used"] = f"Ollama ({configured_model})"
            except Exception as e:
                logger.error(f"Error with Ollama analysis: {str(e)}")
                try:
                    # Fall back to OpenAI if Ollama fails
                    result["analysis"] = analyze_with_openai(prompt)
                    result["model_used"] = "OpenAI"
                except Exception as e2:
                    logger.error(f"Both Ollama and OpenAI failed: {str(e2)}")
                    result["analysis"] = (
                        "Error: Could not generate analysis. Please check Ollama server "
                        "or network connection."
                    )
                    result["model_used"] = "Error"

            result["success"] = True

        except Exception as e:
            logger.error(f"Error generating analysis for {symbol}: {str(e)}")
            result["analysis"] = "Error generating analysis. Please try again later."
            result["success"] = True  # Still mark as success since we have data

    except Exception as e:
        logger.error(f"Error analyzing cryptocurrency {symbol}: {str(e)}")
        result["analysis"] = f"Error: {str(e)}"

    return result


def get_etf_recommendations(
    risk_profile: str = "moderate", sectors: List[str] = None
) -> Dict:
    """
    Get ETF recommendations based on risk profile and sectors of interest.
    """
    result = {
        "risk_profile": risk_profile,
        "sectors": sectors,
        "recommendations": [],
        "analysis": "",
        "success": False,
    }

    # Define some common ETFs by risk profile
    etf_by_risk = {
        "conservative": ["BND", "SCHZ", "VCSH", "MUB", "VTIP"],
        "moderate": ["VTI", "SPY", "QQQ", "VEA", "VXUS"],
        "aggressive": ["ARKK", "VGT", "XLK", "SOXX", "SMH"],
    }

    # Define some common ETFs by sector
    etf_by_sector = {
        "technology": ["VGT", "XLK", "FTEC", "SMH", "SOXX"],
        "healthcare": ["VHT", "XLV", "IHI", "IBB", "XBI"],
        "finance": ["VFH", "XLF", "KBE", "KBWB", "IAI"],
        "energy": ["VDE", "XLE", "IYE", "XOP", "ICLN"],
        "consumer": ["VCR", "XLY", "XLP", "VDC", "IEDI"],
        "real estate": ["VNQ", "XLRE", "IYR", "REZ", "RWR"],
    }

    try:
        # Choose ETFs based on risk profile and sectors
        selected_etfs = []

        # Add risk-based ETFs
        if risk_profile in etf_by_risk:
            selected_etfs.extend(etf_by_risk[risk_profile])

        # Add sector-based ETFs
        if sectors:
            for sector in sectors:
                if sector in etf_by_sector:
                    selected_etfs.extend(etf_by_sector[sector])

        # Ensure unique ETFs and limit to 10
        selected_etfs = list(set(selected_etfs))[:10]

        # Get ETF data
        recommendations = []
        for etf_ticker in selected_etfs:
            try:
                etf = yf.Ticker(etf_ticker)
                info = etf.info

                # Get 1-year return
                hist = etf.history(period="1y")
                if not hist.empty and len(hist) > 5:
                    start_price = hist["Close"].iloc[0]
                    end_price = hist["Close"].iloc[-1]
                    yearly_change = ((end_price - start_price) / start_price) * 100
                else:
                    yearly_change = 0.0

                # Create recommendation
                recommendation = {
                    "ticker": etf_ticker,
                    "name": info.get("shortName", etf_ticker),
                    "category": info.get("category", ""),
                    "expense_ratio": info.get("expenseRatio", 0.0),
                    "yearly_change": round(yearly_change, 2),
                    "current_price": round(info.get("regularMarketPrice", 0.0), 2),
                }

                recommendations.append(recommendation)
            except Exception as e:
                logger.warning(f"Error getting data for ETF {etf_ticker}: {str(e)}")
                # Add a placeholder with minimal information
                recommendations.append(
                    {
                        "ticker": etf_ticker,
                        "name": etf_ticker,
                        "category": "",
                        "expense_ratio": 0.0,
                        "yearly_change": 0.0,
                        "current_price": 0.0,
                    }
                )

        result["recommendations"] = recommendations

        # Generate analysis
        if len(recommendations) > 0:
            # Prepare recommendation data for analysis
            etf_data = "\n".join(
                [
                    (
                        f"- {r['ticker']} ({r['name']}): 1yr return {r['yearly_change']}%, "
                        f"expense ratio {r['expense_ratio']}%"
                    )
                    for r in recommendations[:5]
                ]
            )

            prompt = f"""
            Analyze the following ETF recommendations for a {risk_profile} risk profile investor:

            {etf_data}

            Provide a concise analysis (maximum 150 words) explaining why
            these ETFs are appropriate for
            a {risk_profile} risk profile.

            Mention the benefits of this allocation and any potential
            considerations or risks.
            """

            try:
                result["analysis"] = analyze_with_openai(prompt, task_type="finance")
                result["model_used"] = "OpenAI"
            except Exception:
                try:
                    # Get user's preferred model from settings
                    _, configured_model = get_ollama_settings()
                    result["analysis"] = analyze_with_ollama(
                        prompt, model=configured_model, task_type="finance"
                    )
                    result["model_used"] = f"Ollama ({configured_model})"
                except Exception as ollama_error:
                    logger.warning(
                        f"ETF analysis with Ollama failed: {str(ollama_error)}"
                    )
                    result[
                        "analysis"
                    ] = f"These ETFs are tailored for {risk_profile} investors, providing a balanced approach to market exposure. Consider individual research before investing."
                    result["model_used"] = "Fallback (no model)"

            result["success"] = True

    except Exception as e:
        logger.error(f"Error generating ETF recommendations: {str(e)}")
        result["analysis"] = f"Error: {str(e)}"

    return result


def analyze_with_best_model(
    prompt: str, task_type: str = "finance", fallback_message: Optional[str] = None
) -> str:
    """
    Use the best available model to perform analysis based on task type

    Args:
        prompt: The analysis prompt
        task_type: Type of task (finance, general, coding)
        fallback_message: Optional fallback message if all models fail

    Returns:
        Analysis result
    """
    # First try OpenAI
    try:
        # For finance tasks, try to use GPT-4 if configured
        openai_model = "gpt-3.5-turbo"
        if task_type == "finance" and os.getenv("OPENAI_API_KEY"):
            # Get API configuration
            from src.utils.api_config import APIConfigManager

            # Check if GPT-4 is enabled in the configuration
            gpt4_enabled = APIConfigManager.get_api_value_from_env(
                "USE_GPT4_FOR_FINANCE"
            )
            if gpt4_enabled and gpt4_enabled.lower() in ["true", "yes", "1"]:
                openai_model = "gpt-4"
                logger.info("Using GPT-4 for finance task based on configuration")

        return analyze_with_openai(prompt, model=openai_model, task_type=task_type)
    except Exception as e:
        logger.warning(f"OpenAI analysis failed, trying Ollama: {str(e)}")

        # First try with user's preferred model
        try:
            # Get user's preferred model from settings
            _, configured_model = get_ollama_settings()
            logger.info(f"Trying user-configured model: {configured_model}")

            result = analyze_with_ollama(
                prompt, model=configured_model, task_type=task_type
            )

            # Check if the result indicates a 404 error (model not found)
            if "Error: 404" in result or "Error: Model" in result:
                # If 404, try one more time with most basic model (llama2)
                logger.warning(
                    f"Model {configured_model} not found, falling back to basic model"
                )
                raise ValueError(f"Model {configured_model} not found: {result}")

            return result

        except Exception as e2:
            logger.warning(
                f"First Ollama attempt failed: {str(e2)}, trying fallback model"
            )

            # Try a final time with a simple, common model that should exist
            try:
                fallback_models = ["llama2", "mistral", "gemma:2b"]

                for fallback_model in fallback_models:
                    try:
                        logger.info(f"Trying fallback model: {fallback_model}")
                        return analyze_with_ollama(
                            prompt, model=fallback_model, task_type=task_type
                        )
                    except Exception as model_error:
                        logger.warning(
                            (
                                f"Fallback model {fallback_model} failed: "
                                f"{str(model_error)}"
                            )
                        )
                        continue

                # If all models failed, raise the error to trigger the fallback message
                raise ValueError("All Ollama models failed")

            except Exception as e3:
                logger.error(f"All AI analysis methods and fallbacks failed: {str(e3)}")

                # Use fallback message or generate a generic one
                if fallback_message:
                    return fallback_message
                return (
                    "Analysis could not be generated at this time. "
                    "Please try again later."
                )


def generate_generic_analysis(prompt: str) -> str:
    """Generate a generic analysis when AI systems fail"""
    return analyze_with_best_model(
        prompt,
        task_type="general",
        fallback_message=(
            "Analysis could not be generated at this time. " "Please try again later."
        ),
    )


def generate_social_post(post_type: str, post_data: dict) -> str:
    """
    Generate social media post content based on type and data.

    Args:
        post_type: Type of post to generate (portfolio_update, market_update,
            investment_tip)
        post_data: Dictionary containing data specific to the post type

    Returns:
        Generated post content as string
    """
    try:
        # Build a prompt based on post type
        if post_type == "portfolio_update":
            portfolio_value = post_data.get("portfolio_value", 10000)
            daily_change = post_data.get("daily_change", 0)
            top_assets = post_data.get("top_assets", [])

            # Format top assets for the prompt
            assets_str = ", ".join(top_assets[:3]) if top_assets else "various assets"

            prompt = f"""
            Create a concise social media post (max 280 characters) about a portfolio
            update with the following details:
            - Portfolio value: ${portfolio_value:,.2f}
            - Daily change: {daily_change}%
            - Top performing assets: {assets_str}

            The tone should be professional but conversational.
            Include a relevant emoji or two.
            Do not use hashtags unless they add significant value.
            """

        elif post_type == "market_update":
            market_data = post_data.get("market_data", {})
            btc_price = market_data.get("btc_price", 69000)
            eth_price = market_data.get("eth_price", 3500)
            market_trend = market_data.get("market_trend", "neutral")
            sentiment = post_data.get("sentiment", "neutral")

            prompt = f"""
            Create a concise social media post (max 280 characters) about
            current market conditions:
            - Bitcoin price: ${btc_price:,.2f}
            - Ethereum price: ${eth_price:,.2f}
            - Overall market trend: {market_trend}
            - Market sentiment: {sentiment}

            The tone should be factual with a hint of {sentiment} outlook. Include 1-2
            relevant emojis. Focus on providing value, not just sharing prices.
            """

        elif post_type == "investment_tip":
            topic = post_data.get("topic", "diversification")

            # Map topic to a more descriptive version
            topic_map = {
                "diversification": "portfolio diversification",
                "dollar_cost_averaging": "dollar-cost averaging (DCA)",
                "risk_management": "risk management",
                "emergency_fund": "maintaining an emergency fund",
                "tax_efficiency": "tax-efficient investing",
            }

            topic_desc = topic_map.get(topic, topic)

            prompt = f"""
            Create a concise social media post (max 280 characters) sharing
            an investment tip about {topic_desc}.
            The tone should be helpful and educational without being preachy.
            Include a relevant emoji or two to make the post more engaging.
            The tip should be actionable and valuable to both beginner
            and intermediate investors.
            """

        else:
            return (
                "Error: Unsupported post type. Please choose from "
                "portfolio_update, market_update, or investment_tip."
            )

        # Generate the post content using the best available model
        return analyze_with_best_model(
            prompt,
            task_type="finance",
            fallback_message=(
                "📊 Market update: BTC at $69k, ETH at $3.5k. Overall sentiment"
                " remains cautiously optimistic despite recent volatility. Keep an eye on"
                " upcoming economic data."
            ),
        )

    except Exception as e:
        logger.error(f"Error generating social post: {str(e)}")
        return f"Error: Could not generate post content - {str(e)}"
