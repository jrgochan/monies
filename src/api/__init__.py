"""API interface modules for external services."""

import logging
import os
from typing import Any, Dict, Optional, Tuple, Union

import requests
from dotenv import load_dotenv

# Load environment variables once at module import
load_dotenv()

# Configure logging once at module level
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Common API keys used across modules
API_KEYS = {
    "openai": os.getenv("OPENAI_API_KEY", ""),
    "alpha_vantage": os.getenv("ALPHA_VANTAGE_KEY", ""),
    "fmp": os.getenv("FMP_API_KEY", ""),
    "polygon": os.getenv("POLYGON_API_KEY", ""),
    "twitter": os.getenv("TWITTER_API_KEY", ""),
    "twitter_secret": os.getenv("TWITTER_API_SECRET", ""),
    "binance": os.getenv("BINANCE_API_KEY", ""),
    "binance_secret": os.getenv("BINANCE_SECRET_KEY", ""),
    "coinbase": os.getenv("COINBASE_API_KEY", ""),
    "coinbase_secret": os.getenv("COINBASE_SECRET_KEY", ""),
}


class APIError(Exception):
    """Custom exception for API errors."""

    pass


def make_api_request(
    url: str,
    method: str = "GET",
    headers: Optional[Dict[str, str]] = None,
    params: Optional[Dict[str, Any]] = None,
    data: Optional[Dict[str, Any]] = None,
    json_data: Optional[Dict[str, Any]] = None,
    timeout: int = 15,
    retries: int = 3,
    retry_delay: int = 1,
    error_msg: str = "API request failed",
) -> Tuple[bool, Union[Dict[str, Any], str]]:
    """
    Make an API request with retry logic and error handling.

    Args:
        url: The URL to request
        method: HTTP method (GET, POST, etc.)
        headers: Request headers
        params: URL parameters
        data: Form data
        json_data: JSON payload
        timeout: Request timeout in seconds
        retries: Number of retry attempts
        retry_delay: Delay between retries in seconds
        error_msg: Custom error message prefix

    Returns:
        Tuple of (success, response_data)
    """
    headers = headers or {}

    # Add a default user agent if not provided
    if "User-Agent" not in headers:
        headers["User-Agent"] = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )

    for attempt in range(retries):
        try:
            response = requests.request(
                method=method,
                url=url,
                headers=headers,
                params=params,
                data=data,
                json=json_data,
                timeout=timeout,
            )

            # Check if the request was successful
            if response.status_code == 200:
                # Try to parse as JSON first
                try:
                    return True, response.json()
                except ValueError:
                    # Return the raw text if not valid JSON
                    return True, response.text
            else:
                logger.warning(
                    f"API request failed with status {response.status_code}: {response.text}"
                )

                # If we have more attempts, retry
                if attempt < retries - 1:
                    import time

                    time.sleep(retry_delay)
                    continue

                return False, f"HTTP Error: {response.status_code}"

        except requests.exceptions.Timeout:
            logger.warning(f"Request timeout: {url}")
            if attempt < retries - 1:
                import time

                time.sleep(retry_delay)
                continue
            return False, "Request timed out"

        except requests.exceptions.ConnectionError:
            logger.warning(f"Connection error: {url}")
            if attempt < retries - 1:
                import time

                time.sleep(retry_delay)
                continue
            return False, "Connection error"

        except Exception as e:
            logger.error(f"{error_msg}: {str(e)}")
            if attempt < retries - 1:
                import time

                time.sleep(retry_delay)
                continue
            return False, f"Error: {str(e)}"

    # If we get here, all retries have failed
    return False, "Maximum retries exceeded"
