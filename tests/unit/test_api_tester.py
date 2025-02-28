"""Unit tests for the APITester class."""
import pytest
from unittest.mock import patch, MagicMock
import requests
import json
from src.utils.api_tester import APITester


@pytest.fixture
def mock_response():
    """Create a mock successful response object."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"success": True}
    mock_resp.text = '{"success": true}'
    return mock_resp


@pytest.fixture
def mock_error_response():
    """Create a mock error response object."""
    mock_resp = MagicMock()
    mock_resp.status_code = 401
    mock_resp.json.return_value = {"error": "Invalid API key"}
    mock_resp.text = '{"error": "Invalid API key"}'
    return mock_resp


@patch("src.utils.api_tester.OpenAI")
def test_test_openai_success(mock_openai_class, mock_response):
    """Test OpenAI API connection success."""
    # Setup the mock
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    mock_client.chat.completions.create.return_value.choices = [
        MagicMock(message=MagicMock(content="Test response"))
    ]

    # Call the test function
    success, message = APITester.test_openai("fake_api_key")

    # Verify
    assert success is True
    assert "Successfully" in message
    mock_openai_class.assert_called_once_with(api_key="fake_api_key")


@patch("src.utils.api_tester.OpenAI")
def test_test_openai_failure(mock_openai_class):
    """Test OpenAI API connection failure."""
    # Setup the mock to raise an exception
    mock_openai_class.side_effect = Exception("Invalid API key")

    # Call the test function
    success, message = APITester.test_openai("fake_api_key")

    # Verify
    assert success is False
    assert "Failed" in message
    assert "Invalid API key" in message


@patch("requests.post")
def test_test_anthropic_success(mock_post, mock_response):
    """Test Anthropic API connection success."""
    mock_post.return_value = mock_response

    # Call the test function
    success, message = APITester.test_anthropic("fake_api_key")

    # Verify
    assert success is True
    assert "Successfully" in message
    mock_post.assert_called_once()


@patch("requests.post")
def test_test_anthropic_failure(mock_post, mock_error_response):
    """Test Anthropic API connection failure."""
    mock_post.return_value = mock_error_response

    # Call the test function
    success, message = APITester.test_anthropic("fake_api_key")

    # Verify
    assert success is False
    # The actual message contains the status code but not "Failed"
    assert "401" in message


@patch("binance.client.Client")
def test_test_binance_success(mock_client_class):
    """Test Binance API connection success."""
    # Setup the mock
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client
    mock_client.get_account.return_value = {"balances": []}

    # Call the test function
    success, message = APITester.test_binance("fake_api_key", "fake_api_secret")

    # Verify
    assert success is True
    assert "Successfully" in message
    mock_client_class.assert_called_once_with("fake_api_key", "fake_api_secret", tld='us')


@patch("binance.client.Client")
def test_test_binance_failure(mock_client_class):
    """Test Binance API connection failure."""
    # Setup the mock to raise an exception
    mock_client_class.side_effect = Exception("Invalid API credentials")

    # Call the test function
    success, message = APITester.test_binance("fake_api_key", "fake_api_secret")

    # Verify
    assert success is False
    assert "Failed" in message
    assert "Invalid API credentials" in message


@patch("requests.get")
def test_test_alpha_vantage_success(mock_get):
    """Test Alpha Vantage API connection success."""
    # Setup the mock
    mock_response = MagicMock()
    mock_response.json.return_value = {"Global Quote": {"01. symbol": "IBM"}}
    mock_get.return_value = mock_response

    # Call the test function
    success, message = APITester.test_alpha_vantage("fake_api_key")

    # Verify
    assert success is True
    assert "Successfully" in message
    mock_get.assert_called_once()
    assert "fake_api_key" in mock_get.call_args[0][0]


@patch("requests.get")
def test_test_alpha_vantage_rate_limit(mock_get):
    """Test Alpha Vantage API rate limit handling."""
    # Setup the mock
    mock_response = MagicMock()
    mock_response.json.return_value = {"Note": "API call frequency"}
    mock_get.return_value = mock_response

    # Call the test function
    success, message = APITester.test_alpha_vantage("fake_api_key")

    # Verify
    assert success is True
    assert "API call frequency" in message


@patch("src.utils.api_tester.yf.Ticker")
def test_test_yahoo_finance_success(mock_ticker):
    """Test Yahoo Finance API connection success."""
    # Setup the mock
    mock_ticker_instance = MagicMock()
    mock_ticker_instance.info = {"symbol": "AAPL", "shortName": "Apple Inc."}
    mock_ticker.return_value = mock_ticker_instance

    # Call the test function
    success, message = APITester.test_yahoo_finance()

    # Verify
    assert success is True
    assert "Successfully" in message
    mock_ticker.assert_called_once_with("AAPL")


@patch("src.utils.api_tester.yf.Ticker")
def test_test_yahoo_finance_failure(mock_ticker):
    """Test Yahoo Finance API connection failure."""
    # Setup the mock
    mock_ticker_instance = MagicMock()
    mock_ticker_instance.info = {}  # Empty info
    mock_ticker.return_value = mock_ticker_instance

    # Call the test function
    success, message = APITester.test_yahoo_finance()

    # Verify
    assert success is False
    assert "empty data" in message


@patch("requests.get")
def test_test_coingecko_success(mock_get):
    """Test CoinGecko API connection success."""
    # Setup the mock
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "market_data": {"current_price": {"usd": 50000}}
    }
    mock_get.return_value = mock_response

    # Call the test function
    success, message = APITester.test_coingecko()

    # Verify
    assert success is True
    assert "Successfully" in message
    assert "50000" in message
    mock_get.assert_called_once()


@patch("requests.get")
def test_test_coingecko_rate_limit(mock_get):
    """Test CoinGecko API rate limit handling."""
    # Setup the mock
    mock_response = MagicMock()
    mock_response.status_code = 429
    mock_get.return_value = mock_response

    # Call the test function
    success, message = APITester.test_coingecko()

    # Verify
    assert success is False
    assert "rate limit" in message.lower()


@patch("src.utils.api_tester.ccxt.kraken")
def test_test_kraken_success(mock_kraken_class):
    """Test Kraken API connection success."""
    # Setup the mock
    mock_kraken = MagicMock()
    mock_kraken_class.return_value = mock_kraken
    mock_kraken.fetch_balance.return_value = {"total": {"BTC": 1.0}}

    # Call the test function
    success, message = APITester.test_kraken("fake_api_key", "fake_api_secret")

    # Verify
    assert success is True
    assert "Successfully" in message
    mock_kraken_class.assert_called_once_with({
        'apiKey': "fake_api_key",
        'secret': "fake_api_secret",
        'enableRateLimit': True
    })


@patch("src.utils.api_tester.ccxt.kraken")
def test_test_kraken_failure(mock_kraken_class):
    """Test Kraken API connection failure."""
    # Setup the mock
    mock_kraken = MagicMock()
    mock_kraken_class.return_value = mock_kraken
    mock_kraken.fetch_balance.side_effect = Exception("Invalid API credentials")

    # Call the test function
    success, message = APITester.test_kraken("fake_api_key", "fake_api_secret")

    # Verify
    assert success is False
    assert "Failed" in message
    assert "Invalid API credentials" in message


@patch("requests.get")
def test_test_ollama_success(mock_get, mock_response):
    """Test Ollama API connection success."""
    # Setup the mock
    mock_response.json.return_value = {"models": [{"name": "llama2"}]}
    mock_get.return_value = mock_response

    # Call the test function with default URL
    success, message = APITester.test_ollama()

    # Verify
    assert success is True
    assert "Successfully" in message
    assert "llama2" in message
    mock_get.assert_called_once_with("http://localhost:11434/api/tags")


@patch("requests.get")
def test_test_ollama_custom_url(mock_get, mock_response):
    """Test Ollama API connection with custom URL."""
    # Setup the mock
    mock_response.json.return_value = {"models": [{"name": "llama2"}]}
    mock_get.return_value = mock_response

    # Call the test function with custom URL
    success, message = APITester.test_ollama("http://custom-server:11434")

    # Verify
    assert success is True
    assert "Successfully" in message
    mock_get.assert_called_once_with("http://custom-server:11434/api/tags")


def test_get_api_config_info():
    """Test the API configuration information retrieval."""
    configs = APITester.get_api_config_info()
    
    # Check that we have a list of dictionaries
    assert isinstance(configs, list)
    assert len(configs) > 0
    assert isinstance(configs[0], dict)
    
    # Check required fields in each config
    for config in configs:
        assert "name" in config
        assert "service_id" in config
        assert "description" in config
        assert "category" in config
        assert "needs_key" in config
        
        # If service needs a key, check for env_var_key
        if config["needs_key"]:
            assert "env_var_key" in config