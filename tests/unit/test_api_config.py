"""Unit tests for the APIConfigManager class."""
import pytest
import os
from unittest.mock import patch, MagicMock, mock_open
import tempfile
from pathlib import Path

from src.utils.api_config import APIConfigManager


@pytest.fixture
def temp_env_file():
    """Create a temporary .env file for testing."""
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as f:
        f.write("TEST_API_KEY=test_value\n")
        f.write("TEST_SECRET_KEY=secret_value\n")
        temp_path = f.name
    
    yield temp_path
    
    # Clean up the temporary file
    os.unlink(temp_path)


@patch("src.utils.api_config.APITester")
def test_get_api_configs(mock_api_tester):
    """Test getting API configurations."""
    # Setup mock
    mock_configs = [
        {"name": "Test API", "service_id": "test_api"},
        {"name": "Another API", "service_id": "another_api"}
    ]
    mock_api_tester.get_api_config_info.return_value = mock_configs
    
    # Call the function
    result = APIConfigManager.get_api_configs()
    
    # Verify
    assert result == mock_configs
    mock_api_tester.get_api_config_info.assert_called_once()


@patch("src.utils.api_config.APITester")
def test_get_api_config_by_service_found(mock_api_tester):
    """Test getting API configuration by service ID when it exists."""
    # Setup mock
    mock_configs = [
        {"name": "Test API", "service_id": "test_api"},
        {"name": "Target API", "service_id": "target_api"}
    ]
    mock_api_tester.get_api_config_info.return_value = mock_configs
    
    # Call the function
    result = APIConfigManager.get_api_config_by_service("target_api")
    
    # Verify
    assert result == {"name": "Target API", "service_id": "target_api"}
    mock_api_tester.get_api_config_info.assert_called_once()


@patch("src.utils.api_config.APITester")
def test_get_api_config_by_service_not_found(mock_api_tester):
    """Test getting API configuration by service ID when it doesn't exist."""
    # Setup mock
    mock_configs = [
        {"name": "Test API", "service_id": "test_api"}
    ]
    mock_api_tester.get_api_config_info.return_value = mock_configs
    
    # Call the function
    result = APIConfigManager.get_api_config_by_service("nonexistent")
    
    # Verify
    assert result is None
    mock_api_tester.get_api_config_info.assert_called_once()


@patch("src.utils.api_config.APITester")
def test_get_api_categories(mock_api_tester):
    """Test getting unique API categories."""
    # Setup mock
    mock_configs = [
        {"name": "API 1", "service_id": "api1", "category": "Category A"},
        {"name": "API 2", "service_id": "api2", "category": "Category B"},
        {"name": "API 3", "service_id": "api3", "category": "Category A"}
    ]
    mock_api_tester.get_api_config_info.return_value = mock_configs
    
    # Call the function
    result = APIConfigManager.get_api_categories()
    
    # Verify
    assert set(result) == {"Category A", "Category B"}
    assert len(result) == 2
    mock_api_tester.get_api_config_info.assert_called_once()


@patch("src.utils.api_config.APITester")
def test_get_api_configs_by_category(mock_api_tester):
    """Test getting API configurations by category."""
    # Setup mock
    mock_configs = [
        {"name": "API 1", "service_id": "api1", "category": "Category A"},
        {"name": "API 2", "service_id": "api2", "category": "Category B"},
        {"name": "API 3", "service_id": "api3", "category": "Category A"}
    ]
    mock_api_tester.get_api_config_info.return_value = mock_configs
    
    # Call the function
    result = APIConfigManager.get_api_configs_by_category("Category A")
    
    # Verify
    assert len(result) == 2
    assert result[0]["name"] == "API 1"
    assert result[1]["name"] == "API 3"
    mock_api_tester.get_api_config_info.assert_called_once()


@patch("src.utils.api_config.os.getenv")
def test_get_api_value_from_env(mock_getenv):
    """Test getting API value from environment variable."""
    # Setup mock
    mock_getenv.return_value = "test_api_key_value"
    
    # Call the function
    result = APIConfigManager.get_api_value_from_env("TEST_API_KEY")
    
    # Verify
    assert result == "test_api_key_value"
    mock_getenv.assert_called_once_with("TEST_API_KEY", "")


@patch("src.utils.api_config.APIConfigManager.get_env_path")
@patch("src.utils.api_config.set_key")
@patch("src.utils.api_config.load_dotenv")
def test_update_env_file_success(mock_load_dotenv, mock_set_key, mock_get_env_path):
    """Test updating environment variable in .env file successfully."""
    # Setup mocks
    mock_get_env_path.return_value = "/path/to/.env"
    mock_set_key.return_value = None  # No return value when successful
    
    # Call the function
    result = APIConfigManager.update_env_file("TEST_KEY", "new_value")
    
    # Verify
    assert result is True
    mock_get_env_path.assert_called_once()
    mock_set_key.assert_called_once_with("/path/to/.env", "TEST_KEY", "new_value")
    mock_load_dotenv.assert_called_once()


@patch("src.utils.api_config.APIConfigManager.get_env_path")
@patch("src.utils.api_config.set_key")
def test_update_env_file_failure(mock_set_key, mock_get_env_path):
    """Test updating environment variable in .env file with failure."""
    # Setup mocks
    mock_get_env_path.return_value = "/path/to/.env"
    mock_set_key.side_effect = Exception("File not found")
    
    # Call the function
    result = APIConfigManager.update_env_file("TEST_KEY", "new_value")
    
    # Verify
    assert result is False
    mock_get_env_path.assert_called_once()
    mock_set_key.assert_called_once_with("/path/to/.env", "TEST_KEY", "new_value")


@patch("src.utils.api_config.APIConfigManager.get_api_config_by_service")
@patch("src.utils.api_config.APIConfigManager.get_api_value_from_env")
def test_get_api_credentials_existing(mock_get_value, mock_get_config):
    """Test getting API credentials for a service that exists."""
    # Setup mocks
    mock_get_config.return_value = {
        "name": "Test API",
        "service_id": "test_api", 
        "env_var_key": "TEST_API_KEY",
        "env_var_secret": "TEST_SECRET_KEY"
    }
    mock_get_value.side_effect = ["api_key_value", "secret_value"]
    
    # Call the function
    key, secret = APIConfigManager.get_api_credentials("test_api")
    
    # Verify
    assert key == "api_key_value"
    assert secret == "secret_value"
    mock_get_config.assert_called_once_with("test_api")
    assert mock_get_value.call_count == 2
    mock_get_value.assert_any_call("TEST_API_KEY")
    mock_get_value.assert_any_call("TEST_SECRET_KEY")


@patch("src.utils.api_config.APIConfigManager.get_api_config_by_service")
def test_get_api_credentials_nonexistent(mock_get_config):
    """Test getting API credentials for a service that doesn't exist."""
    # Setup mock
    mock_get_config.return_value = None
    
    # Call the function
    key, secret = APIConfigManager.get_api_credentials("nonexistent")
    
    # Verify
    assert key == ""
    assert secret == ""
    mock_get_config.assert_called_once_with("nonexistent")


@patch("src.utils.api_config.APIConfigManager.get_api_config_by_service")
@patch("src.utils.api_config.APITester")
def test_test_api_connection_nonexistent_service(mock_api_tester, mock_get_config):
    """Test API connection testing for a nonexistent service."""
    # Setup mock
    mock_get_config.return_value = None
    
    # Call the function
    success, message = APIConfigManager.test_api_connection("nonexistent")
    
    # Verify
    assert success is False
    assert "Unknown service" in message
    mock_get_config.assert_called_once_with("nonexistent")
    mock_api_tester.assert_not_called()


@patch("src.utils.api_config.APIConfigManager.get_api_config_by_service")
@patch("src.utils.api_config.APIConfigManager.get_api_value_from_env")
@patch("src.utils.api_config.APITester")
def test_test_api_connection_url_based(mock_api_tester, mock_get_value, mock_get_config):
    """Test API connection testing for a URL-based service (like Ollama)."""
    # Setup mocks
    mock_get_config.return_value = {
        "name": "Ollama",
        "service_id": "ollama",
        "is_url": True,
        "default_url": "http://localhost:11434",
        "env_var_key": "OLLAMA_BASE_URL"
    }
    mock_get_value.return_value = "http://custom-url:11434"
    mock_test_func = MagicMock(return_value=(True, "Successfully connected"))
    mock_api_tester.test_ollama = mock_test_func
    
    # Call the function
    success, message = APIConfigManager.test_api_connection("ollama")
    
    # Verify
    assert success is True
    assert message == "Successfully connected"
    mock_get_config.assert_called_once_with("ollama")
    mock_get_value.assert_called_once_with("OLLAMA_BASE_URL")
    mock_test_func.assert_called_once_with("http://custom-url:11434")


@patch("src.utils.api_config.APIConfigManager.get_api_config_by_service")
@patch("src.utils.api_config.APITester")
def test_test_api_connection_no_key_needed(mock_api_tester, mock_get_config):
    """Test API connection testing for a service that doesn't require an API key."""
    # Setup mocks
    mock_get_config.return_value = {
        "name": "Yahoo Finance",
        "service_id": "yahoofinance",
        "needs_key": False
    }
    mock_test_func = MagicMock(return_value=(True, "Successfully connected"))
    mock_api_tester.test_yahoofinance = mock_test_func
    
    # Call the function
    success, message = APIConfigManager.test_api_connection("yahoofinance")
    
    # Verify
    assert success is True
    assert message == "Successfully connected"
    mock_get_config.assert_called_once_with("yahoofinance")
    mock_test_func.assert_called_once()


@patch("src.utils.api_config.APIConfigManager.get_api_config_by_service")
@patch("src.utils.api_config.APIConfigManager.get_api_value_from_env")
def test_test_api_connection_missing_required_key(mock_get_value, mock_get_config):
    """Test API connection testing when a required key is missing."""
    # Setup mocks
    mock_get_config.return_value = {
        "name": "OpenAI",
        "service_id": "openai",
        "needs_key": True,
        "env_var_key": "OPENAI_API_KEY"
    }
    mock_get_value.return_value = ""  # Empty key
    
    # Call the function
    success, message = APIConfigManager.test_api_connection("openai")
    
    # Verify
    assert success is False
    assert "API key required" in message
    mock_get_config.assert_called_once_with("openai")
    mock_get_value.assert_called_once_with("OPENAI_API_KEY")


@patch("src.utils.api_config.APIConfigManager.get_api_config_by_service")
@patch("src.utils.api_config.APIConfigManager.get_api_value_from_env")
def test_test_api_connection_missing_required_secret(mock_get_value, mock_get_config):
    """Test API connection testing when a required secret is missing."""
    # Setup mocks
    mock_get_config.return_value = {
        "name": "Binance",
        "service_id": "binance",
        "needs_key": True,
        "needs_secret": True,
        "env_var_key": "BINANCE_API_KEY",
        "env_var_secret": "BINANCE_SECRET_KEY"
    }
    mock_get_value.side_effect = ["valid_key", ""]  # Valid key, empty secret
    
    # Call the function
    success, message = APIConfigManager.test_api_connection("binance")
    
    # Verify
    assert success is False
    assert "API secret required" in message
    mock_get_config.assert_called_once_with("binance")
    assert mock_get_value.call_count == 2


@patch("src.utils.api_config.APIConfigManager.get_api_config_by_service")
@patch("src.utils.api_config.APIConfigManager.get_api_value_from_env")
@patch("src.utils.api_config.APITester")
def test_test_api_connection_with_key_and_secret(mock_api_tester, mock_get_value, mock_get_config):
    """Test API connection testing for a service requiring key and secret."""
    # Setup mocks
    mock_get_config.return_value = {
        "name": "Binance",
        "service_id": "binance",
        "needs_key": True,
        "needs_secret": True,
        "env_var_key": "BINANCE_API_KEY",
        "env_var_secret": "BINANCE_SECRET_KEY"
    }
    mock_get_value.side_effect = ["valid_key", "valid_secret"]
    mock_test_func = MagicMock(return_value=(True, "Successfully connected"))
    mock_api_tester.test_binance = mock_test_func
    
    # Call the function
    success, message = APIConfigManager.test_api_connection("binance")
    
    # Verify
    assert success is True
    assert message == "Successfully connected"
    mock_get_config.assert_called_once_with("binance")
    assert mock_get_value.call_count == 2
    mock_test_func.assert_called_once_with("valid_key", "valid_secret")


@patch("src.utils.api_config.APIConfigManager.get_api_config_by_service")
@patch("src.utils.api_config.APIConfigManager.update_env_file")
def test_save_api_credentials_successful(mock_update_env, mock_get_config):
    """Test saving API credentials successfully."""
    # Setup mocks
    mock_get_config.return_value = {
        "name": "OpenAI",
        "service_id": "openai",
        "needs_key": True,
        "env_var_key": "OPENAI_API_KEY"
    }
    mock_update_env.return_value = True
    
    # No need to create db_mock as we won't be storing in DB now
    # Call the function (omitting db and user_id args)
    success, message = APIConfigManager.save_api_credentials("openai", "new_key", None)
    
    # Verify
    assert success is True
    assert "Successfully" in message
    mock_get_config.assert_called_once_with("openai")
    mock_update_env.assert_called_once_with("OPENAI_API_KEY", "new_key")


@patch("src.utils.api_config.APIConfigManager.get_api_config_by_service")
def test_save_api_credentials_nonexistent(mock_get_config):
    """Test saving API credentials for a nonexistent service."""
    # Setup mock
    mock_get_config.return_value = None
    
    # Call the function
    success, message = APIConfigManager.save_api_credentials("nonexistent", "key", "secret")
    
    # Verify
    assert success is False
    assert "Unknown service" in message
    mock_get_config.assert_called_once_with("nonexistent")


@patch("src.utils.api_config.APIConfigManager.get_api_config_by_service")
@patch("src.utils.api_config.APIConfigManager.update_env_file")
def test_save_api_credentials_env_failure(mock_update_env, mock_get_config):
    """Test saving API credentials with environment file update failure."""
    # Setup mocks
    mock_get_config.return_value = {
        "name": "OpenAI",
        "service_id": "openai",
        "needs_key": True,
        "env_var_key": "OPENAI_API_KEY"
    }
    mock_update_env.return_value = False  # Update fails
    
    # Call the function
    success, message = APIConfigManager.save_api_credentials("openai", "new_key")
    
    # Verify
    assert success is False
    assert "Failed to update" in message
    mock_get_config.assert_called_once_with("openai")
    mock_update_env.assert_called_once_with("OPENAI_API_KEY", "new_key")


# Database storage functionality is no longer supported,
# so we don't need to test database failure case