"""Unit tests for OAuth-related security features."""
import os
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest
from cryptography.fernet import Fernet

from src.models.database import ApiKey, SessionLocal, User
from src.utils.security import (
    decrypt_data,
    encrypt_data,
    get_api_key,
    get_api_keys_for_service,
    set_default_api_key,
    store_api_key,
    store_oauth_api_key,
)


# Fixtures for security testing
@pytest.fixture
def mock_crypto_key():
    """Generate a test encryption key."""
    return Fernet.generate_key()


@pytest.fixture
def mock_cipher(mock_crypto_key):
    """Create a Fernet cipher for testing."""
    return Fernet(mock_crypto_key)


@pytest.fixture
def mock_db_session():
    """Mock database session for testing."""
    db = MagicMock()
    with patch("src.utils.security.SessionLocal", return_value=db):
        yield db


@pytest.fixture
def mock_api_key():
    """Create a mock API key object."""
    api_key = MagicMock(spec=ApiKey)
    api_key.id = 1
    api_key.user_id = 123
    api_key.service = "coinbase"
    api_key.encrypted_key = "encrypted_key_value"
    api_key.encrypted_secret = "encrypted_secret_value"
    api_key.created_at = datetime.utcnow()
    api_key.is_oauth = True
    api_key.oauth_provider = "coinbase"
    api_key.is_default = True
    api_key.display_name = "Coinbase OAuth"
    return api_key


@pytest.fixture
def mock_user():
    """Create a mock user object."""
    user = MagicMock(spec=User)
    user.id = 123
    user.username = "testuser"
    user.email = "test@example.com"
    user.created_at = datetime.utcnow()
    user.oauth_provider = "coinbase"
    user.oauth_id = "coinbase_user_id"
    user.oauth_access_token = "encrypted_token_value"
    user.oauth_refresh_token = "encrypted_refresh_token_value"
    user.oauth_token_expiry = datetime.utcnow() + timedelta(hours=1)
    user.password_hash = None  # OAuth user with no password
    return user


# Test data encryption and decryption
def test_oauth_token_encryption_decryption(mock_crypto_key, mock_cipher):
    """Test encryption and decryption of OAuth tokens."""
    # Patch the cipher in the security module
    with patch("src.utils.security.cipher", mock_cipher):
        # Test data
        test_data = "oauth_access_token_value"

        # Encrypt data
        encrypted = encrypt_data(test_data)

        # Verify data was encrypted (should be different from original)
        assert encrypted != test_data
        assert isinstance(encrypted, str)

        # Decrypt data
        decrypted = decrypt_data(encrypted)

        # Verify decryption works and returns original value
        assert decrypted == test_data


# Test storing OAuth API keys
@patch("src.utils.security.encrypt_data")
def test_store_oauth_api_key(mock_encrypt, mock_db_session):
    """Test storing an OAuth API key."""
    # Mock encrypt function to return predictable values
    mock_encrypt.side_effect = lambda x: f"encrypted_{x}"

    # Mock database queries to simulate no existing key
    mock_query = MagicMock()
    mock_filter = MagicMock()
    mock_filter.first.return_value = None  # No existing key
    mock_query.filter.return_value = mock_filter
    mock_db_session.query.return_value = mock_query

    # Mock get_api_keys_for_service to return no keys
    with patch("src.utils.security.get_api_keys_for_service") as mock_get_keys:
        mock_get_keys.return_value = []  # No existing keys

        # Call store_oauth_api_key
        with patch("src.models.database.ApiKey", spec=ApiKey):
            result = store_oauth_api_key(
                mock_db_session,
                123,
                "coinbase",
                "access_token_123",
                "refresh_token_456",
                "coinbase",
                "My Coinbase API",
            )

    # Verify key was created and added to the database
    assert mock_db_session.add.called

    # Get the created key from the mock
    created_key = mock_db_session.add.call_args[0][0]

    # Verify key properties
    assert created_key.user_id == 123
    assert created_key.service == "coinbase"
    assert created_key.encrypted_key == "encrypted_access_token_123"
    assert created_key.encrypted_secret == "encrypted_refresh_token_456"
    assert created_key.is_oauth is True
    assert created_key.oauth_provider == "coinbase"
    assert created_key.is_default is True  # First key should be default
    assert created_key.display_name == "My Coinbase API"

    # Verify database was committed
    assert mock_db_session.commit.called
    assert mock_db_session.refresh.called


# Test retrieving API keys
@patch("src.utils.security.decrypt_data")
def test_get_api_key_with_oauth(mock_decrypt, mock_db_session, mock_api_key):
    """Test retrieving an OAuth API key."""
    # Mock decrypt function
    mock_decrypt.side_effect = lambda x: x.replace("encrypted_", "decrypted_")

    # Mock database queries to return our test key
    mock_query = MagicMock()
    mock_filter = MagicMock()
    mock_filter.first.return_value = mock_api_key  # Return our mock key
    mock_query.filter.return_value = mock_filter
    mock_db_session.query.return_value = mock_query

    # Call get_api_key
    with patch("src.models.database.ApiKey", spec=ApiKey):
        key, secret, api_key_obj = get_api_key(mock_db_session, 123, "coinbase")

    # Verify correct values were returned
    assert key == "decrypted_key_value"
    assert secret == "decrypted_secret_value"
    assert api_key_obj is mock_api_key

    # Verify decrypt was called with the encrypted values
    mock_decrypt.assert_any_call("encrypted_key_value")
    mock_decrypt.assert_any_call("encrypted_secret_value")


# Test setting a specific OAuth key as default
def test_set_default_oauth_api_key(mock_db_session, mock_api_key):
    """Test setting an OAuth API key as the default."""
    # Create a second key to test with
    second_key = MagicMock(spec=ApiKey)
    second_key.id = 2
    second_key.user_id = 123
    second_key.service = "coinbase"
    second_key.is_oauth = True
    second_key.oauth_provider = "coinbase"
    second_key.is_default = False

    # Mock query to return our second key
    mock_query = MagicMock()
    mock_filter = MagicMock()
    mock_filter.first.return_value = second_key
    mock_query.filter.return_value = mock_filter
    mock_db_session.query.return_value = mock_query

    # Call set_default_api_key
    with patch("src.models.database.ApiKey", spec=ApiKey):
        result = set_default_api_key(mock_db_session, 123, "coinbase", 2)

    # Verify update was called to clear defaults
    mock_db_session.query.assert_called()

    # Verify second key was updated to be default
    assert second_key.is_default is True

    # Verify database was committed
    assert mock_db_session.commit.called

    # Verify function returned success
    assert result is True


# Test retrieving all API keys for a service
def test_get_api_keys_for_service(mock_db_session, mock_api_key):
    """Test retrieving all API keys for a service, including OAuth keys."""
    # Create a regular API key to test with
    regular_key = MagicMock(spec=ApiKey)
    regular_key.id = 2
    regular_key.user_id = 123
    regular_key.service = "coinbase"
    regular_key.is_oauth = False
    regular_key.is_default = False

    # Mock query to return both keys
    mock_query = MagicMock()
    mock_filter = MagicMock()
    mock_filter.all.return_value = [mock_api_key, regular_key]
    mock_query.filter.return_value = mock_filter
    mock_db_session.query.return_value = mock_query

    # Call get_api_keys_for_service
    with patch("src.models.database.ApiKey", spec=ApiKey):
        keys = get_api_keys_for_service(mock_db_session, 123, "coinbase")

    # Verify correct keys were returned
    assert len(keys) == 2
    assert mock_api_key in keys
    assert regular_key in keys

    # Verify query was called with correct filters
    mock_query.filter.assert_called_once()


# Test storing a regular API key with token refresh
@patch("src.utils.security.encrypt_data")
def test_store_api_key_update_existing(mock_encrypt, mock_db_session, mock_api_key):
    """Test updating an existing API key with new credentials."""
    # Mock encrypt function
    mock_encrypt.side_effect = lambda x: f"encrypted_{x}"

    # Mock database queries to return our test key
    mock_query = MagicMock()
    mock_filter = MagicMock()
    mock_filter.first.return_value = mock_api_key  # Return our mock key
    mock_query.filter.return_value = mock_filter
    mock_db_session.query.return_value = mock_query

    # Call store_api_key to update existing key
    with patch("src.models.database.ApiKey", spec=ApiKey):
        result = store_api_key(
            mock_db_session, 123, "coinbase", "new_access_token", "new_refresh_token"
        )

    # Verify key was updated, not added
    assert not mock_db_session.add.called

    # Verify key properties were updated
    assert mock_api_key.encrypted_key == "encrypted_new_access_token"
    assert mock_api_key.encrypted_secret == "encrypted_new_refresh_token"

    # Verify database was committed
    assert mock_db_session.commit.called

    # Verify correct key was returned
    assert result is mock_api_key
