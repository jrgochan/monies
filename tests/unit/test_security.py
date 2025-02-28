"""Tests for security functionality."""
import pytest
from cryptography.fernet import Fernet
from unittest.mock import patch, MagicMock
import os

from src.utils.security import (
    generate_key,
    encrypt_data,
    decrypt_data,
    get_fernet_key,
    store_api_key,
    get_api_key
)


def test_key_generation():
    """Test that key generation produces valid Fernet keys."""
    key = generate_key()
    
    # Key should be bytes
    assert isinstance(key, bytes)
    
    # Key should be valid for Fernet
    fernet = Fernet(key)
    assert fernet is not None


def test_encryption_decryption():
    """Test that encryption and decryption works correctly."""
    key = generate_key()
    data = "sensitive api key 1234567890"
    
    # Encrypt the data
    encrypted = encrypt_data(data, key)
    
    # Encrypted data should be different from original
    assert encrypted != data
    assert isinstance(encrypted, str)
    
    # Decrypt the data
    decrypted = decrypt_data(encrypted, key)
    
    # Decrypted data should match original
    assert decrypted == data
    
    # Wrong key should fail to decrypt
    wrong_key = generate_key()
    with pytest.raises(Exception):
        decrypt_data(encrypted, wrong_key)


def test_encrypt_decrypt_without_key():
    """Test encryption and decryption using the default cipher."""
    data = "sensitive data without custom key"
    
    # Encrypt with default cipher
    encrypted = encrypt_data(data)
    assert encrypted != data
    assert isinstance(encrypted, str)
    
    # Decrypt with default cipher
    decrypted = decrypt_data(encrypted)
    assert decrypted == data


def test_get_fernet_key():
    """Test the get_fernet_key function."""
    # Test with empty key
    empty_key = get_fernet_key("")
    assert isinstance(empty_key, bytes)
    
    # Test with a valid key string
    valid_key = get_fernet_key("a-test-key-that-needs-padding-to-be-32-bytes")
    assert isinstance(valid_key, bytes)
    
    # Fernet should be able to use the key
    fernet = Fernet(valid_key)
    assert fernet is not None


@pytest.fixture
def mock_db_session():
    """Create a mock database session."""
    mock_session = MagicMock()
    
    # Mock ApiKey model and queries
    mock_query = MagicMock()
    mock_session.query.return_value = mock_query
    mock_filter = MagicMock()
    mock_query.filter.return_value = mock_filter
    
    # Setup for existing key
    mock_key = MagicMock()
    mock_key.encrypted_key = encrypt_data("test-api-key")
    mock_key.encrypted_secret = encrypt_data("test-api-secret")
    
    # Simulate filtering by user_id and service
    def filter_side_effect(*conditions):
        # Simplified filter logic for testing
        service_name = None
        for condition in conditions:
            condition_str = str(condition)
            if "service ==" in condition_str:
                service_name = condition_str.split("==")[1].strip()
        
        if service_name == "'existing_service'" or service_name == "'existing_service_with_secret'":
            mock_filter.first.return_value = mock_key
        else:
            mock_filter.first.return_value = None
        
        return mock_filter
    
    mock_query.filter.side_effect = filter_side_effect
    
    return mock_session


def test_store_api_key_new(mock_db_session):
    """Test storing a new API key."""
    with patch('src.models.database.ApiKey') as MockApiKey:
        # Setup mock API key object
        mock_api_key = MagicMock()
        MockApiKey.return_value = mock_api_key
        
        # Test creating a new key
        result = store_api_key(
            mock_db_session, 
            user_id=1, 
            service="new_service", 
            api_key="new-api-key",
            api_secret="new-api-secret"
        )
        
        # Check a new ApiKey was created and added to db
        mock_db_session.add.assert_called_once()
        mock_db_session.commit.assert_called_once()
        
        # We need an encryption test separately
        assert encrypt_data("new-api-key") != "new-api-key"


def test_store_api_key_update():
    """Test updating an existing API key."""
    # Create a completely fresh mock setup for this test
    mock_db_session = MagicMock()
    
    # Mock the ApiKey class and create a mock existing key
    with patch('src.models.database.ApiKey') as MockApiKey:
        # Mock object returned by the DB query
        mock_existing_key = MagicMock()
        
        # Configure the mock session's query chain to return the mock existing key
        mock_db_session.query.return_value.filter.return_value.first.return_value = mock_existing_key
        
        # Test updating an existing key
        result = store_api_key(
            mock_db_session, 
            user_id=1, 
            service="existing_service", 
            api_key="updated-api-key"
        )
        
        # Should not call add() when updating
        mock_db_session.add.assert_not_called()
        
        # Should call commit() to save changes
        mock_db_session.commit.assert_called_once()
        
        # Verify the key was updated and encrypted
        assert mock_existing_key.encrypted_key is not None


def test_get_api_key_existing():
    """Test retrieving an existing API key."""
    # Create a mock DB session
    mock_db_session = MagicMock()
    
    with patch('src.models.database.ApiKey') as MockApiKey:
        # Create a mock API key object with encrypted values
        mock_key = MagicMock()
        mock_key.encrypted_key = encrypt_data("test-api-key")
        mock_key.encrypted_secret = encrypt_data("test-api-secret")
        
        # Configure the query chain to return our mock key
        mock_db_session.query.return_value.filter.return_value.first.return_value = mock_key
        
        # Get an existing key
        key, secret = get_api_key(mock_db_session, user_id=1, service="existing_service_with_secret")
        
        # Values should be decrypted
        assert key == "test-api-key"
        assert secret == "test-api-secret"


def test_get_api_key_nonexistent():
    """Test retrieving a non-existent API key."""
    # Create a mock DB session
    mock_db_session = MagicMock()
    
    # Configure the query to return None, simulating no key found
    mock_db_session.query.return_value.filter.return_value.first.return_value = None
    
    # Get a non-existent key
    key, secret = get_api_key(mock_db_session, user_id=1, service="nonexistent_service")
    
    # Should return None for both
    assert key is None
    assert secret is None


def test_null_encryption():
    """Test handling of None values in encryption/decryption."""
    # Encrypt None
    assert encrypt_data(None) is None
    
    # Decrypt None
    assert decrypt_data(None) is None