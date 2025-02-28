"""Tests for OAuth configuration and functionality."""
import json
import os
from datetime import datetime, timedelta
from unittest.mock import MagicMock, PropertyMock, patch

import jwt
import pytest
import requests
from authlib.integrations.requests_client import OAuth2Session

from src.models.database import User
from src.utils.oauth_config import (
    OAUTH_CONFIGS,
    create_or_update_oauth_user,
    exchange_code_for_token,
    generate_oauth_authorize_url,
    get_oauth_access_token,
    get_oauth_client,
    get_user_info,
    refresh_oauth_token,
)


# Test the OAuth provider configurations
def test_oauth_configs_structure():
    """Test that all OAuth provider configurations have the required keys."""
    required_keys = [
        "client_id",
        "client_secret",
        "authorize_url",
        "token_url",
        "userinfo_url",
        "scope",
        "redirect_uri",
        "icon",
        "color",
        "display_name",
    ]

    for provider, config in OAUTH_CONFIGS.items():
        for key in required_keys:
            assert key in config, f"'{key}' missing for provider '{provider}'"


# Mock environment variables for testing
@pytest.fixture
def mock_oauth_env():
    """Mock environment variables for OAuth testing."""
    with patch.dict(
        os.environ,
        {
            "GOOGLE_CLIENT_ID": "google-test-client-id",
            "GOOGLE_CLIENT_SECRET": "google-test-client-secret",
            "COINBASE_CLIENT_ID": "coinbase-test-client-id",
            "COINBASE_CLIENT_SECRET": "coinbase-test-client-secret",
            "FACEBOOK_CLIENT_ID": "facebook-test-client-id",
            "FACEBOOK_CLIENT_SECRET": "facebook-test-client-secret",
            "TWITTER_CLIENT_ID": "twitter-test-client-id",
            "TWITTER_CLIENT_SECRET": "twitter-test-client-secret",
            "GITHUB_CLIENT_ID": "github-test-client-id",
            "GITHUB_CLIENT_SECRET": "github-test-client-secret",
            "MICROSOFT_CLIENT_ID": "microsoft-test-client-id",
            "MICROSOFT_CLIENT_SECRET": "microsoft-test-client-secret",
        },
    ):
        # Create a fresh module with these environment variables
        with patch(
            "src.utils.oauth_config.OAUTH_CONFIGS",
            {
                "google": {
                    "client_id": "google-test-client-id",
                    "client_secret": "google-test-client-secret",
                    "authorize_url": "https://accounts.google.com/o/oauth2/auth",
                    "token_url": "https://oauth2.googleapis.com/token",
                    "userinfo_url": "https://www.googleapis.com/oauth2/v3/userinfo",
                    "scope": "openid email profile",
                    "redirect_uri": "http://localhost:8501/callback/google",
                    "icon": "google",
                    "color": "#DB4437",
                    "display_name": "Google",
                },
                "coinbase": {
                    "client_id": "coinbase-test-client-id",
                    "client_secret": "coinbase-test-client-secret",
                    "authorize_url": "https://www.coinbase.com/oauth/authorize",
                    "token_url": "https://api.coinbase.com/oauth/token",
                    "userinfo_url": "https://api.coinbase.com/v2/user",
                    "scope": "wallet:user:read,wallet:accounts:read",
                    "redirect_uri": "http://localhost:8501/callback/coinbase",
                    "icon": "bitcoin",
                    "color": "#0052FF",
                    "display_name": "Coinbase",
                },
                "github": {
                    "client_id": "github-test-client-id",
                    "client_secret": "github-test-client-secret",
                    "authorize_url": "https://github.com/login/oauth/authorize",
                    "token_url": "https://github.com/login/oauth/access_token",
                    "userinfo_url": "https://api.github.com/user",
                    "scope": "read:user user:email",
                    "redirect_uri": "http://localhost:8501/callback/github",
                    "icon": "github",
                    "color": "#24292E",
                    "display_name": "GitHub",
                }
                # Add other providers as needed for test cases
            },
        ):
            yield


# Test get_oauth_client function
def test_get_oauth_client(mock_oauth_env):
    """Test that get_oauth_client returns the correct client for valid providers."""
    # Test with valid provider
    client = get_oauth_client("google")
    assert client is not None
    assert isinstance(client, OAuth2Session)

    # Test with invalid provider
    client = get_oauth_client("invalid_provider")
    assert client is None


# Test generate_oauth_authorize_url function
def test_generate_oauth_authorize_url(mock_oauth_env):
    """Test that generate_oauth_authorize_url returns the correct URL and state."""
    # Test with valid provider
    url, state = generate_oauth_authorize_url("google")
    assert url is not None
    assert url.startswith("https://accounts.google.com/o/oauth2/auth")
    assert "client_id=google-test-client-id" in url
    assert "redirect_uri=" in url
    assert "state=" in url
    assert state is not None

    # Test with provided state
    test_state = "test-state-123"
    url, returned_state = generate_oauth_authorize_url("google", test_state)
    assert returned_state == test_state
    assert f"state={test_state}" in url

    # Test with invalid provider
    url, state = generate_oauth_authorize_url("invalid_provider")
    assert url is None
    assert state is None


# Test exchange_code_for_token function
@patch("src.utils.oauth_config.OAuth2Session")
def test_exchange_code_for_token(mock_oauth_session, mock_oauth_env):
    """Test that exchange_code_for_token correctly exchanges a code for a token."""
    # Mock the OAuth2Session instance and its fetch_token method
    mock_instance = MagicMock()
    mock_oauth_session.return_value = mock_instance
    mock_instance.fetch_token.return_value = {
        "access_token": "test-access-token",
        "refresh_token": "test-refresh-token",
        "expires_in": 3600,
    }

    # Test with valid provider and code
    token = exchange_code_for_token("google", "test-code")
    assert token is not None
    assert token["access_token"] == "test-access-token"
    assert token["refresh_token"] == "test-refresh-token"
    assert token["expires_in"] == 3600

    # Verify the mock was called with the correct parameters
    mock_oauth_session.assert_called_once_with(
        client_id="google-test-client-id",
        client_secret="google-test-client-secret",
        redirect_uri="http://localhost:8501/callback/google",
    )
    mock_instance.fetch_token.assert_called_once_with(
        url="https://oauth2.googleapis.com/token",
        code="test-code",
        client_id="google-test-client-id",
        client_secret="google-test-client-secret",
    )

    # Test with exception
    mock_instance.fetch_token.side_effect = Exception("Test error")
    token = exchange_code_for_token("google", "test-code")
    assert token is None


# Test get_user_info function for different providers
@patch("src.utils.oauth_config.requests.get")
def test_get_user_info_google(mock_get, mock_oauth_env):
    """Test get_user_info function for Google."""
    # Mock the response for Google
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "sub": "123456789",
        "email": "test@example.com",
        "name": "Test User",
        "picture": "https://example.com/picture.jpg",
    }
    mock_get.return_value = mock_response

    # Test with Google provider
    user_info = get_user_info("google", "test-access-token")
    assert user_info is not None
    assert user_info["id"] == "123456789"
    assert user_info["email"] == "test@example.com"
    assert user_info["name"] == "Test User"
    assert user_info["picture"] == "https://example.com/picture.jpg"

    # Verify the mock was called with the correct parameters
    mock_get.assert_called_once_with(
        "https://www.googleapis.com/oauth2/v3/userinfo",
        headers={"Authorization": "Bearer test-access-token"},
    )


@patch("src.utils.oauth_config.requests.get")
def test_get_user_info_coinbase(mock_get, mock_oauth_env):
    """Test get_user_info function for Coinbase."""
    # Mock the response for Coinbase
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "data": {
            "id": "123456789",
            "email": "test@example.com",
            "name": "Test User",
            "username": "testuser",
        }
    }
    mock_get.return_value = mock_response

    # Test with Coinbase provider
    user_info = get_user_info("coinbase", "test-access-token")
    assert user_info is not None
    assert user_info["id"] == "123456789"
    assert user_info["email"] == "test@example.com"
    assert user_info["name"] == "Test User"
    assert user_info["username"] == "testuser"


@patch("src.utils.oauth_config.requests.get")
def test_get_user_info_github(mock_get, mock_oauth_env):
    """Test get_user_info function for GitHub."""
    # First mock response for the user info
    first_response = MagicMock()
    first_response.status_code = 200
    first_response.json.return_value = {
        "id": 123456789,
        "login": "testuser",
        "name": "Test User",
        "email": None,  # GitHub might not return email if not public
        "avatar_url": "https://example.com/avatar.jpg",
    }

    # Second mock response for the email endpoint
    second_response = MagicMock()
    second_response.status_code = 200
    second_response.json.return_value = [
        {"email": "private@example.com", "primary": True, "verified": True},
        {"email": "secondary@example.com", "primary": False, "verified": True},
    ]

    # Configure the mock to return different responses for different URLs
    def side_effect(url, **kwargs):
        if url == "https://api.github.com/user":
            return first_response
        elif url == "https://api.github.com/user/emails":
            return second_response
        return MagicMock(status_code=404)

    mock_get.side_effect = side_effect

    # Test with GitHub provider
    user_info = get_user_info("github", "test-access-token")
    assert user_info is not None
    assert user_info["id"] == "123456789"  # Should be converted to string
    assert user_info["email"] == "private@example.com"  # Should get primary email
    assert user_info["name"] == "Test User"
    assert user_info["username"] == "testuser"
    assert user_info["picture"] == "https://example.com/avatar.jpg"

    # Verify the mock was called with the correct parameters
    assert mock_get.call_count == 2
    mock_get.assert_any_call(
        "https://api.github.com/user",
        headers={
            "Authorization": "Bearer test-access-token",
            "Accept": "application/vnd.github.v3+json",
        },
    )
    mock_get.assert_any_call(
        "https://api.github.com/user/emails",
        headers={
            "Authorization": "Bearer test-access-token",
            "Accept": "application/vnd.github.v3+json",
        },
    )


# Test create_or_update_oauth_user function
@patch("src.utils.oauth_config.encrypt_data")
@patch("src.utils.oauth_config.SessionLocal")
def test_create_or_update_oauth_user_new_user(
    mock_session_local, mock_encrypt, mock_oauth_env
):
    """Test create_or_update_oauth_user creates a new user when none exists."""
    # Mock the database session
    mock_db = MagicMock()
    mock_session_local.return_value = mock_db

    # Mock the query and filter results (no existing user)
    mock_query = MagicMock()
    mock_db.query.return_value = mock_query
    mock_filter = MagicMock()
    mock_query.filter.return_value = mock_filter
    mock_filter.first.return_value = None

    # Mock the encrypt function
    mock_encrypt.side_effect = lambda x: f"encrypted_{x}"

    # Test user info and token data
    user_info = {
        "id": "123456789",
        "email": "test@example.com",
        "name": "Test User",
        "username": "testuser",
    }
    token_data = {
        "access_token": "test-access-token",
        "refresh_token": "test-refresh-token",
        "expires_in": 3600,
    }

    # Call the function
    user = create_or_update_oauth_user("google", user_info, token_data)

    # Verify a new user was created
    assert mock_db.add.called
    assert mock_db.commit.called
    assert mock_db.refresh.called

    # Verify the user object properties
    created_user = mock_db.add.call_args[0][0]
    assert created_user.username == "testuser"
    assert created_user.email == "test@example.com"
    assert created_user.oauth_provider == "google"
    assert created_user.oauth_id == "123456789"
    assert created_user.oauth_access_token == "encrypted_test-access-token"
    assert created_user.oauth_refresh_token == "encrypted_test-refresh-token"


@patch("src.utils.oauth_config.encrypt_data")
@patch("src.utils.oauth_config.SessionLocal")
def test_create_or_update_oauth_user_existing_user(
    mock_session_local, mock_encrypt, mock_oauth_env
):
    """Test create_or_update_oauth_user updates an existing user."""
    # Mock the database session
    mock_db = MagicMock()
    mock_session_local.return_value = mock_db

    # Mock the query and filter results (existing user)
    mock_query = MagicMock()
    mock_db.query.return_value = mock_query
    mock_filter = MagicMock()
    mock_query.filter.return_value = mock_filter

    # Create an existing user
    existing_user = MagicMock()
    existing_user.id = 1
    existing_user.username = "existinguser"
    existing_user.email = "existing@example.com"
    mock_filter.first.return_value = existing_user

    # Mock the encrypt function
    mock_encrypt.side_effect = lambda x: f"encrypted_{x}"

    # Test user info and token data
    user_info = {
        "id": "123456789",
        "email": "test@example.com",
        "name": "Test User",
        "username": "testuser",
    }
    token_data = {
        "access_token": "test-access-token",
        "refresh_token": "test-refresh-token",
        "expires_in": 3600,
    }

    # Call the function
    user = create_or_update_oauth_user("google", user_info, token_data)

    # Verify the user was updated, not created
    assert not mock_db.add.called
    assert mock_db.commit.called
    assert mock_db.refresh.called

    # Verify the user object properties were updated
    assert existing_user.oauth_provider == "google"
    assert existing_user.oauth_id == "123456789"
    assert existing_user.oauth_access_token == "encrypted_test-access-token"
    assert existing_user.oauth_refresh_token == "encrypted_test-refresh-token"


# Test refresh_oauth_token function
@patch("src.utils.oauth_config.decrypt_data")
@patch("src.utils.oauth_config.encrypt_data")
@patch("src.utils.oauth_config.OAuth2Session")
@patch("src.utils.oauth_config.SessionLocal")
def test_refresh_oauth_token(
    mock_session_local, mock_oauth_session, mock_encrypt, mock_decrypt, mock_oauth_env
):
    """Test that refresh_oauth_token correctly refreshes an OAuth token."""
    # Mock the database session
    mock_db = MagicMock()
    mock_session_local.return_value = mock_db

    # Mock the OAuth2Session instance and its refresh_token method
    mock_instance = MagicMock()
    mock_oauth_session.return_value = mock_instance
    mock_instance.refresh_token.return_value = {
        "access_token": "new-access-token",
        "refresh_token": "new-refresh-token",
        "expires_in": 3600,
    }

    # Mock decrypt function
    mock_decrypt.return_value = "decrypted-refresh-token"

    # Mock encrypt function
    mock_encrypt.side_effect = lambda x: f"encrypted_{x}"

    # Create a mock user
    mock_user = MagicMock()
    mock_user.oauth_provider = "google"
    mock_user.oauth_refresh_token = "encrypted-refresh-token"

    # Call the function
    result = refresh_oauth_token(mock_user)

    # Verify the result
    assert result is True

    # Verify the mock calls
    mock_decrypt.assert_called_once_with("encrypted-refresh-token")
    mock_oauth_session.assert_called_once_with(
        client_id="google-test-client-id", client_secret="google-test-client-secret"
    )
    mock_instance.refresh_token.assert_called_once_with(
        url="https://oauth2.googleapis.com/token",
        refresh_token="decrypted-refresh-token",
        client_id="google-test-client-id",
        client_secret="google-test-client-secret",
    )

    # Verify the user object was updated
    assert mock_user.oauth_access_token == "encrypted_new-access-token"
    assert mock_user.oauth_refresh_token == "encrypted_new-refresh-token"
    assert mock_db.commit.called


# Test get_oauth_access_token function
@patch("src.utils.oauth_config.refresh_oauth_token")
@patch("src.utils.oauth_config.decrypt_data")
def test_get_oauth_access_token_valid(mock_decrypt, mock_refresh, mock_oauth_env):
    """Test get_oauth_access_token with a valid token."""
    # Mock user with a valid token
    mock_user = MagicMock()
    mock_user.oauth_access_token = "encrypted-access-token"
    mock_user.oauth_token_expiry = datetime.utcnow() + timedelta(hours=1)  # Not expired

    # Mock decrypt function
    mock_decrypt.return_value = "decrypted-access-token"

    # Call the function
    token = get_oauth_access_token(mock_user)

    # Verify the result
    assert token == "decrypted-access-token"
    mock_decrypt.assert_called_once_with("encrypted-access-token")
    mock_refresh.assert_not_called()


@patch("src.utils.oauth_config.refresh_oauth_token")
@patch("src.utils.oauth_config.decrypt_data")
def test_get_oauth_access_token_expired(mock_decrypt, mock_refresh, mock_oauth_env):
    """Test get_oauth_access_token with an expired token that needs refresh."""
    # Mock user with an expired token
    mock_user = MagicMock()
    mock_user.oauth_access_token = "encrypted-access-token"
    mock_user.oauth_token_expiry = datetime.utcnow() - timedelta(hours=1)  # Expired

    # Mock decrypt function
    mock_decrypt.return_value = "decrypted-access-token"

    # Mock refresh function to simulate successful refresh
    mock_refresh.return_value = True

    # Call the function
    token = get_oauth_access_token(mock_user)

    # Verify the result
    assert token == "decrypted-access-token"
    mock_decrypt.assert_called_once_with("encrypted-access-token")
    mock_refresh.assert_called_once_with(mock_user)


@patch("src.utils.oauth_config.refresh_oauth_token")
@patch("src.utils.oauth_config.decrypt_data")
def test_get_oauth_access_token_refresh_failed(
    mock_decrypt, mock_refresh, mock_oauth_env
):
    """Test get_oauth_access_token when token refresh fails."""
    # Mock user with an expired token
    mock_user = MagicMock()
    mock_user.oauth_access_token = "encrypted-access-token"
    mock_user.oauth_token_expiry = datetime.utcnow() - timedelta(hours=1)  # Expired

    # Mock refresh function to simulate failed refresh
    mock_refresh.return_value = False

    # Call the function
    token = get_oauth_access_token(mock_user)

    # Verify the result
    assert token is None
    mock_refresh.assert_called_once_with(mock_user)
    mock_decrypt.assert_not_called()  # Shouldn't decrypt if refresh failed
