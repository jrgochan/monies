"""Integration tests for OAuth functionality across the application."""
import json
import os
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import jwt
import pytest
import requests
import streamlit as st
from authlib.integrations.requests_client import OAuth2Session

from src.models.database import ApiKey, SessionLocal, User
from src.utils.auth import handle_oauth_callback
from src.utils.oauth_config import (
    OAUTH_CONFIGS,
    create_api_keys_from_oauth,
    create_or_update_oauth_user,
    exchange_code_for_token,
    generate_oauth_authorize_url,
    get_oauth_access_token,
    get_oauth_client,
    get_user_info,
    refresh_oauth_token,
)
from src.utils.security import decrypt_data, encrypt_data, store_oauth_api_key


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
        # Create fresh configs with these environment variables
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
                    "supports_api_keys": False,
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
                    "supports_api_keys": True,
                    "api_services": ["coinbase"],
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
                    "supports_api_keys": True,
                    "api_services": ["github"],
                },
                "facebook": {
                    "client_id": "facebook-test-client-id",
                    "client_secret": "facebook-test-client-secret",
                    "authorize_url": "https://www.facebook.com/v16.0/dialog/oauth",
                    "token_url": "https://graph.facebook.com/v16.0/oauth/access_token",
                    "userinfo_url": "https://graph.facebook.com/me?fields=id,name,email,picture",
                    "scope": "email,public_profile",
                    "redirect_uri": "http://localhost:8501/callback/facebook",
                    "icon": "facebook",
                    "color": "#1877F2",
                    "display_name": "Facebook",
                    "supports_api_keys": True,
                    "api_services": ["facebook"],
                },
            },
        ):
            yield


# Create fixture for mocked database session
@pytest.fixture
def mock_db_session():
    """Mock database session for testing."""
    db = MagicMock()
    with patch("src.utils.oauth_config.SessionLocal", return_value=db):
        yield db


# Create fixture for a mock user object
@pytest.fixture
def mock_user():
    """Create a mock user object for testing."""
    user = MagicMock(spec=User)
    user.id = 123
    user.username = "testuser"
    user.email = "test@example.com"
    user.oauth_provider = "google"
    user.oauth_id = "123456789"
    user.oauth_access_token = encrypt_data("test-access-token")
    user.oauth_refresh_token = encrypt_data("test-refresh-token")
    user.oauth_token_expiry = datetime.utcnow() + timedelta(hours=1)
    user.created_at = datetime.utcnow()
    user.updated_at = datetime.utcnow()
    return user


# Create fixture for streamlit session state
@pytest.fixture
def mock_session_state():
    """Mock Streamlit session state for testing."""

    class SessionState(dict):
        def __getattr__(self, key):
            if key in self:
                return self[key]
            raise AttributeError(f"'SessionState' object has no attribute '{key}'")

        def __setattr__(self, key, value):
            self[key] = value

        def __delattr__(self, key):
            if key in self:
                del self[key]
            else:
                raise AttributeError(f"'SessionState' object has no attribute '{key}'")

    # Create session state with OAuth data
    session_state = SessionState({"oauth_state": "test-state", "oauth_flow": "google"})

    with patch("src.utils.auth.st.session_state", session_state):
        yield session_state


# Integration test for the complete OAuth authentication flow
@patch("src.utils.oauth_config.exchange_code_for_token")
@patch("src.utils.oauth_config.get_user_info")
@patch("src.utils.oauth_config.create_or_update_oauth_user")
@patch("src.utils.auth.st")
def test_complete_oauth_flow(
    mock_st,
    mock_create_user,
    mock_get_user_info,
    mock_exchange_code,
    mock_session_state,
    mock_oauth_env,
):
    """Test the complete OAuth authentication flow from callback to user creation."""
    # Mock query parameters
    mock_st.query_params = {
        "code": ["test-code"],
        "state": ["test-state"],
    }

    # Mock token data
    mock_exchange_code.return_value = {
        "access_token": "test-access-token",
        "refresh_token": "test-refresh-token",
        "expires_in": 3600,
    }

    # Mock user info
    mock_get_user_info.return_value = {
        "id": "123456789",
        "email": "test@example.com",
        "name": "Test User",
        "username": "testuser",
    }

    # Mock user creation result
    mock_user = MagicMock()
    mock_user.id = 123
    mock_user.username = "testuser"
    mock_user.email = "test@example.com"
    mock_user.oauth_provider = "google"
    mock_create_user.return_value = mock_user

    # Call the OAuth callback handler
    handle_oauth_callback()

    # Verify token exchange
    mock_exchange_code.assert_called_once_with("google", "test-code")

    # Verify user info retrieval
    mock_get_user_info.assert_called_once_with("google", "test-access-token")

    # Verify user creation/update
    mock_create_user.assert_called_once_with(
        "google", mock_get_user_info.return_value, mock_exchange_code.return_value
    )

    # Verify session state updates
    assert "user" in mock_st.session_state
    assert "token" in mock_st.session_state
    assert "oauth_state" not in mock_st.session_state
    assert "oauth_flow" not in mock_st.session_state

    # Verify redirect
    mock_st.markdown.assert_called_once()
    mock_st.stop.assert_called_once()


# Test token refresh flow with API key updates
@patch("src.utils.oauth_config.decrypt_data")
@patch("src.utils.oauth_config.encrypt_data")
@patch("src.utils.oauth_config.OAuth2Session")
def test_token_refresh_with_api_key_updates(
    mock_oauth_session,
    mock_encrypt,
    mock_decrypt,
    mock_db_session,
    mock_user,
    mock_oauth_env,
):
    """Test OAuth token refresh with associated API key updates."""
    # Setup API key in database
    api_key = MagicMock(spec=ApiKey)
    api_key.id = 1
    api_key.user_id = mock_user.id
    api_key.service = "coinbase"
    api_key.is_oauth = True
    api_key.oauth_provider = "coinbase"
    api_key.encrypted_key = encrypt_data("old-access-token")
    api_key.encrypted_secret = encrypt_data("old-refresh-token")

    # Setup mocks for database queries
    mock_query = MagicMock()
    mock_filter = MagicMock()
    mock_filter.all.return_value = [api_key]
    mock_query.filter.return_value = mock_filter
    mock_db_session.query.return_value = mock_query

    # Setup OAuth refresh mock
    mock_oauth_instance = MagicMock()
    mock_oauth_session.return_value = mock_oauth_instance
    mock_oauth_instance.refresh_token.return_value = {
        "access_token": "new-access-token",
        "refresh_token": "new-refresh-token",
        "expires_in": 7200,
    }

    # Setup decrypt/encrypt mocks
    mock_decrypt.return_value = "decrypted-refresh-token"
    mock_encrypt.side_effect = lambda x: f"encrypted_{x}"

    # Update user for this test
    mock_user.oauth_provider = "coinbase"

    # Execute token refresh
    with patch("src.models.database.ApiKey", spec=ApiKey):
        result = refresh_oauth_token(mock_user)

    # Verify token refresh was successful
    assert result is True

    # Verify user tokens were updated
    assert mock_user.oauth_access_token == "encrypted_new-access-token"
    assert mock_user.oauth_refresh_token == "encrypted_new-refresh-token"

    # Verify API key was also updated
    assert api_key.encrypted_key == "encrypted_new-access-token"
    assert api_key.encrypted_secret == "encrypted_new-refresh-token"

    # Verify database was committed
    mock_db_session.commit.assert_called_once()


# Test API key creation from OAuth tokens
@patch("src.utils.oauth_config.get_oauth_access_token")
@patch("src.utils.security.store_oauth_api_key")
def test_create_api_keys_from_oauth(
    mock_store_oauth_key, mock_get_token, mock_db_session, mock_user, mock_oauth_env
):
    """Test creating API keys from OAuth tokens."""
    # Setup the user's provider
    mock_user.oauth_provider = "coinbase"

    # Setup token retrieval mock
    mock_get_token.return_value = "test-access-token"

    # Setup decrypt for refresh token
    with patch("src.utils.oauth_config.decrypt_data") as mock_decrypt:
        mock_decrypt.return_value = "test-refresh-token"

        # Execute API key creation
        results = create_api_keys_from_oauth(mock_user, mock_db_session)

    # Verify API key was created
    mock_store_oauth_key.assert_called_once_with(
        mock_db_session,
        mock_user.id,
        "coinbase",
        "test-access-token",
        "test-refresh-token",
        "coinbase",
        "Coinbase Coinbase",
    )

    # Verify results
    assert "coinbase" in results
    assert results["coinbase"] is True


# Test OAuth user creation and API key setup
@patch("src.utils.oauth_config.encrypt_data")
@patch("src.utils.oauth_config.create_api_keys_from_oauth")
def test_create_or_update_oauth_user_with_api_keys(
    mock_create_api_keys, mock_encrypt, mock_db_session, mock_oauth_env
):
    """Test creating a new OAuth user and setting up API keys."""
    # Mock the database query to return no existing user
    mock_query = MagicMock()
    mock_filter = MagicMock()
    mock_filter.first.return_value = None  # No existing user
    mock_query.filter.return_value = mock_filter
    mock_db_session.query.return_value = mock_query

    # Mock encrypt function
    mock_encrypt.side_effect = lambda x: f"encrypted_{x}"

    # User info and token data
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

    # Execute user creation
    with patch("src.models.database.User", spec=User):
        user = create_or_update_oauth_user("coinbase", user_info, token_data)

    # Verify user was created
    assert mock_db_session.add.called

    # Verify API keys were created
    mock_create_api_keys.assert_called_once()

    # Get the created user from the mock
    created_user = mock_db_session.add.call_args[0][0]

    # Verify user properties
    assert created_user.username == "testuser"
    assert created_user.email == "test@example.com"
    assert created_user.oauth_provider == "coinbase"
    assert created_user.oauth_id == "123456789"
    assert created_user.oauth_access_token == "encrypted_test-access-token"
    assert created_user.oauth_refresh_token == "encrypted_test-refresh-token"


# Test API key creation with OAuth tokens
@patch("src.utils.security.encrypt_data")
def test_store_oauth_api_key(mock_encrypt, mock_db_session, mock_oauth_env):
    """Test creating API keys from OAuth tokens."""
    # Mock encrypt function
    mock_encrypt.side_effect = lambda x: f"encrypted_{x}"

    # Setup an existing API key in the database
    api_key = MagicMock(spec=ApiKey)
    api_key.id = 1
    api_key.user_id = 123
    api_key.service = "github"
    api_key.is_oauth = True
    api_key.oauth_provider = "github"

    # Mock query for existing key
    mock_query = MagicMock()
    mock_filter = MagicMock()
    mock_filter.first.return_value = api_key  # Existing key
    mock_query.filter.return_value = mock_filter
    mock_db_session.query.return_value = mock_query

    # Execute API key storage (updating existing key)
    result = store_oauth_api_key(
        mock_db_session,
        123,
        "github",
        "new-access-token",
        "new-refresh-token",
        "github",
        "GitHub OAuth",
    )

    # Verify the key was updated, not added
    assert not mock_db_session.add.called
    assert api_key.encrypted_key == "encrypted_new-access-token"
    assert api_key.encrypted_secret == "encrypted_new-refresh-token"
    assert api_key.display_name == "GitHub OAuth"

    # Now test creating a new key
    mock_filter.first.return_value = None  # No existing key

    # Mock get_api_keys_for_service to return no keys
    with patch("src.utils.security.get_api_keys_for_service") as mock_get_keys:
        mock_get_keys.return_value = []  # No existing keys

        # Execute API key storage (creating new key)
        with patch("src.models.database.ApiKey", spec=ApiKey):
            result = store_oauth_api_key(
                mock_db_session,
                123,
                "github",
                "new-access-token",
                "new-refresh-token",
                "github",
                "GitHub OAuth",
            )

    # Verify a new key was added
    assert mock_db_session.add.called

    # Get the created API key from the mock
    created_key = mock_db_session.add.call_args[0][0]

    # Verify key properties
    assert created_key.user_id == 123
    assert created_key.service == "github"
    assert created_key.encrypted_key == "encrypted_new-access-token"
    assert created_key.encrypted_secret == "encrypted_new-refresh-token"
    assert created_key.is_oauth is True
    assert created_key.oauth_provider == "github"
    assert created_key.is_default is True  # Should be default as it's the first key
    assert created_key.display_name == "GitHub OAuth"


# Test handling expired tokens
@patch("src.utils.oauth_config.refresh_oauth_token")
@patch("src.utils.oauth_config.decrypt_data")
def test_get_oauth_access_token_expired(
    mock_decrypt, mock_refresh, mock_user, mock_oauth_env
):
    """Test getting an OAuth access token when the token is expired."""
    # Set token as expired
    mock_user.oauth_token_expiry = datetime.utcnow() - timedelta(hours=1)

    # Mock refresh function with success
    mock_refresh.return_value = True

    # Mock decrypt function
    mock_decrypt.return_value = "decrypted-token"

    # Get the token
    token = get_oauth_access_token(mock_user)

    # Verify refresh was called and token was decrypted
    mock_refresh.assert_called_once_with(mock_user)
    mock_decrypt.assert_called_once_with(mock_user.oauth_access_token)

    # Verify correct token was returned
    assert token == "decrypted-token"

    # Now test with refresh failure
    mock_refresh.reset_mock()
    mock_decrypt.reset_mock()
    mock_refresh.return_value = False

    # Get the token with failed refresh
    token = get_oauth_access_token(mock_user)

    # Verify refresh was called but decrypt was not
    mock_refresh.assert_called_once_with(mock_user)
    mock_decrypt.assert_not_called()

    # Verify no token was returned
    assert token is None
