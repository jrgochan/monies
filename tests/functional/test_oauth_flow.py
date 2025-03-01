"""End-to-end tests for OAuth flow."""
import os
from unittest.mock import MagicMock, patch

import pytest

from src.models.database import ApiKey, User
from src.utils.auth import handle_oauth_callback
from src.utils.oauth_config import create_or_update_oauth_user


# Create fixture for mocked OAuth environment
@pytest.fixture
def mock_oauth_env():
    """Mock OAuth environment variables."""
    with patch.dict(
        os.environ,
        {
            "GOOGLE_CLIENT_ID": "test-client-id",
            "GOOGLE_CLIENT_SECRET": "test-client-secret",
            "GOOGLE_REDIRECT_URI": "http://localhost:8501/callback/google",
            "COINBASE_CLIENT_ID": "test-client-id",
            "COINBASE_CLIENT_SECRET": "test-client-secret",
            "COINBASE_REDIRECT_URI": "http://localhost:8501/callback/coinbase",
            "JWT_SECRET": "test-jwt-secret",
            "SECRET_KEY": "12345678901234567890123456789012",  # 32 characters for Fernet
        },
    ):
        yield


# Create fixture for mocked Streamlit session state
@pytest.fixture
def mock_session_state():
    """Mock Streamlit session state."""

    class SessionState(dict):
        def __getattr__(self, key):
            if key in self:
                return self[key]
            return None

        def __setattr__(self, key, value):
            self[key] = value

        def __delattr__(self, key):
            if key in self:
                del self[key]

    session_state = SessionState()
    with patch("streamlit.session_state", session_state):
        with patch("app.st.session_state", session_state):
            with patch("src.utils.auth.st.session_state", session_state):
                with patch("src.pages.settings.st.session_state", session_state):
                    yield session_state


# Create fixture for mocked Streamlit functions
@pytest.fixture
def mock_streamlit():
    """Mock Streamlit functions."""
    mock_st = MagicMock()

    # Create session state attribute
    mock_st.session_state = {}

    # Mock query_params to simulate callback
    mock_st.query_params = {
        "code": ["test-auth-code"],
        "state": ["test-oauth-state"],
    }

    with patch("streamlit.query_params", mock_st.query_params):
        with patch("streamlit.session_state", mock_st.session_state):
            with patch("streamlit.markdown", mock_st.markdown):
                with patch("streamlit.stop", mock_st.stop):
                    with patch("streamlit.error", mock_st.error):
                        with patch("app.st", mock_st):
                            with patch("src.utils.auth.st", mock_st):
                                with patch("src.pages.settings.st", mock_st):
                                    yield mock_st


# Test complete OAuth login flow
@patch("src.utils.oauth_config.exchange_code_for_token")
@patch("src.utils.oauth_config.get_user_info")
@patch("src.utils.oauth_config.create_or_update_oauth_user")
def test_oauth_login_flow(
    mock_create_user,
    mock_get_user_info,
    mock_exchange_code,
    mock_oauth_env,
    mock_session_state,
    mock_streamlit,
):
    """Test end-to-end OAuth login flow."""
    # Setup session state with OAuth flow in progress
    mock_session_state.oauth_state = "test-oauth-state"
    mock_session_state.oauth_flow = "google"

    # Ensure query params are properly set for the test
    mock_streamlit.query_params = {
        "code": ["test-auth-code"],
        "state": ["test-oauth-state"],
    }

    # Mock token exchange
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
        "picture": "https://example.com/picture.jpg",
    }

    # Mock user creation
    mock_user = MagicMock(spec=User)
    mock_user.id = 123
    mock_user.username = "testuser"
    mock_user.email = "test@example.com"
    mock_user.oauth_provider = "google"
    mock_create_user.return_value = mock_user

    # Need to patch the import itself
    oauth_config_module = MagicMock()
    oauth_config_module.exchange_code_for_token = mock_exchange_code
    oauth_config_module.get_user_info = mock_get_user_info
    oauth_config_module.create_or_update_oauth_user = mock_create_user

    # Mock the OAuth_CONFIGS import as well that our updated code uses
    oauth_config_module.OAUTH_CONFIGS = {
        "google": {
            "client_id": "test-client-id",
            "client_secret": "test-client-secret",
            "display_name": "Google",
            "color": "#DB4437",
        }
    }
    oauth_config_module.generate_oauth_authorize_url = MagicMock(
        return_value=("https://accounts.google.com/oauth", "test-state")
    )

    # Mock the dynamic import
    with patch.dict("sys.modules", {"src.utils.oauth_config": oauth_config_module}):
        # Call the OAuth callback handler
        handle_oauth_callback()

    # Verify the flow
    mock_exchange_code.assert_called_once_with("google", "test-auth-code")
    mock_get_user_info.assert_called_once_with("google", "test-access-token")
    mock_create_user.assert_called_once_with(
        "google", mock_get_user_info.return_value, mock_exchange_code.return_value
    )

    # Verify session state changes
    assert "user" in mock_session_state
    assert "token" in mock_session_state
    assert "oauth_state" not in mock_session_state
    assert "oauth_flow" not in mock_session_state

    # Verify redirect happened
    mock_streamlit.markdown.assert_called_once()
    mock_streamlit.stop.assert_called_once()


# Test OAuth API connection flow
@patch("src.utils.oauth_config.exchange_code_for_token")
@patch("src.utils.oauth_config.get_user_info")
@patch("src.utils.oauth_config.create_or_update_oauth_user")
@patch("src.utils.security.store_oauth_api_key")
def test_oauth_api_connection_flow(
    mock_store_api_key,
    mock_create_user,
    mock_get_user_info,
    mock_exchange_code,
    mock_oauth_env,
    mock_session_state,
    mock_streamlit,
):
    """Test OAuth flow for API connection."""
    # Setup session state with OAuth flow for API connection
    mock_session_state.oauth_state = "test-oauth-state"
    mock_session_state.oauth_flow = "coinbase"
    mock_session_state.oauth_for_api = True
    mock_session_state.user = {
        "id": 123,
        "username": "testuser",
        "email": "test@example.com",
    }

    # Ensure query params are properly set for the test
    mock_streamlit.query_params = {
        "code": ["test-auth-code"],
        "state": ["test-oauth-state"],
    }

    # Mock token exchange
    mock_exchange_code.return_value = {
        "access_token": "test-access-token",
        "refresh_token": "test-refresh-token",
        "expires_in": 3600,
    }

    # Mock user info
    mock_get_user_info.return_value = {
        "id": "coinbase_user_id",
        "email": "test@example.com",
        "name": "Test User",
    }

    # Mock user creation/update
    mock_user = MagicMock(spec=User)
    mock_user.id = 123
    mock_user.username = "testuser"
    mock_user.email = "test@example.com"
    mock_user.oauth_provider = "coinbase"
    mock_create_user.return_value = mock_user

    # Mock API key creation
    mock_api_key = MagicMock(spec=ApiKey)
    mock_api_key.id = 1
    mock_api_key.service = "coinbase"
    mock_api_key.is_oauth = True
    mock_store_api_key.return_value = mock_api_key

    # Create database session mock
    # Need to patch the import itself
    oauth_config_module = MagicMock()
    oauth_config_module.exchange_code_for_token = mock_exchange_code
    oauth_config_module.get_user_info = mock_get_user_info
    oauth_config_module.create_or_update_oauth_user = mock_create_user
    oauth_config_module.create_api_keys_from_oauth = MagicMock(
        return_value={"coinbase": True}
    )

    # Mock the OAuth_CONFIGS import as well that our updated code uses
    oauth_config_module.OAUTH_CONFIGS = {
        "coinbase": {
            "client_id": "test-client-id",
            "client_secret": "test-client-secret",
            "display_name": "Coinbase",
            "color": "#0052FF",
        }
    }
    oauth_config_module.generate_oauth_authorize_url = MagicMock(
        return_value=("https://www.coinbase.com/oauth/authorize", "test-state")
    )

    # Mock the dynamic import
    with patch.dict("sys.modules", {"src.utils.oauth_config": oauth_config_module}):
        # Call the OAuth callback handler
        handle_oauth_callback()

    # Verify OAuth flow
    mock_exchange_code.assert_called_once_with("coinbase", "test-auth-code")
    mock_get_user_info.assert_called_once_with("coinbase", "test-access-token")
    mock_create_user.assert_called_once_with(
        "coinbase", mock_get_user_info.return_value, mock_exchange_code.return_value
    )

    # Verify API keys were created automatically during user creation
    # This happens in create_or_update_oauth_user via create_api_keys_from_oauth

    # Verify session state changes
    assert "user" in mock_session_state
    assert "token" in mock_session_state
    assert "oauth_state" not in mock_session_state
    assert "oauth_flow" not in mock_session_state
    assert "oauth_for_api" not in mock_session_state

    # Verify redirect happened
    mock_streamlit.markdown.assert_called_once()
    mock_streamlit.stop.assert_called_once()


# Test error handling in OAuth flow
@patch("src.utils.oauth_config.exchange_code_for_token")
def test_oauth_flow_error_handling(
    mock_exchange_code, mock_oauth_env, mock_session_state, mock_streamlit
):
    """Test error handling in OAuth flow."""
    # Setup session state with OAuth flow
    mock_session_state.oauth_state = "test-oauth-state"
    mock_session_state.oauth_flow = "google"

    # Ensure query params are properly set for the test
    mock_streamlit.query_params = {
        "code": ["test-auth-code"],
        "state": ["test-oauth-state"],
    }

    # Mock token exchange to fail
    mock_exchange_code.return_value = None

    # Need to patch the import itself
    oauth_config_module = MagicMock()
    oauth_config_module.exchange_code_for_token = mock_exchange_code

    # Mock the OAuth_CONFIGS import as well that our updated code uses
    oauth_config_module.OAUTH_CONFIGS = {
        "google": {
            "client_id": "test-client-id",
            "client_secret": "test-client-secret",
            "display_name": "Google",
            "color": "#DB4437",
        }
    }
    oauth_config_module.generate_oauth_authorize_url = MagicMock(
        return_value=("https://accounts.google.com/oauth", "test-state")
    )

    # Mock the dynamic import
    with patch.dict("sys.modules", {"src.utils.oauth_config": oauth_config_module}):
        # Call the OAuth callback handler
        handle_oauth_callback()

    # Verify error was displayed
    mock_streamlit.error.assert_called_once_with("Failed to exchange code for token")

    # Verify no redirect happened
    mock_streamlit.markdown.assert_not_called()

    # Now test with invalid state
    mock_session_state.oauth_state = "different-state"
    mock_session_state.oauth_flow = "google"

    # Reset mocks
    mock_streamlit.reset_mock()

    # Ensure query params are properly set for the test
    mock_streamlit.query_params = {
        "code": ["test-auth-code"],
        "state": ["test-oauth-state"],
    }

    # Call the OAuth callback handler
    handle_oauth_callback()

    # Verify security error was displayed
    mock_streamlit.error.assert_called_once_with(
        "Invalid OAuth state. Possible security attack."
    )

    # Verify OAuth state was cleared
    assert "oauth_state" not in mock_session_state
    assert "oauth_flow" not in mock_session_state


# Test OAuth user merging (existing email)
@patch("src.utils.oauth_config.encrypt_data")
def test_oauth_user_merging(mock_encrypt, mock_oauth_env):
    """Test merging OAuth user with existing account by email."""
    # Mock encrypt function
    mock_encrypt.side_effect = lambda x: f"encrypted_{x}"

    # Create mock database session
    mock_db = MagicMock()

    # Create mock existing user
    existing_user = MagicMock(spec=User)
    existing_user.id = 123
    existing_user.username = "existinguser"
    existing_user.email = "test@example.com"
    existing_user.oauth_provider = None
    existing_user.oauth_id = None

    # Mock database queries to find existing user by email
    mock_query = MagicMock()
    mock_filter = MagicMock()
    # First query for OAuth ID returns no user
    mock_filter.first.side_effect = [None, existing_user]
    mock_query.filter.return_value = mock_filter
    mock_db.query.return_value = mock_query

    # OAuth user info
    user_info = {
        "id": "google_user_id",
        "email": "test@example.com",
        "name": "Test User",
    }

    # Token data
    token_data = {
        "access_token": "test-access-token",
        "refresh_token": "test-refresh-token",
        "expires_in": 3600,
    }

    # Mock creating API keys
    with patch(
        "src.utils.oauth_config.create_api_keys_from_oauth"
    ) as mock_create_api_keys:
        mock_create_api_keys.return_value = {}

        # Call create_or_update_oauth_user
        with patch("src.utils.oauth_config.SessionLocal", return_value=mock_db):
            user = create_or_update_oauth_user("google", user_info, token_data)

    # Verify no new user was created
    assert not mock_db.add.called

    # Verify existing user was updated with OAuth info
    assert existing_user.oauth_provider == "google"
    assert existing_user.oauth_id == "google_user_id"
    assert existing_user.oauth_access_token == "encrypted_test-access-token"
    assert existing_user.oauth_refresh_token == "encrypted_test-refresh-token"

    # Verify API keys were created
    mock_create_api_keys.assert_called_once()

    # Verify database was committed
    assert mock_db.commit.called

    # Verify the user returned is the existing user
    assert user is existing_user
