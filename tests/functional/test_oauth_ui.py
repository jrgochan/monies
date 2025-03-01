"""Functional tests for OAuth UI components."""
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from src.models.database import ApiKey, User
from src.pages.settings import show_account_settings, show_api_key_settings
from src.utils.auth import show_login_page


# Create fixture for mocked OAuth configs
@pytest.fixture
def mock_oauth_configs():
    """Mock OAuth configuration for testing."""
    mock_configs = {
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
    }
    with patch("src.utils.oauth_config.OAUTH_CONFIGS", mock_configs):
        with patch("src.pages.settings.OAUTH_CONFIGS", mock_configs):
            yield mock_configs


# Create fixture for mocked Streamlit components
@pytest.fixture
def mock_st():
    """Mock Streamlit components."""
    mock_streamlit = MagicMock()

    # Create mock columns
    mock_column = MagicMock()
    mock_columns = [mock_column, mock_column, mock_column]
    mock_streamlit.columns.return_value = mock_columns

    # Create mock containers
    mock_container = MagicMock()
    mock_column.__enter__.return_value = mock_column
    mock_column.container.return_value.__enter__.return_value = mock_container

    # Create mock tabs
    mock_tab = MagicMock()
    mock_streamlit.tabs.return_value = [mock_tab, mock_tab, mock_tab]

    # Create mock form
    mock_form = MagicMock()
    mock_streamlit.form.return_value.__enter__.return_value = mock_form

    # Create mock expander
    mock_expander = MagicMock()
    mock_streamlit.expander.return_value.__enter__.return_value = mock_expander

    with patch("src.pages.settings.st", mock_streamlit):
        with patch("src.utils.auth.st", mock_streamlit):
            yield mock_streamlit


# Create fixture for mocked session state
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

    session_state = SessionState()

    with patch("src.pages.settings.st.session_state", session_state):
        with patch("src.utils.auth.st.session_state", session_state):
            yield session_state


# Create fixture for mock OAuth user
@pytest.fixture
def mock_oauth_user():
    """Create a mock OAuth user dict for testing."""
    return {
        "id": 123,
        "username": "testuser",
        "email": "test@example.com",
        "created_at": "2025-01-01T00:00:00",
        "oauth_provider": "google",
    }


# Create fixture for mock database user
@pytest.fixture
def mock_db_user():
    """Create a mock database user object."""
    user = MagicMock(spec=User)
    user.id = 123
    user.username = "testuser"
    user.email = "test@example.com"
    user.created_at = datetime.utcnow()
    user.oauth_provider = "google"
    user.oauth_id = "123456789"
    user.oauth_access_token = "encrypted_access_token"
    user.oauth_refresh_token = "encrypted_refresh_token"
    user.oauth_token_expiry = datetime.utcnow() + timedelta(hours=1)
    user.password_hash = "hashed_password"
    return user


# Create fixture for mock API keys
@pytest.fixture
def mock_api_keys():
    """Create mock API keys for testing."""
    regular_key = MagicMock(spec=ApiKey)
    regular_key.id = 1
    regular_key.user_id = 123
    regular_key.service = "binance"
    regular_key.encrypted_key = "encrypted_key"
    regular_key.encrypted_secret = "encrypted_secret"
    regular_key.created_at = datetime.utcnow()
    regular_key.is_oauth = False
    regular_key.oauth_provider = None
    regular_key.is_default = True
    regular_key.display_name = "My Binance Key"

    oauth_key = MagicMock(spec=ApiKey)
    oauth_key.id = 2
    oauth_key.user_id = 123
    oauth_key.service = "github"
    oauth_key.encrypted_key = "encrypted_oauth_key"
    oauth_key.encrypted_secret = "encrypted_oauth_secret"
    oauth_key.created_at = datetime.utcnow()
    oauth_key.is_oauth = True
    oauth_key.oauth_provider = "github"
    oauth_key.is_default = True
    oauth_key.display_name = "GitHub OAuth"

    return [regular_key, oauth_key]


# Test login page OAuth display
@patch("src.utils.oauth_config.generate_oauth_authorize_url")
def test_login_page_oauth_display(
    mock_gen_url, mock_st, mock_session_state, mock_oauth_configs
):
    """Test that the login page correctly displays OAuth providers."""
    # Mock generate_oauth_authorize_url function
    mock_gen_url.return_value = ("https://example.com/auth", "test-state")

    # Mock the query parameters
    mock_st.query_params = {}

    # Create database session mock
    db_mock = MagicMock()
    with patch("src.utils.auth.SessionLocal", return_value=db_mock):
        # Call the login page function
        show_login_page()

    # Verify tabs were created
    mock_st.tabs.assert_called_with(["Login", "Register"])

    # Verify OAuth section was displayed
    mock_st.subheader.assert_any_call("Or login with")

    # Verify columns were created for OAuth buttons (3 providers, so 1 row of 3)
    mock_st.columns.assert_any_call(3)


# Test OAuth query parameter handling in login page
@patch("src.utils.oauth_config.generate_oauth_authorize_url")
def test_login_page_oauth_redirect(
    mock_gen_url, mock_st, mock_session_state, mock_oauth_configs
):
    """Test that the login page correctly handles OAuth redirect requests."""
    # Mock generate_oauth_authorize_url function
    mock_gen_url.return_value = ("https://example.com/auth", "test-state")

    # Mock the query parameters to simulate clicking the Google button
    mock_st.query_params = {"oauth_provider": ["google"]}

    # Create database session mock
    db_mock = MagicMock()
    with patch("src.utils.auth.SessionLocal", return_value=db_mock):
        # Call the login page function
        show_login_page()

    # Verify OAuth URL was generated
    mock_gen_url.assert_called_once_with("google")

    # Verify OAuth state was stored in session
    assert mock_session_state.oauth_state == "test-state"
    assert mock_session_state.oauth_flow == "google"

    # Verify redirect was triggered
    mock_st.markdown.assert_called()
    mock_st.stop.assert_called_once()


# Test account settings with OAuth user
@patch("src.pages.settings.SessionLocal")
def test_account_settings_oauth_user(
    mock_session_local,
    mock_st,
    mock_session_state,
    mock_oauth_user,
    mock_db_user,
    mock_oauth_configs,
):
    """Test account settings page for an OAuth-connected user."""
    # Mock the database session
    mock_db = MagicMock()
    mock_session_local.return_value = mock_db

    # Setup database query to return the mock user
    mock_query = MagicMock()
    mock_filter = MagicMock()
    mock_filter.first.return_value = mock_db_user
    mock_query.filter.return_value = mock_filter
    mock_db.query.return_value = mock_query

    # Call the account settings function
    show_account_settings(mock_oauth_user)

    # Verify OAuth provider info is displayed
    mock_st.write.assert_any_call("Connected with: **Google**")
    mock_st.info.assert_any_call(
        "Your account is linked to Google. You can use Google to log in."
    )

    # Verify "Connected Accounts" section is shown
    mock_st.subheader.assert_any_call("Connected Accounts")

    # Verify columns are created for provider cards
    mock_st.columns.assert_any_call(3)


# Test account settings with OAuth Connect button
@patch("src.pages.settings.SessionLocal")
@patch("src.utils.oauth_config.generate_oauth_authorize_url")
def test_account_settings_oauth_connect_button(
    mock_gen_url,
    mock_session_local,
    mock_st,
    mock_session_state,
    mock_oauth_user,
    mock_db_user,
    mock_oauth_configs,
):
    """Test OAuth connect button in account settings."""
    # Modify user to not have an OAuth provider for this test
    mock_db_user.oauth_provider = None
    mock_oauth_user_no_oauth = mock_oauth_user.copy()
    mock_oauth_user_no_oauth.pop("oauth_provider", None)

    # Mock the database session
    mock_db = MagicMock()
    mock_session_local.return_value = mock_db

    # Setup database query to return the mock user
    mock_query = MagicMock()
    mock_filter = MagicMock()
    mock_filter.first.return_value = mock_db_user
    mock_query.filter.return_value = mock_filter
    mock_db.query.return_value = mock_query

    # Mock generate_oauth_authorize_url function
    mock_gen_url.return_value = ("https://example.com/auth", "test-state")

    # Simulate clicking the connect button by setting up button return value
    mock_button = MagicMock(return_value=True)
    mock_st.button = mock_button

    # Call the account settings function
    show_account_settings(mock_oauth_user_no_oauth)

    # Verify OAuth URL was generated (at least one call)
    assert mock_gen_url.called

    # Verify OAuth state was stored in session when button clicked
    assert mock_session_state.oauth_state == "test-state"
    assert mock_session_state.oauth_flow in ["google", "coinbase", "github"]

    # Verify redirect was triggered
    mock_st.markdown.assert_called()


# Test API keys section showing OAuth-connected keys
@patch("src.pages.settings.get_user_api_keys")
@patch("src.pages.settings.SessionLocal")
def test_api_key_settings_oauth_keys(
    mock_session_local,
    mock_get_keys,
    mock_st,
    mock_oauth_user,
    mock_api_keys,
    mock_oauth_configs,
):
    """Test API key settings page displays OAuth-connected keys correctly."""
    # Mock the database session
    mock_db = MagicMock()
    mock_session_local.return_value = mock_db

    # Mock get_user_api_keys to return our test keys
    mock_get_keys.return_value = mock_api_keys

    # Additional mocks for functions called within the component
    with patch("src.utils.security.get_api_keys_for_service") as mock_get_service_keys:
        mock_get_service_keys.return_value = mock_api_keys

        # Call the API key settings function
        show_api_key_settings(mock_oauth_user["id"])

    # Verify tabs were created for the services
    mock_st.tabs.assert_called()

    # Check that the OAuth API key was displayed correctly
    # This is complex to verify exactly, so we'll check that certain methods were called
    # that would be needed to display the OAuth key info
    mock_st.container.assert_called()
    mock_st.info.assert_called()  # Should show "Connected via GitHub OAuth" for the OAuth key


# Test the OAuth API connection section
@patch("src.pages.settings.SessionLocal")
def test_oauth_api_connections_section(
    mock_session_local, mock_st, mock_oauth_user, mock_oauth_configs
):
    """Test the OAuth API connection section in settings."""
    # Mock the database session
    mock_db = MagicMock()
    mock_session_local.return_value = mock_db

    # We need to mock generate_oauth_authorize_url since it's imported inside the function
    with patch("src.pages.settings.generate_oauth_authorize_url") as mock_gen_url:
        mock_gen_url.return_value = ("https://example.com/auth", "test-state")

        # Simulate clicking the connect button by setting up button return value
        mock_button = MagicMock(return_value=True)
        mock_st.button = mock_button

        # Call the API key settings function
        show_api_key_settings(mock_oauth_user["id"])

    # Verify OAuth section header was displayed
    mock_st.subheader.assert_any_call("API Key Management")
    mock_st.subheader.assert_any_call("Connect via OAuth")

    # Verify OAuth info message was displayed
    mock_st.info.assert_any_call(
        "You can also connect API services by authenticating with OAuth providers."
    )

    # Since we mocked the button to return True, the OAuth flow should be triggered
    # Verify OAuth state was stored in session
    assert mock_st.session_state.oauth_state == "test-state"
    assert mock_st.session_state.oauth_flow in ["coinbase", "github"]

    # Verify oauth_for_api flag was set to indicate this is for API access
    assert mock_st.session_state.oauth_for_api is True

    # Verify redirect was triggered
    mock_st.markdown.assert_called()


# Test password form for OAuth users with no password
@patch("src.pages.settings.SessionLocal")
def test_account_settings_password_form_oauth_user(
    mock_session_local,
    mock_st,
    mock_session_state,
    mock_oauth_user,
    mock_db_user,
    mock_oauth_configs,
):
    """Test the password setup form for OAuth users without passwords."""
    # Modify user to have no password
    mock_db_user.password_hash = None

    # Mock the database session
    mock_db = MagicMock()
    mock_session_local.return_value = mock_db

    # Setup database query to return the mock user
    mock_query = MagicMock()
    mock_filter = MagicMock()
    mock_filter.first.return_value = mock_db_user
    mock_query.filter.return_value = mock_filter
    mock_db.query.return_value = mock_query

    # Call the account settings function
    show_account_settings(mock_oauth_user)

    # Verify setup password form is shown
    mock_st.form.assert_any_call("setup_password_form")

    # Verify info message about password setup is shown
    mock_st.info.assert_any_call(
        "You're currently using Google to log in. Setting up a password will allow you to log in with your username and password as well."
    )
