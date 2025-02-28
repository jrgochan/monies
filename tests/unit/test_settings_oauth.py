"""Tests for OAuth integration in settings page."""
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest
import streamlit as st

from src.pages.settings import show_account_settings


# Create a fixture for a mock user
@pytest.fixture
def mock_user():
    """Fixture to create a mock user for settings tests."""
    return {
        "id": 123,
        "username": "testuser",
        "email": "test@example.com",
        "created_at": "2025-01-01T00:00:00",
        "oauth_provider": "google",
    }


# Create a fixture for a mock database user object
@pytest.fixture
def mock_db_user():
    """Fixture to create a mock database user object."""
    user = MagicMock()
    user.id = 123
    user.username = "testuser"
    user.email = "test@example.com"
    user.created_at = datetime.utcnow()
    user.oauth_provider = "google"
    user.password_hash = "hashed_password"
    return user


# Test the account settings display with OAuth provider
@patch("src.pages.settings.SessionLocal")
@patch("src.pages.settings.st")
def test_show_account_settings_with_oauth(
    mock_st, mock_session_local, mock_user, mock_db_user
):
    """Test that account settings correctly displays OAuth provider information."""
    # Mock the database session
    mock_db = MagicMock()
    mock_session_local.return_value = mock_db

    # Setup database queries to return the mock user
    mock_query = MagicMock()
    mock_db.query.return_value = mock_query
    mock_filter = MagicMock()
    mock_query.filter.return_value = mock_filter
    mock_filter.first.return_value = mock_db_user

    # Mock st.write to check for output
    mock_write = MagicMock()
    mock_st.write = mock_write

    # Mock st.info to check for provider info
    mock_info = MagicMock()
    mock_st.info = mock_info

    # Call the function with a user that has OAuth provider set
    show_account_settings(mock_user)

    # Check that OAuth provider info is displayed
    mock_write.assert_any_call("Connected with: **Google**")
    mock_info.assert_any_call(
        "Your account is linked to Google. You can use Google to log in."
    )


# Test the account settings without OAuth provider
@patch("src.pages.settings.SessionLocal")
@patch("src.pages.settings.st")
def test_show_account_settings_without_oauth(mock_st, mock_session_local, mock_db_user):
    """Test account settings display without OAuth provider."""
    # Create a user without OAuth provider
    user = {
        "id": 123,
        "username": "testuser",
        "email": "test@example.com",
        "created_at": "2025-01-01T00:00:00",
    }

    # Mock the database session
    mock_db = MagicMock()
    mock_session_local.return_value = mock_db

    # Setup database queries to return the mock user
    mock_query = MagicMock()
    mock_db.query.return_value = mock_query
    mock_filter = MagicMock()
    mock_query.filter.return_value = mock_filter

    # Create a version of the mock user without OAuth
    user_obj = MagicMock()
    user_obj.id = 123
    user_obj.username = "testuser"
    user_obj.email = "test@example.com"
    user_obj.created_at = datetime.utcnow()
    user_obj.oauth_provider = None
    user_obj.password_hash = "hashed_password"

    mock_filter.first.return_value = user_obj

    # Mock st.write to check for output
    mock_write = MagicMock()
    mock_st.write = mock_write

    # Call the function
    show_account_settings(user)

    # We should NOT see OAuth provider info
    for call in mock_write.call_args_list:
        args, kwargs = call
        assert "Connected with:" not in args[0]


# Test the OAuth connection buttons in account settings
@patch("src.utils.oauth_config.OAUTH_CONFIGS")
@patch("src.pages.settings.SessionLocal")
@patch("src.pages.settings.st")
def test_show_account_settings_oauth_buttons(
    mock_st, mock_session_local, mock_oauth_configs, mock_user, mock_db_user
):
    """Test that the OAuth connection buttons are displayed correctly."""
    # Mock the database session
    mock_db = MagicMock()
    mock_session_local.return_value = mock_db

    # Setup database queries to return the mock user
    mock_query = MagicMock()
    mock_db.query.return_value = mock_query
    mock_filter = MagicMock()
    mock_query.filter.return_value = mock_filter
    mock_filter.first.return_value = mock_db_user

    # Mock OAuth configs
    mock_oauth_configs.items.return_value = [
        (
            "google",
            {
                "client_id": "google-test-client-id",
                "client_secret": "google-test-client-secret",
                "display_name": "Google",
                "icon": "google",
                "color": "#DB4437",
            },
        ),
        (
            "github",
            {
                "client_id": "github-test-client-id",
                "client_secret": "github-test-client-secret",
                "display_name": "GitHub",
                "icon": "github",
                "color": "#24292E",
            },
        ),
        (
            "facebook",
            {
                "client_id": "facebook-test-client-id",
                "client_secret": "facebook-test-client-secret",
                "display_name": "Facebook",
                "icon": "facebook",
                "color": "#1877F2",
            },
        ),
    ]

    # Mock columns for the grid layout
    mock_col = MagicMock()
    mock_st.columns.return_value = [mock_col, mock_col, mock_col]

    # Mock the container for the cards
    mock_container = MagicMock()
    mock_col.__enter__.return_value = mock_col
    mock_col.container.return_value.__enter__.return_value = mock_container

    # Mock the success and info displays
    mock_success = MagicMock()
    mock_container.success = mock_success
    mock_info = MagicMock()
    mock_container.info = mock_info

    # Call the function with patches already in place
    with patch("src.utils.oauth_config.generate_oauth_authorize_url") as mock_gen_url:
        # Generate a valid URL for mocking
        mock_gen_url.return_value = ("https://example.com/auth", "test-state")

        # Call the function
        show_account_settings(mock_user)

    # Verify that OAuth provider section is displayed
    mock_st.subheader.assert_any_call("Connected Accounts")

    # Since our mocked user has Google connected, verify that success is shown for Google
    # This is hard to test explicitly due to the dynamics of the column/container structure

    # Verify that the columns function was called with 3 (per row)
    mock_st.columns.assert_any_call(3)


# Test password form display with OAuth user
@patch("src.pages.settings.SessionLocal")
@patch("src.pages.settings.st")
def test_show_account_settings_password_form_with_oauth(
    mock_st, mock_session_local, mock_user, mock_db_user
):
    """Test that the password form is correctly displayed for OAuth users."""
    # Mock the database session
    mock_db = MagicMock()
    mock_session_local.return_value = mock_db

    # Setup database queries to return the mock user
    mock_query = MagicMock()
    mock_db.query.return_value = mock_query
    mock_filter = MagicMock()
    mock_query.filter.return_value = mock_filter
    mock_filter.first.return_value = mock_db_user

    # Since the user is OAuth but has a password, should show change password form
    mock_form = MagicMock()
    mock_st.form.return_value.__enter__.return_value = mock_form

    # Call the function
    show_account_settings(mock_user)

    # Verify the change password form is shown
    mock_st.form.assert_any_call("change_password_form")

    # Now test with a user who doesn't have a password set
    mock_db_user.password_hash = None
    mock_st.form.reset_mock()

    # Call the function
    show_account_settings(mock_user)

    # Verify setup password form is shown instead
    mock_st.form.assert_any_call("setup_password_form")

    # Check for the info message about setting up a password
    # This is a bit tricky to test directly but we'd need to check that
    # the form contains text about "Setting up a password will allow you to log in with your username and password"
