"""Functional tests for the settings page."""
import pytest
from unittest.mock import patch, MagicMock
import streamlit as st
import pandas as pd

from src.pages.settings import (
    show_settings,
    show_api_connections,
    show_connection_status,
    test_all_connections,
    show_api_troubleshooting
)
from src.utils.api_config import APIConfigManager


@pytest.fixture
def mock_user():
    """Create a mock authenticated user."""
    return {
        'id': 1,
        'username': 'testuser',
        'email': 'test@example.com',
        'created_at': '2023-01-01'
    }


@pytest.fixture
def mock_api_configs():
    """Create mock API configurations."""
    return [
        {
            "name": "OpenAI",
            "service_id": "openai",
            "description": "AI API",
            "category": "AI/LLM",
            "needs_key": True,
            "needs_secret": False,
            "env_var_key": "OPENAI_API_KEY",
            "env_var_secret": None,
            "website": "https://openai.com",
            "api_docs": "https://docs.openai.com"
        },
        {
            "name": "Binance",
            "service_id": "binance",
            "description": "Crypto exchange",
            "category": "Crypto Exchange",
            "needs_key": True,
            "needs_secret": True,
            "env_var_key": "BINANCE_API_KEY",
            "env_var_secret": "BINANCE_SECRET_KEY",
            "website": "https://binance.com",
            "api_docs": "https://binance-docs.github.io"
        },
        {
            "name": "Yahoo Finance",
            "service_id": "yahoofinance",
            "description": "Financial data",
            "category": "Financial Data",
            "needs_key": False,
            "needs_secret": False,
            "env_var_key": None,
            "env_var_secret": None,
            "website": "https://finance.yahoo.com",
            "api_docs": "https://pypi.org/project/yfinance/"
        }
    ]


@patch("src.pages.settings.require_login")
def test_show_settings_requires_login(mock_require_login, mock_user):
    """Test that the settings page requires login."""
    # Setup mock
    mock_require_login.return_value = mock_user
    
    # Mock the tabs to avoid streamlit execution
    with patch("streamlit.tabs") as mock_tabs:
        mock_tab1 = MagicMock()
        mock_tab2 = MagicMock()
        mock_tab3 = MagicMock()
        mock_tab4 = MagicMock()
        mock_tab5 = MagicMock()
        mock_tabs.return_value = [mock_tab1, mock_tab2, mock_tab3, mock_tab4, mock_tab5]
        
        # Also mock the individual tab functions
        with patch("src.pages.settings.show_account_settings"), \
             patch("src.pages.settings.show_api_key_settings"), \
             patch("src.pages.settings.show_api_connections"), \
             patch("src.pages.settings.show_preferences"), \
             patch("src.pages.settings.show_data_export"):
            
            # Call the function
            show_settings()
            
            # Verify
            mock_require_login.assert_called_once()
            mock_tabs.assert_called_once()


@patch("streamlit.tabs")  # Use global patch to catch all uses of st.tabs
@patch("src.pages.settings.show_connection_status")
@patch("src.pages.settings.show_api_troubleshooting")
@patch("src.utils.api_config.APIConfigManager.get_api_categories")
@patch("src.utils.api_config.APIConfigManager.get_api_configs_by_category")
def test_show_api_connections(
    mock_get_configs_by_category, mock_get_categories, 
    mock_show_troubleshooting, mock_show_status, mock_tabs
):
    """Test showing API connections tab."""
    # Setup mocks with side_effect to handle different tab calls
    def mock_tabs_side_effect(*args, **kwargs):
        if args and args[0] == ["Connection Status", "Configuration", "Troubleshooting"]:
            # For the main tabs
            return [MagicMock(), MagicMock(), MagicMock()]
        else:
            # For the category tabs
            return [MagicMock(), MagicMock()]
    
    mock_tabs.side_effect = mock_tabs_side_effect
    
    # Setup categories and configs
    mock_get_categories.return_value = ["AI/LLM", "Crypto Exchange"]
    mock_get_configs_by_category.side_effect = lambda category: [
        {
            "name": f"{category} API 1", 
            "key": "", 
            "secret": "", 
            "enabled": False, 
            "service_id": f"{category.lower()}_api1",
            "description": f"Test {category} API 1 description",
            "website": "https://example.com",
            "api_docs": "https://example.com/docs",
            "required_fields": ["key", "secret"],
            "base_url": "https://api.example.com",
            "config_id": f"{category.lower()}_api1_config"
        },
        {
            "name": f"{category} API 2", 
            "key": "", 
            "secret": "", 
            "enabled": False, 
            "service_id": f"{category.lower()}_api2",
            "description": f"Test {category} API 2 description",
            "website": "https://example2.com",
            "api_docs": "https://example2.com/docs",
            "required_fields": ["key", "secret"],
            "base_url": "https://api.example2.com",
            "config_id": f"{category.lower()}_api2_config"
        }
    ]
    
    # Mock expander
    with patch("streamlit.expander") as mock_expander:
        mock_expander_instance = MagicMock()
        mock_expander.return_value.__enter__.return_value = mock_expander_instance
        
        # Mock form
        with patch("streamlit.form") as mock_form:
            mock_form_instance = MagicMock()
            mock_form.return_value.__enter__.return_value = mock_form_instance
            
            # Call the function
            show_api_connections(user_id=1)
            
            # Verify function calls
            mock_show_status.assert_called_once()
            mock_show_troubleshooting.assert_called_once()
            mock_get_categories.assert_called_once()


@patch("src.pages.settings.test_all_connections")
@patch("src.utils.api_config.APIConfigManager.get_api_categories")
def test_show_connection_status_initializes_results(mock_get_categories, mock_test_all):
    """Test that connection status initializes results if not in session state."""
    # Mock categories
    mock_get_categories.return_value = ["AI/LLM", "Exchange"]
    
    # Setup mock with the full expected result format
    mock_test_all.return_value = [
        {
            "name": "Test API 1", 
            "success": True, 
            "has_credentials": True,
            "category": "AI/LLM",
            "message": "Connected successfully",
            "error": None,
            "service_id": "test_api_1",
            "api_config": {
                "name": "Test API 1",
                "service_id": "test_api_1",
                "needs_key": True,
                "category": "AI/LLM"
            }
        },
        {
            "name": "Test API 2", 
            "success": False, 
            "has_credentials": False,
            "category": "Exchange",
            "message": "Not configured",
            "error": "API key is missing",
            "service_id": "test_api_2",
            "api_config": {
                "name": "Test API 2",
                "service_id": "test_api_2",
                "needs_key": True,
                "category": "Exchange"
            }
        }
    ]
    
    # Create a proper session state mock that allows dict-like and attribute-like access
    class SessionStateMock(dict):
        def __getattr__(self, key):
            if key not in self:
                return None
            return self[key]
        
        def __setattr__(self, key, value):
            self[key] = value
    
    session_state = SessionStateMock()
    
    # Patch the session state and mock streamlit components
    with patch("src.pages.settings.st.session_state", session_state), \
         patch("streamlit.button") as mock_button, \
         patch("streamlit.spinner") as mock_spinner, \
         patch("streamlit.columns") as mock_columns, \
         patch("streamlit.metric") as mock_metric:
        
        mock_button.return_value = False
        mock_spinner_ctx = MagicMock()
        mock_spinner.return_value.__enter__.return_value = mock_spinner_ctx
        mock_col1, mock_col2, mock_col3 = MagicMock(), MagicMock(), MagicMock()
        mock_columns.return_value = [mock_col1, mock_col2, mock_col3]
        
        # Call the function
        show_connection_status()
        
        # Verify
        mock_test_all.assert_called_once()
        assert "connection_results" in session_state
        # Just check a few key elements rather than the entire object
        assert len(session_state["connection_results"]) == 2
        assert session_state["connection_results"][0]["name"] == "Test API 1"
        assert session_state["connection_results"][0]["success"] is True
        assert session_state["connection_results"][1]["name"] == "Test API 2"
        assert session_state["connection_results"][1]["success"] is False


@patch("src.utils.api_config.APIConfigManager.get_api_configs")
@patch("src.utils.api_config.APIConfigManager.get_api_credentials")
@patch("src.utils.api_config.APIConfigManager.test_api_connection")
def test_test_all_connections(mock_test_connection, mock_get_credentials, mock_get_configs, mock_api_configs):
    """Test the function that tests all API connections."""
    # Setup mocks
    mock_get_configs.return_value = mock_api_configs
    mock_get_credentials.side_effect = [
        ("api_key", None),  # OpenAI
        ("api_key", "api_secret"),  # Binance
        ("", "")  # Yahoo Finance (doesn't need keys)
    ]
    mock_test_connection.side_effect = [
        (True, "OpenAI connected"),
        (True, "Binance connected"),
        (False, "Yahoo Finance failed")
    ]
    
    # Call the function
    results = test_all_connections()
    
    # Verify
    assert len(results) == 3
    assert results[0]["service_id"] == "openai"
    assert results[0]["success"] is True
    assert results[1]["service_id"] == "binance"
    assert results[1]["success"] is True
    assert results[2]["service_id"] == "yahoofinance"
    assert results[2]["success"] is False
    
    assert mock_get_configs.call_count == 1
    assert mock_get_credentials.call_count == 3
    assert mock_test_connection.call_count == 3


@patch("streamlit.expander")
def test_show_api_troubleshooting(mock_expander):
    """Test showing the API troubleshooting section."""
    # Setup mock
    mock_expander_instance = MagicMock()
    mock_expander.return_value.__enter__.return_value = mock_expander_instance
    
    # Call the function
    show_api_troubleshooting()
    
    # Verify
    assert mock_expander.call_count == 4  # Four expanders for different API types