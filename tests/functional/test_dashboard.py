"""Functional tests for the dashboard page."""
import pytest
from unittest.mock import patch, MagicMock
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

from src.pages.dashboard import (
    show_dashboard,
    get_portfolio_value,
    get_portfolio_history,
    get_asset_allocation,
    get_recent_transactions
)


@pytest.fixture
def mock_user():
    """Create a mock authenticated user."""
    return {
        'id': 1,
        'username': 'testuser',
        'email': 'test@example.com'
    }


@pytest.fixture
def mock_wallets():
    """Create mock wallet data."""
    wallet1 = MagicMock()
    wallet1.id = 1
    wallet1.name = "Test Wallet 1"
    wallet1.exchange = "binance"
    
    wallet2 = MagicMock()
    wallet2.id = 2
    wallet2.name = "Test Wallet 2"
    wallet2.exchange = "coinbase"
    
    return [wallet1, wallet2]


@pytest.fixture
def mock_balances():
    """Create mock balance data."""
    balance1 = MagicMock()
    balance1.currency = "BTC"
    balance1.amount = 1.5
    
    balance2 = MagicMock()
    balance2.currency = "ETH"
    balance2.amount = 10.0
    
    return [balance1, balance2]


@pytest.fixture
def mock_transactions():
    """Create mock transaction data."""
    transaction1 = MagicMock()
    transaction1.id = 1
    transaction1.transaction_type = "buy"
    transaction1.currency = "BTC"
    transaction1.amount = 0.5
    transaction1.price = 30000.0
    transaction1.timestamp = datetime.now() - timedelta(days=1)
    transaction1.status = "completed"
    transaction1.notes = "Test transaction"
    
    transaction2 = MagicMock()
    transaction2.id = 2
    transaction2.transaction_type = "sell"
    transaction2.currency = "ETH"
    transaction2.amount = 2.0
    transaction2.price = 1500.0
    transaction2.timestamp = datetime.now() - timedelta(days=2)
    transaction2.status = "completed"
    transaction2.notes = "Another test"
    
    return [transaction1, transaction2]


@patch("src.pages.dashboard.require_login")
def test_show_dashboard_requires_login(mock_require_login, mock_user):
    """Test that the dashboard page requires login."""
    # Setup mock
    mock_require_login.return_value = mock_user
    
    # Mock load_user_wallets to return no wallets
    with patch("src.pages.dashboard.load_user_wallets") as mock_load_wallets:
        mock_load_wallets.return_value = []
        
        # Mock streamlit components
        with patch("streamlit.info") as mock_info:
            # Call the function
            show_dashboard()
            
            # Verify
            mock_require_login.assert_called_once()
            mock_load_wallets.assert_called_once_with(mock_user['id'])
            mock_info.assert_called_once()


@patch("src.pages.dashboard.load_user_wallets")
@patch("src.pages.dashboard.get_portfolio_value")
def test_show_dashboard_with_wallets(mock_get_value, mock_load_wallets, mock_user, mock_wallets):
    """Test showing the dashboard with wallets."""
    # Setup mocks
    mock_load_wallets.return_value = mock_wallets
    
    # Mock portfolio value calculation
    portfolio_value = 50000.0
    portfolio = {"BTC": 1.5, "ETH": 10.0}
    prices = {"BTC": 30000.0, "ETH": 2000.0}
    mock_get_value.return_value = (portfolio_value, portfolio, prices)
    
    # Mock streamlit components
    with patch("streamlit.subheader") as mock_subheader, \
         patch("streamlit.columns") as mock_columns, \
         patch("streamlit.metric") as mock_metric, \
         patch("streamlit.tabs") as mock_tabs, \
         patch("src.pages.dashboard.get_portfolio_history") as mock_get_history, \
         patch("src.pages.dashboard.get_asset_allocation") as mock_get_allocation, \
         patch("src.pages.dashboard.get_recent_transactions") as mock_get_transactions:
        
        # Mock columns behavior to return different lists based on parameters
        def mock_columns_side_effect(*args, **kwargs):
            if args and len(args) > 0 and args[0] == [3, 1]:  # For the insight columns
                return [MagicMock(), MagicMock()]
            elif args and len(args) > 0 and args[0] == [1, 3, 1]:  # For the transaction columns
                return [MagicMock(), MagicMock(), MagicMock()]
            else:  # Default for any other calls (including the main columns)
                return [MagicMock(), MagicMock(), MagicMock()]
            
        mock_columns.side_effect = mock_columns_side_effect
        
        # Create mock tabs
        tab1, tab2, tab3 = MagicMock(), MagicMock(), MagicMock()
        mock_tabs.return_value = [tab1, tab2, tab3]
        
        # Mock history data
        mock_get_history.return_value = pd.DataFrame({
            'date': pd.date_range(start=datetime.now()-timedelta(days=30), end=datetime.now()),
            'value': [portfolio_value * (1 + i*0.01) for i in range(31)]
        })
        
        # Mock allocation data
        mock_get_allocation.return_value = [
            {'currency': 'BTC', 'amount': 1.5, 'value': 45000.0},
            {'currency': 'ETH', 'amount': 10.0, 'value': 20000.0}
        ]
        
        # Mock transaction data with proper attributes
        tx1 = MagicMock()
        tx1.transaction_type = 'buy'
        tx1.amount = 0.5
        tx1.currency = 'BTC'
        tx1.price = 30000.0  # Use float instead of MagicMock
        tx1.notes = 'Test transaction'
        tx1.timestamp = datetime.now()
        tx1.status = 'completed'
        
        tx2 = MagicMock()
        tx2.transaction_type = 'sell'
        tx2.amount = 1.0
        tx2.currency = 'ETH'
        tx2.price = 2000.0  # Use float instead of MagicMock
        tx2.notes = None
        tx2.timestamp = datetime.now()
        tx2.status = 'completed'
        
        mock_get_transactions.return_value = [tx1, tx2]
        
        # Require login (already mocked)
        with patch("src.pages.dashboard.require_login") as mock_require_login:
            mock_require_login.return_value = mock_user
            
            # Call the function
            show_dashboard()
            
            # Verify
            mock_require_login.assert_called_once()
            mock_load_wallets.assert_called_once()
            mock_get_value.assert_called_once()
            mock_subheader.assert_called()
            mock_columns.assert_called()
            mock_tabs.assert_called_once()
            mock_get_history.assert_called_once()
            mock_get_allocation.assert_called_once()
            mock_get_transactions.assert_called_once()


@patch("src.pages.dashboard.load_wallet_balances")
@patch("src.pages.dashboard.get_current_prices")
def test_get_portfolio_value_success(mock_get_prices, mock_load_balances, mock_wallets, mock_balances):
    """Test portfolio value calculation with successful API connection."""
    # Setup mocks
    mock_load_balances.side_effect = [[mock_balances[0]], [mock_balances[1]]]
    mock_get_prices.return_value = {"BTC": 30000.0, "ETH": 2000.0}
    
    # Call the function
    value, portfolio, prices = get_portfolio_value(mock_wallets)
    
    # Verify
    assert value == 1.5 * 30000.0 + 10.0 * 2000.0  # 65000.0
    assert portfolio == {"BTC": 1.5, "ETH": 10.0}
    assert prices == {"BTC": 30000.0, "ETH": 2000.0}
    assert mock_load_balances.call_count == 2
    
    # Check the call but allow for either order of coins since it comes from a set
    mock_get_prices.assert_called_once()
    call_args = mock_get_prices.call_args[0]
    assert call_args[0] == "binanceus"
    assert sorted(call_args[1]) == sorted(["BTC", "ETH"])


@patch("src.pages.dashboard.load_wallet_balances")
@patch("src.pages.dashboard.get_current_prices")
def test_get_portfolio_value_api_error(mock_get_prices, mock_load_balances, mock_wallets, mock_balances):
    """Test portfolio value calculation with API error."""
    # Setup mocks
    mock_load_balances.side_effect = [[mock_balances[0]], [mock_balances[1]]]
    mock_get_prices.return_value = {"error": "Connection failed", "BTC": 30000.0, "ETH": 2000.0}
    
    # Call the function
    value, portfolio, prices = get_portfolio_value(mock_wallets)
    
    # Verify
    assert value == 0  # Should return 0 when there's an error
    assert portfolio == {"BTC": 1.5, "ETH": 10.0}
    assert "error" in prices
    assert mock_load_balances.call_count == 2
    
    # Check the call but allow for either order of coins since it comes from a set
    mock_get_prices.assert_called_once()
    call_args = mock_get_prices.call_args[0]
    assert call_args[0] == "binanceus"
    assert sorted(call_args[1]) == sorted(["BTC", "ETH"])


@patch("src.pages.dashboard.load_wallet_balances")
@patch("src.pages.dashboard.get_current_prices")
def test_get_portfolio_history_with_api_error(mock_get_prices, mock_load_balances, mock_wallets, mock_balances):
    """Test portfolio history generation with API error."""
    # Setup mocks
    mock_load_balances.side_effect = [[mock_balances[0]], [mock_balances[1]]]
    mock_get_prices.return_value = {"error": "Connection failed"}
    
    # Call the function
    result = get_portfolio_history(mock_wallets)
    
    # Verify
    assert isinstance(result, pd.DataFrame)
    assert 'error' in result.columns
    assert result['error'].iloc[0] == True  # Use == instead of 'is'
    assert 'error_message' in result.columns
    assert "Connection failed" in result['error_message'].iloc[0]
    assert mock_load_balances.call_count == 2
    
    # Make sure get_prices was called once, don't need to check parameters here
    mock_get_prices.assert_called_once()


@patch("src.pages.dashboard.load_wallet_balances")
@patch("src.pages.dashboard.get_current_prices")
def test_get_portfolio_history_success(mock_get_prices, mock_load_balances, mock_wallets, mock_balances):
    """Test portfolio history generation with successful API."""
    # Setup mocks
    mock_load_balances.side_effect = [[mock_balances[0]], [mock_balances[1]]]
    
    # Need to handle multiple calls to get_current_prices
    prices = {"BTC": 30000.0, "ETH": 2000.0}
    mock_get_prices.return_value = prices
    
    # Call the function - in normal operation, this would still indicate an error
    # because we can't generate full historical data without actual API access
    result = get_portfolio_history(mock_wallets)
    
    # Verify
    assert isinstance(result, pd.DataFrame)
    assert 'error' in result.columns
    assert result['error'].iloc[0] == True  # Use == instead of 'is'
    assert 'value' in result.columns
    assert result['value'].iloc[0] > 0  # Should have a positive value
    assert mock_load_balances.call_count == 2
    
    # Make sure get_prices was called once, don't need to check parameters here
    mock_get_prices.assert_called_once()


def test_get_asset_allocation():
    """Test asset allocation calculation."""
    # Setup test data
    portfolio = {"BTC": 1.5, "ETH": 10.0}
    prices = {"BTC": 30000.0, "ETH": 2000.0}
    
    # Call the function
    result = get_asset_allocation(portfolio, prices)
    
    # Verify
    assert len(result) == 2
    assert result[0]['currency'] == "BTC"
    assert result[0]['amount'] == 1.5
    assert result[0]['value'] == 45000.0
    assert result[1]['currency'] == "ETH"
    assert result[1]['amount'] == 10.0
    assert result[1]['value'] == 20000.0


@patch("src.pages.dashboard.SessionLocal")
def test_get_recent_transactions(mock_session_local, mock_transactions):
    """Test getting recent transactions."""
    # Setup mock
    session_mock = MagicMock()
    mock_session_local.return_value = session_mock
    session_mock.query.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = mock_transactions
    
    # Call the function
    result = get_recent_transactions(user_id=1, limit=5)
    
    # Verify
    assert result == mock_transactions
    session_mock.query.assert_called_once()
    session_mock.close.assert_called_once()