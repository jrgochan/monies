"""Tests for exchange API integration."""
import pytest
from unittest.mock import patch, MagicMock

from src.api.exchanges import (
    get_wallet_balance_for_currency as get_wallet_balance,
    get_market_price,
    execute_trade
)


@pytest.fixture
def mock_exchange_client():
    """Create a mock exchange client."""
    with patch('src.api.exchanges.get_exchange_client') as mock_get_client:
        client_instance = MagicMock()
        mock_get_client.return_value = client_instance
        
        # Mock methods for Binance or other exchange APIs
        client_instance.get_account.return_value = {
            'balances': [
                {'asset': 'BTC', 'free': '1.0', 'locked': '0.5'},
                {'asset': 'ETH', 'free': '10.0', 'locked': '2.0'},
                {'asset': 'USDT', 'free': '5000.0', 'locked': '1000.0'}
            ]
        }
        
        client_instance.get_symbol_ticker.return_value = {'price': '30000.00'}
        
        client_instance.create_order.return_value = {
            'orderId': '12345',
            'status': 'FILLED',
            'executedQty': '0.1',
            'cummulativeQuoteQty': '3000.00'
        }
        
        yield client_instance


def test_get_wallet_balance(mock_exchange_client):
    """Test retrieving wallet balance from exchange."""
    # Test with Binance exchange
    balance = get_wallet_balance('binance', 'BTC', 'api_key', 'api_secret')
    
    # Check that the function returned the expected values
    assert balance == {'free': 1.0, 'locked': 0.5, 'total': 1.5}
    
    # Verify mock was called
    mock_exchange_client.get_account.assert_called_once()


def test_get_market_price(mock_exchange_client):
    """Test retrieving market price from exchange."""
    # Test with Binance exchange and BTC/USDT pair
    price = get_market_price('binance', 'BTCUSDT', 'api_key', 'api_secret')
    
    # Check that the function returned the expected price
    assert price == 30000.00
    
    # Verify mock was called with correct parameters
    mock_exchange_client.get_symbol_ticker.assert_called_once_with(symbol='BTCUSDT')


def test_execute_trade(mock_exchange_client):
    """Test executing a trade on an exchange."""
    # Test buying BTC with USDT on Binance
    # Patch the place_order function since that's what execute_trade calls
    with patch('src.api.exchanges.place_order') as mock_place_order:
        mock_place_order.return_value = {
            'success': True,
            'order_id': '12345',
            'message': 'Order placed successfully'
        }
        
        result = execute_trade(
            exchange='binance',
            symbol='BTCUSDT',
            trade_type='buy',
            quantity=0.1,
            price=30000.00,
            api_key='api_key',
            api_secret='api_secret'
        )
        
        # Check that the function returned the expected result
        assert result['status'] == 'FILLED'
        assert result['quantity'] == '0.1'
        
        # Verify mock was called with correct parameters
        mock_place_order.assert_called_once()