"""Test data fixtures for use across tests."""
import datetime


# Test user data
TEST_USERS = [
    {
        "id": 1,
        "username": "testuser",
        "email": "test@example.com",
        "password": "testpassword"
    },
    {
        "id": 2,
        "username": "demouser",
        "email": "demo@example.com",
        "password": "demopassword"
    }
]

# Test wallet data
TEST_WALLETS = [
    {
        "id": 1,
        "user_id": 1,
        "name": "BTC Wallet",
        "wallet_type": "cryptocurrency",
        "address": "0x1234567890abcdef",
        "currency": "BTC",
        "exchange": "binance",
        "balance": 1.5,
        "available_balance": 1.0,
        "locked_balance": 0.5
    },
    {
        "id": 2,
        "user_id": 1,
        "name": "ETH Wallet",
        "wallet_type": "cryptocurrency",
        "address": "0xabcdef1234567890",
        "currency": "ETH",
        "exchange": "coinbase",
        "balance": 10.0,
        "available_balance": 8.0,
        "locked_balance": 2.0
    }
]

# Test transaction data
TEST_TRANSACTIONS = [
    {
        "id": 1,
        "wallet_id": 1,
        "transaction_type": "buy",
        "amount": 1.0,
        "price": 30000.0,
        "timestamp": datetime.datetime(2023, 1, 1, 12, 0, 0),
        "status": "completed",
        "transaction_hash": "0x1234567890abcdef"
    },
    {
        "id": 2,
        "wallet_id": 1,
        "transaction_type": "sell",
        "amount": 0.5,
        "price": 35000.0,
        "timestamp": datetime.datetime(2023, 1, 15, 12, 0, 0),
        "status": "completed",
        "transaction_hash": "0xfedcba0987654321"
    },
    {
        "id": 3,
        "wallet_id": 2,
        "transaction_type": "buy",
        "amount": 5.0,
        "price": 2000.0,
        "timestamp": datetime.datetime(2023, 1, 5, 12, 0, 0),
        "status": "completed",
        "transaction_hash": "0xabcdef1234567890"
    }
]

# Test API key data
TEST_API_KEYS = [
    {
        "exchange": "binance",
        "api_key": "test_binance_api_key",
        "api_secret": "test_binance_api_secret"
    },
    {
        "exchange": "coinbase",
        "api_key": "test_coinbase_api_key",
        "api_secret": "test_coinbase_api_secret"
    }
]

# Test social media account data
TEST_SOCIAL_ACCOUNTS = [
    {
        "platform": "twitter",
        "username": "test_twitter_user",
        "api_key": "test_twitter_api_key",
        "api_secret": "test_twitter_api_secret",
        "access_token": "test_twitter_access_token",
        "access_secret": "test_twitter_access_secret"
    }
]

# Test market data
TEST_MARKET_DATA = {
    "BTC": {
        "price": 35000.0,
        "24h_change": 2.5,
        "market_cap": 650000000000,
        "volume": 25000000000
    },
    "ETH": {
        "price": 2200.0,
        "24h_change": 1.5,
        "market_cap": 260000000000,
        "volume": 15000000000
    },
    "SOL": {
        "price": 85.0,
        "24h_change": 3.2,
        "market_cap": 32000000000,
        "volume": 2000000000
    }
}