import base64
import hashlib
import hmac
import json
import time
from typing import Any, Dict, List, Optional, Tuple, Union, cast
from urllib.parse import urlencode

import ccxt
from binance.client import Client as BinanceClient

from src.api import API_KEYS, logger, make_api_request


class BaseExchangeClient:
    """
    Base class for all exchange clients with common interface
    """

    def __init__(
        self,
        exchange_name: str,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        oauth_token: Optional[str] = None,
    ):
        """
        Initialize the base exchange client

        Args:
            exchange_name: Name of the exchange
            api_key: API key for the exchange (optional if using OAuth)
            api_secret: API secret for the exchange (optional if using OAuth)
            oauth_token: OAuth token for exchanges that support it
        """
        self.exchange_name = exchange_name.lower()
        self.api_key = api_key
        self.api_secret = api_secret
        self.oauth_token = oauth_token
        self.client = None

    def get_wallet_balance(self) -> Dict[str, Dict[str, float]]:
        """
        Get wallet balances from the exchange

        Returns:
            Dictionary of currency -> balance details
        """
        raise NotImplementedError("Subclass must implement get_wallet_balance method")

    def get_current_prices(
        self, currencies: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Get current prices for currencies against USD/USDT

        Args:
            currencies: List of currency symbols to get prices for

        Returns:
            Dictionary of currency -> price
        """
        raise NotImplementedError("Subclass must implement get_current_prices method")

    def place_order(
        self,
        symbol: str,
        order_type: str,
        side: str,
        amount: float,
        price: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Place an order on the exchange

        Args:
            symbol: Trading pair symbol
            order_type: Type of order ('market' or 'limit')
            side: Side of the order ('buy' or 'sell')
            amount: Amount to buy or sell
            price: Price for limit orders

        Returns:
            Order result information
        """
        raise NotImplementedError("Subclass must implement place_order method")

    def get_transaction_history(
        self, symbol: Optional[str] = None, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get transaction history from the exchange

        Args:
            symbol: Trading pair symbol (optional)
            limit: Maximum number of transactions to return

        Returns:
            List of transaction dictionaries
        """
        raise NotImplementedError(
            "Subclass must implement get_transaction_history method"
        )

    def test_connection(self) -> bool:
        """
        Test the connection to the exchange

        Returns:
            True if successful, False otherwise
        """
        raise NotImplementedError("Subclass must implement test_connection method")


class ExchangeClientFactory:
    """
    Factory for creating appropriate exchange clients
    """

    @staticmethod
    def create_client(
        exchange_name: str,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        oauth_token: Optional[str] = None,
    ) -> Optional[BaseExchangeClient]:
        """
        Create and return the appropriate exchange client

        Args:
            exchange_name: Name of the exchange (e.g., 'binance', 'coinbase')
            api_key: API key for the exchange (optional if using OAuth)
            api_secret: API secret for the exchange (optional if using OAuth)
            oauth_token: OAuth access token for the exchange (optional)

        Returns:
            Exchange client instance if successful, None otherwise
        """
        # Check if we have valid credentials
        has_api_keys = api_key and api_secret
        has_oauth = oauth_token is not None

        if not has_api_keys and not has_oauth:
            logger.error(f"Missing API credentials for {exchange_name}")
            return None

        exchange_name = exchange_name.lower()

        try:
            if exchange_name in ["binance", "binanceus"]:
                # Binance doesn't support OAuth, so we need API keys
                if not has_api_keys:
                    logger.error("Binance requires API key and secret")
                    return None

                client = BinanceExchangeClient(api_key, api_secret)
                if not client.test_connection():
                    return None
                return client

            elif exchange_name == "coinbase":
                # Coinbase supports both API key and OAuth authentication
                if has_oauth:
                    client = CoinbaseExchangeClient(oauth_token=oauth_token)
                else:
                    client = CoinbaseExchangeClient(api_key, api_secret)

                if not client.test_connection():
                    return None
                return client

            else:
                # Other exchanges via CCXT
                if not has_api_keys:
                    logger.error(f"{exchange_name} requires API key and secret")
                    return None

                client = CCXTExchangeClient(exchange_name, api_key, api_secret)
                if not client.test_connection():
                    return None
                return client

        except Exception as e:
            error_msg = str(e)
            ExchangeUtility.handle_geographical_restrictions(exchange_name, error_msg)
            return None


# Legacy function to maintain backward compatibility
def get_exchange_client(
    exchange_name: str,
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
    oauth_token: Optional[str] = None,
) -> Any:
    """
    Get the appropriate exchange client based on exchange name.
    Supports both API key authentication and OAuth authentication for some exchanges.

    Args:
        exchange_name: Name of the exchange (e.g., 'binance', 'coinbase')
        api_key: API key for the exchange (optional if using OAuth)
        api_secret: API secret for the exchange (optional if using OAuth)
        oauth_token: OAuth access token for the exchange (optional)

    Returns:
        Exchange client instance if successful, None otherwise
    """
    client = ExchangeClientFactory.create_client(
        exchange_name, api_key, api_secret, oauth_token
    )

    # For backward compatibility, return the internal client object
    return client.client if client else None


class ExchangeUtility:
    """
    Utility class for exchange-related operations
    """

    @staticmethod
    def handle_geographical_restrictions(exchange_name: str, error_msg: str) -> None:
        """
        Handle geographical restriction errors

        Args:
            exchange_name: Name of the exchange
            error_msg: Error message from the exchange
        """
        if (
            "restricted location" in error_msg
            or "geographical restrictions" in error_msg
        ):
            if exchange_name.lower() == "binance":
                logger.error(
                    "Binance access error: Service unavailable due to geographical restrictions. "
                    "Using Binance.US or a VPN may resolve this issue."
                )
            else:
                logger.error(
                    f"Access error for {exchange_name}: Service unavailable due to geographical restrictions."
                )
        else:
            logger.error(f"Error connecting to {exchange_name}: {error_msg}")

    @staticmethod
    def format_binance_transactions(
        trades: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Format Binance transactions to standard format

        Args:
            trades: Raw Binance transactions

        Returns:
            Formatted transactions
        """
        formatted = []
        for trade in trades:
            formatted.append(
                {
                    "id": trade["id"],
                    "symbol": trade["symbol"],
                    "side": "buy" if trade["isBuyer"] else "sell",
                    "amount": float(trade["qty"]),
                    "price": float(trade["price"]),
                    "cost": float(trade["quoteQty"]),
                    "fee": float(trade["commission"]),
                    "fee_currency": trade["commissionAsset"],
                    "timestamp": trade["time"],
                }
            )
        return formatted

    @staticmethod
    def fallback_to_alternative_api(
        currencies: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Try to get price data from alternative API (CoinGecko)

        Args:
            currencies: List of currency symbols to get prices for

        Returns:
            Dictionary of currency -> price
        """
        result = {}

        try:
            if currencies:
                crypto_ids = [c.lower() for c in currencies]
                ids_param = ",".join(crypto_ids)

                success, data = make_api_request(
                    url="https://api.coingecko.com/api/v3/simple/price",
                    params={"ids": ids_param, "vs_currencies": "usd"},
                    error_msg="Error fetching from CoinGecko API",
                )

                if success and isinstance(data, dict):
                    for crypto_id, price_data in data.items():
                        # Convert to uppercase for consistency
                        crypto_symbol = crypto_id.upper()
                        if "usd" in price_data:
                            result[crypto_symbol] = price_data["usd"]

            # If we still don't have results, log error
            if not result and currencies:
                missing = [c for c in currencies if c not in result]
                logger.error(f"Failed to retrieve prices for: {', '.join(missing)}")

        except Exception as e:
            logger.error(f"Error fetching from alternative API: {str(e)}")

        return result


class BinanceExchangeClient(BaseExchangeClient):
    """
    Binance exchange client implementation
    """

    def __init__(self, api_key: str, api_secret: str):
        """
        Initialize Binance client

        Args:
            api_key: Binance API key
            api_secret: Binance API secret
        """
        super().__init__("binance", api_key, api_secret)
        # Always use US endpoint for safer operation
        self.client = BinanceClient(api_key, api_secret, tld="us")

    def test_connection(self) -> bool:
        """
        Test connection to Binance

        Returns:
            True if successful, False otherwise
        """
        try:
            if self.client is None:
                return False
            self.client.get_account()
            return True
        except Exception as e:
            error_msg = str(e)
            ExchangeUtility.handle_geographical_restrictions("binance", error_msg)
            return False

    def get_wallet_balance(self) -> Dict[str, Dict[str, float]]:
        """
        Get wallet balances from Binance

        Returns:
            Dictionary of currency -> balance details
        """
        result = {}

        try:
            if self.client is None:
                result["error"] = "Binance client not initialized"
                return result

            account = self.client.get_account()

            # Extract non-zero balances
            for asset in account["balances"]:
                currency = asset["asset"]
                free = float(asset["free"])
                locked = float(asset["locked"])
                total = free + locked

                if total > 0:
                    result[currency] = {"free": free, "locked": locked, "total": total}

        except Exception as e:
            error_msg = str(e)
            ExchangeUtility.handle_geographical_restrictions("binance", error_msg)
            # Return empty result with error flag
            result = {
                "error": (
                    "Cannot connect to Binance. "
                    "Please check your API credentials and network connection."
                )
            }

        return result

    def get_current_prices(
        self, currencies: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Get current prices for currencies against USDT

        Args:
            currencies: List of currency symbols to get prices for

        Returns:
            Dictionary of currency -> price
        """
        result = {}

        try:
            if self.client is None:
                result["error"] = "Binance client not initialized"
                return result

            # Get all ticker prices
            all_tickers = self.client.get_all_tickers()

            # Filter for USDT pairs if currencies specified
            for ticker in all_tickers:
                symbol = ticker["symbol"]
                if symbol.endswith("USDT"):
                    base_currency = symbol[:-4]

                    if currencies is None or base_currency in currencies:
                        result[base_currency] = float(ticker["price"])

        except Exception as e:
            error_msg = str(e)
            ExchangeUtility.handle_geographical_restrictions("binance", error_msg)

            # Try alternative API if we don't have results
            if not result or (currencies and not all(c in result for c in currencies)):
                fallback_result = ExchangeUtility.fallback_to_alternative_api(
                    currencies
                )
                result.update(fallback_result)

            # If we still don't have results, set error
            if not result or (currencies and not all(c in result for c in currencies)):
                result[
                    "error"
                ] = "Cannot connect to price data source. Please check your network connection."

        return result

    def place_order(
        self,
        symbol: str,
        order_type: str,
        side: str,
        amount: float,
        price: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Place an order on Binance

        Args:
            symbol: Trading pair symbol
            order_type: Type of order ('market' or 'limit')
            side: Side of the order ('buy' or 'sell')
            amount: Amount to buy or sell
            price: Price for limit orders

        Returns:
            Order result information
        """
        result = {"success": False, "order_id": None, "message": ""}

        try:
            if self.client is None:
                result["message"] = "Binance client not initialized"
                return result

            # Format symbol for Binance (no slash)
            binance_symbol = symbol.replace("/", "")

            # Place order based on type
            if order_type.lower() == "market":
                if side.lower() == "buy":
                    order = self.client.order_market_buy(
                        symbol=binance_symbol, quantity=amount
                    )
                else:
                    order = self.client.order_market_sell(
                        symbol=binance_symbol, quantity=amount
                    )
            elif order_type.lower() == "limit":
                if not price:
                    result["message"] = "Price is required for limit orders"
                    return result

                if side.lower() == "buy":
                    order = self.client.order_limit_buy(
                        symbol=binance_symbol, quantity=amount, price=price
                    )
                else:
                    order = self.client.order_limit_sell(
                        symbol=binance_symbol, quantity=amount, price=price
                    )
            else:
                result["message"] = f"Unsupported order type: {order_type}"
                return result

            result["success"] = True
            result["order_id"] = order["orderId"]
            result["message"] = "Order placed successfully"

        except Exception as e:
            error_msg = str(e)
            ExchangeUtility.handle_geographical_restrictions("binance", error_msg)
            result["message"] = f"Error placing order: {error_msg}"

        return result

    def get_transaction_history(
        self, symbol: Optional[str] = None, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get transaction history from Binance

        Args:
            symbol: Trading pair symbol (optional)
            limit: Maximum number of transactions to return

        Returns:
            List of transaction dictionaries
        """
        result = []

        try:
            if self.client is None:
                return [{"error": "Binance client not initialized"}]

            # Format symbol for Binance if provided
            binance_symbol = symbol.replace("/", "") if symbol else None

            # Get trade history
            if binance_symbol:
                trades = self.client.get_my_trades(symbol=binance_symbol, limit=limit)
                result = ExchangeUtility.format_binance_transactions(trades)
            else:
                # Get trades for multiple symbols
                all_trades = []
                tickers = self.client.get_all_tickers()
                for ticker in tickers[:10]:  # Limit to first 10 for performance
                    try:
                        symbol_trades = self.client.get_my_trades(
                            symbol=ticker["symbol"], limit=10
                        )
                        all_trades.extend(symbol_trades)
                    except Exception as e:
                        # Skip if error for this symbol
                        logger.debug(
                            f"Error getting trades for {ticker['symbol']}: {e}"
                        )

                # Sort by time and limit
                all_trades.sort(key=lambda x: x["time"], reverse=True)
                trades = all_trades[:limit]
                result = ExchangeUtility.format_binance_transactions(trades)

        except Exception as e:
            error_msg = str(e)
            ExchangeUtility.handle_geographical_restrictions("binance", error_msg)
            # Add a special dictionary with error message instead of empty list
            result = [
                {
                    "error": f"Cannot connect to Binance to retrieve transaction history. Please check your API credentials and network connection."
                }
            ]

        return result


class CoinbaseExchangeClient(BaseExchangeClient):
    """
    Coinbase exchange client implementation
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        oauth_token: Optional[str] = None,
    ):
        """
        Initialize Coinbase client

        Args:
            api_key: Coinbase API key (optional if using OAuth)
            api_secret: Coinbase API secret (optional if using OAuth)
            oauth_token: OAuth token for authentication
        """
        super().__init__("coinbase", api_key, api_secret, oauth_token)
        # The internal client uses the existing CoinbaseClient class
        self.client = CoinbaseClient(api_key, api_secret, oauth_token)

    def test_connection(self) -> bool:
        """
        Test connection to Coinbase

        Returns:
            True if successful, False otherwise
        """
        try:
            if self.client is None:
                return False
            self.client.get_accounts()
            return True
        except Exception as e:
            logger.error(f"Error connecting to Coinbase: {str(e)}")
            return False

    def get_wallet_balance(self) -> Dict[str, Dict[str, float]]:
        """
        Get wallet balances from Coinbase

        Returns:
            Dictionary of currency -> balance details
        """
        result = {}

        try:
            if self.client is None:
                result["error"] = "Coinbase client not initialized"
                return result

            accounts_data = self.client.get_accounts()

            # Extract non-zero balances
            for account in accounts_data["data"]:
                currency = account["balance"]["currency"]
                amount = float(account["balance"]["amount"])

                if amount > 0:
                    # Coinbase doesn't distinguish between free and locked balances
                    # Use available balance as free and the difference as locked
                    if "available" in account:
                        available = float(account["available"])
                        locked = amount - available
                    else:
                        available = amount
                        locked = 0.0

                    result[currency] = {
                        "free": available,
                        "locked": locked,
                        "total": amount,
                    }

        except Exception as e:
            logger.error(f"Error fetching balances from Coinbase: {str(e)}")
            # Return empty result with error flag
            result = {
                "error": (
                    "Cannot connect to Coinbase. "
                    "Please check your API credentials and network connection."
                )
            }

        return result

    def get_current_prices(
        self, currencies: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Get current prices for currencies against USD

        Args:
            currencies: List of currency symbols to get prices for

        Returns:
            Dictionary of currency -> price
        """
        result = {}

        try:
            if self.client is None:
                result["error"] = "Coinbase client not initialized"
                return result

            # If currencies are specified, get prices for each one
            if currencies:
                for currency in currencies:
                    try:
                        # Get spot price
                        spot_price = self.client.get_spot_price(f"{currency}-USD")
                        result[currency] = spot_price
                    except Exception as e:
                        logger.warning(
                            f"Could not get price for {currency} from Coinbase: {str(e)}"
                        )
            else:
                # Without specific currencies, get exchange rates based on USD
                try:
                    # Using exchange rates for better efficiency when getting multiple prices
                    rates = self.client.get_exchange_rates("USD")

                    # Convert rates to prices (1/rate for USD base)
                    for currency, rate in rates.items():
                        if currency != "USD" and rate != "0":
                            # We need to invert the rate since we want price in USD
                            try:
                                price = 1 / float(rate)
                                result[currency] = price
                            except (ValueError, ZeroDivisionError):
                                # Skip currencies with invalid rates
                                pass
                except Exception as e:
                    logger.error(
                        f"Error fetching exchange rates from Coinbase: {str(e)}"
                    )

        except Exception as e:
            logger.error(f"Error fetching prices from Coinbase: {str(e)}")

            # Try alternative API if we don't have results
            if not result or (currencies and not all(c in result for c in currencies)):
                fallback_result = ExchangeUtility.fallback_to_alternative_api(
                    currencies
                )
                result.update(fallback_result)

            # If we still don't have results, set error
            if not result or (currencies and not all(c in result for c in currencies)):
                result[
                    "error"
                ] = "Cannot connect to price data source. Please check your network connection."

        return result

    def place_order(
        self,
        symbol: str,
        order_type: str,
        side: str,
        amount: float,
        price: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Place an order on Coinbase

        Args:
            symbol: Trading pair symbol
            order_type: Type of order ('market' or 'limit')
            side: Side of the order ('buy' or 'sell')
            amount: Amount to buy or sell
            price: Price for limit orders

        Returns:
            Order result information
        """
        result = {"success": False, "order_id": None, "message": ""}

        try:
            if self.client is None:
                result["message"] = "Coinbase client not initialized"
                return result

            # Extract currency from symbol (e.g., BTC/USD -> BTC)
            if "/" in symbol:
                currency = symbol.split("/")[0]
            else:
                # Try to extract currency assuming format like BTCUSD
                if symbol.endswith("USD"):
                    currency = symbol[:-3]
                else:
                    result[
                        "message"
                    ] = f"Invalid symbol format: {symbol}. Expected format: BTC/USD or BTCUSD"
                    return result

            # Convert amount to string for Coinbase API
            amount_str = str(amount)

            # Place order based on type
            if order_type.lower() == "market":
                order = self.client.place_market_order(
                    action=side.lower(), amount=amount_str, currency=currency
                )
            elif order_type.lower() == "limit":
                if not price:
                    result["message"] = "Price is required for limit orders"
                    return result

                order = self.client.place_limit_order(
                    action=side.lower(),
                    amount=amount_str,
                    price=str(price),
                    currency=currency,
                )
            else:
                result["message"] = f"Unsupported order type: {order_type}"
                return result

            result["success"] = True
            result["order_id"] = order.get("id", "unknown")
            result["message"] = "Order placed successfully"

        except Exception as e:
            logger.error(f"Error placing Coinbase order: {str(e)}")
            result["message"] = f"Error placing order: {str(e)}"

        return result

    def get_transaction_history(
        self, symbol: Optional[str] = None, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get transaction history from Coinbase

        Args:
            symbol: Trading pair symbol (optional)
            limit: Maximum number of transactions to return

        Returns:
            List of transaction dictionaries
        """
        result = []

        try:
            if self.client is None:
                return [{"error": "Coinbase client not initialized"}]

            # For Coinbase, we need to get accounts first, then transactions for each account
            accounts_data = self.client.get_accounts()

            # Process accounts
            all_transactions = []

            # Extract currency from symbol if provided
            target_currency = None
            if symbol:
                if "/" in symbol:
                    target_currency = symbol.split("/")[0]
                elif symbol.endswith("USD"):
                    target_currency = symbol[:-3]
                else:
                    target_currency = symbol

            # Get transactions for relevant accounts
            for account in accounts_data["data"]:
                # If symbol is specified, only get transactions for that currency
                if (
                    target_currency
                    and account["balance"]["currency"] != target_currency
                ):
                    continue

                # Get transactions for this account
                try:
                    account_id = account["id"]
                    account_currency = account["balance"]["currency"]

                    # Get transactions for this account
                    account_txs = self.client.get_transaction_history(account_id, limit)

                    # Add to all transactions list with currency info
                    for tx in account_txs:
                        tx["currency"] = account_currency
                        all_transactions.append(tx)
                except Exception as e:
                    logger.warning(
                        f"Error getting transactions for account {account['id']}: {str(e)}"
                    )

            # Sort all transactions by timestamp (newest first) and apply limit
            all_transactions.sort(key=lambda x: x.get("created_at", ""), reverse=True)
            transactions = all_transactions[:limit]

            # Format transactions to match our standard format
            for tx in transactions:
                # Determine transaction side (buy/sell)
                tx_type = tx.get("type", "")

                if tx_type == "buy":
                    side = "buy"
                elif tx_type == "sell":
                    side = "sell"
                else:
                    # Handle other transaction types
                    amount_str = tx.get("amount", {"amount": "0"}).get("amount", "0")
                    try:
                        amount_float = float(amount_str)
                        side = "buy" if amount_float > 0 else "sell"
                    except ValueError:
                        side = "unknown"

                # Get amount and currency
                currency = tx.get("currency", "")
                amount_str = tx.get("amount", {"amount": "0"}).get("amount", "0")
                try:
                    amount = abs(float(amount_str))
                except ValueError:
                    amount = 0.0

                # Get price if available
                native_amount_str = tx.get("native_amount", {"amount": "0"}).get(
                    "amount", "0"
                )
                try:
                    native_amount = abs(float(native_amount_str))
                    # Calculate price as native amount / amount
                    price = native_amount / amount if amount > 0 else 0
                except (ValueError, ZeroDivisionError):
                    price = 0.0
                    native_amount = 0.0

                # Format for our standard output
                result.append(
                    {
                        "id": tx.get("id", "unknown"),
                        "symbol": f"{currency}/USD",
                        "side": side,
                        "amount": amount,
                        "price": price,
                        "cost": native_amount,
                        "fee": 0.0,  # Coinbase doesn't expose fees in transactions API
                        "fee_currency": "USD",
                        "timestamp": tx.get("created_at", ""),
                    }
                )

        except Exception as e:
            logger.error(f"Error fetching transaction history from Coinbase: {str(e)}")
            # Add a special dictionary with error message
            result = [
                {
                    "error": f"Cannot connect to Coinbase to retrieve transaction history. Please check your API credentials and network connection."
                }
            ]

        return result


class CCXTExchangeClient(BaseExchangeClient):
    """
    CCXT-based client for other exchanges
    """

    def __init__(self, exchange_name: str, api_key: str, api_secret: str):
        """
        Initialize CCXT client

        Args:
            exchange_name: Name of the exchange
            api_key: API key
            api_secret: API secret
        """
        super().__init__(exchange_name, api_key, api_secret)

        try:
            exchange_class = getattr(ccxt, exchange_name.lower())
            self.client = exchange_class(
                {"apiKey": api_key, "secret": api_secret, "enableRateLimit": True}
            )
        except Exception as e:
            logger.error(f"Error initializing {exchange_name} client: {str(e)}")
            self.client = None

    def test_connection(self) -> bool:
        """
        Test connection to the exchange

        Returns:
            True if successful, False otherwise
        """
        if not self.client:
            return False

        try:
            self.client.fetch_balance()
            return True
        except Exception as e:
            logger.error(f"Error connecting to {self.exchange_name}: {str(e)}")
            return False

    def get_wallet_balance(self) -> Dict[str, Dict[str, float]]:
        """
        Get wallet balances from the exchange

        Returns:
            Dictionary of currency -> balance details
        """
        result = {}

        if not self.client:
            result["error"] = f"Client for {self.exchange_name} not initialized"
            return result

        try:
            balance = self.client.fetch_balance()

            # Extract non-zero balances
            for currency, amount in balance["total"].items():
                if amount > 0:
                    result[currency] = {
                        "free": balance["free"].get(currency, 0),
                        "locked": balance["used"].get(currency, 0),
                        "total": amount,
                    }

        except Exception as e:
            logger.error(f"Error fetching balances from {self.exchange_name}: {str(e)}")
            # Return error message
            result["error"] = (
                f"Cannot connect to {self.exchange_name}. "
                f"Please check your API credentials and network connection."
            )

        return result

    def get_current_prices(
        self, currencies: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Get current prices for currencies against USD/USDT

        Args:
            currencies: List of currency symbols to get prices for

        Returns:
            Dictionary of currency -> price
        """
        result = {}

        if not self.client:
            result["error"] = f"Client for {self.exchange_name} not initialized"
            return result

        try:
            # Get ticker prices
            tickers = self.client.fetch_tickers()

            # Filter for USD(T) pairs if currencies specified
            for symbol, ticker in tickers.items():
                if "/USDT" in symbol or "/USD" in symbol:
                    base_currency = symbol.split("/")[0]

                    if currencies is None or base_currency in currencies:
                        result[base_currency] = ticker["last"]

        except Exception as e:
            logger.error(f"Error fetching prices from {self.exchange_name}: {str(e)}")

            # Try alternative API if we don't have results
            if not result or (currencies and not all(c in result for c in currencies)):
                fallback_result = ExchangeUtility.fallback_to_alternative_api(
                    currencies
                )
                result.update(fallback_result)

            # If we still don't have results, set error
            if not result or (currencies and not all(c in result for c in currencies)):
                result[
                    "error"
                ] = "Cannot connect to price data source. Please check your network connection."

        return result

    def place_order(
        self,
        symbol: str,
        order_type: str,
        side: str,
        amount: float,
        price: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Place an order on the exchange

        Args:
            symbol: Trading pair symbol
            order_type: Type of order ('market' or 'limit')
            side: Side of the order ('buy' or 'sell')
            amount: Amount to buy or sell
            price: Price for limit orders

        Returns:
            Order result information
        """
        result = {"success": False, "order_id": None, "message": ""}

        if not self.client:
            result["message"] = f"Client for {self.exchange_name} not initialized"
            return result

        try:
            # Place order
            if order_type.lower() == "market":
                order = self.client.create_market_order(
                    symbol=symbol, side=side.lower(), amount=amount
                )
            elif order_type.lower() == "limit":
                if not price:
                    result["message"] = "Price is required for limit orders"
                    return result

                order = self.client.create_limit_order(
                    symbol=symbol, side=side.lower(), amount=amount, price=price
                )
            else:
                result["message"] = f"Unsupported order type: {order_type}"
                return result

            result["success"] = True
            result["order_id"] = order["id"]
            result["message"] = "Order placed successfully"

        except Exception as e:
            logger.error(f"Error placing order on {self.exchange_name}: {str(e)}")
            result["message"] = f"Error placing order: {str(e)}"

        return result

    def get_transaction_history(
        self, symbol: Optional[str] = None, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get transaction history from the exchange

        Args:
            symbol: Trading pair symbol (optional)
            limit: Maximum number of transactions to return

        Returns:
            List of transaction dictionaries
        """
        result = []

        if not self.client:
            return [{"error": f"Client for {self.exchange_name} not initialized"}]

        try:
            # Check if fetchMyTrades is supported
            if self.client.has["fetchMyTrades"]:
                # Get trades
                if symbol:
                    trades = self.client.fetch_my_trades(symbol=symbol, limit=limit)
                else:
                    # Get trades for multiple symbols
                    all_trades = []
                    markets = self.client.load_markets()

                    for symbol, market in list(markets.items())[
                        :10
                    ]:  # Limit to first 10 for performance
                        if "/USDT" in symbol or "/USD" in symbol:
                            try:
                                symbol_trades = self.client.fetch_my_trades(
                                    symbol=symbol, limit=10
                                )
                                all_trades.extend(symbol_trades)
                            except Exception as e:
                                # Skip if error for this symbol
                                logger.debug(f"Error fetching trades for {symbol}: {e}")

                    # Sort by time and limit
                    all_trades.sort(key=lambda x: x["timestamp"], reverse=True)
                    trades = all_trades[:limit]

                # Format trades
                for trade in trades:
                    result.append(
                        {
                            "id": trade["id"],
                            "symbol": trade["symbol"],
                            "side": trade["side"],
                            "amount": trade["amount"],
                            "price": trade["price"],
                            "cost": trade["cost"],
                            "fee": trade["fee"]["cost"] if trade["fee"] else 0,
                            "fee_currency": trade["fee"]["currency"]
                            if trade["fee"]
                            else None,
                            "timestamp": trade["timestamp"],
                        }
                    )
            else:
                logger.warning(
                    f"{self.exchange_name} does not support fetching trade history"
                )
                result = [
                    {"error": f"{self.exchange_name} does not support trade history"}
                ]

        except Exception as e:
            logger.error(
                f"Error fetching transaction history from {self.exchange_name}: {str(e)}"
            )
            result = [
                {
                    "error": f"Cannot connect to {self.exchange_name} to retrieve transaction history. Please check your API credentials and network connection."
                }
            ]

        return result


class CoinbaseClient:
    """
    Custom Coinbase client to manage API requests
    Supports both API key authentication and OAuth authentication
    """

    def __init__(
        self, api_key: str = None, api_secret: str = None, oauth_token: str = None
    ):
        # Store authentication method
        self.auth_method = "oauth" if oauth_token else "api_key"
        self.oauth_token = oauth_token

        # Handle API key auth if provided
        if api_key:
            # Handle CDP API keys (keys containing organization ID)
            if ":" in api_key:
                # Format is typically: organization_id:api_key
                parts = api_key.split(":")
                if len(parts) == 2:
                    self.organization_id = parts[0]
                    self.api_key = parts[1]
                else:
                    # If format is unexpected, use as-is
                    self.organization_id = None
                    self.api_key = api_key
            else:
                self.organization_id = None
                self.api_key = api_key

            self.api_secret = api_secret
        else:
            self.api_key = None
            self.api_secret = None
            self.organization_id = None

        self.api_url = "https://api.coinbase.com"
        self.api_version = "2021-08-08"

        # Initialize session for API requests
        import requests

        self.session = requests.Session()

    def _generate_signature(
        self, timestamp: str, method: str, request_path: str, body: str = ""
    ) -> str:
        """Generate signature for Coinbase API request"""
        message = timestamp + method + request_path + body
        signature = hmac.new(
            base64.b64decode(self.api_secret),
            message.encode("utf-8"),
            digestmod=hashlib.sha256,
        ).hexdigest()
        return signature

    def _request(
        self, method: str, endpoint: str, params: dict = None, data: dict = None
    ) -> dict:
        """Make request to Coinbase API using either API key or OAuth token"""
        url = f"{self.api_url}{endpoint}"

        # Add query parameters to URL if provided
        if params:
            query_string = urlencode(params)
            url = f"{url}?{query_string}"
            endpoint = f"{endpoint}?{query_string}"

        # Convert data to JSON string if provided
        body = ""
        if data:
            body = json.dumps(data)

        # Create headers based on authentication method
        headers = {
            "Content-Type": "application/json",
        }

        if self.auth_method == "oauth":
            # OAuth authentication
            headers["Authorization"] = f"Bearer {self.oauth_token}"
            headers["CB-VERSION"] = self.api_version
        else:
            # API key authentication
            timestamp = str(int(time.time()))
            signature = self._generate_signature(timestamp, method, endpoint, body)

            headers.update(
                {
                    "CB-ACCESS-KEY": self.api_key,
                    "CB-ACCESS-SIGN": signature,
                    "CB-ACCESS-TIMESTAMP": timestamp,
                    "CB-VERSION": self.api_version,
                }
            )

            # Add organization ID if present (for CDP API keys)
            if self.organization_id:
                headers["CB-ACCESS-PROJECT"] = self.organization_id

        # Make request
        response = self.session.request(method, url, headers=headers, data=body)

        # Handle errors
        if response.status_code != 200:
            error_msg = f"Coinbase API error: {response.status_code}, {response.text}"
            logger.error(error_msg)
            raise Exception(error_msg)

        return response.json()

    def get_accounts(self) -> dict:
        """Get all accounts (used to test connection)"""
        return self._request("GET", "/v2/accounts")

    def get_balance(self) -> dict:
        """Get balance information for all accounts"""
        accounts = self._request("GET", "/v2/accounts")
        return accounts

    def get_spot_price(self, currency_pair: str) -> float:
        """Get current spot price for a currency pair"""
        response = self._request("GET", f"/v2/prices/{currency_pair}/spot")
        return float(response["data"]["amount"])

    def get_exchange_rates(self, currency: str = "USD") -> dict:
        """Get exchange rates for a base currency"""
        response = self._request("GET", "/v2/exchange-rates", {"currency": currency})
        return response["data"]["rates"]

    def place_market_order(self, action: str, amount: str, currency: str) -> dict:
        """Place a market order to buy or sell"""
        data = {
            "type": "market",
            "side": action,  # 'buy' or 'sell'
            "product_id": f"{currency}-USD",
            "size": amount,  # Amount in base currency
        }
        return self._request("POST", "/v3/brokerage/orders", data=data)

    def place_limit_order(
        self, action: str, amount: str, price: str, currency: str
    ) -> dict:
        """Place a limit order to buy or sell"""
        data = {
            "type": "limit",
            "side": action,  # 'buy' or 'sell'
            "product_id": f"{currency}-USD",
            "price": price,
            "size": amount,
        }
        return self._request("POST", "/v3/brokerage/orders", data=data)

    def get_transaction_history(self, account_id: str, limit: int = 50) -> list:
        """Get transaction history for an account"""
        transactions = self._request(
            "GET", f"/v2/accounts/{account_id}/transactions", {"limit": limit}
        )
        return transactions["data"]


def get_wallet_balance(
    exchange_name: str,
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
    oauth_token: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Get wallet balances from the specified exchange.

    Args:
        exchange_name: Name of the exchange (e.g., 'binance', 'coinbase')
        api_key: API key for the exchange (optional if using OAuth)
        api_secret: API secret for the exchange (optional if using OAuth)
        oauth_token: OAuth access token for the exchange (optional)

    Returns:
        A dictionary of currency -> balance
    """
    client = ExchangeClientFactory.create_client(
        exchange_name, api_key, api_secret, oauth_token
    )

    if client:
        return client.get_wallet_balance()

    # Return error if client creation failed
    return {
        "error": (
            f"Cannot connect to {exchange_name}. "
            f"Please check your API credentials and network connection."
        )
    }


def get_current_prices(
    exchange_name: str,
    currencies: Optional[List[str]] = None,
    oauth_token: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Get current prices for currencies against USD or USDT.

    Args:
        exchange_name: Name of the exchange (e.g., 'binance', 'coinbase')
        currencies: List of currency symbols to get prices for (optional)
        oauth_token: OAuth access token for the exchange (optional)

    Returns:
        A dictionary of currency -> price
    """
    # Use API keys from environment if not using OAuth
    api_key = None
    api_secret = None

    if not oauth_token:
        if exchange_name.lower() in ["binance", "binanceus"]:
            api_key = API_KEYS["binance"]
            api_secret = API_KEYS["binance_secret"]
        elif exchange_name.lower() == "coinbase":
            api_key = API_KEYS["coinbase"]
            api_secret = API_KEYS["coinbase_secret"]

    # Create appropriate client
    client = ExchangeClientFactory.create_client(
        exchange_name, api_key, api_secret, oauth_token
    )

    if client:
        return client.get_current_prices(currencies)

    # If client creation failed or getting prices failed,
    # try to use alternative API directly
    result = ExchangeUtility.fallback_to_alternative_api(currencies)

    # Add error message if no results or missing currencies
    if not result or (currencies and not all(c in result for c in currencies)):
        missing = []
        if currencies:
            missing = [c for c in currencies if c not in result]

        error_msg = "Failed to retrieve current prices"
        if missing:
            error_msg += f" for: {', '.join(missing)}"

        logger.error(f"{error_msg}. Check network connection and API access.")

        # Set error flag in the result dictionary
        result[
            "error"
        ] = "Cannot connect to price data source. Please check your network connection."

    return result


def place_order(
    exchange_name: str,
    symbol: str,
    order_type: str,  # 'market', 'limit'
    side: str,  # 'buy', 'sell'
    amount: float,
    price: Optional[float] = None,  # Required for limit orders
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
    oauth_token: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Place an order on the exchange.

    Args:
        exchange_name: Name of the exchange (e.g., 'binance', 'coinbase')
        symbol: Trading pair symbol (e.g., 'BTC/USD', 'BTCUSD')
        order_type: Type of order ('market' or 'limit')
        side: Side of the order ('buy' or 'sell')
        amount: Amount to buy or sell
        price: Price for limit orders (optional)
        api_key: API key for the exchange (optional if using OAuth)
        api_secret: API secret for the exchange (optional if using OAuth)
        oauth_token: OAuth access token for the exchange (optional)

    Returns:
        Dictionary with order result information
    """
    # Create appropriate client
    client = ExchangeClientFactory.create_client(
        exchange_name, api_key, api_secret, oauth_token
    )

    if client:
        return client.place_order(symbol, order_type, side, amount, price)

    # If client creation failed
    return {
        "success": False,
        "order_id": None,
        "message": f"Failed to connect to {exchange_name}. Check your credentials.",
    }


def get_transaction_history(
    exchange_name: str,
    symbol: Optional[str] = None,
    limit: int = 50,
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
    oauth_token: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Get transaction history from the exchange.

    Args:
        exchange_name: Name of the exchange (e.g., 'binance', 'coinbase')
        symbol: Trading pair symbol (optional)
        limit: Maximum number of transactions to return
        api_key: API key for the exchange (optional if using OAuth)
        api_secret: API secret for the exchange (optional if using OAuth)
        oauth_token: OAuth access token for the exchange (optional)

    Returns:
        List of transaction dictionaries
    """
    # Create appropriate client
    client = ExchangeClientFactory.create_client(
        exchange_name, api_key, api_secret, oauth_token
    )

    if client:
        return client.get_transaction_history(symbol, limit)

    # If client creation failed
    return [
        {
            "error": f"Cannot connect to {exchange_name} to retrieve transaction history. "
            f"Please check your API credentials and network connection."
        }
    ]


def get_supported_exchanges() -> List[str]:
    """
    Get a list of supported exchanges.
    """
    # CCXT supported exchanges + special handling for Binance and Coinbase
    exchanges = ccxt.exchanges

    # Make sure Binance is in the list (we have special handling for it)
    if "binance" not in exchanges:
        exchanges.append("binance")

    # Add Binance.US explicitly for users in restricted regions
    if "binanceus" not in exchanges:
        exchanges.append("binanceus")

    # Add Coinbase explicitly (we have custom implementation)
    if "coinbase" not in exchanges:
        exchanges.append("coinbase")

    return sorted(exchanges)


class TestHelpers:
    """
    Helper functions to support tests
    """

    @staticmethod
    def get_wallet_balance_for_currency(
        exchange: str,
        currency: str,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Get wallet balance for a specific currency (for test compatibility)

        Args:
            exchange: Exchange name
            currency: Currency symbol
            api_key: API key
            api_secret: API secret

        Returns:
            Balance information for the currency
        """
        balances = get_wallet_balance(exchange, api_key, api_secret)
        if currency in balances:
            return balances[currency]
        return {"free": 0.0, "locked": 0.0, "total": 0.0}

    @staticmethod
    def get_market_price(
        exchange: str,
        symbol: str,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
    ) -> float:
        """
        Get market price for a specific symbol (for test compatibility)

        Args:
            exchange: Exchange name
            symbol: Trading pair symbol
            api_key: API key
            api_secret: API secret

        Returns:
            Current market price
        """
        client = ExchangeClientFactory.create_client(exchange, api_key, api_secret)
        if not client:
            return 0.0

        try:
            # Get prices for the currency from the symbol
            if "/" in symbol:
                currency = symbol.split("/")[0]
            elif symbol.endswith("USD") or symbol.endswith("USDT"):
                currency = symbol[:-3] if symbol.endswith("USD") else symbol[:-4]
            else:
                currency = symbol

            prices = client.get_current_prices([currency])
            return prices.get(currency, 0.0)
        except Exception as e:
            logger.error(f"Error getting price for {symbol}: {str(e)}")
            return 0.0

    @staticmethod
    def execute_trade(
        exchange: str,
        symbol: str,
        trade_type: str,
        quantity: float,
        price: float,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute a trade (for test compatibility)

        Args:
            exchange: Exchange name
            symbol: Trading pair symbol
            trade_type: Side of the trade (buy/sell)
            quantity: Amount to trade
            price: Price for the trade
            api_key: API key
            api_secret: API secret

        Returns:
            Trade result information
        """
        result = place_order(
            exchange_name=exchange,
            api_key=api_key,
            api_secret=api_secret,
            symbol=symbol,
            order_type="limit",
            side=trade_type,
            amount=quantity,
            price=price,
        )

        if result["success"]:
            return {
                "status": "FILLED",
                "quantity": str(quantity),
                "price": str(price * quantity),
            }
        return {"status": "FAILED", "message": result["message"]}


# Legacy functions to maintain backward compatibility
def get_wallet_balance_for_currency(
    exchange: str, currency: str, api_key: str, api_secret: str
) -> Dict[str, Any]:
    """
    Legacy wrapper for TestHelpers.get_wallet_balance_for_currency
    """
    return TestHelpers.get_wallet_balance_for_currency(
        exchange, currency, api_key, api_secret
    )


def get_market_price(
    exchange: str, symbol: str, api_key: str, api_secret: str
) -> Dict[str, Any]:
    """
    Legacy wrapper for TestHelpers.get_market_price
    """
    return TestHelpers.get_market_price(exchange, symbol, api_key, api_secret)


def execute_trade(
    exchange: str,
    symbol: str,
    trade_type: str,
    quantity: float,
    price: float,
    api_key: str,
    api_secret: str,
) -> Dict[str, Any]:
    """
    Legacy wrapper for TestHelpers.execute_trade
    """
    return TestHelpers.execute_trade(
        exchange, symbol, trade_type, quantity, price, api_key, api_secret
    )
