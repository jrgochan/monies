import ccxt
from binance.client import Client as BinanceClient
import time
import logging
from typing import Dict, List, Tuple, Optional
from dotenv import load_dotenv
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def get_exchange_client(exchange_name: str, api_key: str, api_secret: str):
    """
    Get the appropriate exchange client based on exchange name.
    """
    if not api_key or not api_secret:
        logger.error(f"Missing API credentials for {exchange_name}")
        return None
    
    try:
        if exchange_name.lower() == "binance":
            # Use python-binance library with US endpoint for Binance
            client = BinanceClient(api_key, api_secret, tld='us')
            # Test connection
            client.get_account()
            return client
        elif exchange_name.lower() == "binanceus":
            # Use python-binance library with explicit US endpoint
            client = BinanceClient(api_key, api_secret, tld='us')
            # Test connection
            client.get_account()
            return client
        else:
            # Use CCXT for other exchanges
            exchange_class = getattr(ccxt, exchange_name.lower())
            exchange = exchange_class({
                'apiKey': api_key,
                'secret': api_secret,
                'enableRateLimit': True
            })
            # Test connection
            exchange.fetch_balance()
            return exchange
    except Exception as e:
        error_msg = str(e)
        if "restricted location" in error_msg or "geographical restrictions" in error_msg:
            logger.error(f"Binance access error: Service unavailable due to geographical restrictions. Using Binance.US or a VPN may resolve this issue.")
        else:
            logger.error(f"Error connecting to {exchange_name}: {error_msg}")
        return None

def get_wallet_balance(exchange_name: str, api_key: str, api_secret: str) -> Dict:
    """
    Get wallet balances from the specified exchange.
    Returns a dictionary of currency -> balance
    """
    result = {}
    
    try:
        if exchange_name.lower() in ["binance", "binanceus"]:
            # Always use binanceus mode regardless of what was selected
            client = get_exchange_client("binanceus", api_key, api_secret)
            if not client:
                logger.warning(f"{exchange_name} access failed. Check API keys are from Binance.US.")
                return result
            
            # Get account info from Binance
            account = client.get_account()
            
            # Extract non-zero balances
            for asset in account['balances']:
                currency = asset['asset']
                free = float(asset['free'])
                locked = float(asset['locked'])
                total = free + locked
                
                if total > 0:
                    result[currency] = {
                        'free': free,
                        'locked': locked,
                        'total': total
                    }
        else:
            # Use CCXT for other exchanges
            exchange = get_exchange_client(exchange_name, api_key, api_secret)
            if not exchange:
                return result
            
            # Get balances using CCXT
            balance = exchange.fetch_balance()
            
            # Extract non-zero balances
            for currency, amount in balance['total'].items():
                if amount > 0:
                    result[currency] = {
                        'free': balance['free'].get(currency, 0),
                        'locked': balance['used'].get(currency, 0),
                        'total': amount
                    }
    
    except Exception as e:
        error_msg = str(e)
        if "restricted location" in error_msg or "geographical restrictions" in error_msg:
            logger.error(f"Access error: Service unavailable due to geographical restrictions. Make sure your API keys are from Binance.US, not regular Binance.")
        else:
            logger.error(f"Error fetching balances from {exchange_name}: {error_msg}")
    
    # If we have no results, return an empty dictionary with error flag
    if not result:
        logger.error(f"Failed to retrieve wallet data from {exchange_name}. Check API credentials and network connection.")
        # Return empty result with error indicator
        result = {"error": f"Cannot connect to {exchange_name}. Please check your API credentials and network connection."}
    
    return result

def get_current_prices(exchange_name: str, currencies: List[str] = None) -> Dict:
    """
    Get current prices for currencies against USD or USDT.
    """
    result = {}
    
    try:
        # Always use binanceus for Binance
        if exchange_name.lower() in ["binance", "binanceus"]:
            # Use API key from environment with US endpoint
            api_key = os.getenv("BINANCE_API_KEY", "")
            api_secret = os.getenv("BINANCE_SECRET_KEY", "")
            
            # Create client with US endpoint
            client = BinanceClient(api_key, api_secret, tld='us')
            
            # Get all ticker prices
            all_tickers = client.get_all_tickers()
            
            # Filter for USD(T) pairs if currencies specified
            for ticker in all_tickers:
                symbol = ticker['symbol']
                if symbol.endswith('USDT'):
                    base_currency = symbol[:-4]
                    
                    if currencies is None or base_currency in currencies:
                        result[base_currency] = float(ticker['price'])
        else:
            # Use CCXT for other exchanges
            exchange_class = getattr(ccxt, exchange_name.lower(), None)
            if not exchange_class:
                logger.error(f"Unsupported exchange: {exchange_name}")
                return result
            
            exchange = exchange_class({'enableRateLimit': True})
            
            # Get ticker prices
            tickers = exchange.fetch_tickers()
            
            # Filter for USD(T) pairs if currencies specified
            for symbol, ticker in tickers.items():
                if '/USDT' in symbol or '/USD' in symbol:
                    base_currency = symbol.split('/')[0]
                    
                    if currencies is None or base_currency in currencies:
                        result[base_currency] = ticker['last']
    
    except Exception as e:
        logger.error(f"Error fetching prices from {exchange_name}: {str(e)}")
    
    # If we have no results or need specific currencies, try alternative API
    if not result or (currencies and not all(c in result for c in currencies)):
        try:
            # Try CoinGecko API as an alternative source
            if currencies:
                crypto_ids = [c.lower() for c in currencies]
                ids_param = ",".join(crypto_ids)
                api_url = f"https://api.coingecko.com/api/v3/simple/price?ids={ids_param}&vs_currencies=usd"
                response = requests.get(api_url)
                
                if response.status_code == 200:
                    data = response.json()
                    for crypto_id, price_data in data.items():
                        # Convert to uppercase for consistency
                        crypto_symbol = crypto_id.upper()
                        if "usd" in price_data:
                            result[crypto_symbol] = price_data["usd"]
            
            # If still missing currencies, fall back to defaults for those
            missing_currencies = [c for c in currencies if c not in result] if currencies else []
            if missing_currencies:
                logger.info(f"Missing price data for: {missing_currencies}, using defaults")
        except Exception as e:
            logger.error(f"Error fetching from alternative API: {str(e)}")
    
    # If we have no results or are missing requested currencies, return error
    if not result or (currencies and not all(c in result for c in currencies)):
        missing = []
        if currencies:
            missing = [c for c in currencies if c not in result]
        
        error_msg = "Failed to retrieve current prices"
        if missing:
            error_msg += f" for: {', '.join(missing)}"
        
        logger.error(f"{error_msg}. Check network connection and API access.")
        
        # Set error flag in the result dictionary
        result["error"] = f"Cannot connect to price data source. Please check your network connection."
    
    return result

def place_order(
    exchange_name: str, 
    api_key: str, 
    api_secret: str,
    symbol: str,
    order_type: str,  # 'market', 'limit'
    side: str,  # 'buy', 'sell'
    amount: float,
    price: float = None  # Required for limit orders
) -> Dict:
    """
    Place an order on the exchange.
    """
    result = {
        'success': False,
        'order_id': None,
        'message': ''
    }
    
    try:
        # Always use binanceus for Binance
        if exchange_name.lower() in ["binance", "binanceus"]:
            # Use binanceus client
            client = get_exchange_client("binanceus", api_key, api_secret)
            if not client:
                result['message'] = "Failed to connect to Binance.US"
                return result
            
            # Format symbol for Binance (no slash)
            symbol = symbol.replace('/', '')
            
            # Place order based on type
            if order_type.lower() == 'market':
                if side.lower() == 'buy':
                    order = client.order_market_buy(
                        symbol=symbol,
                        quantity=amount
                    )
                else:
                    order = client.order_market_sell(
                        symbol=symbol,
                        quantity=amount
                    )
            elif order_type.lower() == 'limit':
                if not price:
                    result['message'] = "Price is required for limit orders"
                    return result
                
                if side.lower() == 'buy':
                    order = client.order_limit_buy(
                        symbol=symbol,
                        quantity=amount,
                        price=price
                    )
                else:
                    order = client.order_limit_sell(
                        symbol=symbol,
                        quantity=amount,
                        price=price
                    )
            else:
                result['message'] = f"Unsupported order type: {order_type}"
                return result
            
            result['success'] = True
            result['order_id'] = order['orderId']
            result['message'] = "Order placed successfully"
            
        else:
            # Use CCXT for other exchanges
            exchange = get_exchange_client(exchange_name, api_key, api_secret)
            if not exchange:
                result['message'] = f"Failed to connect to {exchange_name}"
                return result
            
            # Place order
            if order_type.lower() == 'market':
                order = exchange.create_market_order(
                    symbol=symbol,
                    side=side.lower(),
                    amount=amount
                )
            elif order_type.lower() == 'limit':
                if not price:
                    result['message'] = "Price is required for limit orders"
                    return result
                
                order = exchange.create_limit_order(
                    symbol=symbol,
                    side=side.lower(),
                    amount=amount,
                    price=price
                )
            else:
                result['message'] = f"Unsupported order type: {order_type}"
                return result
            
            result['success'] = True
            result['order_id'] = order['id']
            result['message'] = "Order placed successfully"
    
    except Exception as e:
        error_msg = str(e)
        # Check for geographic restriction errors
        if "restricted location" in error_msg or "geographical restrictions" in error_msg:
            result['message'] = "Error placing order: Service unavailable due to geographical restrictions. Make sure you're using Binance.US API keys."
        else:
            result['message'] = f"Error placing order: {error_msg}"
    
    return result

def get_transaction_history(
    exchange_name: str, 
    api_key: str, 
    api_secret: str,
    symbol: str = None,
    limit: int = 50
) -> List[Dict]:
    """
    Get transaction history from the exchange.
    """
    result = []
    
    try:
        # Always use binanceus for Binance
        if exchange_name.lower() in ["binance", "binanceus"]:
            # Always use the US endpoint
            client = get_exchange_client("binanceus", api_key, api_secret)
            if not client:
                return result
            
            # Format symbol for Binance if provided
            binance_symbol = symbol.replace('/', '') if symbol else None
            
            # Get trade history
            if binance_symbol:
                trades = client.get_my_trades(symbol=binance_symbol, limit=limit)
            else:
                # Get trades for multiple symbols
                all_trades = []
                tickers = client.get_all_tickers()
                for ticker in tickers[:10]:  # Limit to first 10 for performance
                    try:
                        symbol_trades = client.get_my_trades(symbol=ticker['symbol'], limit=10)
                        all_trades.extend(symbol_trades)
                    except:
                        # Skip if error for this symbol
                        pass
                
                # Sort by time and limit
                all_trades.sort(key=lambda x: x['time'], reverse=True)
                trades = all_trades[:limit]
            
            # Format trades
            for trade in trades:
                result.append({
                    'id': trade['id'],
                    'symbol': trade['symbol'],
                    'side': 'buy' if trade['isBuyer'] else 'sell',
                    'amount': float(trade['qty']),
                    'price': float(trade['price']),
                    'cost': float(trade['quoteQty']),
                    'fee': float(trade['commission']),
                    'fee_currency': trade['commissionAsset'],
                    'timestamp': trade['time']
                })
        else:
            # Use CCXT for other exchanges
            exchange = get_exchange_client(exchange_name, api_key, api_secret)
            if not exchange:
                return result
            
            # Check if fetchMyTrades is supported
            if exchange.has['fetchMyTrades']:
                # Get trades
                if symbol:
                    trades = exchange.fetch_my_trades(symbol=symbol, limit=limit)
                else:
                    # Get trades for multiple symbols
                    all_trades = []
                    markets = exchange.load_markets()
                    for symbol, market in list(markets.items())[:10]:  # Limit to first 10 for performance
                        if '/USDT' in symbol or '/USD' in symbol:
                            try:
                                symbol_trades = exchange.fetch_my_trades(symbol=symbol, limit=10)
                                all_trades.extend(symbol_trades)
                            except:
                                # Skip if error for this symbol
                                pass
                    
                    # Sort by time and limit
                    all_trades.sort(key=lambda x: x['timestamp'], reverse=True)
                    trades = all_trades[:limit]
                
                # Format trades
                for trade in trades:
                    result.append({
                        'id': trade['id'],
                        'symbol': trade['symbol'],
                        'side': trade['side'],
                        'amount': trade['amount'],
                        'price': trade['price'],
                        'cost': trade['cost'],
                        'fee': trade['fee']['cost'] if trade['fee'] else 0,
                        'fee_currency': trade['fee']['currency'] if trade['fee'] else None,
                        'timestamp': trade['timestamp']
                    })
            else:
                logger.warning(f"{exchange_name} does not support fetching trade history")
    
    except Exception as e:
        logger.error(f"Error fetching transaction history from {exchange_name}: {str(e)}")
    
    # Return empty result with error indication if we couldn't get transactions
    if not result:
        logger.error(f"Failed to retrieve transaction history from {exchange_name}. Check API credentials and network connection.")
        # Add a special dictionary with error message instead of empty list
        # This will help UI display an error message
        result = [{"error": f"Cannot connect to {exchange_name} to retrieve transaction history. Please check your API credentials and network connection."}]
    
    return result

def get_supported_exchanges():
    """
    Get a list of supported exchanges.
    """
    # CCXT supported exchanges + special handling for Binance
    exchanges = ccxt.exchanges
    
    # Make sure Binance is in the list (we have special handling for it)
    if 'binance' not in exchanges:
        exchanges.append('binance')
    
    # Add Binance.US explicitly for users in restricted regions
    if 'binanceus' not in exchanges:
        exchanges.append('binanceus')
    
    return sorted(exchanges)


# Add functions to support tests
def get_wallet_balance_for_currency(exchange, currency, api_key, api_secret):
    """
    Get wallet balance for a specific currency (for test compatibility).
    """
    balances = get_wallet_balance(exchange, api_key, api_secret)
    if currency in balances:
        return balances[currency]
    return {'free': 0.0, 'locked': 0.0, 'total': 0.0}


def get_market_price(exchange, symbol, api_key, api_secret):
    """
    Get market price for a specific symbol (for test compatibility).
    """
    client = get_exchange_client(exchange, api_key, api_secret)
    if not client:
        return 0.0
    
    try:
        if exchange.lower() == "binance":
            ticker = client.get_symbol_ticker(symbol=symbol)
            return float(ticker['price'])
        else:
            ticker = client.fetch_ticker(symbol)
            return ticker['last']
    except Exception as e:
        logger.error(f"Error getting price for {symbol}: {str(e)}")
        return 0.0


def execute_trade(exchange, symbol, trade_type, quantity, price, api_key, api_secret):
    """
    Execute a trade (for test compatibility).
    """
    result = place_order(
        exchange_name=exchange,
        api_key=api_key,
        api_secret=api_secret,
        symbol=symbol,
        order_type='limit',
        side=trade_type,
        amount=quantity,
        price=price
    )
    
    if result['success']:
        return {
            'status': 'FILLED',
            'quantity': str(quantity),
            'price': str(price * quantity)
        }
    return {
        'status': 'FAILED',
        'message': result['message']
    }