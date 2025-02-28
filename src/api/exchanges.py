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
            # Use python-binance library
            client = BinanceClient(api_key, api_secret)
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
        logger.error(f"Error connecting to {exchange_name}: {str(e)}")
        return None

def get_wallet_balance(exchange_name: str, api_key: str, api_secret: str) -> Dict:
    """
    Get wallet balances from the specified exchange.
    Returns a dictionary of currency -> balance
    """
    result = {}
    
    try:
        if exchange_name.lower() == "binance":
            client = get_exchange_client(exchange_name, api_key, api_secret)
            if not client:
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
        logger.error(f"Error fetching balances from {exchange_name}: {str(e)}")
    
    return result

def get_current_prices(exchange_name: str, currencies: List[str] = None) -> Dict:
    """
    Get current prices for currencies against USD or USDT.
    """
    result = {}
    
    try:
        if exchange_name.lower() == "binance":
            # Use API key from environment or params
            api_key = os.getenv("BINANCE_API_KEY", "")
            api_secret = os.getenv("BINANCE_SECRET_KEY", "")
            
            client = BinanceClient(api_key, api_secret)
            
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
        if exchange_name.lower() == "binance":
            client = get_exchange_client(exchange_name, api_key, api_secret)
            if not client:
                result['message'] = "Failed to connect to Binance"
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
        result['message'] = f"Error placing order: {str(e)}"
    
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
        if exchange_name.lower() == "binance":
            client = get_exchange_client(exchange_name, api_key, api_secret)
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
    
    return sorted(exchanges)