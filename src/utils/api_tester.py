import requests
import json
import os
import logging
import time
from typing import Dict, Tuple, List, Optional
from dotenv import load_dotenv
import yfinance as yf
import openai
from openai import OpenAI
import ccxt

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class APITester:
    """Class for testing connections to various financial and AI APIs"""
    
    @staticmethod
    def test_openai(api_key: str) -> Tuple[bool, str]:
        """Test connection to OpenAI API"""
        try:
            client = OpenAI(api_key=api_key)
            # Try a simple completion to check if the API key works
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hello, this is a test!"}],
                max_tokens=10
            )
            return True, "Successfully connected to OpenAI API"
        except Exception as e:
            logger.error(f"OpenAI API test failed: {str(e)}")
            return False, f"Failed to connect to OpenAI API: {str(e)}"
    
    @staticmethod
    def test_anthropic(api_key: str) -> Tuple[bool, str]:
        """Test connection to Anthropic API"""
        try:
            headers = {
                "x-api-key": api_key,
                "content-type": "application/json"
            }
            
            data = {
                "model": "claude-3-haiku-20240307",
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "Hello, this is a test!"}]
            }
            
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=data
            )
            
            if response.status_code == 200:
                return True, "Successfully connected to Anthropic API"
            else:
                return False, f"Anthropic API returned status code {response.status_code}: {response.text}"
        except Exception as e:
            logger.error(f"Anthropic API test failed: {str(e)}")
            return False, f"Failed to connect to Anthropic API: {str(e)}"
    
    @staticmethod
    def test_binance(api_key: str, api_secret: str) -> Tuple[bool, str]:
        """Test connection to Binance API"""
        try:
            from binance.client import Client
            
            # Try to connect to Binance.US first
            try:
                client = Client(api_key, api_secret, tld='us')
                # Test connection by requesting account info
                info = client.get_account()
                return True, "Successfully connected to Binance.US API"
            except Exception as e:
                # If US endpoint fails, try regular Binance
                if "restricted location" in str(e) or "geographical restrictions" in str(e):
                    # Don't try regular Binance as it will likely fail for the same reason
                    return False, f"Failed to connect to Binance.US API due to geographical restrictions: {str(e)}"
                
                try:
                    client = Client(api_key, api_secret)
                    # Test connection by requesting account info
                    info = client.get_account()
                    return True, "Successfully connected to Binance API"
                except Exception as e2:
                    return False, f"Failed to connect to both Binance.US and Binance APIs: {str(e2)}"
        except Exception as e:
            logger.error(f"Binance API test failed: {str(e)}")
            return False, f"Failed to connect to Binance API: {str(e)}"
    
    @staticmethod
    def test_coinbase(api_key: str, api_secret: str) -> Tuple[bool, str]:
        """Test connection to Coinbase API"""
        try:
            import hmac
            import hashlib
            import time
            import base64
            
            # Coinbase API endpoint for testing
            url = "https://api.coinbase.com/v2/user"
            
            # Create timestamp for the request
            timestamp = str(int(time.time()))
            
            # Create signature
            message = timestamp + "GET" + "/v2/user"
            
            # The API secret is Base64 encoded, so we need to decode it first
            try:
                secret_bytes = base64.b64decode(api_secret)
            except Exception:
                # If decoding fails, try using the raw string (for backward compatibility)
                secret_bytes = api_secret.encode('utf-8')
            
            signature = hmac.new(
                secret_bytes,
                message.encode('utf-8'),
                digestmod=hashlib.sha256
            ).hexdigest()
            
            # Set up headers
            headers = {
                "CB-ACCESS-KEY": api_key,
                "CB-ACCESS-SIGN": signature,
                "CB-ACCESS-TIMESTAMP": timestamp,
                "CB-VERSION": "2021-08-08"  # Updated to newer version
            }
            
            # Make request
            response = requests.get(url, headers=headers)
            
            if response.status_code == 200:
                # Get user information from response
                user_data = response.json()
                user_name = user_data.get('data', {}).get('name', 'User')
                return True, f"Successfully connected to Coinbase API (User: {user_name})"
            else:
                return False, f"Coinbase API returned status code {response.status_code}: {response.text}"
        except Exception as e:
            logger.error(f"Coinbase API test failed: {str(e)}")
            return False, f"Failed to connect to Coinbase API: {str(e)}"
    
    @staticmethod
    def test_alpha_vantage(api_key: str) -> Tuple[bool, str]:
        """Test connection to Alpha Vantage API"""
        try:
            # Try a simple request for IBM stock data
            url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=IBM&apikey={api_key}"
            response = requests.get(url)
            
            data = response.json()
            
            # Check if we got a proper response
            if "Global Quote" in data and len(data["Global Quote"]) > 0:
                return True, "Successfully connected to Alpha Vantage API"
            elif "Note" in data and "API call frequency" in data["Note"]:
                # API limit reached but connection works
                return True, "Successfully connected to Alpha Vantage API (note: API call frequency limit reached)"
            else:
                return False, f"Alpha Vantage API returned unexpected response: {data}"
        except Exception as e:
            logger.error(f"Alpha Vantage API test failed: {str(e)}")
            return False, f"Failed to connect to Alpha Vantage API: {str(e)}"
    
    @staticmethod
    def test_yahoo_finance() -> Tuple[bool, str]:
        """Test connection to Yahoo Finance API (yfinance)"""
        # For Yahoo Finance API tests, we'll just return success since:
        # 1. Yahoo Finance doesn't need API keys 
        # 2. It's primarily rate-limiting our test attempts
        # 3. The actual yfinance module will handle errors during real usage
        
        # Import socket to check basic internet connectivity
        import socket
        
        try:
            # Test if we can resolve yahoo's domain - this is a light test
            # that doesn't trigger rate limiting
            socket.gethostbyname("finance.yahoo.com")
            return True, "Yahoo Finance API available (no authentication required)"
        except Exception:
            # Check if we can resolve a common domain as a basic connectivity test
            try:
                socket.gethostbyname("google.com")
                return False, "Cannot resolve Yahoo Finance domain - service may be temporarily unavailable"
            except:
                return False, "Internet connectivity issue - please check your network connection"
            
    @staticmethod
    def test_yahoofinance() -> Tuple[bool, str]:
        """Alias for test_yahoo_finance"""
        return APITester.test_yahoo_finance()
    
    @staticmethod
    def test_coingecko() -> Tuple[bool, str]:
        """Test connection to CoinGecko API (free tier)"""
        try:
            # Try to get data for Bitcoin
            url = "https://api.coingecko.com/api/v3/coins/bitcoin?localization=false&tickers=false&market_data=true&community_data=false&developer_data=false"
            response = requests.get(url)
            
            if response.status_code == 200:
                data = response.json()
                if "market_data" in data and "current_price" in data["market_data"]:
                    price = data["market_data"]["current_price"]["usd"]
                    return True, f"Successfully connected to CoinGecko API (BTC price: ${price})"
                else:
                    return False, "CoinGecko API returned incomplete data"
            elif response.status_code == 429:
                return False, "CoinGecko API rate limit exceeded, try again later"
            else:
                return False, f"CoinGecko API returned status code {response.status_code}: {response.text}"
        except Exception as e:
            logger.error(f"CoinGecko API test failed: {str(e)}")
            return False, f"Failed to connect to CoinGecko API: {str(e)}"
    
    @staticmethod
    def test_gemini(api_key: str, api_secret: str) -> Tuple[bool, str]:
        """Test connection to Gemini API"""
        try:
            import hmac
            import hashlib
            import base64
            import json
            
            # Base URL for Gemini API
            url = "https://api.gemini.com/v1/account"
            
            # Create nonce and payload
            nonce = int(time.time() * 1000)
            payload = {
                "request": "/v1/account",
                "nonce": nonce
            }
            
            encoded_payload = json.dumps(payload).encode()
            b64 = base64.b64encode(encoded_payload)
            signature = hmac.new(api_secret.encode(), b64, hashlib.sha384).hexdigest()
            
            # Set up headers
            headers = {
                'Content-Type': "text/plain",
                'Content-Length': "0",
                'X-GEMINI-APIKEY': api_key,
                'X-GEMINI-PAYLOAD': b64.decode(),
                'X-GEMINI-SIGNATURE': signature,
                'Cache-Control': "no-cache"
            }
            
            # Make request
            response = requests.post(url, headers=headers)
            
            if response.status_code == 200:
                return True, "Successfully connected to Gemini API"
            else:
                return False, f"Gemini API returned status code {response.status_code}: {response.text}"
        except Exception as e:
            logger.error(f"Gemini API test failed: {str(e)}")
            return False, f"Failed to connect to Gemini API: {str(e)}"
    
    @staticmethod
    def test_kraken(api_key: str, api_secret: str) -> Tuple[bool, str]:
        """Test connection to Kraken API"""
        try:
            # Use CCXT for easier testing
            kraken = ccxt.kraken({
                'apiKey': api_key,
                'secret': api_secret,
                'enableRateLimit': True
            })
            
            # Try to fetch account balance
            balance = kraken.fetch_balance()
            
            if balance and 'total' in balance:
                return True, "Successfully connected to Kraken API"
            else:
                return False, "Kraken API returned incomplete data"
        except Exception as e:
            logger.error(f"Kraken API test failed: {str(e)}")
            return False, f"Failed to connect to Kraken API: {str(e)}"
    
    @staticmethod
    def test_coinmarketcap(api_key: str) -> Tuple[bool, str]:
        """Test connection to CoinMarketCap API"""
        try:
            url = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest'
            headers = {
                'X-CMC_PRO_API_KEY': api_key,
                'Accept': 'application/json'
            }
            params = {
                'start': '1',
                'limit': '1',
                'convert': 'USD'
            }
            
            response = requests.get(url, headers=headers, params=params)
            
            if response.status_code == 200:
                data = response.json()
                if 'data' in data and len(data['data']) > 0:
                    return True, "Successfully connected to CoinMarketCap API"
                else:
                    return False, "CoinMarketCap API returned incomplete data"
            else:
                return False, f"CoinMarketCap API returned status code {response.status_code}: {response.text}"
        except Exception as e:
            logger.error(f"CoinMarketCap API test failed: {str(e)}")
            return False, f"Failed to connect to CoinMarketCap API: {str(e)}"
    
    @staticmethod
    def test_hugging_face(api_key: str) -> Tuple[bool, str]:
        """Test connection to Hugging Face API"""
        try:
            # Try a simple inference API call
            url = "https://api-inference.huggingface.co/models/gpt2"
            headers = {"Authorization": f"Bearer {api_key}"}
            payload = {"inputs": "Hello, I'm a test"}
            
            response = requests.post(url, headers=headers, json=payload)
            
            if response.status_code == 200:
                return True, "Successfully connected to Hugging Face API"
            else:
                return False, f"Hugging Face API returned status code {response.status_code}: {response.text}"
        except Exception as e:
            logger.error(f"Hugging Face API test failed: {str(e)}")
            return False, f"Failed to connect to Hugging Face API: {str(e)}"
    
    @staticmethod
    def test_fmp(api_key: str) -> Tuple[bool, str]:
        """Test connection to Financial Modeling Prep API"""
        try:
            url = f"https://financialmodelingprep.com/api/v3/profile/AAPL?apikey={api_key}"
            response = requests.get(url)
            
            if response.status_code == 200:
                data = response.json()
                if len(data) > 0:
                    return True, "Successfully connected to Financial Modeling Prep API"
                else:
                    return False, "Financial Modeling Prep API returned empty data"
            else:
                return False, f"Financial Modeling Prep API returned status code {response.status_code}: {response.text}"
        except Exception as e:
            logger.error(f"Financial Modeling Prep API test failed: {str(e)}")
            return False, f"Failed to connect to Financial Modeling Prep API: {str(e)}"
    
    @staticmethod
    def test_polygon(api_key: str) -> Tuple[bool, str]:
        """Test connection to Polygon.io API"""
        try:
            url = f"https://api.polygon.io/v2/aggs/ticker/AAPL/prev?apiKey={api_key}"
            response = requests.get(url)
            
            if response.status_code == 200:
                data = response.json()
                if 'results' in data:
                    return True, "Successfully connected to Polygon.io API"
                else:
                    return False, "Polygon.io API returned incomplete data"
            else:
                return False, f"Polygon.io API returned status code {response.status_code}: {response.text}"
        except Exception as e:
            logger.error(f"Polygon.io API test failed: {str(e)}")
            return False, f"Failed to connect to Polygon.io API: {str(e)}"
    
    @staticmethod
    def test_ollama(base_url: str = "http://localhost:11434") -> Tuple[bool, str]:
        """Test connection to Ollama API"""
        try:
            # Make sure URL ends with trailing slash
            if not base_url.endswith('/'):
                base_url += '/'
                
            # Try to list available models
            url = f"{base_url}api/tags"
            response = requests.get(url)
            
            if response.status_code == 200:
                data = response.json()
                if 'models' in data:
                    models = [model['name'] for model in data['models']]
                    return True, f"Successfully connected to Ollama API (Available models: {', '.join(models[:3])}...)"
                else:
                    return True, "Successfully connected to Ollama API"
            else:
                return False, f"Ollama API returned status code {response.status_code}: {response.text}"
        except Exception as e:
            logger.error(f"Ollama API test failed: {str(e)}")
            return False, f"Failed to connect to Ollama API: {str(e)}"

    @staticmethod
    def get_api_config_info() -> List[Dict]:
        """Get configuration information for all supported APIs"""
        return [
            {
                "name": "OpenAI",
                "service_id": "openai",
                "description": "Access to GPT models like GPT-3.5, GPT-4 for text generation, analysis and insights",
                "needs_key": True,
                "needs_secret": False,
                "env_var_key": "OPENAI_API_KEY",
                "env_var_secret": None,
                "category": "AI/LLM",
                "website": "https://openai.com/",
                "api_docs": "https://platform.openai.com/docs/api-reference",
            },
            {
                "name": "Anthropic",
                "service_id": "anthropic",
                "description": "Access to Claude models for AI and natural language processing",
                "needs_key": True,
                "needs_secret": False,
                "env_var_key": "ANTHROPIC_API_KEY",
                "env_var_secret": None,
                "category": "AI/LLM",
                "website": "https://www.anthropic.com/",
                "api_docs": "https://docs.anthropic.com/claude/reference/",
            },
            {
                "name": "Binance",
                "service_id": "binance",
                "description": "Access to Binance cryptocurrency exchange (use Binance.US in restricted regions)",
                "needs_key": True,
                "needs_secret": True,
                "env_var_key": "BINANCE_API_KEY",
                "env_var_secret": "BINANCE_SECRET_KEY",
                "category": "Crypto Exchange",
                "website": "https://www.binance.com/",
                "api_docs": "https://binance-docs.github.io/apidocs/",
            },
            {
                "name": "Coinbase",
                "service_id": "coinbase",
                "description": "Access to Coinbase cryptocurrency exchange",
                "needs_key": True,
                "needs_secret": True,
                "env_var_key": "COINBASE_API_KEY",
                "env_var_secret": "COINBASE_SECRET_KEY",
                "category": "Crypto Exchange",
                "website": "https://www.coinbase.com/",
                "api_docs": "https://docs.cloud.coinbase.com/sign-in-with-coinbase/docs/",
            },
            {
                "name": "Alpha Vantage",
                "service_id": "alphavantage",
                "description": "Real-time and historical stock, forex, and crypto data",
                "needs_key": True,
                "needs_secret": False,
                "env_var_key": "ALPHA_VANTAGE_KEY",
                "env_var_secret": None,
                "category": "Financial Data",
                "website": "https://www.alphavantage.co/",
                "api_docs": "https://www.alphavantage.co/documentation/",
            },
            {
                "name": "Yahoo Finance",
                "service_id": "yahoofinance",
                "description": "Stock, ETF, and crypto market data (via yfinance, no API key required)",
                "needs_key": False,
                "needs_secret": False,
                "env_var_key": None,
                "env_var_secret": None,
                "category": "Financial Data",
                "website": "https://finance.yahoo.com/",
                "api_docs": "https://pypi.org/project/yfinance/",
            },
            {
                "name": "CoinGecko",
                "service_id": "coingecko",
                "description": "Cryptocurrency data and market information (free tier)",
                "needs_key": False,
                "needs_secret": False,
                "env_var_key": None,
                "env_var_secret": None,
                "category": "Crypto Data",
                "website": "https://www.coingecko.com/",
                "api_docs": "https://www.coingecko.com/api/documentation",
            },
            {
                "name": "Gemini",
                "service_id": "gemini",
                "description": "Access to Gemini cryptocurrency exchange",
                "needs_key": True,
                "needs_secret": True,
                "env_var_key": "GEMINI_API_KEY",
                "env_var_secret": "GEMINI_API_SECRET",
                "category": "Crypto Exchange",
                "website": "https://www.gemini.com/",
                "api_docs": "https://docs.gemini.com/",
            },
            {
                "name": "Kraken",
                "service_id": "kraken",
                "description": "Access to Kraken cryptocurrency exchange",
                "needs_key": True,
                "needs_secret": True,
                "env_var_key": "KRAKEN_API_KEY",
                "env_var_secret": "KRAKEN_SECRET_KEY",
                "category": "Crypto Exchange",
                "website": "https://www.kraken.com/",
                "api_docs": "https://docs.kraken.com/rest/",
            },
            {
                "name": "CoinMarketCap",
                "service_id": "coinmarketcap",
                "description": "Comprehensive cryptocurrency market data",
                "needs_key": True,
                "needs_secret": False,
                "env_var_key": "COINMARKETCAP_API_KEY",
                "env_var_secret": None,
                "category": "Crypto Data",
                "website": "https://coinmarketcap.com/",
                "api_docs": "https://coinmarketcap.com/api/documentation/v1/",
            },
            {
                "name": "Hugging Face",
                "service_id": "huggingface",
                "description": "Access to open source AI models for NLP, image generation and more",
                "needs_key": True,
                "needs_secret": False,
                "env_var_key": "HUGGINGFACE_API_KEY",
                "env_var_secret": None,
                "category": "AI/LLM",
                "website": "https://huggingface.co/",
                "api_docs": "https://huggingface.co/docs/api-inference/",
            },
            {
                "name": "Financial Modeling Prep",
                "service_id": "fmp",
                "description": "Financial data, statements, and real-time stock and crypto prices",
                "needs_key": True,
                "needs_secret": False,
                "env_var_key": "FMP_API_KEY",
                "env_var_secret": None,
                "category": "Financial Data",
                "website": "https://financialmodelingprep.com/",
                "api_docs": "https://site.financialmodelingprep.com/developer/docs/",
            },
            {
                "name": "Polygon.io",
                "service_id": "polygon",
                "description": "Real-time and historical stock, options, forex and crypto data",
                "needs_key": True,
                "needs_secret": False,
                "env_var_key": "POLYGON_API_KEY",
                "env_var_secret": None,
                "category": "Financial Data",
                "website": "https://polygon.io/",
                "api_docs": "https://polygon.io/docs/",
            },
            {
                "name": "Ollama",
                "service_id": "ollama",
                "description": "Local LLM hosting platform (requires Ollama installation)",
                "needs_key": False,
                "needs_secret": False,
                "env_var_key": "OLLAMA_BASE_URL",
                "env_var_secret": None,
                "is_url": True,
                "default_url": "http://localhost:11434",
                "category": "AI/LLM",
                "website": "https://ollama.com/",
                "api_docs": "https://github.com/ollama/ollama/blob/main/docs/api.md",
            },
        ]