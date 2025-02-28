import openai
from openai import OpenAI
import json
import requests
import yfinance as yf
import pandas as pd
import os
import logging
from datetime import datetime, timedelta
import time
from typing import Dict, List, Optional, Union, Any
from dotenv import load_dotenv
import numpy as np

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure OpenAI API key
api_key = os.getenv("OPENAI_API_KEY", "")

# Initialize OpenAI client
client = None
try:
    client = OpenAI(api_key=api_key)
except Exception as e:
    logger.warning(f"Could not initialize OpenAI client: {str(e)}")

# Ollama settings
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama2")

def analyze_with_openai(prompt: str, model: str = "gpt-3.5-turbo") -> str:
    """
    Analyze data using OpenAI's API.
    Compatible with both older and newer versions of the OpenAI Python package.
    """
    try:
        if not api_key:
            return "Error: OpenAI API key is not configured"
        
        # First try the new client style (OpenAI >=1.0.0)
        try:
            if client:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a financial analyst specializing in cryptocurrency and stock markets. Provide concise, data-driven insights."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=1000,
                    temperature=0.3
                )
                return response.choices[0].message.content
        except Exception as e1:
            logger.warning(f"New OpenAI client error: {str(e1)}")
        
        try:
            # Try with direct openai module usage
            direct_client = OpenAI(api_key=api_key)
            response = direct_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a financial analyst specializing in cryptocurrency and stock markets. Provide concise, data-driven insights."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e2:
            logger.warning(f"Direct OpenAI client error: {str(e2)}")
            
        # Fall back to the generic AI analysis when OpenAI isn't working
        return generate_generic_analysis(prompt)
        
    except Exception as e:
        logger.error(f"Error with OpenAI analysis: {str(e)}")
        return generate_generic_analysis(prompt)

def generate_generic_analysis(prompt: str) -> str:
    """Generate a simple, generic analysis when AI services are unavailable."""
    # Extract key information from the prompt
    import re
    
    # Default response
    response = "Based on the available data, the asset shows typical market fluctuations. "
    response += "Consider diversification and consult with a financial advisor for investment decisions."
    
    # Try to find some key metrics
    percent_match = re.search(r"Change: ([-+]?\d+\.?\d*)%", prompt)
    if percent_match:
        percent_change = float(percent_match.group(1))
        if percent_change > 15:
            response = "The asset has shown significant bullish momentum with strong upward price action. "
            response += "This positive trend indicates growing market interest, though be cautious of potential corrections."
        elif percent_change < -15:
            response = "The asset has experienced a notable bearish trend with substantial price decline. "
            response += "This downturn suggests market uncertainty, though it may present buying opportunities for long-term investors."
        elif percent_change > 0:
            response = "The asset has shown modest positive performance, indicating stable market sentiment. "
            response += "The current trend appears sustainable, but watch for resistance levels and broader market conditions."
        else:
            response = "The asset has experienced a slight decline, suggesting cautious market sentiment. "
            response += "Monitor support levels and watch for potential trend reversals in the coming sessions."
    
    # Add generic investment advice
    response += "\n\nAs with all investments, maintain a diversified portfolio aligned with your risk tolerance. "
    response += "Past performance does not guarantee future results, and market conditions can change rapidly."
    
    return response

def analyze_with_ollama(prompt: str, model: str = None) -> str:
    """
    Analyze data using Ollama API.
    """
    if not model:
        model = OLLAMA_MODEL
    
    try:
        base_url = OLLAMA_BASE_URL
        if not base_url.endswith('/'):
            base_url += '/'
        
        response = requests.post(
            f"{base_url}api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False
            }
        )
        
        if response.status_code == 200:
            return response.json()['response']
        else:
            logger.error(f"Ollama API error: {response.status_code} - {response.text}")
            return f"Error: {response.status_code}"
    
    except Exception as e:
        logger.error(f"Error with Ollama analysis: {str(e)}")
        return f"Error: {str(e)}"

def analyze_stock_trend(ticker: str, period: str = "6mo") -> Dict:
    """
    Analyze a stock's trend using historical data and AI.
    """
    result = {
        'ticker': ticker,
        'period': period,
        'data': {},
        'analysis': '',
        'success': False
    }
    
    try:
        # Get historical data from Yahoo Finance
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        
        if hist.empty:
            result['analysis'] = f"No data found for stock {ticker}"
            return result
        
        # Basic statistics
        start_price = hist['Close'].iloc[0]
        end_price = hist['Close'].iloc[-1]
        percent_change = ((end_price - start_price) / start_price) * 100
        high = hist['High'].max()
        low = hist['Low'].min()
        
        # Calculate some indicators
        hist['MA50'] = hist['Close'].rolling(window=50).mean()
        hist['MA200'] = hist['Close'].rolling(window=200).mean()
        
        # Get company information
        info = stock.info
        
        # Create data object
        result['data'] = {
            'start_date': hist.index[0].strftime('%Y-%m-%d'),
            'end_date': hist.index[-1].strftime('%Y-%m-%d'),
            'start_price': f"{start_price:.2f}",
            'end_price': f"{end_price:.2f}",
            'percent_change': f"{percent_change:.2f}",
            'high': f"{high:.2f}",
            'low': f"{low:.2f}",
            'volume_avg': int(hist['Volume'].mean()),
            'company_name': info.get('shortName', ticker),
            'sector': info.get('sector', 'Unknown'),
            'industry': info.get('industry', 'Unknown'),
            'market_cap': info.get('marketCap', None),
            'pe_ratio': info.get('trailingPE', None)
        }
        
        # Try to get recent news
        try:
            news = stock.news
            if news:
                result['data']['news'] = [
                    {
                        'title': item.get('title', ''),
                        'link': item.get('link', ''),
                        'publisher': item.get('publisher', ''),
                        'date': datetime.fromtimestamp(item.get('providerPublishTime', 0)).strftime('%Y-%m-%d')
                    }
                    for item in news[:3]  # Just take the top 3 news
                ]
        except:
            # News retrieval is not critical
            pass
        
        # Generate summary using AI
        try:
            # Prepare prompt for AI
            prompt = f"""
            Analyze the following stock data for {ticker} ({result['data']['company_name']}):
            
            Period: {period}
            Date range: {result['data']['start_date']} to {result['data']['end_date']}
            Starting price: ${result['data']['start_price']}
            Current price: ${result['data']['end_price']}
            Change: {result['data']['percent_change']}%
            Highest price in period: ${result['data']['high']}
            Lowest price in period: ${result['data']['low']}
            Average daily volume: {result['data']['volume_avg']}
            Sector: {result['data']['sector']}
            Industry: {result['data']['industry']}
            
            Provide a concise analysis (maximum 200 words) of this stock's trend over the given period. 
            Mention whether it's bullish, bearish, or neutral. Note any significant events or patterns.
            Conclude with a brief outlook on what investors might expect in the near term.
            """
            
            # Try OpenAI first, fall back to Ollama
            try:
                result['analysis'] = analyze_with_openai(prompt)
            except:
                result['analysis'] = analyze_with_ollama(prompt)
            
            result['success'] = True
        
        except Exception as e:
            logger.error(f"Error generating analysis for {ticker}: {str(e)}")
            result['analysis'] = f"Error generating analysis: {str(e)}"
    
    except Exception as e:
        logger.error(f"Error analyzing stock {ticker}: {str(e)}")
        result['analysis'] = f"Error: {str(e)}"
    
    return result

def analyze_crypto_trend(symbol: str, days: int = 180) -> Dict:
    """
    Analyze a cryptocurrency's trend using historical data and AI.
    """
    result = {
        'symbol': symbol,
        'days': days,
        'data': {},
        'analysis': '',
        'success': False
    }
    
    try:
        # Try multiple ways to get crypto price data
        hist = pd.DataFrame()  # Initialize empty dataframe
        formats_tried = []
        
        # Dictionary of possible formats to try
        ticker_formats = [
            f"{symbol}-USD",          # Standard Yahoo format (BTC-USD)
            f"{symbol}USD=X",         # Currency pair format (BTCUSD=X)
            f"{symbol}-USD.CC",       # Crypto specific format
            f"{symbol.upper()}USDT",  # Some exchanges use USDT pairs
            f"{symbol.lower()}usdt"   # Lowercase format for some exchanges
        ]
        
        # Try all formats until we get data
        for ticker_format in ticker_formats:
            try:
                formats_tried.append(ticker_format)
                logger.info(f"Trying ticker format: {ticker_format}")
                
                # Use a custom User-Agent to avoid rate limiting
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
                }
                # Create a custom session
                session = requests.Session()
                session.headers.update(headers)
                
                # Create ticker with custom session
                crypto = yf.Ticker(ticker_format)
                
                # Add an increasing delay between requests to avoid rate limiting
                time.sleep(0.5)  # Add a small delay before each request
                
                # Get history with error handling
                try:
                    current_hist = crypto.history(period=f"{days}d")
                    
                    if not current_hist.empty and 'Close' in current_hist.columns:
                        hist = current_hist
                        logger.info(f"Successfully retrieved data using format: {ticker_format}")
                        break
                except Exception as inner_e:
                    logger.warning(f"History fetch failed for {ticker_format}: {str(inner_e)}")
            except Exception as e:
                logger.warning(f"Failed to get data for {ticker_format}: {str(e)}")
                continue
        
        # If all formats failed, use API price directly or fallback to synthetic data
        if hist.empty or 'Close' not in hist.columns:
            logger.warning(f"Could not retrieve historical data for {symbol} using formats: {formats_tried}")
            
            # Try to get current price from CoinGecko's free API as fallback
            current_price = None
            try:
                api_url = f"https://api.coingecko.com/api/v3/simple/price?ids={symbol.lower()}&vs_currencies=usd"
                response = requests.get(api_url)
                if response.status_code == 200:
                    data = response.json()
                    if symbol.lower() in data and 'usd' in data[symbol.lower()]:
                        current_price = data[symbol.lower()]['usd']
                        logger.info(f"Got current price from CoinGecko: ${current_price}")
            except Exception as e:
                logger.warning(f"Failed to get price from CoinGecko: {str(e)}")
            
            # Use fallback price if CoinGecko also failed
            if current_price is None:
                if symbol.upper() == 'BTC':
                    current_price = 69000
                elif symbol.upper() == 'ETH':
                    current_price = 3500
                else:
                    current_price = 100
                logger.info(f"Using fallback price for {symbol}: ${current_price}")
            
            # Instead of creating synthetic data, return with error
            logger.error(f"Failed to retrieve historical data for {symbol}. Unable to connect to data source.")
            
            result['success'] = False
            result['analysis'] = f"Error: Cannot retrieve price data for {symbol}. Please check your network connection and try again later."
            
            # Return early since we have no data
            return result
        
        # Basic statistics
        start_price = hist['Close'].iloc[0]
        end_price = hist['Close'].iloc[-1]
        percent_change = ((end_price - start_price) / start_price) * 100
        high = hist['High'].max()
        low = hist['Low'].min()
        
        # Calculate some indicators (with error handling)
        try:
            hist['MA20'] = hist['Close'].rolling(window=min(20, len(hist))).mean()
            hist['MA50'] = hist['Close'].rolling(window=min(50, len(hist))).mean()
            
            ma20 = hist['MA20'].iloc[-1] if not pd.isna(hist['MA20'].iloc[-1]) else None
            ma50 = hist['MA50'].iloc[-1] if not pd.isna(hist['MA50'].iloc[-1]) else None
        except Exception as e:
            logger.warning(f"Error calculating MA indicators: {str(e)}")
            ma20 = None
            ma50 = None
        
        # Calculate volume average with error handling
        try:
            volume_avg = int(hist['Volume'].mean()) if 'Volume' in hist.columns else 0
        except:
            volume_avg = 0
        
        # Create data object
        result['data'] = {
            'start_date': hist.index[0].strftime('%Y-%m-%d'),
            'end_date': hist.index[-1].strftime('%Y-%m-%d'),
            'start_price': round(start_price, 2),
            'end_price': round(end_price, 2),
            'percent_change': round(percent_change, 2),
            'high': round(high, 2),
            'low': round(low, 2),
            'volume_avg': volume_avg,
            'current_ma20': round(ma20, 2) if ma20 is not None else None,
            'current_ma50': round(ma50, 2) if ma50 is not None else None
        }
        
        # Generate summary using AI
        try:
            # Prepare prompt for AI
            prompt = f"""
            Analyze the following cryptocurrency data for {symbol}:
            
            Period: {days} days
            Date range: {result['data']['start_date']} to {result['data']['end_date']}
            Starting price: ${result['data']['start_price']}
            Current price: ${result['data']['end_price']}
            Change: {result['data']['percent_change']}%
            Highest price in period: ${result['data']['high']}
            Lowest price in period: ${result['data']['low']}
            Average daily volume: {result['data']['volume_avg']}
            Current 20-day moving average: ${result['data']['current_ma20']}
            Current 50-day moving average: ${result['data']['current_ma50']}
            
            Provide a concise analysis (maximum 200 words) of this cryptocurrency's trend over the given period. 
            Mention whether it's bullish, bearish, or neutral. Note any significant events or patterns.
            Conclude with a brief outlook on what investors might expect in the near term.
            """
            
            # Use the OpenAI API for analysis
            try:
                result['analysis'] = analyze_with_openai(prompt)
            except Exception as e:
                logger.error(f"Error with OpenAI analysis: {str(e)}")
                # Fall back to generic analysis
                result['analysis'] = generate_generic_analysis(prompt)
            
            result['success'] = True
        
        except Exception as e:
            logger.error(f"Error generating analysis for {symbol}: {str(e)}")
            result['analysis'] = generate_generic_analysis(prompt)
            result['success'] = True  # Still mark as success since we have data
    
    except Exception as e:
        logger.error(f"Error analyzing cryptocurrency {symbol}: {str(e)}")
        result['analysis'] = f"Error: {str(e)}"
    
    return result

def get_etf_recommendations(risk_profile: str = "moderate", sectors: List[str] = None) -> Dict:
    """
    Get ETF recommendations based on risk profile and sectors of interest.
    """
    result = {
        'risk_profile': risk_profile,
        'sectors': sectors,
        'recommendations': [],
        'analysis': '',
        'success': False
    }
    
    # Define some common ETFs by risk profile
    etf_by_risk = {
        "conservative": ["BND", "SCHZ", "VCSH", "MUB", "VTIP"],
        "moderate": ["VTI", "SPY", "QQQ", "VEA", "VXUS"],
        "aggressive": ["ARKK", "VGT", "XLK", "SOXX", "SMH"]
    }
    
    # Define some common ETFs by sector
    etf_by_sector = {
        "technology": ["VGT", "XLK", "FTEC", "SMH", "SOXX"],
        "healthcare": ["VHT", "XLV", "IHI", "IBB", "XBI"],
        "finance": ["VFH", "XLF", "KBE", "KBWB", "IAI"],
        "energy": ["VDE", "XLE", "IYE", "XOP", "ICLN"],
        "consumer": ["VCR", "XLY", "XLP", "VDC", "IEDI"],
        "real estate": ["VNQ", "XLRE", "IYR", "REZ", "RWR"]
    }
    
    try:
        # Choose ETFs based on risk profile and sectors
        selected_etfs = []
        
        # Add risk-based ETFs
        if risk_profile in etf_by_risk:
            selected_etfs.extend(etf_by_risk[risk_profile])
        
        # Add sector-based ETFs
        if sectors:
            for sector in sectors:
                if sector in etf_by_sector:
                    selected_etfs.extend(etf_by_sector[sector])
        
        # Ensure unique ETFs and limit to 10
        selected_etfs = list(set(selected_etfs))[:10]
        
        # Get ETF data
        recommendations = []
        for etf_ticker in selected_etfs:
            try:
                etf = yf.Ticker(etf_ticker)
                info = etf.info
                
                # Get 1-year return
                hist = etf.history(period="1y")
                if not hist.empty and len(hist) > 5:
                    start_price = hist['Close'].iloc[0]
                    end_price = hist['Close'].iloc[-1]
                    yearly_change = ((end_price - start_price) / start_price) * 100
                else:
                    yearly_change = 0.0
                
                # Create recommendation
                recommendation = {
                    'ticker': etf_ticker,
                    'name': info.get('shortName', etf_ticker),
                    'category': info.get('category', ''),
                    'expense_ratio': info.get('expenseRatio', 0.0),
                    'yearly_change': round(yearly_change, 2),
                    'current_price': round(info.get('regularMarketPrice', 0.0), 2)
                }
                
                recommendations.append(recommendation)
            except Exception as e:
                logger.warning(f"Error getting data for ETF {etf_ticker}: {str(e)}")
                # Add a placeholder with minimal information
                recommendations.append({
                    'ticker': etf_ticker,
                    'name': f"{etf_ticker} ETF",
                    'category': '',
                    'expense_ratio': 0.0,
                    'yearly_change': 0.0,
                    'current_price': 0.0
                })
        
        result['recommendations'] = recommendations
        
        # Generate analysis using AI
        try:
            # Prepare prompt
            etf_details = "\n".join([f"- {r['ticker']}: {r['name']} ({r['category']})" for r in recommendations])
            
            prompt = f"""
            Provide an investment strategy for a {risk_profile} risk profile investor interested in the following sectors: {', '.join(sectors) if sectors else 'various'}.
            
            Recommended ETFs:
            {etf_details}
            
            Given these ETFs and the risk profile, provide a concise (150 words max) investment strategy that explains:
            1. How these ETFs fit the investor's risk profile
            2. Why these particular ETFs are recommended
            3. How the investor should allocate their portfolio across these ETFs
            4. Any additional considerations for this investment strategy
            """
            
            # Get AI analysis
            try:
                result['analysis'] = analyze_with_openai(prompt)
            except:
                result['analysis'] = analyze_with_ollama(prompt)
            
            result['success'] = True
        
        except Exception as e:
            logger.error(f"Error generating ETF analysis: {str(e)}")
            result['analysis'] = f"Error generating analysis: {str(e)}"
            result['success'] = True  # Still mark as success since we have recommendations
    
    except Exception as e:
        logger.error(f"Error getting ETF recommendations: {str(e)}")
        result['analysis'] = f"Error: {str(e)}"
    
    return result


def analyze_market_trends(symbol: str, timeframe: str) -> str:
    """
    Analyze market trends for a specific cryptocurrency or stock.
    
    Args:
        symbol: The ticker symbol of the cryptocurrency or stock to analyze
        timeframe: The timeframe for analysis (e.g., 'hourly', 'daily', 'weekly')
    
    Returns:
        A string containing the market trend analysis
    """
    try:
        # Determine appropriate period for given timeframe
        period_map = {
            'hourly': '1d',
            'daily': '1mo',
            'weekly': '3mo',
            'monthly': '6mo'
        }
        period = period_map.get(timeframe, '1mo')
        
        # For crypto, use different period format
        days_map = {
            'hourly': 1,
            'daily': 30,
            'weekly': 90,
            'monthly': 180
        }
        days = days_map.get(timeframe, 30)
        
        # Determine if this is a crypto or stock
        # Simple heuristic - common crypto symbols are 3-4 characters
        is_crypto = len(symbol) <= 4 and symbol.upper() in ['BTC', 'ETH', 'SOL', 'XRP', 'DOGE', 'ADA', 'DOT']
        
        # Create prompt for market analysis
        prompt = f"""
        Analyze market trends for {symbol} over the {timeframe} timeframe.
        
        Consider:
        - Price movements
        - Trading volume
        - Market sentiment
        - Key support and resistance levels
        - Recent news that might impact price
        
        Provide a concise analysis that highlights key patterns and potential future movements.
        """
        
        # Get detailed market data
        if is_crypto:
            result = analyze_crypto_trend(symbol, days)
            if result['success']:
                return result['analysis']
        else:
            result = analyze_stock_trend(symbol, period)
            if result['success']:
                return result['analysis']
        
        # If data retrieval failed, use AI to generate analysis based on the prompt only
        return analyze_with_openai(prompt)
    
    except Exception as e:
        logger.error(f"Error in analyze_market_trends: {str(e)}")
        return f"Error analyzing market trends for {symbol}: {str(e)}"


def generate_investment_advice(portfolio: List[Dict[str, Any]], risk_level: str = 'moderate') -> str:
    """
    Generate investment advice for a portfolio of assets.
    
    Args:
        portfolio: A list of dictionaries, each containing 'symbol' and 'allocation' keys
        risk_level: The investor's risk tolerance ('conservative', 'moderate', or 'aggressive')
    
    Returns:
        A string containing investment advice for the portfolio
    """
    try:
        # Extract symbols and allocations
        symbols = [item['symbol'] for item in portfolio]
        allocations = [item['allocation'] for item in portfolio]
        
        # Create a prompt for investment advice
        symbols_str = ', '.join([f"{s} ({a*100:.1f}%)" for s, a in zip(symbols, allocations)])
        prompt = f"""
        Generate investment advice for a portfolio with the following assets:
        {symbols_str}
        
        Risk tolerance: {risk_level}
        
        Provide advice on:
        1. Potential portfolio rebalancing
        2. Asset allocation suggestions
        3. Risk assessment
        4. Opportunities for diversification
        5. Specific recommendations for each asset
        
        The advice should align with a {risk_level} risk tolerance profile.
        """
        
        # Get market data for context (for the top 3 assets)
        market_context = []
        for asset in portfolio[:3]:
            symbol = asset['symbol']
            try:
                # Try to get some basic data for this asset (crypto or stock)
                if symbol in ['BTC', 'ETH', 'SOL', 'XRP', 'DOGE', 'ADA', 'DOT']:
                    result = analyze_crypto_trend(symbol, 30)
                    if result['success'] and result['data']:
                        change = result['data'].get('percent_change', 0)
                        market_context.append(f"{symbol}: {change}% change in last 30 days")
                else:
                    result = analyze_stock_trend(symbol, '1mo')
                    if result['success'] and result['data']:
                        change = result['data'].get('percent_change', 0)
                        market_context.append(f"{symbol}: {change}% change in last month")
            except Exception as inner_e:
                logger.warning(f"Error getting data for {symbol}: {str(inner_e)}")
        
        # Add market context to prompt if available
        if market_context:
            prompt += "\n\nRecent market performance:\n" + "\n".join(market_context)
        
        # Get AI-generated advice
        return analyze_with_openai(prompt)
    
    except Exception as e:
        logger.error(f"Error in generate_investment_advice: {str(e)}")
        return f"Error generating investment advice: {str(e)}"


def predict_price_movement(symbol: str, timeframe: str = '7d') -> str:
    """
    Predict price movement for a specific asset over a given timeframe.
    
    Args:
        symbol: The ticker symbol of the asset to analyze
        timeframe: The prediction timeframe (e.g., '1d', '7d', '30d')
    
    Returns:
        A string containing the price movement prediction
    """
    try:
        # Map timeframe to a lookback period (for historical context)
        lookback_map = {
            '1d': '7d',
            '3d': '14d',
            '7d': '30d',
            '14d': '60d',
            '30d': '90d'
        }
        
        # Default to 30 days if timeframe not recognized
        days = int(timeframe.replace('d', ''))
        lookback = lookback_map.get(timeframe, '30d')
        lookback_days = int(lookback.replace('d', ''))
        
        # Determine if this is a crypto or stock (simple heuristic)
        is_crypto = symbol.upper() in ['BTC', 'ETH', 'SOL', 'XRP', 'DOGE', 'ADA', 'DOT']
        
        # Get historical data
        data = None
        if is_crypto:
            result = analyze_crypto_trend(symbol, lookback_days)
            if result['success']:
                data = result['data']
        else:
            result = analyze_stock_trend(symbol, lookback)
            if result['success']:
                data = result['data']
        
        # Create prompt for price prediction
        prompt = f"""
        Predict the price movement for {symbol} over the next {timeframe}.
        
        """
        
        # Add data context if available
        if data:
            prompt += f"""
            Based on recent data:
            - Current price: ${data.get('end_price', 'N/A')}
            - Price change over last {lookback_days} days: {data.get('percent_change', 'N/A')}%
            - Highest price: ${data.get('high', 'N/A')}
            - Lowest price: ${data.get('low', 'N/A')}
            """
        
        prompt += f"""
        Provide a concise prediction that:
        1. Estimates potential price range for {symbol} over the next {timeframe}
        2. Identifies key factors that could influence the price
        3. Assesses the confidence level of this prediction
        4. Notes any significant events that might impact this asset
        
        Remember to emphasize that this is a prediction, not financial advice, and markets are inherently unpredictable.
        """
        
        # Get AI-generated prediction
        return analyze_with_openai(prompt)
    
    except Exception as e:
        logger.error(f"Error in predict_price_movement: {str(e)}")
        return f"Error predicting price movement for {symbol}: {str(e)}"