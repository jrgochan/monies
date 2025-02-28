import openai
import json
import requests
import yfinance as yf
import pandas as pd
import os
import logging
from datetime import datetime, timedelta
import time
from typing import Dict, List, Optional, Union
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY", "")

# Ollama settings
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama2")

def analyze_with_openai(prompt: str, model: str = "gpt-3.5-turbo") -> str:
    """
    Analyze data using OpenAI's API.
    """
    try:
        if not openai.api_key:
            return "Error: OpenAI API key is not configured"
        
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a financial analyst specializing in cryptocurrency and stock markets. Provide concise, data-driven insights."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.3
        )
        
        return response['choices'][0]['message']['content']
    
    except Exception as e:
        logger.error(f"Error with OpenAI analysis: {str(e)}")
        return f"Error: {str(e)}"

def analyze_with_ollama(prompt: str, model: str = None) -> str:
    """
    Analyze data using a local Ollama model.
    """
    if model is None:
        model = OLLAMA_MODEL
    
    try:
        # Prepare the API request payload
        payload = {
            "model": model,
            "prompt": f"System: You are a financial analyst specializing in cryptocurrency and stock markets. Provide concise, data-driven insights.\n\nUser: {prompt}",
            "stream": False
        }
        
        # Send request to Ollama API
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            return response.json()['response']
        else:
            return f"Error: Ollama returned status code {response.status_code}"
    
    except requests.RequestException as e:
        logger.error(f"Error with Ollama analysis: {str(e)}")
        if "connection" in str(e).lower():
            return "Error: Unable to connect to Ollama. Make sure the Ollama server is running."
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
        # Get stock data
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        
        if hist.empty:
            result['analysis'] = f"No data found for ticker {ticker}"
            return result
        
        # Basic statistics
        start_price = hist['Close'].iloc[0]
        end_price = hist['Close'].iloc[-1]
        percent_change = ((end_price - start_price) / start_price) * 100
        high = hist['High'].max()
        low = hist['Low'].min()
        
        # Calculate some indicators
        hist['MA20'] = hist['Close'].rolling(window=20).mean()
        hist['MA50'] = hist['Close'].rolling(window=50).mean()
        
        # Collect recent news
        news = []
        try:
            news_items = stock.news[:5]  # Get the last 5 news items
            for item in news_items:
                news.append({
                    'title': item['title'],
                    'publisher': item['publisher'],
                    'link': item['link'],
                    'date': datetime.fromtimestamp(item['providerPublishTime']).strftime('%Y-%m-%d')
                })
        except:
            # News might not be available
            pass
        
        # Create data object
        result['data'] = {
            'start_date': hist.index[0].strftime('%Y-%m-%d'),
            'end_date': hist.index[-1].strftime('%Y-%m-%d'),
            'start_price': round(start_price, 2),
            'end_price': round(end_price, 2),
            'percent_change': round(percent_change, 2),
            'high': round(high, 2),
            'low': round(low, 2),
            'volume_avg': int(hist['Volume'].mean()),
            'current_ma20': round(hist['MA20'].iloc[-1], 2) if not pd.isna(hist['MA20'].iloc[-1]) else None,
            'current_ma50': round(hist['MA50'].iloc[-1], 2) if not pd.isna(hist['MA50'].iloc[-1]) else None,
            'news': news
        }
        
        # Generate summary using AI
        try:
            # Get info
            info = stock.info
            company_name = info.get('shortName', ticker)
            sector = info.get('sector', 'Unknown')
            industry = info.get('industry', 'Unknown')
            
            # Prepare prompt for AI
            prompt = f"""
            Analyze the following stock data for {company_name} ({ticker}), a company in the {sector} sector, {industry} industry:
            
            Period: {period}
            Date range: {result['data']['start_date']} to {result['data']['end_date']}
            Starting price: ${result['data']['start_price']}
            Current price: ${result['data']['end_price']}
            Change: {result['data']['percent_change']}%
            Highest price in period: ${result['data']['high']}
            Lowest price in period: ${result['data']['low']}
            Average daily volume: {result['data']['volume_avg']}
            Current 20-day moving average: ${result['data']['current_ma20']}
            Current 50-day moving average: ${result['data']['current_ma50']}
            
            Recent news:
            {json.dumps(result['data']['news'], indent=2)}
            
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
        # For crypto, we can still use yfinance with -USD tickers
        ticker = f"{symbol}-USD"
        crypto = yf.Ticker(ticker)
        hist = crypto.history(period=f"{days}d")
        
        if hist.empty:
            # Try alternative format
            ticker = f"{symbol}USD=X"
            crypto = yf.Ticker(ticker)
            hist = crypto.history(period=f"{days}d")
            
            if hist.empty:
                result['analysis'] = f"No data found for cryptocurrency {symbol}"
                return result
        
        # Basic statistics
        start_price = hist['Close'].iloc[0]
        end_price = hist['Close'].iloc[-1]
        percent_change = ((end_price - start_price) / start_price) * 100
        high = hist['High'].max()
        low = hist['Low'].min()
        
        # Calculate some indicators
        hist['MA20'] = hist['Close'].rolling(window=20).mean()
        hist['MA50'] = hist['Close'].rolling(window=50).mean()
        
        # Create data object
        result['data'] = {
            'start_date': hist.index[0].strftime('%Y-%m-%d'),
            'end_date': hist.index[-1].strftime('%Y-%m-%d'),
            'start_price': round(start_price, 2),
            'end_price': round(end_price, 2),
            'percent_change': round(percent_change, 2),
            'high': round(high, 2),
            'low': round(low, 2),
            'volume_avg': int(hist['Volume'].mean()),
            'current_ma20': round(hist['MA20'].iloc[-1], 2) if not pd.isna(hist['MA20'].iloc[-1]) else None,
            'current_ma50': round(hist['MA50'].iloc[-1], 2) if not pd.isna(hist['MA50'].iloc[-1]) else None
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
            
            # Try OpenAI first, fall back to Ollama
            try:
                result['analysis'] = analyze_with_openai(prompt)
            except:
                result['analysis'] = analyze_with_ollama(prompt)
            
            result['success'] = True
        
        except Exception as e:
            logger.error(f"Error generating analysis for {symbol}: {str(e)}")
            result['analysis'] = f"Error generating analysis: {str(e)}"
    
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
        else:
            # Default to moderate
            selected_etfs.extend(etf_by_risk["moderate"])
        
        # Add sector ETFs if specified
        if sectors:
            for sector in sectors:
                if sector.lower() in etf_by_sector:
                    selected_etfs.extend(etf_by_sector[sector.lower()])
        
        # Remove duplicates and limit to 10
        selected_etfs = list(set(selected_etfs))[:10]
        
        # Get data for selected ETFs
        for ticker in selected_etfs:
            try:
                etf = yf.Ticker(ticker)
                info = etf.info
                hist = etf.history(period="1y")
                
                # Calculate performance
                start_price = hist['Close'].iloc[0]
                end_price = hist['Close'].iloc[-1]
                percent_change = ((end_price - start_price) / start_price) * 100
                
                # Get basic info
                etf_data = {
                    'ticker': ticker,
                    'name': info.get('shortName', ticker),
                    'expense_ratio': info.get('expenseRatio', 0) * 100 if 'expenseRatio' in info else None,
                    'category': info.get('category', ''),
                    'description': info.get('longBusinessSummary', '')[:200] + '...',
                    'current_price': round(end_price, 2),
                    'yearly_change': round(percent_change, 2)
                }
                
                result['recommendations'].append(etf_data)
            
            except Exception as e:
                logger.error(f"Error getting data for ETF {ticker}: {str(e)}")
                # Skip this ETF
        
        # Generate analysis using AI
        try:
            # Prepare prompt for AI
            etf_list = [f"{etf['ticker']} ({etf['name']}): {etf['yearly_change']}% 1-year return" 
                      for etf in result['recommendations']]
            
            prompt = f"""
            Based on a {risk_profile} risk profile{' and interest in the following sectors: ' + ', '.join(sectors) if sectors else ''}, 
            I've identified the following ETFs:
            
            {chr(10).join(etf_list)}
            
            Provide a brief analysis of these recommendations. Explain why they match the risk profile, what their performance has been,
            and how they might fit into a diversified portfolio. Include any potential drawbacks or considerations.
            Finally, suggest an allocation strategy across these ETFs for a {risk_profile} investor.
            Keep your response under 250 words.
            """
            
            # Try OpenAI first, fall back to Ollama
            try:
                result['analysis'] = analyze_with_openai(prompt)
            except:
                result['analysis'] = analyze_with_ollama(prompt)
            
            result['success'] = True
        
        except Exception as e:
            logger.error(f"Error generating ETF recommendations analysis: {str(e)}")
            result['analysis'] = f"Error generating analysis: {str(e)}"
    
    except Exception as e:
        logger.error(f"Error getting ETF recommendations: {str(e)}")
        result['analysis'] = f"Error: {str(e)}"
    
    return result

def generate_social_post(post_type: str, data: Dict) -> str:
    """
    Generate a social media post based on market data.
    """
    try:
        if post_type == "portfolio_update":
            # Generate portfolio update post
            portfolio_value = data.get('portfolio_value', 0)
            daily_change = data.get('daily_change', 0)
            top_assets = data.get('top_assets', [])
            
            prompt = f"""
            Generate a short social media post (maximum 280 characters) about my investment portfolio:
            
            Portfolio value: ${portfolio_value}
            Daily change: {daily_change}%
            Top performing assets: {', '.join(top_assets)}
            
            The post should be engaging, professional, and appropriate for platforms like Twitter or LinkedIn.
            Include relevant hashtags. Do not use emojis excessively.
            """
        
        elif post_type == "market_update":
            # Generate market update post
            market_data = data.get('market_data', {})
            sentiment = data.get('sentiment', 'neutral')
            
            prompt = f"""
            Generate a short social media post (maximum 280 characters) about today's market conditions:
            
            Market data: {json.dumps(market_data)}
            Overall sentiment: {sentiment}
            
            The post should be engaging, professional, and appropriate for platforms like Twitter or LinkedIn.
            Include relevant hashtags. Do not use emojis excessively.
            """
        
        elif post_type == "investment_tip":
            # Generate investment tip post
            tip_topic = data.get('topic', 'diversification')
            
            prompt = f"""
            Generate a short social media post (maximum 280 characters) with an investment tip about {tip_topic}.
            
            The post should be educational, engaging, and appropriate for platforms like Twitter or LinkedIn.
            Include relevant hashtags. Do not use emojis excessively.
            """
        
        else:
            return "Error: Unknown post type"
        
        # Try OpenAI first, fall back to Ollama
        try:
            post_content = analyze_with_openai(prompt, model="gpt-3.5-turbo")
        except:
            post_content = analyze_with_ollama(prompt)
        
        return post_content
    
    except Exception as e:
        logger.error(f"Error generating social post: {str(e)}")
        return f"Error: {str(e)}"