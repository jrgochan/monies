import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import yfinance as yf
import numpy as np
import json

from src.utils.auth import require_login
from src.models.database import SessionLocal, AiAnalysis
from src.api.ai_analysis import (
    analyze_stock_trend,
    analyze_crypto_trend,
    get_etf_recommendations,
    analyze_with_openai,
    analyze_with_ollama
)

def cache_analysis(query, result, model_used, expiry_hours=24):
    """Cache analysis results in the database"""
    db = SessionLocal()
    try:
        # Check if analysis already exists
        existing = db.query(AiAnalysis).filter(AiAnalysis.query == query).first()
        
        # Convert result to JSON string if it's a dictionary
        result_text = json.dumps(result) if isinstance(result, dict) else result
        
        if existing:
            # Update existing
            existing.result = result_text
            existing.model_used = model_used
            existing.timestamp = datetime.utcnow()
            existing.cached_until = datetime.utcnow() + timedelta(hours=expiry_hours)
        else:
            # Create new
            analysis = AiAnalysis(
                query=query,
                result=result_text,
                model_used=model_used,
                cached_until=datetime.utcnow() + timedelta(hours=expiry_hours)
            )
            db.add(analysis)
        
        db.commit()
        return True
    except Exception as e:
        db.rollback()
        st.error(f"Error caching analysis: {str(e)}")
        # Don't let caching errors block the analysis functionality
        return False
    finally:
        db.close()

def get_cached_analysis(query):
    """Get cached analysis if available and not expired"""
    db = SessionLocal()
    try:
        analysis = db.query(AiAnalysis).filter(
            AiAnalysis.query == query,
            AiAnalysis.cached_until > datetime.utcnow()
        ).first()
        
        if analysis:
            # Try to parse as JSON, fall back to raw text if not valid JSON
            try:
                result = json.loads(analysis.result)
            except (json.JSONDecodeError, TypeError):
                result = analysis.result
            return result, analysis.model_used
        
        return None, None
    except Exception as e:
        st.warning(f"Cache retrieval error: {str(e)}")
        return None, None
    finally:
        db.close()

def show_stock_analysis():
    """Show stock analysis section"""
    st.subheader("Stock Trend Analysis")
    
    # Input for stock ticker
    ticker = st.text_input("Enter Stock Ticker Symbol", placeholder="AAPL, MSFT, TSLA, etc.")
    
    # Period selection
    period = st.selectbox(
        "Analysis Period",
        options=["1mo", "3mo", "6mo", "1y", "2y", "5y"],
        index=2,
        format_func=lambda x: {
            "1mo": "1 Month",
            "3mo": "3 Months",
            "6mo": "6 Months",
            "1y": "1 Year",
            "2y": "2 Years",
            "5y": "5 Years"
        }.get(x, x)
    )
    
    if st.button("Analyze Stock", key="analyze_stock"):
        if not ticker:
            st.error("Please enter a ticker symbol")
            return
        
        # Check cache first
        cache_key = f"stock_{ticker}_{period}"
        cached_result, model = get_cached_analysis(cache_key)
        
        if cached_result:
            st.success(f"Using cached analysis (via {model})")
            result = cached_result
        else:
            # Run new analysis
            with st.spinner("Analyzing stock data..."):
                result = analyze_stock_trend(ticker, period)
                
                # Cache the result (ignore errors)
                try:
                    if result.get('success', False):
                        cache_analysis(cache_key, result, "OpenAI")
                except Exception as e:
                    st.warning(f"Could not cache result: {str(e)}")
        
        # Display results
        if isinstance(result, str):
            # If cached result was a string
            st.markdown(result)
        elif result.get('success', False):
            # Display stock data
            data = result['data']
            
            # Create metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    label=f"Price ({data['end_date']})",
                    value=f"${data['end_price']}",
                    delta=f"{data['percent_change']}%"
                )
            with col2:
                st.metric(
                    label="Period High",
                    value=f"${data['high']}"
                )
            with col3:
                st.metric(
                    label="Period Low",
                    value=f"${data['low']}"
                )
            
            # Try to display price chart with improved error handling
            try:
                # Create a chart using Yahoo Finance data
                st.write("Loading historical data...")
                
                try:
                    # First attempt with standard ticker format
                    stock = yf.Ticker(ticker)
                    hist = stock.history(period=period)
                    
                    # Check if we got valid data
                    if hist.empty or 'Close' not in hist.columns:
                        raise ValueError("No valid price data found")
                        
                except Exception as e:
                    # Try alternative ticker formats
                    alternatives = [f"^{ticker}", f"{ticker}-USD", f"{ticker}.N"]
                    success = False
                    hist = None
                    
                    for alt_ticker in alternatives:
                        try:
                            st.info(f"Trying alternative ticker format: {alt_ticker}")
                            stock = yf.Ticker(alt_ticker)
                            hist = stock.history(period=period)
                            if not hist.empty and 'Close' in hist.columns:
                                success = True
                                break
                        except:
                            continue
                    
                    if not success:
                        # Create synthetic data for demonstration
                        st.warning(f"Could not retrieve data for {ticker}. Using simulated data for visualization.")
                        
                        # Try to extract starting price
                        start_price = 100.0  # Default
                        
                        # Generate time series
                        dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
                        
                        # Create a random walk price series
                        volatility = 0.01  # 1% daily volatility
                        returns = np.random.normal(0, volatility, len(dates))
                        cumulative_returns = np.exp(np.cumsum(returns)) - 1
                        prices = start_price * (1 + cumulative_returns)
                        
                        hist = pd.DataFrame({'Close': prices}, index=dates)
                
                # Create and display chart
                if hist is not None and not hist.empty and 'Close' in hist.columns:
                    fig = px.line(
                        hist, 
                        y='Close',
                        title=f"{ticker} Price History ({period})"
                    )
                    
                    fig.update_layout(
                        xaxis_title="Date",
                        yaxis_title="Price (USD)",
                        hovermode="x unified"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("Failed to generate price chart")
            
            except Exception as chart_error:
                st.error(f"Could not display price chart: {str(chart_error)}")
            
            # Display AI analysis
            st.subheader("AI Analysis")
            st.markdown(result['analysis'])
            
            # Display recent news if available
            if data.get('news'):
                st.subheader("Recent News")
                for news_item in data['news']:
                    st.markdown(f"### [{news_item['title']}]({news_item['link']})")
                    st.caption(f"{news_item['publisher']} • {news_item['date']}")
        else:
            st.error(f"Error in analysis: {result.get('analysis', 'Unknown error')}")

def show_crypto_analysis():
    """Show cryptocurrency analysis section"""
    st.subheader("Cryptocurrency Trend Analysis")
    
    # Input for crypto symbol
    symbol = st.text_input("Enter Cryptocurrency Symbol", placeholder="BTC, ETH, SOL, etc.")
    
    # Period selection
    days = st.slider("Analysis Period (Days)", min_value=7, max_value=365, value=180, step=7)
    
    if st.button("Analyze Cryptocurrency", key="analyze_crypto"):
        if not symbol:
            st.error("Please enter a cryptocurrency symbol")
            return
        
        # Check cache first
        cache_key = f"crypto_{symbol}_{days}"
        cached_result, model = get_cached_analysis(cache_key)
        
        if cached_result:
            st.success(f"Using cached analysis (via {model})")
            result = cached_result
        else:
            # Run new analysis
            with st.spinner("Analyzing cryptocurrency data..."):
                result = analyze_crypto_trend(symbol, days)
                
                # Cache the result (ignore errors)
                try:
                    if result.get('success', False):
                        cache_analysis(cache_key, result, "OpenAI")
                except Exception as e:
                    st.warning(f"Could not cache result: {str(e)}")
        
        # Display results
        if isinstance(result, str):
            # If cached result was a string
            st.markdown(result)
        elif result.get('success', False):
            # Display crypto data
            data = result['data']
            
            # Create metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    label=f"Price ({data['end_date']})",
                    value=f"${data['end_price']}",
                    delta=f"{data['percent_change']}%"
                )
            with col2:
                st.metric(
                    label="Period High",
                    value=f"${data['high']}"
                )
            with col3:
                st.metric(
                    label="Period Low",
                    value=f"${data['low']}"
                )
            
            # Display price chart safely
            try:
                # Try to get price history with different formats
                st.write("Loading historical data...")
                hist = None
                
                try:
                    # First attempt - standard crypto format
                    # Use Alpha Vantage API as primary source for crypto data (free tier)
                    # This is more reliable than Yahoo Finance for crypto
                    alpha_vantage_key = os.getenv("ALPHA_VANTAGE_KEY", "demo")
                    crypto_url = f"https://www.alphavantage.co/query?function=DIGITAL_CURRENCY_DAILY&symbol={symbol}&market=USD&apikey={alpha_vantage_key}"
                    
                    response = requests.get(crypto_url)
                    if response.status_code == 200 and "Time Series (Digital Currency Daily)" in response.json():
                        # Process Alpha Vantage data
                        data = response.json()["Time Series (Digital Currency Daily)"]
                        
                        # Convert to DataFrame
                        av_data = []
                        for date, values in data.items():
                            av_data.append({
                                'Date': date,
                                'Open': float(values['1a. open (USD)']),
                                'High': float(values['2a. high (USD)']),
                                'Low': float(values['3a. low (USD)']),
                                'Close': float(values['4a. close (USD)']),
                                'Volume': float(values['5. volume'])
                            })
                        
                        # Create DataFrame and sort by date
                        hist = pd.DataFrame(av_data)
                        hist['Date'] = pd.to_datetime(hist['Date'])
                        hist = hist.sort_values('Date')
                        hist = hist.set_index('Date')
                        
                        # Limit to requested days
                        hist = hist.tail(days)
                    else:
                        # Fallback to Yahoo Finance if Alpha Vantage fails
                        ticker = f"{symbol}-USD"
                        crypto = yf.Ticker(ticker)
                        hist = crypto.history(period=f"{days}d")
                        
                        # Verify data
                        if hist.empty or 'Close' not in hist.columns:
                            raise ValueError("No data found")
                except Exception as e:
                    # Try alternative formats
                    alternatives = [f"{symbol}USD=X", f"{symbol}-USD.CC", f"{symbol}USDT=X"]
                    success = False
                    
                    for alt_ticker in alternatives:
                        try:
                            st.info(f"Trying alternative ticker format: {alt_ticker}")
                            crypto = yf.Ticker(alt_ticker)
                            hist = crypto.history(period=f"{days}d")
                            if not hist.empty and 'Close' in hist.columns:
                                success = True
                                break
                        except:
                            continue
                    
                    if not success:
                        # Create synthetic data for demonstration
                        st.warning(f"Could not retrieve data for {symbol}. Using simulated data for visualization.")
                        
                        # Use a default price 
                        start_price = 1000.0 if symbol.upper() in ['BTC', 'ETH'] else 10.0
                        
                        # Generate time series
                        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
                        
                        # Create a random walk price series
                        volatility = 0.02  # 2% daily volatility for crypto
                        returns = np.random.normal(0, volatility, len(dates))
                        cumulative_returns = np.exp(np.cumsum(returns)) - 1
                        prices = start_price * (1 + cumulative_returns)
                        
                        hist = pd.DataFrame({'Close': prices}, index=dates)
                
                # Create and display chart if we have data
                if hist is not None and not hist.empty and 'Close' in hist.columns:
                    fig = px.line(
                        hist, 
                        y='Close',
                        title=f"{symbol} Price History ({days} days)"
                    )
                    
                    fig.update_layout(
                        xaxis_title="Date",
                        yaxis_title="Price (USD)",
                        hovermode="x unified"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("Failed to generate price chart")
            except Exception as e:
                st.error(f"Error displaying chart: {str(e)}")
            
            # Display AI analysis
            st.subheader("AI Analysis")
            st.markdown(result['analysis'])
        else:
            st.error(f"Error in analysis: {result.get('analysis', 'Unknown error')}")

def show_etf_recommendations():
    """Show ETF recommendations section"""
    st.subheader("ETF Investment Recommendations")
    
    # Risk profile selection
    risk_profile = st.selectbox(
        "Risk Profile",
        options=["conservative", "moderate", "aggressive"],
        format_func=lambda x: x.capitalize()
    )
    
    # Sector selection
    sectors = st.multiselect(
        "Sectors of Interest",
        options=["technology", "healthcare", "finance", "energy", "consumer", "real estate"],
        format_func=lambda x: x.capitalize()
    )
    
    if st.button("Get Recommendations", key="get_etf_recs"):
        # Check cache first
        cache_key = f"etf_{risk_profile}_{'-'.join(sorted(sectors))}"
        cached_result, model = get_cached_analysis(cache_key)
        
        if cached_result:
            st.success(f"Using cached recommendations (via {model})")
            result = cached_result
        else:
            # Run new analysis
            with st.spinner("Generating ETF recommendations..."):
                result = get_etf_recommendations(risk_profile, sectors)
                
                # Cache the result (ignore errors)
                try:
                    if result.get('success', False):
                        cache_analysis(cache_key, result, "OpenAI")
                except Exception as e:
                    st.warning(f"Could not cache result: {str(e)}")
        
        # Display results
        if isinstance(result, str):
            # If cached result was a string
            st.markdown(result)
        elif result.get('success', False):
            # Display recommendations
            st.subheader("Recommended ETFs")
            
            # Create a table of ETFs
            if result.get('recommendations', []):
                etf_data = []
                for etf in result['recommendations']:
                    etf_data.append({
                        "Ticker": etf['ticker'],
                        "Name": etf['name'],
                        "1-Year Return": f"{etf['yearly_change']}%",
                        "Expense Ratio": f"{etf['expense_ratio']:.2f}%" if etf['expense_ratio'] else "N/A",
                        "Category": etf['category']
                    })
                
                etf_df = pd.DataFrame(etf_data)
                st.dataframe(etf_df, hide_index=True, use_container_width=True)
                
                # Display AI analysis
                st.subheader("Investment Strategy")
                st.markdown(result['analysis'])
                
                # Show example allocation chart
                st.subheader("Example Allocation")
                
                # Create a simple allocation based on ETF count
                etf_count = len(result['recommendations'])
                if risk_profile == "conservative":
                    # More weight to first few ETFs (likely bonds/stable)
                    weights = [0.3, 0.25, 0.15, 0.1, 0.05, 0.05, 0.025, 0.025, 0.025, 0.025][:etf_count]
                    # Normalize weights
                    weights = [w/sum(weights) for w in weights]
                elif risk_profile == "aggressive":
                    # More evenly distributed
                    equal_weight = 1.0 / etf_count
                    weights = [equal_weight] * etf_count
                else:  # moderate
                    # Somewhat balanced
                    weights = [0.25, 0.2, 0.15, 0.1, 0.1, 0.05, 0.05, 0.05, 0.025, 0.025][:etf_count]
                    # Normalize weights
                    weights = [w/sum(weights) for w in weights]
                
                # Create pie chart
                fig = px.pie(
                    values=weights,
                    names=[etf['ticker'] for etf in result['recommendations']],
                    title='Suggested Allocation'
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No ETF recommendations available")
        else:
            st.error(f"Error in analysis: {result.get('analysis', 'Unknown error')}")

def show_custom_analysis():
    """Show custom AI analysis section"""
    st.subheader("Custom Market Analysis")
    
    # Text area for custom query
    query = st.text_area(
        "Enter your financial analysis question",
        placeholder="Example: What's the relationship between Bitcoin price and tech stocks? Or: Analyze the impact of recent inflation data on growth stocks."
    )
    
    # Model selection
    model = st.radio(
        "Analysis Model",
        options=["OpenAI (GPT)", "Ollama (Local)"],
        horizontal=True
    )
    
    if st.button("Analyze", key="custom_analysis"):
        if not query:
            st.error("Please enter a question to analyze")
            return
        
        # Check cache first
        cache_key = f"custom_{hash(query)}"
        cached_result, cached_model = get_cached_analysis(cache_key)
        
        if cached_result:
            st.success(f"Using cached analysis (via {cached_model})")
            result = cached_result
        else:
            # Run new analysis
            with st.spinner("Analyzing..."):
                try:
                    if model == "OpenAI (GPT)":
                        result = analyze_with_openai(query)
                        model_name = "OpenAI"
                    else:
                        result = analyze_with_ollama(query)
                        model_name = "Ollama"
                    
                    # Cache the result (ignore errors)
                    try:
                        cache_analysis(cache_key, result, model_name)
                    except Exception as e:
                        st.warning(f"Could not cache result: {str(e)}")
                except Exception as e:
                    result = f"Error in analysis: {str(e)}"
        
        # Display result
        st.subheader("Analysis Result")
        st.markdown(result)

def show_ai_insights():
    """Display the AI Insights page"""
    # Require login
    user = require_login()
    
    # Add page title
    st.title("AI-Powered Market Insights")
    
    # Create tabs for different analysis types
    tab1, tab2, tab3, tab4 = st.tabs([
        "Stock Analysis", 
        "Crypto Analysis", 
        "ETF Recommendations",
        "Custom Analysis"
    ])
    
    with tab1:
        show_stock_analysis()
    
    with tab2:
        show_crypto_analysis()
    
    with tab3:
        show_etf_recommendations()
    
    with tab4:
        show_custom_analysis()