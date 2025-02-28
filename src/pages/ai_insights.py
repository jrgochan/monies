import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import yfinance as yf

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
        
        if existing:
            # Update existing
            existing.result = result
            existing.model_used = model_used
            existing.timestamp = datetime.utcnow()
            existing.cached_until = datetime.utcnow() + timedelta(hours=expiry_hours)
        else:
            # Create new
            analysis = AiAnalysis(
                query=query,
                result=result,
                model_used=model_used,
                cached_until=datetime.utcnow() + timedelta(hours=expiry_hours)
            )
            db.add(analysis)
        
        db.commit()
        return True
    except Exception as e:
        db.rollback()
        st.error(f"Error caching analysis: {str(e)}")
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
            return analysis.result, analysis.model_used
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
                
                # Cache the result
                if result['success']:
                    cache_analysis(cache_key, result, "OpenAI")
        
        # Display results
        if isinstance(result, str):
            # If cached result was a string
            st.markdown(result)
        elif result['success']:
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
            
            # Display price chart
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            
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
            st.error(f"Error in analysis: {result['analysis']}")

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
                
                # Cache the result
                if result['success']:
                    cache_analysis(cache_key, result, "OpenAI")
        
        # Display results
        if isinstance(result, str):
            # If cached result was a string
            st.markdown(result)
        elif result['success']:
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
            
            # Display price chart
            ticker = f"{symbol}-USD"
            try:
                crypto = yf.Ticker(ticker)
                hist = crypto.history(period=f"{days}d")
                
                if hist.empty:
                    # Try alternative format
                    ticker = f"{symbol}USD=X"
                    crypto = yf.Ticker(ticker)
                    hist = crypto.history(period=f"{days}d")
                
                if not hist.empty:
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
            except Exception as e:
                st.error(f"Error displaying chart: {str(e)}")
            
            # Display AI analysis
            st.subheader("AI Analysis")
            st.markdown(result['analysis'])
        else:
            st.error(f"Error in analysis: {result['analysis']}")

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
                
                # Cache the result
                if result['success']:
                    cache_analysis(cache_key, result, "OpenAI")
        
        # Display results
        if isinstance(result, str):
            # If cached result was a string
            st.markdown(result)
        elif result['success']:
            # Display recommendations
            st.subheader("Recommended ETFs")
            
            # Create a table of ETFs
            if result['recommendations']:
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
            st.error(f"Error in analysis: {result['analysis']}")

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
                    
                    # Cache the result
                    cache_analysis(cache_key, result, model_name)
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