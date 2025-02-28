"""Integration tests for the new AI analysis module functionality."""
import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import os
import json
from datetime import datetime, timedelta

from src.api.ai_analysis import (
    analyze_with_openai,
    analyze_with_ollama,
    generate_generic_analysis,
    analyze_stock_trend,
    analyze_crypto_trend,
    get_etf_recommendations
)


@pytest.fixture
def sample_stock_data():
    """Create sample historical stock data."""
    dates = pd.date_range(start=datetime.now()-timedelta(days=180), end=datetime.now(), freq='D')
    
    # Create a dataframe with price data
    df = pd.DataFrame({
        'Open': [100 + i * 0.1 for i in range(len(dates))],
        'High': [102 + i * 0.1 for i in range(len(dates))],
        'Low': [98 + i * 0.1 for i in range(len(dates))],
        'Close': [101 + i * 0.1 for i in range(len(dates))],
        'Volume': [1000000 for _ in range(len(dates))]
    }, index=dates)
    
    return df


@pytest.fixture
def sample_crypto_data():
    """Create sample historical cryptocurrency data."""
    dates = pd.date_range(start=datetime.now()-timedelta(days=180), end=datetime.now(), freq='D')
    
    # Create a dataframe with price data
    df = pd.DataFrame({
        'Open': [30000 + i * 10 for i in range(len(dates))],
        'High': [31000 + i * 10 for i in range(len(dates))],
        'Low': [29000 + i * 10 for i in range(len(dates))],
        'Close': [30500 + i * 10 for i in range(len(dates))],
        'Volume': [5000000 for _ in range(len(dates))]
    }, index=dates)
    
    return df


@patch("src.api.ai_analysis.OpenAI")
def test_analyze_with_openai(mock_openai_class):
    """Test analyzing with OpenAI API."""
    # Setup the mock
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    mock_client.chat.completions.create.return_value.choices = [
        MagicMock(message=MagicMock(content="This is a test analysis"))
    ]
    
    # Call the function
    result = analyze_with_openai("Test prompt", model="gpt-3.5-turbo")
    
    # Verify
    assert result == "This is a test analysis"
    mock_openai_class.assert_called_once()
    mock_client.chat.completions.create.assert_called_once()


@patch("src.api.ai_analysis.requests.post")
def test_analyze_with_ollama(mock_post):
    """Test analyzing with Ollama API."""
    # Setup the mock
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"response": "This is a test analysis from Ollama"}
    mock_post.return_value = mock_response
    
    # Call the function
    result = analyze_with_ollama("Test prompt", model="llama2")
    
    # Verify
    assert result == "This is a test analysis from Ollama"
    mock_post.assert_called_once()


def test_generate_generic_analysis():
    """Test generating generic analysis without AI."""
    # Test with positive percent change
    result = generate_generic_analysis("Change: 20.5%")
    assert "bullish momentum" in result.lower()
    
    # Test with negative percent change
    result = generate_generic_analysis("Change: -16.3%")
    assert "bearish trend" in result.lower()
    
    # Test with small positive change
    result = generate_generic_analysis("Change: 2.1%")
    assert "modest positive performance" in result.lower()
    
    # Test with small negative change
    result = generate_generic_analysis("Change: -1.5%")
    assert "slight decline" in result.lower()
    
    # Test with no percent change
    result = generate_generic_analysis("No percentage info")
    assert "typical market fluctuations" in result.lower()


@patch("src.api.ai_analysis.yf.Ticker")
@patch("src.api.ai_analysis.analyze_with_openai")
def test_analyze_stock_trend_success(mock_analyze, mock_ticker, sample_stock_data):
    """Test analyzing stock trend with successful API connections."""
    # Setup mocks
    mock_ticker_instance = MagicMock()
    mock_ticker_instance.history.return_value = sample_stock_data
    mock_ticker_instance.info = {
        "shortName": "Test Company",
        "sector": "Technology",
        "industry": "Software",
        "marketCap": 1000000000,
        "trailingPE": 20.5
    }
    mock_ticker_instance.news = [
        {"title": "Test News", "link": "https://example.com", "publisher": "Test Publisher", "providerPublishTime": int(datetime.now().timestamp())}
    ]
    mock_ticker.return_value = mock_ticker_instance
    
    # Mock the AI analysis
    mock_analyze.return_value = "This is an AI analysis of the stock trend."
    
    # Call the function
    result = analyze_stock_trend("AAPL")
    
    # Verify
    assert result["ticker"] == "AAPL"
    assert result["success"] is True
    assert "data" in result
    assert "analysis" in result
    assert result["analysis"] == "This is an AI analysis of the stock trend."
    mock_ticker.assert_called_once_with("AAPL")
    mock_analyze.assert_called_once()


@patch("src.api.ai_analysis.yf.Ticker")
def test_analyze_stock_trend_api_error(mock_ticker):
    """Test analyzing stock trend with API connection error."""
    # Setup mock to raise an error
    mock_ticker.side_effect = Exception("API connection error")
    
    # Call the function
    result = analyze_stock_trend("AAPL")
    
    # Verify
    assert result["ticker"] == "AAPL"
    assert result["success"] is False
    assert "Error" in result["analysis"]
    mock_ticker.assert_called_once_with("AAPL")


@patch("src.api.ai_analysis.yf.Ticker")
@patch("src.api.ai_analysis.analyze_with_openai")
def test_analyze_crypto_trend_historical_data_error(mock_analyze, mock_ticker):
    """Test analyzing crypto trend when historical data can't be retrieved."""
    # Setup mocks
    mock_ticker_instance = MagicMock()
    mock_ticker_instance.history.return_value = pd.DataFrame()  # Empty dataframe
    mock_ticker.return_value = mock_ticker_instance
    
    # Mock requests for fallback data retrieval
    with patch("src.api.ai_analysis.requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"btc": {"usd": 40000}}
        mock_get.return_value = mock_response
        
        # Call the function
        result = analyze_crypto_trend("BTC")
        
        # Verify
        assert result["symbol"] == "BTC"
        assert result["success"] is False
        assert "Cannot retrieve price data" in result["analysis"]
        mock_ticker.assert_called()
        mock_analyze.assert_not_called()


@patch("src.api.ai_analysis.yf.Ticker")
@patch("src.api.ai_analysis.requests.get")
@patch("src.api.ai_analysis.analyze_with_openai")
def test_analyze_crypto_trend_with_coingecko_fallback(mock_analyze, mock_get, mock_ticker, sample_crypto_data):
    """Test analyzing crypto trend with Yahoo Finance failure but CoinGecko success."""
    # Setup YF mock to fail
    mock_ticker_instance = MagicMock()
    mock_ticker_instance.history.return_value = pd.DataFrame()  # Empty dataframe
    mock_ticker.return_value = mock_ticker_instance
    
    # Setup CoinGecko mock to succeed
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"btc": {"usd": 40000}}
    mock_get.return_value = mock_response
    
    # Since we can't get historical data, we'll still get an error
    result = analyze_crypto_trend("BTC")
    
    # Verify
    assert result["symbol"] == "BTC"
    assert result["success"] is False
    assert "Cannot retrieve price data" in result["analysis"]
    mock_ticker.assert_called()
    mock_get.assert_called()


@patch("src.api.ai_analysis.yf.Ticker")
@patch("src.api.ai_analysis.generate_generic_analysis")
def test_analyze_crypto_trend_openai_error(mock_generate_generic, mock_ticker, sample_crypto_data):
    """Test analyzing crypto trend with successful history but OpenAI error."""
    # Setup YF mock to succeed
    mock_ticker_instance = MagicMock()
    mock_ticker_instance.history.return_value = sample_crypto_data
    mock_ticker.return_value = mock_ticker_instance
    
    # Mock OpenAI to fail but generic analysis to work
    with patch("src.api.ai_analysis.analyze_with_openai") as mock_analyze:
        mock_analyze.side_effect = Exception("OpenAI API error")
        mock_generate_generic.return_value = "This is a generic analysis."
        
        # Call the function - this would still return an early error because of our fixed no-simulated-data change
        result = analyze_crypto_trend("BTC")
        
        # Verify
        assert result["symbol"] == "BTC"
        # Since we've updated the function to use the generic analysis instead of failing
        # with "Cannot retrieve price data", we should check for the generic analysis
        assert result["analysis"] == "This is a generic analysis."


@patch("src.api.ai_analysis.yf.Ticker")
def test_get_etf_recommendations(mock_ticker):
    """Test getting ETF recommendations."""
    # Setup mock
    mock_ticker_instance = MagicMock()
    mock_ticker_instance.info = {
        "shortName": "Test ETF",
        "category": "Technology",
        "expenseRatio": 0.05,
        "regularMarketPrice": 150.0
    }
    mock_ticker_instance.history.return_value = pd.DataFrame({
        'Close': [100.0, 150.0]  # 50% return
    }, index=[datetime.now() - timedelta(days=365), datetime.now()])
    mock_ticker.return_value = mock_ticker_instance
    
    # Mock the AI analysis
    with patch("src.api.ai_analysis.analyze_with_openai") as mock_analyze:
        mock_analyze.return_value = "This is an ETF recommendation analysis."
        
        # Call the function
        result = get_etf_recommendations("moderate", ["technology"])
        
        # Verify
        assert result["risk_profile"] == "moderate"
        assert result["sectors"] == ["technology"]
        assert len(result["recommendations"]) > 0
        assert result["analysis"] == "This is an ETF recommendation analysis."
        assert result["success"] is True
        assert mock_ticker.call_count > 0
        mock_analyze.assert_called_once()