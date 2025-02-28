"""Unit tests for AI analysis functions."""
import pytest
from unittest.mock import patch, MagicMock
import openai
import pandas as pd

from src.api.ai_analysis import (
    analyze_with_openai,
    analyze_with_ollama,
    analyze_stock_trend,
    analyze_market_trends
)


@pytest.fixture
def mock_openai_response():
    """Mock OpenAI API response."""
    mock_resp = MagicMock()
    # Mock for the new OpenAI client
    mock_resp.choices = [MagicMock()]
    mock_resp.choices[0].message.content = "This is a mock analysis from OpenAI"
    return mock_resp


@pytest.fixture
def mock_ollama_response():
    """Mock Ollama API response."""
    return {
        "model": "llama2",
        "response": "This is a mock analysis from Ollama",
        "done": True
    }


def test_analyze_with_openai_new_client(mock_openai_response):
    """Test analyzing data with OpenAI's new client."""
    with patch('src.api.ai_analysis.client') as mock_client:
        with patch('src.api.ai_analysis.openai.api_key', 'mock-api-key'):
            # Configure the mock
            mock_client.chat.completions.create.return_value = mock_openai_response
            
            # Call the function
            result = analyze_with_openai("Test prompt", model="gpt-3.5-turbo")
            
            # Verify the result
            assert result == "This is a mock analysis from OpenAI"
            
            # Verify the mock was called with correct parameters
            mock_client.chat.completions.create.assert_called_once()
            call_kwargs = mock_client.chat.completions.create.call_args[1]
            assert call_kwargs["model"] == "gpt-3.5-turbo"
            assert len(call_kwargs["messages"]) >= 2  # System message + user message
            assert call_kwargs["messages"][1]["content"] == "Test prompt"


@pytest.mark.parametrize("api_key_set", [True, False])
def test_analyze_with_openai_api_key_validation(api_key_set):
    """Test API key validation in analyze_with_openai."""
    with patch('src.api.ai_analysis.api_key', 'mock-api-key' if api_key_set else ''):
        with patch('src.api.ai_analysis.client') as mock_client:
            if api_key_set:
                # Set up the mock response for successful API call
                mock_resp = MagicMock()
                mock_resp.choices = [MagicMock()]
                mock_resp.choices[0].message.content = "This is a mock analysis"
                mock_client.chat.completions.create.return_value = mock_resp
                
                result = analyze_with_openai("Test prompt")
                assert "This is a mock analysis" in result
            else:
                # Make sure we're also patching the direct OpenAI call fallback
                with patch('src.api.ai_analysis.OpenAI') as mock_openai_class:
                    # Need to mock generate_generic_analysis since it will be called as a fallback
                    with patch('src.api.ai_analysis.generate_generic_analysis') as mock_generic:
                        mock_generic.return_value = "Generic analysis fallback"
                        
                        result = analyze_with_openai("Test prompt")
                        # For empty API key, it should use the fallback generic analysis
                        assert result == "Error: OpenAI API key is not configured"
                        mock_client.chat.completions.create.assert_not_called()


def test_analyze_with_ollama(mock_ollama_response):
    """Test analyzing data with Ollama."""
    with patch('requests.post') as mock_post:
        # Configure the mock response
        mock_resp = MagicMock()
        mock_resp.json.return_value = mock_ollama_response
        mock_resp.status_code = 200
        mock_post.return_value = mock_resp
        
        # Call the function
        result = analyze_with_ollama("Test prompt")
        
        # Verify the result
        assert result == "This is a mock analysis from Ollama"
        
        # Verify the mock was called with correct parameters
        mock_post.assert_called_once()
        url = mock_post.call_args[0][0]
        assert "llama2" in url or "generate" in url


def test_analyze_stock_trend():
    """Test stock trend analysis."""
    with patch('src.api.ai_analysis.yf.Ticker') as mock_ticker:
        # Configure the mock
        mock_ticker_instance = MagicMock()
        mock_ticker.return_value = mock_ticker_instance
        
        # Create test data for history
        dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
        hist_data = pd.DataFrame({
            'Open': [100 + i*0.1 for i in range(100)],
            'High': [101 + i*0.1 for i in range(100)],
            'Low': [99 + i*0.1 for i in range(100)],
            'Close': [100.5 + i*0.1 for i in range(100)],
            'Volume': [1000000 for _ in range(100)]
        }, index=dates)
        
        mock_ticker_instance.history.return_value = hist_data
        
        # Mock news data
        mock_ticker_instance.news = [
            {
                'title': 'Test News Article',
                'publisher': 'Test Publisher',
                'link': 'https://example.com',
                'providerPublishTime': 1612137600  # 2021-02-01
            }
        ]
        
        # Call the function
        result = analyze_stock_trend("AAPL", period="3mo")
        
        # Verify the result structure
        assert 'ticker' in result
        assert 'data' in result
        assert 'percent_change' in result['data']
        assert result['ticker'] == "AAPL"


def test_analyze_market_trends_with_openai():
    """Test market trends analysis using OpenAI."""
    with patch('src.api.ai_analysis.analyze_with_openai') as mock_analyze:
        # Configure the mock
        mock_analyze.return_value = "BTC shows a bullish trend with increasing volume."
        
        # Call the function
        result = analyze_market_trends('BTC', timeframe='daily')
        
        # Verify the result
        assert "BTC" in result
        assert "bullish" in result
        
        # Verify the mock was called
        mock_analyze.assert_called_once()
        # Verify the prompt contains relevant keywords
        prompt = mock_analyze.call_args[0][0]
        assert "BTC" in prompt
        assert "daily" in prompt