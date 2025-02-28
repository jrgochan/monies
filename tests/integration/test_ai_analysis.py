"""Tests for AI analysis integration."""
import pytest
from unittest.mock import patch, MagicMock

from src.api.ai_analysis import (
    analyze_market_trends,
    generate_investment_advice,
    predict_price_movement
)


@pytest.fixture
def mock_analyze_with_openai():
    """Mock the analyze_with_openai function."""
    with patch('src.api.ai_analysis.analyze_with_openai') as mock_analyze:
        mock_analyze.return_value = (
            "Bitcoin is showing bullish patterns with increased volume. "
            "Key resistance at $35,000. Recommend cautious accumulation."
        )
        yield mock_analyze


def test_analyze_market_trends(mock_analyze_with_openai):
    """Test market trend analysis using AI."""
    # Test analysis for Bitcoin
    analysis = analyze_market_trends('BTC', 'daily')
    
    # Check that function returned a non-empty string with expected content
    assert isinstance(analysis, str)
    assert len(analysis) > 0
    assert "Bitcoin" in analysis or "bullish" in analysis
    
    # Verify mock was called with correct parameters
    mock_analyze_with_openai.assert_called_once()
    call_args = mock_analyze_with_openai.call_args[0][0]  # First positional arg
    assert "BTC" in call_args
    assert "daily" in call_args


def test_generate_investment_advice(mock_analyze_with_openai):
    """Test generating investment advice using AI."""
    # Test advice generation for a portfolio
    portfolio = [
        {'symbol': 'BTC', 'allocation': 0.5},
        {'symbol': 'ETH', 'allocation': 0.3},
        {'symbol': 'SOL', 'allocation': 0.2}
    ]
    advice = generate_investment_advice(portfolio, risk_level='moderate')
    
    # Check that function returned a non-empty string
    assert isinstance(advice, str)
    assert len(advice) > 0
    
    # Verify mock was called with correct parameters
    assert mock_analyze_with_openai.call_count > 0
    call_args = mock_analyze_with_openai.call_args[0][0]  # First positional arg
    assert "BTC" in call_args or "portfolio" in call_args
    assert "moderate" in call_args


def test_predict_price_movement(mock_analyze_with_openai):
    """Test predicting price movement using AI."""
    # Test prediction for Bitcoin
    prediction = predict_price_movement('BTC', timeframe='7d')
    
    # Check that function returned a non-empty string
    assert isinstance(prediction, str)
    assert len(prediction) > 0
    
    # Verify mock was called with correct parameters
    assert mock_analyze_with_openai.call_count > 0
    call_args = mock_analyze_with_openai.call_args[0][0]  # First positional arg
    assert "BTC" in call_args
    assert "7d" in call_args or "timeframe" in call_args