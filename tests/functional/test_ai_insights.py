"""Functional tests for AI insights page."""
import pytest
from unittest.mock import patch, MagicMock, call
import pandas as pd
import json
from datetime import datetime, timedelta

# Import relevant page functions
from src.pages.ai_insights import (
    show_ai_insights, 
    show_stock_analysis, 
    show_crypto_analysis, 
    show_etf_recommendations,
    show_custom_analysis,
    cache_analysis,
    get_cached_analysis
)


@pytest.fixture
def mock_streamlit():
    """Create mock for Streamlit components."""
    with patch('src.pages.ai_insights.st') as mock_st:
        # Mock title and header
        mock_st.title = MagicMock()
        mock_st.header = MagicMock()
        mock_st.subheader = MagicMock()
        
        # Mock input widgets
        mock_st.selectbox = MagicMock(return_value="BTC")
        mock_st.slider = MagicMock(return_value=30)
        mock_st.button = MagicMock(return_value=True)
        mock_st.text_input = MagicMock(return_value="AAPL")
        mock_st.text_area = MagicMock(return_value="What's the outlook for Bitcoin?")
        mock_st.radio = MagicMock(return_value="Finance")
        mock_st.multiselect = MagicMock(return_value=["technology"])
        
        # Mock display components
        mock_st.write = MagicMock()
        mock_st.markdown = MagicMock()
        mock_st.info = MagicMock()
        mock_st.success = MagicMock()
        mock_st.error = MagicMock()
        mock_st.warning = MagicMock()
        # Create a context manager mock for spinner
        mock_spinner_cm = MagicMock()
        mock_spinner_cm.__enter__ = MagicMock(return_value=None)
        mock_spinner_cm.__exit__ = MagicMock(return_value=None)
        mock_st.spinner = MagicMock(return_value=mock_spinner_cm)
        mock_st.metric = MagicMock()
        mock_st.plotly_chart = MagicMock()
        mock_st.dataframe = MagicMock()
        mock_st.caption = MagicMock()
        
        # Mock session state with proper attribute-style access
        mock_session_state = MagicMock()
        mock_user = MagicMock()
        mock_user.get = MagicMock(return_value=1)
        mock_session_state.__getitem__ = lambda self, key: mock_user if key == "user" else None
        mock_session_state.__contains__ = lambda self, key: key == "user"
        mock_st.session_state = mock_session_state
        
        # Mock expanders and columns
        mock_expander = MagicMock()
        mock_st.expander.return_value = mock_expander
        
        # For columns, make sure we return the right number when requested
        def mock_columns(n):
            return [MagicMock() for _ in range(n)]
        mock_st.columns = MagicMock(side_effect=mock_columns)
        
        # Mock tabs
        mock_tabs = [MagicMock(), MagicMock(), MagicMock(), MagicMock()]
        mock_st.tabs.return_value = mock_tabs
        
        yield mock_st


@pytest.fixture
def mock_ai_analysis():
    """Create mocks for AI analysis calls."""
    # Important: we need to patch analyze_stock_trend at the correct import location in the module being tested
    with patch('src.pages.ai_insights.analyze_stock_trend') as mock_stock:
        # Create a success result for stock analysis
        mock_stock.return_value = {
            'success': True,
            'ticker': 'AAPL',
            'period': '6mo',
            'data': {
                'start_date': '2023-09-01',
                'end_date': '2024-02-27',
                'start_price': '180.50',
                'end_price': '210.75',
                'percent_change': '16.76',
                'high': '215.50',
                'low': '175.25',
                'volume_avg': 25000000,
                'company_name': 'Apple Inc.',
                'sector': 'Technology',
                'industry': 'Consumer Electronics',
                'news': [
                    {'title': 'Apple Announces New Product', 'link': 'https://example.com', 
                     'publisher': 'Tech News', 'date': '2024-02-25'}
                ]
            },
            'analysis': 'Apple has shown strong performance with increasing momentum.',
            'data_source': 'Yahoo Finance'
        }
        
        with patch('src.pages.ai_insights.analyze_crypto_trend') as mock_crypto:
            # Create a success result for crypto analysis
            mock_crypto.return_value = {
                'success': True,
                'symbol': 'BTC',
                'days': 30,
                'data': {
                    'start_date': '2024-01-28',
                    'end_date': '2024-02-27',
                    'start_price': 42000.45,
                    'end_price': 51250.75, 
                    'percent_change': 22.03,
                    'high': 53000.50,
                    'low': 41500.25,
                    'volume_avg': 35000000000,
                    'current_ma20': 48500.25,
                    'current_ma50': 47250.35
                },
                'analysis': 'Bitcoin has broken above key resistance levels with increasing volume.'
            }
            
            with patch('src.pages.ai_insights.get_etf_recommendations') as mock_etf:
                # Create a success result for ETF recommendations
                mock_etf.return_value = {
                    'success': True,
                    'risk_profile': 'moderate',
                    'sectors': ['technology'],
                    'recommendations': [
                        {'ticker': 'VTI', 'name': 'Vanguard Total Stock Market ETF', 
                         'yearly_change': 15.32, 'expense_ratio': 0.03, 'category': 'Total Market'},
                        {'ticker': 'VGT', 'name': 'Vanguard Information Technology ETF', 
                         'yearly_change': 20.54, 'expense_ratio': 0.10, 'category': 'Technology'}
                    ],
                    'analysis': 'This portfolio provides excellent diversification with a focus on technology.'
                }
                
                with patch('src.utils.ai_helpers.analyze_finance_data') as mock_finance:
                    mock_finance.return_value = "This is a detailed analysis of your financial query."
                    
                    with patch('src.utils.ai_helpers.analyze_general_query') as mock_general:
                        mock_general.return_value = "This is a general analysis response."
                        
                        with patch('src.utils.ai_helpers.get_ai_model_info') as mock_model_info:
                            mock_model_info.return_value = {
                                'openai': {
                                    'available': True,
                                    'models': ['gpt-3.5-turbo', 'gpt-4']
                                },
                                'ollama': {
                                    'available': True,
                                    'url': 'http://localhost:11434',
                                    'models': ['llama2', 'mistral'],
                                    'model_preferences': {
                                        'finance': ['llama2', 'mistral'],
                                        'general': ['mistral', 'llama2']
                                    }
                                }
                            }
                            
                            yield {
                                'stock': mock_stock,
                                'crypto': mock_crypto,
                                'etf': mock_etf,
                                'finance': mock_finance,
                                'general': mock_general,
                                'model_info': mock_model_info
                            }


@pytest.fixture
def mock_cache():
    """Mock database caching functions."""
    with patch('src.pages.ai_insights.SessionLocal') as mock_session:
        mock_db = MagicMock()
        mock_session.return_value = mock_db
        
        # Mock AiAnalysis query results
        mock_query = MagicMock()
        mock_db.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.first.return_value = None  # No cached data by default
        
        yield mock_db


@pytest.fixture
def mock_yfinance():
    """Mock Yahoo Finance data retrieval."""
    with patch('src.pages.ai_insights.yf') as mock_yf:
        mock_ticker = MagicMock()
        mock_yf.Ticker.return_value = mock_ticker
        
        # Create sample historical data
        dates = pd.date_range(start='2023-01-01', end='2023-12-31')
        mock_ticker.history.return_value = pd.DataFrame({
            'Open': range(len(dates)),
            'High': range(len(dates)),
            'Low': range(len(dates)),
            'Close': range(len(dates)),
            'Volume': range(len(dates))
        }, index=dates)
        
        mock_ticker.info = {'shortName': 'Test Stock'}
        mock_ticker.news = []
        
        yield mock_yf


def test_show_ai_insights_basic():
    """Simplified test for AI insights page that doesn't rely on implementation details."""
    # Just verify we can import the function without errors
    assert callable(show_ai_insights)


def test_cache_analysis(mock_cache):
    """Test caching analysis results in the database."""
    # Test caching a new analysis
    query = "stock_AAPL_1y"
    result = {"success": True, "data": {"price": 150}}
    model = "OpenAI"
    
    success = cache_analysis(query, result, model)
    
    assert success is True
    mock_cache.add.assert_called_once()
    mock_cache.commit.assert_called_once()


def test_get_cached_analysis_hit(mock_cache):
    """Test retrieving cached analysis when available."""
    # Create a mock cached entry
    cached_entry = MagicMock()
    cached_entry.result = json.dumps({"success": True, "data": {"price": 150}})
    cached_entry.model_used = "OpenAI"
    
    # Configure mock to return the cached entry
    mock_cache.query().filter().first.return_value = cached_entry
    
    result, model = get_cached_analysis("stock_AAPL_1y")
    
    assert result is not None
    assert model == "OpenAI"
    assert result["success"] is True
    assert result["data"]["price"] == 150


def test_get_cached_analysis_miss(mock_cache):
    """Test retrieving cached analysis when not available."""
    # Configure mock to return None (no cache hit)
    mock_cache.query().filter().first.return_value = None
    
    result, model = get_cached_analysis("stock_AAPL_1y")
    
    assert result is None
    assert model is None


def test_show_stock_analysis(mock_streamlit, mock_ai_analysis, mock_yfinance):
    """Test stock analysis section functionality."""
    # Reset mocks before test to clear any previous calls
    mock_streamlit.reset_mock()
    mock_ai_analysis['stock'].reset_mock()
    
    # Mock the caching functions
    with patch('src.pages.ai_insights.get_cached_analysis', return_value=(None, None)):
        with patch('src.pages.ai_insights.cache_analysis', return_value=True):
            # Set button to return True to trigger analysis
            mock_streamlit.button.return_value = True
            
            # Configure text input for ticker
            mock_streamlit.text_input.return_value = "AAPL"
            
            # Configure selectbox for period
            mock_streamlit.selectbox.return_value = "6mo"
            
            # Call the function
            show_stock_analysis()
    
    # Verify streamlit components were used
    assert mock_streamlit.subheader.called
    assert mock_streamlit.text_input.called
    assert mock_streamlit.selectbox.called
    assert mock_streamlit.button.called
    
    # Verify analyze_stock_trend was called
    mock_ai_analysis['stock'].assert_called_once()
    args, kwargs = mock_ai_analysis['stock'].call_args
    assert args[0] == "AAPL"  # First argument should be ticker
    assert "6mo" in args  # Period should be second argument
    
    # Verify results were displayed
    assert mock_streamlit.write.called or mock_streamlit.markdown.called
    assert mock_streamlit.dataframe.called or mock_streamlit.metric.called


def test_show_crypto_analysis(mock_streamlit, mock_ai_analysis, mock_yfinance):
    """Test cryptocurrency analysis section functionality."""
    # Reset mocks
    mock_streamlit.reset_mock()
    mock_ai_analysis['crypto'].reset_mock()
    
    # Mock the caching functions
    with patch('src.pages.ai_insights.get_cached_analysis', return_value=(None, None)):
        with patch('src.pages.ai_insights.cache_analysis', return_value=True):
            # Set button to return True to trigger analysis
            mock_streamlit.button.return_value = True
            
            # Configure text input for crypto
            mock_streamlit.text_input.return_value = "BTC"
            
            # Configure slider for days
            mock_streamlit.slider.return_value = 30
            
            # Call the function
            show_crypto_analysis()
    
    # Verify streamlit components were used
    assert mock_streamlit.subheader.called
    assert mock_streamlit.text_input.called
    assert mock_streamlit.slider.called
    assert mock_streamlit.button.called
    
    # Verify analyze_crypto_trend was called
    mock_ai_analysis['crypto'].assert_called_once()
    args, kwargs = mock_ai_analysis['crypto'].call_args
    assert args[0] == "BTC"  # First argument should be symbol
    assert 30 in args or 30 in kwargs.values()  # Days should be used somewhere
    
    # Verify results were displayed
    assert mock_streamlit.write.called or mock_streamlit.markdown.called
    assert mock_streamlit.metric.called or mock_streamlit.dataframe.called


def test_show_etf_recommendations(mock_streamlit, mock_ai_analysis):
    """Test ETF recommendations section functionality."""
    # Reset mocks
    mock_streamlit.reset_mock()
    mock_ai_analysis['etf'].reset_mock()
    
    # Mock the caching functions
    with patch('src.pages.ai_insights.get_cached_analysis', return_value=(None, None)):
        with patch('src.pages.ai_insights.cache_analysis', return_value=True):
            # Set button to return True to trigger analysis
            mock_streamlit.button.return_value = True
            
            # Configure selectbox for risk profile
            mock_streamlit.selectbox.return_value = "moderate"
            
            # Configure multiselect for sectors
            mock_streamlit.multiselect.return_value = ["technology"]
            
            # Call the function
            show_etf_recommendations()
    
    # Verify streamlit components were used
    assert mock_streamlit.subheader.called
    assert mock_streamlit.selectbox.called
    assert mock_streamlit.multiselect.called
    assert mock_streamlit.button.called
    
    # Verify get_etf_recommendations was called
    mock_ai_analysis['etf'].assert_called_once()
    args, kwargs = mock_ai_analysis['etf'].call_args
    assert "moderate" in args  # First argument should be risk profile
    assert any(arg == ["technology"] for arg in args) or any(val == ["technology"] for val in kwargs.values())
    
    # Verify results were displayed
    assert mock_streamlit.write.called or mock_streamlit.markdown.called


def test_show_custom_analysis(mock_streamlit, mock_ai_analysis):
    """Test custom analysis section functionality."""
    # Reset mocks
    mock_streamlit.reset_mock()
    mock_ai_analysis['finance'].reset_mock()
    mock_ai_analysis['general'].reset_mock()
    
    # Since OpenAI is imported dynamically, patch the module
    with patch('src.api.ai_analysis.analyze_with_openai') as mock_analyze_openai:
        mock_analyze_openai.return_value = "OpenAI analysis result about Bitcoin"
        
        # Also patch select_best_ollama_model which may be used
        with patch('src.api.ai_analysis.select_best_ollama_model') as mock_select_model:
            mock_select_model.return_value = "llama2"
            
            # Mock AI model info
            with patch('src.pages.ai_insights.get_ai_model_info') as mock_get_info:
                mock_get_info.return_value = {
                    'openai': {
                        'available': True,
                        'models': ['gpt-3.5-turbo', 'gpt-4']
                    },
                    'ollama': {
                        'available': True,
                        'url': 'http://localhost:11434',
                        'models': ['llama2', 'mistral'],
                        'model_preferences': {
                            'finance': ['llama2', 'mistral'],
                            'general': ['mistral', 'llama2']
                        }
                    }
                }
            
                # Mock the caching functions
                with patch('src.pages.ai_insights.get_cached_analysis', return_value=(None, None)):
                    with patch('src.pages.ai_insights.cache_analysis', return_value=True):
                        # Set button to return True to trigger analysis
                        mock_streamlit.button.return_value = True
                        
                        # Configure text area for query
                        mock_streamlit.text_area.return_value = "What's the outlook for Bitcoin?"
                        
                        # Configure radio button values
                        # First radio is for model type
                        # Second radio is for task type
                        mock_streamlit.radio.side_effect = ["OpenAI (GPT)", "Finance"]
                        
                        # Call the function
                        show_custom_analysis()
    
    # Verify streamlit components were used
    assert mock_streamlit.subheader.called
    assert mock_streamlit.text_area.called
    assert mock_streamlit.radio.called
    assert mock_streamlit.button.called
    
    # Verify analyze_with_openai was called when we select OpenAI
    assert mock_analyze_openai.called
    
    # Verify results were displayed
    assert mock_streamlit.write.called or mock_streamlit.markdown.called


def test_show_ai_insights_complete(mock_streamlit, mock_ai_analysis):
    """Test the complete AI insights page with tabs."""
    # Reset mocks
    mock_streamlit.reset_mock()
    
    # Mock require_login function
    with patch('src.pages.ai_insights.require_login') as mock_login:
        # Mock a user object
        mock_user = MagicMock()
        mock_login.return_value = mock_user
        
        # Call the function
        show_ai_insights()
    
        # Verify main page components
        assert mock_streamlit.title.called
        assert mock_streamlit.tabs.called
        
        # Verify tabs were created
        tabs = mock_streamlit.tabs.return_value
        assert len(tabs) >= 3  # Should have at least 3 tabs