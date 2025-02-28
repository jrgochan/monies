"""Tests for social media integration."""
import pytest
from unittest.mock import patch, MagicMock

from src.api.social_media import (
    analyze_social_sentiment,
    post_to_twitter,
    get_twitter_trending_topics
)


@pytest.fixture
def mock_tweepy_client():
    """Create a mock Tweepy client."""
    with patch('src.api.social_media.tweepy.Client') as mock_client:
        client_instance = MagicMock()
        mock_client.return_value = client_instance
        
        # Mock the create_tweet method with proper return structure
        response = MagicMock()
        response.data = {'id': '12345'}
        client_instance.create_tweet.return_value = response
        
        yield client_instance


@pytest.fixture
def mock_sentiment_analyzer():
    """Create a mock sentiment analyzer."""
    with patch('src.api.social_media.analyze_text_sentiment') as mock_analyzer:
        # Mock the sentiment analysis function
        mock_analyzer.return_value = {
            'positive': 0.65,
            'negative': 0.15,
            'neutral': 0.20,
            'overall': 'positive'
        }
        
        yield mock_analyzer


def test_analyze_social_sentiment():
    """Test analyzing social media sentiment."""
    # Test sentiment analysis for Bitcoin
    sentiment = analyze_social_sentiment('bitcoin')
    
    # Check that function returned the expected structure
    assert isinstance(sentiment, dict)
    assert 'positive' in sentiment
    assert 'negative' in sentiment
    assert 'neutral' in sentiment
    assert 'overall' in sentiment
    
    # Values should be as expected from our mock implementation
    assert sentiment['overall'] == 'positive'
    assert sentiment['positive'] == 0.65


def test_post_to_twitter(mock_tweepy_client):
    """Test posting to Twitter."""
    # Test posting a tweet
    result = post_to_twitter(
        message="Testing my crypto portfolio app! #Bitcoin #Crypto",
        api_key="test_api_key",
        api_secret="test_api_secret",
        access_token="test_access_token",
        access_secret="test_access_secret"
    )
    
    # Check that function returned the expected response
    assert result['success'] is True
    assert result['tweet_id'] == '12345'
    
    # Verify the create_tweet method was called
    mock_tweepy_client.create_tweet.assert_called_once_with(
        text="Testing my crypto portfolio app! #Bitcoin #Crypto"
    )


def test_get_twitter_trending_topics(mock_tweepy_client):
    """Test retrieving trending topics from Twitter."""
    # Test getting crypto trending topics
    trends = get_twitter_trending_topics(
        search_term="crypto",
        api_key="test_api_key",
        api_secret="test_api_secret",
        access_token="test_access_token",
        access_secret="test_access_secret"
    )
    
    # Check that function returned a list with expected content
    assert isinstance(trends, list)
    assert len(trends) == 3
    assert trends[0]['name'] == '#Bitcoin'
    assert trends[0]['tweet_volume'] == 50000