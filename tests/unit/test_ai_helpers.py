"""Unit tests for AI helpers module."""
import pytest
from unittest.mock import patch, MagicMock
import os
import json
import logging

from src.utils.ai_helpers import (
    get_ai_model_info,
    analyze_finance_data,
    analyze_code,
    analyze_general_query,
    analyze_with_best_model
)


@pytest.fixture
def mock_openai_analyze():
    """Mock the OpenAI analyze function."""
    with patch('src.utils.ai_helpers.analyze_with_openai') as mock_analyze:
        mock_analyze.return_value = "OpenAI analysis result"
        yield mock_analyze


@pytest.fixture
def mock_ollama_analyze():
    """Mock the Ollama analyze function."""
    with patch('src.utils.ai_helpers.analyze_with_ollama') as mock_analyze:
        mock_analyze.return_value = "Ollama analysis result"
        yield mock_analyze


@pytest.fixture
def mock_available_models():
    """Mock getting available Ollama models."""
    with patch('src.utils.ai_helpers.get_available_ollama_models') as mock_models:
        mock_models.return_value = ["llama2", "mistral", "codellama"]
        yield mock_models


@pytest.fixture
def mock_ollama_settings():
    """Mock getting Ollama settings."""
    with patch('src.utils.ai_helpers.get_ollama_settings') as mock_settings:
        mock_settings.return_value = ("http://localhost:11434/", "llama2")
        yield mock_settings


@pytest.fixture
def mock_env_vars():
    """Mock environment variables."""
    original_env = os.environ.copy()
    os.environ["OPENAI_API_KEY"] = "fake-api-key"
    os.environ["OLLAMA_MODEL"] = "llama2"
    
    yield
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


def test_get_ai_model_info(mock_available_models, mock_ollama_settings, mock_env_vars):
    """Test getting AI model information."""
    info = get_ai_model_info()
    
    # Check OpenAI info
    assert info["openai"]["available"] is True
    assert info["openai"]["api_key_set"] is True
    assert "gpt-3.5-turbo" in info["openai"]["models"]
    
    # Check Ollama info
    assert info["ollama"]["available"] is True
    assert "llama2" in info["ollama"]["models"]
    assert "finance" in info["ollama"]["model_preferences"]
    assert "general" in info["ollama"]["model_preferences"]
    assert "coding" in info["ollama"]["model_preferences"]


def test_analyze_finance_data(mock_openai_analyze, mock_env_vars):
    """Test analyzing financial data."""
    result = analyze_finance_data("What is the current market trend?")
    
    assert result == "OpenAI analysis result"
    mock_openai_analyze.assert_called_once()
    args, kwargs = mock_openai_analyze.call_args
    assert args[0] == "What is the current market trend?"
    assert kwargs["task_type"] == "finance"


def test_analyze_code(mock_openai_analyze, mock_env_vars):
    """Test analyzing code."""
    result = analyze_code("def hello_world(): print('Hello World')")
    
    assert result == "OpenAI analysis result"
    mock_openai_analyze.assert_called_once()
    args, kwargs = mock_openai_analyze.call_args
    assert args[0] == "def hello_world(): print('Hello World')"
    assert kwargs["task_type"] == "coding"


def test_analyze_general_query(mock_openai_analyze, mock_env_vars):
    """Test analyzing general query."""
    result = analyze_general_query("What is the weather like today?")
    
    assert result == "OpenAI analysis result"
    mock_openai_analyze.assert_called_once()
    args, kwargs = mock_openai_analyze.call_args
    assert args[0] == "What is the weather like today?"
    assert kwargs["task_type"] == "general"


def test_analyze_with_best_model_openai_success(mock_openai_analyze, mock_ollama_analyze, mock_env_vars):
    """Test using the best model with OpenAI succeeding."""
    result = analyze_with_best_model("Test prompt", task_type="finance")
    
    assert result == "OpenAI analysis result"
    mock_openai_analyze.assert_called_once()
    mock_ollama_analyze.assert_not_called()


def test_analyze_with_best_model_openai_failure(mock_openai_analyze, mock_ollama_analyze, mock_env_vars):
    """Test using the best model with OpenAI failing and Ollama succeeding."""
    # Make OpenAI fail
    mock_openai_analyze.side_effect = Exception("API error")
    
    result = analyze_with_best_model("Test prompt", task_type="general")
    
    assert result == "Ollama analysis result"
    mock_openai_analyze.assert_called_once()
    mock_ollama_analyze.assert_called_once()


def test_analyze_with_best_model_all_failure(mock_openai_analyze, mock_ollama_analyze, mock_env_vars):
    """Test using the best model with both OpenAI and Ollama failing."""
    # Make both APIs fail
    mock_openai_analyze.side_effect = Exception("OpenAI API error")
    mock_ollama_analyze.side_effect = Exception("Ollama API error")
    
    # With custom fallback message
    result = analyze_with_best_model("Test prompt", fallback_message="Custom error message")
    assert result == "Custom error message"
    
    # Reset call counts
    mock_openai_analyze.reset_mock()
    mock_ollama_analyze.reset_mock()
    
    # Without custom fallback message
    result = analyze_with_best_model("Test prompt")
    assert "Analysis could not be generated" in result
    
    # Verify both were called both times
    assert mock_openai_analyze.call_count == 1
    assert mock_ollama_analyze.call_count == 1


def test_analyze_with_best_model_with_specific_task(mock_openai_analyze, mock_env_vars):
    """Test using the best model with specific task type."""
    # Test different task types
    task_types = ["finance", "coding", "general"]
    
    for task_type in task_types:
        mock_openai_analyze.reset_mock()
        result = analyze_with_best_model("Test prompt", task_type=task_type)
        
        assert result == "OpenAI analysis result"
        mock_openai_analyze.assert_called_once()
        assert mock_openai_analyze.call_args[1]["task_type"] == task_type