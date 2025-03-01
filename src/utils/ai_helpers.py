"""
AI helper utilities for intelligent model selection and analysis.
This file provides a unified interface for AI analysis functions.
"""

import logging
import os
from typing import Any, Dict

from src.api.ai_analysis import (
    analyze_with_ollama,
    analyze_with_openai,
    get_available_ollama_models,
    get_ollama_settings,
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_ai_model_info() -> Dict[str, Any]:
    """
    Get information about available AI models
    """
    # OpenAI status
    openai_api_key = os.getenv("OPENAI_API_KEY", "")
    openai_available = bool(openai_api_key)

    # Ollama status
    base_url, default_model = get_ollama_settings()
    available_models = get_available_ollama_models()
    ollama_available = bool(available_models)

    return {
        "openai": {
            "available": openai_available,
            "api_key_set": bool(openai_api_key),
            "models": ["gpt-3.5-turbo", "gpt-4"] if openai_available else [],
        },
        "ollama": {
            "available": ollama_available,
            "url": base_url,
            "default_model": default_model,
            "models": available_models,
            "model_preferences": {
                "finance": [
                    "mistral-medium",
                    "mixtral",
                    "llama3",
                    "llama3:70b",
                    "llama3:8b",
                    "mistral",
                    "codellama",
                    "llama2:70b",
                    "llama2",
                ],
                "general": [
                    "llama3",
                    "llama3:70b",
                    "llama3:8b",
                    "mistral",
                    "mistral-medium",
                    "mixtral",
                    "llama2:70b",
                    "llama2",
                ],
                "coding": [
                    "codellama",
                    "llama3",
                    "llama3:70b",
                    "mixtral",
                    "mistral-medium",
                    "mistral",
                    "llama2:70b",
                    "llama2",
                ],
            },
        },
    }


def analyze_finance_data(prompt: str, fallback_message: str = None) -> str:
    """
    Analyze financial data using the best available model
    """
    return analyze_with_best_model(prompt, "finance", fallback_message)


def analyze_code(prompt: str, fallback_message: str = None) -> str:
    """
    Analyze code using the best available model
    """
    return analyze_with_best_model(prompt, "coding", fallback_message)


def analyze_general_query(prompt: str, fallback_message: str = None) -> str:
    """
    Analyze general query using the best available model
    """
    return analyze_with_best_model(prompt, "general", fallback_message)


def analyze_with_best_model(
    prompt: str, task_type: str = "finance", fallback_message: str = None
) -> str:
    """
    Use the best available model to analyze based on task type
    """
    # First try OpenAI
    try:
        # For finance tasks, try to use GPT-4 if the API key exists
        openai_model = "gpt-3.5-turbo"
        if task_type == "finance" and os.getenv("OPENAI_API_KEY"):
            # Note: this is a simplistic check, in reality we'd use OpenAI API to list available models
            openai_model = (
                "gpt-4"
                if "gpt-4" in os.getenv("OPENAI_API_KEY", "")
                else "gpt-3.5-turbo"
            )

        return analyze_with_openai(prompt, model=openai_model, task_type=task_type)
    except Exception as e:
        logger.warning(f"OpenAI analysis failed, trying Ollama: {str(e)}")

        # Fall back to Ollama with automatic model selection
        try:
            return analyze_with_ollama(prompt, model=None, task_type=task_type)
        except Exception as e2:
            logger.error(f"All AI analysis methods failed: {str(e2)}")

            # Use fallback message or generate a generic one
            if fallback_message:
                return fallback_message
            return (
                "Analysis could not be generated at this time. Please try again later."
            )
