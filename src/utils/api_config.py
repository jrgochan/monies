import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv, set_key

from src.utils.api_tester import APITester

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class APIConfigManager:
    """Class for managing API configurations"""

    @staticmethod
    def get_env_path() -> str:
        """Get the path to the .env file"""
        # Find the .env file in the project root directory
        root_dir = Path(__file__).parent.parent.parent
        env_path = os.path.join(root_dir, ".env")
        return env_path

    @staticmethod
    def get_api_configs() -> List[Dict]:
        """Get all API configurations"""
        return APITester.get_api_config_info()

    @staticmethod
    def get_api_config_by_service(service_id: str) -> Optional[Dict]:
        """Get API configuration by service ID"""
        configs = APITester.get_api_config_info()
        for config in configs:
            if config["service_id"] == service_id:
                return config
        return None

    @staticmethod
    def get_api_categories() -> List[str]:
        """Get unique API categories"""
        configs = APITester.get_api_config_info()
        categories = set()
        for config in configs:
            categories.add(config.get("category", "Other"))
        return sorted(list(categories))

    @staticmethod
    def get_api_configs_by_category(category: str) -> List[Dict]:
        """Get API configurations by category"""
        configs = APITester.get_api_config_info()
        return [
            config for config in configs if config.get("category", "Other") == category
        ]

    @staticmethod
    def get_api_value_from_env(env_var: str) -> str:
        """Get API value from environment variable"""
        if not env_var:
            return ""
        return os.getenv(env_var, "")

    @staticmethod
    def update_env_file(key: str, value: str) -> bool:
        """Update environment variable in .env file"""
        try:
            env_path = APIConfigManager.get_env_path()
            set_key(env_path, key, value)
            # Reload environment variables
            load_dotenv()
            return True
        except Exception as e:
            logger.error(f"Error updating environment variable {key}: {str(e)}")
            return False

    @staticmethod
    def get_api_credentials(
        service_id: str, user_id: Optional[int] = None, key_id: Optional[int] = None
    ) -> Tuple[str, str]:
        """Get API credentials for a service

        Args:
            service_id: Service ID to get credentials for
            user_id: User ID to get credentials for (if using database-stored keys)
            key_id: Specific API key ID to use (optional)

        Returns:
            Tuple of (key, secret)
        """
        config = APIConfigManager.get_api_config_by_service(service_id)
        if not config:
            return "", ""

        # If user_id is provided, try to get from database first
        if user_id:
            try:
                from src.models.database import SessionLocal
                from src.utils.security import get_api_key

                db = SessionLocal()
                try:
                    key, secret, _ = get_api_key(db, user_id, service_id, key_id)
                    if key:
                        return key, secret or ""
                finally:
                    db.close()
            except Exception as e:
                # Fall back to environment variables if database retrieval fails
                print(f"Error getting API key from database: {str(e)}")
                pass

        # Fall back to environment variables
        key = APIConfigManager.get_api_value_from_env(config.get("env_var_key", ""))
        secret = APIConfigManager.get_api_value_from_env(
            config.get("env_var_secret", "")
        )

        return key, secret

    @staticmethod
    def test_api_connection(
        service_id: str,
        key: Optional[str] = None,
        secret: Optional[str] = None,
        user_id: Optional[int] = None,
        key_id: Optional[int] = None,
    ) -> Tuple[bool, str]:
        """Test API connection

        Args:
            service_id: Service ID to test
            key: API key to use (optional)
            secret: API secret to use (optional)
            user_id: User ID to get credentials for (optional)
            key_id: Specific API key ID to use (optional)

        Returns:
            Tuple of (success, message)
        """
        config = APIConfigManager.get_api_config_by_service(service_id)
        if not config:
            return False, f"Unknown service: {service_id}"

        # Use provided credentials or get from user's database entry or environment
        if key is None and secret is None:
            if user_id:
                # Try to get user-specific credentials
                key, secret = APIConfigManager.get_api_credentials(
                    service_id, user_id, key_id
                )
            else:
                # Fall back to environment variables
                if config.get("env_var_key"):
                    key = APIConfigManager.get_api_value_from_env(
                        config.get("env_var_key", "")
                    )

                if config.get("env_var_secret"):
                    secret = APIConfigManager.get_api_value_from_env(
                        config.get("env_var_secret", "")
                    )

        # Check if we need a URL instead of a key
        if config.get("is_url", False):
            # For services like Ollama that use a URL instead of an API key
            url = key if key else config.get("default_url", "")
            test_func = getattr(APITester, f"test_{service_id}")
            return test_func(url)

        # Special handling for APIs that don't require keys (like Yahoo Finance)
        if not config.get("needs_key", True):
            test_func = getattr(APITester, f"test_{service_id}")
            return test_func()

        # Check if we have the necessary credentials
        if config.get("needs_key", True) and not key:
            return False, f"API key required for {config['name']}"

        if config.get("needs_secret", False) and not secret:
            return False, f"API secret required for {config['name']}"

        # Call the corresponding test function
        test_func = getattr(APITester, f"test_{service_id}")

        if config.get("needs_secret", False):
            return test_func(key, secret)
        else:
            return test_func(key)

    @staticmethod
    def save_api_credentials(
        service_id: str,
        key: str,
        secret: Optional[str] = None,
        db: Optional[Any] = None,
        user_id: Optional[int] = None,
        model_preference: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """Save API credentials to .env file and/or database"""
        config = APIConfigManager.get_api_config_by_service(service_id)
        if not config:
            return False, f"Unknown service: {service_id}"

        success = True
        message = ""

        # Update environment variables
        if config.get("env_var_key") and key:
            env_success = APIConfigManager.update_env_file(
                config.get("env_var_key"), key
            )
            if not env_success:
                success = False
                message += (
                    f"Failed to update {config.get('env_var_key')} in .env file. "
                )

        if config.get("env_var_secret") and secret:
            env_success = APIConfigManager.update_env_file(
                config.get("env_var_secret"), secret
            )
            if not env_success:
                success = False
                message += (
                    f"Failed to update {config.get('env_var_secret')} in .env file. "
                )

        # Handle model preference for AI services like Ollama
        if model_preference and service_id == "ollama":
            env_success = APIConfigManager.update_env_file(
                "OLLAMA_MODEL", model_preference
            )
            if not env_success:
                success = False
                message += "Failed to update OLLAMA_MODEL in .env file. "
            else:
                # Set for immediate use in this session
                os.environ["OLLAMA_MODEL"] = model_preference

        # Save to database if db and user_id are provided
        if db and user_id:
            try:
                from src.utils.security import store_api_key

                key_obj = store_api_key(db, user_id, service_id, key, secret)
                if not key_obj:
                    success = False
                    message += "Failed to store API key in database. "
            except Exception as e:
                success = False
                message += f"Database error: {str(e)}"

        if success and not message:
            if model_preference and service_id == "ollama":
                message = (
                    f"Successfully saved {config['name']} settings and model preference"
                )
            else:
                message = f"Successfully saved {config['name']} API credentials"

        return success, message
