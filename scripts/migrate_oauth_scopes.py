# \!/usr/bin/env python
"""
Migration script to update OAuth scopes for existing connections.
"""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import logging
from typing import Dict

from src.models.database import SessionLocal, User
from src.utils.oauth_config import (
    OAUTH_CONFIGS,
    get_oauth_access_token,
    refresh_oauth_token,
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def update_coinbase_oauth_scope():
    """
    Update OAuth scopes for Coinbase users to ensure they include the required permissions.
    This is needed when we add new API features that require additional scopes.
    """
    db = SessionLocal()

    try:
        # Find users with Coinbase OAuth
        coinbase_users = db.query(User).filter(User.oauth_provider == "coinbase").all()
        logger.info(f"Found {len(coinbase_users)} users with Coinbase OAuth")

        for user in coinbase_users:
            logger.info(f"Processing user: {user.username}")

            # Get their current OAuth token
            token = get_oauth_access_token(user)
            if token:
                logger.info("User has a valid OAuth token")

                # Force a token refresh to get new scopes
                success = refresh_oauth_token(user)
                if success:
                    logger.info(
                        "Successfully refreshed OAuth token with updated scopes"
                    )
                else:
                    logger.warning("Failed to refresh OAuth token")
            else:
                logger.warning("User does not have a valid OAuth token")

        logger.info("Completed OAuth scope update process")
    except Exception as e:
        logger.error(f"Error updating OAuth scopes: {str(e)}")
    finally:
        db.close()


def create_coinbase_api_keys():
    """
    Create API keys from OAuth tokens for Coinbase users who don't have them yet.
    """
    db = SessionLocal()

    try:
        # Find users with Coinbase OAuth
        coinbase_users = db.query(User).filter(User.oauth_provider == "coinbase").all()
        logger.info(f"Found {len(coinbase_users)} users with Coinbase OAuth")

        from src.utils.oauth_config import create_api_keys_from_oauth

        for user in coinbase_users:
            logger.info(f"Processing user: {user.username}")

            # Create API keys from OAuth token
            results = create_api_keys_from_oauth(user, db)

            for service, success in results.items():
                if success:
                    logger.info(f"Successfully created {service} API key")
                else:
                    logger.warning(f"Failed to create {service} API key")

        logger.info("Completed API key creation process")
    except Exception as e:
        logger.error(f"Error creating API keys: {str(e)}")
    finally:
        db.close()


if __name__ == "__main__":
    logger.info("Starting OAuth migration script")

    # Update OAuth scopes first
    update_coinbase_oauth_scope()

    # Then create API keys from OAuth tokens
    create_coinbase_api_keys()

    logger.info("OAuth migration completed")
