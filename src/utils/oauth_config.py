"""
OAuth Configuration for authentication and API platform integration
"""
import os
import secrets
from datetime import datetime, timedelta
from typing import Any, Dict, Optional
from urllib.parse import urlencode

import requests
from authlib.integrations.requests_client import OAuth2Session
from dotenv import load_dotenv

from src.models.database import SessionLocal, User
from src.utils.security import decrypt_data, encrypt_data

# Load environment variables
load_dotenv()

# Google OAuth Configuration
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")
GOOGLE_REDIRECT_URI = os.getenv(
    "GOOGLE_REDIRECT_URI", "http://localhost:8501/callback/google"
)

# Coinbase OAuth Configuration
COINBASE_CLIENT_ID = os.getenv("COINBASE_CLIENT_ID", "")
COINBASE_CLIENT_SECRET = os.getenv("COINBASE_CLIENT_SECRET", "")
COINBASE_REDIRECT_URI = os.getenv(
    "COINBASE_REDIRECT_URI", "http://localhost:8501/callback/coinbase"
)

# Additional OAuth provider environment variables
FACEBOOK_CLIENT_ID = os.getenv("FACEBOOK_CLIENT_ID", "")
FACEBOOK_CLIENT_SECRET = os.getenv("FACEBOOK_CLIENT_SECRET", "")
FACEBOOK_REDIRECT_URI = os.getenv(
    "FACEBOOK_REDIRECT_URI", "http://localhost:8501/callback/facebook"
)

TWITTER_CLIENT_ID = os.getenv("TWITTER_CLIENT_ID", "")
TWITTER_CLIENT_SECRET = os.getenv("TWITTER_CLIENT_SECRET", "")
TWITTER_REDIRECT_URI = os.getenv(
    "TWITTER_REDIRECT_URI", "http://localhost:8501/callback/twitter"
)

GITHUB_CLIENT_ID = os.getenv("GITHUB_CLIENT_ID", "")
GITHUB_CLIENT_SECRET = os.getenv("GITHUB_CLIENT_SECRET", "")
GITHUB_REDIRECT_URI = os.getenv(
    "GITHUB_REDIRECT_URI", "http://localhost:8501/callback/github"
)

MICROSOFT_CLIENT_ID = os.getenv("MICROSOFT_CLIENT_ID", "")
MICROSOFT_CLIENT_SECRET = os.getenv("MICROSOFT_CLIENT_SECRET", "")
MICROSOFT_REDIRECT_URI = os.getenv(
    "MICROSOFT_REDIRECT_URI", "http://localhost:8501/callback/microsoft"
)

# OAuth provider configurations
OAUTH_CONFIGS = {
    "google": {
        "client_id": GOOGLE_CLIENT_ID,
        "client_secret": GOOGLE_CLIENT_SECRET,
        "authorize_url": "https://accounts.google.com/o/oauth2/auth",
        "token_url": "https://oauth2.googleapis.com/token",
        "userinfo_url": "https://www.googleapis.com/oauth2/v3/userinfo",
        "scope": "openid email profile",
        "redirect_uri": GOOGLE_REDIRECT_URI,
        "icon": "google",
        "color": "#DB4437",
        "display_name": "Google",
        "supports_api_keys": False,
    },
    "coinbase": {
        "client_id": COINBASE_CLIENT_ID,
        "client_secret": COINBASE_CLIENT_SECRET,
        "authorize_url": "https://www.coinbase.com/oauth/authorize",
        "token_url": "https://api.coinbase.com/oauth/token",
        "userinfo_url": "https://api.coinbase.com/v2/user",
        "scope": "wallet:user:read,wallet:accounts:read,wallet:transactions:read",
        "redirect_uri": COINBASE_REDIRECT_URI,
        "icon": "bitcoin",
        "color": "#0052FF",
        "display_name": "Coinbase",
        "supports_api_keys": True,
        "api_services": ["coinbase"],
    },
    "facebook": {
        "client_id": FACEBOOK_CLIENT_ID,
        "client_secret": FACEBOOK_CLIENT_SECRET,
        "authorize_url": "https://www.facebook.com/v16.0/dialog/oauth",
        "token_url": "https://graph.facebook.com/v16.0/oauth/access_token",
        "userinfo_url": "https://graph.facebook.com/me?fields=id,name,email,picture",
        "scope": "email,public_profile",
        "redirect_uri": FACEBOOK_REDIRECT_URI,
        "icon": "facebook",
        "color": "#1877F2",
        "display_name": "Facebook",
        "supports_api_keys": True,
        "api_services": ["facebook"],
    },
    "twitter": {
        "client_id": TWITTER_CLIENT_ID,
        "client_secret": TWITTER_CLIENT_SECRET,
        "authorize_url": "https://twitter.com/i/oauth2/authorize",
        "token_url": "https://api.twitter.com/2/oauth2/token",
        "userinfo_url": "https://api.twitter.com/2/users/me",
        "scope": "tweet.read users.read offline.access",
        "redirect_uri": TWITTER_REDIRECT_URI,
        "icon": "twitter",
        "color": "#1DA1F2",
        "display_name": "Twitter",
        "supports_api_keys": True,
        "api_services": ["twitter"],
    },
    "github": {
        "client_id": GITHUB_CLIENT_ID,
        "client_secret": GITHUB_CLIENT_SECRET,
        "authorize_url": "https://github.com/login/oauth/authorize",
        "token_url": "https://github.com/login/oauth/access_token",
        "userinfo_url": "https://api.github.com/user",
        "scope": "read:user user:email",
        "redirect_uri": GITHUB_REDIRECT_URI,
        "icon": "github",
        "color": "#24292E",
        "display_name": "GitHub",
        "supports_api_keys": True,
        "api_services": ["github"],
    },
    "microsoft": {
        "client_id": MICROSOFT_CLIENT_ID,
        "client_secret": MICROSOFT_CLIENT_SECRET,
        "authorize_url": "https://login.microsoftonline.com/common/oauth2/v2.0/authorize",
        "token_url": "https://login.microsoftonline.com/common/oauth2/v2.0/token",
        "userinfo_url": "https://graph.microsoft.com/v1.0/me",
        "scope": "openid email profile User.Read offline_access",
        "redirect_uri": MICROSOFT_REDIRECT_URI,
        "icon": "microsoft",
        "color": "#00A4EF",
        "display_name": "Microsoft",
        "supports_api_keys": False,
    },
}


def get_oauth_client(provider: str) -> Optional[OAuth2Session]:
    """
    Create an OAuth2 client for the specified provider
    """
    config = OAUTH_CONFIGS.get(provider)
    if not config or not config["client_id"] or not config["client_secret"]:
        return None

    return OAuth2Session(
        client_id=config["client_id"],
        client_secret=config["client_secret"],
        scope=config["scope"],
    )


def generate_oauth_authorize_url(provider: str, state: Optional[str] = None) -> tuple:
    """
    Generate the authorization URL for the specified OAuth provider
    Returns a tuple of (url, state)
    """
    config = OAUTH_CONFIGS.get(provider)
    if not config:
        return None, None

    # Create new state if not provided
    if not state:
        state = secrets.token_urlsafe(16)

    # Generate the authorize URL
    params = {
        "client_id": config["client_id"],
        "redirect_uri": config["redirect_uri"],
        "response_type": "code",
        "scope": config["scope"],
        "state": state,
        "access_type": "offline",  # For refresh tokens
        "prompt": "consent",  # To always ask for consent
    }

    # Special handling for Coinbase - force approval prompt
    if provider == "coinbase":
        params["account"] = "all"

    authorize_url = f"{config['authorize_url']}?{urlencode(params)}"
    return authorize_url, state


def exchange_code_for_token(provider: str, code: str) -> Optional[Dict[str, Any]]:
    """
    Exchange the authorization code for an access token
    """
    config = OAUTH_CONFIGS.get(provider)
    if not config:
        return None

    try:
        # Create OAuth session
        oauth = OAuth2Session(
            client_id=config["client_id"],
            client_secret=config["client_secret"],
            redirect_uri=config["redirect_uri"],
        )

        # Exchange code for token
        token = oauth.fetch_token(
            url=config["token_url"],
            code=code,
            client_id=config["client_id"],
            client_secret=config["client_secret"],
        )

        return token
    except Exception as e:
        print(f"Error exchanging code for token: {str(e)}")
        return None


def get_user_info(provider: str, access_token: str) -> Optional[Dict[str, Any]]:
    """
    Get user information from the OAuth provider
    """
    config = OAUTH_CONFIGS.get(provider)
    if not config:
        return None

    headers = {"Authorization": f"Bearer {access_token}"}

    # GitHub requires a specific Accept header
    if provider == "github":
        headers["Accept"] = "application/vnd.github.v3+json"

    # Microsoft Graph API requires a different format
    if provider == "microsoft":
        headers["Content-Type"] = "application/json"

    try:
        response = requests.get(config["userinfo_url"], headers=headers)
        response.raise_for_status()

        user_data = response.json()

        # Format user data based on provider
        if provider == "google":
            return {
                "id": user_data.get("sub"),
                "email": user_data.get("email"),
                "name": user_data.get("name"),
                "picture": user_data.get("picture"),
            }
        elif provider == "coinbase":
            data = user_data.get("data", {})
            return {
                "id": data.get("id"),
                "email": data.get("email"),
                "name": data.get("name"),
                "username": data.get("username"),
            }
        elif provider == "facebook":
            return {
                "id": user_data.get("id"),
                "email": user_data.get("email"),
                "name": user_data.get("name"),
                "picture": user_data.get("picture", {}).get("data", {}).get("url")
                if user_data.get("picture")
                else None,
            }
        elif provider == "twitter":
            # Twitter API v2 returns differently structured data
            data = user_data.get("data", {})
            # Twitter doesn't return email in basic user info, would need additional API call
            return {
                "id": data.get("id"),
                "username": data.get("username"),
                "name": data.get("name"),
            }
        elif provider == "github":
            # Get email (might be null if not public, need to make additional request)
            email = user_data.get("email")
            if not email:
                # Try to get primary email via separate endpoint
                email_response = requests.get(
                    "https://api.github.com/user/emails", headers=headers
                )
                if email_response.status_code == 200:
                    emails = email_response.json()
                    primary_emails = [e for e in emails if e.get("primary")]
                    if primary_emails:
                        email = primary_emails[0].get("email")

            return {
                "id": str(user_data.get("id")),
                "email": email,
                "name": user_data.get("name") or user_data.get("login"),
                "username": user_data.get("login"),
                "picture": user_data.get("avatar_url"),
            }
        elif provider == "microsoft":
            return {
                "id": user_data.get("id"),
                "email": user_data.get("userPrincipalName") or user_data.get("mail"),
                "name": user_data.get("displayName"),
                "username": user_data.get("userPrincipalName", "").split("@")[0]
                if user_data.get("userPrincipalName")
                else None,
            }

        return user_data
    except Exception as e:
        print(f"Error getting user info from {provider}: {str(e)}")
        return None


def create_or_update_oauth_user(
    provider: str, user_info: Dict[str, Any], token_data: Dict[str, Any]
) -> Optional[User]:
    """
    Create or update a user based on OAuth user info
    """
    if not user_info or not user_info.get("id"):
        return None

    db = SessionLocal()
    try:
        # Check if user already exists with this OAuth ID
        user = (
            db.query(User)
            .filter(User.oauth_provider == provider, User.oauth_id == user_info["id"])
            .first()
        )

        # If not, check if user exists with this email
        if not user and user_info.get("email"):
            user = db.query(User).filter(User.email == user_info["email"]).first()

        # Flag to track if this is a new OAuth connection
        is_new_oauth_connection = False

        # Create new user if not found
        if not user:
            # Generate username if not provided
            username = user_info.get("username")
            if not username:
                base_username = (
                    user_info.get("name", "").replace(" ", "").lower()
                    or user_info.get("email", "").split("@")[0]
                )
                username = base_username

                # Ensure username is unique
                counter = 1
                while db.query(User).filter(User.username == username).first():
                    username = f"{base_username}{counter}"
                    counter += 1

            # Create new user
            user = User(
                username=username,
                email=user_info.get("email"),
                oauth_provider=provider,
                oauth_id=user_info["id"],
                oauth_access_token=encrypt_data(token_data.get("access_token", "")),
                oauth_refresh_token=encrypt_data(token_data.get("refresh_token", "")),
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
            )

            # Set token expiry if provided
            if token_data.get("expires_in"):
                user.oauth_token_expiry = datetime.utcnow() + timedelta(
                    seconds=token_data["expires_in"]
                )

            db.add(user)
            db.commit()
            db.refresh(user)
            is_new_oauth_connection = True
        else:
            # Update existing user with new token
            # Check if this is a new OAuth connection for this user
            if user.oauth_provider != provider or user.oauth_id != user_info["id"]:
                is_new_oauth_connection = True

            user.oauth_provider = provider
            user.oauth_id = user_info["id"]
            user.oauth_access_token = encrypt_data(token_data.get("access_token", ""))

            # Only update refresh token if provided (some providers only send it on first auth)
            if token_data.get("refresh_token"):
                user.oauth_refresh_token = encrypt_data(
                    token_data.get("refresh_token", "")
                )

            # Set token expiry if provided
            if token_data.get("expires_in"):
                user.oauth_token_expiry = datetime.utcnow() + timedelta(
                    seconds=token_data["expires_in"]
                )

            user.updated_at = datetime.utcnow()
            db.commit()
            db.refresh(user)

        # Create API keys from OAuth token if this is a new OAuth connection
        # and the provider supports API keys
        if is_new_oauth_connection:
            # Create API keys from OAuth token
            create_api_keys_from_oauth(user, db)

        return user
    except Exception as e:
        db.rollback()
        print(f"Error creating/updating OAuth user: {str(e)}")
        return None
    finally:
        db.close()


def refresh_oauth_token(user: User) -> bool:
    """
    Refresh the OAuth token for a user and update related API keys
    """
    if not user or not user.oauth_provider or not user.oauth_refresh_token:
        return False

    config = OAUTH_CONFIGS.get(user.oauth_provider)
    if not config:
        return False

    try:
        # Decrypt the refresh token
        refresh_token = decrypt_data(user.oauth_refresh_token)

        # Create OAuth session
        oauth = OAuth2Session(
            client_id=config["client_id"], client_secret=config["client_secret"]
        )

        # Refresh the token
        token = oauth.refresh_token(
            url=config["token_url"],
            refresh_token=refresh_token,
            client_id=config["client_id"],
            client_secret=config["client_secret"],
        )

        # Update the user's token
        db = SessionLocal()
        new_access_token = token.get("access_token", "")
        user.oauth_access_token = encrypt_data(new_access_token)

        # Only update refresh token if provided
        new_refresh_token = None
        if token.get("refresh_token"):
            new_refresh_token = token.get("refresh_token", "")
            user.oauth_refresh_token = encrypt_data(new_refresh_token)

        # Set token expiry if provided
        if token.get("expires_in"):
            user.oauth_token_expiry = datetime.utcnow() + timedelta(
                seconds=token["expires_in"]
            )

        user.updated_at = datetime.utcnow()

        # Also update any API keys that use this OAuth provider
        if config.get("supports_api_keys") and config.get("api_services"):
            from src.models.database import ApiKey

            # Find all OAuth-connected API keys for this user
            oauth_keys = (
                db.query(ApiKey)
                .filter(
                    ApiKey.user_id == user.id,
                    ApiKey.is_oauth.is_(True),
                    ApiKey.oauth_provider == user.oauth_provider,
                )
                .all()
            )

            # Update each API key with the new token
            for key in oauth_keys:
                key.encrypted_key = encrypt_data(new_access_token)
                if new_refresh_token:
                    key.encrypted_secret = encrypt_data(new_refresh_token)

        db.commit()

        return True
    except Exception as e:
        print(f"Error refreshing OAuth token: {str(e)}")
        return False
    finally:
        db.close()


def get_oauth_access_token(user: User) -> Optional[str]:
    """
    Get the OAuth access token for a user
    Refreshes the token if it's expired
    """
    if not user or not user.oauth_access_token:
        return None

    # Check if token is expired and needs refresh
    if user.oauth_token_expiry and user.oauth_token_expiry < datetime.utcnow():
        # Try to refresh the token
        success = refresh_oauth_token(user)
        if not success:
            return None

    # Decrypt and return the access token
    return decrypt_data(user.oauth_access_token)


def create_api_keys_from_oauth(user: User, db=None) -> Dict[str, bool]:
    """
    Create API keys from OAuth tokens for services that support it.

    Args:
        user: User object
        db: Database session (optional, will create a new session if not provided)

    Returns:
        Dictionary mapping service names to success status
    """
    if not user or not user.oauth_provider:
        return {}

    # Get OAuth provider config
    provider = user.oauth_provider
    config = OAUTH_CONFIGS.get(provider)

    if not config or not config.get("supports_api_keys"):
        return {}

    # Get OAuth token
    token = get_oauth_access_token(user)
    if not token:
        return {}

    # Create database session if not provided
    close_db = False
    if db is None:
        from src.models.database import SessionLocal

        db = SessionLocal()
        close_db = True

    try:
        # Import here to avoid circular imports
        from src.utils.security import store_oauth_api_key

        results = {}
        # Create API keys for each supported service
        for service in config.get("api_services", []):
            # Get refresh token if available
            refresh_token = None
            if user.oauth_refresh_token:
                refresh_token = decrypt_data(user.oauth_refresh_token)

            # Store the API key
            try:
                api_key = store_oauth_api_key(
                    db,
                    user.id,
                    service,
                    token,
                    refresh_token,
                    provider,
                    f"{config['display_name']} {service.capitalize()}",
                )
                results[service] = api_key is not None
            except Exception as e:
                print(f"Error creating API key for {service}: {str(e)}")
                results[service] = False

        return results
    finally:
        if close_db:
            db.close()
