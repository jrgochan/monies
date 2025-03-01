import os
from typing import Any, List, Optional, Tuple

from cryptography.fernet import Fernet
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get encryption key from environment or generate a new one
SECRET_KEY = os.getenv("SECRET_KEY")

# If no key is provided, generate a warning
if not SECRET_KEY or len(SECRET_KEY) < 32:
    import warnings

    warnings.warn(
        "SECRET_KEY is missing or too short. A temporary key will be used, "
        "but all encrypted data will be lost when the application restarts. "
        "Set a secure SECRET_KEY in .env file (min. 32 characters)."
    )
    SECRET_KEY = Fernet.generate_key().decode()


# Ensure the key is in bytes and padded/truncated to 32 bytes for Fernet
def get_fernet_key(key_str: Optional[str]) -> bytes:
    """Convert string key to a valid Fernet key."""
    if not key_str:
        return Fernet.generate_key()

    import base64

    # Encode and pad/truncate to 32 bytes
    key_bytes = key_str.encode("utf-8")
    key_bytes = key_bytes.ljust(32, b"0")[:32]
    # Return base64-encoded key required by Fernet
    b64_encoded = base64.urlsafe_b64encode(key_bytes)
    return b64_encoded


# Initialize Fernet cipher for encryption/decryption
cipher = Fernet(get_fernet_key(SECRET_KEY))


def generate_key() -> bytes:
    """Generate a new Fernet key (test-compatible function)."""
    return Fernet.generate_key()


def encrypt_data(data: str, key: Optional[bytes] = None) -> Optional[str]:
    """Encrypt a string using Fernet symmetric encryption."""
    if not data:
        return None

    # Use provided key or default cipher
    if key:
        custom_cipher = Fernet(key)
        return custom_cipher.encrypt(data.encode("utf-8")).decode("utf-8")
    else:
        return cipher.encrypt(data.encode("utf-8")).decode("utf-8")


def decrypt_data(encrypted_data: str, key: Optional[bytes] = None) -> Optional[str]:
    """Decrypt a Fernet-encrypted string."""
    if not encrypted_data:
        return None

    # Use provided key or default cipher
    if key:
        custom_cipher = Fernet(key)
        return custom_cipher.decrypt(encrypted_data.encode("utf-8")).decode("utf-8")
    else:
        return cipher.decrypt(encrypted_data.encode("utf-8")).decode("utf-8")


def store_api_key(
    db: Any, user_id: int, service: str, api_key: str, api_secret: Optional[str] = None
) -> Any:
    """Securely store API key and secret for a user."""
    from src.models.database import ApiKey

    # Check if key for this service already exists
    existing_key = (
        db.query(ApiKey)
        .filter(ApiKey.user_id == user_id, ApiKey.service == service)
        .first()
    )

    if existing_key:
        # Update existing key without creating a new object
        existing_key.encrypted_key = encrypt_data(api_key)
        if api_secret is not None:
            existing_key.encrypted_secret = encrypt_data(api_secret)
        db.commit()
        return existing_key
    else:
        # Create new key entry
        api_key_obj = ApiKey(
            user_id=user_id,
            service=service,
            encrypted_key=encrypt_data(api_key),
            encrypted_secret=encrypt_data(api_secret) if api_secret else None,
        )
        db.add(api_key_obj)
        db.commit()
        db.refresh(api_key_obj)
        return api_key_obj


def get_api_key(
    db: Any, user_id: int, service: str, key_id: Optional[int] = None
) -> Tuple[Optional[str], Optional[str], Any]:
    """Retrieve and decrypt API key and secret for a user.

    Args:
        db: Database session
        user_id: User ID
        service: Service name
        key_id: Specific API key ID to retrieve (optional)

    Returns:
        Tuple of (decrypted_key, decrypted_secret, api_key_obj)
    """
    from src.models.database import ApiKey

    query = db.query(ApiKey).filter(
        ApiKey.user_id == user_id, ApiKey.service == service
    )

    if key_id:
        # Get specific key by ID
        api_key = query.filter(ApiKey.id == key_id).first()
    else:
        # Get default key or first available
        default_key = query.filter(ApiKey.is_default.is_(True)).first()
        if default_key:
            api_key = default_key
        else:
            api_key = query.first()

    if not api_key:
        return None, None, None

    decrypted_key = decrypt_data(api_key.encrypted_key)
    decrypted_secret = (
        decrypt_data(api_key.encrypted_secret) if api_key.encrypted_secret else None
    )

    return decrypted_key, decrypted_secret, api_key


def get_api_keys_for_service(db: Any, user_id: int, service: str) -> List[Any]:
    """Get all API keys for a specific service.

    Args:
        db: Database session
        user_id: User ID
        service: Service name

    Returns:
        List of ApiKey objects
    """
    from src.models.database import ApiKey

    keys = (
        db.query(ApiKey)
        .filter(ApiKey.user_id == user_id, ApiKey.service == service)
        .all()
    )

    return keys


def set_default_api_key(db: Any, user_id: int, service: str, key_id: int) -> bool:
    """Set a specific API key as the default for a service.

    Args:
        db: Database session
        user_id: User ID
        service: Service name
        key_id: API key ID to set as default

    Returns:
        Boolean indicating success
    """
    from src.models.database import ApiKey

    try:
        # Clear default status from all keys for this service
        db.query(ApiKey).filter(
            ApiKey.user_id == user_id, ApiKey.service == service
        ).update({"is_default": False})

        # Set the specified key as default
        key = (
            db.query(ApiKey)
            .filter(ApiKey.id == key_id, ApiKey.user_id == user_id)  # Security check
            .first()
        )

        if not key:
            return False

        key.is_default = True
        db.commit()
        return True
    except Exception as e:
        db.rollback()
        print(f"Error setting default API key: {str(e)}")
        return False


def store_oauth_api_key(
    db: Any,
    user_id: int,
    service: str,
    api_key: str,
    api_secret: Optional[str] = None,
    oauth_provider: Optional[str] = None,
    display_name: Optional[str] = None,
) -> Any:
    """Store an API key obtained via OAuth.

    Args:
        db: Database session
        user_id: User ID
        service: Service name (e.g., 'coinbase', 'twitter')
        api_key: API key or access token
        api_secret: API secret or refresh token (optional)
        oauth_provider: Name of the OAuth provider (e.g., 'google', 'coinbase')
        display_name: Display name for this key

    Returns:
        ApiKey object
    """
    from src.models.database import ApiKey

    # Check if this OAuth key already exists
    existing_key = (
        db.query(ApiKey)
        .filter(
            ApiKey.user_id == user_id,
            ApiKey.service == service,
            ApiKey.is_oauth.is_(True),
            ApiKey.oauth_provider == oauth_provider,
        )
        .first()
    )

    if existing_key:
        # Update existing key
        existing_key.encrypted_key = encrypt_data(api_key)
        if api_secret is not None:
            existing_key.encrypted_secret = encrypt_data(api_secret)
        if display_name:
            existing_key.display_name = display_name
        db.commit()
        return existing_key
    else:
        # Create new key entry
        # If this is the first key for this service, make it the default
        existing_keys = get_api_keys_for_service(db, user_id, service)
        is_default = len(existing_keys) == 0

        api_key_obj = ApiKey(
            user_id=user_id,
            service=service,
            encrypted_key=encrypt_data(api_key),
            encrypted_secret=encrypt_data(api_secret) if api_secret else None,
            is_oauth=True,
            oauth_provider=oauth_provider,
            is_default=is_default,
            display_name=display_name or f"{oauth_provider.capitalize()} {service}",
        )
        db.add(api_key_obj)
        db.commit()
        db.refresh(api_key_obj)
        return api_key_obj
