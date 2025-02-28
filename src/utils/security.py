import os
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
def get_fernet_key(key_str):
    """Convert string key to a valid Fernet key."""
    # Encode and pad/truncate to 32 bytes
    key_bytes = key_str.encode('utf-8')
    key_bytes = key_bytes.ljust(32, b'0')[:32]
    # Return base64-encoded key required by Fernet
    return Fernet.generate_key() if not key_str else Fernet(key_bytes)

# Initialize Fernet cipher for encryption/decryption
cipher = Fernet(get_fernet_key(SECRET_KEY))

def encrypt_data(data: str) -> str:
    """Encrypt a string using Fernet symmetric encryption."""
    if not data:
        return None
    return cipher.encrypt(data.encode('utf-8')).decode('utf-8')

def decrypt_data(encrypted_data: str) -> str:
    """Decrypt a Fernet-encrypted string."""
    if not encrypted_data:
        return None
    return cipher.decrypt(encrypted_data.encode('utf-8')).decode('utf-8')

def store_api_key(db, user_id, service, api_key, api_secret=None):
    """Securely store API key and secret for a user."""
    from src.models.database import ApiKey
    
    # Check if key for this service already exists
    existing_key = db.query(ApiKey).filter(
        ApiKey.user_id == user_id,
        ApiKey.service == service
    ).first()
    
    if existing_key:
        # Update existing key
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
            encrypted_secret=encrypt_data(api_secret) if api_secret else None
        )
        db.add(api_key_obj)
        db.commit()
        db.refresh(api_key_obj)
        return api_key_obj

def get_api_key(db, user_id, service):
    """Retrieve and decrypt API key and secret for a user."""
    from src.models.database import ApiKey
    
    api_key = db.query(ApiKey).filter(
        ApiKey.user_id == user_id,
        ApiKey.service == service
    ).first()
    
    if not api_key:
        return None, None
    
    decrypted_key = decrypt_data(api_key.encrypted_key)
    decrypted_secret = decrypt_data(api_key.encrypted_secret) if api_key.encrypted_secret else None
    
    return decrypted_key, decrypted_secret