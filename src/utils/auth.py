"""Authentication utilities for the Monies application."""

import os
import secrets
from datetime import datetime, timedelta
from typing import Optional

import bcrypt
import jwt
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy.orm import Session

from src.models.database import SessionLocal, User

# Load environment variables
load_dotenv()

# Get JWT secret from environment or generate a secure random one
JWT_SECRET = os.getenv("JWT_SECRET", secrets.token_hex(32))


def hash_password(password: str) -> str:
    """Hash a password for storing."""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a stored password against a provided password."""
    return bcrypt.checkpw(
        plain_password.encode("utf-8"), hashed_password.encode("utf-8")
    )


def create_access_token(data: dict, expires_delta: timedelta = timedelta(days=1)):
    """Create a JWT token."""
    to_encode = data.copy()
    expire = datetime.utcnow() + expires_delta
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET, algorithm="HS256")
    return encoded_jwt


def verify_token(token: str) -> dict:
    """Verify a JWT token and return the payload."""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        return payload
    except jwt.PyJWTError:
        return None


def authenticate_user(db: Session, username: str, password: str):
    """Authenticate a user using username/email and password."""
    # Try to find user by username
    user = db.query(User).filter(User.username == username).first()

    # If not found, try using email
    if not user:
        user = db.query(User).filter(User.email == username).first()

    # If not found or user is an OAuth user without password, return None
    if not user or user.password_hash is None:
        return None

    # Verify password
    if not verify_password(password, user.password_hash):
        return None

    return user


def login_user(user: User):
    """Set the user in session state after successful login."""
    # Create a user dict without sensitive info
    user_data = {
        "id": user.id,
        "username": user.username,
        "email": user.email,
        "oauth_provider": user.oauth_provider,
        "created_at": user.created_at.isoformat(),
    }

    # Create a JWT token
    token = create_access_token({"sub": user.username})

    # Store in session state
    st.session_state.user = user_data
    st.session_state.token = token


def logout_user():
    """Clear user data from session state."""
    # Also clear OAuth state if present
    keys_to_clear = ["user", "token", "oauth_state", "oauth_flow"]

    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]


def require_login():
    """Check if user is logged in, redirect to login page if not."""
    if "user" not in st.session_state:
        show_login_page()
        # Stop execution to prevent showing the rest of the page
        st.stop()

    return st.session_state.user


def show_login_page():
    """Display the login page with simple OAuth login and traditional options."""
    st.warning("Please log in to access this page")

    # Import OAuth config here to avoid circular imports
    from src.utils.oauth_config import OAUTH_CONFIGS, generate_oauth_authorize_url

    # Get available OAuth providers
    available_providers = []
    for provider, config in OAUTH_CONFIGS.items():
        if config["client_id"] and config["client_secret"]:
            available_providers.append(provider)

    # Helper function to handle OAuth button clicks
    def handle_oauth_button_click(provider, config):
        # Generate OAuth URL and store state in session
        auth_url, state = generate_oauth_authorize_url(provider)
        if auth_url:
            # Store state in session
            st.session_state.oauth_state = state
            st.session_state.oauth_flow = provider

            # Redirect to OAuth provider
            st.markdown(
                f'<meta http-equiv="refresh" content="0;url={auth_url}">',
                unsafe_allow_html=True,
            )
            st.stop()
        else:
            st.error(
                f"Failed to generate {config['display_name']} OAuth URL. Please check your configuration."
            )

    # Social media login section with logo buttons
    if available_providers:
        st.write("### Sign in with")

        # Define logos and styles for known providers
        logo_styles = {
            "google": """
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="24" height="24">
                    <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
                    <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
                    <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/>
                    <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/>
                </svg>
            """,
            "facebook": """
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="24" height="24">
                    <path fill="#1877F2" d="M24 12.073c0-6.627-5.373-12-12-12s-12 5.373-12 12c0 5.99 4.388 10.954 10.125 11.854v-8.385H7.078v-3.47h3.047V9.43c0-3.007 1.792-4.669 4.533-4.669 1.312 0 2.686.235 2.686.235v2.953H15.83c-1.491 0-1.956.925-1.956 1.874v2.25h3.328l-.532 3.47h-2.796v8.385C19.612 23.027 24 18.062 24 12.073z"/>
                </svg>
            """,
            "twitter": """
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="24" height="24">
                    <path fill="#1DA1F2" d="M23.953 4.57a10 10 0 01-2.825.775 4.958 4.958 0 002.163-2.723c-.951.555-2.005.959-3.127 1.184a4.92 4.92 0 00-8.384 4.482C7.69 8.095 4.067 6.13 1.64 3.162a4.822 4.822 0 00-.666 2.475c0 1.71.87 3.213 2.188 4.096a4.904 4.904 0 01-2.228-.616v.06a4.923 4.923 0 003.946 4.827 4.996 4.996 0 01-2.212.085 4.936 4.936 0 004.604 3.417 9.867 9.867 0 01-6.102 2.105c-.39 0-.779-.023-1.17-.067a13.995 13.995 0 007.557 2.209c9.053 0 13.998-7.496 13.998-13.985 0-.21 0-.42-.015-.63A9.935 9.935 0 0024 4.59z"/>
                </svg>
            """,
            "github": """
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="24" height="24">
                    <path fill="#24292E" d="M12 .297c-6.63 0-12 5.373-12 12 0 5.303 3.438 9.8 8.205 11.385.6.113.82-.258.82-.577 0-.285-.01-1.04-.015-2.04-3.338.724-4.042-1.61-4.042-1.61C4.422 18.07 3.633 17.7 3.633 17.7c-1.087-.744.084-.729.084-.729 1.205.084 1.838 1.236 1.838 1.236 1.07 1.835 2.809 1.305 3.495.998.108-.776.417-1.305.76-1.605-2.665-.3-5.466-1.332-5.466-5.93 0-1.31.465-2.38 1.235-3.22-.135-.303-.54-1.523.105-3.176 0 0 1.005-.322 3.3 1.23.96-.267 1.98-.399 3-.405 1.02.006 2.04.138 3 .405 2.28-1.552 3.285-1.23 3.285-1.23.645 1.653.24 2.873.12 3.176.765.84 1.23 1.91 1.23 3.22 0 4.61-2.805 5.625-5.475 5.92.42.36.81 1.096.81 2.22 0 1.606-.015 2.896-.015 3.286 0 .315.21.69.825.57C20.565 22.092 24 17.592 24 12.297c0-6.627-5.373-12-12-12"/>
                </svg>
            """,
            "microsoft": """
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 23 23" width="24" height="24">
                    <rect fill="#F25022" x="1" y="1" width="10" height="10" />
                    <rect fill="#00A4EF" x="1" y="12" width="10" height="10" />
                    <rect fill="#7FBA00" x="12" y="1" width="10" height="10" />
                    <rect fill="#FFB900" x="12" y="12" width="10" height="10" />
                </svg>
            """,
            "coinbase": """
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="24" height="24">
                    <path fill="#0052FF" d="M12 0C5.373 0 0 5.373 0 12s5.373 12 12 12 12-5.373 12-12S18.627 0 12 0zm0 21.6a9.6 9.6 0 110-19.2 9.6 9.6 0 010 19.2zm-2.4-9.6l2.4 2.4 2.4-2.4-2.4-2.4-2.4 2.4z"/>
                </svg>
            """,
        }

        # Define CSS for OAuth buttons
        st.markdown(
            """
        <style>
        .sso-container {
            display: flex;
            flex-wrap: wrap;
            gap: 16px;
            margin: 20px 0;
            justify-content: center;
        }
        .sso-button {
            display: flex;
            align-items: center;
            justify-content: center;
            width: 60px;
            height: 60px;
            border-radius: 50%;
            border: 1px solid #e0e0e0;
            background-color: white;
            cursor: pointer;
            transition: all 0.2s ease;
            padding: 0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .sso-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        }
        </style>
        """,
            unsafe_allow_html=True,
        )

        # Create container for SSO buttons
        html_buttons = '<div class="sso-container">'

        # Generate SSO buttons for all available providers
        for provider in available_providers:
            config = OAUTH_CONFIGS[provider]
            logo = logo_styles.get(
                provider,
                f'<span style="font-weight: bold; font-size: 20px;">{config["display_name"][0]}</span>',
            )

            # Create button with logo
            html_buttons += f"""
            <button class="sso-button" id="oauth-{provider}" title="{config["display_name"]}">
                {logo}
            </button>
            <script>
                document.getElementById("oauth-{provider}").addEventListener("click", function() {{
                    window.location.href = "/?oauth_provider={provider}";
                }});
            </script>
            """

            # Handle the button click via a query parameter check
            query_params = st.query_params
            if query_params.get("oauth_provider") == [provider]:
                handle_oauth_button_click(provider, config)

        html_buttons += "</div>"
        st.markdown(html_buttons, unsafe_allow_html=True)

        # Provide explanation text
        st.markdown(
            """
        <p style="color: #666; font-size: 0.9em; text-align: center; margin-bottom: 20px;">
        Sign in with your social account. We'll create a new account if you don't have one yet.
        </p>
        """,
            unsafe_allow_html=True,
        )

        # Add a divider
        st.markdown("<hr style='margin: 20px 0;'>", unsafe_allow_html=True)

    # Standard login form (in tabs)
    st.markdown("### Email login")

    # Create tabs for login/register with email
    login_tab, register_tab = st.tabs(["Login", "Register"])

    with login_tab:
        with st.form("login_form"):
            username = st.text_input("Username or Email")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login")

            if submit:
                db = SessionLocal()
                user = authenticate_user(db, username, password)
                db.close()

                if user:
                    login_user(user)
                    st.rerun()
                else:
                    st.error("Invalid username or password")

    with register_tab:
        with st.form("register_form"):
            new_username = st.text_input("Username")
            new_email = st.text_input("Email")
            new_password = st.text_input("Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            register_submit = st.form_submit_button("Register")

            if register_submit:
                # Validate input
                if not new_username or not new_email or not new_password:
                    st.error("All fields are required")
                elif new_password != confirm_password:
                    st.error("Passwords do not match")
                else:
                    # Check if username or email already exists
                    db = SessionLocal()
                    existing_user = (
                        db.query(User)
                        .filter(
                            (User.username == new_username) | (User.email == new_email)
                        )
                        .first()
                    )

                    if existing_user:
                        if existing_user.username == new_username:
                            st.error("Username already exists")
                        else:
                            st.error("Email already exists")
                    else:
                        # Create new user
                        user = User(
                            username=new_username,
                            email=new_email,
                            password_hash=hash_password(new_password),
                            created_at=datetime.utcnow(),
                            updated_at=datetime.utcnow(),
                        )
                        db.add(user)
                        db.commit()
                        db.refresh(user)

                        # Log in the new user
                        login_user(user)
                        st.success("Account created successfully!")
                        st.rerun()

                    db.close()


def handle_oauth_callback():
    """Handle OAuth callback and login the user.

    This should be called at the beginning of the app.py file.
    """
    # Check for OAuth callback with code parameter
    query_params = st.query_params
    if "code" in query_params and "state" in query_params:
        # Get code and state from query parameters
        code = query_params["code"][0]
        state = query_params["state"][0]

        # Check if we have a stored state and it matches
        if "oauth_state" in st.session_state and st.session_state.oauth_state == state:
            provider = st.session_state.get("oauth_flow")

            if provider:
                # Import OAuth functions here to avoid circular imports
                from src.utils.oauth_config import (
                    create_or_update_oauth_user,
                    exchange_code_for_token,
                    get_user_info,
                )

                # Exchange code for token
                token_data = exchange_code_for_token(provider, code)
                if token_data and "access_token" in token_data:
                    # Get user info
                    user_info = get_user_info(provider, token_data["access_token"])
                    if user_info:
                        # Create or update user
                        user = create_or_update_oauth_user(
                            provider, user_info, token_data
                        )
                        if user:
                            # Login user
                            login_user(user)

                            # Clear OAuth flow parameters
                            if "oauth_state" in st.session_state:
                                del st.session_state.oauth_state
                            if "oauth_flow" in st.session_state:
                                del st.session_state.oauth_flow

                            # Clear URL parameters by redirecting to the main app
                            st.markdown(
                                '<meta http-equiv="refresh" content="0;url=/">',
                                unsafe_allow_html=True,
                            )
                            st.stop()
                        else:
                            st.error("Failed to create or update user")
                    else:
                        st.error("Failed to get user info")
                else:
                    st.error("Failed to exchange code for token")
            else:
                st.error("Unknown OAuth provider")
        else:
            st.error("Invalid OAuth state. Possible security attack.")

            # Clear any OAuth related session state
            if "oauth_state" in st.session_state:
                del st.session_state.oauth_state
            if "oauth_flow" in st.session_state:
                del st.session_state.oauth_flow


def get_user_by_id(user_id: int) -> Optional[User]:
    """Get a user by ID."""
    db = SessionLocal()
    try:
        return db.query(User).filter(User.id == user_id).first()
    finally:
        db.close()


def generate_jwt_token(user_id):
    """Generate a JWT token for a user (test-compatible function)."""
    return create_access_token({"user_id": user_id})


def validate_jwt_token(token):
    """Validate a JWT token and return the payload (test-compatible function)."""
    payload = verify_token(token)
    if not payload:
        raise Exception("Invalid or expired token")
    return payload
