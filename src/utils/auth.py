import json
import os
import secrets
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

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
    """
    Display the login page with both traditional and OAuth login options
    """
    st.warning("Please log in to access this page")

    # Create tabs for different login methods
    login_tab, register_tab = st.tabs(["Login", "Register"])

    with login_tab:
        # Standard login form
        with st.form("login_form"):
            st.subheader("Login with Username/Email")
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

        # Import OAuth config here to avoid circular imports
        from src.utils.oauth_config import OAUTH_CONFIGS, generate_oauth_authorize_url

        # OAuth login options
        st.subheader("Or login with")

        # Get available OAuth providers
        available_providers = []
        for provider, config in OAUTH_CONFIGS.items():
            if config["client_id"] and config["client_secret"]:
                available_providers.append(provider)

        if not available_providers:
            st.info(
                "No OAuth providers are configured. Please set up OAuth credentials in your environment variables."
            )
        else:
            # Display OAuth buttons in rows of 3
            providers_per_row = 3
            rows = (
                len(available_providers) + providers_per_row - 1
            ) // providers_per_row

            for row in range(rows):
                cols = st.columns(providers_per_row)

                for i in range(providers_per_row):
                    idx = row * providers_per_row + i
                    if idx < len(available_providers):
                        provider = available_providers[idx]
                        config = OAUTH_CONFIGS[provider]

                        with cols[i]:
                            # Create a styled button with the provider's color and name
                            button_html = f"""
                            <style>
                            .oauth-button-{provider} {{
                                background-color: {config["color"]};
                                color: white;
                                padding: 8px 12px;
                                border: none;
                                border-radius: 4px;
                                cursor: pointer;
                                width: 100%;
                                font-weight: bold;
                                margin-bottom: 10px;
                            }}
                            .oauth-button-{provider}:hover {{
                                opacity: 0.9;
                            }}
                            </style>
                            <button class="oauth-button-{provider}" id="oauth-{provider}">{config["display_name"]}</button>
                            <script>
                                document.getElementById("oauth-{provider}").addEventListener("click", function() {{
                                    window.location.href = "/?oauth_provider={provider}";
                                }});
                            </script>
                            """
                            st.markdown(button_html, unsafe_allow_html=True)

                            # Handle the button click via a query parameter check
                            query_params = st.query_params
                            if query_params.get("oauth_provider") == [provider]:
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

            # Display Streamlit native buttons as a fallback if the HTML buttons don't work
            # (For example, in some Streamlit Cloud environments)
            st.write("---")
            st.write("Alternatively, use these buttons:")

            for provider in available_providers:
                config = OAUTH_CONFIGS[provider]
                if st.button(config["display_name"], key=f"oauth_{provider}_button"):
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

    with register_tab:
        with st.form("register_form"):
            st.subheader("Create an Account")
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
    """
    Handle OAuth callback and login the user
    This should be called at the beginning of the app.py file
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
