import bcrypt
import jwt
from datetime import datetime, timedelta
import os
from sqlalchemy.orm import Session
from src.models.database import User
import streamlit as st
from dotenv import load_dotenv
import secrets

# Load environment variables
load_dotenv()

# Get JWT secret from environment or generate a secure random one
JWT_SECRET = os.getenv("JWT_SECRET", secrets.token_hex(32))

def hash_password(password: str) -> str:
    """Hash a password for storing."""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a stored password against a provided password."""
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))

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
    
    # If still not found or password doesn't match, return None
    if not user or not verify_password(password, user.password_hash):
        return None
    
    return user

def login_user(user: User):
    """Set the user in session state after successful login."""
    # Create a user dict without sensitive info
    user_data = {
        "id": user.id,
        "username": user.username,
        "email": user.email,
        "created_at": user.created_at.isoformat()
    }
    
    # Create a JWT token
    token = create_access_token({"sub": user.username})
    
    # Store in session state
    st.session_state.user = user_data
    st.session_state.token = token

def logout_user():
    """Clear user data from session state."""
    if 'user' in st.session_state:
        del st.session_state.user
    if 'token' in st.session_state:
        del st.session_state.token

def require_login():
    """Check if user is logged in, redirect to login page if not."""
    if 'user' not in st.session_state:
        st.warning("Please log in to access this page")
        
        # Create a simple login form
        with st.form("login_form"):
            username = st.text_input("Username or Email")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login")
            
            if submit:
                from src.models.database import SessionLocal
                db = SessionLocal()
                user = authenticate_user(db, username, password)
                db.close()
                
                if user:
                    login_user(user)
                    st.experimental_rerun()
                else:
                    st.error("Invalid username or password")
        
        # Show registration link
        st.markdown("Don't have an account? [Register here](#)")
        
        # Stop execution to prevent showing the rest of the page
        st.stop()
        
    return st.session_state.user