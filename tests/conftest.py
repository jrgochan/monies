"""Test configuration and fixtures for the Monies application."""
import os
import sys
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

# Add the project root to the path so we can import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.database import Base, User, Wallet, Transaction, Balance
from src.utils.auth import hash_password, generate_jwt_token
from src.utils.security import generate_key


@pytest.fixture(scope="session")
def test_db_url():
    """Create a test database URL."""
    return "sqlite:///:memory:"


@pytest.fixture(scope="session")
def engine(test_db_url):
    """Create a SQLAlchemy engine for tests."""
    return create_engine(test_db_url)


@pytest.fixture(scope="session")
def tables(engine):
    """Create all tables in the test database."""
    Base.metadata.create_all(engine)
    yield
    Base.metadata.drop_all(engine)


@pytest.fixture(scope="function")
def db_session(engine, tables):
    """Create a new database session for a test."""
    connection = engine.connect()
    transaction = connection.begin()
    session = sessionmaker(bind=connection)()
    
    yield session
    
    session.close()
    # Safely handle transaction rollback
    try:
        transaction.rollback()
    except Exception:
        # Transaction might have been deassociated or closed already
        pass
    connection.close()


@pytest.fixture
def test_user(db_session):
    """Create a test user."""
    user = User(
        username="testuser",
        email="test@example.com",
        password_hash=hash_password("testpassword")
    )
    db_session.add(user)
    db_session.commit()
    return user


@pytest.fixture
def test_wallet(db_session, test_user):
    """Create a test wallet for the test user."""
    wallet = Wallet(
        user_id=test_user.id,
        name="Test Wallet",
        wallet_type="cryptocurrency",
        address="0x1234567890abcdef",
        exchange="test_exchange"
    )
    db_session.add(wallet)
    db_session.commit()
    
    # Add a balance for this wallet
    balance = Balance(
        wallet_id=wallet.id,
        currency="BTC",
        amount=1.5
    )
    db_session.add(balance)
    db_session.commit()
    
    return wallet


@pytest.fixture
def auth_token(test_user):
    """Generate a JWT token for the test user."""
    return generate_jwt_token(test_user.id)


@pytest.fixture
def encryption_key():
    """Generate an encryption key for tests."""
    return generate_key()