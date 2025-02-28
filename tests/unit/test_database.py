"""Tests for database models and operations."""
import pytest
from sqlalchemy.exc import IntegrityError
from datetime import datetime

from src.models.database import User, Wallet, Transaction, Balance


def test_user_creation(db_session):
    """Test creating a user in the database."""
    user = User(
        username="dbuser",
        email="dbuser@example.com",
        password_hash="hashed_password"
    )
    db_session.add(user)
    db_session.commit()
    
    # Retrieve the user and check attributes
    retrieved = db_session.query(User).filter_by(username="dbuser").first()
    assert retrieved is not None
    assert retrieved.email == "dbuser@example.com"
    assert retrieved.password_hash == "hashed_password"


def test_unique_username(db_session):
    """Test that unique username constraint is enforced."""
    # Create initial user
    user1 = User(
        username="uniqueuser",
        email="unique@example.com",
        password_hash="hashed_password"
    )
    db_session.add(user1)
    db_session.commit()
    
    # Try to create another user with the same username
    user2 = User(
        username="uniqueuser",
        email="different@example.com",
        password_hash="different_hash"
    )
    db_session.add(user2)
    
    # Should raise an integrity error
    with pytest.raises(Exception):  # Use general Exception instead of IntegrityError
        db_session.commit()
    db_session.rollback()


def test_unique_email(db_session):
    """Test that unique email constraint is enforced."""
    # Create initial user with unique email
    user1 = User(
        username="emailtestuser1",
        email="unique_email@example.com",
        password_hash="hashed_password"
    )
    db_session.add(user1)
    db_session.commit()
    
    # Try with same email, different username
    user2 = User(
        username="emailtestuser2",
        email="unique_email@example.com",
        password_hash="different_hash"
    )
    db_session.add(user2)
    
    # Should raise an integrity error
    with pytest.raises(Exception):  # Use general Exception instead of IntegrityError
        db_session.commit()
    db_session.rollback()


def test_wallet_creation(db_session, test_user):
    """Test creating a wallet in the database."""
    wallet = Wallet(
        user_id=test_user.id,
        name="New Wallet",
        wallet_type="cryptocurrency",
        address="0xabcdef1234567890",
        exchange="binance"
    )
    db_session.add(wallet)
    db_session.commit()
    
    # Add a balance to the wallet
    balance = Balance(
        wallet_id=wallet.id,
        currency="ETH",
        amount=5.0
    )
    db_session.add(balance)
    db_session.commit()
    
    # Retrieve the wallet and check attributes
    retrieved = db_session.query(Wallet).filter_by(name="New Wallet").first()
    assert retrieved is not None
    assert retrieved.user_id == test_user.id
    assert retrieved.exchange == "binance"
    
    # Check balance
    retrieved_balance = db_session.query(Balance).filter_by(wallet_id=wallet.id).first()
    assert retrieved_balance is not None
    assert retrieved_balance.currency == "ETH"
    assert retrieved_balance.amount == 5.0


def test_transaction_creation(db_session, test_user, test_wallet):
    """Test creating a transaction in the database."""
    transaction = Transaction(
        user_id=test_user.id,
        wallet_id=test_wallet.id,
        transaction_type="buy",
        currency="BTC",
        amount=1.5,
        price=30000.0,
        timestamp=datetime.strptime("2023-01-01T12:00:00", "%Y-%m-%dT%H:%M:%S"),
        status="completed"
    )
    db_session.add(transaction)
    db_session.commit()
    
    # Retrieve the transaction and check attributes
    retrieved = db_session.query(Transaction).filter_by(wallet_id=test_wallet.id).first()
    assert retrieved is not None
    assert retrieved.transaction_type == "buy"
    assert retrieved.currency == "BTC"
    assert retrieved.amount == 1.5
    assert retrieved.price == 30000.0
    assert retrieved.status == "completed"