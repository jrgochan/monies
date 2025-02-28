from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get database URL from environment or use default SQLite
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/crypto_wallet.db")

# Create SQLAlchemy engine and session
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Define database models
class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True)
    email = Column(String(100), unique=True, index=True)
    password_hash = Column(String(255))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    wallets = relationship("Wallet", back_populates="user")
    api_keys = relationship("ApiKey", back_populates="user")
    social_accounts = relationship("SocialAccount", back_populates="user")
    scheduled_posts = relationship("ScheduledPost", back_populates="user")
    transactions = relationship("Transaction", back_populates="user")

class Wallet(Base):
    __tablename__ = "wallets"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    name = Column(String(50))
    wallet_type = Column(String(20))  # 'exchange', 'on-chain'
    exchange = Column(String(50), nullable=True)  # if wallet_type is 'exchange'
    address = Column(String(255), nullable=True)  # if wallet_type is 'on-chain'
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="wallets")
    balances = relationship("Balance", back_populates="wallet")

class Balance(Base):
    __tablename__ = "balances"
    
    id = Column(Integer, primary_key=True, index=True)
    wallet_id = Column(Integer, ForeignKey("wallets.id"))
    currency = Column(String(10))
    amount = Column(Float)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    wallet = relationship("Wallet", back_populates="balances")

class Transaction(Base):
    __tablename__ = "transactions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    wallet_id = Column(Integer, ForeignKey("wallets.id"))
    transaction_type = Column(String(20))  # 'buy', 'sell', 'send', 'receive'
    currency = Column(String(10))
    amount = Column(Float)
    price = Column(Float, nullable=True)  # price per unit if applicable
    timestamp = Column(DateTime, default=datetime.utcnow)
    status = Column(String(20))  # 'pending', 'completed', 'failed'
    notes = Column(Text, nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="transactions")

class ApiKey(Base):
    __tablename__ = "api_keys"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    service = Column(String(50))  # 'binance', 'coinbase', 'openai'
    encrypted_key = Column(Text)
    encrypted_secret = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="api_keys")

class SocialAccount(Base):
    __tablename__ = "social_accounts"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    platform = Column(String(20))  # 'twitter', 'facebook', 'instagram', 'linkedin'
    username = Column(String(50))
    encrypted_token = Column(Text)
    encrypted_token_secret = Column(Text, nullable=True)  # For Twitter OAuth1
    connected_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="social_accounts")

class ScheduledPost(Base):
    __tablename__ = "scheduled_posts"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    platform = Column(String(20))  # 'twitter', 'facebook', 'instagram', 'linkedin'
    content = Column(Text)
    media_path = Column(String(255), nullable=True)
    scheduled_time = Column(DateTime)
    posted = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="scheduled_posts")

class AiAnalysis(Base):
    __tablename__ = "ai_analyses"
    
    id = Column(Integer, primary_key=True, index=True)
    query = Column(Text)
    result = Column(Text)
    model_used = Column(String(50))  # 'gpt-4', 'ollama-llama2', etc.
    timestamp = Column(DateTime, default=datetime.utcnow)
    cached_until = Column(DateTime, nullable=True)

# Create all tables
def init_db():
    Base.metadata.create_all(bind=engine)

# Get a database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()