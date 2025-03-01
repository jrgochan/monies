import os
import logging
import json
from src.models.database import init_db, SessionLocal, User, DataSource
from src.utils.auth import hash_password
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


def initialize_database():
    """Initialize the database and create tables"""
    try:
        # Create database tables
        init_db()
        logger.info("Database initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        return False


def create_demo_user():
    """Create a demo user for testing"""
    db = SessionLocal()
    try:
        # Check if demo user already exists
        existing_user = db.query(User).filter(User.username == "demo").first()

        if existing_user:
            logger.info("Demo user already exists")
            return existing_user

        # Create demo user
        demo_user = User(
            username="demo",
            email="demo@example.com",
            password_hash=hash_password("password123"),
        )

        db.add(demo_user)
        db.commit()
        db.refresh(demo_user)

        logger.info(f"Demo user created with id: {demo_user.id}")
        return demo_user

    except Exception as e:
        db.rollback()
        logger.error(f"Error creating demo user: {str(e)}")
        return None

    finally:
        db.close()


def create_default_data_sources():
    """Create the default data sources"""
    db = SessionLocal()
    try:
        # Check if data sources already exist
        existing_sources = db.query(DataSource).count()
        if existing_sources > 0:
            logger.info("Data sources already exist")
            return True

        # Define default data sources
        default_sources = [
            # Stock data sources
            {
                "name": "financial_modeling_prep",
                "display_name": "Financial Modeling Prep",
                "category": "stocks",
                "api_required": True,
                "api_key_field": "FMP_API_KEY",
                "priority": 10,
                "enabled": True,
            },
            {
                "name": "alpha_vantage",
                "display_name": "Alpha Vantage",
                "category": "stocks",
                "api_required": True,
                "api_key_field": "ALPHA_VANTAGE_KEY",
                "priority": 20,
                "enabled": True,
            },
            {
                "name": "yahoo_finance",
                "display_name": "Yahoo Finance",
                "category": "stocks",
                "api_required": False,
                "api_key_field": None,
                "priority": 30,
                "enabled": True,
            },
            {
                "name": "polygon",
                "display_name": "Polygon.io",
                "category": "stocks",
                "api_required": True,
                "api_key_field": "POLYGON_API_KEY",
                "priority": 40,
                "enabled": True,
            },
            # Cryptocurrency data sources
            {
                "name": "coingecko",
                "display_name": "CoinGecko",
                "category": "crypto",
                "api_required": False,
                "api_key_field": None,
                "priority": 10,
                "enabled": True,
            },
            {
                "name": "coinmarketcap",
                "display_name": "CoinMarketCap",
                "category": "crypto",
                "api_required": True,
                "api_key_field": "COINMARKETCAP_API_KEY",
                "priority": 20,
                "enabled": True,
            },
            {
                "name": "binance",
                "display_name": "Binance",
                "category": "crypto",
                "api_required": True,
                "api_key_field": "BINANCE_API_KEY",
                "priority": 30,
                "enabled": True,
            },
            {
                "name": "coinbase",
                "display_name": "Coinbase",
                "category": "crypto",
                "api_required": True,
                "api_key_field": "COINBASE_API_KEY",
                "priority": 40,
                "enabled": True,
            },
            # ETF/Mutual Fund data sources
            {
                "name": "yahoo_finance_etf",
                "display_name": "Yahoo Finance ETF",
                "category": "etf",
                "api_required": False,
                "api_key_field": None,
                "priority": 10,
                "enabled": True,
            },
            {
                "name": "financial_modeling_prep_etf",
                "display_name": "Financial Modeling Prep ETF",
                "category": "etf",
                "api_required": True,
                "api_key_field": "FMP_API_KEY",
                "priority": 20,
                "enabled": True,
            },
        ]

        # Add data sources to database
        for source_data in default_sources:
            source = DataSource(**source_data)
            db.add(source)

        db.commit()
        logger.info("Default data sources created successfully")
        return True

    except Exception as e:
        db.rollback()
        logger.error(f"Error creating default data sources: {str(e)}")
        return False

    finally:
        db.close()


def setup_database():
    """Initialize database and create demo data"""
    if initialize_database():
        # Create demo user and default data
        demo_user = create_demo_user()
        data_sources_created = create_default_data_sources()

        return demo_user is not None and data_sources_created
    return False


if __name__ == "__main__":
    # Can be run as a standalone script to initialize the database
    success = setup_database()
    if success:
        print("Database setup completed successfully")
    else:
        print("Database setup failed")
