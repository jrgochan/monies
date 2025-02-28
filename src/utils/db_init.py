import os
import logging
from src.models.database import init_db, SessionLocal, User
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
            password_hash=hash_password("password123")
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

def setup_database():
    """Initialize database and create demo data"""
    if initialize_database():
        demo_user = create_demo_user()
        return demo_user is not None
    return False

if __name__ == "__main__":
    # Can be run as a standalone script to initialize the database
    success = setup_database()
    if success:
        print("Database setup completed successfully")
    else:
        print("Database setup failed")