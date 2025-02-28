"""
Database initialization script.
Run this script to create the database and tables, and optionally a demo user.
"""

from src.utils.db_init import setup_database

if __name__ == "__main__":
    print("Initializing database...")
    success = setup_database()
    
    if success:
        print("✅ Database setup completed successfully!")
        print("  - Database tables created")
        print("  - Demo user created (username: demo, password: password123)")
        print("\nYou can now run the application with:")
        print("  streamlit run app.py")
    else:
        print("❌ Database setup failed. Check the logs for details.")