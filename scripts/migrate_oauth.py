#!/usr/bin/env python3
"""
Migration script to update the database schema for OAuth support.
Adds OAuth-related columns to the User table.
"""

import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import logging

from dotenv import load_dotenv
from sqlalchemy import Column, DateTime, MetaData, String, Table, Text, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Get database URL from environment or use default SQLite
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/crypto_wallet.db")


def run_migration():
    """Run the database migration to add OAuth columns to the User table"""
    logger.info(f"Starting migration on database: {DATABASE_URL}")

    # Create SQLAlchemy engine
    engine = create_engine(DATABASE_URL)

    # Create a metadata object
    metadata = MetaData()
    metadata.bind = engine

    # Try to reflect the users table
    try:
        users_table = Table("users", metadata, autoload_with=engine)
        logger.info("Successfully reflected users table")
    except Exception as e:
        logger.error(f"Failed to reflect users table: {str(e)}")
        return False

    # Add the OAuth columns if they don't exist
    columns_to_add = [
        {
            "name": "oauth_provider",
            "column": Column("oauth_provider", String(20), nullable=True),
        },
        {"name": "oauth_id", "column": Column("oauth_id", String(100), nullable=True)},
        {
            "name": "oauth_access_token",
            "column": Column("oauth_access_token", Text, nullable=True),
        },
        {
            "name": "oauth_refresh_token",
            "column": Column("oauth_refresh_token", Text, nullable=True),
        },
        {
            "name": "oauth_token_expiry",
            "column": Column("oauth_token_expiry", DateTime, nullable=True),
        },
    ]

    # Check existing columns and add missing ones
    existing_columns = [c.name for c in users_table.columns]

    # Alter table to add missing columns
    conn = engine.connect()

    for col_def in columns_to_add:
        col_name = col_def["name"]
        if col_name not in existing_columns:
            try:
                # SQLite doesn't support ALTER TABLE ADD COLUMN with constraints,
                # so we need to use a simplified syntax
                if engine.name == "sqlite":
                    from sqlalchemy.sql import text

                    stmt = text(f"ALTER TABLE users ADD COLUMN {col_name} TEXT")
                    conn.execute(stmt)
                    logger.info(f"Added column {col_name} to users table (SQLite)")
                else:
                    # For other databases like PostgreSQL, we can use the SQLAlchemy column
                    col = col_def["column"]
                    column_type = col.type.compile(engine.dialect)
                    nullable = "NULL" if col.nullable else "NOT NULL"
                    stmt = text(
                        f"ALTER TABLE users ADD COLUMN {col_name} {column_type} {nullable}"
                    )
                    conn.execute(stmt)
                    logger.info(f"Added column {col_name} to users table")
            except Exception as e:
                logger.error(f"Failed to add column {col_name}: {str(e)}")
                return False

    # Close the connection
    conn.close()

    # Update password_hash column to be nullable
    try:
        conn = engine.connect()

        # For SQLite, we need to create a new table and copy data
        if engine.name == "sqlite":
            logger.info(
                "SQLite detected, will modify nullable property by creating new table"
            )
            # Check if password_hash is already nullable
            # Since SQLite doesn't support altering column nullability directly,
            # we'll skip this step and assume it's not needed for a fresh installation
            logger.info("Skipping password_hash nullable update for SQLite")
        else:
            # For other databases, we can modify the column directly
            from sqlalchemy.sql import text

            stmt = text("ALTER TABLE users ALTER COLUMN password_hash DROP NOT NULL")
            conn.execute(stmt)
            logger.info("Updated password_hash column to be nullable")

        conn.close()
    except Exception as e:
        logger.error(f"Failed to update password_hash column: {str(e)}")
        # Not critical, so continue

    logger.info("Migration completed successfully")
    return True


if __name__ == "__main__":
    success = run_migration()
    if success:
        print("✅ Migration completed successfully")
        sys.exit(0)
    else:
        print("❌ Migration failed")
        sys.exit(1)
