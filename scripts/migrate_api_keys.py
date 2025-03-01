# \!/usr/bin/env python
"""
Migration script to add new columns to ApiKey table for OAuth integration
"""
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from sqlalchemy import Boolean, Column, String, create_engine, text
from sqlalchemy.orm import sessionmaker

from src.models.database import ApiKey, Base, SessionLocal


def run_migration():
    """Run the migration to add new columns to ApiKey table"""
    print("Starting API key table migration...")

    # Create a session
    db = SessionLocal()
    engine = db.get_bind()

    try:
        # Check if columns already exist
        columns_to_add = [
            ("is_oauth", "BOOLEAN", "False"),
            ("oauth_provider", "VARCHAR(20)", "NULL"),
            ("is_default", "BOOLEAN", "False"),
            ("display_name", "VARCHAR(100)", "NULL"),
        ]

        # Get existing columns
        if engine.dialect.name == "sqlite":
            result = db.execute(text("PRAGMA table_info(api_keys)")).fetchall()
            existing_columns = [row[1] for row in result]
        else:
            # For PostgreSQL
            query = text(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = 'api_keys'
            """
            )
            result = db.execute(query).fetchall()
            existing_columns = [row[0] for row in result]

        # Add missing columns
        for column_name, column_type, default in columns_to_add:
            if column_name.lower() not in [col.lower() for col in existing_columns]:
                print(f"Adding column {column_name} to api_keys table...")

                if engine.dialect.name == "sqlite":
                    # SQLite syntax
                    db.execute(
                        text(
                            f"ALTER TABLE api_keys ADD COLUMN {column_name} {column_type} DEFAULT {default}"
                        )
                    )
                else:
                    # PostgreSQL syntax
                    db.execute(
                        text(
                            f"ALTER TABLE api_keys ADD COLUMN {column_name} {column_type} DEFAULT {default}"
                        )
                    )
            else:
                print(f"Column {column_name} already exists in api_keys table.")

        # Set all existing keys as default for their service
        print("Setting existing keys as default...")
        db.execute(
            text(
                """
            WITH RankedKeys AS (
                SELECT
                    id,
                    user_id,
                    service,
                    ROW_NUMBER() OVER (PARTITION BY user_id, service ORDER BY created_at) as rn
                FROM api_keys
                WHERE is_default = 0 OR is_default IS NULL
            )
            UPDATE api_keys
            SET is_default = 1
            WHERE id IN (SELECT id FROM RankedKeys WHERE rn = 1)
        """
            )
        )

        db.commit()
        print("Migration completed successfully.")
    except Exception as e:
        db.rollback()
        print(f"Error during migration: {str(e)}")
    finally:
        db.close()


if __name__ == "__main__":
    run_migration()
