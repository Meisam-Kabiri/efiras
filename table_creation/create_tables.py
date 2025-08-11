#!/usr/bin/env python3
"""
Run this script to create database tables
Usage: python create_tables.py
"""

import sys
import os
from pathlib import Path

# Add the current directory to Python path so we can import our modules
sys.path.append(str(Path(__file__).parent))

from db_models.database import create_tables, engine
from dotenv import load_dotenv

def main():
    # Load environment variables
    load_dotenv()
    
    print("🗄️  Creating database tables...")
    
    try:
        # Test connection
        with engine.connect() as conn:
            print("✅ Database connection successful!")
        
        # Create all tables
        create_tables()
        print("✅ All tables created successfully!")
        
        print("\n📋 Tables created:")
        print("   - users")
        print("   - user_sessions") 
        print("   - query_logs")
        
        print("\n🎉 Database setup complete!")
        print("   You can now run your FastAPI app with authentication.")
        
    except Exception as e:
        print(f"❌ Database setup failed: {e}")
        print("\n🔧 Troubleshooting:")
        print("   1. Check your DATABASE_URL in .env file")
        print("   2. Ensure Azure PostgreSQL server is running")
        print("   3. Verify firewall allows your IP")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())