#!/usr/bin/env python3
"""
Add website_visits table to track who opens the website
Usage: python website_visits_table.py
"""

import sys
import os
from pathlib import Path
from datetime import datetime

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

from sqlalchemy import Column, Integer, String, DateTime, create_engine
from sqlalchemy.ext.declarative import declarative_base
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

Base = declarative_base()

class WebsiteVisit(Base):
    __tablename__ = 'website_visits'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    ip_address = Column(String(45), nullable=False)  # IPv6 can be up to 45 chars
    user_agent = Column(String(500), nullable=True)  # Browser info
    visit_time = Column(DateTime, default=datetime.utcnow, nullable=False)

def create_website_visits_table():
    """Create the website_visits table"""
    
    # Get database URL from environment
    database_url = os.getenv('DATABASE_URL')
    if not database_url:
        raise ValueError("DATABASE_URL not found in environment variables")
    
    # Create engine
    engine = create_engine(database_url)
    
    # Create the table
    Base.metadata.create_all(engine, tables=[WebsiteVisit.__table__])
    print("✅ website_visits table created successfully!")

def main():
    print("🗄️  Creating website_visits table...")
    
    try:
        create_website_visits_table()
        
        print("\n📋 New table created:")
        print("   - website_visits (id, ip_address, user_agent, visit_time)")
        
        print("\n🎉 Website visit tracking table ready!")
        
    except Exception as e:
        print(f"❌ Failed to create website_visits table: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())