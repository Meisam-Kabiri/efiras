"""
Simple website visit tracker for FastAPI
"""

import os
from datetime import datetime, timezone
from sqlalchemy import create_engine, text
from fastapi import Request

def track_visit(request: Request):
    """Track a website visit with IP, user agent, and timestamp"""
    
    # Get client IP (handle proxy headers if needed)
    client_ip = request.client.host
    if "x-forwarded-for" in request.headers:
        client_ip = request.headers["x-forwarded-for"].split(",")[0].strip()
    
    # Get user agent (browser info)
    user_agent = request.headers.get("user-agent", "Unknown")
    
    # Get database connection
    database_url = os.getenv('DATABASE_URL')
    if not database_url:
        print("❌ DATABASE_URL not found - cannot track visit")
        return
    
    try:
        engine = create_engine(database_url)
        
        # Insert visit record
        with engine.connect() as conn:
            conn.execute(
                text("""
                    INSERT INTO website_visits (ip_address, user_agent, visit_time) 
                    VALUES (:ip, :agent, :time)
                """),
                {
                    "ip": client_ip,
                    "agent": user_agent,
                    "time": datetime.now(timezone.utc)
                }
            )
            conn.commit()
        
        print(f"✅ Visit tracked: {client_ip}")
        
    except Exception as e:
        print(f"❌ Failed to track visit: {e}")

def get_visit_stats():
    """Get basic visit statistics"""
    
    database_url = os.getenv('DATABASE_URL')
    if not database_url:
        return {}
    
    try:
        engine = create_engine(database_url)
        
        with engine.connect() as conn:
            # Total visits
            total_result = conn.execute(text("SELECT COUNT(*) FROM website_visits"))
            total_visits = total_result.scalar()
            
            # Unique IPs
            unique_result = conn.execute(text("SELECT COUNT(DISTINCT ip_address) FROM website_visits"))
            unique_visitors = unique_result.scalar()
            
            # Today's visits
            today_result = conn.execute(
                text("SELECT COUNT(*) FROM website_visits WHERE DATE(visit_time) = CURRENT_DATE")
            )
            today_visits = today_result.scalar()
            
            return {
                "total_visits": total_visits,
                "unique_visitors": unique_visitors,
                "today_visits": today_visits
            }
    
    except Exception as e:
        print(f"❌ Failed to get visit stats: {e}")
        return {}

def get_recent_visits(limit=20):
    """Get recent visits for admin view"""
    
    database_url = os.getenv('DATABASE_URL')
    if not database_url:
        return []
    
    try:
        engine = create_engine(database_url)
        
        with engine.connect() as conn:
            result = conn.execute(
                text("""
                    SELECT ip_address, user_agent, visit_time 
                    FROM website_visits 
                    ORDER BY visit_time DESC 
                    LIMIT :limit
                """),
                {"limit": limit}
            )
            
            return [
                {
                    "ip": row[0],
                    "user_agent": row[1],
                    "visit_time": row[2].isoformat()
                }
                for row in result
            ]
    
    except Exception as e:
        print(f"❌ Failed to get recent visits: {e}")
        return []