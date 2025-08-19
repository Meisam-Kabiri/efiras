"""
Simple website visit tracker for FastAPI with bot detection
"""

import os
import re
from datetime import datetime, timezone
from sqlalchemy import create_engine, text
from fastapi import Request

def is_bot(user_agent: str, headers: dict) -> bool:
    """
    Comprehensive bot detection based on user agent and headers
    Returns True if the request appears to be from a bot/crawler
    """
    if not user_agent:
        return True
    
    user_agent_lower = user_agent.lower()
    
    # Known bot patterns in user agent
    bot_patterns = [
        # Search engine crawlers
        r'googlebot', r'bingbot', r'slurp', r'duckduckbot', r'baiduspider', r'yandexbot',
        r'facebookexternalhit', r'twitterbot', r'linkedinbot', r'whatsapp', r'telegrambot',
        
        # Generic bot indicators
        r'\bbot\b', r'\bcrawl', r'\bspider\b', r'\bscrape', r'\bfetch', r'\bindex',
        r'scanner', r'monitor', r'check', r'test', r'probe', r'validator',
        
        # Specific tools and services
        r'censys', r'zgrab', r'masscan', r'nmap', r'curl', r'wget', r'python-requests',
        r'postman', r'insomnia', r'httpie', r'apache-httpclient', r'java/', r'okhttp',
        
        # Security scanners
        r'nessus', r'qualys', r'rapid7', r'shodan', r'nuclei', r'sqlmap',
        r'nikto', r'burp', r'zap', r'acunetix',
        
        # Monitoring services
        r'pingdom', r'newrelic', r'datadog', r'uptimerobot', r'statuscake',
        r'synthetic', r'monitoring', r'uptime',
        
        # SEO tools
        r'semrush', r'ahrefs', r'majestic', r'screaming frog', r'moz\.com',
        
        # Screenshot/preview services
        r'screenshot', r'preview', r'thumbnail', r'capture', r'render',
        r'vercel-screenshot', r'puppeteer', r'headless', r'phantom',
        
        # Development tools
        r'axios', r'node-fetch', r'got/', r'superagent', r'undici',
        
        # Malicious/suspicious
        r'hack', r'exploit', r'penetration', r'security', r'vulnerability'
    ]
    
    # Check if user agent matches any bot pattern
    for pattern in bot_patterns:
        if re.search(pattern, user_agent_lower):
            return True
    
    # Suspicious user agent characteristics
    if (
        len(user_agent) < 10 or  # Too short
        len(user_agent) > 1000 or  # Too long (suspicious)
        user_agent in ['Mozilla/5.0', '-', ''] or  # Generic/empty
        user_agent.count('(') != user_agent.count(')') or  # Malformed parentheses
        'http://' in user_agent_lower or 'https://' in user_agent_lower  # URLs in UA (spam)
    ):
        return True
    
    # Check for missing browser headers that real browsers always send
    browser_headers = ['accept', 'accept-language', 'accept-encoding']
    missing_headers = sum(1 for header in browser_headers if header not in headers)
    
    # If missing more than 1 essential browser header, likely a bot
    if missing_headers > 1:
        return True
    
    # Check for suspicious header combinations
    accept_header = headers.get('accept', '').lower()
    if accept_header and (
        accept_header == '*/*' or  # Too generic
        'text/html' not in accept_header  # Not requesting HTML (API bots)
    ):
        return True
    
    return False

def is_likely_real_user(user_agent: str, headers: dict) -> bool:
    """
    Additional validation for real users based on browser characteristics
    """
    if not user_agent:
        return False
        
    user_agent_lower = user_agent.lower()
    
    # Real browser indicators
    real_browser_patterns = [
        r'chrome/\d+', r'firefox/\d+', r'safari/\d+', r'edge/\d+',
        r'opera/\d+', r'chromium/\d+', r'vivaldi/\d+'
    ]
    
    has_real_browser = any(re.search(pattern, user_agent_lower) for pattern in real_browser_patterns)
    
    # Check for common browser headers
    has_accept_language = 'accept-language' in headers
    has_accept_encoding = 'accept-encoding' in headers
    accepts_html = 'text/html' in headers.get('accept', '').lower()
    
    return has_real_browser and has_accept_language and has_accept_encoding and accepts_html

def track_visit(request: Request):
    """Track a website visit with IP, user agent, and timestamp (real users only)"""
    
    # Get client IP (handle proxy headers if needed)
    client_ip = request.client.host
    if "x-forwarded-for" in request.headers:
        client_ip = request.headers["x-forwarded-for"].split(",")[0].strip()
    
    # Get user agent (browser info)
    user_agent = request.headers.get("user-agent", "Unknown")
    
    # Convert headers to lowercase dict for case-insensitive checking
    headers_dict = {k.lower(): v for k, v in request.headers.items()}
    
    # Bot detection - skip tracking for bots
    if is_bot(user_agent, headers_dict):
        print(f"🤖 Bot detected, skipping tracking: {client_ip} - {user_agent[:100]}...")
        return
    
    # Additional validation for real users
    if not is_likely_real_user(user_agent, headers_dict):
        print(f"⚠️ Suspicious request, skipping tracking: {client_ip} - {user_agent[:100]}...")
        return
    
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
        
        print(f"✅ Real user visit tracked: {client_ip} - {user_agent[:50]}...")
        
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