"""
Simple website visit tracker for FastAPI with bot detection.

Stores visits in a local SQLite file rather than Postgres - this is
low-volume analytics data, not query-critical, so it doesn't need a
managed database. Note: on Cloud Run each instance has its own ephemeral
filesystem, so visit counts are per-instance and not durable across
restarts/scale-events unless min-instances=1.
"""

import os
import re
import sqlite3
from datetime import datetime, timezone

from fastapi import Request

VISITS_DB_PATH = os.getenv("VISITS_DB_PATH", "data/visits.db")


def get_connection() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(VISITS_DB_PATH) or ".", exist_ok=True)
    return sqlite3.connect(VISITS_DB_PATH)


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
        r"googlebot",
        r"bingbot",
        r"slurp",
        r"duckduckbot",
        r"baiduspider",
        r"yandexbot",
        r"facebookexternalhit",
        r"twitterbot",
        r"linkedinbot",
        r"whatsapp",
        r"telegrambot",
        # Generic bot indicators
        r"\bbot\b",
        r"\bcrawl",
        r"\bspider\b",
        r"\bscrape",
        r"\bfetch",
        r"\bindex",
        r"scanner",
        r"monitor",
        r"check",
        r"test",
        r"probe",
        r"validator",
        # Specific tools and services
        r"censys",
        r"zgrab",
        r"masscan",
        r"nmap",
        r"curl",
        r"wget",
        r"python-requests",
        r"postman",
        r"insomnia",
        r"httpie",
        r"apache-httpclient",
        r"java/",
        r"okhttp",
        # Security scanners
        r"nessus",
        r"qualys",
        r"rapid7",
        r"shodan",
        r"nuclei",
        r"sqlmap",
        r"nikto",
        r"burp",
        r"zap",
        r"acunetix",
        # Monitoring services
        r"pingdom",
        r"newrelic",
        r"datadog",
        r"uptimerobot",
        r"statuscake",
        r"synthetic",
        r"monitoring",
        r"uptime",
        # SEO tools
        r"semrush",
        r"ahrefs",
        r"majestic",
        r"screaming frog",
        r"moz\.com",
        # Screenshot/preview services
        r"screenshot",
        r"preview",
        r"thumbnail",
        r"capture",
        r"render",
        r"vercel-screenshot",
        r"puppeteer",
        r"headless",
        r"phantom",
        # Development tools
        r"axios",
        r"node-fetch",
        r"got/",
        r"superagent",
        r"undici",
        # Malicious/suspicious
        r"hack",
        r"exploit",
        r"penetration",
        r"security",
        r"vulnerability",
    ]

    # Check if user agent matches any bot pattern
    for pattern in bot_patterns:
        if re.search(pattern, user_agent_lower):
            return True

    # Suspicious user agent characteristics
    if (
        len(user_agent) < 10  # Too short
        or len(user_agent) > 1000  # Too long (suspicious)
        or user_agent in ["Mozilla/5.0", "-", ""]  # Generic/empty
        or user_agent.count("(") != user_agent.count(")")  # Malformed parentheses
        or "http://" in user_agent_lower
        or "https://" in user_agent_lower  # URLs in UA (spam)
    ):
        return True

    # Check for missing browser headers that real browsers always send
    browser_headers = ["accept", "accept-language", "accept-encoding"]
    missing_headers = sum(1 for header in browser_headers if header not in headers)

    # If missing ALL essential browser headers, likely a bot (relaxed from >1)
    if missing_headers >= 3:
        return True

    # Check for suspicious header combinations (only if accept header exists)
    accept_header = headers.get("accept", "").lower()
    if accept_header and accept_header == "*/*" and missing_headers > 1:
        # Only flag as bot if both conditions: generic accept AND missing other headers
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
        r"chrome/\d+",
        r"firefox/\d+",
        r"safari/\d+",
        r"edge/\d+",
        r"opera/\d+",
        r"chromium/\d+",
        r"vivaldi/\d+",
    ]

    has_real_browser = any(
        re.search(pattern, user_agent_lower) for pattern in real_browser_patterns
    )

    # If it has a real browser pattern, be more lenient with headers
    if has_real_browser:
        return True

    # For non-standard browsers, check headers more carefully
    has_accept_language = "accept-language" in headers
    has_accept_encoding = "accept-encoding" in headers
    accepts_html = "text/html" in headers.get("accept", "").lower()

    # Allow if at least 2 out of 3 browser characteristics are present
    browser_score = sum([has_accept_language, has_accept_encoding, accepts_html])
    return browser_score >= 2


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
        print(
            f"🤖 Bot detected, skipping tracking: {client_ip} - {user_agent[:100]}..."
        )
        return

    # Additional validation for real users
    if not is_likely_real_user(user_agent, headers_dict):
        print(
            f"⚠️ Suspicious request, skipping tracking: {client_ip} - {user_agent[:100]}..."
        )
        print(f"   Headers: {list(headers_dict.keys())}")
        return

    try:
        conn = get_connection()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS website_visits (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ip_address TEXT NOT NULL,
                    user_agent TEXT,
                    visit_time TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "INSERT INTO website_visits (ip_address, user_agent, visit_time) VALUES (?, ?, ?)",
                (client_ip, user_agent, datetime.now(timezone.utc).isoformat()),
            )
            conn.commit()
        finally:
            conn.close()

        print(f"✅ Real user visit tracked: {client_ip} - {user_agent[:50]}...")

    except Exception as e:
        print(f"❌ Failed to track visit: {e}")


def get_visit_stats():
    """Get basic visit statistics"""

    try:
        conn = get_connection()
        try:
            total_visits = conn.execute(
                "SELECT COUNT(*) FROM website_visits"
            ).fetchone()[0]

            unique_visitors = conn.execute(
                "SELECT COUNT(DISTINCT ip_address) FROM website_visits"
            ).fetchone()[0]

            today_visits = conn.execute(
                "SELECT COUNT(*) FROM website_visits WHERE DATE(visit_time) = DATE('now')"
            ).fetchone()[0]

            return {
                "total_visits": total_visits,
                "unique_visitors": unique_visitors,
                "today_visits": today_visits,
            }
        finally:
            conn.close()

    except Exception as e:
        print(f"❌ Failed to get visit stats: {e}")
        return {}


def get_recent_visits(limit=20):
    """Get recent visits for admin view"""

    try:
        conn = get_connection()
        try:
            rows = conn.execute(
                """
                SELECT ip_address, user_agent, visit_time
                FROM website_visits
                ORDER BY visit_time DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

            return [
                {"ip": row[0], "user_agent": row[1], "visit_time": row[2]}
                for row in rows
            ]
        finally:
            conn.close()

    except Exception as e:
        print(f"❌ Failed to get recent visits: {e}")
        return []
