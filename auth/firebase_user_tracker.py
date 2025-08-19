# usage_tracker_azure.py
# WHAT: PostgreSQL-based usage tracking for Azure
# WHY: Production-ready usage limits with proper database storage


# PostgreSQL Driver Comparison:
# 1. asyncpg (Recommended for FastAPI)
#    - Fastest, async/await, connection pooling, best for high concurrency
#    - Only supports PostgreSQL, async syntax required
#
# 2. psycopg2 (Traditional)
#    - Mature, simple, lots of resources
#    - Blocking, poor for async apps
#
# 3. SQLAlchemy (ORM)
#    - Database agnostic, ORM features, migrations
#    - Slower, complex for simple queries, heavier dependency

import asyncio
import asyncpg
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
import os
from dataclasses import dataclass
import logging

@dataclass
class UsageInfo:
    daily_queries: int
    daily_limit: int
    remaining: int
    plan: str
    total_queries: int
    last_query_date: str

class AzurePostgresUsageTracker:
    """
    WHAT: Production usage tracker using Azure PostgreSQL
    WHY: Scalable, reliable usage tracking with proper database storage
    """
    
    def __init__(self):
        self.connection_pool = None
        self.logger = logging.getLogger(__name__)
        
        # Azure PostgreSQL connection settings
        self.db_config = {
            "host": os.getenv("PGHOST", "your-server.postgres.database.azure.com"),
            "port": int(os.getenv("PGPORT", "5432")),
            "database": os.getenv("PGDATABASE", "efiras_db"),
            "user": os.getenv("PGUSER", "efirasadmindb"),
            "password": os.getenv("PGPASSWORD"),
            "ssl": "require"  # Azure PostgreSQL requires SSL
        }
    
    async def initialize(self):
        """
        WHAT: Initialize database connection pool and create tables
        WHY: Set up database connections and ensure tables exist
        """
        try:
            # Create connection pool for better performance
            self.connection_pool = await asyncpg.create_pool(
                **self.db_config,
                min_size=2,
                max_size=10,
                command_timeout=60
            )
            
            # Create tables if they don't exist
            await self._create_tables()
            self.logger.info("✅ Azure PostgreSQL connection established")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to connect to Azure PostgreSQL: {e}")
            raise
    
    async def _create_tables(self):
        """
        WHAT: Create necessary tables for usage tracking
        WHY: Ensure database schema exists
        """
        async with self.connection_pool.acquire() as conn:
            # Users table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id VARCHAR(255) PRIMARY KEY,
                    email VARCHAR(255) UNIQUE NOT NULL,
                    plan VARCHAR(50) DEFAULT 'free',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    total_queries INTEGER DEFAULT 0,
                    is_active BOOLEAN DEFAULT true
                );
            """)
            
            # Daily usage table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS daily_usage (
                    id SERIAL PRIMARY KEY,
                    user_id VARCHAR(255),
                    usage_date DATE DEFAULT CURRENT_DATE,
                    query_count INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id, usage_date)
                );
            """)
            
            # Query logs table (for analytics)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS query_logs (
                    id SERIAL PRIMARY KEY,
                    user_id VARCHAR(255),
                    query_type VARCHAR(100) DEFAULT 'chatbot',
                    query_text TEXT,
                    response_time_ms INTEGER,
                    success BOOLEAN DEFAULT true,
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # Create indexes for better performance
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_daily_usage_user_date 
                ON daily_usage(user_id, usage_date);
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_query_logs_user_created 
                ON query_logs(user_id, created_at);
            """)
    
    async def ensure_user_exists(self, user_id: str, email: str, plan: str = "free"):
        """
        WHAT: Ensure user exists in database
        WHY: Create user record if they don't exist
        """
        async with self.connection_pool.acquire() as conn:
            try:
                # Try to insert the user
                await conn.execute("""
                    INSERT INTO users (user_id, email, plan) 
                    VALUES ($1, $2, $3)
                    ON CONFLICT (user_id) DO UPDATE SET
                        last_active = CURRENT_TIMESTAMP,
                        email = EXCLUDED.email
                """, user_id, email, plan)
            except asyncpg.UniqueViolationError as e:
                # Handle email uniqueness violation
                if "users_email_key" in str(e):
                    # Just update the timestamp, don't change user_id to avoid foreign key issues
                    await conn.execute("""
                        UPDATE users 
                        SET last_active = CURRENT_TIMESTAMP
                        WHERE email = $1
                    """, email)
                else:
                    # Re-raise if it's a different uniqueness violation
                    raise
    
    async def can_make_query(self, user_id: str, email: str) -> Tuple[bool, UsageInfo]:
        """
        WHAT: Check if user can make another query
        WHY: Enforce usage limits based on user plan
        RETURNS: (can_query: bool, usage_info: UsageInfo)
        """
        async with self.connection_pool.acquire() as conn:
            # Get user info and today's usage
            user_data = await conn.fetchrow("""
                SELECT u.plan, u.total_queries,
                       COALESCE(du.query_count, 0) as daily_queries
                FROM users u
                LEFT JOIN daily_usage du ON u.user_id = du.user_id 
                    AND du.usage_date = CURRENT_DATE
                WHERE u.user_id = $1
            """, user_id)
            
            if not user_data:
                # New user - create them
                await self.ensure_user_exists(user_id, email)
                user_data = await conn.fetchrow("""
                    SELECT plan, total_queries, 0 as daily_queries
                    FROM users WHERE user_id = $1
                """, user_id)
            
            # Define limits by plan
            limits = {
                "free": 10,        # 10 queries per day
                "premium": 20,   # 1000 queries per day
                "enterprise": 20 # Enterprise limit
            }
            
            plan = user_data['plan']
            daily_limit = limits.get(plan, 10)
            daily_queries = user_data['daily_queries']
            
            can_query = daily_queries < daily_limit
            
            usage_info = UsageInfo(
                daily_queries=daily_queries,
                daily_limit=daily_limit,
                remaining=max(0, daily_limit - daily_queries),
                plan=plan,
                total_queries=user_data['total_queries'],
                last_query_date=datetime.now().date().isoformat()
            )
            
            return can_query, usage_info
    
    async def record_query(self, user_id: str, email: str, query_text: str = "", 
                          response_time_ms: int = 0, success: bool = True, 
                          error_message: str = None) -> UsageInfo:
        """
        WHAT: Record that user made a query
        WHY: Increment counters and log the query
        RETURNS: Updated usage info
        """
        async with self.connection_pool.acquire() as conn:
            async with conn.transaction():
                # Ensure user exists
                await self.ensure_user_exists(user_id, email)
                
                # Update or insert daily usage
                await conn.execute("""
                    INSERT INTO daily_usage (user_id, usage_date, query_count)
                    VALUES ($1, CURRENT_DATE, 1)
                    ON CONFLICT (user_id, usage_date)
                    DO UPDATE SET 
                        query_count = daily_usage.query_count + 1,
                        updated_at = CURRENT_TIMESTAMP
                """, user_id)
                
                # Update total queries in users table
                await conn.execute("""
                    UPDATE users 
                    SET total_queries = total_queries + 1,
                        last_active = CURRENT_TIMESTAMP
                    WHERE user_id = $1
                """, user_id)
                
                # Log the query
                await conn.execute("""
                    INSERT INTO query_logs 
                    (user_id, query_text, response_time_ms, success, error_message)
                    VALUES ($1, $2, $3, $4, $5)
                """, user_id, query_text[:1000], response_time_ms, success, error_message)
                
                # Get updated usage info
                can_query, usage_info = await self.can_make_query(user_id, email)
                return usage_info
    
    async def get_user_analytics(self, user_id: str) -> Dict:
        """
        WHAT: Get detailed analytics for a user
        WHY: Provide insights for user dashboard
        """
        async with self.connection_pool.acquire() as conn:
            # Basic user stats
            user_stats = await conn.fetchrow("""
                SELECT plan, total_queries, created_at, last_active
                FROM users WHERE user_id = $1
            """, user_id)
            
            if not user_stats:
                return {}
            
            # Last 7 days usage
            weekly_usage = await conn.fetch("""
                SELECT usage_date, query_count
                FROM daily_usage
                WHERE user_id = $1 
                    AND usage_date >= CURRENT_DATE - INTERVAL '7 days'
                ORDER BY usage_date DESC
            """, user_id)
            
            # Monthly total
            monthly_total = await conn.fetchval("""
                SELECT COALESCE(SUM(query_count), 0)
                FROM daily_usage
                WHERE user_id = $1 
                    AND usage_date >= CURRENT_DATE - INTERVAL '30 days'
            """, user_id)
            
            # Average response time
            avg_response_time = await conn.fetchval("""
                SELECT COALESCE(AVG(response_time_ms), 0)
                FROM query_logs
                WHERE user_id = $1 
                    AND created_at >= CURRENT_TIMESTAMP - INTERVAL '7 days'
                    AND success = true
            """, user_id)
            
            return {
                "plan": user_stats['plan'],
                "total_queries": user_stats['total_queries'],
                "member_since": user_stats['created_at'].isoformat(),
                "last_active": user_stats['last_active'].isoformat(),
                "monthly_total": monthly_total,
                "weekly_usage": [
                    {"date": row['usage_date'].isoformat(), "queries": row['query_count']}
                    for row in weekly_usage
                ],
                "avg_response_time_ms": round(float(avg_response_time or 0), 2)
            }
    
    async def get_system_stats(self) -> Dict:
        """
        WHAT: Get system-wide statistics
        WHY: Monitor overall platform usage
        """
        async with self.connection_pool.acquire() as conn:
            stats = await conn.fetchrow("""
                SELECT 
                    COUNT(DISTINCT user_id) as total_users,
                    COUNT(DISTINCT CASE WHEN last_active >= CURRENT_DATE THEN user_id END) as daily_active_users,
                    COUNT(DISTINCT CASE WHEN last_active >= CURRENT_DATE - INTERVAL '7 days' THEN user_id END) as weekly_active_users,
                    COALESCE(SUM(total_queries), 0) as total_queries,
                    COUNT(DISTINCT CASE WHEN plan = 'premium' THEN user_id END) as premium_users
                FROM users
                WHERE is_active = true
            """)
            
            # Today's queries
            todays_queries = await conn.fetchval("""
                SELECT COALESCE(SUM(query_count), 0)
                FROM daily_usage
                WHERE usage_date = CURRENT_DATE
            """)
            
            return {
                "total_users": stats['total_users'],
                "daily_active_users": stats['daily_active_users'],
                "weekly_active_users": stats['weekly_active_users'],
                "total_queries": stats['total_queries'],
                "todays_queries": todays_queries,
                "premium_users": stats['premium_users']
            }
    
    async def delete_user_account(self, user_id: str, email: str) -> bool:
        """
        WHAT: Completely delete user account and all related data
        WHY: Handle frontend account deletion properly
        RETURNS: True if deletion was successful
        """
        try:
            async with self.connection_pool.acquire() as conn:
                async with conn.transaction():
                    # Delete user data (no foreign key constraints, so order doesn't matter)
                    deleted_daily = await conn.execute("DELETE FROM daily_usage WHERE user_id = $1", user_id)
                    deleted_logs = await conn.execute("DELETE FROM query_logs WHERE user_id = $1", user_id)
                    deleted_user = await conn.execute("DELETE FROM users WHERE user_id = $1", user_id)
                    
                    # Also clean up by email in case of user_id mismatches
                    await conn.execute("DELETE FROM daily_usage WHERE user_id IN (SELECT user_id FROM users WHERE email = $1)", email)
                    await conn.execute("DELETE FROM query_logs WHERE user_id IN (SELECT user_id FROM users WHERE email = $1)", email)
                    await conn.execute("DELETE FROM users WHERE email = $1", email)
                    
                    self.logger.info(f"✅ Deleted user account: {email} (user_id: {user_id})")
                    return True
                    
        except Exception as e:
            self.logger.error(f"❌ Failed to delete user account {email}: {e}")
            return False

    async def close(self):
        """Close database connections"""
        if self.connection_pool:
            await self.connection_pool.close()

# Global instance
usage_tracker = AzurePostgresUsageTracker()

# Startup function
async def initialize_usage_tracker():
    """Call this when FastAPI app starts"""
    await usage_tracker.initialize()