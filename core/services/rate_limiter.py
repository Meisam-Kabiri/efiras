
import time
import hashlib
from fastapi import Request, HTTPException
from collections import defaultdict
from threading import Lock

class SimpleMemoryRateLimiter:
    def __init__(self):
        self.usage_data = defaultdict(lambda: {"minute": {}, "hour": {}, "day": {}})
        self.lock = Lock()  # Thread safety
        
        # Set your limits here
        self.LIMITS = {
            "minute": 1,   # 3 requests per minute
            "hour": 1,    # 20 requests per hour
            "day": 10     # 100 requests per day
        }
    def get_user_fingerprint(self, request: Request) -> str:
        """Create unique fingerprint for each user based on browser characteristics"""
        
        # Combine IP + User Agent + Accept Language for uniqueness
        # Safely get headers with fallbacks for None values
        fingerprint_parts = [
            str(request.client.host or "unknown"),  # IP address
            str(request.headers.get("user-agent") or "unknown"),  # Browser info
            str(request.headers.get("accept-language") or "unknown"),  # Language
            str(request.headers.get("accept-encoding") or "unknown"),   # Encoding
            str(request.headers.get("cache-control") or "unknown"),
            str(request.headers.get("sec-ch-ua") or "unknown"),
        ]
        
        # Join them and create a hash
        fingerprint_string = "|".join(fingerprint_parts)
        fingerprint_hash = hashlib.sha256(fingerprint_string.encode()).hexdigest()
        
        # Return first 16 characters (enough for uniqueness)
        return fingerprint_hash[:16]


    def clean_old_data(self, user_data, window_type, current_window):
        """Remove old time windows to prevent memory buildup"""
        to_remove = []
        for window_time in user_data[window_type]:
            if window_time < current_window - 2:  # Keep last 2 windows for safety
                to_remove.append(window_time)
        
        for old_window in to_remove:
            del user_data[window_type][old_window]
    
    def check_and_increment(self, request):
        """Check rate limit and increment if allowed"""
        user_id = self.get_user_fingerprint(request)
        current_time = int(time.time())

        print(f"🔍 Debug - User ID: {user_id}")
        print(f"🔍 Debug - Current time: {current_time}")
        
        # Calculate current time windows
        current_minute = current_time // 60
        current_hour = current_time // 3600
        current_day = current_time // 86400

        print(f"🔍 Debug - Current minute window: {current_minute}")
        
        with self.lock:
            user_data = self.usage_data[user_id]
            
            # Clean old data to prevent memory leak
            self.clean_old_data(user_data, "minute", current_minute)
            self.clean_old_data(user_data, "hour", current_hour)
            self.clean_old_data(user_data, "day", current_day)
            
            # Get current counts
            minute_count = user_data["minute"].get(current_minute, 0)
            hour_count = user_data["hour"].get(current_hour, 0)
            day_count = user_data["day"].get(current_day, 0)

            print(f"🔍 Debug - Current counts: minute={minute_count}, hour={hour_count}, day={day_count}")
            print(f"🔍 Debug - Limits: minute={self.LIMITS['minute']}, hour={self.LIMITS['hour']}, day={self.LIMITS['day']}")
            
            # # Check limits
            # if minute_count >= self.LIMITS["minute"]:
            #     return {
            #         "allowed": False,
            #         "reason": "Too many requests per minute. Please wait a moment.",
            #         "reset_in": 60 - (current_time % 60)
            #     }
            
            # if hour_count >= self.LIMITS["hour"]:
            #     return {
            #         "allowed": False,
            #         "reason": "Hourly limit reached. Please wait or consider our paid plans.",
            #         "reset_in": 3600 - (current_time % 3600)
            #     }
                
            if day_count >= self.LIMITS["day"]:
                return {
                    "allowed": False,
                    "reason": "Daily limit reached.!",
                    "reset_in": 86400 - (current_time % 86400)
                }
            
            print(f"✅ Debug - ALLOWED: incrementing counters")
            # All good - increment counters
            user_data["minute"][current_minute] = minute_count + 1
            user_data["hour"][current_hour] = hour_count + 1
            user_data["day"][current_day] = day_count + 1
            
            print(f"🔍 Debug - After increment: minute={minute_count + 1}")
            return {
                "allowed": True,
                "remaining": min(
                    self.LIMITS["minute"] - minute_count - 1,
                    self.LIMITS["hour"] - hour_count - 1,
                    self.LIMITS["day"] - day_count - 1
                ),
                "usage": {
                    "minute": f"{minute_count + 1}/{self.LIMITS['minute']}",
                    "hour": f"{hour_count + 1}/{self.LIMITS['hour']}",
                    "day": f"{day_count + 1}/{self.LIMITS['day']}"
                }
            }
