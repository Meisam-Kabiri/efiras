# auth_middleware.py
# WHAT: Middleware to check if user is authenticated on protected routes
# WHY: Automatically verify tokens before accessing chatbot/protected features

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from auth.firebase_admin_config import verify_firebase_token
from auth.firebase_user_tracker import usage_tracker

# STEP 1: Set up HTTP Bearer token security
security = HTTPBearer()


# FUNCTION: Get current authenticated user
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    """
    WHAT: Extracts and verifies the Firebase token from request headers
    WHY: Every protected endpoint uses this to verify the user
    HOW TO USE: Add as dependency to any route that needs authentication

    Example usage in the routes:
    @app.post("/chat")
    async def chat_endpoint(current_user: dict = Depends(get_current_user)):
        # current_user contains user info
        user_id = current_user["user_id"]
        email = current_user["email"]
    """

    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required"
        )

    try:
        # Extract token from "Bearer <token>" format
        token = credentials.credentials

        # Verify token with Firebase
        user_info = await verify_firebase_token(token)

        # Automatically ensure user exists in database
        try:
            await usage_tracker.ensure_user_exists(
                user_info["user_id"], user_info["email"]
            )
        except Exception as db_error:
            # Log but don't fail authentication if database user creation fails
            print(f"Warning: Failed to create user in database: {db_error}")

        return user_info

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid authentication: {str(e)}",
        )


# FUNCTION: Optional authentication (for routes that work with/without login)
async def get_current_user_optional(
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    """
    WHAT: Same as get_current_user but doesn't fail if no token provided
    WHY: For endpoints that work differently for logged-in vs anonymous users
    RETURNS: User info if logged in, None if anonymous
    """

    if not credentials:
        return None

    try:
        token = credentials.credentials
        user_info = await verify_firebase_token(token)

        # Automatically ensure user exists in database
        try:
            await usage_tracker.ensure_user_exists(
                user_info["user_id"], user_info["email"]
            )
        except Exception as db_error:
            print(f"Warning: Failed to create user in database: {db_error}")

        return user_info
    except:
        return None
