# firebase_admin_config.py
# WHAT: Connects your FastAPI backend to Firebase
# WHY: Lets your backend verify user tokens and get user info

import firebase_admin
from firebase_admin import credentials, auth

# STEP 1: Initialize Firebase Admin SDK

cred = credentials.Certificate("auth/firebase-service-account.json")
firebase_admin.initialize_app(cred)

# FUNCTION: Verify Firebase token from frontend
async def verify_firebase_token(token: str):
    """
    WHAT: Takes token from React frontend and verifies it with Firebase
    WHY: Proves the user is really logged in (not fake)
    RETURNS: User info if valid, raises exception if invalid
    """
    try:
        # Ask Firebase: "Is this token real?"
        decoded_token = auth.verify_id_token(token)
        
        # Extract user information
        user_info = {
            "user_id": decoded_token['uid'],           # Unique user ID
            "email": decoded_token.get('email'),       # User's email
            "email_verified": decoded_token.get('email_verified', False),
            "name": decoded_token.get('name'),          # Display name (if available)
            "picture": decoded_token.get('picture'),    # Profile photo (if from Google)
            "firebase_claims": decoded_token            # Full Firebase data
        }
        
        return user_info
        
    except auth.InvalidIdTokenError:
        raise Exception("Invalid or expired token")
    except auth.ExpiredIdTokenError:
        raise Exception("Token has expired")
    except Exception as e:
        raise Exception(f"Token verification failed: {str(e)}")

# FUNCTION: Get user info by user ID
async def get_user_by_id(user_id: str):
    """
    WHAT: Gets full user information from Firebase by user ID
    WHY: Sometimes we need more user details beyond the token
    """
    try:
        user_record = auth.get_user(user_id)
        return {
            "user_id": user_record.uid,
            "email": user_record.email,
            "display_name": user_record.display_name,
            "photo_url": user_record.photo_url,
            "email_verified": user_record.email_verified,
            "creation_time": user_record.user_metadata.creation_timestamp,
            "last_sign_in": user_record.user_metadata.last_sign_in_timestamp
        }
    except auth.UserNotFoundError:
        raise Exception("User not found")
    except Exception as e:
        raise Exception(f"Failed to get user: {str(e)}")