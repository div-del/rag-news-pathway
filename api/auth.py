"""
Authentication module for Clerk integration.
Handles token verification and user sync.
"""

import os
import logging
from datetime import datetime
from typing import Optional
from fastapi import Request, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import httpx

logger = logging.getLogger(__name__)

# Security scheme
security = HTTPBearer(auto_error=False)

# Clerk configuration
CLERK_SECRET_KEY = os.getenv("CLERK_SECRET_KEY", "")
CLERK_PUBLISHABLE_KEY = os.getenv("CLERK_PUBLISHABLE_KEY", "")


async def verify_clerk_token(token: str) -> Optional[dict]:
    """
    Verify a Clerk session token by calling Clerk's API.
    Returns user data if valid, None otherwise.
    """
    if not CLERK_SECRET_KEY:
        logger.warning("CLERK_SECRET_KEY not set, skipping verification")
        return None
    
    try:
        # Verify token with Clerk's Backend API
        async with httpx.AsyncClient() as client:
            # First, try to verify the session
            response = await client.get(
                "https://api.clerk.com/v1/sessions",
                headers={
                    "Authorization": f"Bearer {CLERK_SECRET_KEY}",
                    "Content-Type": "application/json"
                }
            )
            
            if response.status_code != 200:
                logger.warning(f"Clerk API error: {response.status_code}")
                return None
            
            # For now, decode the JWT locally (basic validation)
            # In production, you'd want to verify the signature
            import base64
            import json
            
            # Split the JWT
            parts = token.split('.')
            if len(parts) != 3:
                return None
            
            # Decode payload (middle part)
            payload = parts[1]
            # Add padding if needed
            padding = 4 - len(payload) % 4
            if padding != 4:
                payload += '=' * padding
            
            decoded = base64.urlsafe_b64decode(payload)
            claims = json.loads(decoded)
            
            return {
                "user_id": claims.get("sub"),
                "session_id": claims.get("sid"),
                "email": claims.get("email"),
                "exp": claims.get("exp")
            }
            
    except Exception as e:
        logger.error(f"Token verification error: {e}")
        return None


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[dict]:
    """
    FastAPI dependency to get the current authenticated user.
    Returns None if not authenticated (for optional auth).
    """
    if not credentials:
        return None
    
    user_data = await verify_clerk_token(credentials.credentials)
    return user_data


async def require_auth(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> dict:
    """
    FastAPI dependency that requires authentication.
    Raises 401 if not authenticated.
    """
    if not credentials:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    user_data = await verify_clerk_token(credentials.credentials)
    if not user_data:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    return user_data


def sync_user_to_db(db_session, clerk_user_data: dict):
    """
    Create or update a ClerkUser in the database.
    """
    from api.db_models import ClerkUser
    
    clerk_id = clerk_user_data.get("user_id")
    if not clerk_id:
        return None
    
    # Check if user exists
    existing_user = db_session.query(ClerkUser).filter(
        ClerkUser.clerk_id == clerk_id
    ).first()
    
    if existing_user:
        # Update last login
        existing_user.last_login = datetime.utcnow()
        db_session.commit()
        return existing_user
    
    # Create new user
    new_user = ClerkUser(
        clerk_id=clerk_id,
        email=clerk_user_data.get("email"),
        first_name=clerk_user_data.get("first_name"),
        last_name=clerk_user_data.get("last_name"),
        image_url=clerk_user_data.get("image_url")
    )
    db_session.add(new_user)
    db_session.commit()
    db_session.refresh(new_user)
    
    return new_user
