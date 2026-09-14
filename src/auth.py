# src/auth.py
import datetime
import time
import os
import jwt
from dotenv import load_dotenv
from typing import List
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from src.schemas import UserToken
from src.utils.constants import UserRole

load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY")  # Store in .env
ALGORITHM = "HS256"

security = HTTPBearer()

def create_access_token(user_id: str, username: str, role: UserRole, expires_minutes: int = 120) -> str:
    """Generates a signed JWT with role claim."""
    # Compute expiration as a clean integer timestamp
    expire_timestamp = int(time.time()) + (expires_minutes * 60)
    
    payload = {
        "sub": user_id,
        "username": username,
        "role": role.value if hasattr(role, "value") else str(role),
        "exp": expire_timestamp
    }
    
    # Ensure payload is encoded as a string
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> UserToken:
    """Validates incoming Bearer token and returns authenticated user details."""
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return UserToken(
            user_id=payload["sub"],
            username=payload["username"],
            role=UserRole(payload["role"])
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token has expired")
    except jwt.PyJWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication token")

def RequireRoles(allowed_roles: List[UserRole]):
    """RBAC Dependency: Enforces that current user possesses an allowed role."""
    def role_checker(user: UserToken = Depends(get_current_user)):
        if user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, 
                detail=f"Access denied. Required role: {[r.value for r in allowed_roles]}"
            )
        return user
    return role_checker