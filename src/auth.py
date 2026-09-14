# src/auth.py
import os
import time
from typing import List, Optional

import jwt
from dotenv import load_dotenv
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from src.database import SessionLocal, User
from src.schemas import UserToken
from src.utils.constants import UserRole

load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY", "super-secret-key-change-in-production")
ALGORITHM = "HS256"

security = HTTPBearer()


def get_db():
    """Dependency providing database session scope per auth request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_access_token(
    user_id: str, username: str, role: UserRole, expires_minutes: int = 120
) -> str:
    """Generates a signed JWT with user identity and role claims."""
    expire_timestamp = int(time.time()) + (expires_minutes * 60)

    payload = {
        "sub": str(user_id),
        "username": username,
        "role": role.value if hasattr(role, "value") else str(role),
        "exp": expire_timestamp,
    }

    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db),
) -> UserToken:
    """
    Validates incoming Bearer token AND re-checks current user state against the database.
    Prevents deleted, deactivated, or stale-role users from maintaining unauthorized access.
    """
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: Optional[str] = payload.get("sub")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication token payload.",
            )
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
        )
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
        )

    # Query DB to ensure user exists and active status/role is fresh
    db_user = db.query(User).filter(User.user_id == user_id).first()
    if not db_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account no longer exists or is inactive.",
        )

    db_role_str = (
        db_user.role.value if hasattr(db_user.role, "value") else str(db_user.role)
    )

    return UserToken(
        user_id=str(db_user.user_id),
        username=db_user.full_name,
        role=UserRole(db_role_str),
    )


def RequireRoles(allowed_roles: List[UserRole]):
    """RBAC Dependency: Enforces that current user possesses an allowed role."""

    def role_checker(user: UserToken = Depends(get_current_user)):
        if user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. Required role: {[r.value for r in allowed_roles]}",
            )
        return user

    return role_checker