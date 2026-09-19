import os
import sys
from collections.abc import Generator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

os.environ.setdefault("SECRET_KEY", "test-secret-key")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import api, auth
from src.database import Base, User
from src.utils.constants import UserRole
from src.utils.security import hash_password


@pytest.fixture()
def db_session() -> Generator[Session, None, None]:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    session = session_factory()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(engine)
        engine.dispose()


@pytest.fixture()
def client(db_session: Session) -> Generator[TestClient, None, None]:
    def override_get_db():
        yield db_session

    api.app.dependency_overrides[api.get_db] = override_get_db
    api.app.dependency_overrides[auth.get_db] = override_get_db
    with TestClient(api.app) as test_client:
        yield test_client
    api.app.dependency_overrides.clear()


@pytest.fixture()
def users(db_session: Session) -> dict[UserRole, User]:
    result = {}
    for role in UserRole:
        user = User(
            full_name=f"{role.value.title()} User",
            email=f"{role.value.lower()}@example.com",
            password_hash=hash_password("Password123!"),
            role=role,
        )
        db_session.add(user)
        result[role] = user
    db_session.commit()
    for user in result.values():
        db_session.refresh(user)
    return result


@pytest.fixture()
def access_token(client: TestClient, users: dict[UserRole, User]):
    def make_token(role: UserRole) -> str:
        response = client.post(
            "/auth/login",
            json={
                "username": users[role].email,
                "password": "Password123!",
            },
        )
        assert response.status_code == 200
        return response.json()["access_token"]

    return make_token