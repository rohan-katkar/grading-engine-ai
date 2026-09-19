from fastapi.testclient import TestClient

from src.utils.constants import UserRole


def test_login_returns_bearer_token(client: TestClient, users):
    response = client.post(
        "/auth/login",
        json={"username": users[UserRole.STUDENT].email, "password": "Password123!"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["token_type"] == "bearer"
    assert body["user"]["role"] == UserRole.STUDENT.value
    assert body["access_token"]


def test_login_rejects_invalid_password(client: TestClient, users):
    response = client.post(
        "/auth/login",
        json={"username": users[UserRole.STUDENT].email, "password": "wrong-pass"},
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid email or password."


def test_student_registration_rejects_non_student_role(client: TestClient):
    response = client.post(
        "/api/v1/auth/register/student",
        json={
            "username": "reviewer",
            "email": "reviewer2@example.com",
            "password": "Password123!",
            "role": UserRole.EXAM_REVIEWER.value,
        },
    )

    assert response.status_code == 403