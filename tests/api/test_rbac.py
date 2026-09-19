from fastapi.testclient import TestClient

from src.utils.constants import UserRole


def test_student_cannot_read_review_queue(client: TestClient, access_token):
    token = access_token(UserRole.STUDENT)
    response = client.get(
        "/api/v1/review/queue",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 403


def test_reviewer_can_read_review_queue(client: TestClient, access_token):
    token = access_token(UserRole.EXAM_REVIEWER)
    response = client.get(
        "/api/v1/review/queue",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    assert response.json() == []


def test_invalid_token_is_rejected(client: TestClient):
    response = client.get(
        "/api/v1/review/queue",
        headers={"Authorization": "Bearer definitely-not-a-jwt"},
    )

    assert response.status_code == 401