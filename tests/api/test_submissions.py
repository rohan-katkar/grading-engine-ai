from fastapi.testclient import TestClient

from src.utils.constants import UserRole


def test_student_cannot_submit_for_another_student(
    client: TestClient, access_token, users, monkeypatch
):
    token = access_token(UserRole.STUDENT)
    monkeypatch.setattr(
        "src.api.grading_workflow.invoke",
        lambda _: (_ for _ in ()).throw(AssertionError("workflow should not run")),
    )
    response = client.post(
        "/api/v1/submit",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "student_id": str(users[UserRole.ADMIN].user_id),
            "question_id": "question-1",
            "question_text": "What is ATP?",
            "student_answer": "A molecule used for energy.",
            "max_marks": 2,
            "rubric": "Mention energy transfer.",
        },
    )

    assert response.status_code == 403
    assert response.json()["detail"] == "Cannot submit for another student."


def test_registered_student_submission_uses_workflow_result(
    client: TestClient, access_token, users, monkeypatch
):
    token = access_token(UserRole.STUDENT)
    monkeypatch.setattr(
        "src.api.grading_workflow.invoke",
        lambda _: {
            "raw_eval": {
                "score": 2,
                "feedback": "Correct.",
                "key_points_matched": ["energy"],
                "key_points_missing": [],
            },
            "composite_confidence": 0.9,
            "final_status": "AUTO_APPROVED",
            "sanitized_answer": "A molecule used for energy.",
            "pii_detected": False,
        },
    )
    student_id = str(users[UserRole.STUDENT].user_id)

    response = client.post(
        "/api/v1/submit",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "student_id": student_id,
            "question_id": "question-1",
            "question_text": "What is ATP?",
            "student_answer": "A molecule used for energy.",
            "max_marks": 2,
            "rubric": "Mention energy transfer.",
        },
    )

    assert response.status_code == 200
    assert response.json()["student_id"] == student_id
    assert response.json()["assigned_score"] == 2.0
    assert response.json()["final_status"] == "AUTO_APPROVED"