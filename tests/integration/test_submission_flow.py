from types import SimpleNamespace

from fastapi.testclient import TestClient

from src.utils.constants import UserRole


def test_authenticated_mcq_submission_runs_real_workflow(
    client: TestClient, access_token, users, monkeypatch
):
    recorded_states = []
    monkeypatch.setattr(
        "src.workflow.log_grading_run",
        lambda state, submission_id=None: recorded_states.append(dict(state)),
    )

    student_id = str(users[UserRole.STUDENT].user_id)
    token = access_token(UserRole.STUDENT)
    response = client.post(
        "/api/v1/submit",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "student_id": student_id,
            "question_id": "question-1",
            "question_type": "MCQ",
            "question_text": "Which molecule stores cellular energy?",
            "student_answer": "b",
            "max_marks": 2,
            "rubric": {"correct_option": "B"},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["student_id"] == student_id
    assert body["assigned_score"] == 2.0
    assert body["final_status"] == "COMPLETED"
    assert body["flagged_for_human_review"] is False
    assert recorded_states[0]["raw_eval"]["score"] == 2.0
    assert recorded_states[0]["final_status"] == "COMPLETED"


def test_authenticated_long_answer_runs_retrieval_evaluation_and_approval(
    client: TestClient, access_token, users, monkeypatch
):
    recorded_states = []
    monkeypatch.setattr(
        "src.workflow.log_grading_run",
        lambda state, submission_id=None: recorded_states.append(dict(state)),
    )
    monkeypatch.setattr(
        "src.workflow.vector_store.query_context",
        lambda **kwargs: "Mitochondria generate ATP through cellular respiration.",
    )
    monkeypatch.setattr(
        "src.workflow.chat",
        lambda **kwargs: SimpleNamespace(
            message=SimpleNamespace(
                content=(
                    '{"score": 4, "max_marks": 5, '
                    '"llm_self_confidence": 0.95, '
                    '"key_points_matched": ["ATP production"], '
                    '"key_points_missing": [], '
                    '"feedback": "Correct explanation."}'
                )
            )
        ),
    )

    student_id = str(users[UserRole.STUDENT].user_id)
    token = access_token(UserRole.STUDENT)
    response = client.post(
        "/api/v1/submit",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "student_id": student_id,
            "question_id": "question-2",
            "question_type": "LONG_ANSWER",
            "question_text": "Explain how mitochondria produce cellular energy.",
            "student_answer": "Mitochondria produce ATP through cellular respiration.",
            "max_marks": 5,
            "rubric": "Must explain ATP production through respiration.",
            "required_keywords": ["mitochondria", "ATP", "respiration"],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["assigned_score"] == 4.0
    assert body["confidence_score"] >= 0.80
    assert body["final_status"] == "AUTO_APPROVED"
    assert body["retrieved_context"] == [
        "Mitochondria generate ATP through cellular respiration."
    ]
    assert recorded_states[0]["raw_eval"]["feedback"] == "Correct explanation."


