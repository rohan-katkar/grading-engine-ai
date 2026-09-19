from hypothesis import given, settings
from hypothesis import strategies as st

from src.utils.security import hash_password, verify_password
from src.workflow import calculate_deterministic_confidence


@settings(max_examples=15)
@given(password=st.text(min_size=0, max_size=80))
def test_password_hash_round_trips_for_arbitrary_text(password):
    password_hash = hash_password(password)

    assert verify_password(password_hash, password)


@settings(max_examples=25)
@given(
    max_marks=st.floats(min_value=0.01, max_value=1000, allow_nan=False, allow_infinity=False),
    assigned_ratio=st.floats(min_value=0, max_value=1, allow_nan=False, allow_infinity=False),
    llm_confidence=st.floats(min_value=-100, max_value=200, allow_nan=False, allow_infinity=False),
)
def test_confidence_score_stays_in_valid_range(max_marks, assigned_ratio, llm_confidence):
    score = calculate_deterministic_confidence(
        max_marks=max_marks,
        assigned_score=max_marks * assigned_ratio,
        key_points_matched=["correct point"],
        key_points_missing=["missing point"],
        student_answer="mitochondria produce ATP",
        required_keywords=["mitochondria", "ATP"],
        llm_self_confidence=llm_confidence,
    )

    assert 0.0 <= score <= 1.0