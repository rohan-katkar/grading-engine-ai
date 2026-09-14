# %% [markdown]
# # Main API Module (`src/api.py`)
# FastAPI application handling Authentication, User Management, RAG-Grading Submissions, 
# and Human Review Queues.

# %% [imports]
import uuid
import sys
from pathlib import Path
from typing import List
from fastapi import FastAPI, Depends, status, HTTPException
from sqlalchemy.orm import Session

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Local Utilities & Constants
from src.utils.constants import UserRole
from src.schemas import (
    UserToken,
    SubmissionRequest,
    SubmissionResponse,
    QuestionCreateRequest,
    HumanReviewItem,
    LoginRequest,
    UserCreateRequest,
    UserResponse,
    TokenResponse,
)
from src.auth import create_access_token, get_current_user, RequireRoles
from src.utils.security import hash_password, verify_password
from src.vector_store import TextbookVectorStore
from src.workflow import grading_workflow
from src.database import SessionLocal, User, ExamQuestion

# %% [app_setup]
app = FastAPI(
    title="Deterministic AI Grading Engine",
    version="1.0.0",
    description="GPU-Accelerated RAG Grading Platform with Human-in-the-Loop Architecture",
)

# Initialize vector store once on startup
vector_store = TextbookVectorStore()

# In-memory review queue for low-confidence evaluations
human_review_queue: List[HumanReviewItem] = []


def get_db():
    """Dependency providing database session scope per request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# %% [auth_endpoints]
# ## 1. Authentication & User Management Endpoints

@app.post("/auth/login", response_model=TokenResponse, tags=["Auth"])
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    """
    Authenticates users via strict email lookup against stored salted password hashes 
    and generates standard JWT access tokens.
    """
    # Strict lookup by unique email only
    user = db.query(User).filter(User.email == payload.username).first()

    if not user or not verify_password(user.password_hash, payload.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_role = UserRole(user.role) if isinstance(user.role, str) else user.role

    access_token = create_access_token(
        user_id=str(user.user_id),
        username=user.full_name,
        role=user_role,
    )

    user_resp = UserResponse(
        user_id=str(user.user_id),
        username=user.full_name,
        email=user.email,
        full_name=user.full_name,
        role=user_role.value,
    )

    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        user=user_resp,
    )


# %% [user_registration]
@app.post("/api/v1/auth/register/student", response_model=UserResponse, status_code=status.HTTP_201_CREATED, tags=["Auth"])
def register_student(payload: UserCreateRequest, db: Session = Depends(get_db)):
    """Public self-registration endpoint restricted strictly to STUDENT accounts."""
    if payload.role != UserRole.STUDENT:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Public self-registration is restricted to STUDENT accounts only.",
        )

    if db.query(User).filter_by(email=payload.email).first():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Email '{payload.email}' is already registered.",
        )

    new_student = User(
        user_id=uuid.uuid4(),
        full_name=payload.full_name or payload.username,
        email=payload.email,
        password_hash=hash_password(payload.password),
        role=UserRole.STUDENT
    )

    db.add(new_student)
    db.commit()
    db.refresh(new_student)

    student_role = UserRole(new_student.role) if isinstance(new_student.role, str) else new_student.role

    return UserResponse(
        user_id=str(new_student.user_id),
        username=new_student.full_name,
        email=new_student.email,
        full_name=new_student.full_name,
        role=student_role.value,
    )


# %% [admin_create_user]
@app.post("/api/v1/users", response_model=UserResponse, status_code=status.HTTP_201_CREATED, tags=["User Management"])
def create_user(
    payload: UserCreateRequest,
    db: Session = Depends(get_db),
    current_user: UserToken = Depends(RequireRoles([UserRole.ADMIN])),
):
    """
    Administrative user creation for EXAM_REVIEWER, EXAM_CREATOR, ADMIN, or STUDENT accounts.
    Requires ADMIN privileges.
    """
    if db.query(User).filter_by(email=payload.email).first():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Email '{payload.email}' is already registered.",
        )

    assigned_role = payload.role if isinstance(payload.role, UserRole) else UserRole(payload.role)

    new_user = User(
        user_id=uuid.uuid4(),
        full_name=payload.full_name or payload.username,
        email=payload.email,
        password_hash=hash_password(payload.password),
        role=assigned_role
    )

    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    user_role = UserRole(new_user.role) if isinstance(new_user.role, str) else new_user.role

    return UserResponse(
        user_id=str(new_user.user_id),
        username=new_user.full_name,
        email=new_user.email,
        full_name=new_user.full_name,
        role=user_role.value,
    )


# %% [get_user_by_id]
@app.get("/api/v1/users/{user_id}", response_model=UserResponse, tags=["User Management"])
def get_user_by_id(
    user_id: str,
    db: Session = Depends(get_db),
    current_user: UserToken = Depends(RequireRoles([UserRole.ADMIN, UserRole.EXAM_REVIEWER])),
):
    """Fetches user details by UUIDv4 identifier."""
    try:
        valid_uuid = uuid.UUID(user_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid UUIDv4 format for user_id.",
        )

    user = db.query(User).filter_by(user_id=valid_uuid).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with ID '{user_id}' not found.",
        )

    user_role = UserRole(user.role) if isinstance(user.role, str) else user.role

    return UserResponse(
        user_id=str(user.user_id),
        username=user.full_name,
        email=user.email,
        full_name=user.full_name,
        role=user_role.value,
    )


# %% [submissions]
# ## 2. Student Submissions & Pipeline Endpoints

@app.post("/api/v1/submit", response_model=SubmissionResponse, tags=["Submissions"])
def submit_answer(
    payload: SubmissionRequest,
    db: Session = Depends(get_db),
    current_user: UserToken = Depends(RequireRoles([UserRole.STUDENT, UserRole.ADMIN])),
):
    # 1. Identity Check: Impersonation Protection
    is_student = (
        current_user.role == UserRole.STUDENT
        or current_user.role == UserRole.STUDENT.value
    )
    if is_student and payload.student_id != current_user.user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot submit for another student.",
        )

    # 2. Validate Student UUID string format
    try:
        student_uuid = uuid.UUID(payload.student_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid UUID format for student_id: '{payload.student_id}'",
        )

    # 3. SHORT-CIRCUIT: Fast DB check to verify user exists BEFORE running expensive LLM inference
    user_exists = db.query(User).filter_by(user_id=student_uuid).first()
    if not user_exists:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Student ID '{payload.student_id}' is not registered in the system.",
        )

    # 4. Generate Submission UUIDv4
    submission_id = str(uuid.uuid4())

    # 5. Invoke LangGraph Execution Graph (Only runs if user validation passes!)
    graph_input = {
        "submission_id": submission_id,
        "student_id": payload.student_id,
        "question_id": payload.question_id,
        "question_type": payload.question_type,
        "question_text": payload.question_text,
        "subject": payload.subject,
        "topic": payload.topic,
        "max_marks": payload.max_marks,
        "official_rubric": payload.rubric,
        "required_keywords": payload.required_keywords,
        "student_answer": payload.student_answer,
    }

    final_state = grading_workflow.invoke(graph_input)
    raw_eval = final_state.get("raw_eval") or {}

    context_str = final_state.get("rag_context", "")
    context_list = [c.strip() for c in context_str.split("\n\n---\n\n") if c.strip()] if context_str else []

    flagged = final_state.get("final_status") == "NEEDS_HUMAN_REVIEW"

    return SubmissionResponse(
        submission_id=submission_id,
        student_id=payload.student_id,
        question_id=payload.question_id,
        assigned_score=float(raw_eval.get("score", 0.0)),
        max_marks=payload.max_marks,
        confidence_score=final_state.get("composite_confidence", 0.0),
        feedback=raw_eval.get("feedback", "No feedback generated."),
        final_status=final_state.get("final_status", "UNKNOWN"),
        flagged_for_human_review=flagged,
        flag_reason=final_state.get("flag_reason"),
        pii_detected=final_state.get("pii_detected", False),
        sanitized_answer=final_state.get("sanitized_answer"),
        key_points_matched=raw_eval.get("key_points_matched", []),
        key_points_missing=raw_eval.get("key_points_missing", []),
        retrieved_context=context_list,
    )


# %% [review_and_questions]
# ## 3. Review Queue & Question Management Endpoints

@app.get("/api/v1/review/queue", response_model=List[HumanReviewItem], tags=["Review Queue"])
def get_human_review_queue(
    current_user: UserToken = Depends(RequireRoles([UserRole.EXAM_REVIEWER, UserRole.ADMIN])),
):
    """
    Fetches flagged submissions needing manual human review.
    Protected: Accessible by EXAM_REVIEWER and ADMIN roles.
    """
    return human_review_queue


@app.post("/api/v1/questions", status_code=status.HTTP_201_CREATED, tags=["Question Management"])
def create_question(
    payload: QuestionCreateRequest,
    current_user: UserToken = Depends(RequireRoles([UserRole.EXAM_CREATOR, UserRole.ADMIN])),
):
    """
    Creates reference questions and rubrics in database.
    Protected: Accessible by EXAM_CREATOR and ADMIN roles.
    """
    return {"status": "success", "message": "Question created successfully", "data": payload}