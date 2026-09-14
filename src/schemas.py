# src/schemas.py
import sys
from pathlib import Path
from typing import List, Optional, Any
from pydantic import BaseModel, Field, EmailStr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.utils.constants import UserRole, QuestionType

# --------------------------------------------------------------------------
# Authentication & User Management Schemas
# --------------------------------------------------------------------------

class LoginRequest(BaseModel):
    username: str = Field(json_schema_extra={"example": "johndoe"})
    password: str = Field(json_schema_extra={"example": "SuperSecretPassword123!"})

class UserCreateRequest(BaseModel):
    username: str = Field(min_length=3, max_length=64, json_schema_extra={"example": "johndoe"})
    email: EmailStr = Field(json_schema_extra={"example": "john.doe@university.edu"})
    password: str = Field(min_length=8, description="Minimum 8 characters")
    full_name: Optional[str] = Field(default=None, json_schema_extra={"example": "John Doe"})
    role: UserRole = Field(default=UserRole.STUDENT, json_schema_extra={"example": UserRole.STUDENT.value})

class UserResponse(BaseModel):
    user_id: str
    username: str
    email: EmailStr
    full_name: Optional[str] = None
    role: UserRole

    class Config:
        from_attributes = True

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse

class UserToken(BaseModel):
    user_id: str
    username: str
    role: UserRole

# --------------------------------------------------------------------------
# Grading & Submission Schemas
# --------------------------------------------------------------------------

class SubmissionRequest(BaseModel):
    student_id: str
    question_id: str
    question_type: QuestionType = Field(default=QuestionType.LONG_ANSWER, description="LONG_ANSWER or MCQ")
    question_text: str
    student_answer: str
    max_marks: float = Field(default=5.0, ge=0.0)
    rubric: Any = Field(description="Rubric criteria string for long answer, or dict with correct_option for MCQ")
    required_keywords: List[str] = Field(default_factory=list, description="Keywords used for deterministic confidence scoring")
    subject: Optional[str] = "Biology"
    topic: Optional[str] = "General Biology"

class SubmissionResponse(BaseModel):
    submission_id: str
    student_id: str
    question_id: str
    assigned_score: float
    max_marks: float
    confidence_score: float
    feedback: str
    final_status: str  # e.g., "AUTO_APPROVED", "NEEDS_HUMAN_REVIEW", "COMPLETED"
    flagged_for_human_review: bool
    flag_reason: Optional[str] = None
    pii_detected: Optional[bool] = False
    sanitized_answer: Optional[str] = None
    key_points_matched: List[str] = Field(default_factory=list)
    key_points_missing: List[str] = Field(default_factory=list)
    retrieved_context: List[str] = Field(default_factory=list)

class QuestionCreateRequest(BaseModel):
    subject: str
    topic: str
    question_text: str
    question_type: QuestionType = QuestionType.LONG_ANSWER
    max_marks: float = 5.0
    rubric: Any
    required_keywords: List[str] = Field(default_factory=list)

class HumanReviewItem(BaseModel):
    submission_id: str
    student_id: str
    question_text: str
    student_answer: str
    sanitized_answer: Optional[str] = None
    ai_grade: str
    ai_feedback: str
    confidence_score: float
    flag_reason: Optional[str] = None
    status: str = "PENDING_REVIEW"