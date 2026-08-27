# %% [imports]
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sqlalchemy import (
    create_engine, Column, String, Float, DateTime, Text, JSON, ForeignKey, Integer
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

# %% [database_config]
# Default to SQLite for quick local development if POSTGRES_URL isn't set
DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "sqlite:///" + str(PROJECT_ROOT / "eval_ai.db")  # e.g., "postgresql://user:password@localhost:5432/eval_ai_db"
)

engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# %% [models]
class StudentSubmission(Base):
    """Stores incoming student exam submissions."""
    __tablename__ = "student_submissions"

    id = Column(String, primary_key=True)
    question_id = Column(String, nullable=False, index=True)
    question_text = Column(Text, nullable=False)
    student_answer = Column(Text, nullable=False)
    max_marks = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    grading_runs = relationship("GradingRun", back_populates="submission")


class GradingRun(Base):
    """Audit log for each execution of the LangGraph grading workflow."""
    __tablename__ = "grading_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    submission_id = Column(String, ForeignKey("student_submissions.id"), nullable=False, index=True)
    
    # Context & Scores
    rag_context = Column(Text, nullable=True)
    assigned_score = Column(Float, nullable=True)
    max_marks = Column(Float, nullable=False)
    composite_confidence = Column(Float, nullable=True)
    llm_self_confidence = Column(Float, nullable=True)
    
    # Detailed Outputs
    status = Column(String, nullable=False)  # e.g., 'AUTO_APPROVED', 'NEEDS_HUMAN_REVIEW'
    raw_llm_output = Column(JSON, nullable=True)
    feedback = Column(Text, nullable=True)
    
    evaluated_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    submission = relationship("StudentSubmission", back_populates="grading_runs")
    audit_overrides = relationship("HumanAuditLog", back_populates="grading_run")


class HumanAuditLog(Base):
    """Tracks manual reviews and overrides by human educators."""
    __tablename__ = "human_audit_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    grading_run_id = Column(Integer, ForeignKey("grading_runs.id"), nullable=False, index=True)
    reviewer_id = Column(String, nullable=False)
    original_score = Column(Float, nullable=False)
    overridden_score = Column(Float, nullable=False)
    reason_for_override = Column(Text, nullable=False)
    reviewed_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    grading_run = relationship("GradingRun", back_populates="audit_overrides")


# %% [db_init]
def init_db():
    """Creates database tables if they do not exist."""
    Base.metadata.create_all(bind=engine)
    print("✅ Database tables initialized successfully.")


# %% [persistence_helpers]
def log_grading_run(state: Dict[str, Any], submission_id: str) -> Optional[int]:
    """
    Persists a completed or flagged LangGraph workflow state to the database.
    """
    session = SessionLocal()
    try:
        # 1. Upsert Student Submission record
        submission = session.query(StudentSubmission).filter_by(id=submission_id).first()
        if not submission:
            submission = StudentSubmission(
                id=submission_id,
                question_id=state.get("question_id", "Q_UNKNOWN"),
                question_text=state.get("question_text", ""),
                student_answer=state.get("student_answer", ""),
                max_marks=state.get("max_marks", 0.0)
            )
            session.add(submission)
            session.flush()

        # 2. Parse Raw LLM Output safely
        raw_eval = state.get("raw_eval") or {}
        
        # 3. Save Grading Run Record
        run = GradingRun(
            submission_id=submission.id,
            rag_context=state.get("rag_context"),
            assigned_score=raw_eval.get("score"),
            max_marks=state.get("max_marks", 0.0),
            composite_confidence=state.get("composite_confidence"),
            llm_self_confidence=raw_eval.get("llm_self_confidence"),
            status=state.get("final_status", "PENDING"),
            raw_llm_output=raw_eval,
            feedback=raw_eval.get("feedback")
        )
        session.add(run)
        session.commit()
        session.refresh(run)
        
        print(f"💾 Logged Grading Run ID #{run.id} for Submission '{submission_id}' (Status: {run.status}).")
        return run.id

    except Exception as e:
        session.rollback()
        print(f"❌ Database error while saving grading run: {e}")
        return None
    finally:
        session.close()


# %% [test_execution]
if __name__ == "__main__":
    init_db()

    # Test saving a dummy state object
    test_submission_id = "SUB_TEST_101"
    sample_workflow_state = {
        "question_id": "Q101",
        "question_text": "What is the primary function of mitochondria in eukaryotic cells?",
        "max_marks": 5.0,
        "student_answer": "Mitochondria produce ATP by breaking down glucose during cellular respiration.",
        "rag_context": "Mitochondria generate ATP through oxidative phosphorylation.",
        "raw_eval": {
            "score": 5.0,
            "max_marks": 5.0,
            "llm_self_confidence": 0.95,
            "key_points_matched": ["site of cellular respiration", "uses ATP"],
            "key_points_missing": [],
            "feedback": "Correct explanation."
        },
        "composite_confidence": 0.99,
        "final_status": "AUTO_APPROVED"
    }

    run_id = log_grading_run(sample_workflow_state, submission_id=test_submission_id)
    print(f"Test complete. Created Run ID: {run_id}")