# %% [imports]
import sys
import os
import uuid
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sqlalchemy import (
    create_engine, Column, String, Float, DateTime, Text, JSON, ForeignKey, Numeric, Integer
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID, JSONB
from sqlalchemy.types import TypeDecorator, CHAR
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

# %% [database_config]
DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "sqlite:///" + str(PROJECT_ROOT / "eval_ai.db")
)

engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Portable UUID type (PostgreSQL native UUID, SQLite fallback CHAR(36))
class GUID(TypeDecorator):
    impl = CHAR
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(PG_UUID(as_uuid=True))
        return dialect.type_descriptor(CHAR(36))

    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        if isinstance(value, uuid.UUID):
            return str(value) if dialect.name != "postgresql" else value
        return str(uuid.UUID(value)) if dialect.name != "postgresql" else uuid.UUID(value)

    def process_result_value(self, value, dialect):
        if value is None:
            return value
        if not isinstance(value, uuid.UUID):
            return uuid.UUID(value)
        return value

# %% [models]
class ExamQuestion(Base):
    """1. QUESTIONS & RUBRICS"""
    __tablename__ = "exam_questions"

    question_id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    subject = Column(String(100), nullable=False)
    topic = Column(String(150), nullable=False)
    question_text = Column(Text, nullable=False)
    max_marks = Column(Integer, nullable=False, default=10)
    official_rubric = Column(JSON, nullable=False)  # JSONB on PG
    vector_chunk_id = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    submissions = relationship("StudentSubmission", back_populates="question", cascade="all, delete-orphan")


class StudentSubmission(Base):
    """2. STUDENT SUBMISSIONS"""
    __tablename__ = "student_submissions"

    submission_id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    student_id = Column(GUID(), nullable=False)
    question_id = Column(GUID(), ForeignKey("exam_questions.question_id", ondelete="CASCADE"), nullable=True)
    typed_answer = Column(Text, nullable=False)
    submitted_at = Column(DateTime, default=datetime.utcnow)
    processing_status = Column(String(50), default="PENDING")

    # Relationships
    question = relationship("ExamQuestion", back_populates="submissions")
    evaluations = relationship("EvaluationResult", back_populates="submission", cascade="all, delete-orphan")
    reviews = relationship("HumanReview", back_populates="submission", cascade="all, delete-orphan")


class EvaluationResult(Base):
    """3. AI EVALUATION LOGS"""
    __tablename__ = "evaluation_results"

    evaluation_id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    submission_id = Column(GUID(), ForeignKey("student_submissions.submission_id", ondelete="CASCADE"), nullable=False)
    ai_score = Column(Numeric(4, 2), nullable=True)
    ai_confidence = Column(Numeric(3, 2), nullable=True)
    ai_reasoning_feedback = Column(Text, nullable=True)
    model_version = Column(String(50), default="qwen3:8b")
    retrieved_rag_context = Column(Text, nullable=True)
    evaluated_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    submission = relationship("StudentSubmission", back_populates="evaluations")
    human_reviews = relationship("HumanReview", back_populates="evaluation")


class HumanReview(Base):
    """4. HUMAN-IN-THE-LOOP OVERRIDE QUEUE"""
    __tablename__ = "human_reviews"

    review_id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    submission_id = Column(GUID(), ForeignKey("student_submissions.submission_id", ondelete="CASCADE"), nullable=False)
    evaluation_id = Column(GUID(), ForeignKey("evaluation_results.evaluation_id"), nullable=True)
    flag_reason = Column(String(100), nullable=False)
    reviewer_id = Column(GUID(), nullable=True)
    final_human_score = Column(Numeric(4, 2), nullable=True)
    reviewer_comments = Column(Text, nullable=True)
    review_status = Column(String(30), default="UNASSIGNED")
    reviewed_at = Column(DateTime, nullable=True)

    # Relationships
    submission = relationship("StudentSubmission", back_populates="reviews")
    evaluation = relationship("EvaluationResult", back_populates="human_reviews")


# %% [db_init]
def init_db():
    Base.metadata.create_all(bind=engine)
    print("✅ Database tables initialized matching DDL Specification (UUID PKs across all tables).")

# %% [add exam questions]
def seed_exam_questions(questions: List[Dict[str, Any]]) -> List[uuid.UUID]:
    """
    Seeds initial exam questions into the exam_questions table.
    """
    session = SessionLocal()
    question_ids = []
    try:
        for q in questions:
            q_id = uuid.UUID(q["question_id"]) if isinstance(q.get("question_id"), str) else q.get("question_id", uuid.uuid4())
            existing = session.query(ExamQuestion).filter_by(question_id=q_id).first()
            if not existing:
                question_obj = ExamQuestion(
                    question_id=q_id,
                    subject=q.get("subject", "General Science"),
                    topic=q.get("topic", "Biology"),
                    question_text=q["question_text"],
                    max_marks=int(q["max_marks"]),
                    official_rubric=q["official_rubric"],
                    vector_chunk_id=q.get("vector_chunk_id")
                )
                session.add(question_obj)
                question_ids.append(q_id)
            else:
                question_ids.append(existing.question_id)
        session.commit()
        print(f"✅ Seeded {len(question_ids)} exam question(s) into database.")
        return question_ids
    except Exception as e:
        session.rollback()
        print(f"❌ Error seeding exam questions: {e}")
        return []
    finally:
        session.close()


# %% [persistence_helpers]
def log_grading_run(state: Dict[str, Any], submission_id: Optional[Any] = None) -> Optional[uuid.UUID]:
    """
    Persists evaluation results to student_submissions, evaluation_results, and human_reviews.
    """
    session = SessionLocal()
    try:
        # Cast submission_id
        sub_id = state.get("submission_id") or submission_id
        if isinstance(sub_id, str):
            sub_id = uuid.UUID(sub_id)
        elif not sub_id:
            sub_id = uuid.uuid4()

        student_id = state.get("student_id")
        if isinstance(student_id, str):
            student_id = uuid.UUID(student_id)
        elif not student_id:
            student_id = uuid.uuid4()

        q_id = state.get("question_id")
        if isinstance(q_id, str):
            q_id = uuid.UUID(q_id)

        final_status = state.get("final_status", "PENDING")

        # 1. Ensure exam_questions record exists or link FK
        question_ref = None
        if q_id:
            question_ref = session.query(ExamQuestion).filter_by(question_id=q_id).first()
            if not question_ref:
                # Upsert fallback for dynamically passed questions
                question_ref = ExamQuestion(
                    question_id=q_id,
                    subject=state.get("subject", "Biology"),
                    topic=state.get("topic", "Cellular Processes"),
                    question_text=state.get("question_text", ""),
                    max_marks=int(state.get("max_marks", 10)),
                    official_rubric={"text": state.get("official_rubric", "")}
                )
                session.add(question_ref)
                session.flush()

        # 2. Save Student Submission Record with FK
        submission = session.query(StudentSubmission).filter_by(submission_id=sub_id).first()
        if not submission:
            submission = StudentSubmission(
                submission_id=sub_id,
                student_id=student_id,
                question_id=question_ref.question_id if question_ref else None,
                typed_answer=state.get("student_answer", ""),
                processing_status=final_status
            )
            session.add(submission)
        else:
            submission.processing_status = final_status

        session.flush()

        # 3. Save AI Evaluation Log
        raw_eval = state.get("raw_eval") or {}
        eval_record = EvaluationResult(
            evaluation_id=uuid.uuid4(),
            submission_id=submission.submission_id,
            ai_score=raw_eval.get("score"),
            ai_confidence=state.get("composite_confidence"),
            ai_reasoning_feedback=raw_eval.get("feedback"),
            model_version="qwen3:8b",
            retrieved_rag_context=state.get("rag_context")
        )
        session.add(eval_record)
        session.flush()

        # 4. Queue Human Review if Flagged
        if final_status == "NEEDS_HUMAN_REVIEW":
            review_entry = HumanReview(
                review_id=uuid.uuid4(),
                submission_id=submission.submission_id,
                evaluation_id=eval_record.evaluation_id,
                flag_reason="LOW_CONFIDENCE_SCORE",
                review_status="UNASSIGNED"
            )
            session.add(review_entry)

        session.commit()
        print(f"💾 Logged Evaluation '{eval_record.evaluation_id}' for Submission '{submission.submission_id}' [{final_status}]")
        return eval_record.evaluation_id

    except Exception as e:
        session.rollback()
        print(f"❌ Database error while logging evaluation: {e}")
        return None
    finally:
        session.close()


# %% [test_execution]
if __name__ == "__main__":
    init_db()

    # Seeded batch of 10 realistic grading examples for smoke testing
    rng = __import__("random").Random(42)
    sample_workflow_states = [
        {
            "submission_id": str(uuid.uuid4()),
            "student_id": str(uuid.uuid4()),
            "question_id": str(uuid.uuid4()),
            "question_text": "What is the primary function of mitochondria in eukaryotic cells?",
            "max_marks": 5.0,
            "official_rubric": """
            1. Identifies mitochondria as the site of cellular respiration / ATP production (2 marks).
            2. Uses the term 'ATP' or 'adenosine triphosphate' (1 mark).
            3. Mentions converting nutrients/glucose into usable chemical energy (2 marks).
            """,
            "required_keywords": ["mitochondria", "ATP", "respiration", "glucose"],
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
        },
        {
            "submission_id": str(uuid.uuid4()),
            "student_id": str(uuid.uuid4()),
            "question_id": str(uuid.uuid4()),
            "question_text": "Explain how photosynthesis converts light energy into chemical energy.",
            "max_marks": 6.0,
            "official_rubric": """
            1. Identifies chloroplasts as the site of photosynthesis (1 mark).
            2. Mentions absorption of sunlight or light energy (1 mark).
            3. Explains conversion of carbon dioxide and water into glucose (2 marks).
            4. Notes oxygen is produced as a by-product (1 mark).
            5. Connects this to stored chemical energy in glucose (1 mark).
            """,
            "required_keywords": ["chloroplast", "sunlight", "glucose", "carbon dioxide", "water"],
            "student_answer": "Chloroplasts absorb sunlight and use it to turn carbon dioxide and water into glucose and oxygen.",
            "rag_context": "Photosynthesis converts light energy into glucose through chlorophyll-mediated reactions.",
            "raw_eval": {
                "score": 5.5,
                "max_marks": 6.0,
                "llm_self_confidence": 0.90,
                "key_points_matched": ["absorbs sunlight", "produces glucose", "uses carbon dioxide and water"],
                "key_points_missing": ["mention of oxygen as by-product"],
                "feedback": "Good explanation with one missing detail."
            },
            "composite_confidence": 0.88,
            "final_status": "AUTO_APPROVED"
        },
        {
            "submission_id": str(uuid.uuid4()),
            "student_id": str(uuid.uuid4()),
            "question_id": str(uuid.uuid4()),
            "question_text": "What is the role of ribosomes in a cell?",
            "max_marks": 4.0,
            "official_rubric": """
            1. Identifies ribosomes as sites of protein synthesis (2 marks).
            2. Mentions translation of mRNA (1 mark).
            3. Relates this to assembly of amino acids into proteins (1 mark).
            """,
            "required_keywords": ["ribosomes", "protein", "mRNA", "amino acids"],
            "student_answer": "Ribosomes are responsible for building proteins from amino acids.",
            "rag_context": "Ribosomes synthesize proteins by translating mRNA into polypeptide chains.",
            "raw_eval": {
                "score": 3.5,
                "max_marks": 4.0,
                "llm_self_confidence": 0.79,
                "key_points_matched": ["protein synthesis"],
                "key_points_missing": ["translation of mRNA"],
                "feedback": "Mostly correct, but the mechanism could be clearer."
            },
            "composite_confidence": 0.82,
            "final_status": "AUTO_APPROVED"
        },
        {
            "submission_id": str(uuid.uuid4()),
            "student_id": str(uuid.uuid4()),
            "question_id": str(uuid.uuid4()),
            "question_text": "Describe the function of the cell membrane.",
            "max_marks": 5.0,
            "official_rubric": """
            1. States it controls entry and exit of substances (2 marks).
            2. Mentions selective permeability / barrier function (1 mark).
            3. Notes communication or structural role (1 mark).
            4. Identifies phospholipid bilayer or membrane structure (1 mark).
            """,
            "required_keywords": ["membrane", "selective", "cell", "transport"],
            "student_answer": "It controls what enters and leaves the cell, protecting the contents and helping cells communicate.",
            "rag_context": "The plasma membrane is selectively permeable and regulates transport and signalling.",
            "raw_eval": {
                "score": 4.0,
                "max_marks": 5.0,
                "llm_self_confidence": 0.87,
                "key_points_matched": ["selective barrier", "controls movement"],
                "key_points_missing": ["structure of phospholipid bilayer"],
                "feedback": "Strong answer but missing the membrane structure detail."
            },
            "composite_confidence": 0.83,
            "final_status": "AUTO_APPROVED"
        },
        {
            "submission_id": str(uuid.uuid4()),
            "student_id": str(uuid.uuid4()),
            "question_id": str(uuid.uuid4()),
            "question_text": "How does the circulatory system support cellular respiration?",
            "max_marks": 6.0,
            "official_rubric": """
            1. Explains delivery of oxygen to tissues (2 marks).
            2. Mentions removal of carbon dioxide (1 mark).
            3. Connects supply of nutrients to metabolism and ATP production (2 marks).
            4. Relates this to cellular respiration efficiency (1 mark).
            """,
            "required_keywords": ["oxygen", "carbon dioxide", "ATP", "circulatory", "respiration"],
            "student_answer": "It delivers oxygen and removes carbon dioxide so cells can make energy.",
            "rag_context": "Blood transports oxygen, nutrients, and wastes to support metabolism and respiration.",
            "raw_eval": {
                "score": 4.0,
                "max_marks": 6.0,
                "llm_self_confidence": 0.68,
                "key_points_matched": ["oxygen delivery", "carbon dioxide removal"],
                "key_points_missing": ["nutrient transport", "link to ATP production"],
                "feedback": "Partially correct; the role in fuel delivery and energy production is incomplete."
            },
            "composite_confidence": 0.71,
            "final_status": "NEEDS_HUMAN_REVIEW"
        },
        {
            "submission_id": str(uuid.uuid4()),
            "student_id": str(uuid.uuid4()),
            "question_id": str(uuid.uuid4()),
            "question_text": "Explain why enzymes are important in metabolism.",
            "max_marks": 5.0,
            "official_rubric": """
            1. States enzymes speed up reactions (1 mark).
            2. Mentions they lower activation energy (1 mark).
            3. Connects this to metabolic pathways and cell function (2 marks).
            4. Applies to control of biochemical reactions (1 mark).
            """,
            "required_keywords": ["enzymes", "reaction", "activation", "metabolism"],
            "student_answer": "Enzymes speed up chemical reactions in the body and help maintain life processes.",
            "rag_context": "Enzymes lower activation energy and catalyze biochemical reactions in cells.",
            "raw_eval": {
                "score": 4.5,
                "max_marks": 5.0,
                "llm_self_confidence": 0.92,
                "key_points_matched": ["speed up reactions", "maintain metabolic processes"],
                "key_points_missing": ["lower activation energy"],
                "feedback": "Very good answer; only the activation-energy mechanism is missing."
            },
            "composite_confidence": 0.86,
            "final_status": "AUTO_APPROVED"
        },
        {
            "submission_id": str(uuid.uuid4()),
            "student_id": str(uuid.uuid4()),
            "question_id": str(uuid.uuid4()),
            "question_text": "State the significance of meiosis in sexual reproduction.",
            "max_marks": 4.0,
            "official_rubric": """
            1. States meiosis halves chromosome number (2 marks).
            2. Explains gamete formation (1 mark).
            3. Links this to restoration of diploid number at fertilisation (1 mark).
            """,
            "required_keywords": ["meiosis", "chromosome", "gametes", "fertilisation"],
            "student_answer": "Meiosis creates gametes with half the number of chromosomes so fertilisation restores the diploid number.",
            "rag_context": "Meiosis reduces chromosome number and introduces genetic variation in gametes.",
            "raw_eval": {
                "score": 4.0,
                "max_marks": 4.0,
                "llm_self_confidence": 0.96,
                "key_points_matched": ["halves chromosome number", "supports fertilisation"],
                "key_points_missing": [],
                "feedback": "Excellent explanation."
            },
            "composite_confidence": 0.98,
            "final_status": "AUTO_APPROVED"
        },
        {
            "submission_id": str(uuid.uuid4()),
            "student_id": str(uuid.uuid4()),
            "question_id": str(uuid.uuid4()),
            "question_text": "What happens to a plant cell in a hypertonic solution?",
            "max_marks": 5.0,
            "official_rubric": """
            1. States water leaves the cell by osmosis (2 marks).
            2. Mentions reduced turgor or cell shrinkage (1 mark).
            3. Identifies plasmolysis or wilting (2 marks).
            """,
            "required_keywords": ["hypertonic", "osmosis", "water", "shrink"],
            "student_answer": "It loses water and shrinks because water moves out of the cell by osmosis.",
            "rag_context": "In a hypertonic solution, cells lose water and undergo plasmolysis.",
            "raw_eval": {
                "score": 5.0,
                "max_marks": 5.0,
                "llm_self_confidence": 0.93,
                "key_points_matched": ["water leaves cell", "osmosis", "cell shrinks"],
                "key_points_missing": [],
                "feedback": "Correct and complete."
            },
            "composite_confidence": 0.96,
            "final_status": "AUTO_APPROVED"
        },
        {
            "submission_id": str(uuid.uuid4()),
            "student_id": str(uuid.uuid4()),
            "question_id": str(uuid.uuid4()),
            "question_text": "Describe the role of neurotransmitters in the nervous system.",
            "max_marks": 5.0,
            "official_rubric": """
            1. States neurotransmitters are chemical messengers (2 marks).
            2. Explains they cross synapses or transmit signals between neurons (2 marks).
            3. Mentions effect on target cells or receptors (1 mark).
            """,
            "required_keywords": ["neurotransmitters", "synapse", "signal", "neurons"],
            "student_answer": "They carry signals across synapses between neurons.",
            "rag_context": "Neurotransmitters are chemical messengers released into synapses to transmit signals.",
            "raw_eval": {
                "score": 3.0,
                "max_marks": 5.0,
                "llm_self_confidence": 0.62,
                "key_points_matched": ["cross synapse"],
                "key_points_missing": ["chemical messenger role", "signal transmission detail"],
                "feedback": "This is a partial answer and needs more detail on chemical signalling."
            },
            "composite_confidence": 0.66,
            "final_status": "NEEDS_HUMAN_REVIEW"
        },
        {
            "submission_id": str(uuid.uuid4()),
            "student_id": str(uuid.uuid4()),
            "question_id": str(uuid.uuid4()),
            "question_text": "Why is ATP important for cells?",
            "max_marks": 5.0,
            "official_rubric": """
            1. States ATP is the cell's immediate energy source (2 marks).
            2. Mentions it powers cellular processes such as transport, movement, or synthesis (2 marks).
            3. Relates this to energy transfer and metabolism (1 mark).
            """,
            "required_keywords": ["ATP", "energy", "cell", "processes"],
            "student_answer": "ATP stores and provides energy for cellular processes such as transport, movement and synthesis.",
            "rag_context": "ATP is the cell's immediate energy currency used in biosynthesis, transport, and movement.",
            "raw_eval": {
                "score": 5.0,
                "max_marks": 5.0,
                "llm_self_confidence": 0.94,
                "key_points_matched": ["energy currency", "supports cellular processes"],
                "key_points_missing": [],
                "feedback": "Clear and accurate explanation."
            },
            "composite_confidence": 0.97,
            "final_status": "AUTO_APPROVED"
        }
    ]

    rng.shuffle(sample_workflow_states)

    for index, sample in enumerate(sample_workflow_states, start=1):
        submission_id = sample["submission_id"]
        run_id = log_grading_run(sample, submission_id=submission_id)
        print(f"[{index}/10] {sample['question_id']} -> Run ID {run_id} ({sample['final_status']})")

    print("Test complete. Seeded batch of 10 workflow states saved to the database.")