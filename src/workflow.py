# %% [imports]
import sys
import uuid
import json
import re
from pathlib import Path
from typing import TypedDict, List, Optional, Dict, Any

# Fix relative import paths for standalone and interactive runs
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langgraph.graph import StateGraph, END
from ollama import chat
from pydantic import BaseModel, Field

from src.vector_store import TextbookVectorStore
from src.database import init_db, log_grading_run, seed_exam_questions, seed_default_users
from src.utils.sanitizer import sanitize_student_answer
from src.utils.constants import QuestionType, UserRole
from src.utils.constants import PROMPT_INJECTION_PATTERNS, PLEA_DETECTION_PATTERNS


# %% [schemas]
class EvaluationOutput(BaseModel):
    score: float = Field(description="Assigned mark out of max_marks")
    max_marks: float = Field(description="Maximum possible marks")
    llm_self_confidence: float = Field(description="LLM self-reported confidence score between 0.0 and 1.0 or percentage")
    key_points_matched: List[str] = Field(default_factory=list, description="Rubric points the student correctly hit")
    key_points_missing: List[str] = Field(default_factory=list, description="Rubric points the student missed")
    feedback: str = Field(description="Constructive justification for the score")


class GradingState(TypedDict):
    submission_id: str
    student_id: str
    question_id: str
    question_type: str  # QuestionType.MCQ vs QuestionType.LONG_ANSWER
    question_text: str
    subject: Optional[str]
    topic: Optional[str]
    max_marks: float
    official_rubric: Any  # Can be JSON dict (for MCQ) or string (for Long Answer)
    required_keywords: List[str]
    student_answer: str
    
    # Security/Defensive State
    sanitized_answer: Optional[str]
    pii_detected: Optional[bool]
    is_injection_attempt: Optional[bool]
    flag_reason: Optional[str]
    has_plea_attempt: Optional[bool]
    
    rag_context: Optional[str]
    raw_eval: Optional[dict]
    composite_confidence: Optional[float]
    final_status: Optional[str]


# %% [confidence_tool]
def calculate_deterministic_confidence(
    max_marks: float,
    assigned_score: float,
    key_points_matched: List[str],
    key_points_missing: List[str],
    student_answer: str,
    required_keywords: List[str],
    llm_self_confidence: float
) -> float:
    if llm_self_confidence > 1.0:
        llm_self_confidence = llm_self_confidence / 100.0
    llm_self_confidence = max(0.0, min(1.0, llm_self_confidence))

    total_rubric_items = len(key_points_matched) + len(key_points_missing)
    rubric_clarity = (len(key_points_matched) / total_rubric_items) if total_rubric_items > 0 else 0.5

    matched_keywords = [
        kw for kw in required_keywords 
        if re.search(r'\b' + re.escape(kw) + r'\b', student_answer, re.IGNORECASE)
    ]
    keyword_coverage = len(matched_keywords) / len(required_keywords) if required_keywords else 1.0

    score_ratio = assigned_score / max_marks if max_marks > 0 else 0.0
    decisiveness = 1.0 if (score_ratio == 1.0 or score_ratio == 0.0) else 0.75

    composite_confidence = (
        (0.40 * rubric_clarity) +
        (0.30 * keyword_coverage) +
        (0.15 * decisiveness) +
        (0.15 * llm_self_confidence)
    )

    return round(composite_confidence, 2)


# %% [prompt_injection_guard]
def detect_prompt_injection(text: str) -> bool:
    if not text:
        return False
    return any(pattern.search(text) for pattern in PROMPT_INJECTION_PATTERNS)


# %% [plea_detection_patterns]
def detect_plea_attempt(text: str) -> bool:
    """Detects emotional appeals or pleas for marks in student answers."""
    if not text:
        return False
    return any(pattern.search(text) for pattern in PLEA_DETECTION_PATTERNS)


# %% [nodes]
vector_store = TextbookVectorStore()

def sanitize_input_node(state: GradingState) -> dict:
    """Node 1: Redacts student PII while preserving domain allowlisted terms."""
    print("🧹 [Node 1: Sanitizer] Running Presidio PII check...")
    raw_answer = state.get("student_answer", "")
    q_text = state.get("question_text", "")
    rubric = str(state.get("official_rubric", ""))
    
    sanitized, pii_found = sanitize_student_answer(
        student_answer=raw_answer,
        question_text=q_text,
        rubric_text=rubric,
        rag_context=""
    )
    return {
        "sanitized_answer": sanitized,
        "pii_detected": pii_found
    }


def injection_guard_node(state: GradingState) -> dict:
    """Node 2: Detects adversarial prompt injection attempts."""
    print("🛡️ [Node 2: Injection Guard] Scanning for prompt jailbreaks...")
    text_to_check = state.get("sanitized_answer") or state.get("student_answer", "")
    is_injection = detect_prompt_injection(text_to_check)
    
    if is_injection:
        print("⚠️ Prompt injection attempt flagged!")
        return {
            "is_injection_attempt": True,
            "final_status": "NEEDS_HUMAN_REVIEW",
            "flag_reason": "SUSPECTED_PROMPT_INJECTION"
        }
    return {"is_injection_attempt": False}


def plea_detector_node(state: GradingState) -> dict:
    """Node 2B: Detects emotional pleas or begging in student submissions."""
    print("🥺 [Node 2B: Plea Detector] Scanning for emotional appeals/pleas...")
    text_to_check = state.get("sanitized_answer") or state.get("student_answer", "")
    is_plea = detect_plea_attempt(text_to_check)
    
    if is_plea:
        print("⚠️ Emotional plea detected in submission!")
        return {
            "has_plea_attempt": True,
            "final_status": "NEEDS_HUMAN_REVIEW",
            "flag_reason": "STUDENT_PLEA_DETECTED"
        }
    return {"has_plea_attempt": False}


def evaluate_mcq_node(state: GradingState) -> dict:
    """Node 3A: Instant, deterministic evaluation for Multiple Choice Questions."""
    print("⚡ [Node 3A: MCQ Evaluator] Deterministically matching option key...")
    student_choice = (state.get("sanitized_answer") or state.get("student_answer") or "").strip().upper()
    rubric = state.get("official_rubric") or {}
    
    correct_option = ""
    if isinstance(rubric, dict):
        correct_option = str(rubric.get("correct_option", "")).strip().upper()
    elif isinstance(rubric, str):
        match = re.search(r'correct_option["\']?\s*:\s*["\']?([A-D])', rubric, re.IGNORECASE)
        if match:
            correct_option = match.group(1).upper()

    max_marks = float(state.get("max_marks", 1.0))
    is_correct = (student_choice == correct_option) and len(student_choice) > 0
    assigned_score = max_marks if is_correct else 0.0

    raw_eval = {
        "score": assigned_score,
        "max_marks": max_marks,
        "llm_self_confidence": 1.0,
        "key_points_matched": [f"Option {student_choice}"] if is_correct else [],
        "key_points_missing": [f"Option {correct_option}"] if not is_correct else [],
        "feedback": f"Selected option '{student_choice}'. Correct option was '{correct_option}'."
    }

    return {
        "raw_eval": raw_eval,
        "composite_confidence": 1.0,
        "final_status": "COMPLETED"
    }


def retrieve_context_node(state: GradingState) -> dict:
    """Node 3B: Calibrated RAG Context Retrieval from Vector Store."""
    print("📚 [Node 3B: RAG Context] Retrieving reference context...")
    
    # Fetch using calibrated 0.45 max_distance threshold and metadata fallback
    context = vector_store.query_context(
        question_text=state["question_text"],
        subject=state.get("subject"),
        topic=state.get("topic"),
        max_distance=0.45
    )
    return {"rag_context": context}


def evaluate_answer_node(state: GradingState) -> dict:
    """Node 4: LLM Semantic Evaluation using qwen3:8b."""
    print("🤖 [Node 4: LLM Evaluator] Evaluating semantic concepts using qwen3:8b...")
    
    answer_to_grade = state.get("sanitized_answer") or state.get("student_answer")
    prompt = f"""
You are an academic exam evaluator. Grade the student answer strictly based on the rubric and context.

QUESTION: {state['question_text']}
MAX MARKS: {state['max_marks']}

TEXTBOOK CONTEXT:
{state.get('rag_context', 'No textbook context provided.')}

OFFICIAL RUBRIC:
{state['official_rubric']}

STUDENT ANSWER:
{answer_to_grade}

Evaluate step-by-step and output your verdict matching the schema.
"""
    response = chat(
        model="qwen3:8b",
        messages=[
            {"role": "system", "content": "You are a precise grading system that outputs strictly structured JSON."},
            {"role": "user", "content": prompt}
        ],
        format=EvaluationOutput.model_json_schema(),
        options={"temperature": 0.0}
    )
    
    raw_content = response.message.content.strip()
    # Strip markdown codeblocks if model encloses output in standard json blocks
    if raw_content.startswith("```"):
        raw_content = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_content, flags=re.MULTILINE)

    try:
        parsed_eval = EvaluationOutput.model_validate_json(raw_content)
        raw_eval = parsed_eval.model_dump()
    except Exception as e:
        print(f"⚠️ Pydantic parsing failed: {e}. Falling back to raw json load...")
        raw_eval = json.loads(raw_content)

    return {"raw_eval": raw_eval}


def compute_confidence_node(state: GradingState) -> dict:
    """Node 5: Deterministic confidence scoring on LLM evaluation."""
    raw = state["raw_eval"]
    answer_for_confidence = state.get("sanitized_answer") or state.get("student_answer", "")
    
    score = calculate_deterministic_confidence(
        max_marks=state["max_marks"],
        assigned_score=raw["score"],
        key_points_matched=raw.get("key_points_matched", []),
        key_points_missing=raw.get("key_points_missing", []),
        student_answer=answer_for_confidence,
        required_keywords=state.get("required_keywords", []),
        llm_self_confidence=raw.get("llm_self_confidence", 0.85)
    )
    return {"composite_confidence": score}


def auto_approve_node(state: GradingState) -> dict:
    print("🟢 [AUTO_APPROVE] Evaluation passed deterministic confidence check.")
    return {"final_status": "AUTO_APPROVED"}


def human_review_node(state: GradingState) -> dict:
    print("🟡 [REQUIRES_HUMAN_REVIEW] Low confidence or security flag detected. Flagged for review.")
    return {"final_status": "NEEDS_HUMAN_REVIEW"}


def log_to_db_node(state: GradingState) -> dict:
    """Final Node: Persists evaluation state and review records into DB."""
    print("💾 [DB Logger] Persisting run execution log...")
    try:
        log_grading_run(state, submission_id=state["submission_id"])
    except Exception as e:
        print(f"⚠️ Error logging grading run to DB: {e}")
    return {}


# %% [routers]
def security_and_type_router(state: GradingState) -> str:
    """Routes based on injection detection, plea detection, and question type."""
    if state.get("is_injection_attempt") or state.get("has_plea_attempt"):
        return "human_review"
    
    q_type = state.get("question_type", QuestionType.LONG_ANSWER)
    if q_type == QuestionType.MCQ:
        return "evaluate_mcq"
        
    return "retrieve_context"


def confidence_router(state: GradingState) -> str:
    if state.get("composite_confidence", 0.0) >= 0.80:
        return "auto_approve"
    return "human_review"


# %% [graph_builder]
builder = StateGraph(GradingState)

# Register All Nodes
builder.add_node("sanitize_input", sanitize_input_node)
builder.add_node("injection_guard", injection_guard_node)
builder.add_node("plea_detector", plea_detector_node)
builder.add_node("evaluate_mcq", evaluate_mcq_node)
builder.add_node("retrieve_context", retrieve_context_node)
builder.add_node("evaluate_answer", evaluate_answer_node)
builder.add_node("compute_confidence", compute_confidence_node)
builder.add_node("auto_approve", auto_approve_node)
builder.add_node("human_review", human_review_node)
builder.add_node("log_to_db", log_to_db_node)

# Entry Point & Sequential Guard Edges
builder.set_entry_point("sanitize_input")
builder.add_edge("sanitize_input", "injection_guard")
builder.add_edge("injection_guard", "plea_detector")

# Conditional Router 1: Security Scan & Question Type Route
builder.add_conditional_edges(
    "plea_detector",
    security_and_type_router,
    {
        "human_review": "human_review",
        "evaluate_mcq": "evaluate_mcq",
        "retrieve_context": "retrieve_context"
    }
)

# MCQ Branch -> Log to DB -> End
builder.add_edge("evaluate_mcq", "log_to_db")

# Long Answer Branch -> RAG -> LLM -> Confidence
builder.add_edge("retrieve_context", "evaluate_answer")
builder.add_edge("evaluate_answer", "compute_confidence")

# Conditional Router 2: Confidence Threshold Route
builder.add_conditional_edges(
    "compute_confidence",
    confidence_router,
    {
        "auto_approve": "auto_approve",
        "human_review": "human_review"
    }
)

# Approval/Review Status -> Log to DB -> End
builder.add_edge("auto_approve", "log_to_db")
builder.add_edge("human_review", "log_to_db")
builder.add_edge("log_to_db", END)

grading_workflow = builder.compile()