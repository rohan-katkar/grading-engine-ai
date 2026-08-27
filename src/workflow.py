# %% [markdown]
# # LangGraph Grading Workflow Engine
# This notebook/script defines the state graph combining RAG context retrieval, 
# Qwen 8B evaluation, deterministic confidence calculations, and automated routing.

# %% [imports]
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import re
from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END
from ollama import chat
from pydantic import BaseModel, Field

from src.vector_store import TextbookVectorStore

# %% [schemas]
# 1. Define Pydantic Schema for LLM Grading Output
class EvaluationOutput(BaseModel):
    score: float = Field(description="Assigned mark out of max_marks")
    max_marks: float = Field(description="Maximum possible marks")
    llm_self_confidence: float = Field(description="LLM self-reported confidence score between 0.0 and 1.0 or percentage")
    key_points_matched: List[str] = Field(description="Rubric points the student correctly hit")
    key_points_missing: List[str] = Field(description="Rubric points the student missed")
    feedback: str = Field(description="Constructive justification for the score")

# 2. Define LangGraph State Schema
class GradingState(TypedDict):
    question_id: str
    question_text: str
    max_marks: float
    official_rubric: str
    required_keywords: List[str]
    student_answer: str
    
    # State fields populated during workflow execution
    rag_context: Optional[str]
    raw_eval: Optional[dict]
    composite_confidence: Optional[float]
    final_status: Optional[str]

# %% [confidence_tool]
# 3. Deterministic Confidence Calculation Tool
def calculate_deterministic_confidence(
    max_marks: float,
    assigned_score: float,
    key_points_matched: List[str],
    key_points_missing: List[str],
    student_answer: str,
    required_keywords: List[str],
    llm_self_confidence: float
) -> float:
    # Scale normalization
    if llm_self_confidence > 1.0:
        llm_self_confidence = llm_self_confidence / 100.0
    llm_self_confidence = max(0.0, min(1.0, llm_self_confidence))

    # Rubric Coverage Ratio
    total_rubric_items = len(key_points_matched) + len(key_points_missing)
    rubric_clarity = (len(key_points_matched) / total_rubric_items) if total_rubric_items > 0 else 0.5

    # Hard Keyword Verification
    matched_keywords = [
        kw for kw in required_keywords 
        if re.search(r'\b' + re.escape(kw) + r'\b', student_answer, re.IGNORECASE)
    ]
    keyword_coverage = len(matched_keywords) / len(required_keywords) if required_keywords else 1.0

    # Score Boundary Check
    score_ratio = assigned_score / max_marks if max_marks > 0 else 0.0
    decisiveness = 1.0 if (score_ratio == 1.0 or score_ratio == 0.0) else 0.75

    # Weighted Composite Formula
    composite_confidence = (
        (0.40 * rubric_clarity) +
        (0.30 * keyword_coverage) +
        (0.15 * decisiveness) +
        (0.15 * llm_self_confidence)
    )

    return round(composite_confidence, 2)

# %% [nodes]
# Initialize Vector Store Client
vector_store = TextbookVectorStore()

def retrieve_context_node(state: GradingState) -> dict:
    """Queries ChromaDB for context using the question text."""
    context = vector_store.query_context(state["question_text"])
    return {"rag_context": context}

def evaluate_answer_node(state: GradingState) -> dict:
    """Invokes Qwen 8B to evaluate student response against context & rubric."""
    prompt = f"""
You are an academic exam evaluator. Grade the student answer strictly based on the rubric and context.

QUESTION: {state['question_text']}
MAX MARKS: {state['max_marks']}

TEXTBOOK CONTEXT:
{state['rag_context']}

OFFICIAL RUBRIC:
{state['official_rubric']}

STUDENT ANSWER:
{state['student_answer']}

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
    
    raw_eval = json.loads(response.message.content)
    return {"raw_eval": raw_eval}

def compute_confidence_node(state: GradingState) -> dict:
    """Runs the deterministic confidence calculator on LLM output."""
    raw = state["raw_eval"]
    score = calculate_deterministic_confidence(
        max_marks=state["max_marks"],
        assigned_score=raw["score"],
        key_points_matched=raw["key_points_matched"],
        key_points_missing=raw["key_points_missing"],
        student_answer=state["student_answer"],
        required_keywords=state["required_keywords"],
        llm_self_confidence=raw["llm_self_confidence"]
    )
    return {"composite_confidence": score}

def auto_approve_node(state: GradingState) -> dict:
    """Finalizes evaluation for high-confidence output."""
    print("\n🟢 [AUTO_APPROVE] Evaluation passed deterministic confidence check.")
    return {"final_status": "AUTO_APPROVED"}

def human_review_node(state: GradingState) -> dict:
    """Flags evaluation for human review queue."""
    print("\n🟡 [REQUIRES_HUMAN_REVIEW] Low confidence or edge case detected. Flagged for review.")
    return {"final_status": "NEEDS_HUMAN_REVIEW"}

# %% [router]
def confidence_router(state: GradingState) -> str:
    """Routes state based on composite confidence threshold."""
    if state["composite_confidence"] >= 0.80:
        return "auto_approve"
    return "human_review"

# %% [graph_builder]
builder = StateGraph(GradingState)

# Add Nodes
builder.add_node("retrieve_context", retrieve_context_node)
builder.add_node("evaluate_answer", evaluate_answer_node)
builder.add_node("compute_confidence", compute_confidence_node)
builder.add_node("auto_approve", auto_approve_node)
builder.add_node("human_review", human_review_node)

# Set Entry Point & Edges
builder.set_entry_point("retrieve_context")
builder.add_edge("retrieve_context", "evaluate_answer")
builder.add_edge("evaluate_answer", "compute_confidence")

# Add Conditional Routing Edge
builder.add_conditional_edges(
    "compute_confidence",
    confidence_router,
    {
        "auto_approve": "auto_approve",
        "human_review": "human_review"
    }
)

builder.add_edge("auto_approve", END)
builder.add_edge("human_review", END)

# Compile Graph
grading_workflow = builder.compile()

# %% [test_execution]
if __name__ == "__main__":
    # Ensure ChromaDB has seed data
    vector_store.seed_data()

    # Test Sample Submission
    sample_input = {
        "question_id": "Q101",
        "question_text": "What is the primary function of mitochondria in eukaryotic cells?",
        "max_marks": 5.0,
        "official_rubric": """
        1. Identifies mitochondria as site of cellular respiration / ATP production (2 marks).
        2. Uses the term 'ATP' or 'adenosine triphosphate' (1 mark).
        3. Mentions converting nutrients/glucose into usable chemical energy (2 marks).
        """,
        "required_keywords": ["mitochondria", "ATP", "respiration", "glucose"],
        "student_answer": "Mitochondria produce ATP by breaking down glucose during cellular respiration."
    }

    print("\n🚀 Executing LangGraph Workflow...")
    final_state = grading_workflow.invoke(sample_input)

    print("\n=== FINAL WORKFLOW SUMMARY ===")
    print(f"Status              : {final_state['final_status']}")
    print(f"Assigned Score      : {final_state['raw_eval']['score']} / {final_state['max_marks']}")
    print(f"Composite Confidence: {final_state['composite_confidence']}")
    print(f"Feedback            : {final_state['raw_eval']['feedback']}")