# %% [imports]
import sys
import uuid
from pathlib import Path

# Fix relative import paths for standalone and interactive runs
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
from src.database import init_db, log_grading_run, seed_exam_questions

# %% [schemas]
class EvaluationOutput(BaseModel):
    score: float = Field(description="Assigned mark out of max_marks")
    max_marks: float = Field(description="Maximum possible marks")
    llm_self_confidence: float = Field(description="LLM self-reported confidence score between 0.0 and 1.0 or percentage")
    key_points_matched: List[str] = Field(description="Rubric points the student correctly hit")
    key_points_missing: List[str] = Field(description="Rubric points the student missed")
    feedback: str = Field(description="Constructive justification for the score")

class GradingState(TypedDict):
    submission_id: str
    student_id: str
    question_id: str
    question_text: str
    max_marks: float
    official_rubric: str
    required_keywords: List[str]
    student_answer: str
    
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

# %% [nodes]
vector_store = TextbookVectorStore()

def retrieve_context_node(state: GradingState) -> dict:
    context = vector_store.query_context(state["question_text"])
    return {"rag_context": context}

def evaluate_answer_node(state: GradingState) -> dict:
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
    print("\n🟢 [AUTO_APPROVE] Evaluation passed deterministic confidence check.")
    status = "AUTO_APPROVED"
    updated_state = {**state, "final_status": status}
    log_grading_run(updated_state, submission_id=state["submission_id"])
    return {"final_status": status}

def human_review_node(state: GradingState) -> dict:
    print("\n🟡 [REQUIRES_HUMAN_REVIEW] Low confidence or edge case detected. Flagged for review.")
    status = "NEEDS_HUMAN_REVIEW"
    updated_state = {**state, "final_status": status}
    log_grading_run(updated_state, submission_id=state["submission_id"])
    return {"final_status": status}

# %% [router]
def confidence_router(state: GradingState) -> str:
    if state["composite_confidence"] >= 0.80:
        return "auto_approve"
    return "human_review"

# %% [graph_builder]
builder = StateGraph(GradingState)

builder.add_node("retrieve_context", retrieve_context_node)
builder.add_node("evaluate_answer", evaluate_answer_node)
builder.add_node("compute_confidence", compute_confidence_node)
builder.add_node("auto_approve", auto_approve_node)
builder.add_node("human_review", human_review_node)

builder.set_entry_point("retrieve_context")
builder.add_edge("retrieve_context", "evaluate_answer")
builder.add_edge("evaluate_answer", "compute_confidence")

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

grading_workflow = builder.compile()

# %% [test_execution]
if __name__ == "__main__":
    init_db()
    vector_store.seed_data()

    # Pre-seed question bank into PostgreSQL / SQLite
    sample_questions = [
        {
            "question_id": str(uuid.uuid5(uuid.NAMESPACE_DNS, "q101")),
            "subject": "Biology",
            "topic": "Cell Biology",
            "question_text": "What is the primary function of mitochondria in eukaryotic cells?",
            "max_marks": 5.0,
            "official_rubric": {
                "criteria": [
                    "Identifies mitochondria as site of cellular respiration / ATP production (2 marks)",
                    "Uses the term 'ATP' or 'adenosine triphosphate' (1 mark)",
                    "Mentions converting nutrients/glucose into usable chemical energy (2 marks)"
                ]
            }
        },
        {
            "question_id": str(uuid.uuid5(uuid.NAMESPACE_DNS, "q102")),
            "subject": "Biology",
            "topic": "Plant Physiology",
            "question_text": "Explain how photosynthesis converts light energy into chemical energy.",
            "max_marks": 6.0,
            "official_rubric": {
                "criteria": [
                    "Identifies chloroplasts as the site of photosynthesis (1 mark)",
                    "Mentions absorption of sunlight or light energy (1 mark)",
                    "Explains conversion of carbon dioxide and water into glucose (2 marks)",
                    "Notes oxygen is produced as a by-product (1 mark)",
                    "Connects this to stored chemical energy in glucose (1 mark)"
                ]
            }
        },
        {
            "question_id": str(uuid.uuid5(uuid.NAMESPACE_DNS, "q103")),
            "subject": "Biology",
            "topic": "Cell Structure",
            "question_text": "What is the role of ribosomes in a cell?",
            "max_marks": 4.0,
            "official_rubric": {
                "criteria": [
                    "Identifies ribosomes as sites of protein synthesis (2 marks)",
                    "Mentions translation of mRNA (1 mark)",
                    "Relates this to assembly of amino acids into proteins (1 mark)"
                ]
            }
        },
        {
            "question_id": str(uuid.uuid5(uuid.NAMESPACE_DNS, "q104")),
            "subject": "Biology",
            "topic": "Cell Membranes",
            "question_text": "Describe the function of the cell membrane.",
            "max_marks": 5.0,
            "official_rubric": {
                "criteria": [
                    "States it controls entry and exit of substances (2 marks)",
                    "Mentions selective permeability or barrier function (1 mark)",
                    "Notes communication or structural role (1 mark)",
                    "Identifies phospholipid bilayer or membrane structure (1 mark)"
                ]
            }
        },
        {
            "question_id": str(uuid.uuid5(uuid.NAMESPACE_DNS, "q105")),
            "subject": "Biology",
            "topic": "Biochemistry",
            "question_text": "Explain why enzymes are important in metabolism.",
            "max_marks": 5.0,
            "official_rubric": {
                "criteria": [
                    "States enzymes speed up reactions (1 mark)",
                    "Mentions they lower activation energy (1 mark)",
                    "Connects this to metabolic pathways and cell function (2 marks)",
                    "Applies to control of biochemical reactions (1 mark)"
                ]
            }
        },
        {
            "question_id": str(uuid.uuid5(uuid.NAMESPACE_DNS, "q106")),
            "subject": "Biology",
            "topic": "Genetics",
            "question_text": "State the significance of meiosis in sexual reproduction.",
            "max_marks": 4.0,
            "official_rubric": {
                "criteria": [
                    "States meiosis halves chromosome number (2 marks)",
                    "Explains gamete formation (1 mark)",
                    "Links this to restoration of diploid number at fertilisation (1 mark)"
                ]
            }
        }
    ]

    seeded_ids = seed_exam_questions(sample_questions)
    target_q_id = str(seeded_ids[0])

    sample_inputs = [
        {
            "submission_id": str(uuid.uuid4()),
            "student_id": str(uuid.uuid4()),
            "question_id": str(seeded_ids[0]),
            "question_text": "What is the primary function of mitochondria in eukaryotic cells?",
            "max_marks": 5.0,
            "official_rubric": """
            1. Identifies mitochondria as site of cellular respiration / ATP production (2 marks).
            2. Uses the term 'ATP' or 'adenosine triphosphate' (1 mark).
            3. Mentions converting nutrients/glucose into usable chemical energy (2 marks).
            """,
            "required_keywords": ["mitochondria", "ATP", "respiration", "glucose"],
            "student_answer": "Mitochondria produce ATP by breaking down glucose during cellular respiration."
        },
        {
            "submission_id": str(uuid.uuid4()),
            "student_id": str(uuid.uuid4()),
            "question_id": str(seeded_ids[1]),
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
            "student_answer": "Chloroplasts absorb sunlight and use it to turn carbon dioxide and water into glucose and oxygen."
        },
        {
            "submission_id": str(uuid.uuid4()),
            "student_id": str(uuid.uuid4()),
            "question_id": str(seeded_ids[2]),
            "question_text": "What is the role of ribosomes in a cell?",
            "max_marks": 4.0,
            "official_rubric": """
            1. Identifies ribosomes as sites of protein synthesis (2 marks).
            2. Mentions translation of mRNA (1 mark).
            3. Relates this to assembly of amino acids into proteins (1 mark).
            """,
            "required_keywords": ["ribosomes", "protein", "mRNA", "amino acids"],
            "student_answer": "Ribosomes are responsible for building proteins from amino acids."
        },
        {
            "submission_id": str(uuid.uuid4()),
            "student_id": str(uuid.uuid4()),
            "question_id": str(seeded_ids[3]),
            "question_text": "Describe the function of the cell membrane.",
            "max_marks": 5.0,
            "official_rubric": """
            1. States it controls entry and exit of substances (2 marks).
            2. Mentions selective permeability or barrier function (1 mark).
            3. Notes communication or structural role (1 mark).
            4. Identifies phospholipid bilayer or membrane structure (1 mark).
            """,
            "required_keywords": ["membrane", "selective", "cell", "transport"],
            "student_answer": "It controls what enters and leaves the cell and helps the cell communicate with its environment."
        },
        {
            "submission_id": str(uuid.uuid4()),
            "student_id": str(uuid.uuid4()),
            "question_id": str(seeded_ids[4]),
            "question_text": "Explain why enzymes are important in metabolism.",
            "max_marks": 5.0,
            "official_rubric": """
            1. States enzymes speed up reactions (1 mark).
            2. Mentions they lower activation energy (1 mark).
            3. Connects this to metabolic pathways and cell function (2 marks).
            4. Applies to control of biochemical reactions (1 mark).
            """,
            "required_keywords": ["enzymes", "reaction", "activation", "metabolism"],
            "student_answer": "Enzymes speed up chemical reactions in the body and help maintain life processes."
        },
        {
            "submission_id": str(uuid.uuid4()),
            "student_id": str(uuid.uuid4()),
            "question_id": str(seeded_ids[5]),
            "question_text": "State the significance of meiosis in sexual reproduction.",
            "max_marks": 4.0,
            "official_rubric": """
            1. States meiosis halves chromosome number (2 marks).
            2. Explains gamete formation (1 mark).
            3. Links this to restoration of diploid number at fertilisation (1 mark).
            """,
            "required_keywords": ["meiosis", "chromosome", "gametes", "fertilisation"],
            "student_answer": "Meiosis creates gametes with half the number of chromosomes so fertilisation restores the diploid number."
        }
    ]

    print("\n🚀 Executing LangGraph Workflow synced with SCHEMA.md...")
    for index, sample_input in enumerate(sample_inputs, start=1):
        print(f"\n=== SAMPLE {index} ===")
        final_state = grading_workflow.invoke(sample_input)
        print(f"Submission ID       : {final_state['submission_id']}")
        print(f"Status              : {final_state['final_status']}")
        print(f"Assigned Score      : {final_state['raw_eval']['score']} / {final_state['max_marks']}")
        print(f"Composite Confidence: {final_state['composite_confidence']}")