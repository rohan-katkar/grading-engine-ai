import json
import re
from ollama import chat
from pydantic import BaseModel, Field

# 1. Define the exact JSON output structure using Pydantic
class EvaluationOutput(BaseModel):
    score: float = Field(description="Assigned mark out of max_marks")
    max_marks: float = Field(description="Maximum possible marks")
    llm_self_confidence: float = Field(description="LLM self-reported confidence score between 0.0 and 1.0")
    key_points_matched: list[str] = Field(description="Rubric points the student correctly hit")
    key_points_missing: list[str] = Field(description="Rubric points the student missed")
    feedback: str = Field(description="Constructive justification for the score")

# 2. Deterministic Confidence Calculator Tool
def calculate_deterministic_confidence(
    max_marks: float,
    assigned_score: float,
    key_points_matched: list[str],
    key_points_missing: list[str],
    student_answer: str,
    required_keywords: list[str],
    llm_self_confidence: float
) -> float:
    """
    Calculates a verified composite confidence score combining:
    - Rubric coverage clarity (40%)
    - Hard keyword verification (30%)
    - Score boundary decisiveness (15%)
    - LLM self-reported confidence (15%)
    """
    # Fix scaling if LLM outputs 100 or 100.0 instead of 1.0
    if llm_self_confidence > 1.0:
        llm_self_confidence = llm_self_confidence / 100.0
    llm_self_confidence = max(0.0, min(1.0, llm_self_confidence))

    # 1. Rubric Coverage Ratio
    total_rubric_items = len(key_points_matched) + len(key_points_missing)
    rubric_clarity = (len(key_points_matched) / total_rubric_items) if total_rubric_items > 0 else 0.5

    # 2. Hard Keyword Verification (Exact Regex Matching)
    matched_keywords = [
        kw for kw in required_keywords 
        if re.search(r'\b' + re.escape(kw) + r'\b', student_answer, re.IGNORECASE)
    ]
    keyword_coverage = len(matched_keywords) / len(required_keywords) if required_keywords else 1.0

    # 3. Score Boundary Check (Edge-case boundary detection)
    score_ratio = assigned_score / max_marks if max_marks > 0 else 0.0
    decisiveness = 1.0 if (score_ratio == 1.0 or score_ratio == 0.0) else 0.75

    # 4. Weighted Composite Score
    composite_confidence = (
        (0.40 * rubric_clarity) +
        (0.30 * keyword_coverage) +
        (0.15 * decisiveness) +
        (0.15 * llm_self_confidence)
    )

    return round(composite_confidence, 2)

# 3. Mock Exam & Rubric Data
question = "What is the primary function of mitochondria in eukaryotic cells?"
max_marks = 5.0
required_keywords = ["mitochondria", "ATP", "respiration", "glucose"]

official_rubric = """
1. Identifies mitochondria as the site of cellular respiration / ATP production (2 marks).
2. Uses the term 'ATP' or 'adenosine triphosphate' (1 mark).
3. Mentions converting nutrients/glucose into usable chemical energy (2 marks).
"""

# Test with a partially correct student answer to see how confidence adjusts
student_answer = "Mitochondria provides energy to the cells. It creates ATP."

# 4. Construct Evaluation Prompt
prompt = f"""
You are a strict, impartial academic exam evaluator. 
Grade the student answer using ONLY the provided official rubric.

QUESTION: {question}
MAX MARKS: {max_marks}

OFFICIAL RUBRIC:
{official_rubric}

STUDENT ANSWER:
{student_answer}

Evaluate step-by-step and output your verdict in strict JSON matching the schema.
"""

print("Running evaluation through qwen3:8b...\n")

# 5. Invoke Model
response = chat(
    model="qwen3:8b",
    messages=[
        {"role": "system", "content": "You are a precise grading system that outputs strictly structured JSON."},
        {"role": "user", "content": prompt}
    ],
    format=EvaluationOutput.model_json_schema(),
    options={"temperature": 0.0}
)

# 6. Parse Output & Compute Composite Confidence
raw_result = json.loads(response.message.content)

# print(f"Raw Results:\n{json.dumps(raw_result, indent=2)}")

composite_confidence = calculate_deterministic_confidence(
    max_marks=max_marks,
    assigned_score=raw_result["score"],
    key_points_matched=raw_result["key_points_matched"],
    key_points_missing=raw_result["key_points_missing"],
    student_answer=student_answer,
    required_keywords=required_keywords,
    llm_self_confidence=raw_result["llm_self_confidence"]
)

# 7. Print Detailed Comparison
print("=== EVALUATION RESULT ===")
print(f"Assigned Score        : {raw_result['score']} / {max_marks}")
print(f"LLM Self Confidence   : {raw_result['llm_self_confidence']}")
print(f"Deterministic Confidence: {composite_confidence}")
print(f"Routing Decision      : {'AUTO_APPROVE' if composite_confidence >= 0.80 else 'REQUIRES_HUMAN_REVIEW'}")

print("\n--- Breakdown ---")
print(f"Key Points Matched    : {raw_result['key_points_matched']}")
print(f"Key Points Missing    : {raw_result['key_points_missing']}")
print(f"Feedback              : {raw_result['feedback']}")