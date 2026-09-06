import re
from typing import Set, Tuple
from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

# Initialize Presidio Engines once
analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

# Fallback Regex Patterns (Guarantees catch even if NLP confidence threshold drops)
EMAIL_REGEX = re.compile(r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b')
PHONE_REGEX = re.compile(r'\b(?:\+?1[-. ]?)?\(?\d{3}\)?[-. ]?\d{3}[-. ]?\d{4}\b')
STUDENT_ID_REGEX = re.compile(r'\b(?:ID|Id|id|StudentID|UUID)[:#\s]*[A-Za-z0-9\-]{5,36}\b')


def extract_domain_allowlist(*context_sources: str) -> Set[str]:
    """
    Extracts key domain terms and capitalized phrases from trusted reference materials.
    """
    allowlist: Set[str] = set()
    
    for text in context_sources:
        if not text:
            continue
            
        # 1. Extract multi-word capitalized phrases (e.g., "Jonas Salk")
        capitalized_phrases = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', text)
        for phrase in capitalized_phrases:
            allowlist.add(phrase.strip())
            
        # 2. Extract single capitalized nouns
        words = re.findall(r'\b[A-Z][a-z]{2,}\b', text)
        for word in words:
            allowlist.add(word.strip())
            
    return allowlist


def sanitize_student_answer(
    student_answer: str, 
    question_text: str = "", 
    rubric_text: str = "", 
    rag_context: str = ""
) -> Tuple[str, bool]:
    """
    Sanitizes student answers using Presidio NLP + Guaranteed Fallback Regex,
    filtering out allowlisted domain entities.
    """
    if not student_answer:
        return student_answer, False

    pii_detected = False
    sanitized_text = student_answer

    # Step 1: Extract dynamic domain allowlist
    allowlist = extract_domain_allowlist(question_text, rubric_text, rag_context)

    # Step 2: Presidio Analyzer with explicit low threshold so emails/phones aren't dropped
    results = analyzer.analyze(
        text=sanitized_text,
        entities=["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "STUDENT_ID"],
        language="en",
        score_threshold=0.2  # 👈 CRITICAL FIX: Ensures email patterns pass confidence filter
    )

    # Step 3: Filter Presidio results against Allowlist
    filtered_results = []
    for res in results:
        entity_text = sanitized_text[res.start:res.end].strip()
        
        is_allowlisted = any(
            entity_text.lower() == term.lower() or term.lower() in entity_text.lower()
            for term in allowlist
        )
        
        if not is_allowlisted:
            filtered_results.append(res)

    if filtered_results:
        pii_detected = True
        operators = {
            "PERSON": OperatorConfig("replace", {"new_value": "[REDACTED_NAME]"}),
            "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "[REDACTED_EMAIL]"}),
            "PHONE_NUMBER": OperatorConfig("replace", {"new_value": "[REDACTED_PHONE]"}),
            "STUDENT_ID": OperatorConfig("replace", {"new_value": "[REDACTED_ID]"}),
        }

        anonymized_result = anonymizer.anonymize(
            text=sanitized_text,
            analyzer_results=filtered_results,
            operators=operators
        )
        sanitized_text = anonymized_result.text

    # Step 4: Deterministic Regex Backup (Ensures emails/IDs never escape even if Presidio misses)
    if EMAIL_REGEX.search(sanitized_text):
        sanitized_text = EMAIL_REGEX.sub("[REDACTED_EMAIL]", sanitized_text)
        pii_detected = True

    if PHONE_REGEX.search(sanitized_text):
        sanitized_text = PHONE_REGEX.sub("[REDACTED_PHONE]", sanitized_text)
        pii_detected = True

    if STUDENT_ID_REGEX.search(sanitized_text):
        sanitized_text = STUDENT_ID_REGEX.sub("[REDACTED_ID]", sanitized_text)
        pii_detected = True

    return sanitized_text, pii_detected


# %% [unit_test]
if __name__ == "__main__":
    q_text = "Explain the contribution of Jonas Salk to modern medicine."
    r_text = "Must mention Jonas Salk, polio vaccine, and 1955 clinical trials."
    rag_ctx = "Jonas Salk developed the first successful inactivated polio vaccine."

    student_raw = (
        "Hi, my name is Alex Vance (Student ID: 987654). "
        "Jonas Salk developed the inactivated polio vaccine in 1955. "
        "Reach out at alex.vance@university.edu or student_test@gmail.com."
    )

    clean_answer, pii_found = sanitize_student_answer(
        student_answer=student_raw,
        question_text=q_text,
        rubric_text=r_text,
        rag_context=rag_ctx
    )

    print("--- Presidio + Regex Backup Test ---")
    print(f"PII Flagged: {pii_found}")
    print("\n[Original Answer]:\n", student_raw)
    print("\n[Sanitized Answer]:\n", clean_answer)